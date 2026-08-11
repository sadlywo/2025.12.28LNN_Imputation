"""Formal signal + inertial-mechanization objective for IMU imputation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from validation_v2.data.normalization import (
    denormalize_imu_tensor,
    imu_dataset_units_to_si,
)
from validation_v2.models.hybrid import complete_signal
from validation_v2.physics.mechanization import propagate_imu
from validation_v2.physics.so3 import so3_log

from .reconstruction import missing_mse


@dataclass(frozen=True)
class PhysicsLossConfig:
    """Fixed, non-learnable physical objective scales and conventions."""

    lambda_physics: float = 0.1
    sigma_rotation_rad: float = 0.1
    sigma_velocity_mps: float = 0.5
    sigma_position_m: float = 0.5
    acceleration_mode: str = "gravity_compensated"
    acceleration_unit: str = "G"

    def __post_init__(self) -> None:
        finite = (
            self.lambda_physics,
            self.sigma_rotation_rad,
            self.sigma_velocity_mps,
            self.sigma_position_m,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("physics loss weights/scales must be finite")
        if self.lambda_physics < 0:
            raise ValueError("lambda_physics must be non-negative")
        if min(finite[1:]) <= 0:
            raise ValueError("physics residual scales must be positive")
        if self.acceleration_mode not in {"gravity_compensated", "specific_force"}:
            raise ValueError("unsupported acceleration_mode")
        if self.acceleration_unit not in {"G", "m/s^2"}:
            raise ValueError("unsupported acceleration_unit")


def _require_shape(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")


class IMUPhysicsInformedLoss(nn.Module):
    """Optimize exactly ``L_signal + lambda_physics * L_physics``.

    Vicon labels initialize and supervise the differentiable propagation only;
    they are never model inputs.  The six-axis IMU is denormalized to rad/s and
    m/s² before integration.  ``R`` denotes body-to-world rotation throughout.
    """

    def __init__(self, config: PhysicsLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or PhysicsLossConfig()

    def forward(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        dt: torch.Tensor,
        normalization_center: torch.Tensor,
        normalization_scale: torch.Tensor,
        vicon_position_m: torch.Tensor,
        vicon_rotation_body_to_world: torch.Tensor,
        vicon_velocity_mps: torch.Tensor,
        completed: torch.Tensor | None = None,
        gyro_bias_radps: torch.Tensor | None = None,
        acc_bias_mps2: torch.Tensor | None = None,
        gravity_world_mps2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.ndim != 3 or prediction.shape[-1] != 6:
            raise ValueError("prediction must have shape (B,T,6)")
        if target.shape != prediction.shape or mask.shape != prediction.shape:
            raise ValueError("target and mask must match prediction")
        batch, steps, _ = prediction.shape
        if dt.shape == (batch, steps, 1):
            dt = dt[..., 0]
        _require_shape(dt, (batch, steps), "dt")
        _require_shape(vicon_position_m, (batch, steps, 3), "vicon_position_m")
        _require_shape(vicon_velocity_mps, (batch, steps, 3), "vicon_velocity_mps")
        _require_shape(
            vicon_rotation_body_to_world,
            (batch, steps, 3, 3),
            "vicon_rotation_body_to_world",
        )
        for name, value in (
            ("target", target), ("mask", mask), ("dt", dt),
            ("normalization_center", normalization_center),
            ("normalization_scale", normalization_scale),
            ("vicon_position_m", vicon_position_m),
            ("vicon_rotation_body_to_world", vicon_rotation_body_to_world),
            ("vicon_velocity_mps", vicon_velocity_mps),
        ):
            if value.device != prediction.device or value.dtype != prediction.dtype:
                raise ValueError(f"{name} must share prediction dtype and device")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite values")
        if normalization_center.shape not in {(6,), (batch, 6)}:
            raise ValueError("normalization_center must have shape (6,) or (B,6)")
        if normalization_scale.shape not in {(6,), (batch, 6)}:
            raise ValueError("normalization_scale must have shape (6,) or (B,6)")
        if normalization_center.ndim == 2:
            normalization_center = normalization_center[:, None, :]
            normalization_scale = normalization_scale[:, None, :]

        signal = missing_mse(prediction, target, mask)
        observed = target
        expected_completed = complete_signal(observed, mask, prediction)
        if completed is None:
            completed = expected_completed
        else:
            if completed.shape != prediction.shape:
                raise ValueError("completed must match prediction shape")
            if not torch.equal(completed, expected_completed):
                raise ValueError("completed must exactly preserve observed values")

        physical_dataset_units = denormalize_imu_tensor(
            completed, normalization_center, normalization_scale
        )
        physical_si = imu_dataset_units_to_si(
            physical_dataset_units,
            acceleration_unit=self.config.acceleration_unit,
        )
        propagated = propagate_imu(
            physical_si[..., :3],
            physical_si[..., 3:],
            dt,
            vicon_rotation_body_to_world[:, 0],
            vicon_velocity_mps[:, 0],
            vicon_position_m[:, 0],
            gyro_bias=gyro_bias_radps,
            acc_bias=acc_bias_mps2,
            acceleration_mode=self.config.acceleration_mode,
            gravity_world_mps2=gravity_world_mps2,
        )

        rotation_residual = so3_log(
            vicon_rotation_body_to_world[:, -1].transpose(-1, -2)
            @ propagated.rotation_body_to_world[:, -1]
        )
        velocity_residual = (
            propagated.velocity_world_mps[:, -1] - vicon_velocity_mps[:, -1]
        )
        position_residual = (
            propagated.position_world_m[:, -1] - vicon_position_m[:, -1]
        )
        rotation_term = rotation_residual.square().sum(dim=-1) / (
            self.config.sigma_rotation_rad ** 2
        )
        velocity_term = velocity_residual.square().sum(dim=-1) / (
            self.config.sigma_velocity_mps ** 2
        )
        position_term = position_residual.square().sum(dim=-1) / (
            self.config.sigma_position_m ** 2
        )
        physics = (rotation_term + velocity_term + position_term).mean()
        total = signal + self.config.lambda_physics * physics

        rotation_error = torch.linalg.vector_norm(rotation_residual, dim=-1).mean()
        velocity_error = torch.linalg.vector_norm(velocity_residual, dim=-1).mean()
        position_error = torch.linalg.vector_norm(position_residual, dim=-1).mean()
        components = {
            "total": total,
            "signal": signal,
            "physics": physics,
            "physics_rotation": rotation_term.mean(),
            "physics_velocity": velocity_term.mean(),
            "physics_position": position_term.mean(),
            "rotation_endpoint_error_rad": rotation_error,
            "rotation_endpoint_error_deg": rotation_error * (180.0 / math.pi),
            "velocity_endpoint_error_mps": velocity_error,
            "position_endpoint_error_m": position_error,
        }
        return total, components


__all__ = ["IMUPhysicsInformedLoss", "PhysicsLossConfig"]
