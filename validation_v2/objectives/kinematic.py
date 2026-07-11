"""Optional, physically gated full-sequence kinematic objective.

This objective is not enabled by the default training flow.  It only accepts
denormalized acceleration in a world/reference frame and real seconds so that
normalized network values cannot silently be integrated as physical data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class KinematicLoss:
    """Weighted total and the two unweighted MSE components."""

    total: torch.Tensor
    velocity: torch.Tensor
    displacement: torch.Tensor


def _require_physical_metadata(frame_metadata: Mapping[str, object]) -> None:
    if not isinstance(frame_metadata, Mapping):
        raise ValueError("frame_metadata must be a mapping")
    required = {
        "acceleration_unit": "m/s^2",
        "time_unit": "s",
        "acceleration_frame": "world",
    }
    for key, expected in required.items():
        if frame_metadata.get(key) != expected:
            raise ValueError(
                f"frame_metadata must declare {key}={expected!r}; "
                "kinematic loss accepts physical world-frame inputs only"
            )


def _validate_label(
    label: torch.Tensor | None,
    acceleration: torch.Tensor,
    name: str,
) -> None:
    if label is None:
        return
    if not isinstance(label, torch.Tensor) or label.shape != acceleration.shape:
        raise ValueError(f"{name} label must be a tensor matching acceleration shape")
    if label.dtype != acceleration.dtype or label.device != acceleration.device:
        raise ValueError(f"{name} label must match acceleration dtype and device")
    if not torch.isfinite(label).all():
        raise ValueError(f"{name} label must contain only finite values")


def kinematic_consistency_loss(
    acceleration_world_mps2: torch.Tensor,
    time_s: torch.Tensor,
    *,
    frame_metadata: Mapping[str, object],
    velocity_mps: torch.Tensor | None = None,
    displacement_m: torch.Tensor | None = None,
    velocity_weight: float = 1.0,
    displacement_weight: float = 1.0,
) -> KinematicLoss:
    """Integrate one complete sequence and compare physical kinematic labels.

    Trapezoidal acceleration integration produces velocity, followed by
    trapezoidal velocity integration for displacement.  Initial state is taken
    exactly once from the first available label (or zero when that label type is
    absent).  The function is fully differentiable with respect to acceleration.
    """

    _require_physical_metadata(frame_metadata)
    if not isinstance(acceleration_world_mps2, torch.Tensor):
        raise ValueError("acceleration_world_mps2 must be a torch tensor")
    if (
        acceleration_world_mps2.ndim != 2
        or acceleration_world_mps2.shape[1] != 3
        or acceleration_world_mps2.shape[0] < 2
        or not acceleration_world_mps2.is_floating_point()
    ):
        raise ValueError("acceleration_world_mps2 must have floating shape (N, 3), N >= 2")
    if not torch.isfinite(acceleration_world_mps2).all():
        raise ValueError("acceleration_world_mps2 must contain only finite values")
    if not isinstance(time_s, torch.Tensor) or time_s.shape != (acceleration_world_mps2.shape[0],):
        raise ValueError("time_s must be a length-N torch tensor")
    if time_s.dtype != acceleration_world_mps2.dtype or time_s.device != acceleration_world_mps2.device:
        raise ValueError("time_s must match acceleration dtype and device")
    if not torch.isfinite(time_s).all() or not torch.all(torch.diff(time_s) > 0):
        raise ValueError("time_s must contain finite, strictly increasing seconds")
    if velocity_mps is None or displacement_m is None:
        raise ValueError("both velocity and displacement labels are required")
    _validate_label(velocity_mps, acceleration_world_mps2, "velocity_mps")
    _validate_label(displacement_m, acceleration_world_mps2, "displacement_m")
    weights = torch.as_tensor(
        [velocity_weight, displacement_weight],
        dtype=acceleration_world_mps2.dtype,
        device=acceleration_world_mps2.device,
    )
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("kinematic weights must be finite and non-negative")

    velocity_values = [
        velocity_mps[0] if velocity_mps is not None else torch.zeros_like(acceleration_world_mps2[0])
    ]
    position_values = [
        displacement_m[0]
        if displacement_m is not None
        else torch.zeros_like(acceleration_world_mps2[0])
    ]
    for index in range(1, acceleration_world_mps2.shape[0]):
        dt = time_s[index] - time_s[index - 1]
        next_velocity = velocity_values[-1] + 0.5 * (
            acceleration_world_mps2[index - 1] + acceleration_world_mps2[index]
        ) * dt
        next_position = position_values[-1] + 0.5 * (
            velocity_values[-1] + next_velocity
        ) * dt
        velocity_values.append(next_velocity)
        position_values.append(next_position)
    integrated_velocity = torch.stack(velocity_values)
    integrated_position = torch.stack(position_values)

    differentiable_zero = acceleration_world_mps2.sum() * 0.0
    velocity_loss = (
        torch.mean((integrated_velocity - velocity_mps).square())
        if velocity_mps is not None
        else differentiable_zero
    )
    displacement_loss = (
        torch.mean((integrated_position - displacement_m).square())
        if displacement_m is not None
        else differentiable_zero
    )
    total = velocity_weight * velocity_loss + displacement_weight * displacement_loss
    return KinematicLoss(total=total, velocity=velocity_loss, displacement=displacement_loss)
