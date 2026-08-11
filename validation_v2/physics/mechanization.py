"""Differentiable six-axis IMU mechanization.

Units and frames:

* gyro: rad/s, body frame
* acceleration: m/s^2, body frame
* dt: seconds; ``dt[:, i]`` is the interval from ``i-1`` to ``i``
* rotation: body-to-world
* velocity: m/s, world frame
* position: m, world frame
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .so3 import so3_exp


@dataclass(frozen=True)
class IMUPropagation:
    rotation_body_to_world: torch.Tensor
    velocity_world_mps: torch.Tensor
    position_world_m: torch.Tensor


def _bias(
    value: torch.Tensor | None,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(
            (*reference.shape[:-2], 3), dtype=reference.dtype, device=reference.device
        )
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    expected = (*reference.shape[:-2], 3)
    if value.shape not in {(3,), expected}:
        raise ValueError(f"{name} must have shape (3,) or {expected}")
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f"{name} must match IMU dtype and device")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value.expand(expected)


def _validate(
    gyro: torch.Tensor,
    acc: torch.Tensor,
    dt: torch.Tensor,
    rotation0: torch.Tensor,
    velocity0: torch.Tensor,
    position0: torch.Tensor,
) -> None:
    if not all(
        isinstance(value, torch.Tensor)
        for value in (gyro, acc, dt, rotation0, velocity0, position0)
    ):
        raise TypeError("all propagation inputs must be torch tensors")
    if gyro.ndim != 3 or gyro.shape[-1] != 3 or gyro.shape[1] < 2:
        raise ValueError("gyro must have shape (B,T,3), T >= 2")
    if acc.shape != gyro.shape:
        raise ValueError("acc must match gyro shape")
    if dt.shape == (*gyro.shape[:2], 1):
        dt = dt[..., 0]
    if dt.shape != gyro.shape[:2]:
        raise ValueError("dt must have shape (B,T) or (B,T,1)")
    if rotation0.shape != (gyro.shape[0], 3, 3):
        raise ValueError("rotation0 must have shape (B,3,3)")
    if velocity0.shape != (gyro.shape[0], 3) or position0.shape != (gyro.shape[0], 3):
        raise ValueError("velocity0 and position0 must have shape (B,3)")
    dtype, device = gyro.dtype, gyro.device
    if not dtype.is_floating_point:
        raise TypeError("propagation tensors must have floating dtypes")
    for name, value in (
        ("acc", acc), ("dt", dt), ("rotation0", rotation0),
        ("velocity0", velocity0), ("position0", position0),
    ):
        if value.dtype != dtype or value.device != device:
            raise ValueError(f"{name} must match gyro dtype and device")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if not torch.all(dt[:, 1:] > 0):
        raise ValueError("dt[:,1:] must be strictly positive seconds")


def propagate_imu(
    gyro: torch.Tensor,
    acc: torch.Tensor,
    dt: torch.Tensor,
    rotation0: torch.Tensor,
    velocity0: torch.Tensor,
    position0: torch.Tensor,
    *,
    gyro_bias: torch.Tensor | None = None,
    acc_bias: torch.Tensor | None = None,
    acceleration_mode: str = "gravity_compensated",
    gravity_world_mps2: torch.Tensor | None = None,
) -> IMUPropagation:
    """Propagate batched IMU sequences using midpoint/trapezoidal updates.

    ``gravity_compensated`` is the OxIOD ``user_acc`` mode and does not add
    gravity. ``specific_force`` adds ``gravity_world_mps2`` after rotating the
    bias-corrected specific force into the world frame.
    """

    if dt.ndim == 3 and dt.shape[-1] == 1:
        dt = dt[..., 0]
    _validate(gyro, acc, dt, rotation0, velocity0, position0)
    if acceleration_mode not in {"gravity_compensated", "specific_force"}:
        raise ValueError("unsupported acceleration_mode")
    bg = _bias(gyro_bias, gyro, "gyro_bias")
    ba = _bias(acc_bias, acc, "acc_bias")
    if gravity_world_mps2 is None:
        gravity = gyro.new_tensor([0.0, 0.0, -9.80665]).expand(gyro.shape[0], 3)
    else:
        gravity = _bias(gravity_world_mps2, gyro, "gravity_world_mps2")

    rotations = [rotation0]
    velocities = [velocity0]
    positions = [position0]
    corrected_gyro = gyro - bg[:, None, :]
    corrected_acc = acc - ba[:, None, :]
    for index in range(1, gyro.shape[1]):
        interval = dt[:, index:index + 1]
        omega_mid = 0.5 * (
            corrected_gyro[:, index - 1] + corrected_gyro[:, index]
        )
        next_rotation = rotations[-1] @ so3_exp(omega_mid * interval)
        acceleration_previous = (
            rotations[-1] @ corrected_acc[:, index - 1, :, None]
        )[..., 0]
        acceleration_next = (
            next_rotation @ corrected_acc[:, index, :, None]
        )[..., 0]
        if acceleration_mode == "specific_force":
            acceleration_previous = acceleration_previous + gravity
            acceleration_next = acceleration_next + gravity
        acceleration_mid = 0.5 * (acceleration_previous + acceleration_next)
        next_velocity = velocities[-1] + acceleration_mid * interval
        next_position = positions[-1] + 0.5 * (
            velocities[-1] + next_velocity
        ) * interval
        rotations.append(next_rotation)
        velocities.append(next_velocity)
        positions.append(next_position)
    return IMUPropagation(
        rotation_body_to_world=torch.stack(rotations, dim=1),
        velocity_world_mps=torch.stack(velocities, dim=1),
        position_world_m=torch.stack(positions, dim=1),
    )


__all__ = ["IMUPropagation", "propagate_imu"]
