"""Six-axis inertial endpoint diagnostics shared by tests and experiments."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
import torch

from validation_v2.data.normalization import imu_dataset_units_to_si
from validation_v2.physics import propagate_imu, so3_log


def physics_endpoint_diagnostics(
    completed_imu_dataset_units: np.ndarray,
    target_imu_dataset_units: np.ndarray,
    mask: np.ndarray,
    dt_s: np.ndarray,
    vicon_position_m: np.ndarray,
    vicon_rotation_body_to_world: np.ndarray,
    vicon_velocity_mps: np.ndarray,
    *,
    acceleration_unit: str = "G",
    acceleration_mode: str = "gravity_compensated",
    dtype: torch.dtype = torch.float64,
) -> Mapping[str, float]:
    """Return missing-channel RMSE and full-mechanization endpoint errors."""

    completed = np.asarray(completed_imu_dataset_units, dtype=np.float64)
    target = np.asarray(target_imu_dataset_units, dtype=np.float64)
    mask_values = np.asarray(mask)
    if completed.shape != target.shape or completed.ndim != 2 or completed.shape[1] != 6:
        raise ValueError("completed and target IMU must have shape (T,6)")
    if mask_values.shape != completed.shape or not np.all((mask_values == 0) | (mask_values == 1)):
        raise ValueError("mask must match IMU and contain 0/1")
    if not np.all(np.isfinite(completed)) or not np.all(np.isfinite(target)):
        raise ValueError("IMU values must be finite")
    time_steps = len(completed)
    dt = np.asarray(dt_s, dtype=np.float64)
    position = np.asarray(vicon_position_m, dtype=np.float64)
    rotation = np.asarray(vicon_rotation_body_to_world, dtype=np.float64)
    velocity = np.asarray(vicon_velocity_mps, dtype=np.float64)
    if dt.shape != (time_steps,) or position.shape != (time_steps, 3):
        raise ValueError("dt and Vicon position must align with IMU")
    if rotation.shape != (time_steps, 3, 3) or velocity.shape != (time_steps, 3):
        raise ValueError("Vicon rotation/velocity must align with IMU")

    completed_tensor = imu_dataset_units_to_si(
        torch.as_tensor(completed, dtype=dtype), acceleration_unit=acceleration_unit
    )
    target_tensor = imu_dataset_units_to_si(
        torch.as_tensor(target, dtype=dtype), acceleration_unit=acceleration_unit
    )
    propagated = propagate_imu(
        completed_tensor[None, :, :3],
        completed_tensor[None, :, 3:],
        torch.as_tensor(dt, dtype=dtype)[None],
        torch.as_tensor(rotation[0], dtype=dtype)[None],
        torch.as_tensor(velocity[0], dtype=dtype)[None],
        torch.as_tensor(position[0], dtype=dtype)[None],
        acceleration_mode=acceleration_mode,
    )
    end_rotation = torch.as_tensor(rotation[-1], dtype=dtype)
    rotation_error = so3_log(
        end_rotation.transpose(-1, -2)
        @ propagated.rotation_body_to_world[0, -1]
    )
    velocity_error = propagated.velocity_world_mps[0, -1] - torch.as_tensor(
        velocity[-1], dtype=dtype
    )
    position_error = propagated.position_world_m[0, -1] - torch.as_tensor(
        position[-1], dtype=dtype
    )

    missing = mask_values == 0
    completed_si = completed_tensor.numpy()
    target_si = target_tensor.numpy()

    def channel_rmse(channel_slice: slice) -> float:
        selected = missing[:, channel_slice]
        if not np.any(selected):
            return 0.0
        error = completed_si[:, channel_slice] - target_si[:, channel_slice]
        return float(np.sqrt(np.mean(np.square(error[selected]))))

    rotation_rad = float(torch.linalg.vector_norm(rotation_error))
    return {
        "missing_gyro_rmse_radps": channel_rmse(slice(0, 3)),
        "missing_accelerometer_rmse_mps2": channel_rmse(slice(3, 6)),
        "rotation_endpoint_error_rad": rotation_rad,
        "rotation_endpoint_error_deg": rotation_rad * 180.0 / math.pi,
        "velocity_endpoint_error_mps": float(torch.linalg.vector_norm(velocity_error)),
        "position_endpoint_error_m": float(torch.linalg.vector_norm(position_error)),
    }


__all__ = ["physics_endpoint_diagnostics"]
