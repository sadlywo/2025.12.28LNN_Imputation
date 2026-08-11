"""Training-only robust normalization for IMU channels."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from validation_v2.types import Recording


@dataclass(frozen=True)
class RobustTrainScaler:
    """Median/MAD scaler fitted exclusively from explicitly allowed recordings."""

    center_: np.ndarray
    scale_: np.ndarray
    training_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            center = np.asarray(self.center_, dtype=np.float64)
            scale = np.asarray(self.scale_, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("center_ and scale_ must be numeric arrays") from exc
        if center.ndim != 1 or scale.ndim != 1:
            raise ValueError("center_ and scale_ must be one-dimensional")
        if center.size == 0 or scale.size == 0:
            raise ValueError("center_ and scale_ must be non-empty")
        if center.shape != scale.shape:
            raise ValueError("center_ and scale_ must have the same shape")
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)):
            raise ValueError("center_ and scale_ must contain only finite values")
        if np.any(scale <= 0):
            raise ValueError("scale_ values must be strictly positive")

        training_ids = tuple(self.training_ids)
        if not training_ids or any(
            not isinstance(recording_id, str) or not recording_id
            for recording_id in training_ids
        ):
            raise ValueError("training_ids must contain non-empty strings")
        if len(set(training_ids)) != len(training_ids):
            raise ValueError("training_ids must be unique")
        if training_ids != tuple(sorted(training_ids)):
            raise ValueError("training_ids must be sorted")

        # Arrays backed directly by immutable ``bytes`` cannot have NumPy's
        # WRITEABLE flag re-enabled, unlike owning arrays marked read-only.
        frozen_center = np.frombuffer(center.tobytes(), dtype=np.float64).reshape(
            center.shape
        )
        frozen_scale = np.frombuffer(scale.tobytes(), dtype=np.float64).reshape(
            scale.shape
        )
        object.__setattr__(self, "center_", frozen_center)
        object.__setattr__(self, "scale_", frozen_scale)
        object.__setattr__(self, "training_ids", training_ids)

    @classmethod
    def fit(
        cls,
        recordings: Sequence[Recording],
        *,
        allowed_ids: Collection[str],
    ) -> "RobustTrainScaler":
        if not recordings:
            raise ValueError("fit requires at least one training recording")
        ids = [recording.id for recording in recordings]
        if len(set(ids)) != len(ids):
            raise ValueError("training recording ids must be unique")
        allowed = {str(recording_id) for recording_id in allowed_ids}
        disallowed = sorted(set(ids) - allowed)
        if disallowed:
            raise ValueError(
                "fit accepts train recordings only; disallowed recording ids: "
                f"{disallowed}"
            )

        arrays: list[np.ndarray] = []
        feature_count: int | None = None
        for recording in recordings:
            values = np.asarray(recording.imu_six, dtype=np.float64)
            if values.ndim != 2:
                raise ValueError("each recording.imu_six must be a two-dimensional array")
            if values.shape[0] == 0 or values.shape[1] == 0:
                raise ValueError("recording.imu_six must not be empty")
            if feature_count is None:
                feature_count = values.shape[1]
            elif values.shape[1] != feature_count:
                raise ValueError("recording.imu_six feature dimensions must be consistent")
            if not np.all(np.isfinite(values)):
                raise ValueError("recording.imu_six must contain only finite values")
            arrays.append(values)

        samples = np.concatenate(arrays, axis=0)
        center = np.median(samples, axis=0)
        mad = np.median(np.abs(samples - center), axis=0)
        scale = np.maximum(1.4826 * mad, 1e-6)
        center = np.array(center, dtype=np.float64, copy=True)
        scale = np.array(scale, dtype=np.float64, copy=True)
        center.setflags(write=False)
        scale.setflags(write=False)
        return cls(center_=center, scale_=scale, training_ids=tuple(sorted(ids)))

    @property
    def train_recording_ids(self) -> tuple[str, ...]:
        """Alias emphasizing that the persisted identifiers are train-only."""
        return self.training_ids

    def _validated_values(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("values must be a two-dimensional array")
        if array.shape[0] == 0:
            raise ValueError("values must not be empty")
        if array.shape[1] != self.center_.shape[0]:
            raise ValueError("values feature dimension does not match the fitted scaler")
        if not np.all(np.isfinite(array)):
            raise ValueError("values must contain only finite values")
        return array

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Return normalized values without mutating the caller's array."""
        array = self._validated_values(values)
        return (array - self.center_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Undo :meth:`transform` without mutating the caller's array."""
        array = self._validated_values(values)
        return array * self.scale_ + self.center_

    def transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor's final six-channel axis without leaving PyTorch."""

        if not isinstance(values, torch.Tensor) or values.shape[-1:] != self.center_.shape:
            raise ValueError("values final dimension must match scaler channels")
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError("values must be a finite floating tensor")
        center = torch.tensor(self.center_, dtype=values.dtype, device=values.device)
        scale = torch.tensor(self.scale_, dtype=values.dtype, device=values.device)
        return (values - center) / scale

    def inverse_transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        """Recover physical dataset units while preserving autograd and device."""

        if not isinstance(values, torch.Tensor) or values.shape[-1:] != self.center_.shape:
            raise ValueError("values final dimension must match scaler channels")
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError("values must be a finite floating tensor")
        center = torch.tensor(self.center_, dtype=values.dtype, device=values.device)
        scale = torch.tensor(self.scale_, dtype=values.dtype, device=values.device)
        return values * scale + center


def denormalize_imu_tensor(
    normalized: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Denormalize ``[...,6]`` IMU values using train-fitted median/MAD stats."""

    if not all(isinstance(value, torch.Tensor) for value in (normalized, center, scale)):
        raise TypeError("normalized, center, and scale must be torch tensors")
    if normalized.shape[-1:] != (6,) or center.shape[-1:] != (6,) or scale.shape[-1:] != (6,):
        raise ValueError("IMU values and normalization stats must end in six channels")
    if center.device != normalized.device or scale.device != normalized.device:
        raise ValueError("normalization stats must share the IMU device")
    if center.dtype != normalized.dtype or scale.dtype != normalized.dtype:
        raise ValueError("normalization stats must share the IMU dtype")
    if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("normalization stats must be finite with positive scale")
    return normalized * scale + center


def imu_dataset_units_to_si(values: torch.Tensor, *, acceleration_unit: str) -> torch.Tensor:
    """Convert six-channel IMU from dataset units to rad/s and m/s².

    Gyroscope channels are already rad/s.  OxIOD ``user_acc`` is stored in G.
    EuRoC MAV and IDOL adapters declare accelerometer values in m/s².
    """

    if not isinstance(values, torch.Tensor) or values.shape[-1:] != (6,):
        raise ValueError("values must be a torch tensor ending in six channels")
    if acceleration_unit not in {"G", "m/s^2"}:
        raise ValueError("acceleration_unit must be 'G' or 'm/s^2'")
    gyro, acceleration = values[..., :3], values[..., 3:]
    if acceleration_unit == "G":
        acceleration = acceleration * values.new_tensor(9.80665)
    return torch.cat((gyro, acceleration), dim=-1)


__all__ = [
    "RobustTrainScaler",
    "denormalize_imu_tensor",
    "imu_dataset_units_to_si",
]
