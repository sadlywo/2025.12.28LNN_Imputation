"""Training-only robust normalization for IMU channels."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np

from validation_v2.types import Recording


@dataclass(frozen=True)
class RobustTrainScaler:
    """Median/MAD scaler fitted exclusively from explicitly allowed recordings."""

    center_: np.ndarray
    scale_: np.ndarray
    training_ids: tuple[str, ...]

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
