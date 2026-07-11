"""Shared types for the validation v2 package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class Recording:
    """One raw IMU/Vicon pair with explicit units and time overlap."""

    id: str
    imu_time_s: np.ndarray
    imu_six: np.ndarray
    vicon_time_s: np.ndarray
    vicon_position_m: np.ndarray
    vicon_quaternion_xyzw: np.ndarray
    overlap_s: tuple[float, float]
    metadata: Mapping[str, object]


def _tensor_copy(value: torch.Tensor) -> torch.Tensor:
    """Detach dataclass state from mutable caller-owned tensor storage."""

    return value.detach().clone()


@dataclass(frozen=True)
class FeatureBatch:
    """Leakage-safe model inputs and their explicit timing/mask context."""

    values: torch.Tensor
    dt: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _tensor_copy(self.values))
        object.__setattr__(self, "dt", _tensor_copy(self.dt))
        object.__setattr__(self, "mask", _tensor_copy(self.mask))


@dataclass(frozen=True)
class MaskResult:
    """A generated value mask with requested and actually realized rates."""

    mask: torch.Tensor
    requested_fraction: float
    realized_fraction: float
    topology: str
    seed: int
    masked_channels: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask", _tensor_copy(self.mask))


@dataclass(frozen=True)
class IrregularTimeResult:
    """Timestamp-only perturbation result, deliberately separate from value masks."""

    time: torch.Tensor
    dt: torch.Tensor
    retained_indices: torch.Tensor
    requested_irregularity: float
    realized_irregularity: float
    method: str
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _tensor_copy(self.time))
        object.__setattr__(self, "dt", _tensor_copy(self.dt))
        object.__setattr__(self, "retained_indices", _tensor_copy(self.retained_indices))


@dataclass(frozen=True)
class WindowBatch:
    """One window cut from one contiguous segment of one recording."""

    target: torch.Tensor
    mask: torch.Tensor
    dt: torch.Tensor
    index: torch.Tensor
    time: torch.Tensor
    recording_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _tensor_copy(self.target))
        object.__setattr__(self, "mask", _tensor_copy(self.mask))
        object.__setattr__(self, "dt", _tensor_copy(self.dt))
        object.__setattr__(self, "index", _tensor_copy(self.index))
        object.__setattr__(self, "time", _tensor_copy(self.time))
