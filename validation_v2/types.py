"""Shared types for the validation v2 package."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True, init=False, repr=False, eq=False)
class FeatureBatch:
    """Leakage-safe model inputs and their explicit timing/mask context."""

    _values: torch.Tensor = field(repr=False)
    _dt: torch.Tensor = field(repr=False)
    _mask: torch.Tensor = field(repr=False)

    def __init__(self, values: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor) -> None:
        object.__setattr__(self, "_values", _tensor_copy(values))
        object.__setattr__(self, "_dt", _tensor_copy(dt))
        object.__setattr__(self, "_mask", _tensor_copy(mask))

    @property
    def values(self) -> torch.Tensor:
        return _tensor_copy(self._values)

    @property
    def dt(self) -> torch.Tensor:
        return _tensor_copy(self._dt)

    @property
    def mask(self) -> torch.Tensor:
        return _tensor_copy(self._mask)


@dataclass(frozen=True, init=False, repr=False, eq=False)
class MaskResult:
    """A generated value mask with requested and actually realized rates."""

    _mask: torch.Tensor = field(repr=False)
    requested_fraction: float
    realized_fraction: float
    topology: str
    seed: int
    masked_channels: int | None = None

    def __init__(
        self,
        mask: torch.Tensor,
        requested_fraction: float,
        realized_fraction: float,
        topology: str,
        seed: int,
        masked_channels: int | None = None,
    ) -> None:
        object.__setattr__(self, "_mask", _tensor_copy(mask))
        object.__setattr__(self, "requested_fraction", requested_fraction)
        object.__setattr__(self, "realized_fraction", realized_fraction)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "masked_channels", masked_channels)

    @property
    def mask(self) -> torch.Tensor:
        return _tensor_copy(self._mask)


@dataclass(frozen=True, init=False, repr=False, eq=False)
class IrregularTimeResult:
    """Timestamp-only perturbation result, deliberately separate from value masks."""

    _time: torch.Tensor = field(repr=False)
    _dt: torch.Tensor = field(repr=False)
    _retained_indices: torch.Tensor = field(repr=False)
    requested_irregularity: float
    realized_irregularity: float
    method: str
    seed: int

    def __init__(
        self,
        time: torch.Tensor,
        dt: torch.Tensor,
        retained_indices: torch.Tensor,
        requested_irregularity: float,
        realized_irregularity: float,
        method: str,
        seed: int,
    ) -> None:
        object.__setattr__(self, "_time", _tensor_copy(time))
        object.__setattr__(self, "_dt", _tensor_copy(dt))
        object.__setattr__(self, "_retained_indices", _tensor_copy(retained_indices))
        object.__setattr__(self, "requested_irregularity", requested_irregularity)
        object.__setattr__(self, "realized_irregularity", realized_irregularity)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "seed", seed)

    @property
    def time(self) -> torch.Tensor:
        return _tensor_copy(self._time)

    @property
    def dt(self) -> torch.Tensor:
        return _tensor_copy(self._dt)

    @property
    def retained_indices(self) -> torch.Tensor:
        return _tensor_copy(self._retained_indices)


@dataclass(frozen=True, init=False, repr=False, eq=False)
class WindowBatch:
    """One window cut from one contiguous segment of one recording."""

    _target: torch.Tensor = field(repr=False)
    _mask: torch.Tensor = field(repr=False)
    _dt: torch.Tensor = field(repr=False)
    _index: torch.Tensor = field(repr=False)
    _time: torch.Tensor = field(repr=False)
    recording_id: str

    def __init__(
        self,
        target: torch.Tensor,
        mask: torch.Tensor,
        dt: torch.Tensor,
        index: torch.Tensor,
        time: torch.Tensor,
        recording_id: str,
    ) -> None:
        object.__setattr__(self, "_target", _tensor_copy(target))
        object.__setattr__(self, "_mask", _tensor_copy(mask))
        object.__setattr__(self, "_dt", _tensor_copy(dt))
        object.__setattr__(self, "_index", _tensor_copy(index))
        object.__setattr__(self, "_time", _tensor_copy(time))
        object.__setattr__(self, "recording_id", recording_id)

    @property
    def target(self) -> torch.Tensor:
        return _tensor_copy(self._target)

    @property
    def mask(self) -> torch.Tensor:
        return _tensor_copy(self._mask)

    @property
    def dt(self) -> torch.Tensor:
        return _tensor_copy(self._dt)

    @property
    def index(self) -> torch.Tensor:
        return _tensor_copy(self._index)

    @property
    def time(self) -> torch.Tensor:
        return _tensor_copy(self._time)
