"""Immutable tensor records shared by imputation v3."""

from dataclasses import dataclass, field
import math
from numbers import Real

import torch


def _copy(value: torch.Tensor) -> torch.Tensor:
    """Return tensor storage isolated from callers and autograd history."""

    return value.detach().clone()


@dataclass(frozen=True, init=False, repr=False, eq=False)
class FeatureBatch:
    """Leakage-safe features with their aligned time steps and mask."""

    _values: torch.Tensor = field(repr=False)
    _dt: torch.Tensor = field(repr=False)
    _mask: torch.Tensor = field(repr=False)

    def __init__(self, values: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor) -> None:
        object.__setattr__(self, "_values", _copy(values))
        object.__setattr__(self, "_dt", _copy(dt))
        object.__setattr__(self, "_mask", _copy(mask))

    @property
    def values(self) -> torch.Tensor:
        return _copy(self._values)

    @property
    def dt(self) -> torch.Tensor:
        return _copy(self._dt)

    @property
    def mask(self) -> torch.Tensor:
        return _copy(self._mask)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a non-empty string")
    if not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real fraction in [0, 1]")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be a finite real fraction in [0, 1]")
    return converted


@dataclass(frozen=True, init=False, repr=False, eq=False)
class PreparedWindow:
    """One fully materialized, immutable teacher-training condition."""

    _features: torch.Tensor = field(repr=False)
    _target: torch.Tensor = field(repr=False)
    _observed: torch.Tensor = field(repr=False)
    _mask: torch.Tensor = field(repr=False)
    _dt: torch.Tensor = field(repr=False)
    _time: torch.Tensor = field(repr=False)
    _baseline: torch.Tensor = field(repr=False)
    window_id: str
    recording_id: str
    topology: str
    requested_fraction: float
    realized_fraction: float

    def __init__(
        self,
        *,
        features: torch.Tensor,
        target: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
        dt: torch.Tensor,
        time: torch.Tensor,
        baseline: torch.Tensor,
        window_id: str,
        recording_id: str,
        topology: str,
        requested_fraction: float,
        realized_fraction: float,
    ) -> None:
        tensors = {
            "features": features,
            "target": target,
            "observed": observed,
            "mask": mask,
            "dt": dt,
            "time": time,
            "baseline": baseline,
        }
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch tensor")
            if not value.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")

        if target.ndim != 2 or target.shape[0] == 0 or target.shape[1] != 6:
            raise ValueError("target must have non-empty shape (T, 6)")
        samples = target.shape[0]
        for name, value in {
            "observed": observed,
            "mask": mask,
            "baseline": baseline,
        }.items():
            if value.shape != target.shape:
                raise ValueError(f"{name} must have the same (T, 6) shape as target")
        if features.shape != (samples, 31):
            raise ValueError("features must have shape (T, 31)")
        for name, value in {"dt": dt, "time": time}.items():
            if value.shape != (samples,):
                raise ValueError(f"{name} must have shape (T,)")

        reference_dtype = target.dtype
        reference_device = target.device
        for name, value in tensors.items():
            if value.dtype != reference_dtype:
                raise ValueError(f"{name} dtype must match target dtype")
            if value.device != reference_device:
                raise ValueError(f"{name} device must match target device")

        for name, value in tensors.items():
            if value.device.type == "meta":
                raise ValueError(f"{name} must contain materialized, non-meta values")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must contain only finite values")
        if not torch.all((mask == 0) | (mask == 1)).item():
            raise ValueError("mask values must be exactly binary 0 or 1")
        if not (dt > 0).all().item():
            raise ValueError("dt must contain strictly positive values")
        if samples > 1 and not (time[1:] > time[:-1]).all().item():
            raise ValueError("time must be strictly increasing")

        for name, value in tensors.items():
            object.__setattr__(self, f"_{name}", _copy(value))
        object.__setattr__(self, "window_id", _identifier(window_id, "window_id"))
        object.__setattr__(
            self, "recording_id", _identifier(recording_id, "recording_id")
        )
        object.__setattr__(self, "topology", _identifier(topology, "topology"))
        object.__setattr__(
            self,
            "requested_fraction",
            _fraction(requested_fraction, "requested_fraction"),
        )
        object.__setattr__(
            self,
            "realized_fraction",
            _fraction(realized_fraction, "realized_fraction"),
        )

    @property
    def features(self) -> torch.Tensor:
        return _copy(self._features)

    @property
    def target(self) -> torch.Tensor:
        return _copy(self._target)

    @property
    def observed(self) -> torch.Tensor:
        return _copy(self._observed)

    @property
    def mask(self) -> torch.Tensor:
        return _copy(self._mask)

    @property
    def dt(self) -> torch.Tensor:
        return _copy(self._dt)

    @property
    def time(self) -> torch.Tensor:
        return _copy(self._time)

    @property
    def baseline(self) -> torch.Tensor:
        return _copy(self._baseline)


__all__ = ["FeatureBatch", "PreparedWindow"]
