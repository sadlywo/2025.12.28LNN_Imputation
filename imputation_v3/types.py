"""Immutable tensor records shared by imputation v3."""

from dataclasses import dataclass, field
import math
from numbers import Real

import torch


_TOPOLOGIES = frozenset(("point", "block", "channel"))


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

        validated_window_id = _identifier(window_id, "window_id")
        validated_recording_id = _identifier(recording_id, "recording_id")
        validated_topology = _identifier(topology, "topology")
        if validated_topology not in _TOPOLOGIES:
            raise ValueError("topology must be exactly 'point', 'block', or 'channel'")
        validated_requested = _fraction(requested_fraction, "requested_fraction")
        validated_realized = _fraction(realized_fraction, "realized_fraction")
        if validated_requested == 0.0:
            raise ValueError("requested_fraction must be positive")

        if time[0].item() != 0.0:
            raise ValueError("time must start at zero")
        if samples > 1:
            time_difference = time[1:] - time[:-1]
            scale = torch.maximum(
                torch.maximum(time[1:].abs(), time[:-1].abs()),
                torch.ones_like(time_difference),
            )
            tolerance = torch.finfo(time.dtype).eps * 8 * scale
            if not (torch.abs(dt[1:] - time_difference) <= tolerance).all().item():
                raise ValueError("dt[1:] must agree with adjacent time differences")

        expected_observed = torch.where(
            mask.bool(), target, torch.zeros_like(target)
        )
        if not torch.equal(observed, expected_observed):
            raise ValueError("observed must equal masked target with deterministic zeros")
        observed_positions = mask.bool()
        if not torch.equal(
            baseline[observed_positions], observed[observed_positions]
        ):
            raise ValueError("baseline must exactly preserve observed positions")
        realized_from_mask = float((mask == 0).to(torch.float64).mean().item())
        if realized_from_mask == 0.0:
            raise ValueError("mask must contain at least one missing entry")
        if not math.isclose(
            validated_realized,
            realized_from_mask,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("realized_fraction must agree with the mask missing mean")
        if not torch.equal(features[:, 0:6], observed):
            raise ValueError("feature columns 0:6 must equal observed")
        if not torch.equal(features[:, 6:12], mask):
            raise ValueError("feature columns 6:12 must equal mask")
        if not torch.equal(features[:, 12], dt):
            raise ValueError("feature column 12 must equal dt")

        for name, value in tensors.items():
            object.__setattr__(self, f"_{name}", _copy(value))
        object.__setattr__(self, "window_id", validated_window_id)
        object.__setattr__(self, "recording_id", validated_recording_id)
        object.__setattr__(self, "topology", validated_topology)
        object.__setattr__(self, "requested_fraction", validated_requested)
        object.__setattr__(self, "realized_fraction", validated_realized)

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


@dataclass(frozen=True, init=False, repr=False, eq=False)
class PreparedBatch:
    """Stacked prepared windows whose tensors are trusted as read-only.

    Tensor inputs are detached and cloned once at construction. Training code may
    read them without another full-batch clone, but must not mutate them in place.
    """

    features: torch.Tensor = field(repr=False)
    target: torch.Tensor = field(repr=False)
    observed: torch.Tensor = field(repr=False)
    mask: torch.Tensor = field(repr=False)
    dt: torch.Tensor = field(repr=False)
    time: torch.Tensor = field(repr=False)
    baseline: torch.Tensor = field(repr=False)
    window_ids: tuple[str, ...]
    recording_ids: tuple[str, ...]
    topologies: tuple[str, ...]
    requested_fractions: tuple[float, ...]
    realized_fractions: tuple[float, ...]

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
        window_ids: tuple[str, ...],
        recording_ids: tuple[str, ...],
        topologies: tuple[str, ...],
        requested_fractions: tuple[float, ...],
        realized_fractions: tuple[float, ...],
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
            if value.device.type == "meta":
                raise ValueError(f"{name} must contain materialized, non-meta values")
        if target.ndim != 3 or target.shape[0] == 0 or target.shape[2] != 6:
            raise ValueError("target must have non-empty shape (B, T, 6)")
        batch_size, samples, _ = target.shape
        for name, value in {
            "observed": observed,
            "mask": mask,
            "baseline": baseline,
        }.items():
            if value.shape != target.shape:
                raise ValueError(f"{name} must have shape (B, T, 6)")
        if features.shape != (batch_size, samples, 31):
            raise ValueError("features must have shape (B, T, 31)")
        for name, value in {"dt": dt, "time": time}.items():
            if value.shape != (batch_size, samples):
                raise ValueError(f"{name} must have shape (B, T)")
        for name, value in tensors.items():
            if value.dtype != target.dtype:
                raise ValueError(f"{name} dtype must match target dtype")
            if value.device != target.device:
                raise ValueError(f"{name} device must match target device")

        metadata = {
            "window_ids": tuple(window_ids),
            "recording_ids": tuple(recording_ids),
            "topologies": tuple(topologies),
            "requested_fractions": tuple(requested_fractions),
            "realized_fractions": tuple(realized_fractions),
        }
        for name, values in metadata.items():
            if len(values) != batch_size:
                raise ValueError(f"{name} must contain one value per batch item")
        validated_window_ids = tuple(
            _identifier(value, "window_ids item") for value in metadata["window_ids"]
        )
        validated_recording_ids = tuple(
            _identifier(value, "recording_ids item")
            for value in metadata["recording_ids"]
        )
        validated_topologies = tuple(
            _identifier(value, "topologies item") for value in metadata["topologies"]
        )
        if any(value not in _TOPOLOGIES for value in validated_topologies):
            raise ValueError("topologies must contain only point, block, or channel")
        validated_requested = tuple(
            _fraction(value, "requested_fractions item")
            for value in metadata["requested_fractions"]
        )
        validated_realized = tuple(
            _fraction(value, "realized_fractions item")
            for value in metadata["realized_fractions"]
        )

        for name, value in tensors.items():
            object.__setattr__(self, name, _copy(value))
        object.__setattr__(self, "window_ids", validated_window_ids)
        object.__setattr__(self, "recording_ids", validated_recording_ids)
        object.__setattr__(self, "topologies", validated_topologies)
        object.__setattr__(self, "requested_fractions", validated_requested)
        object.__setattr__(self, "realized_fractions", validated_realized)


__all__ = ["FeatureBatch", "PreparedBatch", "PreparedWindow"]
