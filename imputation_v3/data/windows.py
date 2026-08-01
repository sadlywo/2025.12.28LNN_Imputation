"""Deterministic, shared teacher-window materialization and batching.

Public topology shorthands map to Validation v2 generators as follows:
``point`` -> ``point_missing``, ``block`` -> ``contiguous_block``, and
``channel`` -> ``channel_outage``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import hashlib
import json
import math
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np
import torch

from imputation_v3.data.features import build_features
from imputation_v3.models.baselines import timestamp_linear
from imputation_v3.types import PreparedBatch, PreparedWindow
from validation_v2.data.masking import (
    channel_outage,
    contiguous_block,
    point_missing,
)
from validation_v2.data.normalization import RobustTrainScaler
from validation_v2.types import Recording


TOPOLOGY_GENERATOR_NAMES = MappingProxyType(
    {
        "point": "point_missing",
        "block": "contiguous_block",
        "channel": "channel_outage",
    }
)
_TOPOLOGY_ORDER = tuple(TOPOLOGY_GENERATOR_NAMES)
_GENERATORS = {
    "point": point_missing,
    "block": contiguous_block,
    "channel": channel_outage,
}
_MASK_GENERATOR_VERSION = "validation-v2-masking-v1"


def _positive_integer(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    converted = int(value)
    if converted < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return converted


def _experiment_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("seed must be an integer")
    return int(value)


def _validated_topologies(values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("topologies must be a non-empty iterable of topology names")
    try:
        supplied = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("topologies must be a non-empty iterable") from exc
    if not supplied:
        raise ValueError("topologies must not be empty")
    for topology in supplied:
        if not isinstance(topology, str):
            raise TypeError("each topology must be a string")
        if topology not in _GENERATORS:
            raise ValueError(
                "topologies must contain only 'point', 'block', and 'channel'"
            )
    if len(set(supplied)) != len(supplied):
        raise ValueError("topologies must not contain duplicates")
    supplied_set = set(supplied)
    return tuple(topology for topology in _TOPOLOGY_ORDER if topology in supplied_set)


def _validated_rates(values: object) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("rates must be a non-empty iterable of finite fractions")
    try:
        raw_rates = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("rates must be a non-empty iterable") from exc
    if not raw_rates:
        raise ValueError("rates must not be empty")
    result: list[float] = []
    for rate in raw_rates:
        if isinstance(rate, bool) or not isinstance(rate, Real):
            raise TypeError("each rate must be a finite real fraction in (0, 1]")
        converted = float(rate)
        if not math.isfinite(converted) or not 0.0 < converted <= 1.0:
            raise ValueError("each rate must be a finite positive fraction in (0, 1]")
        result.append(converted)
    if len(set(result)) != len(result):
        raise ValueError("rates must not contain duplicates")
    return tuple(sorted(result))


def _identity_bytes(*parts: object) -> bytes:
    return json.dumps(
        parts,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_rate(rate: float) -> str:
    return rate.hex()


def _update_length_prefixed(digest: object, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _fingerprint_arrays(label: str, arrays: tuple[np.ndarray, ...], *parts: object) -> str:
    digest = hashlib.sha256()
    _update_length_prefixed(digest, label.encode("utf-8"))
    _update_length_prefixed(digest, _identity_bytes(*parts))
    for array in arrays:
        canonical = np.ascontiguousarray(array, dtype="<f8")
        _update_length_prefixed(digest, _identity_bytes(canonical.shape))
        _update_length_prefixed(digest, canonical.tobytes(order="C"))
    return digest.hexdigest()


def _scaler_fingerprint(scaler: object) -> str:
    if type(scaler) is not RobustTrainScaler:
        raise TypeError("scaler must be an actual frozen RobustTrainScaler")
    if np.iscomplexobj(scaler.center_) or np.iscomplexobj(scaler.scale_):
        raise ValueError("frozen scaler arrays must not be complex")
    center = np.asarray(scaler.center_)
    scale = np.asarray(scaler.scale_)
    if center.shape != (6,) or scale.shape != (6,):
        raise ValueError("frozen scaler center_ and scale_ must have shape (6,)")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)):
        raise ValueError("frozen scaler arrays must be finite")
    if np.any(scale <= 0):
        raise ValueError("frozen scaler scale_ must be strictly positive")
    training_ids = tuple(scaler.training_ids)
    if (
        not training_ids
        or training_ids != tuple(sorted(training_ids))
        or len(set(training_ids)) != len(training_ids)
        or any(not isinstance(value, str) or not value for value in training_ids)
    ):
        raise ValueError("frozen scaler training_ids must be sorted unique strings")
    return _fingerprint_arrays(
        "robust-train-scaler-v1", (center, scale), training_ids
    )


def _condition_seed(
    recording_id: str,
    start: int,
    stop: int,
    experiment_seed: int,
    topology: str,
    rate: float,
) -> int:
    digest = hashlib.sha256(
        _identity_bytes(
            "teacher-mask-v1",
            recording_id,
            start,
            stop,
            experiment_seed,
            topology,
            _canonical_rate(rate),
        )
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _window_id(
    recording_id: str,
    recording_fingerprint: str,
    scaler_fingerprint: str,
    start: int,
    stop: int,
    experiment_seed: int,
    topology: str,
    rate: float,
) -> str:
    digest = hashlib.sha256(
        _identity_bytes(
            "teacher-window-v2",
            recording_id,
            recording_fingerprint,
            scaler_fingerprint,
            start,
            stop,
            experiment_seed,
            topology,
            _canonical_rate(rate),
            _MASK_GENERATOR_VERSION,
            TOPOLOGY_GENERATOR_NAMES[topology],
        )
    ).hexdigest()
    return f"teacher-window-sha256-{digest}"


def _training_condition(
    recording_id: str,
    start: int,
    stop: int,
    seed: int,
    topologies: tuple[str, ...],
    rates: tuple[float, ...],
) -> tuple[str, float]:
    digest = hashlib.sha256(
        _identity_bytes("teacher-curriculum-v1", recording_id, start, stop, seed)
    ).digest()
    topology = topologies[int.from_bytes(digest[:8], "big") % len(topologies)]
    rate = rates[int.from_bytes(digest[8:16], "big") % len(rates)]
    return topology, rate


def _recording_arrays(
    recording: Recording,
) -> tuple[np.ndarray, np.ndarray, str]:
    if not isinstance(recording.id, str):
        raise TypeError("recording id must be a non-empty string")
    if not recording.id:
        raise ValueError("recording id must be a non-empty string")
    raw_time = np.asarray(recording.imu_time_s)
    if np.iscomplexobj(raw_time):
        raise ValueError(f"recording {recording.id!r} IMU time must not be complex")
    try:
        time = np.asarray(raw_time, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"recording {recording.id!r} IMU time must be float64-compatible"
        ) from exc
    if time.ndim != 1:
        raise ValueError(f"recording {recording.id!r} IMU time must be one-dimensional")
    if not np.all(np.isfinite(time)):
        raise ValueError(f"recording {recording.id!r} IMU time must be finite")
    if time.size > 1 and not np.all(np.diff(time) > 0):
        raise ValueError(
            f"recording {recording.id!r} IMU time must be strictly increasing"
        )

    raw_imu = np.asarray(recording.imu_six)
    if np.iscomplexobj(raw_imu):
        raise ValueError(f"recording {recording.id!r} imu_six must not be complex")
    try:
        imu = np.asarray(raw_imu, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"recording {recording.id!r} imu_six must be numeric") from exc
    if imu.ndim != 2 or imu.shape[1:] != (6,):
        raise ValueError(f"recording {recording.id!r} imu_six must have shape (N, 6)")
    if imu.shape[0] != time.shape[0]:
        raise ValueError(
            f"recording {recording.id!r} IMU time and imu_six must be aligned"
        )
    if not np.all(np.isfinite(imu)):
        raise ValueError(f"recording {recording.id!r} imu_six must be finite")
    fingerprint = _fingerprint_arrays(
        "recording-imu-v1", (time, imu), recording.id
    )
    return time, imu, fingerprint


def _scaled_target(
    scaler: RobustTrainScaler, imu: np.ndarray, recording_id: str
) -> torch.Tensor:
    try:
        raw_output = scaler.transform(imu)
    except Exception as exc:
        raise ValueError(
            f"scaler transform failed for recording {recording_id!r}"
        ) from exc
    if np.iscomplexobj(raw_output):
        raise ValueError("scaler transform output must not be complex")
    try:
        scaled = np.asarray(raw_output, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("scaler transform output must be a numeric array") from exc
    if scaled.shape != imu.shape:
        raise ValueError("scaler transform output must have aligned shape (N, 6)")
    if not np.all(np.isfinite(scaled)):
        raise ValueError("scaler transform output must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        scaled32 = scaled.astype(np.float32)
    if not np.all(np.isfinite(scaled32)):
        raise ValueError("scaler transform output must be representable as finite float32")
    owned_c_order = np.array(scaled32, dtype=np.float32, order="C", copy=True)
    return torch.from_numpy(owned_c_order)


def _local_time_and_dt(raw_time: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    with np.errstate(over="ignore", invalid="ignore"):
        local64 = raw_time - raw_time[0]
        internal_intervals = np.diff(raw_time)
    local_dt64 = np.empty(raw_time.shape, dtype=np.float64)
    local_dt64[1:] = internal_intervals
    local_dt64[0] = np.median(internal_intervals)
    with np.errstate(over="ignore", invalid="ignore"):
        local32 = local64.astype(np.float32)
        dt32 = local_dt64.astype(np.float32)
    if not np.all(np.isfinite(local32)) or not np.all(np.isfinite(dt32)):
        raise ValueError("window time and dt must be representable as finite float32")
    if not np.all(dt32 > 0) or not np.all(np.diff(local32) > 0):
        raise ValueError(
            "window time and dt must remain strictly increasing and positive in float32"
        )
    return (
        torch.from_numpy(np.array(local32, copy=True)),
        torch.from_numpy(np.array(dt32, copy=True)),
    )


def _conditions(
    *,
    exhaustive: bool,
    recording_id: str,
    start: int,
    stop: int,
    seed: int,
    topologies: tuple[str, ...],
    rates: tuple[float, ...],
) -> Iterator[tuple[str, float]]:
    if not exhaustive:
        yield _training_condition(
            recording_id, start, stop, seed, topologies, rates
        )
        return
    for topology in topologies:
        for rate in rates:
            yield topology, rate


def iter_teacher_windows(
    recordings: Iterable[Recording],
    scaler: RobustTrainScaler,
    *,
    window_samples: int,
    stride: int,
    seed: int,
    topologies: Iterable[str],
    rates: Iterable[float],
    exhaustive: bool = False,
) -> Iterator[PreparedWindow]:
    """Yield prepared conditions immediately in deterministic semantic order.

    Formal exhaustive evaluation should consume this iterator condition by
    condition so the topology-by-rate Cartesian product is never retained in
    memory as a collection of tensor records.
    """

    window_samples = _positive_integer(window_samples, "window_samples", minimum=2)
    stride = _positive_integer(stride, "stride")
    seed = _experiment_seed(seed)
    topology_values = _validated_topologies(topologies)
    rate_values = _validated_rates(rates)
    if not isinstance(exhaustive, bool):
        raise TypeError("exhaustive must be a bool")
    if isinstance(recordings, (str, bytes)) or not isinstance(recordings, Iterable):
        raise TypeError("recordings must be an iterable of Recording objects")
    scaler_identity = _scaler_fingerprint(scaler)
    recording_values = list(recordings)
    for recording in recording_values:
        if not isinstance(recording, Recording):
            raise TypeError("recordings must contain only Recording objects")
    ids = [recording.id for recording in recording_values]
    for recording_id in ids:
        if not isinstance(recording_id, str):
            raise TypeError("recording id must be a non-empty string")
        if not recording_id:
            raise ValueError("recording id must be a non-empty string")
    if len(set(ids)) != len(ids):
        raise ValueError("recording ids must be unique")

    yielded = False
    seen_window_ids: set[str] = set()
    for recording in sorted(recording_values, key=lambda item: item.id):
        raw_time, imu, recording_identity = _recording_arrays(recording)
        full_target = _scaled_target(scaler, imu, recording.id)
        for start in range(0, len(raw_time) - window_samples + 1, stride):
            stop = start + window_samples
            local_time, local_dt = _local_time_and_dt(raw_time[start:stop])
            target = full_target[start:stop].contiguous()
            for topology, rate in _conditions(
                exhaustive=exhaustive,
                recording_id=recording.id,
                start=start,
                stop=stop,
                seed=seed,
                topologies=topology_values,
                rates=rate_values,
            ):
                generator = _GENERATORS[topology]
                generator_topology = TOPOLOGY_GENERATOR_NAMES[topology]
                condition_seed = _condition_seed(
                    recording.id, start, stop, seed, topology, rate
                )
                mask_result = generator(target, rate, condition_seed)
                if (
                    mask_result.topology != generator_topology
                    or mask_result.requested_fraction != rate
                    or mask_result.seed != condition_seed
                ):
                    raise ValueError("mask generator result does not match its request")
                mask = mask_result.mask.contiguous()
                realized = float((mask == 0).to(torch.float64).mean().item())
                if realized == 0.0:
                    raise ValueError(
                        f"{topology} at requested_fraction={rate} realizes no missing entries"
                    )
                if not math.isclose(
                    mask_result.realized_fraction,
                    realized,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise ValueError("mask generator realized fraction is inconsistent")
                observed = torch.where(mask.bool(), target, torch.zeros_like(target))
                features = build_features(target, mask, local_dt).values
                baseline = timestamp_linear(
                    observed, mask, local_time, empty_fill=0.0
                )
                identifier = _window_id(
                    recording.id,
                    recording_identity,
                    scaler_identity,
                    start,
                    stop,
                    seed,
                    topology,
                    rate,
                )
                if identifier in seen_window_ids:
                    raise ValueError("materialized window ids must be unique")
                seen_window_ids.add(identifier)
                yielded = True
                yield PreparedWindow(
                    features=features,
                    target=target,
                    observed=observed,
                    mask=mask,
                    dt=local_dt,
                    time=local_time,
                    baseline=baseline,
                    window_id=identifier,
                    recording_id=recording.id,
                    topology=topology,
                    requested_fraction=mask_result.requested_fraction,
                    realized_fraction=mask_result.realized_fraction,
                )
    if not yielded:
        raise ValueError("recordings produced no complete windows")


def materialize_teacher_windows(
    recordings: Iterable[Recording],
    scaler: RobustTrainScaler,
    *,
    window_samples: int,
    stride: int,
    seed: int,
    topologies: Iterable[str],
    rates: Iterable[float],
    exhaustive: bool = False,
) -> list[PreparedWindow]:
    """Collect a bounded prepared-window iterator into a list.

    Use :func:`iter_teacher_windows` directly for formal exhaustive evaluation;
    materializing an exhaustive Cartesian product retains every tensor record.
    """

    return list(
        iter_teacher_windows(
            recordings,
            scaler,
            window_samples=window_samples,
            stride=stride,
            seed=seed,
            topologies=topologies,
            rates=rates,
            exhaustive=exhaustive,
        )
    )


def collate_prepared_windows(items: Iterable[PreparedWindow]) -> PreparedBatch:
    """Stack a non-empty, compatible sequence for ``DataLoader.collate_fn``."""

    windows = list(items)
    if not windows:
        raise ValueError("prepared-window batch must be non-empty")
    if any(not isinstance(window, PreparedWindow) for window in windows):
        raise TypeError("batch items must all be PreparedWindow records")
    first = windows[0]
    samples = first.target.shape[0]
    dtype = first.target.dtype
    device = first.target.device
    for window in windows[1:]:
        if window.target.shape[0] != samples:
            raise ValueError("prepared windows must have the same T shape")
        if window.target.dtype != dtype:
            raise ValueError("prepared windows must have the same dtype")
        if window.target.device != device:
            raise ValueError("prepared windows must be on the same device")

    tensor_names = (
        "features",
        "target",
        "observed",
        "mask",
        "dt",
        "time",
        "baseline",
    )
    stacked = {
        name: torch.stack([getattr(window, name) for window in windows], dim=0)
        for name in tensor_names
    }
    return PreparedBatch(
        **stacked,
        window_ids=tuple(window.window_id for window in windows),
        recording_ids=tuple(window.recording_id for window in windows),
        topologies=tuple(window.topology for window in windows),
        requested_fractions=tuple(window.requested_fraction for window in windows),
        realized_fractions=tuple(window.realized_fraction for window in windows),
    )


__all__ = [
    "TOPOLOGY_GENERATOR_NAMES",
    "collate_prepared_windows",
    "iter_teacher_windows",
    "materialize_teacher_windows",
]
