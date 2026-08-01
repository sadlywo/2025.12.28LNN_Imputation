"""Deterministic, shared teacher-window materialization."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch

from imputation_v3.data.features import build_features
from imputation_v3.models.baselines import timestamp_linear
from imputation_v3.types import PreparedWindow
from validation_v2.data.masking import (
    channel_outage,
    contiguous_block,
    point_missing,
)
from validation_v2.types import Recording


_GENERATORS = {
    "point": (point_missing, "point_missing"),
    "block": (contiguous_block, "contiguous_block"),
    "channel": (channel_outage, "channel_outage"),
}


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
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("topologies must be a non-empty iterable") from exc
    if not result:
        raise ValueError("topologies must not be empty")
    for topology in result:
        if not isinstance(topology, str):
            raise TypeError("each topology must be a string")
        if topology not in _GENERATORS:
            raise ValueError(
                "topologies must contain only 'point', 'block', and 'channel'"
            )
    if len(set(result)) != len(result):
        raise ValueError("topologies must not contain duplicates")
    return result


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
            raise TypeError("each rate must be a finite real fraction in [0, 1]")
        converted = float(rate)
        if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
            raise ValueError("each rate must be a finite real fraction in [0, 1]")
        if converted == 0.0:
            converted = 0.0
        result.append(converted)
    if len(set(result)) != len(result):
        raise ValueError("rates must not contain duplicates")
    return tuple(result)


def _identity_bytes(*parts: object) -> bytes:
    return json.dumps(
        parts,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_rate(rate: float) -> str:
    return (0.0 if rate == 0.0 else rate).hex()


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
    start: int,
    stop: int,
    experiment_seed: int,
    topology: str,
    rate: float,
) -> str:
    digest = hashlib.sha256(
        _identity_bytes(
            "teacher-window-v1",
            recording_id,
            start,
            stop,
            experiment_seed,
            topology,
            _canonical_rate(rate),
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
) -> tuple[tuple[str, float], ...]:
    digest = hashlib.sha256(
        _identity_bytes("teacher-curriculum-v1", recording_id, start, stop, seed)
    ).digest()
    topology = topologies[int.from_bytes(digest[:8], "big") % len(topologies)]
    rate = rates[int.from_bytes(digest[8:16], "big") % len(rates)]
    return ((topology, rate),)


def _recording_arrays(recording: Recording) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(recording.id, str):
        raise TypeError("recording id must be a non-empty string")
    if not recording.id:
        raise ValueError("recording id must be a non-empty string")
    try:
        time = np.asarray(recording.imu_time_s, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"recording {recording.id!r} IMU time must be float64-compatible") from exc
    if time.ndim != 1:
        raise ValueError(f"recording {recording.id!r} IMU time must be one-dimensional")
    if not np.all(np.isfinite(time)):
        raise ValueError(f"recording {recording.id!r} IMU time must be finite")
    if time.size > 1 and not np.all(np.diff(time) > 0):
        raise ValueError(
            f"recording {recording.id!r} IMU time must be strictly increasing"
        )
    try:
        imu = np.asarray(recording.imu_six, dtype=np.float64)
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
    return time, imu


def _scaled_target(scaler: Any, imu: np.ndarray, recording_id: str) -> torch.Tensor:
    transform = getattr(scaler, "transform", None)
    if not callable(transform):
        raise TypeError("scaler must provide a callable transform method")
    try:
        raw_output = transform(imu)
    except Exception as exc:
        raise ValueError(
            f"scaler transform failed for recording {recording_id!r}"
        ) from exc
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
    return torch.from_numpy(np.array(scaled32, copy=True))


def _local_time_and_dt(raw_time: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
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


def materialize_teacher_windows(
    recordings: Iterable[Recording],
    scaler: Any,
    *,
    window_samples: int,
    stride: int,
    seed: int,
    topologies: Iterable[str],
    rates: Iterable[float],
    exhaustive: bool = False,
) -> list[PreparedWindow]:
    """Freeze deterministic physical windows and their missingness conditions."""

    window_samples = _positive_integer(window_samples, "window_samples", minimum=2)
    stride = _positive_integer(stride, "stride")
    seed = _experiment_seed(seed)
    topology_values = _validated_topologies(topologies)
    rate_values = _validated_rates(rates)
    if not isinstance(exhaustive, bool):
        raise TypeError("exhaustive must be a bool")
    if isinstance(recordings, (str, bytes)) or not isinstance(recordings, Iterable):
        raise TypeError("recordings must be an iterable of Recording objects")
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

    prepared: list[PreparedWindow] = []
    seen_window_ids: set[str] = set()
    for recording in sorted(recording_values, key=lambda item: item.id):
        raw_time, imu = _recording_arrays(recording)
        full_target = _scaled_target(scaler, imu, recording.id)
        for start in range(0, len(raw_time) - window_samples + 1, stride):
            stop = start + window_samples
            local_time, local_dt = _local_time_and_dt(raw_time[start:stop])
            target = full_target[start:stop]
            conditions = (
                tuple(
                    (topology, rate)
                    for topology in topology_values
                    for rate in rate_values
                )
                if exhaustive
                else _training_condition(
                    recording.id,
                    start,
                    stop,
                    seed,
                    topology_values,
                    rate_values,
                )
            )
            for topology, rate in conditions:
                generator, generator_topology = _GENERATORS[topology]
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
                mask = mask_result.mask
                realized = float((mask == 0).to(torch.float64).mean().item())
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
                    recording.id, start, stop, seed, topology, rate
                )
                if identifier in seen_window_ids:
                    raise ValueError("materialized window ids must be unique")
                seen_window_ids.add(identifier)
                prepared.append(
                    PreparedWindow(
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
                )
    if not prepared:
        raise ValueError("recordings produced no complete windows")
    return prepared


__all__ = ["materialize_teacher_windows"]
