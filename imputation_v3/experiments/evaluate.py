"""Overlap-safe, physical-unit evaluation for imputation v3."""

from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Integral
from typing import Any

import numpy as np


_AXES = ("gx", "gy", "gz", "ax", "ay", "az")


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _real_array(value: object, name: str, *, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    return array


def _binary_mask(value: object, name: str, *, shape: tuple[int, ...]) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != shape:
        raise ValueError(f"{name} shape must match the signal")
    if mask.dtype.kind not in "bifu":
        raise TypeError(f"{name} must be bool or real numeric")
    if not np.isfinite(mask).all() or not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError(f"{name} must contain exact binary values")
    return mask.astype(bool, copy=False)


def aggregate_raw_windows(
    raw_windows: Sequence[np.ndarray],
    *,
    starts: Sequence[int],
    recording_length: int,
) -> np.ndarray:
    """Average overlapping normalized raw predictions on the recording grid."""
    length = _positive_integer(recording_length, "recording_length")
    if isinstance(raw_windows, (str, bytes)) or not isinstance(raw_windows, Sequence):
        raise TypeError("raw_windows must be a non-empty sequence")
    if isinstance(starts, (str, bytes)) or not isinstance(starts, Sequence):
        raise TypeError("starts must be a sequence aligned with raw_windows")
    if not raw_windows:
        raise ValueError("raw_windows must be non-empty")
    if len(raw_windows) != len(starts):
        raise ValueError("raw_windows and starts must be aligned")

    first = _real_array(raw_windows[0], "each raw window", ndim=2)
    if first.shape[0] == 0 or first.shape[1] == 0:
        raise ValueError("raw windows must have non-empty time and channel axes")
    channels = first.shape[1]
    total = np.zeros((length, channels), dtype=np.float64)
    count = np.zeros((length, 1), dtype=np.int64)
    for raw_value, start_value in zip(raw_windows, starts):
        if isinstance(start_value, bool) or not isinstance(start_value, Integral):
            raise TypeError("window starts must be integers")
        start = int(start_value)
        raw = _real_array(raw_value, "each raw window", ndim=2)
        if raw.shape[0] == 0:
            raise ValueError("raw windows must not be empty")
        if raw.shape[1] != channels:
            raise ValueError("raw windows must have consistent channels")
        if not np.isfinite(raw).all():
            raise ValueError("raw windows must contain finite values")
        stop = start + raw.shape[0]
        if start < 0 or stop > length:
            raise ValueError("window lies outside the recording")
        total[start:stop] += raw
        count[start:stop] += 1
    if np.any(count == 0):
        raise ValueError("every evaluated sample must have prediction coverage")
    result = total / count
    if not np.isfinite(result).all():
        raise ValueError("aggregated prediction must be finite")
    return result


def physical_record_metrics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    missing: np.ndarray,
    recording_id: str,
) -> dict[str, Any]:
    """Calculate missing-only RMSE and MAE in physical units."""
    prediction_array = _real_array(prediction, "prediction", ndim=2)
    target_array = _real_array(target, "target", ndim=2)
    if prediction_array.shape != target_array.shape or prediction_array.size == 0:
        raise ValueError("prediction and target arrays must align and be non-empty")
    selected = _binary_mask(missing, "missing mask", shape=target_array.shape)
    if not selected.any():
        raise ValueError("physical metrics require at least one missing value")
    if not isinstance(recording_id, str) or not recording_id:
        raise ValueError("recording_id must be a non-empty string")
    if not np.isfinite(prediction_array[selected]).all() or not np.isfinite(
        target_array[selected]
    ).all():
        raise ValueError("selected prediction and target values must be finite")
    error = prediction_array[selected].astype(np.float64) - target_array[selected]
    absolute = np.abs(error)
    maximum = float(absolute.max())
    if maximum == 0.0:
        rmse = 0.0
        mae = 0.0
    else:
        scaled = absolute / maximum
        rmse = float(maximum * np.sqrt(np.mean(np.square(scaled))))
        mae = float(maximum * np.mean(scaled))
    if not math.isfinite(rmse) or not math.isfinite(mae):
        raise ValueError("physical metrics must be finite")
    return {"recording_id": recording_id, "rmse_physical": rmse, "mae_physical": mae}


def _gap_durations(missing: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Assign each missing run its elapsed duration on an irregular time grid."""
    durations = np.zeros(missing.shape, dtype=np.float64)
    sample_count = len(time)
    for channel in range(missing.shape[1]):
        start: int | None = None
        for index in range(sample_count + 1):
            active = index < sample_count and bool(missing[index, channel])
            if active and start is None:
                start = index
            elif not active and start is not None:
                stop = index
                if stop < sample_count:
                    duration = float(time[stop] - time[start])
                elif start > 0:
                    duration = float(time[-1] - time[start - 1])
                elif sample_count > 1:
                    duration = float(time[-1] - time[0] + (time[-1] - time[-2]))
                else:
                    duration = 0.0
                durations[start:stop, channel] = duration
                start = None
    return durations


def diagnostic_masks(missing: np.ndarray, time: np.ndarray) -> dict[str, np.ndarray]:
    """Return non-empty sensor, axis, and elapsed-gap diagnostic masks."""
    raw_missing = np.asarray(missing)
    if raw_missing.ndim != 2 or raw_missing.shape[1] != 6:
        raise ValueError("missing must have shape (time, six channels)")
    if raw_missing.shape[0] == 0:
        raise ValueError("missing time axis must not be empty")
    selected = _binary_mask(raw_missing, "missing", shape=raw_missing.shape)
    time_array = _real_array(time, "time", ndim=1).astype(np.float64, copy=False)
    if len(time_array) != len(selected):
        raise ValueError("time length must match missing")
    if not np.isfinite(time_array).all():
        raise ValueError("time must contain finite values")
    if len(time_array) > 1 and np.any(np.diff(time_array) <= 0):
        raise ValueError("time must be strictly increasing")

    channel_gyro = np.array([True, True, True, False, False, False])
    channel_accel = ~channel_gyro
    groups: dict[str, np.ndarray] = {
        "overall": selected.copy(),
        "sensor/gyro": selected & channel_gyro,
        "sensor/accelerometer": selected & channel_accel,
    }
    for channel, axis in enumerate(_AXES):
        axis_mask = np.zeros_like(selected)
        axis_mask[:, channel] = selected[:, channel]
        groups[f"axis/{axis}"] = axis_mask

    durations = _gap_durations(selected, time_array)
    groups.update(
        {
            "gap/0-50ms": selected & (durations <= 0.05),
            "gap/50-200ms": selected & (durations > 0.05) & (durations <= 0.2),
            "gap/over-200ms": selected & (durations > 0.2),
        }
    )
    return {
        name: np.array(mask, dtype=bool, order="C", copy=True)
        for name, mask in groups.items()
        if mask.any()
    }


def evaluate_record(
    *,
    raw_windows: Sequence[np.ndarray],
    starts: Sequence[int],
    target_normalized: np.ndarray,
    observed_mask: np.ndarray,
    scaler: object,
    recording_id: str,
) -> dict[str, Any]:
    """Aggregate windows, restore observations, inverse-scale, then score."""
    target = _real_array(target_normalized, "target_normalized", ndim=2)
    if target.shape[0] == 0 or target.shape[1] == 0:
        raise ValueError("target_normalized must be non-empty")
    if not np.isfinite(target).all():
        raise ValueError("target_normalized must contain finite values")
    observed = _binary_mask(observed_mask, "observed_mask", shape=target.shape)
    raw = aggregate_raw_windows(
        raw_windows, starts=starts, recording_length=len(target)
    )
    if raw.shape != target.shape:
        raise ValueError("aggregated prediction channels must match target_normalized")
    inverse = getattr(scaler, "inverse_transform", None)
    if not callable(inverse):
        raise TypeError("scaler must provide a callable inverse_transform")
    completed = np.where(observed, target, raw)
    prediction_physical = np.asarray(inverse(completed))
    target_physical = np.asarray(inverse(target))
    return physical_record_metrics(
        prediction=prediction_physical,
        target=target_physical,
        missing=~observed,
        recording_id=recording_id,
    )


def evaluate_record_diagnostics(
    *,
    raw_windows: Sequence[np.ndarray],
    starts: Sequence[int],
    target_normalized: np.ndarray,
    observed_mask: np.ndarray,
    time: np.ndarray,
    scaler: object,
    recording_id: str,
) -> dict[str, dict[str, Any]]:
    """Score all non-empty diagnostic strata after one recording aggregation."""
    target = _real_array(target_normalized, "target_normalized", ndim=2)
    if target.shape != (len(target), 6) or len(target) == 0:
        raise ValueError("diagnostic target_normalized must have shape (time, 6)")
    if not np.isfinite(target).all():
        raise ValueError("target_normalized must contain finite values")
    observed = _binary_mask(observed_mask, "observed_mask", shape=target.shape)
    raw = aggregate_raw_windows(
        raw_windows, starts=starts, recording_length=len(target)
    )
    if raw.shape != target.shape:
        raise ValueError("aggregated prediction channels must match target_normalized")
    inverse = getattr(scaler, "inverse_transform", None)
    if not callable(inverse):
        raise TypeError("scaler must provide a callable inverse_transform")
    prediction_physical = np.asarray(inverse(np.where(observed, target, raw)))
    target_physical = np.asarray(inverse(target))
    groups = diagnostic_masks(~observed, time)
    return {
        name: physical_record_metrics(
            prediction=prediction_physical,
            target=target_physical,
            missing=mask,
            recording_id=recording_id,
        )
        for name, mask in groups.items()
    }


__all__ = [
    "aggregate_raw_windows",
    "diagnostic_masks",
    "evaluate_record",
    "evaluate_record_diagnostics",
    "physical_record_metrics",
]
