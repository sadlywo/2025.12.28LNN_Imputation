from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np


def _validated_samples(
    samples: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    observed_mask = np.asarray(mask)
    if values.ndim < 2 or values.shape[0] < 2:
        raise ValueError("samples must contain at least two draws on axis 0")
    if values.shape[1:] != truth.shape or observed_mask.shape != truth.shape:
        raise ValueError("sample, target, and mask shapes do not align")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(truth)):
        raise ValueError("samples and target must be finite")
    missing = observed_mask == 0
    if not np.any(missing):
        raise ValueError("probabilistic metrics require at least one missing value")
    return values, truth, missing


def empirical_crps(
    samples: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> float:
    values, truth, missing = _validated_samples(samples, target, mask)
    missing_values = values[:, missing]
    missing_truth = truth[missing]
    first = np.mean(np.abs(missing_values - missing_truth[None, :]), axis=0)
    pairwise = np.mean(
        np.abs(missing_values[:, None, :] - missing_values[None, :, :]),
        axis=(0, 1),
    )
    return float(np.mean(first - 0.5 * pairwise))


def interval_metrics(
    samples: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    level: float,
) -> tuple[float, float]:
    values, truth, missing = _validated_samples(samples, target, mask)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("interval level must be strictly between 0 and 1")
    alpha = 1.0 - level
    low, high = np.quantile(
        values, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0
    )
    missing_truth = truth[missing]
    coverage = np.mean(
        (missing_truth >= low[missing]) & (missing_truth <= high[missing])
    )
    width = np.mean(high[missing] - low[missing])
    return float(coverage), float(width)


def stitch_samples(
    window_samples: np.ndarray, starts: Sequence[int], length: int
) -> np.ndarray:
    values = np.asarray(window_samples, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError(
            "window_samples must have shape (windows, samples, steps, features)"
        )
    window_count, sample_count, step_count, feature_count = values.shape
    if sample_count < 2:
        raise ValueError("stitching requires at least two samples")
    if min(window_count, step_count, feature_count, length) <= 0:
        raise ValueError("window and output dimensions must be positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("window samples must be finite")
    normalized_starts = tuple(starts)
    if len(normalized_starts) != window_count or any(
        isinstance(start, bool) or not isinstance(start, (int, np.integer))
        for start in normalized_starts
    ):
        raise ValueError("one integer start is required per window")
    normalized_starts = tuple(int(start) for start in normalized_starts)
    if tuple(sorted(set(normalized_starts))) != normalized_starts:
        raise ValueError("window starts must be unique and sorted")
    if normalized_starts[0] < 0 or any(
        start + step_count > length for start in normalized_starts
    ):
        raise ValueError("window extends outside the requested output length")

    totals = np.zeros((sample_count, length, feature_count), dtype=np.float64)
    counts = np.zeros(length, dtype=np.int64)
    for window_index, start in enumerate(normalized_starts):
        stop = start + step_count
        totals[:, start:stop, :] += values[window_index]
        counts[start:stop] += 1
    if np.any(counts == 0):
        raise ValueError("window starts leave uncovered output positions")
    return totals / counts[None, :, None]


__all__ = ["empirical_crps", "interval_metrics", "stitch_samples"]
