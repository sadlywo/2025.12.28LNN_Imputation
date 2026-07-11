"""Recording-safe window generation with no target-derived statistics."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from validation_v2.types import WindowBatch


def _recording_ids(recording_id: str | Sequence[str], length: int) -> tuple[str, ...]:
    if isinstance(recording_id, str):
        if not recording_id:
            raise ValueError("recording_id must be non-empty")
        return (recording_id,) * length
    ids = tuple(recording_id)
    if len(ids) != length or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("recording_id must provide one non-empty string per sample")
    return ids


def make_windows(
    target: torch.Tensor,
    mask: torch.Tensor,
    dt: torch.Tensor,
    index: torch.Tensor,
    time: torch.Tensor,
    recording_id: str | Sequence[str],
    *,
    window_size: int,
    stride: int | None = None,
    drop_last: bool = True,
) -> tuple[WindowBatch, ...]:
    """Cut windows within recording and consecutive-index boundaries.

    ``stride=None`` means non-overlapping windows (``stride=window_size``).
    Short trailing segments are dropped by default or retained when
    ``drop_last=False``.
    """

    values = (target, mask, dt, index, time)
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("target, mask, dt, index, and time must be torch tensors")
    if target.ndim != 2 or target.shape[1] != 6 or target.shape[0] == 0:
        raise ValueError("target must have non-empty shape (samples, 6)")
    length = target.shape[0]
    if mask.shape != target.shape or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must match target and contain only 0 or 1")
    for name, value in (("dt", dt), ("index", index), ("time", time)):
        if value.ndim != 1 or value.shape[0] != length:
            raise ValueError(f"{name} must be one-dimensional and align with target")
    if not torch.isfinite(dt).all() or not (dt > 0).all():
        raise ValueError("dt must be finite and strictly positive")
    if not torch.isfinite(time).all():
        raise ValueError("time must be finite")
    if torch.isinf(target).any() or not torch.isfinite(target[mask.bool()]).all():
        raise ValueError("observed target values must be finite")
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if stride is None:
        stride = window_size
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")
    ids = _recording_ids(recording_id, length)

    boundaries = [0]
    for position in range(1, length):
        same_recording = ids[position] == ids[position - 1]
        consecutive = index[position].item() == index[position - 1].item() + 1
        increasing_time = time[position].item() > time[position - 1].item()
        if not (same_recording and consecutive and increasing_time):
            boundaries.append(position)
    boundaries.append(length)

    windows: list[WindowBatch] = []
    for run_start, run_end in zip(boundaries[:-1], boundaries[1:]):
        run_length = run_end - run_start
        for offset in range(0, run_length, stride):
            stop_offset = min(offset + window_size, run_length)
            if stop_offset - offset < window_size and drop_last:
                break
            start = run_start + offset
            stop = run_start + stop_offset
            windows.append(
                WindowBatch(
                    target=target[start:stop],
                    mask=mask[start:stop],
                    dt=dt[start:stop],
                    index=index[start:stop],
                    time=time[start:stop],
                    recording_id=ids[start],
                )
            )
            if stop_offset == run_length:
                break
    return tuple(windows)


__all__ = ["WindowBatch", "make_windows"]
