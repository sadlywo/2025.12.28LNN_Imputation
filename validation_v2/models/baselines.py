"""Observed-only diagnostic imputers and hybrid ablations."""

from __future__ import annotations

import torch

from .hybrid import complete_signal, fuse


def _validate_observed(observed: torch.Tensor, mask: torch.Tensor) -> None:
    if not isinstance(observed, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("observed and mask must be torch tensors")
    if observed.ndim != 3 or observed.shape != mask.shape:
        raise ValueError("observed and mask must share shape (batch, time, channel)")
    if observed.device != mask.device:
        raise ValueError("observed and mask must be on the same device")
    if observed.dtype != mask.dtype:
        raise TypeError("observed and mask must have the same dtype")
    if not observed.is_floating_point():
        raise TypeError("observed signal must be floating point")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must contain only 0 and 1")
    if not torch.isfinite(observed[mask.bool()]).all():
        raise ValueError("observed entries must be finite")


def _empty_series_fill(
    observed: torch.Tensor,
    mask: torch.Tensor,
    value: float | None,
) -> torch.Tensor | None:
    if mask.bool().any(dim=1).all():
        return None
    if value is None:
        raise ValueError("a batch/channel series has no observed value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("empty_series_fill must be a finite number")
    fill = observed.new_tensor(value)
    if not torch.isfinite(fill):
        raise ValueError("empty_series_fill must be a finite number")
    return fill


def locf(
    observed: torch.Tensor,
    mask: torch.Tensor,
    *,
    empty_series_fill: float | None = None,
) -> torch.Tensor:
    """Carry observations forward, optionally filling an all-missing series."""

    _validate_observed(observed, mask)
    fill = _empty_series_fill(observed, mask, empty_series_fill)
    prediction = torch.empty_like(observed)
    batch_size, time_steps, channels = observed.shape
    for batch in range(batch_size):
        for channel in range(channels):
            valid = torch.where(mask[batch, :, channel].bool())[0]
            if valid.numel() == 0:
                prediction[batch, :, channel] = fill
                continue
            current = observed[batch, valid[0], channel]
            for step in range(time_steps):
                if mask[batch, step, channel].bool():
                    current = observed[batch, step, channel]
                prediction[batch, step, channel] = current
    return complete_signal(observed, mask, prediction)


def linear_interpolation(
    observed: torch.Tensor,
    mask: torch.Tensor,
    *,
    empty_series_fill: float | None = None,
) -> torch.Tensor:
    """Interpolate each series, optionally filling one with no observations."""

    _validate_observed(observed, mask)
    fill = _empty_series_fill(observed, mask, empty_series_fill)
    prediction = torch.empty_like(observed)
    batch_size, time_steps, channels = observed.shape
    timeline = torch.arange(time_steps, device=observed.device)
    for batch in range(batch_size):
        for channel in range(channels):
            valid = torch.where(mask[batch, :, channel].bool())[0]
            if valid.numel() == 0:
                prediction[batch, :, channel] = fill
                continue
            values = observed[batch, valid, channel]
            right = torch.searchsorted(valid, timeline).clamp(max=valid.numel() - 1)
            left = (right - 1).clamp(min=0)
            left = torch.where(timeline >= valid[-1], right, left)
            left_index, right_index = valid[left], valid[right]
            left_value, right_value = values[left], values[right]
            span = right_index - left_index
            weight = torch.where(
                span > 0,
                (timeline - left_index).to(observed.dtype) / span.clamp(min=1),
                torch.zeros(time_steps, dtype=observed.dtype, device=observed.device),
            )
            prediction[batch, :, channel] = left_value + weight * (
                right_value - left_value
            )
    return complete_signal(observed, mask, prediction)


def single_branch(
    observed: torch.Tensor, mask: torch.Tensor, branch_prediction: torch.Tensor
) -> torch.Tensor:
    """Complete with one branch under the identical observed-information budget."""

    return complete_signal(observed, mask, branch_prediction)


def fixed_gate(
    observed: torch.Tensor,
    mask: torch.Tensor,
    lnn_prediction: torch.Tensor,
    lstm_prediction: torch.Tensor,
    lnn_gate: float,
) -> torch.Tensor:
    """Complete with a fixed LNN weight (normally 0, 0.5, or 1)."""

    if isinstance(lnn_gate, bool) or not isinstance(lnn_gate, (int, float)):
        raise TypeError("lnn_gate must be a number")
    if not isinstance(lnn_prediction, torch.Tensor) or not isinstance(
        lstm_prediction, torch.Tensor
    ):
        raise TypeError("branch predictions must be torch tensors")
    if not lnn_prediction.is_floating_point() or not lstm_prediction.is_floating_point():
        raise TypeError("branch predictions must be floating point")
    gate = lnn_prediction.new_tensor(lnn_gate)
    return complete_signal(observed, mask, fuse(lnn_prediction, lstm_prediction, gate))


def equal_average(
    observed: torch.Tensor,
    mask: torch.Tensor,
    lnn_prediction: torch.Tensor,
    lstm_prediction: torch.Tensor,
) -> torch.Tensor:
    """Complete with an equal, declared 0.5/0.5 branch average."""

    return fixed_gate(observed, mask, lnn_prediction, lstm_prediction, 0.5)
