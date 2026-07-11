"""Deterministic value-missingness and separate irregular-time generators."""

from __future__ import annotations

import math

import torch

from validation_v2.types import IrregularTimeResult, MaskResult


def _fraction(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite fraction in [0, 1]") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    return result


def _template(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("mask template must be a torch tensor")
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError("mask template must be a non-empty two-dimensional tensor")
    if not value.is_floating_point():
        raise TypeError("mask template must have a floating-point dtype")
    return value


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("seed must be an integer")
    return value


def _half_up_count(total: int, fraction: float) -> int:
    return min(total, int(math.floor(total * fraction + 0.5)))


def _result(
    mask: torch.Tensor,
    requested_fraction: float,
    topology: str,
    seed: int,
    masked_channels: int | None = None,
) -> MaskResult:
    realized = float((mask == 0).to(torch.float64).mean().item())
    return MaskResult(
        mask=mask,
        requested_fraction=requested_fraction,
        realized_fraction=realized,
        topology=topology,
        seed=seed,
        masked_channels=masked_channels,
    )


def point_missing(
    template: torch.Tensor, requested_fraction: float, seed: int
) -> MaskResult:
    """Mask an exact half-up-rounded number of individual values."""

    template = _template(template)
    requested_fraction = _fraction(requested_fraction, "requested_fraction")
    seed = _seed(seed)
    count = _half_up_count(template.numel(), requested_fraction)
    mask = torch.ones_like(template)
    if count:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selected = torch.randperm(template.numel(), generator=generator)[:count]
        mask.reshape(-1)[selected.to(mask.device)] = 0
    return _result(mask, requested_fraction, "point_missing", seed)


def contiguous_block(
    template: torch.Tensor, requested_fraction: float, seed: int
) -> MaskResult:
    """Mask one contiguous, in-bounds time block across all channels."""

    template = _template(template)
    requested_fraction = _fraction(requested_fraction, "requested_fraction")
    seed = _seed(seed)
    rows = _half_up_count(template.shape[0], requested_fraction)
    mask = torch.ones_like(template)
    if rows:
        if rows == template.shape[0]:
            start = 0
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            start = int(
                torch.randint(
                    0, template.shape[0] - rows + 1, (1,), generator=generator
                ).item()
            )
        mask[start : start + rows] = 0
    return _result(mask, requested_fraction, "contiguous_block", seed)


def channel_outage(
    template: torch.Tensor, requested_fraction: float, seed: int
) -> MaskResult:
    """Mask whole channels, with one-channel minimum for a positive request."""

    template = _template(template)
    requested_fraction = _fraction(requested_fraction, "requested_fraction")
    seed = _seed(seed)
    floored = int(math.floor(template.shape[1] * requested_fraction))
    channels = min(
        template.shape[1],
        max(1, floored) if requested_fraction > 0 else 0,
    )
    mask = torch.ones_like(template)
    if channels:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selected = torch.randperm(template.shape[1], generator=generator)[:channels]
        mask[:, selected.to(mask.device)] = 0
    return _result(
        mask, requested_fraction, "channel_outage", seed, masked_channels=channels
    )


def generate_interval_jittered_time(
    time: torch.Tensor,
    requested_irregularity: float,
    seed: int,
    jitter_fraction: float = 0.25,
) -> IrregularTimeResult:
    """Jitter selected original intervals without masking or dropping values.

    The original sample indices are retained. This generator represents only
    interval jitter; it does not claim packet dropping or resampling.
    """

    if not isinstance(time, torch.Tensor):
        raise TypeError("time must be a torch tensor")
    if time.ndim != 1 or time.numel() < 2 or not time.is_floating_point():
        raise ValueError("time must be a floating tensor with at least two samples")
    if not torch.isfinite(time).all() or not (time[1:] > time[:-1]).all():
        raise ValueError("time must be finite and strictly increasing")
    requested_irregularity = _fraction(
        requested_irregularity, "requested_irregularity"
    )
    jitter_fraction = _fraction(jitter_fraction, "jitter_fraction")
    if jitter_fraction == 0 and requested_irregularity > 0:
        raise ValueError("jitter_fraction must be positive when jitter is requested")
    seed = _seed(seed)
    original_intervals = time[1:] - time[:-1]
    intervals = original_intervals.clone()
    count = _half_up_count(intervals.numel(), requested_irregularity)
    if count:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selected_cpu = torch.randperm(intervals.numel(), generator=generator)[:count]
        magnitudes = 0.5 + 0.5 * torch.rand(count, generator=generator)
        signs = torch.where(
            torch.rand(count, generator=generator) < 0.5,
            torch.tensor(-1.0),
            torch.tensor(1.0),
        )
        factors = 1.0 + jitter_fraction * magnitudes * signs
        selected = selected_cpu.to(time.device)
        intervals[selected] = intervals[selected] * factors.to(
            device=time.device, dtype=time.dtype
        )
    jittered_time = torch.empty_like(time)
    jittered_time[0] = time[0]
    jittered_time[1:] = time[0] + torch.cumsum(intervals, dim=0)
    dt = torch.empty_like(time)
    dt[0] = intervals[0]
    dt[1:] = intervals
    realized = float((intervals != original_intervals).to(torch.float64).mean().item())
    return IrregularTimeResult(
        time=jittered_time,
        dt=dt,
        retained_indices=torch.arange(time.numel(), device=time.device),
        requested_irregularity=requested_irregularity,
        realized_irregularity=realized,
        method="interval_jitter",
        seed=seed,
    )


__all__ = [
    "IrregularTimeResult",
    "MaskResult",
    "channel_outage",
    "contiguous_block",
    "generate_interval_jittered_time",
    "point_missing",
]
