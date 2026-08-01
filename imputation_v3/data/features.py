"""Observed-only temporal features for imputation models."""

import torch

from imputation_v3.types import FeatureBatch


def _validate_inputs(
    target: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not all(isinstance(value, torch.Tensor) for value in (target, mask, dt)):
        raise TypeError("target, mask, and dt must be torch tensors")
    if target.ndim != 2 or target.shape[1] != 6:
        raise ValueError("target must have shape (samples, 6)")
    if target.shape[0] == 0:
        raise ValueError("target must contain at least one sample")
    if not target.is_floating_point():
        raise TypeError("target must have a floating-point dtype")
    if mask.shape != target.shape:
        raise ValueError("mask must have the same shape as target")
    if mask.device != target.device:
        raise ValueError("mask and target must be on the same device")
    if mask.is_complex() or mask.is_quantized:
        raise TypeError("mask must have a numeric or boolean dtype")

    mask_values = mask.to(dtype=target.dtype)
    if not torch.all((mask_values == 0) | (mask_values == 1)):
        raise ValueError("mask values must be exactly 0 or 1")

    if dt.ndim != 1 or dt.shape[0] != target.shape[0]:
        raise ValueError("dt must have shape (samples,)")
    if dt.device != target.device:
        raise ValueError("dt and target must be on the same device")
    if not dt.is_floating_point():
        raise TypeError("dt must have a floating-point dtype")
    if not torch.isfinite(dt).all():
        raise ValueError("dt must contain only finite values")
    if not (dt > 0).all():
        raise ValueError("dt must contain strictly positive values")

    dt_values = dt.to(dtype=target.dtype)
    if not torch.isfinite(dt_values).all():
        raise ValueError("dt must remain finite in the target dtype")
    if not (dt_values > 0).all():
        raise ValueError("dt must remain strictly positive in the target dtype")

    mask_bool = mask_values.bool()
    observed = torch.where(mask_bool, target, torch.zeros_like(target))
    if not torch.isfinite(observed).all():
        raise ValueError("observed target values must be finite")
    return observed, mask_values, dt_values


def _temporal_features(
    observed: torch.Tensor, mask_bool: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    age = torch.zeros_like(observed)
    slope = torch.zeros_like(observed)
    slope_valid = torch.zeros_like(observed)

    last_time = torch.zeros(6, dtype=observed.dtype, device=observed.device)
    last_value = torch.zeros(6, dtype=observed.dtype, device=observed.device)
    carried_slope = torch.zeros(6, dtype=observed.dtype, device=observed.device)
    has_observation = torch.zeros(6, dtype=torch.bool, device=observed.device)
    has_slope = torch.zeros(6, dtype=torch.bool, device=observed.device)

    for sample in range(observed.shape[0]):
        is_observed = mask_bool[sample]
        age[sample] = torch.where(
            is_observed | ~has_observation,
            torch.zeros_like(last_time),
            time[sample] - last_time,
        )

        updates_slope = is_observed & has_observation
        interval = torch.where(
            updates_slope, time[sample] - last_time, torch.ones_like(last_time)
        )
        candidate_slope = (observed[sample] - last_value) / interval
        carried_slope = torch.where(updates_slope, candidate_slope, carried_slope)
        has_slope = has_slope | updates_slope
        slope[sample] = carried_slope
        slope_valid[sample] = has_slope.to(dtype=observed.dtype)

        last_time = torch.where(is_observed, time[sample], last_time)
        last_value = torch.where(is_observed, observed[sample], last_value)
        has_observation = has_observation | is_observed

    return age, slope, slope_valid


def build_features(
    target: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor
) -> FeatureBatch:
    """Build the fixed 31-D causal observed-only feature contract."""

    observed, mask_values, dt_values = _validate_inputs(target, mask, dt)
    time = torch.cumsum(dt_values, dim=0) - dt_values[0]
    age, slope, slope_valid = _temporal_features(observed, mask_values.bool(), time)
    values = torch.cat(
        (observed, mask_values, dt_values[:, None], age, slope, slope_valid), dim=1
    )
    return FeatureBatch(values=values, dt=dt_values, mask=mask_values)


__all__ = ["build_features"]
