"""Observed-only temporal features for imputation models."""

import torch

from imputation_v3.types import FeatureBatch


def _validate_inputs(
    target: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask values must be exactly 0 or 1")
    mask_values = mask.to(dtype=target.dtype)

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
    dt_accumulator = dt.to(dtype=torch.float64)
    return observed, mask_values, dt_values, dt_accumulator


def _temporal_features(
    observed: torch.Tensor, mask_bool: torch.Tensor, dt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observed_accumulator = observed.to(dtype=torch.float64)
    age = torch.zeros_like(observed_accumulator)
    slope = torch.zeros_like(observed_accumulator)
    slope_valid = torch.zeros_like(observed_accumulator)

    elapsed_since_last = torch.zeros(6, dtype=torch.float64, device=observed.device)
    last_value = torch.zeros(6, dtype=torch.float64, device=observed.device)
    carried_slope = torch.zeros(6, dtype=torch.float64, device=observed.device)
    has_observation = torch.zeros(6, dtype=torch.bool, device=observed.device)
    has_slope = torch.zeros(6, dtype=torch.bool, device=observed.device)

    for sample in range(observed.shape[0]):
        is_observed = mask_bool[sample]
        if sample > 0:
            elapsed_since_last = torch.where(
                has_observation,
                elapsed_since_last + dt[sample],
                elapsed_since_last,
            )
        age[sample] = torch.where(
            is_observed | ~has_observation,
            torch.zeros_like(elapsed_since_last),
            elapsed_since_last,
        )

        updates_slope = is_observed & has_observation
        interval = torch.where(
            updates_slope,
            elapsed_since_last,
            torch.ones_like(elapsed_since_last),
        )
        candidate_slope = (observed_accumulator[sample] - last_value) / interval
        carried_slope = torch.where(updates_slope, candidate_slope, carried_slope)
        has_slope = has_slope | updates_slope
        slope[sample] = carried_slope
        slope_valid[sample] = has_slope.to(dtype=torch.float64)

        elapsed_since_last = torch.where(
            is_observed, torch.zeros_like(elapsed_since_last), elapsed_since_last
        )
        last_value = torch.where(
            is_observed, observed_accumulator[sample], last_value
        )
        has_observation = has_observation | is_observed

    return age, slope, slope_valid


def build_features(
    target: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor
) -> FeatureBatch:
    """Build the fixed 31-D causal observed-only feature contract."""

    observed, mask_values, dt_values, dt_accumulator = _validate_inputs(
        target, mask, dt
    )
    elapsed_time = torch.zeros_like(dt_accumulator)
    elapsed_time[1:] = torch.cumsum(dt_accumulator[1:], dim=0)
    if not torch.isfinite(elapsed_time).all():
        raise ValueError("dt produces non-finite cumulative elapsed time")

    temporal_accumulator = _temporal_features(
        observed, mask_values.bool(), dt_accumulator
    )
    if not all(torch.isfinite(value).all() for value in temporal_accumulator):
        raise ValueError("derived temporal features must be finite")
    age, slope, slope_valid = (
        value.to(dtype=target.dtype) for value in temporal_accumulator
    )
    if not all(torch.isfinite(value).all() for value in (age, slope, slope_valid)):
        raise ValueError("derived temporal features must be finite in the target dtype")
    values = torch.cat(
        (observed, mask_values, dt_values[:, None], age, slope, slope_valid), dim=1
    )
    return FeatureBatch(values=values, dt=dt_values, mask=mask_values)


__all__ = ["build_features"]
