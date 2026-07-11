"""Observed-only features for imputation models."""

import torch

from validation_v2.types import FeatureBatch


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
    if mask.dtype == torch.bool:
        mask_values = mask.to(dtype=target.dtype)
    elif mask.is_floating_point() or mask.dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        mask_values = mask.to(dtype=target.dtype)
    else:
        raise TypeError("mask must have a numeric or boolean dtype")
    if not torch.all((mask_values == 0) | (mask_values == 1)):
        raise ValueError("mask values must be exactly 0 or 1")
    if dt.ndim != 1 or dt.shape[0] != target.shape[0]:
        raise ValueError("dt must be one-dimensional and align with target length")
    if dt.device != target.device:
        raise ValueError("dt and target must be on the same device")
    if not dt.is_floating_point():
        raise TypeError("dt must have a floating-point dtype")
    if not torch.isfinite(dt).all() or not (dt > 0).all():
        raise ValueError("dt must contain finite, strictly positive values")
    if torch.isinf(target).any():
        raise ValueError("target must not contain infinite values")
    if not torch.isfinite(target[mask_values.bool()]).all():
        raise ValueError("observed target values must be finite")
    return target, mask_values, dt.to(dtype=target.dtype)


def build_features(
    target: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor
) -> FeatureBatch:
    """Build the fixed 25-D observed-only feature contract.

    Column order is ``observed[6], mask[6], dt[1], delta[6],
    valid_delta[6]``. Hidden target values are never evaluated.
    """

    target, mask_values, dt_values = _validate_inputs(target, mask, dt)
    observed = torch.where(mask_values.bool(), target, torch.zeros_like(target))
    valid_delta = torch.zeros_like(target)
    valid_delta[1:] = mask_values[1:] * mask_values[:-1]
    delta = torch.zeros_like(target)
    delta[1:] = torch.where(
        valid_delta[1:].bool(),
        observed[1:] - observed[:-1],
        torch.zeros_like(observed[1:]),
    )
    values = torch.cat(
        (observed, mask_values, dt_values[:, None], delta, valid_delta), dim=1
    )
    return FeatureBatch(values=values, dt=dt_values, mask=mask_values)


__all__ = ["FeatureBatch", "build_features"]
