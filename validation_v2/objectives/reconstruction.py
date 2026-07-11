"""Reconstruction losses evaluated strictly at missing entries.

The mask convention is ``1 = observed`` and ``0 = missing``.  These
functions compare samples at the same time index; there is deliberately no
``t - 1`` shift.
"""

from __future__ import annotations

import torch


def _validated_missing_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not all(isinstance(value, torch.Tensor) for value in (prediction, target, mask)):
        raise ValueError("prediction, target, and mask must be torch tensors")
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    if prediction.numel() == 0:
        raise ValueError("prediction, target, and mask must not be empty")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("prediction and target must have floating dtypes")
    if not mask.is_floating_point():
        raise ValueError("mask must have a floating dtype containing only 0 or 1")
    if prediction.dtype != target.dtype or prediction.dtype != mask.dtype:
        raise ValueError("prediction, target, and mask must have the same dtype")
    if prediction.device != target.device or prediction.device != mask.device:
        raise ValueError("prediction, target, and mask must be on the same device")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must contain only finite values")
    if not torch.isfinite(mask).all() or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask values must be exactly 0 (missing) or 1 (observed)")

    missing = mask == 0
    if not torch.any(missing):
        raise ValueError("missing-entry metric requires at least one missing value")
    return prediction[missing] - target[missing]


def missing_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error normalized by the exact number of missing values."""

    errors = _validated_missing_errors(prediction, target, mask)
    return errors.square().mean()


def missing_rmse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Root mean squared error over missing values only."""

    return torch.sqrt(missing_mse(prediction, target, mask))


def missing_mae(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean absolute error over missing values only."""

    errors = _validated_missing_errors(prediction, target, mask)
    return errors.abs().mean()
