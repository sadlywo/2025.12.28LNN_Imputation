"""Primary missing-only reconstruction objective."""

from __future__ import annotations

import torch


def channel_balanced_missing_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Average missing-only MSE equally across represented sensor channels."""
    for name, value in (
        ("prediction", prediction),
        ("target", target),
        ("mask", mask),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch tensor")

    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating point")
    if prediction.ndim == 0 or prediction.shape[-1] != 6:
        raise ValueError("prediction final dimension must be 6")
    if prediction.numel() == 0:
        raise ValueError("prediction, target, and mask must be nonempty")
    if target.shape != prediction.shape:
        raise ValueError("target shape must match prediction shape")
    if mask.shape != prediction.shape:
        raise ValueError("mask shape must match prediction shape")
    if target.dtype != prediction.dtype:
        raise TypeError("target must have the same dtype as prediction")
    if target.device != prediction.device:
        raise ValueError("target must be on the same device as prediction")
    if mask.device != prediction.device:
        raise ValueError("mask must be on the same device as prediction")
    if mask.dtype != torch.bool and (
        not mask.is_floating_point() or mask.dtype != prediction.dtype
    ):
        raise TypeError(
            "mask must be bool or have the same floating dtype as prediction"
        )
    if not torch.all((mask == 0) | (mask == 1)).item():
        raise ValueError("mask must contain exact binary 0 or 1 values")

    missing = mask == 0
    flat_missing = missing.reshape(-1, 6)
    counts = flat_missing.sum(dim=0)
    represented = counts > 0
    if not represented.any().item():
        raise ValueError("loss requires at least one missing value")

    if not torch.isfinite(prediction[missing]).all().item() or not torch.isfinite(
        target[missing]
    ).all().item():
        raise ValueError("prediction and target values used by the loss must be finite")

    accumulation_dtype = (
        torch.float32
        if prediction.dtype in (torch.float16, torch.bfloat16)
        else prediction.dtype
    )
    prediction_acc = prediction.to(dtype=accumulation_dtype)
    target_acc = target.to(dtype=accumulation_dtype)
    zeros = torch.zeros((), dtype=accumulation_dtype, device=prediction.device)
    safe_error = torch.where(missing, prediction_acc - target_acc, zeros)

    safe_counts = counts.clamp_min(1).to(dtype=accumulation_dtype)
    normalized_error = safe_error / safe_counts.sqrt()
    squared_error = normalized_error.square()
    selected_error = torch.where(missing, squared_error, zeros).reshape(-1, 6)
    channel_means = selected_error.sum(dim=0)[represented]
    valid_channel_count = represented.sum().to(dtype=accumulation_dtype)
    loss = (channel_means / valid_channel_count).sum().to(dtype=prediction.dtype)
    if not torch.isfinite(loss).item():
        raise ValueError("channel-balanced missing MSE must be finite")
    return loss


__all__ = ["channel_balanced_missing_mse"]
