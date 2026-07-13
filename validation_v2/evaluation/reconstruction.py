"""Separate normalized and physical reconstruction reports."""

from __future__ import annotations

from typing import Mapping, Protocol

import numpy as np


class _InverseScaler(Protocol):
    def inverse_transform(self, values: np.ndarray) -> np.ndarray: ...


def _inverse_transform_last_axis(
    scaler: _InverseScaler, values: np.ndarray
) -> np.ndarray:
    """Apply a sklearn-style 2D inverse transform to the final feature axis."""

    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("normalized values must have a non-empty feature axis")
    flattened = values.reshape(-1, values.shape[-1])
    try:
        transformed = np.asarray(scaler.inverse_transform(flattened), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("scaler inverse_transform must return numeric values") from exc
    if transformed.shape != flattened.shape:
        raise ValueError("scaler inverse_transform must preserve the flattened input shape")
    if not np.all(np.isfinite(transformed)):
        raise ValueError("scaler inverse_transform must return only finite values")
    return transformed.reshape(values.shape)


def _missing_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if prediction.shape != target.shape or prediction.shape != mask.shape or prediction.size == 0:
        raise ValueError("prediction, target, and mask must have identical non-empty shapes")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
        raise ValueError("prediction and target must contain only finite values")
    if not np.all(np.isfinite(mask)) or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("mask values must be exactly 0 or 1")
    missing = mask == 0
    if not np.any(missing):
        raise ValueError("reconstruction metrics require at least one missing value")
    errors = prediction[missing] - target[missing]
    mse = float(np.mean(np.square(errors)))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(errors)))}


def reconstruction_metrics(
    normalized_prediction: np.ndarray,
    normalized_target: np.ndarray,
    mask: np.ndarray,
    *,
    scaler: _InverseScaler | None = None,
    physical_prediction: np.ndarray | None = None,
    physical_target: np.ndarray | None = None,
) -> Mapping[str, Mapping[str, float]]:
    """Return honest normalized and physical missing-entry metrics.

    Physical values must either be supplied explicitly or recovered with an
    inverse scaler.  Normalized values are never relabeled as physical values.
    """

    normalized_prediction = np.asarray(normalized_prediction, dtype=np.float64)
    normalized_target = np.asarray(normalized_target, dtype=np.float64)
    normalized = _missing_metrics(normalized_prediction, normalized_target, mask)
    if (physical_prediction is None) != (physical_target is None):
        raise ValueError("physical_prediction and physical_target must be supplied together")
    if physical_prediction is None:
        if scaler is None or not hasattr(scaler, "inverse_transform"):
            raise ValueError("physical metrics require explicit physical tensors or an inverse scaler")
        physical_prediction = _inverse_transform_last_axis(scaler, normalized_prediction)
        physical_target = _inverse_transform_last_axis(scaler, normalized_target)
    physical = _missing_metrics(physical_prediction, physical_target, mask)
    return {"normalized": normalized, "physical": physical}
