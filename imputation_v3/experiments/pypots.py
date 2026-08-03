"""Leakage-safe adapters for the optional PyPOTS 1.5.0 baselines."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from importlib import metadata
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from packaging.version import InvalidVersion, Version


SUPPORTED_PYPOTS_MODELS = ("brits", "saits", "csdi")
_N_FEATURES = 6
_PINNED_PYPOTS_VERSION = "1.5.0"


def installed_pypots_version() -> str:
    """Return the installed PyPOTS version only when it matches the formal pin."""
    try:
        actual = metadata.version("pypots")
    except metadata.PackageNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyPOTS baselines require the optional pinned dependency "
            "pypots==1.5.0 from requirements-imputation-v3-baselines.txt"
        ) from exc
    try:
        matches_pin = Version(actual) == Version(_PINNED_PYPOTS_VERSION)
    except InvalidVersion:
        matches_pin = False
    if not matches_pin:
        raise RuntimeError(
            "formal PyPOTS execution requires exactly pypots==1.5.0; "
            f"installed version is {actual}"
        )
    return _PINNED_PYPOTS_VERSION


def _array_attribute(dataset: object, attribute: str, dataset_name: str) -> np.ndarray:
    try:
        value = getattr(dataset, attribute)
    except AttributeError as exc:
        raise TypeError(f"{dataset_name} must expose a {attribute} numpy array") from exc
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{dataset_name}.{attribute} must be a numpy array")
    return value


def _validate_target_and_mask(
    dataset: object, dataset_name: str, *, require_complete_target: bool
) -> tuple[np.ndarray, np.ndarray]:
    target = _array_attribute(dataset, "target", dataset_name)
    mask = _array_attribute(dataset, "mask", dataset_name)
    if target.ndim != 3 or target.shape[0] == 0 or target.shape[1] == 0:
        raise ValueError(f"{dataset_name}.target must have nonempty shape (N, T, 6)")
    if target.shape[2] != _N_FEATURES:
        raise ValueError(f"{dataset_name}.target must have exactly 6 features")
    if target.dtype.kind != "f":
        raise TypeError(f"{dataset_name}.target must have a real floating dtype")
    if mask.shape != target.shape:
        raise ValueError(f"{dataset_name}.mask shape must match target")
    if mask.dtype.kind not in "bifu":
        raise TypeError(f"{dataset_name}.mask must be bool or real numeric")
    if not np.isfinite(mask).all() or not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError(f"{dataset_name}.mask must contain exact binary 0 or 1")
    mask_bool = mask.astype(bool, copy=False)
    if not np.isfinite(target[mask_bool]).all():
        raise ValueError(f"{dataset_name} observed target values must be finite")
    if require_complete_target and not np.isfinite(target).all():
        raise ValueError(f"{dataset_name}.target ground truth must be fully finite")
    return target, mask_bool


def to_pypots_sets(train: object, validation: object) -> tuple[dict, dict]:
    """Convert immutable full targets and masks to PyPOTS NaN datasets.

    Missing target values never enter ``X``. Validation receives an isolated
    complete ``X_ori`` copy for model selection.
    """
    train_target, train_mask = _validate_target_and_mask(
        train, "train", require_complete_target=False
    )
    validation_target, validation_mask = _validate_target_and_mask(
        validation, "validation", require_complete_target=True
    )
    if (
        train_target.shape[1:] != validation_target.shape[1:]
        or train_target.dtype != validation_target.dtype
    ):
        raise ValueError(
            "train and validation target sequence shape and dtype must match"
        )
    train_x = np.where(train_mask, train_target, np.nan)
    validation_x = np.where(validation_mask, validation_target, np.nan)
    return (
        {"X": np.array(train_x, dtype=train_target.dtype, order="C", copy=True)},
        {
            "X": np.array(
                validation_x,
                dtype=validation_target.dtype,
                order="C",
                copy=True,
            ),
            "X_ori": np.array(validation_target, order="C", copy=True),
        },
    )


def _validate_x(value: object, name: str, *, allow_nan: bool) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.ndim != 3 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must have nonempty shape (N, T, 6)")
    if value.shape[2] != _N_FEATURES:
        raise ValueError(f"{name} must have exactly 6 features")
    if value.dtype.kind != "f":
        raise TypeError(f"{name} must have a real floating dtype")
    if np.isinf(value).any() or (not allow_nan and np.isnan(value).any()):
        raise ValueError(f"{name} must contain only permitted finite/NaN values")
    return value


def _validated_dataset(
    dataset: object,
    name: str,
    *,
    require_x_ori: bool = False,
) -> tuple[dict[str, Any], np.ndarray]:
    if not isinstance(dataset, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if "X" not in dataset:
        raise ValueError(f"{name} must contain X")
    x = _validate_x(dataset["X"], f"{name}['X']", allow_nan=True)
    if require_x_ori and "X_ori" not in dataset:
        raise ValueError(f"{name} must contain X_ori")
    if "X_ori" in dataset:
        x_ori = _validate_x(
            dataset["X_ori"], f"{name}['X_ori']", allow_nan=False
        )
        if x_ori.shape != x.shape or x_ori.dtype != x.dtype:
            raise ValueError(f"{name} X_ori shape and dtype must match X")
        observed = ~np.isnan(x)
        if not np.array_equal(x[observed], x_ori[observed]):
            raise ValueError(f"{name} observed X values must match X_ori")
    copied = {
        key: np.array(value, order="C", copy=True)
        if isinstance(value, np.ndarray)
        else value
        for key, value in dataset.items()
    }
    return copied, x


class PyPOTSAdapter:
    """Normalize PyPOTS fit/impute behavior to the v3 experiment contract."""

    def __init__(
        self,
        model: object,
        *,
        impute_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(getattr(model, "fit", None)):
            raise TypeError("model must provide a callable fit method")
        if not callable(getattr(model, "impute", None)):
            raise TypeError("model must provide a callable impute method")
        if impute_kwargs is not None and not isinstance(impute_kwargs, Mapping):
            raise TypeError("impute_kwargs must be a mapping or None")
        kwargs = dict(impute_kwargs or {})
        if any(not isinstance(key, str) for key in kwargs):
            raise TypeError("impute_kwargs keys must be strings")
        self.model = model
        self.impute_kwargs = kwargs

    def fit(self, train_set: Mapping, validation_set: Mapping) -> "PyPOTSAdapter":
        train_copy, train_x = _validated_dataset(train_set, "train_set")
        validation_copy, validation_x = _validated_dataset(
            validation_set, "validation_set", require_x_ori=True
        )
        if (
            train_x.shape[1:] != validation_x.shape[1:]
            or train_x.dtype != validation_x.dtype
        ):
            raise ValueError(
                "train and validation X sequence shape and dtype must match"
            )
        self.model.fit(train_copy, validation_copy)
        return self

    def impute(self, test_set: Mapping) -> np.ndarray:
        test_copy, source = _validated_dataset(test_set, "test_set")
        result = self.model.impute(test_copy, **dict(self.impute_kwargs))
        if not isinstance(result, np.ndarray):
            raise TypeError("PyPOTS impute result must be a numpy array")
        if result.shape != source.shape:
            raise ValueError("PyPOTS impute result shape must match test X")
        if result.dtype != source.dtype or result.dtype.kind != "f":
            raise TypeError("PyPOTS impute result dtype must match floating test X")
        if not np.isfinite(result).all():
            raise ValueError("PyPOTS impute result must be finite")
        completed = np.where(np.isnan(source), result, source)
        if not np.isfinite(completed).all():
            raise ValueError("completed PyPOTS result must be finite")
        return np.array(completed, dtype=source.dtype, order="C", copy=True)


def build_pypots_model(
    name: str,
    *,
    n_steps: int,
    epochs: int,
    batch_size: int,
    device: str,
    saving_path: str | Path,
) -> PyPOTSAdapter:
    """Build one pinned PyPOTS baseline without importing PyPOTS eagerly."""
    if not isinstance(name, str):
        raise TypeError("PyPOTS model name must be a string")
    if name not in SUPPORTED_PYPOTS_MODELS:
        supported = ", ".join(SUPPORTED_PYPOTS_MODELS)
        raise ValueError(
            f"unsupported PyPOTS model: {name}; supported models: {supported}"
        )

    integers = {"n_steps": n_steps, "epochs": epochs, "batch_size": batch_size}
    validated_integers = {}
    for field, value in integers.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{field} must be a positive integer")
        converted = int(value)
        if converted <= 0:
            raise ValueError(f"{field} must be a positive integer")
        validated_integers[field] = converted
    if not isinstance(device, str):
        raise TypeError("device must be a non-empty string")
    if not device:
        raise ValueError("device must be a non-empty string")
    if not isinstance(saving_path, (str, Path)):
        raise TypeError("saving_path must be a string or Path")
    path_text = str(saving_path)
    if not path_text:
        raise ValueError("saving_path must be non-empty")

    installed_pypots_version()
    try:
        imputation = import_module("pypots.imputation")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyPOTS baselines require the optional pinned dependency "
            "pypots==1.5.0 from requirements-imputation-v3-baselines.txt"
        ) from exc

    common = {
        "n_steps": validated_integers["n_steps"],
        "n_features": _N_FEATURES,
        "batch_size": validated_integers["batch_size"],
        "epochs": validated_integers["epochs"],
        "device": device,
        "saving_path": path_text,
        "model_saving_strategy": "best",
    }
    if name == "brits":
        model = imputation.BRITS(rnn_hidden_size=64, **common)
        return PyPOTSAdapter(model)
    if name == "saits":
        model = imputation.SAITS(
            n_layers=2,
            d_model=64,
            n_heads=4,
            d_k=16,
            d_v=16,
            d_ffn=128,
            dropout=0.1,
            **common,
        )
        return PyPOTSAdapter(model)

    model = imputation.CSDI(
        n_layers=4,
        n_heads=4,
        n_channels=64,
        d_time_embedding=64,
        d_feature_embedding=16,
        d_diffusion_embedding=64,
        n_diffusion_steps=50,
        **common,
    )
    return PyPOTSAdapter(model, impute_kwargs={"n_sampling_times": 20})


__all__ = [
    "PyPOTSAdapter",
    "SUPPORTED_PYPOTS_MODELS",
    "build_pypots_model",
    "installed_pypots_version",
    "to_pypots_sets",
]
