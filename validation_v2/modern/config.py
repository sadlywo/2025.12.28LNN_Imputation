from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REFERENCE_MODELS = ("linear", "locf", "bilstm", "bilnn", "hybrid")
MODERN_MODELS = ("brits", "saits", "csdi", "sssd")
ALL_MODELS = REFERENCE_MODELS + MODERN_MODELS

_KEYS = {
    "data_root",
    "output_root",
    "protocol",
    "seeds",
    "split_seed",
    "seq_len",
    "batch_size",
    "epochs",
    "patience",
    "device",
    "models",
    "rates",
    "topologies",
    "irregular_cases",
    "n_sampling_times",
    "tuning_sampling_times",
    "max_train_windows",
    "max_eval_samples",
    "trajectory_enabled",
}


@dataclass(frozen=True)
class ModernConfig:
    data_root: str
    output_root: str
    protocol: str
    seeds: tuple[int, ...]
    split_seed: int
    seq_len: int
    batch_size: int
    epochs: int
    patience: int
    device: str
    models: tuple[str, ...]
    rates: tuple[float, ...]
    topologies: tuple[str, ...]
    irregular_cases: int
    n_sampling_times: int
    tuning_sampling_times: int
    max_train_windows: int
    max_eval_samples: int | None
    trajectory_enabled: bool


def load_modern_config(path: Path | str) -> ModernConfig:
    source = Path(path)
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("modern config must be a YAML mapping")
    raw: dict[str, Any] = loaded

    unknown = sorted(set(raw) - _KEYS)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    missing = sorted(_KEYS - set(raw))
    if missing:
        raise ValueError(f"missing config keys: {', '.join(missing)}")

    max_eval_samples = raw["max_eval_samples"]
    config = ModernConfig(
        data_root=str(raw["data_root"]),
        output_root=str(raw["output_root"]),
        protocol=str(raw["protocol"]),
        seeds=tuple(int(seed) for seed in raw["seeds"]),
        split_seed=int(raw["split_seed"]),
        seq_len=int(raw["seq_len"]),
        batch_size=int(raw["batch_size"]),
        epochs=int(raw["epochs"]),
        patience=int(raw["patience"]),
        device=str(raw["device"]),
        models=tuple(str(model) for model in raw["models"]),
        rates=tuple(float(rate) for rate in raw["rates"]),
        topologies=tuple(str(topology) for topology in raw["topologies"]),
        irregular_cases=len(raw["irregular_cases"]),
        n_sampling_times=int(raw["n_sampling_times"]),
        tuning_sampling_times=int(raw["tuning_sampling_times"]),
        max_train_windows=int(raw["max_train_windows"]),
        max_eval_samples=(
            None if max_eval_samples is None else int(max_eval_samples)
        ),
        trajectory_enabled=bool(raw["trajectory_enabled"]),
    )

    if config.protocol != "strict_file" or config.seq_len != 30:
        raise ValueError("modern stage A requires strict_file and seq_len 30")
    if not config.seeds or not config.models or not config.rates or not config.topologies:
        raise ValueError("seeds, models, rates, and topologies must not be empty")
    if any(model not in ALL_MODELS for model in config.models):
        raise ValueError("unsupported modern benchmark model")
    if any(topology not in {"point", "block", "channel"} for topology in config.topologies):
        raise ValueError("unsupported missingness topology")
    if any(rate <= 0.0 or rate >= 1.0 for rate in config.rates):
        raise ValueError("missingness rates must be between 0 and 1")
    if min(
        config.batch_size,
        config.epochs,
        config.patience,
        config.n_sampling_times,
        config.tuning_sampling_times,
        config.max_train_windows,
    ) <= 0:
        raise ValueError("training and sampling counts must be positive")
    if config.max_eval_samples is not None and config.max_eval_samples <= 0:
        raise ValueError("max_eval_samples must be positive or null")
    if len(config.seeds) > 1 and (
        config.n_sampling_times != 50 or config.tuning_sampling_times != 5
    ):
        raise ValueError("formal sampling counts must be 50/5")
    return config
