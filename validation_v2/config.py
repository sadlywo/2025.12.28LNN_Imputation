"""Typed experiment configuration loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import yaml

from .types import SelectionSplit


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path
    output_root: Path
    selection_split: SelectionSplit
    seeds: tuple[int, ...]
    seq_len: int
    batch_size: int
    epochs: int


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from YAML."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    data: Mapping[str, Any] = loaded if isinstance(loaded, Mapping) else {}

    selection_split = str(data.get("selection_split", "validation"))
    if selection_split != "validation":
        raise ValueError("selection_split must be validation")

    raw_seeds = data.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("seeds must be a non-empty list")

    return ExperimentConfig(
        data_root=Path(str(data["data_root"])),
        output_root=Path(str(data["output_root"])),
        selection_split=cast(SelectionSplit, selection_split),
        seeds=tuple(int(seed) for seed in raw_seeds),
        seq_len=int(data["seq_len"]),
        batch_size=int(data["batch_size"]),
        epochs=int(data["epochs"]),
    )
