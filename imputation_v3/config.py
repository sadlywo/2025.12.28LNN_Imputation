"""Typed configuration for offline teacher experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

import yaml


ALLOWED_MODELS = frozenset(
    {
        "locf",
        "linear",
        "pchip",
        "rts",
        "bilstm",
        "bilnn",
        "tcn",
        "feature_mlp",
        "teacher",
        "brits",
        "saits",
        "csdi",
    }
)

Number = TypeVar("Number", int, float)


@dataclass(frozen=True)
class TeacherConfig:
    """Immutable settings for teacher training and model evaluation."""

    data_root: Path
    output_root: Path
    selection_split: str
    seeds: tuple[int, ...]
    window_seconds: tuple[float, ...]
    nominal_dt_s: float
    batch_size: int
    epochs: int
    hidden_size: int
    tcn_width: int
    tcn_dilations: tuple[int, ...]
    learning_rate: float
    training_rates: tuple[float, ...]
    training_topologies: tuple[str, ...]
    models: tuple[str, ...]

    @property
    def window_samples(self) -> tuple[int, ...]:
        """Return configured window lengths expressed in samples."""
        return tuple(round(seconds / self.nominal_dt_s) for seconds in self.window_seconds)


def _required(data: Mapping[str, Any], field: str) -> Any:
    try:
        return data[field]
    except KeyError as exc:
        raise ValueError(f"missing required field: {field}") from exc


def _convert(value: Any, field: str, converter: Callable[[Any], Number]) -> Number:
    try:
        return converter(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc


def _positive(value: Any, field: str, converter: Callable[[Any], Number]) -> Number:
    converted = _convert(value, field, converter)
    if converted <= 0:
        raise ValueError(f"{field} must be positive")
    return converted


def _positive_list(
    data: Mapping[str, Any],
    field: str,
    converter: Callable[[Any], Number],
) -> tuple[Number, ...]:
    values = _required(data, field)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field} must be a non-empty list")
    return tuple(_positive(value, field, converter) for value in values)


def _string_list(data: Mapping[str, Any], field: str) -> tuple[str, ...]:
    values = _required(data, field)
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list")
    return tuple(str(value) for value in values)


def load_teacher_config(path: Path) -> TeacherConfig:
    """Load a teacher YAML config while enforcing validation-only selection."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("teacher configuration must be a YAML mapping")
    data: Mapping[str, Any] = loaded

    selection_split = str(data.get("selection_split", "validation"))
    if selection_split != "validation":
        raise ValueError("selection_split must be validation")

    raw_seeds = data.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("seeds must be a non-empty list")
    try:
        seeds = tuple(int(seed) for seed in raw_seeds)
    except (TypeError, ValueError) as exc:
        raise ValueError("seeds must contain integers") from exc

    models = _string_list(data, "models")
    unsupported = sorted(set(models) - ALLOWED_MODELS)
    if unsupported:
        raise ValueError(f"unsupported models: {', '.join(unsupported)}")

    return TeacherConfig(
        data_root=Path(str(_required(data, "data_root"))),
        output_root=Path(str(_required(data, "output_root"))),
        selection_split=selection_split,
        seeds=seeds,
        window_seconds=_positive_list(data, "window_seconds", float),
        nominal_dt_s=_positive(_required(data, "nominal_dt_s"), "nominal_dt_s", float),
        batch_size=_positive(_required(data, "batch_size"), "batch_size", int),
        epochs=_positive(_required(data, "epochs"), "epochs", int),
        hidden_size=_positive(_required(data, "hidden_size"), "hidden_size", int),
        tcn_width=_positive(_required(data, "tcn_width"), "tcn_width", int),
        tcn_dilations=_positive_list(data, "tcn_dilations", int),
        learning_rate=_positive(_required(data, "learning_rate"), "learning_rate", float),
        training_rates=_positive_list(data, "training_rates", float),
        training_topologies=_string_list(data, "training_topologies"),
        models=models,
    )
