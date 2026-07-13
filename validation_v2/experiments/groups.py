"""Shared training-group enumeration for validation matrix execution."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional

from .matrix import enumerate_matrix
from .provenance import canonical_json


_GATE_MODELS = frozenset(
    {
        "hybrid",
        "equal_average",
        "fixed_gate_0",
        "fixed_gate_0.5",
        "fixed_gate_1",
    }
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class TrainingGroup:
    group_index: int
    group_id: str
    training_family: str
    training_model: str
    reported_models: tuple[str, ...]
    seed: int
    protocol: str
    objective: str
    conditions: tuple[Mapping[str, Any], ...]
    combination_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "conditions",
            tuple(_freeze(condition) for condition in self.conditions),
        )


def enumerate_training_groups(
    config: Mapping[str, Any],
    *,
    combinations: Optional[Sequence[Mapping[str, Any]]] = None,
) -> tuple[TrainingGroup, ...]:
    """Group matrix cells that share one trained checkpoint."""

    canonical_cells = enumerate_matrix(config)
    canonical_by_id = {
        cell["combination_id"]: cell for cell in canonical_cells
    }
    if combinations is None:
        cells = canonical_cells
    else:
        cells = []
        for selected in combinations:
            combination_id = selected.get("combination_id")
            if not isinstance(combination_id, str):
                raise ValueError("invalid combination_id")
            canonical_cell = canonical_by_id.get(combination_id)
            if canonical_cell is None:
                raise ValueError("foreign combination_id")
            try:
                matches = canonical_json(selected) == canonical_json(canonical_cell)
            except (TypeError, ValueError) as error:
                raise ValueError("tampered combination") from error
            if not matches:
                raise ValueError("tampered combination")
            cells.append(canonical_cell)

    combination_ids = [cell["combination_id"] for cell in cells]
    if len(set(combination_ids)) != len(combination_ids):
        raise ValueError("duplicate combination_id")

    objective = str(config.get("objective", "reconstruction_only"))
    grouped: dict[
        tuple[str, int, str, str], list[Mapping[str, Any]]
    ] = {}
    for cell in cells:
        model = str(cell["model"])
        training_family = "hybrid_shared" if model in _GATE_MODELS else model
        key = (
            training_family,
            int(cell["seed"]),
            str(cell["protocol"]),
            objective,
        )
        grouped.setdefault(key, []).append(cell)

    groups: list[TrainingGroup] = []
    group_ids: set[str] = set()
    for group_index, (key, conditions) in enumerate(sorted(grouped.items())):
        training_family, seed, protocol, group_objective = key
        ordered_combination_ids = tuple(
            str(condition["combination_id"]) for condition in conditions
        )
        payload = {
            "training_family": training_family,
            "seed": seed,
            "protocol": protocol,
            "objective": group_objective,
            "combination_ids": list(ordered_combination_ids),
        }
        group_id = hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest()
        if group_id in group_ids:
            raise ValueError("duplicate group_id")
        group_ids.add(group_id)
        groups.append(
            TrainingGroup(
                group_index=group_index,
                group_id=group_id,
                training_family=training_family,
                training_model=(
                    "hybrid"
                    if training_family == "hybrid_shared"
                    else training_family
                ),
                reported_models=tuple(
                    sorted({str(condition["model"]) for condition in conditions})
                ),
                seed=seed,
                protocol=protocol,
                objective=group_objective,
                conditions=tuple(conditions),
                combination_ids=ordered_combination_ids,
            )
        )
    return tuple(groups)


def group_execution_config(
    config: Mapping[str, Any], group: TrainingGroup
) -> dict[str, Any]:
    """Build the private runner configuration for one training group."""

    execution = _thaw(config)
    execution.update(
        models=[group.training_model],
        seeds=[group.seed],
        protocols=[group.protocol],
        _execution_conditions=[
            _thaw(condition) for condition in group.conditions
        ],
        _skip_descriptive_summary=True,
        _training_family=group.training_family,
        _reported_models=list(group.reported_models),
    )
    return execution


__all__ = [
    "TrainingGroup",
    "enumerate_training_groups",
    "group_execution_config",
]
