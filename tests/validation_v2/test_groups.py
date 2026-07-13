from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path
from typing import Any, get_type_hints, Optional

import pytest
import yaml

from validation_v2.experiments.groups import (
    TrainingGroup,
    enumerate_training_groups,
    group_execution_config,
)
from validation_v2.experiments.matrix import enumerate_matrix
from validation_v2.experiments.provenance import canonical_json


REPO_ROOT = Path(__file__).resolve().parents[2]


def _server_config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "configs" / "validation_v2" / "server_full.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_server_matrix_enumerates_expected_training_groups():
    config = _server_config()

    combinations = enumerate_matrix(config)
    groups = enumerate_training_groups(config)

    assert len(combinations) == 4095
    assert isinstance(groups, tuple)
    assert len(groups) == 175
    assert [group.group_index for group in groups] == list(range(175))
    assert Counter(group.training_family for group in groups) == {
        "linear": 35,
        "locf": 35,
        "bilstm": 35,
        "bilnn": 35,
        "hybrid_shared": 35,
    }
    assert sum(len(group.conditions) for group in groups) == 4095


def test_group_ids_are_stable_content_hashes_and_groups_are_frozen():
    config = _server_config()

    first = enumerate_training_groups(config)
    second = enumerate_training_groups(config)
    group = first[0]
    payload = {
        "training_family": group.training_family,
        "seed": group.seed,
        "protocol": group.protocol,
        "objective": group.objective,
        "combination_ids": list(group.combination_ids),
    }

    assert [item.group_id for item in first] == [item.group_id for item in second]
    assert group.group_id == hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    assert len({item.group_id for item in first}) == len(first)
    with pytest.raises(FrozenInstanceError):
        group.seed = 0  # type: ignore[misc]


def test_gate_reporters_share_one_hybrid_training_group():
    config = _server_config()
    config.update(
        models=[
            "hybrid",
            "equal_average",
            "fixed_gate_0",
            "fixed_gate_0.5",
            "fixed_gate_1",
        ],
        seeds=[2026],
        protocols=["strict_file"],
        topologies=["point"],
        rates=[0.3],
        irregular_cases=[],
    )

    groups = enumerate_training_groups(config)

    assert len(groups) == 1
    assert groups[0].training_family == "hybrid_shared"
    assert groups[0].training_model == "hybrid"
    assert groups[0].reported_models == (
        "equal_average",
        "fixed_gate_0",
        "fixed_gate_0.5",
        "fixed_gate_1",
        "hybrid",
    )
    assert len(groups[0].conditions) == 5


def test_group_execution_config_sets_the_runner_private_fields():
    config = _server_config()
    group = enumerate_training_groups(config)[0]

    execution = group_execution_config(config, group)

    assert execution == {
        **config,
        "models": [group.training_model],
        "seeds": [group.seed],
        "protocols": [group.protocol],
        "_execution_conditions": list(group.conditions),
        "_skip_descriptive_summary": True,
        "_training_family": group.training_family,
        "_reported_models": list(group.reported_models),
    }
    assert config == _server_config()


def test_bounded_combinations_group_only_the_selected_cells():
    config = _server_config()
    combinations = enumerate_matrix(config)
    selected = combinations[:17]

    groups = enumerate_training_groups(config, combinations=selected)

    grouped_ids = [
        combination_id
        for group in groups
        for combination_id in group.combination_ids
    ]
    assert Counter(grouped_ids) == Counter(
        combination["combination_id"] for combination in selected
    )
    assert sum(len(group.conditions) for group in groups) == len(selected)
    assert not set(grouped_ids) & {
        combination["combination_id"] for combination in combinations[17:]
    }


def test_duplicate_selected_combination_ids_are_rejected():
    config = _server_config()
    combination = enumerate_matrix(config)[0]

    with pytest.raises(ValueError, match="duplicate combination_id"):
        enumerate_training_groups(config, combinations=[combination, combination])


def _nested_group() -> TrainingGroup:
    return TrainingGroup(
        group_index=0,
        group_id="group-id",
        training_family="linear",
        training_model="linear",
        reported_models=("linear",),
        seed=2026,
        protocol="strict_file",
        objective="reconstruction_only",
        conditions=(
            {
                "combination_id": "combination-id",
                "nested": {"items": [{"value": 1}]},
            },
        ),
        combination_ids=("combination-id",),
    )


def test_training_group_conditions_are_recursively_immutable():
    group = _nested_group()

    with pytest.raises(TypeError):
        group.conditions[0]["combination_id"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        group.conditions[0]["nested"]["items"][0]["value"] = 2  # type: ignore[index]


def test_selected_cell_mutation_does_not_change_group_snapshot():
    config = _server_config()
    selected = dict(enumerate_matrix(config)[0])
    original_model = selected["model"]

    group = enumerate_training_groups(config, combinations=[selected])[0]
    selected["model"] = "changed-after-enumeration"

    assert group.conditions[0]["model"] == original_model


def test_execution_configs_recursively_thaw_to_independent_plain_containers():
    group = _nested_group()

    first = group_execution_config({}, group)
    second = group_execution_config({}, group)
    first_condition = first["_execution_conditions"][0]
    second_condition = second["_execution_conditions"][0]

    assert type(first_condition) is dict
    assert type(first_condition["nested"]) is dict
    assert type(first_condition["nested"]["items"]) is list
    first_condition["nested"]["items"][0]["value"] = 2
    assert second_condition["nested"]["items"][0]["value"] == 1
    assert group.conditions[0]["nested"]["items"][0]["value"] == 1


def test_foreign_selected_combination_id_is_rejected():
    config = _server_config()
    selected = dict(enumerate_matrix(config)[0])
    selected["combination_id"] = "0" * 64

    with pytest.raises(ValueError, match="foreign combination_id"):
        enumerate_training_groups(config, combinations=[selected])


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("requested_fraction", 0.987),
        ("model", "changed-model"),
        ("protocol", "changed-protocol"),
    ],
)
def test_tampered_selected_combination_is_rejected(field: str, changed_value: Any):
    config = _server_config()
    selected = dict(enumerate_matrix(config)[0])
    selected[field] = changed_value

    with pytest.raises(ValueError, match="tampered combination"):
        enumerate_training_groups(config, combinations=[selected])


def test_non_string_selected_combination_id_is_rejected():
    config = _server_config()
    selected = dict(enumerate_matrix(config)[0])
    selected["combination_id"] = 7

    with pytest.raises(ValueError, match="invalid combination_id"):
        enumerate_training_groups(config, combinations=[selected])


def test_group_enumerator_annotations_are_python_39_compatible():
    hints = get_type_hints(enumerate_training_groups)

    assert hints["combinations"] == Optional[Sequence[Mapping[str, Any]]]
    assert hints["return"] == tuple[TrainingGroup, ...]
    assert enumerate_training_groups.__annotations__["combinations"].startswith(
        "Optional["
    )
