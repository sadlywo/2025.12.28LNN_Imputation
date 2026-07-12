from collections import Counter
from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path

import pytest
import yaml

from validation_v2.experiments.groups import (
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
