from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from validation_v2.experiments.groups import enumerate_training_groups
from validation_v2.experiments.provenance import canonical_json
from validation_v2.experiments.sharding import (
    SHARD_SCHEMA_VERSION,
    build_shard_plan,
    load_shard_plan,
    write_shard_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GIT_COMMIT = "c34d4cb7d766393bd31f210cc33ad7ae8d30e59b"


def _server_config() -> dict[str, Any]:
    return yaml.safe_load(
        (REPO_ROOT / "configs" / "validation_v2" / "server_full.yaml").read_text(
            encoding="utf-8"
        )
    )


def _plan(shard_count: int = 8) -> dict[str, Any]:
    return build_shard_plan(
        _server_config(),
        shard_count=shard_count,
        git_commit=GIT_COMMIT,
        device="cuda",
    )


def _resign(plan: Mapping[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(dict(plan))
    payload = {
        key: value
        for key, value in changed.items()
        if key not in {"created_at", "plan_sha256"}
    }
    changed["plan_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return changed


def _write_raw(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _temporary_files(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}-*.tmp"))


def test_server_plan_has_expected_counts_and_round_robin_assignment():
    config = _server_config()
    groups = enumerate_training_groups(config)

    plan = _plan()

    assert SHARD_SCHEMA_VERSION == 1
    assert plan["schema_version"] == 1
    assert plan["shard_count"] == 8
    assert plan["total_groups"] == 175
    assert plan["total_cells"] == 4095
    assert len(plan["shards"]) == 8
    for shard_index, shard in enumerate(plan["shards"]):
        expected = list(range(shard_index, len(groups), 8))
        assert shard["shard_index"] == shard_index
        assert shard["group_indices"] == expected
        assert shard["group_ids"] == [groups[index].group_id for index in expected]
        assert shard["group_keys"] == [
            {
                "training_family": groups[index].training_family,
                "seed": groups[index].seed,
                "protocol": groups[index].protocol,
                "objective": groups[index].objective,
            }
            for index in expected
        ]
        assert shard["combination_ids"] == [
            combination_id
            for index in expected
            for combination_id in groups[index].combination_ids
        ]


def test_group_ids_are_stable_across_shard_counts():
    one = _plan(shard_count=1)
    thirteen = _plan(shard_count=13)

    one_ids = one["shards"][0]["group_ids"]
    thirteen_ids = [
        group_id
        for shard in thirteen["shards"]
        for group_id in shard["group_ids"]
    ]

    assert sorted(one_ids) == sorted(thirteen_ids)


def test_source_config_hash_uses_execution_provenance_filtering():
    config = _server_config()
    config["output_root"] = "different/output"
    config["_execution_conditions"] = [{"secret": "ignored"}]
    expected_source = {
        key: value
        for key, value in config.items()
        if key != "output_root" and not key.startswith("_")
    }

    plan = build_shard_plan(
        config, shard_count=2, git_commit=GIT_COMMIT, device="cpu"
    )

    assert plan["source_config_sha256"] == hashlib.sha256(
        canonical_json(expected_source).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("shard_count", [True, False, 0, -1, 1.0, "2"])
def test_invalid_shard_counts_are_rejected(shard_count: Any):
    with pytest.raises(ValueError, match="shard_count"):
        build_shard_plan(
            _server_config(),
            shard_count=shard_count,
            git_commit=GIT_COMMIT,
        )


@pytest.mark.parametrize("device", ["auto", "CUDA", "mps", "", 1, []])
def test_invalid_devices_are_rejected(device: Any):
    with pytest.raises(ValueError, match="device"):
        build_shard_plan(
            _server_config(),
            shard_count=1,
            git_commit=GIT_COMMIT,
            device=device,
        )


@pytest.mark.parametrize("git_commit", ["", "   ", None, 7])
def test_invalid_git_commits_are_rejected(git_commit: Any):
    with pytest.raises(ValueError, match="git_commit"):
        build_shard_plan(
            _server_config(), shard_count=1, git_commit=git_commit
        )


def test_plan_hash_excludes_timestamp_but_covers_the_rest():
    plan = _plan(shard_count=2)
    payload = {
        key: value
        for key, value in plan.items()
        if key not in {"created_at", "plan_sha256"}
    }

    assert plan["created_at"].endswith("Z")
    assert plan["plan_sha256"] == hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    changed_time = copy.deepcopy(plan)
    changed_time["created_at"] = "2000-01-01T00:00:00Z"
    assert _resign(changed_time)["plan_sha256"] == plan["plan_sha256"]


def test_write_is_canonical_idempotent_and_creates_parents(tmp_path: Path):
    plan = _plan(shard_count=2)
    path = tmp_path / "nested" / "plan.json"

    returned = write_shard_plan(path, plan)
    original = path.read_bytes()
    second = write_shard_plan(path, copy.deepcopy(plan))

    assert returned == path
    assert second == path
    assert original == (canonical_json(plan) + "\n").encode("utf-8")
    assert path.read_bytes() == original
    assert _temporary_files(path) == []


def test_write_never_clobbers_different_existing_content(tmp_path: Path):
    path = tmp_path / "plan.json"
    path.write_bytes(b"existing bytes\n")

    with pytest.raises(ValueError, match="different content"):
        write_shard_plan(path, _plan(shard_count=2))

    assert path.read_bytes() == b"existing bytes\n"
    assert _temporary_files(path) == []


def test_write_survives_destination_deleted_after_exists_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "plan.json"
    path.write_bytes(b"transient writer\n")
    plan = _plan(shard_count=2)
    expected = (canonical_json(plan) + "\n").encode("utf-8")
    real_exists = Path.exists
    real_link = os.link
    raced = False

    def delete_destination_once() -> None:
        nonlocal raced
        if not raced:
            raced = True
            path.unlink()

    def racing_exists(candidate: Path) -> bool:
        if candidate == path:
            delete_destination_once()
            return True
        return real_exists(candidate)

    def racing_link(source: Any, destination: Any) -> None:
        delete_destination_once()
        real_link(source, destination)

    monkeypatch.setattr(Path, "exists", racing_exists)
    monkeypatch.setattr(os, "link", racing_link)

    assert write_shard_plan(path, plan) == path
    assert path.read_bytes() == expected
    assert _temporary_files(path) == []


def test_write_retries_when_destination_is_deleted_before_conflict_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "plan.json"
    path.write_bytes(b"transient writer\n")
    plan = _plan(shard_count=2)
    expected = (canonical_json(plan) + "\n").encode("utf-8")
    real_link = os.link
    real_read_bytes = Path.read_bytes
    link_attempts = 0
    deleted = False

    def counting_link(source: Any, destination: Any) -> None:
        nonlocal link_attempts
        link_attempts += 1
        real_link(source, destination)

    def deleting_read_bytes(candidate: Path) -> bytes:
        nonlocal deleted
        if candidate == path and not deleted:
            deleted = True
            candidate.unlink()
            raise FileNotFoundError(candidate)
        return real_read_bytes(candidate)

    monkeypatch.setattr(os, "link", counting_link)
    monkeypatch.setattr(Path, "read_bytes", deleting_read_bytes)

    assert write_shard_plan(path, plan) == path
    assert link_attempts == 2
    assert path.read_bytes() == expected
    assert _temporary_files(path) == []


def _race_at_commit(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    racing_content: bytes,
) -> None:
    real_link = os.link
    raced = False

    def create_racing_destination() -> None:
        nonlocal raced
        if not raced:
            raced = True
            path.write_bytes(racing_content)

    def racing_link(source: Any, destination: Any) -> None:
        create_racing_destination()
        real_link(source, destination)

    monkeypatch.setattr(os, "link", racing_link)


def test_write_does_not_clobber_different_content_created_at_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "plan.json"
    racing_content = b"racing writer's different plan\n"
    _race_at_commit(monkeypatch, path, racing_content)

    with pytest.raises(ValueError, match="different content"):
        write_shard_plan(path, _plan(shard_count=2))

    assert path.read_bytes() == racing_content
    assert _temporary_files(path) == []


def test_write_accepts_same_content_created_at_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "plan.json"
    plan = _plan(shard_count=2)
    content = (canonical_json(plan) + "\n").encode("utf-8")
    _race_at_commit(monkeypatch, path, content)

    assert write_shard_plan(path, plan) == path
    assert path.read_bytes() == content
    assert _temporary_files(path) == []


def test_write_cleans_temporary_file_when_link_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "plan.json"

    def fail_link(source: Any, destination: Any) -> None:
        raise PermissionError("link denied")

    monkeypatch.setattr(os, "link", fail_link)

    with pytest.raises(PermissionError, match="link denied"):
        write_shard_plan(path, _plan(shard_count=2))

    assert not path.exists()
    assert _temporary_files(path) == []


def test_load_round_trip_accepts_json_and_yaml(tmp_path: Path):
    config = _server_config()
    plan = _plan(shard_count=3)
    json_path = write_shard_plan(tmp_path / "plan.json", plan)
    yaml_path = tmp_path / "plan.yaml"
    yaml_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")

    assert load_shard_plan(
        json_path, config=config, git_commit=GIT_COMMIT, device="cuda"
    ) == plan
    assert load_shard_plan(
        yaml_path, config=config, git_commit=GIT_COMMIT, device="cuda"
    ) == plan


def test_load_rejects_non_mapping_and_invalid_yaml(tmp_path: Path):
    path = tmp_path / "plan.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_shard_plan(
            path, config=_server_config(), git_commit=GIT_COMMIT, device="cuda"
        )

    path.write_text("not: [valid", encoding="utf-8")
    with pytest.raises(ValueError, match="shard plan"):
        load_shard_plan(
            path, config=_server_config(), git_commit=GIT_COMMIT, device="cuda"
        )


def test_load_rejects_plan_hash_tampering(tmp_path: Path):
    path = tmp_path / "plan.json"
    changed = _plan(shard_count=2)
    changed["total_cells"] += 1
    _write_raw(path, changed)

    with pytest.raises(ValueError, match="plan_sha256"):
        load_shard_plan(
            path, config=_server_config(), git_commit=GIT_COMMIT, device="cuda"
        )


def test_load_rejects_yaml_implicit_date_as_invalid_plan_value(tmp_path: Path):
    path = tmp_path / "plan.yaml"
    changed = _plan(shard_count=2)
    changed["shards"][0]["combination_ids"][0] = "2026-01-01"
    changed = _resign(changed)
    rendered = yaml.safe_dump(changed, sort_keys=False)
    quoted_date = "- '2026-01-01'"
    assert quoted_date in rendered
    path.write_text(
        rendered.replace(quoted_date, "- 2026-01-01", 1), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="invalid shard plan values"):
        load_shard_plan(
            path, config=_server_config(), git_commit=GIT_COMMIT, device="cuda"
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("source_config_sha256", "0" * 64, "source_config_sha256"),
        ("git_commit", "different", "git_commit"),
        ("device", "cpu", "device"),
        ("total_groups", 176, "total_groups"),
        ("total_cells", 4096, "total_cells"),
        ("shard_count", 3, "shard_count"),
    ],
)
def test_load_rejects_resigned_top_level_tampering(
    tmp_path: Path, field: str, value: Any, message: str
):
    path = tmp_path / "plan.json"
    changed = _plan(shard_count=2)
    changed[field] = value
    _write_raw(path, _resign(changed))

    with pytest.raises(ValueError, match=message):
        load_shard_plan(
            path, config=_server_config(), git_commit=GIT_COMMIT, device="cuda"
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda plan: plan.update(extra="unexpected"), "fields"),
        (lambda plan: plan.pop("total_cells"), "fields"),
        (lambda plan: plan["shards"][0].update(extra="unexpected"), "fields"),
        (lambda plan: plan["shards"][0].pop("group_ids"), "fields"),
        (lambda plan: plan["shards"][0].update(shard_index=1), "shard_index"),
        (lambda plan: plan["shards"].pop(), "shards"),
        (
            lambda plan: plan["shards"].append(copy.deepcopy(plan["shards"][0])),
            "shards",
        ),
        (
            lambda plan: plan["shards"][0]["group_indices"].append(
                plan["shards"][1]["group_indices"][0]
            ),
            "group_indices",
        ),
        (
            lambda plan: plan["shards"][0]["group_indices"].__setitem__(0, False),
            "group_indices",
        ),
        (lambda plan: plan["shards"][0]["group_indices"].pop(), "group_indices"),
        (lambda plan: plan["shards"][0]["group_ids"].pop(), "group_ids"),
        (
            lambda plan: plan["shards"][0]["group_ids"].__setitem__(0, "0" * 64),
            "group_ids",
        ),
        (lambda plan: plan["shards"][0]["group_keys"].pop(), "group_keys"),
        (
            lambda plan: plan["shards"][0]["group_keys"][0].update(seed=-1),
            "group_keys",
        ),
        (
            lambda plan: plan["shards"][0]["group_keys"][0].update(seed=2026.0),
            "group_keys",
        ),
        (
            lambda plan: plan["shards"][0]["group_keys"][0].update(extra=True),
            "group_keys",
        ),
        (
            lambda plan: plan["shards"][0]["combination_ids"].pop(),
            "combination_ids",
        ),
        (
            lambda plan: plan["shards"][0]["combination_ids"].append(
                plan["shards"][0]["combination_ids"][0]
            ),
            "combination_ids",
        ),
        (
            lambda plan: plan["shards"][0]["combination_ids"].append("extra"),
            "combination_ids",
        ),
    ],
)
def test_load_rejects_resigned_structure_and_coverage_tampering(
    tmp_path: Path, mutation: Any, message: str
):
    path = tmp_path / "plan.json"
    changed = _plan(shard_count=2)
    mutation(changed)
    _write_raw(path, _resign(changed))

    with pytest.raises(ValueError, match=message):
        load_shard_plan(
            path, config=_server_config(), git_commit=GIT_COMMIT, device="cuda"
        )


def test_load_rejects_changed_current_config(tmp_path: Path):
    config = _server_config()
    path = write_shard_plan(tmp_path / "plan.json", _plan(shard_count=2))
    config["epochs"] += 1

    with pytest.raises(ValueError, match="source_config_sha256"):
        load_shard_plan(path, config=config, git_commit=GIT_COMMIT, device="cuda")


@pytest.mark.parametrize("device", ["auto", "", "CUDA"])
def test_load_requires_resolved_device(tmp_path: Path, device: str):
    path = write_shard_plan(tmp_path / "plan.json", _plan(shard_count=2))

    with pytest.raises(ValueError, match="device"):
        load_shard_plan(
            path, config=_server_config(), git_commit=GIT_COMMIT, device=device
        )
