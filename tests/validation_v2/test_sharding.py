from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import pytest
import yaml

from validation_v2.experiments import sharding
from validation_v2.experiments.groups import enumerate_training_groups
from validation_v2.experiments.matrix import enumerate_matrix
from validation_v2.experiments.provenance import canonical_json
from validation_v2.experiments.sharding import (
    SHARD_SCHEMA_VERSION,
    build_shard_plan,
    load_shard_plan,
    write_shard_plan,
)
from test_server_handoff import _complete_root

_run_group = getattr(sharding, "_run_group", None)
execute_shard = getattr(sharding, "execute_shard", None)
merge_shards = getattr(sharding, "merge_shards", None)
preflight_shards = getattr(sharding, "preflight_shards", None)


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


RUN_FILES = (
    "run.json",
    "history.json",
    "best.pt",
    "checkpoint.json",
    "test_evaluation.json",
    "per_record_metrics.csv",
)


def _mini_config() -> dict[str, Any]:
    config = _server_config()
    config.update(
        models=["linear", "locf", "bilstm"],
        seeds=[2026],
        protocols=["strict_file"],
        topologies=["point"],
        rates=[0.3],
        irregular_cases=[],
    )
    return config


def _head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _execution_plan(config: Mapping[str, Any], shard_count: int = 2) -> dict[str, Any]:
    return build_shard_plan(
        config, shard_count=shard_count, git_commit=_head(), device="cpu"
    )


def _write_complete_run(
    output_root: Path,
    run_id: str,
    group: Any,
    *,
    git_commit: str | None = None,
    device: str = "cpu",
) -> None:
    run_dir = output_root / run_id
    if run_dir.exists():
        return
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "git_commit": git_commit or _head(),
        "seed": group.seed,
        "config": {
            "model": group.training_model,
            "training_family": group.training_family,
            "reported_models": list(group.reported_models),
            "seed": group.seed,
            "protocol": group.protocol,
            "objective": group.objective,
            "condition_list": list(group.conditions),
            "resolved_device": device,
        },
    }
    _write_raw(run_dir / "run.json", manifest)
    _write_raw(run_dir / "history.json", [])
    (run_dir / "best.pt").write_bytes(b"checkpoint")
    _write_raw(run_dir / "checkpoint.json", {})
    _write_raw(run_dir / "test_evaluation.json", {"status": "completed"})
    (run_dir / "per_record_metrics.csv").write_text(
        "run_id,metric,value\n", encoding="utf-8"
    )


def _fake_group_runner(calls: list[str]):
    def fake(
        config: Mapping[str, Any],
        group: Any,
        repository_root: Path,
        output_root: Path,
        requested_device: str,
    ) -> Mapping[str, Any]:
        calls.append(group.group_id)
        run_id = group.group_id[:16]
        _write_complete_run(
            output_root,
            run_id,
            group,
            git_commit=_head(),
            device=requested_device,
        )
        (output_root / "split_manifest-test.csv").write_text("split\n", encoding="utf-8")
        (output_root / "scaler-test.json").write_text("{}\n", encoding="utf-8")
        return {"status": "completed", "run_ids": [run_id]}

    return fake


def _merge_config() -> dict[str, Any]:
    config = _mini_config()
    config["models"] = ["linear"]
    config["protocols"] = ["global_random", "strict_file"]
    return config


def _write_merge_fixture(tmp_path: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    config = _merge_config()
    commit = "a" * 40
    plan = build_shard_plan(
        config, shard_count=2, git_commit=commit, device="cpu"
    )
    shards_root = tmp_path / "shards"
    shards_root.mkdir(parents=True)
    groups = enumerate_training_groups(config)

    for shard_index, shard in enumerate(plan["shards"]):
        group = groups[shard["group_indices"][0]]
        generated, _, manifest = _complete_root(
            tmp_path / f"generated-{shard_index}",
            protocol=group.protocol,
            condition_id=group.combination_ids[0],
        )
        shard_root = shards_root / f"{shard_index:03d}"
        generated.replace(shard_root)
        (shard_root / "matrix_execution.json").unlink()
        _write_raw(
            shard_root / "shard_execution.json",
            {
                "schema_version": SHARD_SCHEMA_VERSION,
                "plan_sha256": plan["plan_sha256"],
                "source_config_sha256": plan["source_config_sha256"],
                "git_commit": commit,
                "device": "cpu",
                "shard_index": shard_index,
                "shard_count": 2,
                "group_ids": list(shard["group_ids"]),
                "combination_ids": list(shard["combination_ids"]),
                "status": "completed",
                "started_at": "2026-07-12T00:00:00Z",
                "completed_at": "2026-07-12T00:01:00Z",
                "completed_group_ids": list(shard["group_ids"]),
                "run_ids": [manifest["run_id"]],
                "group_runs": [
                    {"group_id": group.group_id, "run_ids": [manifest["run_id"]]}
                ],
            },
        )
        (shard_root / ".shard_execution.lock").write_bytes(b"\0")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    write_shard_plan(plan_path, plan)
    return plan, shards_root, config_path, plan_path


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_preflight_is_read_only_and_returns_complete_promotion_manifest(
    tmp_path: Path,
):
    plan, shards_root, config_path, _ = _write_merge_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    before = _tree_snapshot(shards_root)

    promotion = preflight_shards(config, plan=plan, shards_root=shards_root)

    assert _tree_snapshot(shards_root) == before
    assert promotion["plan_sha256"] == plan["plan_sha256"]
    assert promotion["source_config_sha256"] == plan["source_config_sha256"]
    assert promotion["git_commit"] == plan["git_commit"]
    assert promotion["device"] == "cpu"
    assert promotion["total_groups"] == 2
    assert promotion["total_cells"] == 2
    assert promotion["selected_combination_ids"] == [
        cell["combination_id"] for cell in enumerate_matrix(config)
    ]
    assert promotion["run_ids"] == sorted(promotion["run_ids"])
    assert len(promotion["run_sources"]) == 2
    assert len(promotion["asset_sources"]) == 4


def test_merge_two_formal_shards_validates_and_preserves_sources(tmp_path: Path):
    plan, shards_root, config_path, plan_path = _write_merge_fixture(tmp_path)
    before = _tree_snapshot(shards_root)
    output_root = tmp_path / "merged"

    report = merge_shards(
        config_path=config_path,
        plan_path=plan_path,
        shards_root=shards_root,
        output_root=output_root,
    )

    assert report["status"] == "complete"
    assert _tree_snapshot(shards_root) == before
    assert (output_root / "validation_report.json").is_file()
    assert not (output_root / "shard_execution.json").exists()
    assert not (output_root / ".shard_execution.lock").exists()
    marker = json.loads(
        (output_root / "matrix_execution.json").read_text(encoding="utf-8")
    )
    assert marker == {
        "status": "completed",
        "partial": False,
        "selected_cells": plan["total_cells"],
        "total_cells": plan["total_cells"],
        "training_groups": plan["total_groups"],
        "grouping_key": ["training_family", "seed", "protocol", "objective"],
        "selected_combination_ids": [
            cell["combination_id"]
            for cell in enumerate_matrix(
                yaml.safe_load(config_path.read_text(encoding="utf-8"))
            )
        ],
        "run_ids": sorted(report["run_ids"]),
    }
    assert not list(tmp_path.glob(".merged-merge-*"))
    assert not list(tmp_path.glob(".failed-merge-*"))


@pytest.mark.parametrize("status", ["started", "failed"])
def test_preflight_rejects_incomplete_shard_status(tmp_path: Path, status: str):
    plan, shards_root, config_path, _ = _write_merge_fixture(tmp_path)
    marker_path = shards_root / "000" / "shard_execution.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["status"] = status
    if status == "started":
        marker.pop("completed_at")
    else:
        marker["error_type"] = "RuntimeError"
    _write_raw(marker_path, marker)

    with pytest.raises(ValueError, match="completed|status|failed"):
        preflight_shards(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            plan=plan,
            shards_root=shards_root,
        )


def test_preflight_rejects_foreign_or_missing_shard_directories(tmp_path: Path):
    plan, shards_root, config_path, _ = _write_merge_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    (shards_root / "README.md").write_text("foreign\n", encoding="utf-8")
    with pytest.raises(ValueError, match="foreign|shard director"):
        preflight_shards(config, plan=plan, shards_root=shards_root)
    (shards_root / "README.md").unlink()
    shutil.rmtree(shards_root / "001")
    with pytest.raises(ValueError, match="missing|shard director"):
        preflight_shards(config, plan=plan, shards_root=shards_root)


def test_preflight_rejects_tampered_asset_and_run(tmp_path: Path):
    plan, shards_root, config_path, _ = _write_merge_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    asset = next((shards_root / "000").glob("scaler-*.json"))
    asset.write_bytes(asset.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256|digest|asset"):
        preflight_shards(config, plan=plan, shards_root=shards_root)

    plan, shards_root, config_path, _ = _write_merge_fixture(tmp_path / "run-case")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    marker = json.loads(
        (shards_root / "000" / "shard_execution.json").read_text(encoding="utf-8")
    )
    (shards_root / "000" / marker["run_ids"][0] / "best.pt").unlink()
    with pytest.raises(ValueError, match="incomplete|artifacts"):
        preflight_shards(config, plan=plan, shards_root=shards_root)


def test_preflight_rejects_duplicate_run_id_and_asset_name_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plan, shards_root, config_path, _ = _write_merge_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    first_marker = json.loads(
        (shards_root / "000" / "shard_execution.json").read_text(encoding="utf-8")
    )
    second_marker_path = shards_root / "001" / "shard_execution.json"
    second_marker = json.loads(second_marker_path.read_text(encoding="utf-8"))
    old_run = shards_root / "001" / second_marker["run_ids"][0]
    duplicate_run = shards_root / "001" / first_marker["run_ids"][0]
    old_run.rename(duplicate_run)
    manifest_path = duplicate_run / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = duplicate_run.name
    _write_raw(manifest_path, manifest)
    second_marker["run_ids"] = [duplicate_run.name]
    second_marker["group_runs"][0]["run_ids"] = [duplicate_run.name]
    _write_raw(second_marker_path, second_marker)
    with pytest.raises(ValueError, match="duplicate run"):
        preflight_shards(config, plan=plan, shards_root=shards_root)

    plan, shards_root, config_path, _ = _write_merge_fixture(tmp_path / "asset-case")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = next((shards_root / "000").glob("scaler-*.json"))
    source_digest = source.stem.removeprefix("scaler-")
    second_run = json.loads(
        (shards_root / "001" / "shard_execution.json").read_text(encoding="utf-8")
    )["run_ids"][0]
    second_manifest_path = shards_root / "001" / second_run / "run.json"
    second_manifest = json.loads(second_manifest_path.read_text(encoding="utf-8"))
    old_scaler = shards_root / "001" / f"scaler-{second_manifest['scaler_hash']}.json"
    old_scaler.unlink()
    second_manifest["scaler_hash"] = source_digest
    _write_raw(second_manifest_path, second_manifest)
    target = shards_root / "001" / source.name
    target.write_bytes(b"different")
    real_file_sha256 = sharding._file_sha256

    def accept_named_asset(path: Path) -> str:
        path = Path(path)
        match = sharding._SCALER_ASSET.fullmatch(path.name)
        return match.group(1) if match else real_file_sha256(path)

    monkeypatch.setattr(sharding, "_file_sha256", accept_named_asset)
    with pytest.raises(ValueError, match="conflict"):
        preflight_shards(config, plan=plan, shards_root=shards_root)


def test_merge_existing_destination_is_never_modified(tmp_path: Path):
    _, shards_root, config_path, plan_path = _write_merge_fixture(tmp_path)
    output_root = tmp_path / "merged"
    output_root.mkdir()
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="exist"):
        merge_shards(
            config_path=config_path,
            plan_path=plan_path,
            shards_root=shards_root,
            output_root=output_root,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(output_root.iterdir()) == [sentinel]


@pytest.mark.parametrize("failure", ["copy", "validator", "publish"])
def test_merge_failure_keeps_no_final_and_preserves_failed_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
):
    _, shards_root, config_path, plan_path = _write_merge_fixture(tmp_path)
    before = _tree_snapshot(shards_root)
    output_root = tmp_path / "merged"

    if failure == "copy":
        monkeypatch.setattr(sharding.shutil, "copy2", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("copy boom")))
    elif failure == "validator":
        monkeypatch.setattr(sharding, "validate_artifacts", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("validator boom")))
    else:
        monkeypatch.setattr(sharding.os, "rename", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("publish boom")))

    with pytest.raises((OSError, ValueError), match=f"{failure} boom"):
        merge_shards(
            config_path=config_path,
            plan_path=plan_path,
            shards_root=shards_root,
            output_root=output_root,
        )

    assert not output_root.exists()
    assert _tree_snapshot(shards_root) == before
    failed = list(tmp_path.glob(".failed-merge-*"))
    assert len(failed) == 1
    assert failed[0].is_dir()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("plan_sha256", "0" * 64),
        ("source_config_sha256", "0" * 64),
        ("git_commit", "wrong"),
        ("device", "cuda"),
        ("shard_index", 1),
        ("shard_count", 3),
        ("group_ids", []),
        ("combination_ids", []),
    ],
)
def test_preflight_rejects_marker_plan_config_and_coverage_tampering(
    tmp_path: Path, field: str, value: Any
):
    plan, shards_root, config_path, _ = _write_merge_fixture(tmp_path)
    marker_path = shards_root / "000" / "shard_execution.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker[field] = value
    _write_raw(marker_path, marker)

    with pytest.raises(ValueError, match=field.replace("_sha256", "") + "|immutable|marker"):
        preflight_shards(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            plan=plan,
            shards_root=shards_root,
        )


@pytest.mark.parametrize("mutation", ["foreign-root", "extra-run-file", "bad-asset-name"])
def test_preflight_rejects_foreign_shard_and_run_content(
    tmp_path: Path, mutation: str
):
    plan, shards_root, config_path, _ = _write_merge_fixture(tmp_path)
    shard_root = shards_root / "000"
    marker = json.loads(
        (shard_root / "shard_execution.json").read_text(encoding="utf-8")
    )
    if mutation == "foreign-root":
        (shard_root / "notes.txt").write_text("foreign", encoding="utf-8")
    elif mutation == "extra-run-file":
        (shard_root / marker["run_ids"][0] / "partial.tmp").write_text(
            "partial", encoding="utf-8"
        )
    else:
        (shard_root / "scaler-partial.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="foreign|exactly six|content"):
        preflight_shards(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            plan=plan,
            shards_root=shards_root,
        )


@pytest.mark.parametrize("mutation", ["missing-marker", "missing-run", "extra-asset"])
def test_preflight_rejects_missing_or_extra_artifacts(tmp_path: Path, mutation: str):
    plan, shards_root, config_path, _ = _write_merge_fixture(tmp_path)
    shard_root = shards_root / "000"
    marker = json.loads(
        (shard_root / "shard_execution.json").read_text(encoding="utf-8")
    )
    if mutation == "missing-marker":
        (shard_root / "shard_execution.json").unlink()
    elif mutation == "missing-run":
        shutil.rmtree(shard_root / marker["run_ids"][0])
    else:
        content = b"unreferenced asset"
        digest = hashlib.sha256(content).hexdigest()
        (shard_root / f"scaler-{digest}.json").write_bytes(content)

    with pytest.raises(ValueError, match="missing|extra|match"):
        preflight_shards(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            plan=plan,
            shards_root=shards_root,
        )


def test_preflight_failure_creates_no_merge_temp_or_failed_directory(tmp_path: Path):
    _, shards_root, config_path, plan_path = _write_merge_fixture(tmp_path)
    (shards_root / "foreign").mkdir()

    with pytest.raises(ValueError, match="foreign|shard director"):
        merge_shards(
            config_path=config_path,
            plan_path=plan_path,
            shards_root=shards_root,
            output_root=tmp_path / "merged",
        )

    assert not (tmp_path / "merged").exists()
    assert not list(tmp_path.glob(".merged-merge-*"))
    assert not list(tmp_path.glob(".failed-merge-*"))


def test_merge_creates_output_parent(tmp_path: Path):
    _, shards_root, config_path, plan_path = _write_merge_fixture(tmp_path)
    output_root = tmp_path / "new-parent" / "merged"

    report = merge_shards(
        config_path=config_path,
        plan_path=plan_path,
        shards_root=shards_root,
        output_root=output_root,
    )

    assert report["status"] == "complete"
    assert output_root.is_dir()


def test_merge_rejects_output_inside_shards_without_mutating_sources(tmp_path: Path):
    _, shards_root, config_path, plan_path = _write_merge_fixture(tmp_path)
    before = _tree_snapshot(shards_root)

    with pytest.raises(ValueError, match="overlap|shards_root"):
        merge_shards(
            config_path=config_path,
            plan_path=plan_path,
            shards_root=shards_root,
            output_root=shards_root / "merged",
        )

    assert _tree_snapshot(shards_root) == before
    assert {path.name for path in shards_root.iterdir()} == {"000", "001"}


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


def test_run_group_delegates_group_config_to_run_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _mini_config()
    group = enumerate_training_groups(config)[0]
    captured: dict[str, Any] = {}

    def fake_run_smoke(execution: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]:
        captured.update(execution=execution, kwargs=kwargs)
        return {"run_ids": ["0123456789abcdef"]}

    monkeypatch.setattr("validation_v2.experiments.sharding.run_smoke", fake_run_smoke)
    result = _run_group(config, group, REPO_ROOT, tmp_path, "cpu")

    assert result == {"run_ids": ["0123456789abcdef"]}
    assert captured["execution"]["models"] == [group.training_model]
    assert captured["execution"]["_execution_conditions"] == list(group.conditions)
    assert captured["kwargs"] == {
        "repository_root": REPO_ROOT,
        "output_root": tmp_path,
        "requested_device": "cpu",
    }


def test_two_shards_execute_only_disjoint_exhaustive_assigned_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _mini_config()
    groups = enumerate_training_groups(config)
    plan = _execution_plan(config)
    calls: list[str] = []
    monkeypatch.setattr(
        "validation_v2.experiments.sharding._run_group", _fake_group_runner(calls)
    )

    reports = [
        execute_shard(
            config,
            plan=plan,
            shard_index=index,
            repository_root=REPO_ROOT,
            output_root=tmp_path / f"shard-{index}",
            requested_device="cpu",
        )
        for index in range(2)
    ]

    assert calls == [groups[0].group_id, groups[2].group_id, groups[1].group_id]
    assert set(reports[0]["group_ids"]).isdisjoint(reports[1]["group_ids"])
    assert set(reports[0]["group_ids"] + reports[1]["group_ids"]) == {
        group.group_id for group in groups
    }
    for index, report in enumerate(reports):
        assert report["status"] == "completed"
        assert report["shard_index"] == index
        assert report["completed_group_ids"] == report["group_ids"]
        assert report["group_runs"] == [
            {"group_id": group_id, "run_ids": [group_id[:16]]}
            for group_id in report["group_ids"]
        ]
        assert report["completed_at"].endswith("Z")
        assert json.loads(
            (tmp_path / f"shard-{index}" / "shard_execution.json").read_text(
                encoding="utf-8"
            )
        ) == report
        assert not (tmp_path / f"shard-{index}" / "matrix_execution.json").exists()
        assert not (tmp_path / f"shard-{index}" / "smoke_summary.json").exists()
        assert not (tmp_path / f"shard-{index}" / "summary.json").exists()


def test_completed_shard_rerun_is_idempotent_without_group_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _mini_config()
    plan = _execution_plan(config)
    calls: list[str] = []
    monkeypatch.setattr(
        "validation_v2.experiments.sharding._run_group", _fake_group_runner(calls)
    )
    kwargs = dict(
        plan=plan,
        shard_index=0,
        repository_root=REPO_ROOT,
        output_root=tmp_path / "shard",
        requested_device="cpu",
    )
    first = execute_shard(config, **kwargs)
    calls.clear()

    second = execute_shard(config, **kwargs)

    assert second == first
    assert calls == []


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_interrupt_preserves_started_marker_and_resumes_only_remaining_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
):
    config = _mini_config()
    plan = _execution_plan(config, shard_count=1)
    root = tmp_path / "shard"
    groups = enumerate_training_groups(config)
    calls: list[str] = []
    runner = _fake_group_runner(calls)
    monkeypatch.setattr("validation_v2.experiments.sharding._run_group", runner)

    attempts = 0

    def interrupt_after_one(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise interrupt_type
        return runner(*args, **kwargs)

    monkeypatch.setattr("validation_v2.experiments.sharding._run_group", interrupt_after_one)
    with pytest.raises(interrupt_type):
        execute_shard(
            config,
            plan=plan,
            shard_index=0,
            repository_root=REPO_ROOT,
            output_root=root,
            requested_device="cpu",
        )
    interrupted = json.loads(
        (root / "shard_execution.json").read_text(encoding="utf-8")
    )
    assert interrupted["status"] == "started"
    assert interrupted["completed_group_ids"] == [groups[0].group_id]
    assert interrupted["run_ids"] == [groups[0].group_id[:16]]
    calls.clear()

    report = execute_shard(
        config,
        plan=plan,
        shard_index=0,
        repository_root=REPO_ROOT,
        output_root=root,
        requested_device="cpu",
    )

    assert calls == [group.group_id for group in groups[1:]]
    assert report["completed_group_ids"] == [group.group_id for group in groups]


def test_completed_run_before_marker_update_is_claimed_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _mini_config()
    plan = _execution_plan(config, shard_count=1)
    root = tmp_path / "shard"
    groups = enumerate_training_groups(config)
    calls: list[str] = []
    runner = _fake_group_runner(calls)
    attempts = 0

    def interrupt_after_completed_run(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        nonlocal attempts
        attempts += 1
        report = runner(*args, **kwargs)
        if attempts == 2:
            raise KeyboardInterrupt
        return report

    monkeypatch.setattr(
        "validation_v2.experiments.sharding._run_group", interrupt_after_completed_run
    )
    with pytest.raises(KeyboardInterrupt):
        execute_shard(
            config, plan=plan, shard_index=0, repository_root=REPO_ROOT,
            output_root=root, requested_device="cpu"
        )
    interrupted = json.loads(
        (root / "shard_execution.json").read_text(encoding="utf-8")
    )
    assert interrupted["completed_group_ids"] == [groups[0].group_id]
    assert (root / groups[1].group_id[:16]).is_dir()
    calls.clear()
    monkeypatch.setattr("validation_v2.experiments.sharding._run_group", runner)

    report = execute_shard(
        config, plan=plan, shard_index=0, repository_root=REPO_ROOT,
        output_root=root, requested_device="cpu"
    )

    assert calls == [groups[1].group_id, groups[2].group_id]
    assert report["group_runs"] == [
        {"group_id": group.group_id, "run_ids": [group.group_id[:16]]}
        for group in groups
    ]


def test_unregistered_complete_run_for_wrong_next_group_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _mini_config()
    plan = _execution_plan(config, shard_count=1)
    root = tmp_path / "shard"
    groups = enumerate_training_groups(config)
    runner = _fake_group_runner([])
    attempts = 0

    def forge_next_group(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            wrong = groups[2]
            _write_complete_run(root, wrong.group_id[:16], wrong)
            raise KeyboardInterrupt
        return runner(*args, **kwargs)

    monkeypatch.setattr("validation_v2.experiments.sharding._run_group", forge_next_group)
    with pytest.raises(KeyboardInterrupt):
        execute_shard(
            config, plan=plan, shard_index=0, repository_root=REPO_ROOT,
            output_root=root, requested_device="cpu"
        )

    with pytest.raises(ValueError, match="manifest|assigned group|group binding"):
        execute_shard(
            config, plan=plan, shard_index=0, repository_root=REPO_ROOT,
            output_root=root, requested_device="cpu"
        )


def test_active_shard_lock_rejects_second_holder_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _mini_config()
    plan = _execution_plan(config)
    root = tmp_path / "shard"
    root.mkdir()
    monkeypatch.setattr(
        "validation_v2.experiments.sharding._run_group", _fake_group_runner([])
    )

    with sharding._shard_execution_lock(root):
        with pytest.raises(ValueError, match="active|locked"):
            execute_shard(
                config, plan=plan, shard_index=0, repository_root=REPO_ROOT,
                output_root=root, requested_device="cpu"
            )

    report = execute_shard(
        config, plan=plan, shard_index=0, repository_root=REPO_ROOT,
        output_root=root, requested_device="cpu"
    )
    assert report["status"] == "completed"
    assert (root / ".shard_execution.lock").is_file()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("shard_index", False),
        ("shard_count", 2.0),
    ],
)
def test_marker_integer_immutables_reject_bool_and_float_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
):
    config = _mini_config()
    plan = _execution_plan(config)
    root = tmp_path / "shard"
    monkeypatch.setattr(
        "validation_v2.experiments.sharding._run_group", _fake_group_runner([])
    )
    execute_shard(
        config, plan=plan, shard_index=0, repository_root=REPO_ROOT,
        output_root=root, requested_device="cpu"
    )
    marker_path = root / "shard_execution.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker[field] = value
    _write_raw(marker_path, marker)

    with pytest.raises(ValueError, match=field):
        execute_shard(
            config, plan=plan, shard_index=0, repository_root=REPO_ROOT,
            output_root=root, requested_device="cpu"
        )


def test_started_resume_rejects_partial_run_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _mini_config()
    plan = _execution_plan(config)
    root = tmp_path / "shard"
    calls: list[str] = []
    monkeypatch.setattr(
        "validation_v2.experiments.sharding._run_group", _fake_group_runner(calls)
    )
    execute_shard(
        config,
        plan=plan,
        shard_index=0,
        repository_root=REPO_ROOT,
        output_root=root,
        requested_device="cpu",
    )
    marker = json.loads((root / "shard_execution.json").read_text(encoding="utf-8"))
    marker["status"] = "started"
    marker.pop("completed_at")
    _write_raw(root / "shard_execution.json", marker)
    (root / marker["run_ids"][0] / "best.pt").unlink()

    with pytest.raises(ValueError, match="incomplete|artifacts"):
        execute_shard(
            config,
            plan=plan,
            shard_index=0,
            repository_root=REPO_ROOT,
            output_root=root,
            requested_device="cpu",
        )


@pytest.mark.parametrize("shard_index", [True, False, -1, 2, 1.0, "0"])
def test_execute_rejects_invalid_shard_index(tmp_path: Path, shard_index: Any):
    config = _mini_config()
    with pytest.raises(ValueError, match="shard_index"):
        execute_shard(
            config,
            plan=_execution_plan(config),
            shard_index=shard_index,
            repository_root=REPO_ROOT,
            output_root=tmp_path / "shard",
            requested_device="cpu",
        )


@pytest.mark.parametrize("forbidden", ["matrix_execution.json", "smoke_summary.json", "validation_report.json"])
def test_new_shard_root_rejects_forbidden_markers(tmp_path: Path, forbidden: str):
    config = _mini_config()
    root = tmp_path / "shard"
    root.mkdir()
    (root / forbidden).write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden|exclusive"):
        execute_shard(
            config,
            plan=_execution_plan(config),
            shard_index=0,
            repository_root=REPO_ROOT,
            output_root=root,
            requested_device="cpu",
        )


def test_new_shard_root_rejects_foreign_directory(tmp_path: Path):
    config = _mini_config()
    root = tmp_path / "shard"
    (root / "foreign").mkdir(parents=True)

    with pytest.raises(ValueError, match="foreign|exclusive"):
        execute_shard(
            config,
            plan=_execution_plan(config),
            shard_index=0,
            repository_root=REPO_ROOT,
            output_root=root,
            requested_device="cpu",
        )


def test_failed_and_mismatched_execution_markers_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _mini_config()
    plan = _execution_plan(config)
    root = tmp_path / "shard"

    def fail(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr("validation_v2.experiments.sharding._run_group", fail)
    with pytest.raises(RuntimeError, match="boom"):
        execute_shard(
            config, plan=plan, shard_index=0, repository_root=REPO_ROOT,
            output_root=root, requested_device="cpu"
        )
    marker = json.loads((root / "shard_execution.json").read_text(encoding="utf-8"))
    assert marker["status"] == "failed"
    assert marker["error_type"] == "RuntimeError"
    with pytest.raises(ValueError, match="failed"):
        execute_shard(
            config, plan=plan, shard_index=0, repository_root=REPO_ROOT,
            output_root=root, requested_device="cpu"
        )
    marker["status"] = "started"
    marker.pop("error_type")
    marker["device"] = "cuda"
    _write_raw(root / "shard_execution.json", marker)
    with pytest.raises(ValueError, match="device|immutable"):
        execute_shard(
            config, plan=plan, shard_index=0, repository_root=REPO_ROOT,
            output_root=root, requested_device="cpu"
        )


def test_execute_rejects_plan_git_and_device_mismatch(tmp_path: Path):
    config = _mini_config()
    wrong_git = build_shard_plan(
        config, shard_count=1, git_commit="not-current", device="cpu"
    )
    with pytest.raises(ValueError, match="git_commit"):
        execute_shard(
            config, plan=wrong_git, shard_index=0, repository_root=REPO_ROOT,
            output_root=tmp_path / "git", requested_device="cpu"
        )
    cpu_plan = _execution_plan(config, shard_count=1)
    with pytest.raises(ValueError, match="device"):
        execute_shard(
            config, plan=cpu_plan, shard_index=0, repository_root=REPO_ROOT,
            output_root=tmp_path / "device", requested_device="cuda"
        )


def test_execution_marker_updates_leave_no_atomic_temp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _mini_config()
    root = tmp_path / "shard"
    monkeypatch.setattr(
        "validation_v2.experiments.sharding._run_group", _fake_group_runner([])
    )
    execute_shard(
        config, plan=_execution_plan(config), shard_index=0,
        repository_root=REPO_ROOT, output_root=root, requested_device="cpu"
    )

    assert list(root.glob(".shard_execution.json-*.tmp")) == []
