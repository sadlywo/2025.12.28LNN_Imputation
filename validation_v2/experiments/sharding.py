"""Deterministic, content-addressed plans for sharded validation runs."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Union

import yaml

from .groups import TrainingGroup, enumerate_training_groups, group_execution_config
from .provenance import canonical_json
from .runner import run_smoke


SHARD_SCHEMA_VERSION = 1

_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "created_at",
        "source_config_sha256",
        "git_commit",
        "device",
        "shard_count",
        "total_groups",
        "total_cells",
        "shards",
        "plan_sha256",
    }
)
_SHARD_FIELDS = frozenset(
    {
        "shard_index",
        "group_indices",
        "group_ids",
        "group_keys",
        "combination_ids",
    }
)
_GROUP_KEY_FIELDS = frozenset(
    {"training_family", "seed", "protocol", "objective"}
)
_RUN_ID = re.compile(r"[0-9a-f]{16}")
_RUN_FILES = frozenset(
    {
        "run.json",
        "history.json",
        "best.pt",
        "checkpoint.json",
        "test_evaluation.json",
        "per_record_metrics.csv",
    }
)
_FORBIDDEN_ROOT_FILES = frozenset(
    {"matrix_execution.json", "smoke_summary.json", "validation_report.json"}
)


def _validate_shard_count(shard_count: Any) -> int:
    if isinstance(shard_count, bool) or not isinstance(shard_count, int):
        raise ValueError("shard_count must be a positive integer")
    if shard_count <= 0:
        raise ValueError("shard_count must be a positive integer")
    return shard_count


def _validate_git_commit(git_commit: Any) -> str:
    if not isinstance(git_commit, str) or not git_commit.strip():
        raise ValueError("git_commit must be a non-empty string")
    return git_commit


def _validate_device(device: Any) -> str:
    if not isinstance(device, str) or device not in {"cpu", "cuda"}:
        raise ValueError("device must be a resolved device: cpu or cuda")
    return device


def _source_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key != "output_root" and not key.startswith("_")
    }


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _group_key(group: TrainingGroup) -> dict[str, Any]:
    return {
        "training_family": group.training_family,
        "seed": group.seed,
        "protocol": group.protocol,
        "objective": group.objective,
    }


def _shards(groups: tuple[TrainingGroup, ...], shard_count: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        assigned = [
            group for group in groups if group.group_index % shard_count == shard_index
        ]
        result.append(
            {
                "shard_index": shard_index,
                "group_indices": [group.group_index for group in assigned],
                "group_ids": [group.group_id for group in assigned],
                "group_keys": [_group_key(group) for group in assigned],
                "combination_ids": [
                    combination_id
                    for group in assigned
                    for combination_id in group.combination_ids
                ],
            }
        )
    return result


def _plan_hash(plan: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in plan.items()
        if key not in {"created_at", "plan_sha256"}
    }
    return _sha256(payload)


def build_shard_plan(
    config: Mapping[str, Any],
    *,
    shard_count: int,
    git_commit: str,
    device: str = "cuda",
) -> dict[str, Any]:
    """Build a deterministic round-robin plan for all validation training groups."""

    shard_count = _validate_shard_count(shard_count)
    git_commit = _validate_git_commit(git_commit)
    device = _validate_device(device)
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")

    groups = enumerate_training_groups(config)
    plan: dict[str, Any] = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "source_config_sha256": _sha256(_source_config(config)),
        "git_commit": git_commit,
        "device": device,
        "shard_count": shard_count,
        "total_groups": len(groups),
        "total_cells": sum(len(group.combination_ids) for group in groups),
        "shards": _shards(groups, shard_count),
    }
    plan["plan_sha256"] = _plan_hash(plan)
    return plan


def write_shard_plan(
    path: Union[Path, str], plan: Mapping[str, Any]
) -> Path:
    """Atomically create a canonical plan without replacing different content."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (canonical_json(plan) + "\n").encode("utf-8")

    temporary: Union[Path, None] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        while True:
            try:
                os.link(temporary, path)
                return path
            except FileExistsError:
                try:
                    existing = path.read_bytes()
                except FileNotFoundError:
                    continue
                if existing == content:
                    return path
                raise ValueError(
                    "shard plan already exists with different content"
                )
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _validate_created_at(value: Any) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("created_at must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("created_at must be a UTC timestamp") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("created_at must be a UTC timestamp")


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"invalid {label} fields")


def _validate_plan_structure(plan: Mapping[str, Any]) -> None:
    _require_exact_fields(plan, _PLAN_FIELDS, "shard plan")
    if type(plan["schema_version"]) is not int:
        raise ValueError("schema_version must be an integer")
    if plan["schema_version"] != SHARD_SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    _validate_created_at(plan["created_at"])
    for field in ("source_config_sha256", "plan_sha256"):
        value = plan[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{field} must be 64 lowercase hex")
    _validate_git_commit(plan["git_commit"])
    _validate_device(plan["device"])
    _validate_shard_count(plan["shard_count"])
    for field in ("total_groups", "total_cells"):
        if type(plan[field]) is not int or plan[field] < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    if not isinstance(plan["shards"], list):
        raise ValueError("shards must be a list")


def _validate_shard_shape(shard: Any, expected_index: int) -> None:
    if not isinstance(shard, Mapping):
        raise ValueError("each shard must be a mapping")
    _require_exact_fields(shard, _SHARD_FIELDS, "shard")
    if type(shard["shard_index"]) is not int or shard["shard_index"] != expected_index:
        raise ValueError("invalid shard_index")
    for field in ("group_indices", "group_ids", "group_keys", "combination_ids"):
        if not isinstance(shard[field], list):
            raise ValueError(f"{field} must be a list")
    for group_key in shard["group_keys"]:
        if not isinstance(group_key, Mapping):
            raise ValueError("group_keys must contain mappings")
        _require_exact_fields(group_key, _GROUP_KEY_FIELDS, "group_keys")


def load_shard_plan(
    path: Union[Path, str],
    *,
    config: Mapping[str, Any],
    git_commit: str,
    device: str,
) -> dict[str, Any]:
    """Strictly load and validate a plan against its current execution inputs."""

    git_commit = _validate_git_commit(git_commit)
    device = _validate_device(device)
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    try:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("unable to load shard plan") from error
    if not isinstance(loaded, Mapping):
        raise ValueError("shard plan must be a mapping")

    return _validate_plan_mapping(
        loaded, config=config, git_commit=git_commit, device=device
    )


def _validate_plan_mapping(
    loaded: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    git_commit: str,
    device: str,
) -> dict[str, Any]:
    """Validate an in-memory plan with the same strictness as file loading."""

    git_commit = _validate_git_commit(git_commit)
    device = _validate_device(device)
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    if not isinstance(loaded, Mapping):
        raise ValueError("shard plan must be a mapping")
    plan = dict(loaded)
    _validate_plan_structure(plan)
    try:
        actual_plan_sha256 = _plan_hash(plan)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid shard plan values") from error
    if plan["plan_sha256"] != actual_plan_sha256:
        raise ValueError("plan_sha256 does not match shard plan content")
    if plan["source_config_sha256"] != _sha256(_source_config(config)):
        raise ValueError("source_config_sha256 does not match current config")
    if plan["git_commit"] != git_commit:
        raise ValueError("git_commit does not match requested commit")
    if plan["device"] != device:
        raise ValueError("device does not match requested device")

    groups = enumerate_training_groups(config)
    if plan["total_groups"] != len(groups):
        raise ValueError("total_groups does not match current groups")
    total_cells = sum(len(group.combination_ids) for group in groups)
    if plan["total_cells"] != total_cells:
        raise ValueError("total_cells does not match current groups")
    if len(plan["shards"]) != plan["shard_count"]:
        raise ValueError("shard_count does not match shards")

    expected_shards = _shards(groups, plan["shard_count"])
    for shard_index, (shard, expected) in enumerate(
        zip(plan["shards"], expected_shards)
    ):
        _validate_shard_shape(shard, shard_index)
        for field in (
            "group_indices",
            "group_ids",
            "group_keys",
            "combination_ids",
        ):
            try:
                matches = canonical_json(shard[field]) == canonical_json(
                    expected[field]
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{field} contain invalid values"
                ) from error
            if not matches:
                raise ValueError(f"{field} do not match current training groups")
    return plan


def _current_git_commit(repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("unable to determine current git_commit") from error
    commit = completed.stdout.strip()
    if not commit:
        raise ValueError("unable to determine current git_commit")
    return commit


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    content = (canonical_json(value) + "\n").encode("utf-8")
    temporary: Union[Path, None] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _run_group(
    config: Mapping[str, Any],
    group: TrainingGroup,
    repository_root: Path,
    output_root: Path,
    requested_device: str,
) -> Mapping[str, Any]:
    """Execute exactly one pre-enumerated training group."""

    return run_smoke(
        group_execution_config(config, group),
        repository_root=repository_root,
        output_root=output_root,
        requested_device=requested_device,
    )


def _immutable_marker_fields(
    plan: Mapping[str, Any], shard: Mapping[str, Any], shard_index: int
) -> dict[str, Any]:
    return {
        "schema_version": SHARD_SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "source_config_sha256": plan["source_config_sha256"],
        "git_commit": plan["git_commit"],
        "device": plan["device"],
        "shard_index": shard_index,
        "shard_count": plan["shard_count"],
        "group_ids": list(shard["group_ids"]),
        "combination_ids": list(shard["combination_ids"]),
    }


def _validate_run_directory(run_dir: Path) -> None:
    missing = sorted(name for name in _RUN_FILES if not (run_dir / name).is_file())
    if missing:
        raise ValueError(f"incomplete run artifacts in {run_dir.name}: {missing}")
    try:
        ledger = json.loads(
            (run_dir / "test_evaluation.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid test_evaluation ledger in {run_dir.name}") from error
    if not isinstance(ledger, Mapping) or ledger.get("status") != "completed":
        raise ValueError(f"test_evaluation ledger is not completed in {run_dir.name}")


def _root_run_ids(output_root: Path) -> list[str]:
    run_ids: list[str] = []
    for child in output_root.iterdir():
        if child.name == "shard_execution.json":
            continue
        if child.name in _FORBIDDEN_ROOT_FILES:
            raise ValueError(f"forbidden marker in shard output root: {child.name}")
        if child.is_file() and (
            child.match("split_manifest-*.csv") or child.match("scaler-*.json")
        ):
            continue
        if child.is_dir() and _RUN_ID.fullmatch(child.name):
            run_ids.append(child.name)
            continue
        raise ValueError(f"foreign content in exclusive shard output root: {child.name}")
    return sorted(run_ids)


def _validate_marker(
    marker: Any,
    immutable: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    if not isinstance(marker, Mapping):
        raise ValueError("shard execution marker must be a mapping")
    value = dict(marker)
    status = value.get("status")
    common = set(immutable) | {
        "status",
        "started_at",
        "completed_group_ids",
        "run_ids",
    }
    expected_fields = {
        "started": common,
        "completed": common | {"completed_at"},
        "failed": common | {"error_type"},
    }.get(status)
    if expected_fields is None or set(value) != expected_fields:
        raise ValueError("invalid shard execution marker fields or status")
    for field, expected in immutable.items():
        if value.get(field) != expected:
            raise ValueError(f"immutable shard marker mismatch: {field}")
    _validate_created_at(value["started_at"])
    if status == "completed":
        _validate_created_at(value["completed_at"])
    if status == "failed" and (
        not isinstance(value["error_type"], str) or not value["error_type"]
    ):
        raise ValueError("invalid shard execution error_type")
    assigned = list(immutable["group_ids"])
    completed = value["completed_group_ids"]
    run_ids = value["run_ids"]
    if not isinstance(completed, list) or completed != assigned[: len(completed)]:
        raise ValueError("completed_group_ids must be an assigned prefix")
    if status == "completed" and completed != assigned:
        raise ValueError("completed_group_ids do not cover the completed shard")
    if not isinstance(run_ids, list) or any(
        not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id)
        for run_id in run_ids
    ):
        raise ValueError("invalid run_ids in shard execution marker")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("duplicate run_ids in shard execution marker")
    if completed and not run_ids:
        raise ValueError("completed groups require non-empty run_ids")
    if not assigned and run_ids:
        raise ValueError("empty shard must have empty run_ids")
    actual_run_ids = _root_run_ids(output_root)
    if sorted(run_ids) != actual_run_ids:
        raise ValueError("run_ids do not match shard run directories")
    for run_id in run_ids:
        _validate_run_directory(output_root / run_id)
    return value


def _load_execution_marker(
    marker_path: Path,
    immutable: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("unable to load shard execution marker") from error
    return _validate_marker(marker, immutable, output_root)


def execute_shard(
    config: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    shard_index: int,
    repository_root: Union[Path, str],
    output_root: Union[Path, str],
    requested_device: str,
) -> Mapping[str, Any]:
    """Execute one isolated shard, with atomic resumable state transitions."""

    if type(shard_index) is not int:
        raise ValueError("shard_index must be an integer")
    repository_root = Path(repository_root)
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = repository_root / output_root
    git_commit = _current_git_commit(repository_root)
    validated_plan = _validate_plan_mapping(
        plan,
        config=config,
        git_commit=git_commit,
        device=requested_device,
    )
    if shard_index < 0 or shard_index >= validated_plan["shard_count"]:
        raise ValueError("shard_index is outside the shard plan")
    shard = validated_plan["shards"][shard_index]
    immutable = _immutable_marker_fields(validated_plan, shard, shard_index)
    marker_path = output_root / "shard_execution.json"

    if marker_path.exists():
        marker = _load_execution_marker(marker_path, immutable, output_root)
        if marker["status"] == "failed":
            raise ValueError("failed shard execution cannot be resumed")
        if marker["status"] == "completed":
            return marker
    else:
        if output_root.exists() and any(output_root.iterdir()):
            names = {child.name for child in output_root.iterdir()}
            forbidden = sorted(names & _FORBIDDEN_ROOT_FILES)
            if forbidden:
                raise ValueError(f"forbidden marker in shard output root: {forbidden[0]}")
            raise ValueError("foreign content in exclusive shard output root")
        output_root.mkdir(parents=True, exist_ok=True)
        marker = {
            **immutable,
            "status": "started",
            "started_at": _timestamp(),
            "completed_group_ids": [],
            "run_ids": [],
        }
        _atomic_write_json(marker_path, marker)

    groups = enumerate_training_groups(config)
    assigned_groups = [groups[index] for index in shard["group_indices"]]
    completed_count = len(marker["completed_group_ids"])
    try:
        for group in assigned_groups[completed_count:]:
            report = _run_group(
                config, group, repository_root, output_root, requested_device
            )
            if not isinstance(report, Mapping):
                raise ValueError("group execution report must be a mapping")
            new_run_ids = report.get("run_ids")
            if not isinstance(new_run_ids, list) or not new_run_ids:
                raise ValueError("group execution must return non-empty run_ids")
            if any(
                not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id)
                for run_id in new_run_ids
            ):
                raise ValueError("group execution returned invalid run_ids")
            if set(new_run_ids) & set(marker["run_ids"]):
                raise ValueError("group execution returned duplicate run_ids")
            for run_id in new_run_ids:
                _validate_run_directory(output_root / run_id)
            marker["completed_group_ids"].append(group.group_id)
            marker["run_ids"].extend(new_run_ids)
            if sorted(marker["run_ids"]) != _root_run_ids(output_root):
                raise ValueError("run_ids do not match shard run directories")
            _atomic_write_json(marker_path, marker)
    except BaseException as error:
        marker["status"] = "failed"
        marker["error_type"] = type(error).__name__
        _atomic_write_json(marker_path, marker)
        raise

    marker["status"] = "completed"
    marker["completed_at"] = _timestamp()
    _atomic_write_json(marker_path, marker)
    return marker


__all__ = [
    "SHARD_SCHEMA_VERSION",
    "build_shard_plan",
    "execute_shard",
    "load_shard_plan",
    "write_shard_plan",
]
