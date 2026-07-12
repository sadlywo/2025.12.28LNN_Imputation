"""Deterministic, content-addressed plans for sharded validation runs."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Union
import uuid

import yaml

from .groups import TrainingGroup, enumerate_training_groups, group_execution_config
from .matrix import enumerate_matrix
from .provenance import canonical_json
from .runner import run_smoke
from .validate_artifacts import validate_artifacts


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
_LOCK_FILE = ".shard_execution.lock"
_MERGE_LOCK_FILE = ".validation-v2-merge.lock"
_SPLIT_ASSET = re.compile(r"split_manifest-([0-9a-f]{64})\.csv")
_SCALER_ASSET = re.compile(r"scaler-([0-9a-f]{64})\.json")


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


@contextmanager
def _shard_execution_lock(output_root: Path):
    """Hold a non-blocking process lock for one shard output root."""

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / _LOCK_FILE
    handle = lock_path.open("a+b")
    locked = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise ValueError("shard output root is active or locked") from error
        yield
    finally:
        if locked:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        else:
            handle.close()


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


def _validate_run_directory(
    run_dir: Path,
    group: TrainingGroup,
    *,
    git_commit: str,
    device: str,
) -> None:
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
    try:
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid run manifest in {run_dir.name}") from error
    if not isinstance(manifest, Mapping):
        raise ValueError(f"run manifest must be a mapping in {run_dir.name}")
    if manifest.get("run_id") != run_dir.name:
        raise ValueError("run manifest run_id does not match its directory")
    if manifest.get("git_commit") != git_commit:
        raise ValueError("run manifest git_commit does not match shard")
    if type(manifest.get("seed")) is not int or manifest.get("seed") != group.seed:
        raise ValueError("run manifest seed does not match assigned group")
    resolved = manifest.get("config")
    if not isinstance(resolved, Mapping):
        raise ValueError("run manifest config must be a mapping")
    expected = {
        "model": group.training_model,
        "training_family": group.training_family,
        "reported_models": list(group.reported_models),
        "seed": group.seed,
        "protocol": group.protocol,
        "objective": group.objective,
        "condition_list": list(group.conditions),
        "resolved_device": device,
    }
    for field, expected_value in expected.items():
        actual = resolved.get(field)
        if field == "seed" and type(actual) is not int:
            raise ValueError("run manifest config seed has invalid type")
        try:
            matches = canonical_json(actual) == canonical_json(expected_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid run manifest config field: {field}") from error
        if not matches:
            raise ValueError(
                f"run manifest {field} does not match assigned group binding"
            )


def _root_run_ids(output_root: Path) -> list[str]:
    run_ids: list[str] = []
    for child in output_root.iterdir():
        if child.name in {"shard_execution.json", _LOCK_FILE}:
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
    assigned_groups: list[TrainingGroup],
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
        "group_runs",
    }
    expected_fields = {
        "started": common,
        "completed": common | {"completed_at"},
        "failed": common | {"error_type"},
    }.get(status)
    if expected_fields is None or set(value) != expected_fields:
        raise ValueError("invalid shard execution marker fields or status")
    for field in ("schema_version", "shard_index", "shard_count"):
        if type(value.get(field)) is not int or value.get(field) != immutable[field]:
            raise ValueError(f"immutable shard marker mismatch: {field}")
    for field, expected in immutable.items():
        if field in {"schema_version", "shard_index", "shard_count"}:
            continue
        try:
            matches = canonical_json(value.get(field)) == canonical_json(expected)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid immutable shard marker: {field}") from error
        if not matches:
            raise ValueError(f"immutable shard marker mismatch: {field}")
    _validate_created_at(value["started_at"])
    if status == "completed":
        _validate_created_at(value["completed_at"])
    if status == "failed" and (
        not isinstance(value["error_type"], str) or not value["error_type"]
    ):
        raise ValueError("invalid shard execution error_type")
    completed = value["completed_group_ids"]
    run_ids = value["run_ids"]
    group_runs = value["group_runs"]
    if not isinstance(group_runs, list) or len(group_runs) > len(assigned_groups):
        raise ValueError("invalid group_runs in shard execution marker")
    expected_completed: list[str] = []
    expected_run_ids: list[str] = []
    for index, binding in enumerate(group_runs):
        if not isinstance(binding, Mapping) or set(binding) != {"group_id", "run_ids"}:
            raise ValueError("invalid group binding fields in group_runs")
        group = assigned_groups[index]
        binding_run_ids = binding["run_ids"]
        if binding["group_id"] != group.group_id:
            raise ValueError("group_runs do not match assigned group order")
        if (
            not isinstance(binding_run_ids, list)
            or len(binding_run_ids) != 1
            or not isinstance(binding_run_ids[0], str)
            or not _RUN_ID.fullmatch(binding_run_ids[0])
        ):
            raise ValueError("each group binding requires exactly one run_id")
        expected_completed.append(group.group_id)
        expected_run_ids.append(binding_run_ids[0])
    if completed != expected_completed:
        raise ValueError("completed_group_ids do not match group_runs")
    if run_ids != expected_run_ids:
        raise ValueError("run_ids do not match group_runs")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("duplicate run_ids in shard execution marker")
    if status == "completed" and len(group_runs) != len(assigned_groups):
        raise ValueError("group_runs do not cover the completed shard")
    actual_run_ids = _root_run_ids(output_root)
    missing = set(run_ids) - set(actual_run_ids)
    if missing:
        raise ValueError("registered run_ids do not match shard run directories")
    for index, run_id in enumerate(run_ids):
        _validate_run_directory(
            output_root / run_id,
            assigned_groups[index],
            git_commit=str(immutable["git_commit"]),
            device=str(immutable["device"]),
        )
    unregistered = sorted(set(actual_run_ids) - set(run_ids))
    if unregistered:
        if status != "started" or len(unregistered) != 1:
            raise ValueError("unregistered run directories do not match marker")
        next_index = len(group_runs)
        if next_index >= len(assigned_groups):
            raise ValueError("unregistered run has no assigned group")
        _validate_run_directory(
            output_root / unregistered[0],
            assigned_groups[next_index],
            git_commit=str(immutable["git_commit"]),
            device=str(immutable["device"]),
        )
    return value


def _load_execution_marker(
    marker_path: Path,
    immutable: Mapping[str, Any],
    output_root: Path,
    assigned_groups: list[TrainingGroup],
) -> dict[str, Any]:
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("unable to load shard execution marker") from error
    return _validate_marker(marker, immutable, output_root, assigned_groups)


def _execute_shard_locked(
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
    groups = enumerate_training_groups(config)
    assigned_groups = [groups[index] for index in shard["group_indices"]]

    if marker_path.exists():
        marker = _load_execution_marker(
            marker_path, immutable, output_root, assigned_groups
        )
        if marker["status"] == "failed":
            raise ValueError("failed shard execution cannot be resumed")
        if marker["status"] == "completed":
            return marker
    else:
        existing = [child for child in output_root.iterdir() if child.name != _LOCK_FILE]
        if existing:
            names = {child.name for child in existing}
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
            "group_runs": [],
        }
        _atomic_write_json(marker_path, marker)

    completed_count = len(marker["group_runs"])
    try:
        for group in assigned_groups[completed_count:]:
            report = _run_group(
                config, group, repository_root, output_root, requested_device
            )
            if not isinstance(report, Mapping):
                raise ValueError("group execution report must be a mapping")
            new_run_ids = report.get("run_ids")
            if (
                not isinstance(new_run_ids, list)
                or len(new_run_ids) != 1
                or not isinstance(new_run_ids[0], str)
                or not _RUN_ID.fullmatch(new_run_ids[0])
            ):
                raise ValueError("group execution must return exactly one run_id")
            if set(new_run_ids) & set(marker["run_ids"]):
                raise ValueError("group execution returned duplicate run_ids")
            _validate_run_directory(
                output_root / new_run_ids[0],
                group,
                git_commit=git_commit,
                device=requested_device,
            )
            marker["group_runs"].append(
                {"group_id": group.group_id, "run_ids": list(new_run_ids)}
            )
            marker["completed_group_ids"].append(group.group_id)
            marker["run_ids"].extend(new_run_ids)
            if sorted(marker["run_ids"]) != _root_run_ids(output_root):
                raise ValueError("run_ids do not match shard run directories")
            _atomic_write_json(marker_path, marker)
    except Exception as error:
        marker["status"] = "failed"
        marker["error_type"] = type(error).__name__
        _atomic_write_json(marker_path, marker)
        raise

    marker["status"] = "completed"
    marker["completed_at"] = _timestamp()
    _atomic_write_json(marker_path, marker)
    return marker


def execute_shard(
    config: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    shard_index: int,
    repository_root: Union[Path, str],
    output_root: Union[Path, str],
    requested_device: str,
) -> Mapping[str, Any]:
    """Execute one shard while holding its process-wide output-root lock."""

    repository_path = Path(repository_root)
    output_path = Path(output_root)
    if not output_path.is_absolute():
        output_path = repository_path / output_path
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("shard output root must be a directory")
    with _shard_execution_lock(output_path):
        return _execute_shard_locked(
            config,
            plan=plan,
            shard_index=shard_index,
            repository_root=repository_path,
            output_root=output_path,
            requested_device=requested_device,
        )


def _file_sha256(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"unable to hash artifact: {path}") from error
    return digest.hexdigest()


def _is_linked_source(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    except OSError as error:
        raise ValueError(f"unable to inspect source path: {path}") from error
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _resolve_contained_source(
    path: Path, *, container: Path, label: str
) -> Path:
    if _is_linked_source(path):
        raise ValueError(f"linked or symlink {label} is not allowed: {path}")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"unable to resolve {label}: {path}") from error
    if resolved != container and container not in resolved.parents:
        raise ValueError(f"{label} escapes its resolved shard containment: {path}")
    return resolved


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to load {label}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(loaded)


def _validate_shard_assets(
    shard_root: Path,
    run_ids: list[str],
    resolved_entries: Mapping[str, Path],
) -> list[dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for child in shard_root.iterdir():
        split_match = _SPLIT_ASSET.fullmatch(child.name)
        scaler_match = _SCALER_ASSET.fullmatch(child.name)
        if not (split_match or scaler_match):
            continue
        if not child.is_file():
            raise ValueError(f"asset is not a file: {child.name}")
        expected = (split_match or scaler_match).group(1)  # type: ignore[union-attr]
        source = resolved_entries[child.name]
        actual = _file_sha256(source)
        if actual != expected:
            raise ValueError(f"asset filename digest does not match SHA-256: {child.name}")
        assets[child.name] = {
            "name": child.name,
            "source": source,
            "sha256": actual,
        }

    referenced: set[str] = set()
    for run_id in run_ids:
        manifest = _load_json_mapping(
            shard_root / run_id / "run.json", f"run manifest in {run_id}"
        )
        split_hash = manifest.get("split_hash")
        scaler_hash = manifest.get("scaler_hash")
        for field, value in (("split_hash", split_hash), ("scaler_hash", scaler_hash)):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"run manifest {field} must be 64 lowercase hex")
        referenced.add(f"split_manifest-{split_hash}.csv")
        referenced.add(f"scaler-{scaler_hash}.json")
    if set(assets) != referenced:
        missing = sorted(referenced - set(assets))
        extra = sorted(set(assets) - referenced)
        raise ValueError(f"asset set does not match run manifests; missing={missing}, extra={extra}")
    return [assets[name] for name in sorted(assets)]


def preflight_shards(
    config: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    shards_root: Union[Path, str],
) -> Mapping[str, Any]:
    """Read and strictly validate every completed shard without writing files."""

    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    if not isinstance(plan, Mapping):
        raise ValueError("shard plan must be a mapping")
    validated_plan = _validate_plan_mapping(
        plan,
        config=config,
        git_commit=plan.get("git_commit"),
        device=plan.get("device"),
    )
    root = Path(shards_root)
    if os.path.lexists(root) and _is_linked_source(root):
        raise ValueError("linked or symlink shards_root is not allowed")
    if not root.is_dir():
        raise ValueError("shards_root must be a directory")
    try:
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("unable to resolve shards_root") from error
    expected_names = {
        f"{index:03d}" for index in range(validated_plan["shard_count"])
    }
    actual_names = {child.name for child in root.iterdir()}
    if actual_names != expected_names or any(
        not (root / name).is_dir() for name in actual_names
    ):
        missing = sorted(expected_names - actual_names)
        foreign = sorted(actual_names - expected_names)
        raise ValueError(
            f"shard directories do not match plan; missing={missing}, foreign={foreign}"
        )

    groups = enumerate_training_groups(config)
    run_sources: list[dict[str, Any]] = []
    asset_sources: list[dict[str, Any]] = []
    run_ids: list[str] = []
    group_ids: list[str] = []
    combination_ids: list[str] = []
    assets_by_name: dict[str, tuple[str, bytes]] = {}
    source_paths: list[Path] = []

    for shard_index, shard in enumerate(validated_plan["shards"]):
        shard_root = root / f"{shard_index:03d}"
        resolved_shard_root = _resolve_contained_source(
            shard_root,
            container=resolved_root,
            label=f"shard directory {shard_root.name}",
        )
        if not resolved_shard_root.is_dir():
            raise ValueError(f"shard source is not a directory: {shard_root.name}")
        source_paths.append(resolved_shard_root)
        resolved_entries: dict[str, Path] = {}
        resolved_run_entries: dict[str, dict[str, Path]] = {}
        for child in shard_root.iterdir():
            resolved_child = _resolve_contained_source(
                child,
                container=resolved_shard_root,
                label=f"shard source {child.name}",
            )
            resolved_entries[child.name] = resolved_child
            source_paths.append(resolved_child)
            if _RUN_ID.fullmatch(child.name) and resolved_child.is_dir():
                run_items: dict[str, Path] = {}
                for item in child.iterdir():
                    resolved_item = _resolve_contained_source(
                        item,
                        container=resolved_child,
                        label=f"run artifact {child.name}/{item.name}",
                    )
                    run_items[item.name] = resolved_item
                    source_paths.append(resolved_item)
                resolved_run_entries[child.name] = run_items
        assigned_groups = [groups[index] for index in shard["group_indices"]]
        marker_path = shard_root / "shard_execution.json"
        if not marker_path.is_file():
            raise ValueError(f"missing completed shard_execution.json in {shard_root.name}")
        immutable = _immutable_marker_fields(validated_plan, shard, shard_index)
        marker = _load_execution_marker(
            resolved_entries[marker_path.name], immutable, shard_root, assigned_groups
        )
        if marker["status"] != "completed":
            raise ValueError(
                f"shard {shard_root.name} status must be completed, got {marker['status']}"
            )
        marker_run_ids = list(marker["run_ids"])
        allowed = {"shard_execution.json", _LOCK_FILE, *marker_run_ids}
        shard_assets = _validate_shard_assets(
            shard_root, marker_run_ids, resolved_entries
        )
        allowed.update(asset["name"] for asset in shard_assets)
        actual = {child.name for child in shard_root.iterdir()}
        if actual - allowed or allowed - actual - {_LOCK_FILE}:
            raise ValueError(
                f"foreign or missing content in shard {shard_root.name}: "
                f"foreign={sorted(actual - allowed)}, missing={sorted(allowed - actual - {_LOCK_FILE})}"
            )
        for run_id in marker_run_ids:
            run_dir = shard_root / run_id
            resolved_run_dir = resolved_entries[run_id]
            run_entries = list(run_dir.iterdir()) if run_dir.is_dir() else []
            if not run_dir.is_dir() or {item.name for item in run_entries} != _RUN_FILES:
                raise ValueError(f"run {run_id} must contain exactly six artifacts")
            for item_name, resolved_item in resolved_run_entries[run_id].items():
                if not resolved_item.is_file():
                    raise ValueError(
                        f"run artifact is not a regular file: {run_id}/{item_name}"
                    )
            if run_id in run_ids:
                raise ValueError(f"duplicate run_id across shards: {run_id}")
            run_ids.append(run_id)
            run_sources.append({"run_id": run_id, "source": resolved_run_dir})
        for asset in shard_assets:
            source = asset["source"]
            content = source.read_bytes()
            prior = assets_by_name.get(asset["name"])
            if prior is not None and prior != (asset["sha256"], content):
                raise ValueError(f"same-name asset conflict across shards: {asset['name']}")
            assets_by_name[asset["name"]] = (asset["sha256"], content)
            asset_sources.append(asset)
        group_ids.extend(marker["group_ids"])
        combination_ids.extend(marker["combination_ids"])

    expected_group_ids = [
        identifier
        for shard in validated_plan["shards"]
        for identifier in shard["group_ids"]
    ]
    expected_combination_ids = [
        identifier
        for shard in validated_plan["shards"]
        for identifier in shard["combination_ids"]
    ]
    if group_ids != expected_group_ids or len(set(group_ids)) != len(group_ids):
        raise ValueError("shard group IDs are not disjoint and exhaustive")
    if (
        combination_ids != expected_combination_ids
        or len(set(combination_ids)) != len(combination_ids)
    ):
        raise ValueError("shard combination IDs are not disjoint and exhaustive")
    if len(group_ids) != validated_plan["total_groups"]:
        raise ValueError("shard group coverage does not match total_groups")
    if len(combination_ids) != validated_plan["total_cells"]:
        raise ValueError("shard combination coverage does not match total_cells")
    ordered_combination_ids = [
        cell["combination_id"] for cell in enumerate_matrix(config)
    ]
    if set(ordered_combination_ids) != set(combination_ids):
        raise ValueError("shard combination IDs do not match the config matrix")

    return {
        "plan_sha256": validated_plan["plan_sha256"],
        "source_config_sha256": validated_plan["source_config_sha256"],
        "git_commit": validated_plan["git_commit"],
        "device": validated_plan["device"],
        "total_groups": validated_plan["total_groups"],
        "total_cells": validated_plan["total_cells"],
        "selected_combination_ids": ordered_combination_ids,
        "run_ids": sorted(run_ids),
        "run_sources": run_sources,
        "asset_sources": asset_sources,
        "asset_digests": {
            name: digest for name, (digest, _) in sorted(assets_by_name.items())
        },
        "source_paths": source_paths,
    }


@contextmanager
def _merge_publish_lock(parent: Path):
    lock_path = parent / _MERGE_LOCK_FILE
    if os.path.lexists(lock_path):
        if _is_linked_source(lock_path):
            raise ValueError("linked or symlink merge lock is not allowed")
        if not stat.S_ISREG(os.lstat(lock_path).st_mode):
            raise ValueError("merge lock must be a regular file")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        if os.path.lexists(lock_path) and _is_linked_source(lock_path):
            raise ValueError("linked or symlink merge lock is not allowed") from error
        raise
    locked = False
    primary_error: BaseException | None = None
    try:
        path_status = os.lstat(lock_path)
        descriptor_status = os.fstat(descriptor)
        if (
            _is_linked_source(lock_path)
            or not stat.S_ISREG(path_status.st_mode)
            or not stat.S_ISREG(descriptor_status.st_mode)
            or (path_status.st_dev, path_status.st_ino)
            != (descriptor_status.st_dev, descriptor_status.st_ino)
        ):
            raise ValueError("merge lock path changed or is not a regular file")
        if descriptor_status.st_size == 0:
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise ValueError("merge parent is active or locked") from error
        yield
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_error: BaseException | None = None
        if locked:
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            except BaseException as error:
                cleanup_error = error
        try:
            os.close(descriptor)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if primary_error is None and cleanup_error is not None:
            raise cleanup_error


def _copy_verified_file(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    shutil.copy2(source, destination)
    if _file_sha256(source) != _file_sha256(destination):
        raise ValueError(f"copied artifact SHA-256 mismatch: {source}")


def _preserve_failed_merge(temporary: Path, parent: Path) -> None:
    if not temporary.exists():
        return
    failed = parent / f".failed-merge-{uuid.uuid4().hex}"
    os.replace(temporary, failed)


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _absolute_unresolved(path: Union[Path, str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_linked_components(path: Path, *, label: str) -> None:
    components: list[Path] = []
    current = path
    while True:
        components.append(current)
        if current == current.parent:
            break
        current = current.parent
    for component in reversed(components):
        if os.path.lexists(component) and _is_linked_source(component):
            raise ValueError(f"linked {label} component is not allowed: {component}")


def _destination_exists(raw_output: Path, resolved_output: Path) -> bool:
    return os.path.lexists(raw_output) or os.path.lexists(resolved_output)


def _rename_noreplace(source: Union[Path, str], destination: Union[Path, str]) -> None:
    source_path = os.fspath(source)
    destination_path = os.fspath(destination)
    if os.name == "nt":
        os.rename(source_path, destination_path)
        return
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = library.renameat2
        except AttributeError as error:
            raise OSError(errno.ENOSYS, "libc renameat2 is unavailable") from error
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = renameat2(
            -100,
            os.fsencode(source_path),
            -100,
            os.fsencode(destination_path),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error_number, os.strerror(error_number), destination_path
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            f"{source_path} -> {destination_path}",
        )
    raise OSError(
        errno.ENOTSUP,
        f"atomic no-replace rename is unsupported on {sys.platform}",
    )


def merge_shards(
    *,
    config_path: Union[Path, str],
    plan_path: Union[Path, str],
    shards_root: Union[Path, str],
    output_root: Union[Path, str],
) -> Mapping[str, Any]:
    """Validate, copy, seal, and atomically publish completed shards."""

    raw_output = _absolute_unresolved(output_root)
    if os.path.lexists(raw_output):
        raise ValueError("raw output_root already exists or is linked")
    raw_parent = raw_output.parent
    _reject_linked_components(raw_parent, label="output parent")

    try:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("unable to load merge config") from error
    if not isinstance(config, Mapping):
        raise ValueError("merge config must be a mapping")
    try:
        plan = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("unable to load shard plan") from error
    if not isinstance(plan, Mapping):
        raise ValueError("shard plan must be a mapping")

    promotion = preflight_shards(config, plan=plan, shards_root=shards_root)
    _reject_linked_components(raw_parent, label="output parent")
    resolved_parent_candidate = raw_parent.resolve(strict=False)
    output = resolved_parent_candidate / raw_output.name
    source_root = Path(shards_root).absolute().resolve(strict=False)
    if output == source_root or source_root in output.parents:
        raise ValueError("output_root must not overlap shards_root")
    for source in promotion["source_paths"]:
        resolved_source = Path(source)
        if _paths_overlap(output, resolved_source) or _paths_overlap(
            output, resolved_source.parent
        ):
            raise ValueError("output_root must not overlap resolved shard sources")
    raw_parent.mkdir(parents=True, exist_ok=True)
    _reject_linked_components(raw_parent, label="output parent")
    try:
        parent = raw_parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("unable to resolve output parent") from error
    if parent != resolved_parent_candidate:
        raise ValueError("output parent changed while preparing merge")
    output = parent / raw_output.name

    with _merge_publish_lock(parent):
        _reject_linked_components(raw_parent, label="output parent")
        if raw_parent.resolve(strict=True) != parent:
            raise ValueError("output parent changed while acquiring merge lock")
        if _destination_exists(raw_output, output):
            raise ValueError("output_root already exists")
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output.name}-merge-", dir=parent)
        )
        try:
            copied_assets: set[str] = set()
            for asset in promotion["asset_sources"]:
                name = asset["name"]
                if name in copied_assets:
                    continue
                _copy_verified_file(asset["source"], temporary / name)
                copied_assets.add(name)
            for run in promotion["run_sources"]:
                destination = temporary / run["run_id"]
                shutil.copytree(
                    run["source"], destination, copy_function=_copy_verified_file
                )
            _atomic_write_json(
                temporary / "matrix_execution.json",
                {
                    "status": "completed",
                    "partial": False,
                    "selected_cells": promotion["total_cells"],
                    "total_cells": promotion["total_cells"],
                    "training_groups": promotion["total_groups"],
                    "grouping_key": [
                        "training_family",
                        "seed",
                        "protocol",
                        "objective",
                    ],
                    "selected_combination_ids": promotion[
                        "selected_combination_ids"
                    ],
                    "run_ids": promotion["run_ids"],
                },
            )
            report = validate_artifacts(temporary, config=config_path)
            if (
                not isinstance(report, Mapping)
                or report.get("status") != "complete"
                or not (temporary / "validation_report.json").is_file()
            ):
                raise ValueError("artifact validator did not complete and seal report")
            _reject_linked_components(raw_parent, label="output parent")
            if raw_parent.resolve(strict=True) != parent:
                raise ValueError("output parent changed during merge")
            if _destination_exists(raw_output, output):
                raise ValueError("output_root appeared during merge")
            _rename_noreplace(temporary, raw_output)
            return dict(report)
        except BaseException:
            try:
                _preserve_failed_merge(temporary, parent)
            except OSError:
                pass
            raise


__all__ = [
    "SHARD_SCHEMA_VERSION",
    "build_shard_plan",
    "execute_shard",
    "load_shard_plan",
    "merge_shards",
    "preflight_shards",
    "write_shard_plan",
]
