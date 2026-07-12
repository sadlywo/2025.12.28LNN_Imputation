"""Deterministic, content-addressed plans for sharded validation runs."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Union

import yaml

from .groups import TrainingGroup, enumerate_training_groups
from .provenance import canonical_json


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
    if path.exists():
        if path.read_bytes() == content:
            return path
        raise ValueError("shard plan already exists with different content")

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

    plan = dict(loaded)
    _validate_plan_structure(plan)
    if plan["plan_sha256"] != _plan_hash(plan):
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


__all__ = [
    "SHARD_SCHEMA_VERSION",
    "build_shard_plan",
    "load_shard_plan",
    "write_shard_plan",
]
