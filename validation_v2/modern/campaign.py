from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from .artifacts import canonical_json
from .config import ModernConfig


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _write_once(path: Path, value: object) -> None:
    content = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content); handle.flush(); os.fsync(handle.fileno())


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"state marker must be an object: {path}")
    return value


def build_tasks(
    config: ModernConfig,
    dataset_manifest: Mapping[str, object],
    selection_lock: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    selected = selection_lock.get("selected", {})
    if not isinstance(selected, Mapping):
        raise ValueError("selection lock is missing selected configurations")
    dataset_id = dataset_manifest.get("dataset_id")
    if not isinstance(dataset_id, str):
        raise ValueError("dataset manifest is missing dataset_id")
    conditions = [
        {"topology": topology, "requested_fraction": rate}
        for topology in config.topologies for rate in config.rates
    ]
    if config.irregular_cases:
        conditions.append({"topology": "irregular:interval_jitter+point", "requested_fraction": 0.3})
    tasks = []
    for seed in config.seeds:
        for model in config.models:
            chosen = selected.get(model, {}) if model in selected else {}
            configuration_id = chosen.get("configuration_id", f"reference:{model}") if isinstance(chosen, Mapping) else f"reference:{model}"
            identity = {
                "phase": "formal_training", "model": model, "seed": seed,
                "configuration_id": configuration_id, "dataset_artifact_id": dataset_id,
                "checkpoint_input_hash": None,
                "sampling_count": config.n_sampling_times if model in {"csdi", "sssd"} else 1,
                "condition_list": conditions,
            }
            tasks.append({"task_id": _digest(identity), **identity})
    return tuple(tasks)


def claim_task(root: Path, task: Mapping[str, object]) -> Path:
    task_id = task.get("task_id")
    if not isinstance(task_id, str) or len(task_id) != 64:
        raise ValueError("task_id must be a SHA-256 string")
    directory = Path(root) / task_id
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("claimed.json", "running.json", "completed.json")):
        raise FileExistsError(f"task is already claimed: {task_id}")
    _write_once(directory / "claimed.json", {"schema_version": 1, "task": dict(task), "task_digest": _digest(dict(task))})
    return directory


def run_task(task_dir: Path, command: Sequence[str], *, environment: Mapping[str, str]) -> None:
    task_dir = Path(task_dir)
    claimed = _read(task_dir / "claimed.json")
    _write_once(task_dir / "running.json", {"schema_version": 1, "task_digest": claimed["task_digest"], "command": list(command)})
    stdout_path, stderr_path = task_dir / "stdout.log", task_dir / "stderr.log"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        result = subprocess.run(list(command), cwd=task_dir, env=dict(environment), stdout=stdout, stderr=stderr, check=False)
    if result.returncode:
        _write_once(task_dir / "failed.json", {"schema_version": 1, "returncode": result.returncode, "stdout": stdout_path.name, "stderr": stderr_path.name})
        raise RuntimeError(f"task subprocess failed with return code {result.returncode}")


def complete_task(root: Path, task: Mapping[str, object], outputs: Mapping[str, object]) -> None:
    directory = Path(root) / str(task["task_id"])
    claimed = _read(directory / "claimed.json")
    if claimed.get("task_digest") != _digest(dict(task)):
        raise ValueError("claimed task does not match completion task")
    payload = {"schema_version": 1, "task_digest": claimed["task_digest"], "outputs": dict(outputs)}
    _write_once(directory / "completed.json", {**payload, "completion_hash": _digest(payload)})


def pending_tasks(root: Path, tasks: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    pending = []
    for source in tasks:
        task = dict(source); directory = Path(root) / str(task.get("task_id")); marker = directory / "completed.json"
        if not marker.exists():
            pending.append(task); continue
        try:
            completed = _read(marker)
            payload = {key: value for key, value in completed.items() if key != "completion_hash"}
            valid = (
                set(completed) == {"schema_version", "task_digest", "outputs", "completion_hash"}
                and completed["schema_version"] == 1
                and completed["task_digest"] == _digest(task)
                and completed["completion_hash"] == _digest(payload)
                and isinstance(completed["outputs"], dict)
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            valid = False
        if not valid:
            raise ValueError(f"inconsistent completed task: {task.get('task_id')}")
    return tuple(pending)


__all__ = ["build_tasks", "claim_task", "complete_task", "pending_tasks", "run_task"]
