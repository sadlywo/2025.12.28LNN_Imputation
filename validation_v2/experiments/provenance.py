"""Immutable experiment provenance."""

from __future__ import annotations

import dataclasses
import hashlib
from importlib import metadata
import json
import math
import os
import platform as platform_module
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping


PACKAGE_DISTRIBUTIONS = (
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "PyYAML",
    "pytest",
    "ncps",
)


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("provenance values must not contain NaN or Infinity")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("provenance mapping keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"unsupported provenance value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return strict, stable JSON for supported provenance values."""

    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def run_id(
    config: Any,
    seed: int,
    split_hash: str = "",
    scaler_hash: str = "",
    git_commit: str = "",
    dirty_digest: str = "",
) -> str:
    """Derive a short content identifier without wall-clock state."""

    payload = {
        "config": _json_value(config),
        "seed": seed,
        "split_hash": split_hash,
        "scaler_hash": scaler_hash,
        "git_commit": git_commit,
        "dirty_state_digest": dirty_digest,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def collect_provenance(
    config: Any,
    seed: int,
    split_hash: str = "",
    scaler_hash: str = "",
    git_commit: str = "",
    dirty_digest: str = "",
) -> dict[str, Any]:
    """Collect the resolved, content-addressed inputs needed to reproduce a run."""

    resolved_config = _json_value(config)
    config_json = canonical_json(resolved_config)
    versions: dict[str, str] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return {
        "run_id": run_id(
            resolved_config,
            seed,
            split_hash,
            scaler_hash,
            git_commit,
            dirty_digest,
        ),
        "seed": int(seed),
        "config": resolved_config,
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "split_hash": split_hash,
        "scaler_hash": scaler_hash,
        "git_commit": git_commit,
        "dirty_state_digest": dirty_digest,
        "package_versions": versions,
        "python": sys.version.split()[0],
        "platform": platform_module.platform(),
    }


def write_run_manifest(run_dir: Path | str, manifest: Mapping[str, Any]) -> Path:
    """Atomically create an immutable run manifest; identical writes are idempotent."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    content = (canonical_json(manifest) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"run_id {manifest.get('run_id')} already has different content")
        return path

    expected_run_id = run_id(
        manifest["config"],
        int(manifest["seed"]),
        str(manifest.get("split_hash", "")),
        str(manifest.get("scaler_hash", "")),
        str(manifest.get("git_commit", "")),
        str(manifest.get("dirty_state_digest", "")),
    )
    if manifest.get("run_id") != expected_run_id:
        raise ValueError("run_id does not match provenance content")

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=run_dir, prefix=".run-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path
