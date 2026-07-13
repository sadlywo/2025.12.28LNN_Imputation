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
import re
import subprocess
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
MANIFEST_FIELDS = frozenset(
    {
        "run_id",
        "seed",
        "config",
        "config_sha256",
        "split_hash",
        "scaler_hash",
        "git_commit",
        "dirty_state_digest",
        "package_versions",
        "python",
        "platform",
    }
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


def runtime_fingerprint() -> dict[str, Any]:
    """Return the canonical package, Python, and platform runtime identity."""

    versions: dict[str, str] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return {
        "package_versions": versions,
        "python": sys.version.split()[0],
        "platform": platform_module.platform(),
    }


def git_worktree_identity(repository_root: Path | str) -> dict[str, str]:
    """Return HEAD and the tracked-worktree digest used by experiment manifests."""

    root = Path(repository_root)

    def git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("unable to determine git worktree identity") from error
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    if not commit:
        raise ValueError("unable to determine git worktree identity")
    dirty_text = git("status", "--porcelain=v1", "--untracked-files=no")
    return {
        "git_commit": commit,
        "dirty_state_digest": (
            hashlib.sha256(dirty_text.encode("utf-8")).hexdigest()
            if dirty_text
            else ""
        ),
    }


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
    runtime = runtime_fingerprint()
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
        **runtime,
    }


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    missing = sorted(MANIFEST_FIELDS - manifest.keys())
    if missing:
        raise ValueError("missing manifest fields: " + ", ".join(missing))

    if type(manifest["seed"]) is not int:
        raise ValueError("manifest seed must be an integer")
    string_fields = (
        "run_id",
        "config_sha256",
        "split_hash",
        "scaler_hash",
        "git_commit",
        "dirty_state_digest",
        "python",
        "platform",
    )
    if any(not isinstance(manifest[field], str) for field in string_fields):
        raise ValueError("manifest provenance identifiers must be strings")
    for field in ("split_hash", "scaler_hash", "dirty_state_digest"):
        digest = manifest[field]
        if digest and re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{field} must be empty or 64 lowercase hex")
    versions = manifest["package_versions"]
    if not isinstance(versions, Mapping) or any(
        not isinstance(name, str) or not isinstance(version, str)
        for name, version in versions.items()
    ):
        raise ValueError("package_versions must map strings to strings")

    config_json = canonical_json(manifest["config"])
    expected_config_sha256 = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    if manifest["config_sha256"] != expected_config_sha256:
        raise ValueError("config_sha256 does not match resolved config")
    expected_run_id = run_id(
        manifest["config"],
        manifest["seed"],
        manifest["split_hash"],
        manifest["scaler_hash"],
        manifest["git_commit"],
        manifest["dirty_state_digest"],
    )
    if manifest["run_id"] != expected_run_id:
        raise ValueError("run_id does not match provenance content (different content)")


def write_run_manifest(run_dir: Path | str, manifest: Mapping[str, Any]) -> Path:
    """Validate and atomically seal a manifest without replacing an existing file."""

    _validate_manifest(manifest)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    content = (canonical_json(manifest) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"run_id {manifest.get('run_id')} already has different content")
        return path

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=run_dir, prefix=".run-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        while True:
            try:
                os.link(temporary, path)
                break
            except FileExistsError:
                try:
                    existing = path.read_bytes()
                except FileNotFoundError:
                    continue
                if existing == content:
                    return path
                raise ValueError(
                    f"run_id {manifest.get('run_id')} already has different content"
                )
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path
