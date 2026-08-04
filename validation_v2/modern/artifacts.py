from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

_SCHEMA_VERSION = 1
_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "artifact_id",
    "npz_sha256",
    "arrays",
    "metadata",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _artifact_id(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _temporary_path(parent: Path, prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(dir=parent, prefix=prefix, suffix=".tmp")
    os.close(descriptor)
    return Path(name)


def _publish_no_clobber(temporary: Path, destination: Path) -> None:
    os.link(temporary, destination)


def write_array_artifact(
    base: Path,
    kind: str,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
) -> dict[str, object]:
    base = Path(base)
    npz_path = base.with_suffix(".npz")
    json_path = base.with_suffix(".json")
    base.parent.mkdir(parents=True, exist_ok=True)
    if npz_path.exists():
        raise FileExistsError(npz_path)
    if json_path.exists():
        raise FileExistsError(json_path)
    if not kind:
        raise ValueError("artifact kind must not be empty")
    if not arrays:
        raise ValueError("artifact arrays must not be empty")

    normalized: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise ValueError("array names must be non-empty strings")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"object array is not supported: {name}")
        normalized[name] = array

    npz_temporary = _temporary_path(base.parent, f".{base.name}-npz-")
    json_temporary: Path | None = None
    published_npz = False
    try:
        with npz_temporary.open("wb") as handle:
            np.savez_compressed(handle, **normalized)
            handle.flush()
            os.fsync(handle.fileno())

        payload: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "kind": kind,
            "npz_sha256": sha256_file(npz_temporary),
            "arrays": {
                name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                for name, array in sorted(normalized.items())
            },
            "metadata": dict(metadata),
        }
        manifest = {**payload, "artifact_id": _artifact_id(payload)}
        content = (canonical_json(manifest) + "\n").encode("utf-8")
        json_temporary = _temporary_path(base.parent, f".{base.name}-json-")
        with json_temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        _publish_no_clobber(npz_temporary, npz_path)
        published_npz = True
        try:
            _publish_no_clobber(json_temporary, json_path)
        except BaseException:
            if published_npz and npz_path.exists() and os.path.samefile(
                npz_temporary, npz_path
            ):
                npz_path.unlink()
            raise
        return manifest
    finally:
        if npz_temporary.exists():
            npz_temporary.unlink()
        if json_temporary is not None and json_temporary.exists():
            json_temporary.unlink()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_array_artifact(
    base: Path, *, expected_kind: str
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    base = Path(base)
    npz_path = base.with_suffix(".npz")
    json_path = base.with_suffix(".json")
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid artifact manifest: {json_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("artifact manifest must be a JSON object")
    if set(manifest) != _MANIFEST_KEYS:
        raise ValueError("artifact manifest has unknown or missing top-level keys")
    if manifest["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("unsupported artifact schema version")
    if manifest["kind"] != expected_kind:
        raise ValueError("artifact kind mismatch")

    payload = {key: value for key, value in manifest.items() if key != "artifact_id"}
    if manifest["artifact_id"] != _artifact_id(payload):
        raise ValueError("artifact ID mismatch")
    if sha256_file(npz_path) != manifest["npz_sha256"]:
        raise ValueError("NPZ hash mismatch")

    declared = manifest["arrays"]
    if not isinstance(declared, dict) or not declared:
        raise ValueError("artifact array manifest must be a non-empty object")
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError("duplicate arrays in NPZ artifact")
            if set(archive.files) != set(declared):
                raise ValueError("artifact array set mismatch")
            for name in archive.files:
                specification = declared[name]
                if not isinstance(specification, dict) or set(specification) != {
                    "shape",
                    "dtype",
                }:
                    raise ValueError(f"invalid array specification: {name}")
                array = archive[name]
                if list(array.shape) != specification["shape"]:
                    raise ValueError(f"array shape mismatch: {name}")
                if str(array.dtype) != specification["dtype"]:
                    raise ValueError(f"array dtype mismatch: {name}")
                arrays[name] = np.array(array, copy=True)
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "mismatch" in str(error):
            raise
        raise ValueError(f"invalid NPZ artifact: {npz_path}") from error

    metadata = manifest["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("artifact metadata must be an object")
    return arrays, manifest


__all__ = [
    "canonical_json",
    "read_array_artifact",
    "sha256_file",
    "write_array_artifact",
]
