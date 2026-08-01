"""Read-only, fail-closed validation for completed imputation-v3 artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from imputation_v3.experiments.runner import (
    FORMAL_SEEDS,
    success_gate_payload,
)
from validation_v2.data.splits import MANIFEST_COLUMNS
from validation_v2.evaluation.statistics import (
    GROUP_COLUMNS,
    PER_RECORD_COLUMNS,
    SUMMARY_COLUMNS,
    validate_per_record_metrics,
)
from validation_v2.experiments.provenance import (
    MANIFEST_FIELDS,
    _validate_manifest,
    canonical_json,
)
from validation_v2.experiments.train import select_best_checkpoint


_RUN_FILES = frozenset({"run.json", "history.json", "best.pt", "checkpoint.json"})
_CHECKPOINT_FIELDS = frozenset(
    {"run_id", "best_epoch", "selection_split", "selection_metric", "checkpoint_sha256"}
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_WINDOW_ID = re.compile(r"teacher-window-sha256-[0-9a-f]{64}")
_FORMAL_HASHED_FILES = frozenset(
    {
        "per_record_metrics.csv",
        "summary.csv",
        "success_gate.json",
        "mask_ledger.csv",
        "coverage_ledger.csv",
    }
)
_FORMAL_FIXED_FILES = _FORMAL_HASHED_FILES | frozenset(
    {"artifact_hashes.json", "frozen_models.json"}
)
_MASK_COLUMNS = (
    "seed",
    "recording_id",
    "topology",
    "requested_fraction",
    "realized_fraction",
    "mask_sha256",
)
_COVERAGE_COLUMNS = (
    *GROUP_COLUMNS,
    "included",
    "reason",
    "present_seeds",
    "required_seeds",
    "present_models",
    "required_models",
    "common_recordings",
    "union_recordings",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot read artifact: {path}") from exc
    return digest.hexdigest()


def _strict_canonical_json(path: Path) -> Any:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"missing or unreadable artifact: {path.name}") from exc

    def reject_constant(value: str) -> None:
        raise ValueError(f"{path.name} contains non-finite JSON constant {value}")

    try:
        value = json.loads(content.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} must contain legal UTF-8 JSON") from exc
    expected = (canonical_json(value) + "\n").encode("utf-8")
    if content != expected:
        raise ValueError(f"{path.name} must use canonical JSON encoding")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _safe_directory(path: Path) -> Path:
    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError("artifact root must not be a symlink")
    try:
        root = supplied.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"artifact output does not exist: {supplied}") from exc
    if not root.is_dir():
        raise ValueError("artifact output must be a directory")
    return root


def _safe_child(root: Path, child: Path) -> None:
    if child.is_symlink():
        raise ValueError(f"artifact path must not be a symlink: {child.name}")
    try:
        resolved = child.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"missing artifact: {child.name}") from exc
    if resolved != root / child.name:
        raise ValueError(f"artifact path escapes its run directory: {child.name}")


def _safe_descendant(root: Path, supplied: Path, *, area: Path | None = None) -> Path:
    """Resolve a referenced artifact while rejecting links and path escapes."""
    candidate = supplied if supplied.is_absolute() else root / supplied
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"missing referenced artifact: {supplied}") from exc
    boundary = root if area is None else area.resolve(strict=True)
    if not resolved.is_relative_to(boundary) or not resolved.is_file():
        raise ValueError(f"referenced artifact escapes its allowed directory: {supplied}")
    current = candidate
    while True:
        if current.is_symlink():
            raise ValueError(f"referenced artifact must not traverse a symlink: {supplied}")
        if current.resolve(strict=True) == root:
            break
        if current.parent == current:
            raise ValueError(f"referenced artifact escapes its allowed directory: {supplied}")
        current = current.parent
    return resolved


def _read_csv(path: Path, columns: tuple[str, ...] | list[str], name: str) -> pd.DataFrame:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"missing or unreadable artifact: {path.name}") from exc
    if not content or b"\r" in content or not content.endswith(b"\n"):
        raise ValueError(f"{name} must be canonical UTF-8 CSV with LF line endings")
    try:
        content.decode("utf-8")
        frame = pd.read_csv(path)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"{name} must contain legal UTF-8 CSV") from exc
    if list(frame.columns) != list(columns):
        raise ValueError(f"{name} does not use the exact frozen schema")
    if frame.empty:
        raise ValueError(f"{name} must not be empty")
    return frame


def _finite_frame(frame: pd.DataFrame, name: str) -> None:
    if frame.isna().any().any():
        raise ValueError(f"{name} must not contain missing values")
    for column in frame.select_dtypes(include=[np.number]).columns:
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} numeric values must be finite")


def _string_list(value: Any, name: str, *, pattern: re.Pattern[str] | None = None) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} entries must be non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} entries must be unique")
    if pattern is not None and any(pattern.fullmatch(item) is None for item in value):
        raise ValueError(f"{name} contains a malformed identity")
    return list(value)


def _validate_smoke_config(config: Any, manifest_seed: int) -> tuple[list[str], list[str]]:
    config = _mapping(config, "run config")
    if (
        config.get("mode") != "imputation_v3_teacher_smoke"
        or config.get("selection_split") != "validation"
        or config.get("test_evaluation") is not False
        or config.get("model") != "teacher"
        or config.get("seed") != manifest_seed
    ):
        raise ValueError("smoke run config does not match the frozen teacher contract")
    split_counts = _mapping(config.get("split_counts"), "split_counts")
    if set(split_counts) != {"train", "validation", "test"} or any(
        type(value) is not int or value <= 0 for value in split_counts.values()
    ):
        raise ValueError("split_counts must contain positive train/validation/test counts")
    recording_ids = _mapping(config.get("selected_recording_ids"), "selected_recording_ids")
    if set(recording_ids) != {"train", "validation"}:
        raise ValueError("selected_recording_ids must contain train and validation")
    train_ids = _string_list(recording_ids["train"], "selected train recording IDs")
    validation_ids = _string_list(
        recording_ids["validation"], "selected validation recording IDs"
    )
    if set(train_ids) & set(validation_ids):
        raise ValueError("selected train and validation recording IDs must be disjoint")
    scaler_ids = _string_list(config.get("scaler_training_ids"), "scaler_training_ids")
    if scaler_ids != sorted(train_ids):
        raise ValueError("scaler_training_ids must exactly match selected training IDs")
    windows = _mapping(config.get("selected_window_ids"), "selected_window_ids")
    if set(windows) != {"train", "validation"}:
        raise ValueError("selected_window_ids must contain train and validation")
    train_windows = _string_list(windows["train"], "train window IDs", pattern=_WINDOW_ID)
    validation_windows = _string_list(
        windows["validation"], "validation window IDs", pattern=_WINDOW_ID
    )
    if set(train_windows) & set(validation_windows):
        raise ValueError("train and validation window identities must be disjoint")
    return train_windows, validation_windows


def _validate_smoke_run(run_dir: Path) -> dict[str, Any]:
    if run_dir.is_symlink():
        raise ValueError("smoke run directory must not be a symlink")
    run_dir = run_dir.resolve(strict=True)
    entries = {item.name for item in run_dir.iterdir()}
    if entries != _RUN_FILES:
        missing = sorted(_RUN_FILES - entries)
        extra = sorted(entries - _RUN_FILES)
        raise ValueError(f"smoke run artifacts are incomplete or unexpected; missing={missing}, extra={extra}")
    for name in _RUN_FILES:
        _safe_child(run_dir, run_dir / name)
    manifest = _mapping(_strict_canonical_json(run_dir / "run.json"), "run.json")
    if set(manifest) != set(MANIFEST_FIELDS):
        raise ValueError("run.json must use the exact provenance schema")
    _validate_manifest(manifest)
    if manifest["run_id"] != run_dir.name:
        raise ValueError("run directory name does not match run.json run_id")
    train_windows, validation_windows = _validate_smoke_config(
        manifest["config"], manifest["seed"]
    )
    for name in ("split_hash", "scaler_hash", "config_sha256"):
        if _HEX64.fullmatch(str(manifest[name])) is None:
            raise ValueError(f"run.json {name} must be 64 lowercase hex")
    history = _strict_canonical_json(run_dir / "history.json")
    if not isinstance(history, list):
        raise ValueError("history.json must contain a list")
    best_epoch = select_best_checkpoint(history)
    checkpoint = _mapping(
        _strict_canonical_json(run_dir / "checkpoint.json"), "checkpoint.json"
    )
    if set(checkpoint) != _CHECKPOINT_FIELDS:
        raise ValueError("checkpoint.json must use the exact checkpoint schema")
    if (
        checkpoint["run_id"] != manifest["run_id"]
        or checkpoint["best_epoch"] != best_epoch
        or checkpoint["selection_split"] != "validation"
        or checkpoint["selection_metric"] != "missing_rmse"
    ):
        raise ValueError("checkpoint selection metadata is inconsistent")
    actual_checkpoint_hash = _sha256(run_dir / "best.pt")
    if checkpoint["checkpoint_sha256"] != actual_checkpoint_hash:
        raise ValueError("best.pt SHA-256 does not match checkpoint.json")
    return {
        "run_id": manifest["run_id"],
        "config_sha256": manifest["config_sha256"],
        "split_hash": manifest["split_hash"],
        "scaler_hash": manifest["scaler_hash"],
        "checkpoint_sha256": actual_checkpoint_hash,
        "window_ids_sha256": hashlib.sha256(
            canonical_json(train_windows + validation_windows).encode("utf-8")
        ).hexdigest(),
    }


def _validate_formal_layout(root: Path) -> tuple[Path, Path]:
    entries = list(root.iterdir())
    for entry in entries:
        if entry.is_symlink():
            raise ValueError(f"formal artifact must not be a symlink: {entry.name}")
    names = {entry.name for entry in entries}
    missing = sorted(_FORMAL_FIXED_FILES - names)
    if missing:
        raise ValueError(f"formal artifact set is incomplete; missing={missing}")
    split_files = [entry for entry in entries if re.fullmatch(r"split_manifest-[0-9a-f]{64}\.csv", entry.name)]
    scaler_files = [entry for entry in entries if re.fullmatch(r"scaler-[0-9a-f]{64}\.json", entry.name)]
    if len(split_files) != 1 or len(scaler_files) != 1:
        raise ValueError("formal root must contain exactly one content-addressed split and scaler")
    allowed = set(_FORMAL_FIXED_FILES) | {
        split_files[0].name,
        scaler_files[0].name,
        "candidates",
        "evaluation",
    }
    extra = sorted(names - allowed)
    if extra:
        raise ValueError(f"formal root contains unexpected artifacts: {extra}")
    if "candidates" not in names or not (root / "candidates").is_dir():
        raise ValueError("formal root must contain the candidates directory")
    if "evaluation" in names and not (root / "evaluation").is_dir():
        raise ValueError("formal evaluation artifact must be a directory")
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"formal artifacts must not traverse symlinks: {entry}")
        try:
            resolved = entry.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"unreadable formal artifact: {entry}") from exc
        if not resolved.is_relative_to(root):
            raise ValueError(f"formal artifact escapes output root: {entry}")
    return split_files[0], scaler_files[0]


def _validate_formal_hashes(root: Path) -> None:
    hashes = _mapping(_strict_canonical_json(root / "artifact_hashes.json"), "artifact_hashes.json")
    if set(hashes) != set(_FORMAL_HASHED_FILES):
        raise ValueError("artifact_hashes.json must bind the exact completed formal artifact set")
    for name in sorted(_FORMAL_HASHED_FILES):
        digest = hashes[name]
        if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
            raise ValueError(f"artifact_hashes.json contains malformed digest for {name}")
        if _sha256(root / name) != digest:
            raise ValueError(f"formal artifact hash mismatch: {name}")


def _validate_split(root: Path, path: Path) -> tuple[pd.DataFrame, str]:
    expected_hash = path.stem.removeprefix("split_manifest-")
    if _sha256(path) != expected_hash:
        raise ValueError("split manifest content does not match its content-addressed filename")
    manifest = _read_csv(path, MANIFEST_COLUMNS, "split manifest")
    if manifest["recording_id"].astype(str).duplicated().any():
        raise ValueError("split manifest recording IDs must be unique")
    if set(manifest["split"].astype(str)) != {"train", "validation", "test"}:
        raise ValueError("split manifest must contain train, validation, and test rows")
    for row in manifest.itertuples(index=False):
        for path_field, hash_field in (("imu_path", "imu_sha256"), ("vicon_path", "vicon_sha256")):
            source = Path(str(getattr(row, path_field)))
            if not source.is_absolute():
                source = path.parent / source
            if source.is_symlink():
                raise ValueError("split manifest source paths must not be symlinks")
            try:
                source = source.resolve(strict=True)
            except OSError as exc:
                raise ValueError(f"split manifest source is missing: {source}") from exc
            recorded = str(getattr(row, hash_field))
            if _HEX64.fullmatch(recorded) is None or _sha256(source) != recorded:
                raise ValueError(f"split manifest source hash mismatch: {source}")
    return manifest, expected_hash


def _validate_scaler(path: Path, manifest: pd.DataFrame, split_hash: str) -> str:
    expected_hash = path.stem.removeprefix("scaler-")
    if _sha256(path) != expected_hash:
        raise ValueError("scaler content does not match its content-addressed filename")
    scaler = _mapping(_strict_canonical_json(path), "scaler")
    if set(scaler) != {"center", "scale", "training_ids", "split_hash"}:
        raise ValueError("scaler must use the exact frozen schema")
    if scaler["split_hash"] != split_hash:
        raise ValueError("scaler split_hash does not match the split manifest")
    center = np.asarray(scaler["center"], dtype=float)
    scale = np.asarray(scaler["scale"], dtype=float)
    if center.shape != (6,) or scale.shape != (6,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("scaler center/scale must contain six finite channels with positive scales")
    train_ids = sorted(manifest.loc[manifest["split"] == "train", "recording_id"].astype(str))
    if scaler["training_ids"] != train_ids:
        raise ValueError("scaler training_ids must exactly match the training split")
    return expected_hash


def _validate_frozen_models(root: Path, split_hash: str, scaler_hash: str) -> tuple[Mapping[str, Any], dict[tuple[int, str], str]]:
    frozen = _mapping(_strict_canonical_json(root / "frozen_models.json"), "frozen_models.json")
    required = {
        "selection_split", "split_hash", "scaler_hash", "git_commit",
        "dirty_state_digest", "strongest_baseline", "checkpoints",
    }
    if set(frozen) != required or frozen["selection_split"] != "validation":
        raise ValueError("frozen_models.json does not use the exact validation-selected schema")
    if frozen["split_hash"] != split_hash or frozen["scaler_hash"] != scaler_hash:
        raise ValueError("frozen model provenance does not match split/scaler artifacts")
    strongest = frozen["strongest_baseline"]
    checkpoints = frozen["checkpoints"]
    if not isinstance(strongest, str) or not strongest or not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("frozen model selection must contain a strongest baseline and checkpoints")
    checkpoint_fields = {
        "seed", "condition", "context_samples", "capacity", "validation_rmse",
        "validation_scores", "checkpoint_sha256", "checkpoint_path",
        "inference_config", "constructor_identity",
    }
    identities: dict[tuple[int, str], str] = {}
    candidates = (root / "candidates").resolve(strict=True)
    for item in checkpoints:
        checkpoint = _mapping(item, "frozen checkpoint")
        if set(checkpoint) != checkpoint_fields:
            raise ValueError("frozen checkpoint does not use the exact schema")
        seed = checkpoint["seed"]
        condition = checkpoint["condition"]
        if type(seed) is not int or seed not in FORMAL_SEEDS or not isinstance(condition, str) or not condition:
            raise ValueError("frozen checkpoint has invalid seed or condition")
        identity = (seed, condition)
        if identity in identities:
            raise ValueError("frozen checkpoint identities must be unique")
        if type(checkpoint["context_samples"]) is not int or checkpoint["context_samples"] <= 0:
            raise ValueError("frozen checkpoint context_samples must be positive")
        score = checkpoint["validation_rmse"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("frozen validation score must be finite")
        scores = checkpoint["validation_scores"]
        if not isinstance(scores, list) or not scores:
            raise ValueError("frozen checkpoint must preserve validation-only search scores")
        digest = checkpoint["checkpoint_sha256"]
        if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
            raise ValueError("frozen checkpoint SHA-256 is malformed")
        resolved = _safe_descendant(root, Path(str(checkpoint["checkpoint_path"])), area=candidates)
        if _sha256(resolved) != digest:
            raise ValueError("frozen checkpoint content hash mismatch")
        identities[identity] = digest
    if {seed for seed, _ in identities} != set(FORMAL_SEEDS):
        raise ValueError("frozen checkpoints must cover the exact formal seed set")
    conditions_by_seed = [{condition for candidate_seed, condition in identities if candidate_seed == seed} for seed in FORMAL_SEEDS]
    if any(conditions != conditions_by_seed[0] for conditions in conditions_by_seed[1:]):
        raise ValueError("frozen checkpoint condition matrix is incomplete")
    if strongest not in conditions_by_seed[0] or "teacher_actual_residual" not in conditions_by_seed[0]:
        raise ValueError("frozen checkpoint matrix omits the teacher or strongest baseline")
    return frozen, identities


def _validate_formal_tables(root: Path, frozen: Mapping[str, Any], checkpoint_hashes: Mapping[tuple[int, str], str]) -> None:
    metrics = validate_per_record_metrics(_read_csv(root / "per_record_metrics.csv", PER_RECORD_COLUMNS, "per-record metrics"))
    _finite_frame(metrics, "per-record metrics")
    if set(metrics["seed"].astype(int)) != set(FORMAL_SEEDS):
        raise ValueError("formal metrics must cover the exact preregistered seed set")
    non_primary = metrics.loc[metrics["protocol"] != "teacher_primary"]
    for row in non_primary.itertuples(index=False):
        expected = checkpoint_hashes.get((int(row.seed), str(row.model)))
        if expected is None or row.checkpoint_sha256 != expected:
            raise ValueError("formal metric checkpoint provenance is inconsistent")
    primary = metrics.loc[metrics["protocol"] == "teacher_primary"]
    if primary.empty or set(primary["model"].astype(str)) != {"teacher", str(frozen["strongest_baseline"])}:
        raise ValueError("formal metrics omit the exact teacher-primary comparison")
    if any(_HEX64.fullmatch(str(value)) is None for value in primary["checkpoint_sha256"]):
        raise ValueError("primary metric checkpoint provenance is malformed")

    summary = _read_csv(root / "summary.csv", SUMMARY_COLUMNS, "summary")
    _finite_frame(summary, "summary")
    gate = _mapping(_strict_canonical_json(root / "success_gate.json"), "success gate")
    expected_gate = success_gate_payload(summary, strongest_baseline=str(frozen["strongest_baseline"]))
    if set(gate) != set(expected_gate) or dict(gate) != expected_gate:
        raise ValueError("success gate is inconsistent with summary and frozen baseline")

    ledger = _read_csv(root / "mask_ledger.csv", _MASK_COLUMNS, "mask ledger")
    _finite_frame(ledger, "mask ledger")
    key = ["seed", "recording_id", "topology", "requested_fraction"]
    if ledger.duplicated(key).any() or set(ledger["seed"].astype(int)) != set(FORMAL_SEEDS):
        raise ValueError("mask ledger has duplicate cells or incomplete formal seeds")
    fractions = ledger[["requested_fraction", "realized_fraction"]].to_numpy(dtype=float)
    if ((fractions < 0.0) | (fractions > 1.0)).any() or any(_HEX64.fullmatch(str(value)) is None for value in ledger["mask_sha256"]):
        raise ValueError("mask ledger contains invalid fractions or identities")
    metric_cells = non_primary.loc[:, [*key, "realized_fraction"]].drop_duplicates()
    merged = metric_cells.merge(ledger.loc[:, [*key, "realized_fraction"]], on=key, how="outer", suffixes=("_metric", "_ledger"), indicator=True)
    if (merged["_merge"] != "both").any() or not np.allclose(merged["realized_fraction_metric"], merged["realized_fraction_ledger"], rtol=0.0, atol=1e-12):
        raise ValueError("mask ledger is inconsistent with formal metric cells")

    coverage = _read_csv(root / "coverage_ledger.csv", _COVERAGE_COLUMNS, "coverage ledger")
    _finite_frame(coverage, "coverage ledger")
    if coverage.duplicated(GROUP_COLUMNS).any():
        raise ValueError("coverage ledger comparison groups must be unique")
    if not coverage["included"].astype(str).str.lower().isin({"true", "false"}).all():
        raise ValueError("coverage ledger included must be boolean")


def _validate_formal_root(root: Path) -> dict[str, Any]:
    split_path, scaler_path = _validate_formal_layout(root)
    _validate_formal_hashes(root)
    manifest, split_hash = _validate_split(root, split_path)
    scaler_hash = _validate_scaler(scaler_path, manifest, split_hash)
    frozen, checkpoints = _validate_frozen_models(root, split_hash, scaler_hash)
    _validate_formal_tables(root, frozen, checkpoints)
    return {
        "status": "valid",
        "kind": "formal_root",
        "output": str(root),
        "run_ids": [],
        "checks": {
            "artifact_hashes": True,
            "canonical_manifests": True,
            "checkpoint_hashes": True,
            "config_hashes": True,
            "mask_hashes": True,
            "metrics_hashes": True,
            "scaler_hashes": True,
            "split_hashes": True,
            "window_identities": True,
        },
        "formal": {
            "split_hash": split_hash,
            "scaler_hash": scaler_hash,
            "strongest_baseline": frozen["strongest_baseline"],
        },
    }


def validate_artifacts(output: Path | str) -> dict[str, Any]:
    """Validate a completed smoke run/root or sealed formal output root."""
    root = _safe_directory(Path(output))
    formal_markers = _FORMAL_FIXED_FILES | frozenset({"candidates", "evaluation"})
    if any((root / name).exists() for name in formal_markers) or any(
        root.glob("split_manifest-*.csv")
    ) or any(root.glob("scaler-*.json")):
        return _validate_formal_root(root)
    if (root / "run.json").is_file():
        run = _validate_smoke_run(root)
        return {
            "status": "valid",
            "kind": "smoke_run",
            "output": str(root),
            "run_ids": [run["run_id"]],
            "checks": {
                "canonical_manifests": True,
                "config_hashes": True,
                "split_hashes": True,
                "scaler_hashes": True,
                "window_identities": True,
                "checkpoint_hashes": True,
            },
            "runs": [run],
        }
    children = sorted(root.iterdir(), key=lambda path: path.name)
    if not children or any(not child.is_dir() or child.is_symlink() for child in children):
        raise ValueError("smoke output root must contain only completed run directories")
    runs = [_validate_smoke_run(child) for child in children]
    run_ids = [run["run_id"] for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("smoke output contains duplicate run IDs")
    return {
        "status": "valid",
        "kind": "smoke_root",
        "output": str(root),
        "run_ids": run_ids,
        "checks": {
            "canonical_manifests": True,
            "config_hashes": True,
            "split_hashes": True,
            "scaler_hashes": True,
            "window_identities": True,
            "checkpoint_hashes": True,
        },
        "runs": runs,
    }


__all__ = ["validate_artifacts"]
