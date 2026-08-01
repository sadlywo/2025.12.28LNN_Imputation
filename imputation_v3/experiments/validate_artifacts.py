"""Read-only, fail-closed validation for completed imputation-v3 artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from imputation_v3.config import TeacherConfig
from imputation_v3.data.windows import collate_prepared_windows, iter_teacher_windows
from imputation_v3.experiments.training import make_teacher_callbacks
from imputation_v3.models.teacher import OfflineTeacher
from imputation_v3.experiments.runner import (
    FORMAL_SEEDS,
    formal_mask_seed,
    formal_matrix_plan,
    success_gate_payload,
)
from validation_v2.data.masking import channel_outage, contiguous_block, point_missing
from validation_v2.data.normalization import RobustTrainScaler
from validation_v2.data.oxiod import load_recording
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
from validation_v2.experiments.runner import discover_oxiod_pairs
from validation_v2.data.splits import stratified_file_split


_RUN_FILES = frozenset(
    {"run.json", "history.json", "best.pt", "checkpoint.json", "evidence.json"}
)
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
    {
        "artifact_hashes.json", "frozen_models.json", "resolved_config.json",
        "window_identity_ledger.json",
    }
)
_MASK_COLUMNS = (
    "seed",
    "recording_id",
    "topology",
    "requested_fraction",
    "realized_fraction",
    "mask_sha256",
    "generator",
    "condition_seed",
    "target_source_sha256",
    "target_length",
    "channels",
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


def _validate_smoke_config(
    config: Any, manifest_seed: int, split_hash: str, scaler_hash: str
) -> tuple[list[str], list[str]]:
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

    split_manifest = config.get("split_manifest")
    if not isinstance(split_manifest, list) or not split_manifest:
        raise ValueError("split_manifest must preserve canonical split evidence")
    split_fields = {
        "recording_id", "scenario", "imu_path", "vicon_path", "split",
        "imu_sha256", "vicon_sha256",
    }
    if any(not isinstance(row, Mapping) or set(row) != split_fields for row in split_manifest):
        raise ValueError("split_manifest rows must use the exact evidence schema")
    if split_manifest != sorted(split_manifest, key=lambda row: row["recording_id"]):
        raise ValueError("split_manifest rows must use canonical recording order")
    if hashlib.sha256(canonical_json(split_manifest).encode("utf-8")).hexdigest() != split_hash:
        raise ValueError("split_hash does not match preserved split membership")
    for row in split_manifest:
        for path_name, hash_name in (("imu_path", "imu_sha256"), ("vicon_path", "vicon_sha256")):
            source = Path(str(row[path_name]))
            if source.is_symlink() or not source.is_file():
                raise ValueError("smoke split source is missing or symlinked")
            digest = str(row[hash_name])
            if _HEX64.fullmatch(digest) is None or _sha256(source) != digest:
                raise ValueError("smoke split source hash mismatch")
    computed_counts = {
        name: sum(row["split"] == name for row in split_manifest)
        for name in ("train", "validation", "test")
    }
    if dict(split_counts) != computed_counts:
        raise ValueError("split_counts do not match preserved split membership")

    scaler_state = _mapping(config.get("scaler_state"), "scaler_state")
    if set(scaler_state) != {"center", "scale", "training_ids", "split_hash"}:
        raise ValueError("scaler_state must use the exact frozen schema")
    if hashlib.sha256(canonical_json(scaler_state).encode("utf-8")).hexdigest() != scaler_hash:
        raise ValueError("scaler_hash does not match preserved scaler material")
    if scaler_state["split_hash"] != split_hash or scaler_state["training_ids"] != scaler_ids:
        raise ValueError("scaler state provenance is inconsistent")
    center = np.asarray(scaler_state["center"], dtype=float)
    scale = np.asarray(scaler_state["scale"], dtype=float)
    if center.shape != (6,) or scale.shape != (6,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("scaler evidence must contain six finite channels and positive scales")

    window_evidence = _mapping(config.get("window_evidence"), "window_evidence")
    if set(window_evidence) != {"train", "validation"}:
        raise ValueError("window_evidence must contain train and validation")
    evidence_fields = {
        "window_id", "recording_id", "topology", "requested_fraction",
        "realized_fraction", "mask_sha256", "mask_bytes_hex", "mask_dtype",
        "samples", "channels",
    }
    for split, expected_ids in (("train", train_windows), ("validation", validation_windows)):
        rows = window_evidence[split]
        if not isinstance(rows, list) or [row.get("window_id") for row in rows if isinstance(row, Mapping)] != expected_ids:
            raise ValueError("window evidence identities do not match selected_window_ids")
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != evidence_fields:
                raise ValueError("window evidence row does not use the exact schema")
            if row["mask_dtype"] != "float32-le" or type(row["samples"]) is not int or row["samples"] < 1 or row["channels"] != 6:
                raise ValueError("window mask material has invalid dtype or shape")
            try:
                content = bytes.fromhex(str(row["mask_bytes_hex"]))
            except ValueError as exc:
                raise ValueError("window mask material is not valid hexadecimal") from exc
            if len(content) != row["samples"] * row["channels"] * 4 or hashlib.sha256(content).hexdigest() != row["mask_sha256"]:
                raise ValueError("window mask material hash mismatch")
            mask = np.frombuffer(content, dtype="<f4")
            if not np.isin(mask, (0.0, 1.0)).all():
                raise ValueError("window mask material must be binary")
            realized = float(np.mean(mask == 0.0))
            if not math.isclose(realized, float(row["realized_fraction"]), rel_tol=0.0, abs_tol=1e-15):
                raise ValueError("window mask realized fraction is inconsistent")
    return train_windows, validation_windows


def _replayed_window_evidence(window: Any) -> dict[str, Any]:
    mask = np.ascontiguousarray(window.mask.numpy(), dtype=np.float32)
    content = mask.tobytes()
    return {
        "window_id": window.window_id,
        "recording_id": window.recording_id,
        "topology": window.topology,
        "requested_fraction": window.requested_fraction,
        "realized_fraction": window.realized_fraction,
        "mask_sha256": hashlib.sha256(content).hexdigest(),
        "mask_bytes_hex": content.hex(),
        "mask_dtype": "float32-le",
        "samples": int(mask.shape[0]),
        "channels": int(mask.shape[1]),
    }


def _replay_smoke_pipeline(
    run_dir: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, list[Any]], dict[str, dict[str, float]]]:
    config = _mapping(manifest["config"], "smoke replay config")
    data_root = Path(str(config["data_root"]))
    discovered = discover_oxiod_pairs(data_root)
    replayed_split = stratified_file_split(
        discovered, seed=int(config["split_seed"])
    )
    replayed_rows = [
        {
            "recording_id": str(row["recording_id"]),
            "scenario": str(row["scenario"]),
            "imu_path": str(row["imu_path"]),
            "vicon_path": str(row["vicon_path"]),
            "split": str(row["split"]),
            "imu_sha256": str(row["imu_sha256"]),
            "vicon_sha256": str(row["vicon_sha256"]),
        }
        for row in replayed_split.to_dict(orient="records")
    ]
    replayed_rows.sort(key=lambda row: row["recording_id"])
    if replayed_rows != config["split_manifest"]:
        raise ValueError("smoke replay split membership/source evidence differs")
    selected_rows = {
        split: [row for row in replayed_rows if row["split"] == split][:2]
        for split in ("train", "validation")
    }
    expected_selected = {
        split: [row["recording_id"] for row in selected_rows[split]]
        for split in ("train", "validation")
    }
    if expected_selected != config["selected_recording_ids"]:
        raise ValueError("smoke replay selected recording order differs")
    recordings = {
        split: [
            load_recording(Path(row["imu_path"]), Path(row["vicon_path"]))
            for row in selected_rows[split]
        ]
        for split in ("train", "validation")
    }
    for split in recordings:
        if [recording.id for recording in recordings[split]] != expected_selected[split]:
            raise ValueError("smoke replay loaded recording identities differ")
    fitted = RobustTrainScaler.fit(
        recordings["train"], allowed_ids=set(expected_selected["train"])
    )
    scaler_state = config["scaler_state"]
    if (
        list(fitted.training_ids) != scaler_state["training_ids"]
        or not np.array_equal(fitted.center_, np.asarray(scaler_state["center"], dtype=np.float64))
        or not np.array_equal(fitted.scale_, np.asarray(scaler_state["scale"], dtype=np.float64))
    ):
        raise ValueError("smoke replay fitted scaler differs from sealed scaler")
    hyper = _mapping(config["hyperparameters"], "smoke replay hyperparameters")
    prepared = {
        split: list(
            islice(
                iter_teacher_windows(
                    recordings[split],
                    fitted,
                    window_samples=int(hyper["window_samples"]),
                    stride=int(hyper["stride"]),
                    seed=int(config["seed"]),
                    topologies=tuple(hyper["training_topologies"]),
                    rates=tuple(hyper["training_rates"]),
                    exhaustive=False,
                ),
                int(config["bounds"]["max_windows_per_split"]),
            )
        )
        for split in ("train", "validation")
    }
    replayed_window_evidence = {
        split: [_replayed_window_evidence(window) for window in prepared[split]]
        for split in ("train", "validation")
    }
    if replayed_window_evidence != config["window_evidence"]:
        raise ValueError("smoke replay window or mask evidence differs from source regeneration")
    model = OfflineTeacher(
        31,
        int(hyper["hidden_size"]),
        int(hyper["tcn_width"]),
        tuple(hyper["tcn_dilations"]),
        residual_mode=str(hyper["residual_mode"]),
        time_mode=str(hyper["time_mode"]),
    )
    try:
        state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("smoke replay cannot reconstruct the frozen teacher checkpoint") from exc
    _, evaluate_epoch = make_teacher_callbacks(torch.device("cpu"))
    final_metrics: dict[str, dict[str, float]] = {}
    for split in ("train", "validation"):
        loader = DataLoader(
            prepared[split],
            batch_size=int(hyper["batch_size"]),
            shuffle=False,
            collate_fn=collate_prepared_windows,
        )
        measured = evaluate_epoch(model, loader, 0)
        final_metrics[split] = {"missing_rmse": float(measured["missing_rmse"])}
    return prepared, final_metrics


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
        manifest["config"], manifest["seed"], manifest["split_hash"], manifest["scaler_hash"]
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
    evidence = _mapping(_strict_canonical_json(run_dir / "evidence.json"), "evidence.json")
    expected_evidence = {
        "schema": "imputation-v3-smoke-evidence-v2",
        "run_id": manifest["run_id"],
        "run_manifest_sha256": _sha256(run_dir / "run.json"),
        "history_sha256": _sha256(run_dir / "history.json"),
        "checkpoint_metadata_sha256": _sha256(run_dir / "checkpoint.json"),
        "checkpoint_sha256": actual_checkpoint_hash,
    }
    if set(evidence) != {
        *expected_evidence,
        "final_checkpoint_metrics",
        "final_checkpoint_metrics_sha256",
    } or any(
        evidence.get(name) != value for name, value in expected_evidence.items()
    ):
        raise ValueError("smoke evidence does not bind the completed artifact set")
    _, replayed_metrics = _replay_smoke_pipeline(run_dir, manifest)
    recorded_metrics = _mapping(
        evidence["final_checkpoint_metrics"], "final checkpoint metrics"
    )
    if evidence["final_checkpoint_metrics_sha256"] != hashlib.sha256(
        canonical_json(recorded_metrics).encode("utf-8")
    ).hexdigest():
        raise ValueError("final checkpoint metric digest is inconsistent")
    if set(recorded_metrics) != {"train", "validation"}:
        raise ValueError("final checkpoint metrics must contain train and validation")
    for split in ("train", "validation"):
        recorded = _mapping(recorded_metrics[split], f"{split} final checkpoint metrics")
        if set(recorded) != {"missing_rmse"}:
            raise ValueError("final checkpoint metrics must use the exact replay schema")
        value = recorded["missing_rmse"]
        expected = replayed_metrics[split]["missing_rmse"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("final checkpoint RMSE differs from independent smoke replay")
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


def _validate_formal_hashes(root: Path, split_path: Path, scaler_path: Path) -> None:
    hashes = _mapping(_strict_canonical_json(root / "artifact_hashes.json"), "artifact_hashes.json")
    expected = set(_FORMAL_HASHED_FILES) | {
        "frozen_models.json", "resolved_config.json", "window_identity_ledger.json",
        split_path.name, scaler_path.name,
    }
    if set(hashes) != expected:
        raise ValueError("artifact_hashes.json must bind the exact completed formal artifact set")
    for name in sorted(expected):
        digest = hashes[name]
        if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
            raise ValueError(f"artifact_hashes.json contains malformed digest for {name}")
        if _sha256(root / name) != digest:
            raise ValueError(f"formal artifact hash mismatch: {name}")


def _config_from_payload(payload: Mapping[str, Any]) -> TeacherConfig:
    expected = {
        "data_root", "output_root", "selection_split", "seeds", "window_seconds",
        "nominal_dt_s", "batch_size", "epochs", "hidden_size", "tcn_width",
        "tcn_dilations", "learning_rate", "training_rates", "training_topologies",
        "models",
    }
    if set(payload) != expected:
        raise ValueError("resolved formal config does not use the exact TeacherConfig schema")
    return TeacherConfig(
        data_root=Path(str(payload["data_root"])),
        output_root=Path(str(payload["output_root"])),
        selection_split=str(payload["selection_split"]),
        seeds=tuple(payload["seeds"]),
        window_seconds=tuple(payload["window_seconds"]),
        nominal_dt_s=payload["nominal_dt_s"],
        batch_size=payload["batch_size"],
        epochs=payload["epochs"],
        hidden_size=payload["hidden_size"],
        tcn_width=payload["tcn_width"],
        tcn_dilations=tuple(payload["tcn_dilations"]),
        learning_rate=payload["learning_rate"],
        training_rates=tuple(payload["training_rates"]),
        training_topologies=tuple(payload["training_topologies"]),
        models=tuple(payload["models"]),
    )


def _validate_resolved_config(root: Path) -> tuple[TeacherConfig, str, str]:
    document = _mapping(
        _strict_canonical_json(root / "resolved_config.json"), "resolved_config.json"
    )
    if set(document) != {"schema", "resolved", "resolved_config_sha256", "matrix_plan_sha256"} or document["schema"] != "imputation-v3-formal-resolved-config-v1":
        raise ValueError("resolved_config.json does not use the exact formal schema")
    resolved = _mapping(document["resolved"], "resolved formal payload")
    if set(resolved) != {"config", "device", "output_root"}:
        raise ValueError("resolved formal payload does not use the exact schema")
    if resolved["device"] not in {"cpu", "cuda"}:
        raise ValueError("resolved formal device is invalid")
    try:
        recorded_output = Path(str(resolved["output_root"])).resolve(strict=True)
    except OSError as exc:
        raise ValueError("resolved formal output root does not exist") from exc
    if recorded_output != root:
        raise ValueError("resolved formal output root does not match validated root")
    payload = _mapping(resolved["config"], "resolved formal config")
    config_sha = hashlib.sha256(canonical_json(resolved).encode("utf-8")).hexdigest()
    if document["resolved_config_sha256"] != config_sha:
        raise ValueError("resolved formal config digest mismatch")
    config = _config_from_payload(payload)
    plan_sha = hashlib.sha256(canonical_json(formal_matrix_plan(config)).encode("utf-8")).hexdigest()
    if document["matrix_plan_sha256"] != plan_sha:
        raise ValueError("resolved formal matrix digest mismatch")
    return config, config_sha, plan_sha


def _validate_window_ledger(
    root: Path, config: TeacherConfig
) -> tuple[dict[tuple[int, int, str], tuple[str, int]], str]:
    path = root / "window_identity_ledger.json"
    document = _mapping(_strict_canonical_json(path), "window_identity_ledger.json")
    if set(document) != {"schema", "entries", "entries_sha256"} or document["schema"] != "imputation-v3-formal-window-identities-v1":
        raise ValueError("window identity ledger does not use the exact schema")
    entries = document["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("window identity ledger must contain entries")
    if document["entries_sha256"] != hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest():
        raise ValueError("window identity ledger digest mismatch")
    expected_fields = {
        "seed", "context_samples", "split", "window_ids_sha256", "window_count"
    }
    identities: dict[tuple[int, int, str], tuple[str, int]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != expected_fields:
            raise ValueError("window identity ledger entry schema mismatch")
        key = (entry["seed"], entry["context_samples"], entry["split"])
        digest, count = entry["window_ids_sha256"], entry["window_count"]
        if key in identities or key[0] not in FORMAL_SEEDS or key[2] not in {"train", "validation"} or key[1] not in config.window_samples:
            raise ValueError("window identity ledger contains an unexpected or duplicate cell")
        if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None or type(count) is not int or count <= 0:
            raise ValueError("window identity ledger contains malformed identity material")
        identities[key] = (digest, count)
    expected_keys = {
        (seed, samples, split)
        for seed in FORMAL_SEEDS
        for samples in config.window_samples
        for split in ("train", "validation")
    }
    if set(identities) != expected_keys:
        raise ValueError("window identity ledger does not cover the full matrix")
    return identities, _sha256(path)


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


def _validate_frozen_models(
    root: Path,
    split_hash: str,
    scaler_hash: str,
    config_sha256: str,
    matrix_plan_sha256: str,
    window_identities: Mapping[tuple[int, int, str], tuple[str, int]],
    window_ledger_sha256: str,
    config: TeacherConfig,
) -> tuple[Mapping[str, Any], dict[tuple[int, str], str]]:
    frozen = _mapping(_strict_canonical_json(root / "frozen_models.json"), "frozen_models.json")
    required = {
        "selection_split", "split_hash", "scaler_hash", "git_commit",
        "dirty_state_digest", "strongest_baseline", "checkpoints",
        "resolved_config_sha256", "matrix_plan_sha256",
        "window_identity_ledger_sha256",
    }
    if set(frozen) != required or frozen["selection_split"] != "validation":
        raise ValueError("frozen_models.json does not use the exact validation-selected schema")
    if frozen["split_hash"] != split_hash or frozen["scaler_hash"] != scaler_hash:
        raise ValueError("frozen model provenance does not match split/scaler artifacts")
    if (
        frozen["resolved_config_sha256"] != config_sha256
        or frozen["matrix_plan_sha256"] != matrix_plan_sha256
        or frozen["window_identity_ledger_sha256"] != window_ledger_sha256
    ):
        raise ValueError("frozen model provenance does not match config or window evidence")
    strongest = frozen["strongest_baseline"]
    checkpoints = frozen["checkpoints"]
    if not isinstance(strongest, str) or not strongest or not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("frozen model selection must contain a strongest baseline and checkpoints")
    checkpoint_fields = {
        "seed", "condition", "context_samples", "capacity", "validation_rmse",
        "validation_scores", "checkpoint_sha256", "checkpoint_path",
        "inference_config", "constructor_identity",
        "train_window_ids_sha256", "train_window_count",
        "validation_window_ids_sha256", "validation_window_count",
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
        for split in ("train", "validation"):
            expected_window = window_identities.get(
                (seed, checkpoint["context_samples"], split)
            )
            actual_window = (
                checkpoint[f"{split}_window_ids_sha256"],
                checkpoint[f"{split}_window_count"],
            )
            if actual_window != expected_window:
                raise ValueError("frozen checkpoint window identity disagrees with ledger")
    if {seed for seed, _ in identities} != set(FORMAL_SEEDS):
        raise ValueError("frozen checkpoints must cover the exact formal seed set")
    conditions_by_seed = [{condition for candidate_seed, condition in identities if candidate_seed == seed} for seed in FORMAL_SEEDS]
    if any(conditions != conditions_by_seed[0] for conditions in conditions_by_seed[1:]):
        raise ValueError("frozen checkpoint condition matrix is incomplete")
    if strongest not in conditions_by_seed[0] or "teacher_actual_residual" not in conditions_by_seed[0]:
        raise ValueError("frozen checkpoint matrix omits the teacher or strongest baseline")
    expected_conditions = {
        str(cell["condition"]) for cell in formal_matrix_plan(config)["cells"]
    }
    if conditions_by_seed[0] != expected_conditions:
        raise ValueError("frozen checkpoint conditions do not match the resolved matrix")
    return frozen, identities


def _validate_formal_tables(
    root: Path,
    manifest: pd.DataFrame,
    frozen: Mapping[str, Any],
    checkpoint_hashes: Mapping[tuple[int, str], str],
) -> None:
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
    for row in primary.itertuples(index=False):
        condition = (
            "teacher_actual_residual"
            if row.model == "teacher"
            else str(frozen["strongest_baseline"])
        )
        if row.checkpoint_sha256 != checkpoint_hashes.get((int(row.seed), condition)):
            raise ValueError("primary metric checkpoint provenance is inconsistent")

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
    test_sources = {
        str(row.recording_id): str(row.imu_sha256)
        for row in manifest.loc[manifest["split"] == "test"].itertuples(index=False)
    }
    generators = {
        "point": point_missing,
        "block": contiguous_block,
        "channel": channel_outage,
    }
    for row in ledger.itertuples(index=False):
        if (
            row.generator != "formal-test-mask-v1"
            or row.target_source_sha256 != test_sources.get(str(row.recording_id))
            or type(row.condition_seed) is not int
            or type(row.target_length) is not int
            or type(row.channels) is not int
            or row.target_length <= 0
            or row.channels != 6
        ):
            raise ValueError("mask generator/source/shape evidence is inconsistent")
        expected_seed = formal_mask_seed(
            str(row.recording_id), int(row.seed), str(row.topology),
            float(row.requested_fraction),
        )
        if row.condition_seed != expected_seed or row.topology not in generators:
            raise ValueError("mask condition seed is inconsistent")
        target = torch.zeros((row.target_length, row.channels), dtype=torch.float32)
        recomputed = generators[row.topology](
            target, float(row.requested_fraction), expected_seed
        ).mask
        content = np.ascontiguousarray(recomputed.numpy()).tobytes()
        if hashlib.sha256(content).hexdigest() != row.mask_sha256:
            raise ValueError("mask SHA-256 cannot be recomputed from sealed generator evidence")
        realized = float((recomputed == 0).double().mean())
        if not math.isclose(realized, float(row.realized_fraction), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("mask realized fraction cannot be recomputed")
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
    _validate_formal_hashes(root, split_path, scaler_path)
    manifest, split_hash = _validate_split(root, split_path)
    scaler_hash = _validate_scaler(scaler_path, manifest, split_hash)
    config, config_sha, matrix_sha = _validate_resolved_config(root)
    window_identities, window_ledger_sha = _validate_window_ledger(root, config)
    frozen, checkpoints = _validate_frozen_models(
        root, split_hash, scaler_hash, config_sha, matrix_sha,
        window_identities, window_ledger_sha, config,
    )
    _validate_formal_tables(root, manifest, frozen, checkpoints)
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
                "mask_hashes": True,
                "checkpoint_hashes": True,
                "metrics_hashes": True,
                "history_integrity": True,
                "artifact_hashes": True,
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
            "mask_hashes": True,
            "checkpoint_hashes": True,
            "metrics_hashes": True,
            "history_integrity": True,
            "artifact_hashes": True,
        },
        "runs": runs,
    }


__all__ = ["validate_artifacts"]
