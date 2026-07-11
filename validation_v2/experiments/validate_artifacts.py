"""Fail-closed validation for completed validation-v2 experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import tempfile
from typing import Any

import yaml

from validation_v2.data.oxiod import IMU_CHANNEL_NAMES
from validation_v2.experiments.evaluate import METRIC_COLUMNS
from validation_v2.experiments.matrix import enumerate_matrix
from validation_v2.experiments.provenance import (
    _validate_manifest,
    canonical_json,
)
from validation_v2.experiments.train import select_best_checkpoint


SPLIT_COLUMNS = (
    "recording_id",
    "scenario",
    "imu_path",
    "vicon_path",
    "split",
    "imu_sha256",
    "vicon_sha256",
)
CHECKPOINT_FIELDS = {
    "run_id",
    "best_epoch",
    "selection_split",
    "selection_metric",
    "checkpoint_sha256",
}
RECONSTRUCTION_METRICS = {
    "reconstruction_normalized",
    "reconstruction_physical",
}
TRAJECTORY_GROUPS = {
    "ate": {"ate", "ate_rmse_m"},
    "rpe": {"rpe", "rpe_rmse_m"},
    "endpoint": {"endpoint", "endpoint_drift_m"},
    "velocity": {"velocity", "velocity_rmse_mps"},
}


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except FileNotFoundError as error:
        raise ValueError(f"missing required artifact: {path.name}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path.name} must contain legal JSON") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot read artifact for SHA-256: {path}") from error
    return digest.hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_marker(root: Path, allow_smoke: bool) -> tuple[Mapping[str, Any] | None, bool]:
    path = root / "matrix_execution.json"
    if not path.is_file():
        if not allow_smoke:
            raise ValueError(
                "matrix_execution.json is required for formal validation; "
                "use --allow-smoke only for descriptive smoke output"
            )
        summary = _require_mapping(_strict_json(root / "smoke_summary.json"), "smoke_summary.json")
        if summary.get("descriptive_only") is not True:
            raise ValueError("smoke_summary.json must declare descriptive_only=true")
        return None, True

    marker = _require_mapping(_strict_json(path), "matrix_execution.json")
    if marker.get("status") != "completed":
        raise ValueError("matrix_execution.json status must be completed")
    if marker.get("partial") is not False:
        raise ValueError("partial matrix execution cannot validate as a full run")
    selected = _positive_integer(marker.get("selected_cells"), "selected_cells")
    total = _positive_integer(marker.get("total_cells"), "total_cells")
    if selected != total:
        raise ValueError("selected_cells must equal total_cells for a full matrix")
    combination_ids = marker.get("selected_combination_ids")
    if not isinstance(combination_ids, list) or any(
        not isinstance(item, str) or not item for item in combination_ids
    ):
        raise ValueError("selected_combination_ids must be a list of identifiers")
    if len(combination_ids) != selected or len(set(combination_ids)) != len(combination_ids):
        raise ValueError("selected_combination_ids count must match selected_cells and be unique")
    run_ids = marker.get("run_ids")
    if not isinstance(run_ids, list) or any(not isinstance(item, str) or not item for item in run_ids):
        raise ValueError("matrix_execution.json run_ids must be a list of identifiers")
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError("matrix_execution.json run_ids must be non-empty and unique")
    return marker, False


def _run_directories(root: Path, marker: Mapping[str, Any] | None) -> list[Path]:
    directories = sorted(
        (item for item in root.iterdir() if item.is_dir() and (item / "run.json").is_file()),
        key=lambda item: item.name,
    )
    if not directories:
        raise ValueError("no run directories containing run.json were found")
    if marker is not None:
        marker_ids = set(marker["run_ids"])
        directory_ids = {item.name for item in directories}
        if marker_ids != directory_ids:
            raise ValueError("matrix_execution.json run_ids do not exactly match run directories")
    return directories


def _asset_path(root: Path, prefix: str, digest: str, suffix: str, smoke: bool) -> Path:
    if not isinstance(digest, str) or not digest:
        raise ValueError(f"run manifest {prefix}_hash must be non-empty")
    named = root / f"{prefix}-{digest}{suffix}"
    fixed = root / f"{prefix}{suffix}"
    path = fixed if smoke else named
    if not path.is_file():
        raise ValueError(f"missing {path.name} for referenced {prefix}_hash")
    if _sha256(path) != digest:
        raise ValueError(f"{path.name} SHA-256 does not match {prefix}_hash")
    return path


def _validate_split(path: Path) -> dict[str, str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SPLIT_COLUMNS:
                raise ValueError("split manifest must use the fixed source-traceability schema")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"cannot read split manifest {path.name}") from error
    if not rows:
        raise ValueError("split manifest must not be empty")
    recording_ids: set[str] = set()
    source_paths: set[Path] = set()
    split_by_recording: dict[str, str] = {}
    for row in rows:
        recording_id = row["recording_id"].strip()
        if not recording_id or recording_id in recording_ids:
            raise ValueError("recording split assignments must be disjoint and uniquely identified")
        recording_ids.add(recording_id)
        split = row["split"]
        if split not in {"train", "validation", "test"}:
            raise ValueError("split values must be train, validation, or test")
        split_by_recording[recording_id] = split
        for path_field, hash_field in (("imu_path", "imu_sha256"), ("vicon_path", "vicon_sha256")):
            source = Path(row[path_field]).expanduser()
            if not source.is_absolute():
                source = path.parent / source
            source = source.resolve(strict=False)
            if source in source_paths:
                raise ValueError("recording source files must be disjoint across splits")
            source_paths.add(source)
            if not source.is_file() or _sha256(source) != row[hash_field]:
                raise ValueError(f"split manifest {hash_field} does not match source file")
    present = set(split_by_recording.values())
    if present != {"train", "validation", "test"}:
        raise ValueError("split manifest requires non-empty train, validation, and test splits")
    return split_by_recording


def _validate_scaler(path: Path, split_hash: str, split: Mapping[str, str]) -> None:
    scaler = _require_mapping(_strict_json(path), path.name)
    required = {"center", "scale", "channel_order", "training_ids", "split_hash"}
    if set(scaler) != required or scaler.get("split_hash") != split_hash:
        raise ValueError("scaler schema or split_hash is inconsistent")
    center = scaler["center"]
    scale = scaler["scale"]
    if not isinstance(center, list) or not isinstance(scale, list) or len(center) != len(scale):
        raise ValueError("scaler center and scale must be equal-length arrays")
    if not center or any(not math.isfinite(_finite_number(item, "scaler center")) for item in center):
        raise ValueError("scaler center must contain finite values")
    if any(_finite_number(item, "scaler scale") <= 0 for item in scale):
        raise ValueError("scaler scale must contain finite positive values")
    if scaler["channel_order"] != list(IMU_CHANNEL_NAMES) or len(center) != len(IMU_CHANNEL_NAMES):
        raise ValueError("scaler channel_order does not match the six physical IMU channels")
    training_ids = scaler["training_ids"]
    if not isinstance(training_ids, list) or not training_ids or len(set(training_ids)) != len(training_ids):
        raise ValueError("scaler training_ids must be non-empty and unique")
    if any(split.get(item) != "train" for item in training_ids):
        raise ValueError("scaler training_ids must belong only to the train split")


def _load_metrics(
    path: Path,
    manifest: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(METRIC_COLUMNS):
                raise ValueError("per_record_metrics.csv must use the fixed 12-column schema")
            raw_rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError("cannot read per_record_metrics.csv") from error
    if not raw_rows:
        raise ValueError("each run metrics recording set must be non-empty")

    rows: list[dict[str, Any]] = []
    metric_keys: set[tuple[Any, ...]] = set()
    metrics_by_cell: dict[tuple[Any, ...], set[str]] = {}
    records: set[str] = set()
    cell_columns = tuple(column for column in METRIC_COLUMNS if column not in {"metric", "value"})
    for raw in raw_rows:
        row: dict[str, Any] = dict(raw)
        if row["run_id"] != manifest["run_id"]:
            raise ValueError("CSV run_id does not match run manifest")
        try:
            seed = int(row["seed"])
        except ValueError as error:
            raise ValueError("CSV seed must be an integer") from error
        if str(seed) != row["seed"].strip() or seed != manifest["seed"]:
            raise ValueError("CSV seed does not match run manifest")
        row["seed"] = seed
        if row["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
            raise ValueError("CSV checkpoint_sha256 does not match frozen checkpoint")
        if not row["recording_id"] or not row["model"] or not row["protocol"] or not row["topology"]:
            raise ValueError("CSV recording/model/protocol/topology identifiers must be non-empty")
        records.add(row["recording_id"])
        for field in ("requested_fraction", "realized_fraction"):
            row[field] = _finite_number(row[field], field)
            if not 0 <= row[field] <= 1:
                raise ValueError(f"{field} must be between 0 and 1")
        row["value"] = _finite_number(row["value"], "metric value")
        cell_key = tuple(row[column] for column in cell_columns)
        metric_key = (*cell_key, row["metric"])
        if metric_key in metric_keys:
            raise ValueError("duplicate metric row")
        metric_keys.add(metric_key)
        metrics_by_cell.setdefault(cell_key, set()).add(row["metric"])
        rows.append(row)

    trajectory_enabled = bool(
        manifest.get("config", {}).get("source_config", {}).get("trajectory_enabled", False)
    )
    for metrics in metrics_by_cell.values():
        missing = sorted(RECONSTRUCTION_METRICS - metrics)
        if trajectory_enabled:
            missing.extend(
                name for name, choices in TRAJECTORY_GROUPS.items() if not metrics.intersection(choices)
            )
            if not any(name == "delta" or name.startswith("delta_") for name in metrics):
                missing.append("delta")
        if missing:
            raise ValueError("metrics cell missing required reconstruction/trajectory metrics: " + ", ".join(missing))
    return rows, records


def _condition_key(
    condition: Mapping[str, Any], *, default_model: str, default_protocol: str
) -> tuple[str, str, str, float]:
    case_type = condition.get("case_type", "missingness")
    if case_type == "irregular":
        topology = f"irregular:{condition.get('irregular_method')}+{condition.get('value_topology')}"
        fraction = _finite_number(condition.get("value_requested_fraction"), "value_requested_fraction")
    elif case_type == "missingness":
        topology = str(condition.get("topology", ""))
        fraction = _finite_number(condition.get("requested_fraction"), "requested_fraction")
    else:
        raise ValueError(f"condition_list has unsupported case_type: {case_type}")
    return (
        str(condition.get("model", default_model)),
        str(condition.get("protocol", default_protocol)),
        topology,
        fraction,
    )


def _validate_conditions(
    manifest: Mapping[str, Any],
    rows: list[dict[str, Any]],
    test_recordings: set[str],
    *,
    smoke: bool,
) -> set[str]:
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("run manifest resolved config must be a mapping")
    conditions = config.get("condition_list")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("run manifest condition_list must be non-empty")
    identifiers: set[str] = set()
    for condition in conditions:
        if not isinstance(condition, Mapping):
            raise ValueError("run manifest condition_list entries must be mappings")
        identifier = condition.get("combination_id")
        if smoke and identifier is None:
            identifier = hashlib.sha256(
                canonical_json(
                    {"run_id": manifest["run_id"], "condition": condition}
                ).encode("utf-8")
            ).hexdigest()
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("run manifest condition_list combination IDs must be unique")
        identifiers.add(identifier)
        if condition.get("seed", manifest["seed"] if smoke else None) != manifest["seed"]:
            raise ValueError("condition_list seed does not match run manifest")
        model, protocol, topology, fraction = _condition_key(
            condition,
            default_model=str(config.get("model", "")) if smoke else "",
            default_protocol=str(config.get("protocol", "")) if smoke else "",
        )
        covered = {
            row["recording_id"]
            for row in rows
            if row["model"] == model
            and row["protocol"] == protocol
            and row["topology"] == topology
            and math.isclose(row["requested_fraction"], fraction, rel_tol=0.0, abs_tol=1e-12)
        }
        if covered != test_recordings:
            raise ValueError(
                "run manifest condition_list cell is not covered for every test recording"
            )
    return identifiers


def _validate_run(
    root: Path,
    run_dir: Path,
    *,
    smoke: bool,
    split_cache: dict[str, dict[str, str]],
    scaler_cache: set[tuple[str, str]],
) -> dict[str, Any]:
    manifest = _require_mapping(_strict_json(run_dir / "run.json"), "run.json")
    _validate_manifest(manifest)
    if manifest["run_id"] != run_dir.name:
        raise ValueError("run directory name does not match run.json run_id")

    history = _strict_json(run_dir / "history.json")
    if not isinstance(history, list):
        raise ValueError("history.json must be an array")
    try:
        selected_epoch = select_best_checkpoint(history)
    except (TypeError, ValueError) as error:
        raise ValueError(f"history is invalid: {error}") from error

    metadata = _require_mapping(_strict_json(run_dir / "checkpoint.json"), "checkpoint.json")
    missing_checkpoint_fields = sorted(CHECKPOINT_FIELDS - set(metadata))
    if missing_checkpoint_fields:
        raise ValueError("checkpoint.json missing fields: " + ", ".join(missing_checkpoint_fields))
    if set(metadata) != CHECKPOINT_FIELDS:
        raise ValueError("checkpoint.json has unexpected fields")
    if metadata["run_id"] != manifest["run_id"]:
        raise ValueError("checkpoint run_id does not match run manifest")
    if metadata["selection_split"] != "validation" or metadata["selection_metric"] != "missing_rmse":
        raise ValueError("checkpoint selection must use validation/missing_rmse")
    if metadata["best_epoch"] != selected_epoch:
        raise ValueError("checkpoint best_epoch does not match history")
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != metadata["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA-256 does not match checkpoint_sha256")

    ledger = _require_mapping(_strict_json(run_dir / "test_evaluation.json"), "test_evaluation.json")
    if ledger.get("status") != "completed":
        raise ValueError("test_evaluation.json status must be completed")
    if ledger.get("run_id") != manifest["run_id"] or ledger.get("checkpoint_sha256") != metadata["checkpoint_sha256"]:
        raise ValueError("test_evaluation.json run/checkpoint does not match frozen run")

    split_hash = manifest["split_hash"]
    scaler_hash = manifest["scaler_hash"]
    if split_hash not in split_cache:
        split_cache[split_hash] = _validate_split(
            _asset_path(root, "split_manifest", split_hash, ".csv", smoke)
        )
    split = split_cache[split_hash]
    scaler_key = (scaler_hash, split_hash)
    if scaler_key not in scaler_cache:
        _validate_scaler(
            _asset_path(root, "scaler", scaler_hash, ".json", smoke), split_hash, split
        )
        scaler_cache.add(scaler_key)
    test_recordings = {recording_id for recording_id, split_name in split.items() if split_name == "test"}
    rows, records = _load_metrics(run_dir / "per_record_metrics.csv", manifest, metadata)
    if records != test_recordings:
        raise ValueError("metrics recording set must exactly match split test recordings")
    if ledger.get("records") != len(records):
        raise ValueError("test_evaluation.json records does not match metrics recording set")
    condition_ids = _validate_conditions(
        manifest, rows, test_recordings, smoke=smoke
    )
    return {
        "run_id": manifest["run_id"],
        "seed": manifest["seed"],
        "models": {row["model"] for row in rows},
        "records": records,
        "condition_ids": condition_ids,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
    }


def _load_config(path: Path | str) -> Mapping[str, Any]:
    path = Path(path)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read validation config: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError("validation config must be a YAML mapping")
    return value


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    content = (canonical_json(report) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("validation_report.json already has inconsistent content")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=".validation-report-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if path.read_bytes() != content:
                raise ValueError("validation_report.json already has inconsistent content") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_artifacts(
    root: Path | str,
    *,
    config: Path | str | None = None,
    allow_smoke: bool = False,
) -> dict[str, Any]:
    """Validate a complete artifact root and atomically seal its report."""

    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"artifact root is not a directory: {root}")
    marker, smoke = _validate_marker(root, allow_smoke)
    directories = _run_directories(root, marker)
    split_cache: dict[str, dict[str, str]] = {}
    scaler_cache: set[tuple[str, str]] = set()
    validated = [
        _validate_run(
            root,
            directory,
            smoke=smoke,
            split_cache=split_cache,
            scaler_cache=scaler_cache,
        )
        for directory in directories
    ]
    run_ids = [item["run_id"] for item in validated]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("duplicate run_id")
    condition_ids = [identifier for item in validated for identifier in item["condition_ids"]]
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("duplicate condition_list combination ID across runs")
    if marker is not None:
        selected_ids = set(marker["selected_combination_ids"])
        if set(condition_ids) != selected_ids:
            raise ValueError(
                "run manifest condition_list union does not match selected_combination_ids"
            )

    if config is not None:
        loaded_config = _load_config(config)
        required_seeds = {int(seed) for seed in loaded_config.get("seeds", ())}
        actual_seeds = {int(item["seed"]) for item in validated}
        if actual_seeds != required_seeds:
            missing = sorted(required_seeds - actual_seeds)
            extra = sorted(actual_seeds - required_seeds)
            raise ValueError(f"required seeds mismatch; missing={missing}, unexpected={extra}")
        if marker is not None:
            expected_ids = {item["combination_id"] for item in enumerate_matrix(loaded_config)}
            if set(marker["selected_combination_ids"]) != expected_ids:
                raise ValueError(
                    "matrix marker does not contain every required config protocol/model/family cell"
                )

    seeds = sorted({int(item["seed"]) for item in validated})
    models = sorted({model for item in validated for model in item["models"]})
    records = sorted({record for item in validated for record in item["records"]})
    checkpoints = sorted({item["checkpoint_sha256"] for item in validated})
    report = {
        "status": "complete",
        "descriptive_only": smoke,
        "summary": {
            "runs": len(run_ids),
            "seeds": len(seeds),
            "models": len(models),
            "records": len(records),
            "cells": len(set(condition_ids)),
            "checkpoints": len(checkpoints),
        },
        "run_ids": sorted(run_ids),
        "seeds": seeds,
        "models": models,
        "records": records,
        "cells": sorted(set(condition_ids)),
        "checkpoint_hashes": checkpoints,
    }
    _write_report(root / "validation_report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m validation_v2.experiments.validate_artifacts",
        description="Validate a completed validation-v2 artifact root before summarization.",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = validate_artifacts(
            arguments.root, config=arguments.config, allow_smoke=arguments.allow_smoke
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        print(f"validation-v2 artifacts: {error}", file=os.sys.stderr)
        return 2
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "validate_artifacts"]
