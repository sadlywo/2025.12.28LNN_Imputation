"""One-time evaluation of a frozen validation-selected checkpoint."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping

from .provenance import _validate_manifest, canonical_json


METRIC_COLUMNS = (
    "run_id",
    "seed",
    "recording_id",
    "scenario",
    "protocol",
    "topology",
    "requested_fraction",
    "realized_fraction",
    "model",
    "metric",
    "value",
    "checkpoint_sha256",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_replace(path: Path, value: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}-", suffix=".tmp",
            encoding="utf-8", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_frozen_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    try:
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        metadata = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("run manifest and frozen checkpoint metadata are required") from error
    _validate_manifest(manifest)
    required = {
        "run_id", "best_epoch", "selection_split", "selection_metric", "checkpoint_sha256"
    }
    if set(metadata) != required:
        raise ValueError("invalid frozen checkpoint metadata")
    if metadata["run_id"] != manifest["run_id"]:
        raise ValueError("checkpoint run_id does not match run manifest")
    if metadata["selection_split"] != "validation" or metadata["selection_metric"] != "missing_rmse":
        raise ValueError("checkpoint was not selected by validation missing_rmse")
    checkpoint = run_dir / "best.pt"
    if not checkpoint.is_file() or _sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("checkpoint hash does not match frozen metadata")
    return manifest, metadata, checkpoint


def _claim_once(path: Path, run_id: str, checkpoint_sha256: str) -> dict[str, Any]:
    ledger = {
        "run_id": run_id,
        "checkpoint_sha256": checkpoint_sha256,
        "status": "started",
        "started_at": _now(),
    }
    content = (canonical_json(ledger) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=".test-evaluation-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"run_id {run_id} already evaluated") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return ledger


def _validate_rows(
    rows: list[Mapping[str, Any]], manifest: Mapping[str, Any], metadata: Mapping[str, Any],
    trajectory_enabled: bool,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("evaluation callback returned no metrics")
    normalized: list[dict[str, Any]] = []
    cell_columns = tuple(column for column in METRIC_COLUMNS if column not in {"metric", "value"})
    by_cell: dict[tuple[Any, ...], set[str]] = {}
    metric_keys: set[tuple[Any, ...]] = set()
    for row in rows:
        if set(row) != set(METRIC_COLUMNS):
            raise ValueError("metrics must use the fixed 12-column schema")
        item = dict(row)
        if item["run_id"] != manifest["run_id"] or item["seed"] != manifest["seed"]:
            raise ValueError("metric row does not match run manifest")
        if item["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
            raise ValueError("metric row does not match checkpoint hash")
        value = item["value"]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("metric value must be finite")
        for field in ("requested_fraction", "realized_fraction"):
            fraction = item[field]
            if fraction is not None and (
                not isinstance(fraction, (int, float))
                or isinstance(fraction, bool)
                or not math.isfinite(fraction)
            ):
                raise ValueError(f"{field} must be finite when present")
        recording_id = item["recording_id"]
        if not isinstance(recording_id, str) or not recording_id:
            raise ValueError("recording_id must be a non-empty string")
        cell_key = tuple(item[column] for column in cell_columns)
        metric_key = (*cell_key, item["metric"])
        if metric_key in metric_keys:
            raise ValueError("duplicate metric row")
        metric_keys.add(metric_key)
        by_cell.setdefault(cell_key, set()).add(item["metric"])
        normalized.append(item)
    for cell_key, metrics in by_cell.items():
        required = {"reconstruction_normalized", "reconstruction_physical"}
        missing = sorted(required - metrics)
        if missing:
            raise ValueError(
                f"evaluation cell {cell_key!r} missing required metrics: {', '.join(missing)}"
            )
        if trajectory_enabled:
            groups = {
                "ate": {"ate", "ate_rmse_m"},
                "rpe": {"rpe", "rpe_rmse_m"},
                "endpoint": {"endpoint", "endpoint_drift_m"},
                "velocity": {"velocity", "velocity_rmse_mps"},
                "delta": {name for name in metrics if name == "delta" or name.startswith("delta_")},
            }
            missing_groups = [name for name, choices in groups.items() if not metrics.intersection(choices)]
            if missing_groups:
                raise ValueError(
                    f"evaluation cell {cell_key!r} missing required metrics: {', '.join(missing_groups)}"
                )
    return normalized


def _write_metrics_no_clobber(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", dir=path.parent, prefix=".metrics-", suffix=".tmp",
            encoding="utf-8", delete=False
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if path.read_bytes() != temporary.read_bytes():
                raise ValueError("per_record_metrics.csv already has inconsistent content") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def evaluate_test_once(
    run_dir: Path | str,
    test_loader_factory: Callable[[], Iterable[Any]],
    evaluate_record: Callable[[Any, Path], Iterable[Mapping[str, Any]]],
    *,
    trajectory_enabled: bool = False,
) -> Path:
    """Evaluate each test record exactly once after verifying the frozen checkpoint."""

    run_dir = Path(run_dir)
    manifest, metadata, checkpoint = _load_frozen_run(run_dir)
    ledger_path = run_dir / "test_evaluation.json"
    if (run_dir / "per_record_metrics.csv").exists() and not ledger_path.exists():
        raise ValueError("partial or inconsistent test outputs cannot be resumed")
    ledger = _claim_once(ledger_path, manifest["run_id"], metadata["checkpoint_sha256"])
    try:
        rows = [
            row
            for record in test_loader_factory()
            for row in evaluate_record(record, checkpoint)
        ]
        validated = _validate_rows(rows, manifest, metadata, trajectory_enabled)
        metrics_path = run_dir / "per_record_metrics.csv"
        _write_metrics_no_clobber(metrics_path, validated)
    except BaseException as error:
        ledger.update(status="failed", failed_at=_now(), error=type(error).__name__)
        _write_json_replace(ledger_path, ledger)
        raise
    ledger.update(status="completed", completed_at=_now(), records=len({r["recording_id"] for r in validated}))
    _write_json_replace(ledger_path, ledger)
    return metrics_path
