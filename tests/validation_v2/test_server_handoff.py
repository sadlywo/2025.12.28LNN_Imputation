from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from validation_v2.experiments.evaluate import METRIC_COLUMNS
from validation_v2.experiments.provenance import canonical_json, collect_provenance
from validation_v2.experiments.validate_artifacts import main, validate_artifacts


TRAJECTORY_METRICS = (
    "ate_rmse_m",
    "rpe_rmse_m",
    "endpoint_drift_m",
    "velocity_rmse_mps",
    "delta_ate_rmse_m",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _complete_root(tmp_path: Path, *, matrix: bool = True) -> tuple[Path, Path, dict]:
    root = tmp_path / "results"
    root.mkdir()
    sources = tmp_path / "sources"
    sources.mkdir()
    split_rows = []
    for recording_id, split in (
        ("train-record", "train"),
        ("validation-record", "validation"),
        ("test-record", "test"),
    ):
        imu = sources / f"{recording_id}-imu.csv"
        vicon = sources / f"{recording_id}-vicon.csv"
        imu.write_text(f"imu,{recording_id}\n", encoding="utf-8")
        vicon.write_text(f"vicon,{recording_id}\n", encoding="utf-8")
        split_rows.append(
            {
                "recording_id": recording_id,
                "scenario": "handbag",
                "imu_path": str(imu.resolve()),
                "vicon_path": str(vicon.resolve()),
                "split": split,
                "imu_sha256": _sha256(imu),
                "vicon_sha256": _sha256(vicon),
            }
        )
    split_path = root / "split.csv"
    with split_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(split_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(split_rows)
    split_hash = _sha256(split_path)
    final_split = root / (f"split_manifest-{split_hash}.csv" if matrix else "split_manifest.csv")
    split_path.replace(final_split)

    scaler = {
        "center": [0.0] * 6,
        "scale": [1.0] * 6,
        "channel_order": [
            "rotation_rate_x",
            "rotation_rate_y",
            "rotation_rate_z",
            "user_acc_x",
            "user_acc_y",
            "user_acc_z",
        ],
        "training_ids": ["train-record"],
        "split_hash": split_hash,
    }
    scaler_path = root / "scaler.json"
    _write_json(scaler_path, scaler)
    scaler_hash = _sha256(scaler_path)
    if matrix:
        scaler_path.replace(root / f"scaler-{scaler_hash}.json")

    condition = {
        "combination_id": hashlib.sha256(b"cell-1").hexdigest(),
        "case_type": "missingness",
        "model": "linear",
        "seed": 2026,
        "protocol": "strict_file",
        "topology": "point",
        "requested_fraction": 0.3,
        "realized_fraction": None,
    }
    if not matrix:
        condition = {
            "case_type": "missingness",
            "protocol": "strict_file",
            "topology": "point",
            "requested_fraction": 0.3,
        }
    resolved = {
        "mode": "validation_v2",
        "source_config": {
            "seeds": [2026],
            "models": ["linear"],
            "protocols": ["strict_file"],
            "topologies": ["point"],
            "rates": [0.3],
            "trajectory_enabled": True,
        },
        "model": "linear",
        "training_family": "linear",
        "reported_models": ["linear"],
        "seed": 2026,
        "protocol": "strict_file",
        "objective": "reconstruction_only",
        "condition_list": [condition],
        "resolved_device": "cpu",
        "evaluation_scope": "full_overlap_record",
        "recording_splits": [
            {"recording_id": row["recording_id"], "split": row["split"]}
            for row in split_rows
        ],
    }
    manifest = collect_provenance(
        resolved,
        2026,
        split_hash=split_hash,
        scaler_hash=scaler_hash,
        git_commit="a" * 40,
    )
    run_dir = root / manifest["run_id"]
    run_dir.mkdir()
    _write_json(run_dir / "run.json", manifest)
    _write_json(
        run_dir / "history.json",
        [
            {"epoch": 1, "train": {"missing_rmse": 0.5}, "validation": {"missing_rmse": 0.4}},
            {"epoch": 2, "train": {"missing_rmse": 0.4}, "validation": {"missing_rmse": 0.3}},
        ],
    )
    checkpoint = run_dir / "best.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    checkpoint_hash = _sha256(checkpoint)
    _write_json(
        run_dir / "checkpoint.json",
        {
            "run_id": manifest["run_id"],
            "best_epoch": 2,
            "selection_split": "validation",
            "selection_metric": "missing_rmse",
            "checkpoint_sha256": checkpoint_hash,
        },
    )
    _write_json(
        run_dir / "test_evaluation.json",
        {
            "run_id": manifest["run_id"],
            "checkpoint_sha256": checkpoint_hash,
            "status": "completed",
            "started_at": "2026-07-11T00:00:00+00:00",
            "completed_at": "2026-07-11T00:01:00+00:00",
            "records": 1,
        },
    )
    common = {
        "run_id": manifest["run_id"],
        "seed": 2026,
        "recording_id": "test-record",
        "scenario": "handbag",
        "protocol": "strict_file",
        "topology": "point",
        "requested_fraction": 0.3,
        "realized_fraction": 0.29,
        "model": "linear",
        "checkpoint_sha256": checkpoint_hash,
    }
    metrics = ("reconstruction_normalized", "reconstruction_physical", *TRAJECTORY_METRICS)
    _write_csv(
        run_dir / "per_record_metrics.csv",
        [{**common, "metric": metric, "value": 1.0} for metric in metrics],
    )

    if matrix:
        _write_json(
            root / "matrix_execution.json",
            {
                "status": "completed",
                "partial": False,
                "selected_cells": 1,
                "total_cells": 1,
                "training_groups": 1,
                "selected_combination_ids": [condition["combination_id"]],
                "run_ids": [manifest["run_id"]],
            },
        )
    else:
        _write_json(
            root / "smoke_summary.json",
            {"descriptive_only": True, "n_recordings": 1, "reason": "smoke"},
        )
    return root, run_dir, manifest


def test_complete_formal_run_validates_and_report_is_idempotent(tmp_path: Path) -> None:
    root, _, manifest = _complete_root(tmp_path)

    first = validate_artifacts(root)
    report_bytes = (root / "validation_report.json").read_bytes()
    second = validate_artifacts(root)

    assert first == second
    assert (root / "validation_report.json").read_bytes() == report_bytes
    assert first["status"] == "complete"
    assert first["run_ids"] == [manifest["run_id"]]
    assert first["summary"] == {
        "runs": 1,
        "seeds": 1,
        "models": 1,
        "records": 1,
        "cells": 1,
        "checkpoints": 1,
    }


def test_missing_checkpoint_sha256_is_named_in_failure(tmp_path: Path) -> None:
    root, run_dir, _ = _complete_root(tmp_path)
    metadata = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    del metadata["checkpoint_sha256"]
    _write_json(run_dir / "checkpoint.json", metadata)

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        validate_artifacts(root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("checkpoint", "checkpoint"),
        ("ledger", "completed"),
        ("history", "history"),
        ("duplicate_metric", "duplicate metric"),
        ("condition", "condition_list"),
    ],
)
def test_run_corruption_is_rejected(tmp_path: Path, mutation: str, message: str) -> None:
    root, run_dir, _ = _complete_root(tmp_path)
    if mutation == "checkpoint":
        (run_dir / "best.pt").write_bytes(b"tampered")
    elif mutation == "ledger":
        ledger = json.loads((run_dir / "test_evaluation.json").read_text(encoding="utf-8"))
        ledger["status"] = "started"
        _write_json(run_dir / "test_evaluation.json", ledger)
    elif mutation == "history":
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        history[1]["epoch"] = 3
        _write_json(run_dir / "history.json", history)
    elif mutation == "duplicate_metric":
        path = run_dir / "per_record_metrics.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        rows.append(dict(rows[0]))
        _write_csv(path, rows)
    else:
        path = run_dir / "per_record_metrics.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        for row in rows:
            row["topology"] = "block"
        _write_csv(path, rows)

    with pytest.raises(ValueError, match=message):
        validate_artifacts(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [("topology", "block"), ("model", "undeclared-model"), ("requested_fraction", 0.4)],
)
def test_undeclared_complete_metrics_cell_is_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    root, run_dir, _ = _complete_root(tmp_path)
    path = run_dir / "per_record_metrics.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows.extend([{**row, field: value} for row in rows])
    _write_csv(path, rows)

    with pytest.raises(ValueError, match="condition_list|evaluation cells"):
        validate_artifacts(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [("scenario", "forged-scenario"), ("realized_fraction", 0.31)],
)
def test_projected_cell_cannot_hide_a_second_extended_metrics_group(
    tmp_path: Path, field: str, value: object
) -> None:
    root, run_dir, _ = _complete_root(tmp_path)
    path = run_dir / "per_record_metrics.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows.extend([{**row, field: value} for row in rows])
    _write_csv(path, rows)

    with pytest.raises(ValueError, match="extra|duplicate evaluation cell"):
        validate_artifacts(root)


@pytest.mark.parametrize(
    "mutation",
    ["missing_key", "extra_key", "naive_time", "reverse_time", "boolean_records"],
)
def test_completed_ledger_schema_and_timeline_are_strict(
    tmp_path: Path, mutation: str
) -> None:
    root, run_dir, _ = _complete_root(tmp_path)
    path = run_dir / "test_evaluation.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing_key":
        del ledger["completed_at"]
    elif mutation == "extra_key":
        ledger["error"] = "forged"
    elif mutation == "naive_time":
        ledger["started_at"] = "2026-07-11T00:00:00"
    elif mutation == "reverse_time":
        ledger["completed_at"] = "2026-07-10T23:59:59+00:00"
    else:
        ledger["records"] = True
    _write_json(path, ledger)

    with pytest.raises(ValueError, match="test_evaluation|ledger|records|timestamp"):
        validate_artifacts(root)


def test_smoke_ledger_uses_the_same_fixed_schema(tmp_path: Path) -> None:
    root, run_dir, _ = _complete_root(tmp_path, matrix=False)
    path = run_dir / "test_evaluation.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    ledger["unexpected"] = "not allowed"
    _write_json(path, ledger)

    with pytest.raises(ValueError, match="test_evaluation|ledger"):
        validate_artifacts(root, allow_smoke=True)


@pytest.mark.parametrize("status", ["started", "failed"])
def test_incomplete_matrix_marker_is_rejected(tmp_path: Path, status: str) -> None:
    root, _, _ = _complete_root(tmp_path)
    marker = json.loads((root / "matrix_execution.json").read_text(encoding="utf-8"))
    marker["status"] = status
    _write_json(root / "matrix_execution.json", marker)

    with pytest.raises(ValueError, match="matrix_execution.*completed"):
        validate_artifacts(root)


def test_partial_matrix_can_never_validate_as_full(tmp_path: Path) -> None:
    root, _, _ = _complete_root(tmp_path)
    marker = json.loads((root / "matrix_execution.json").read_text(encoding="utf-8"))
    marker.update(partial=True, selected_cells=1, total_cells=2)
    _write_json(root / "matrix_execution.json", marker)

    with pytest.raises(ValueError, match="partial"):
        validate_artifacts(root, allow_smoke=True)


def test_smoke_requires_explicit_flag_and_descriptive_marker(tmp_path: Path) -> None:
    root, _, _ = _complete_root(tmp_path, matrix=False)

    with pytest.raises(ValueError, match="allow-smoke"):
        validate_artifacts(root)
    assert validate_artifacts(root, allow_smoke=True)["status"] == "complete"

    _write_json(root / "smoke_summary.json", {"descriptive_only": False})
    (root / "validation_report.json").unlink()
    with pytest.raises(ValueError, match="descriptive_only"):
        validate_artifacts(root, allow_smoke=True)


def test_assets_are_content_addressed_and_source_hashes_are_verified(tmp_path: Path) -> None:
    root, _, _ = _complete_root(tmp_path)
    split = next(root.glob("split_manifest-*.csv"))
    rows = list(csv.DictReader(split.open(encoding="utf-8")))
    Path(rows[0]["imu_path"]).write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="imu_sha256"):
        validate_artifacts(root)


def test_scaler_must_be_train_only_finite_and_positive(tmp_path: Path) -> None:
    root, _, _ = _complete_root(tmp_path)
    scaler_path = next(root.glob("scaler-*.json"))
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    scaler["training_ids"] = ["test-record"]
    scaler["scale"][0] = 0.0
    _write_json(scaler_path, scaler)

    with pytest.raises(ValueError, match="scaler"):
        validate_artifacts(root)


def test_config_required_seeds_are_enforced(tmp_path: Path) -> None:
    root, _, _ = _complete_root(tmp_path)
    config = tmp_path / "server.yaml"
    config.write_text(yaml.safe_dump({"seeds": [2026, 2027]}), encoding="utf-8")

    with pytest.raises(ValueError, match="2027"):
        validate_artifacts(root, config=config)


def test_marker_run_ids_and_directory_set_must_match(tmp_path: Path) -> None:
    root, _, _ = _complete_root(tmp_path)
    (root / "untracked-run").mkdir()
    (root / "untracked-run" / "run.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="run_ids"):
        validate_artifacts(root)


def test_module_cli_writes_report_and_returns_nonzero_on_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _, _ = _complete_root(tmp_path)
    assert main(["--root", str(root)]) == 0

    marker = json.loads((root / "matrix_execution.json").read_text(encoding="utf-8"))
    marker["partial"] = True
    _write_json(root / "matrix_execution.json", marker)
    assert main(["--root", str(root)]) != 0
    assert "partial" in capsys.readouterr().err
