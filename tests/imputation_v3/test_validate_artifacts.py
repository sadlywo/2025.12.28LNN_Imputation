from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import pandas as pd

from imputation_v3.experiments.validate_artifacts import validate_artifacts
from imputation_v3.experiments.runner import (
    FORMAL_SEEDS,
    make_primary_rows,
    paired_formal_summaries,
    success_gate_payload,
    write_formal_artifacts,
)
from validation_v2.data.splits import MANIFEST_COLUMNS
from validation_v2.evaluation.statistics import PER_RECORD_COLUMNS
from validation_v2.experiments.provenance import canonical_json, collect_provenance


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes((canonical_json(value) + "\n").encode("utf-8"))


def _smoke_config() -> dict:
    return {
        "mode": "imputation_v3_teacher_smoke",
        "selection_split": "validation",
        "test_evaluation": False,
        "model": "teacher",
        "seed": 2026,
        "device": "cpu",
        "data_root": "Oxford Dataset",
        "source_config": {"selection_split": "validation"},
        "split_seed": 2026,
        "split_counts": {"train": 2, "validation": 1, "test": 1},
        "selected_recording_ids": {
            "train": ["train-a", "train-b"],
            "validation": ["validation-a"],
        },
        "scaler_training_ids": ["train-a", "train-b"],
        "selected_window_ids": {
            "train": ["teacher-window-sha256-" + "1" * 64],
            "validation": ["teacher-window-sha256-" + "2" * 64],
        },
        "bounds": {"max_recordings_per_split": 2, "max_windows_per_split": 4},
        "hyperparameters": {
            "window_samples": 128,
            "stride": 64,
            "batch_size": 2,
            "epochs": 1,
            "hidden_size": 16,
            "tcn_width": 16,
            "tcn_dilations": [1, 2],
            "learning_rate": 0.001,
            "training_rates": [0.2],
            "training_topologies": ["point", "block"],
            "residual_mode": "residual",
            "time_mode": "actual",
        },
    }


def _make_smoke_root(tmp_path: Path) -> tuple[Path, Path, dict]:
    root = tmp_path / "smoke"
    manifest = collect_provenance(
        _smoke_config(),
        2026,
        split_hash="a" * 64,
        scaler_hash="b" * 64,
        git_commit="c" * 40,
        dirty_digest="",
    )
    run_dir = root / manifest["run_id"]
    run_dir.mkdir(parents=True)
    _write_canonical(run_dir / "run.json", manifest)
    history = [
        {
            "epoch": 1,
            "train": {"missing_rmse": 1.25},
            "validation": {"missing_rmse": 1.0},
        }
    ]
    _write_canonical(run_dir / "history.json", history)
    torch.save({"weight": torch.tensor([1.0])}, run_dir / "best.pt")
    checkpoint_hash = hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest()
    checkpoint = {
        "run_id": manifest["run_id"],
        "best_epoch": 1,
        "selection_split": "validation",
        "selection_metric": "missing_rmse",
        "checkpoint_sha256": checkpoint_hash,
    }
    _write_canonical(run_dir / "checkpoint.json", checkpoint)
    return root, run_dir, manifest


def test_validate_smoke_root_and_direct_run_are_read_only(tmp_path):
    root, run_dir, manifest = _make_smoke_root(tmp_path)
    before = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }

    root_report = validate_artifacts(root)
    run_report = validate_artifacts(run_dir)

    assert root_report["kind"] == "smoke_root"
    assert root_report["run_ids"] == [manifest["run_id"]]
    assert root_report["checks"]["checkpoint_hashes"] is True
    assert root_report["checks"]["window_identities"] is True
    assert run_report["kind"] == "smoke_run"
    assert run_report["run_ids"] == [manifest["run_id"]]
    after = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


def _formal_metric_row(
    *, seed: int, recording_id: str, model: str, checkpoint_sha256: str, value: float
) -> dict:
    return {
        "run_id": f"formal-{seed}-{model}",
        "seed": seed,
        "recording_id": recording_id,
        "scenario": "handheld",
        "protocol": "overall",
        "topology": "point",
        "requested_fraction": 0.2,
        "realized_fraction": 0.2,
        "model": model,
        "metric": "rmse_physical",
        "value": value,
        "checkpoint_sha256": checkpoint_sha256,
    }


def _make_formal_root(tmp_path: Path) -> Path:
    root = tmp_path / "formal"
    root.mkdir()
    sources = tmp_path / "sources"
    sources.mkdir()
    split_rows = []
    for recording_id, split in (
        ("train-a", "train"),
        ("validation-a", "validation"),
        ("test-a", "test"),
        ("test-b", "test"),
    ):
        pair = []
        for kind in ("imu", "vicon"):
            source = sources / f"{recording_id}-{kind}.csv"
            source.write_bytes(f"{recording_id}-{kind}\n".encode("ascii"))
            pair.append((str(source.resolve()), hashlib.sha256(source.read_bytes()).hexdigest()))
        split_rows.append(
            [recording_id, "handheld", pair[0][0], pair[1][0], split, pair[0][1], pair[1][1]]
        )
    split = pd.DataFrame(split_rows, columns=MANIFEST_COLUMNS)
    split_bytes = split.to_csv(index=False, lineterminator="\n").encode("utf-8")
    split_hash = hashlib.sha256(split_bytes).hexdigest()
    (root / f"split_manifest-{split_hash}.csv").write_bytes(split_bytes)
    scaler_state = {
        "center": [0.0] * 6,
        "scale": [1.0] * 6,
        "training_ids": ["train-a"],
        "split_hash": split_hash,
    }
    scaler_bytes = (canonical_json(scaler_state) + "\n").encode("utf-8")
    scaler_hash = hashlib.sha256(scaler_bytes).hexdigest()
    (root / f"scaler-{scaler_hash}.json").write_bytes(scaler_bytes)

    candidates = root / "candidates"
    candidates.mkdir()
    checkpoints = []
    checkpoint_hashes = {}
    for seed in FORMAL_SEEDS:
        for condition in ("linear", "teacher_actual_residual"):
            checkpoint = candidates / f"{seed}-{condition}.bin"
            checkpoint.write_bytes(f"{seed}:{condition}".encode("ascii"))
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            checkpoint_hashes[(seed, condition)] = digest
            checkpoints.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "context_samples": 128,
                    "capacity": {"capacity": "fixed"},
                    "validation_rmse": 1.0 if condition == "linear" else 0.5,
                    "validation_scores": [
                        {
                            "context_samples": 128,
                            "capacity": {"capacity": "fixed"},
                            "validation_rmse": 1.0 if condition == "linear" else 0.5,
                        }
                    ],
                    "checkpoint_sha256": digest,
                    "checkpoint_path": str(checkpoint.resolve()),
                    "inference_config": {},
                    "constructor_identity": None,
                }
            )
    frozen = {
        "selection_split": "validation",
        "split_hash": split_hash,
        "scaler_hash": scaler_hash,
        "git_commit": "c" * 40,
        "dirty_state_digest": "",
        "strongest_baseline": "linear",
        "checkpoints": checkpoints,
    }
    _write_canonical(root / "frozen_models.json", frozen)

    rows = []
    for seed in FORMAL_SEEDS:
        for recording_id in ("test-a", "test-b"):
            rows.extend(
                (
                    _formal_metric_row(
                        seed=seed,
                        recording_id=recording_id,
                        model="linear",
                        checkpoint_sha256=checkpoint_hashes[(seed, "linear")],
                        value=2.0,
                    ),
                    _formal_metric_row(
                        seed=seed,
                        recording_id=recording_id,
                        model="teacher_actual_residual",
                        checkpoint_sha256=checkpoint_hashes[
                            (seed, "teacher_actual_residual")
                        ],
                        value=1.0,
                    ),
                )
            )
    metrics = pd.DataFrame(rows, columns=PER_RECORD_COLUMNS)
    primary = make_primary_rows(
        metrics,
        candidate_model="teacher_actual_residual",
        strongest_baseline="linear",
        required_topologies=("point",),
        required_rates=(0.2,),
        required_scenarios=("handheld",),
    )
    summary, _, coverage = paired_formal_summaries(
        metrics,
        candidate_model="teacher_actual_residual",
        strongest_baseline="linear",
        required_topologies=("point",),
        required_rates=(0.2,),
        required_scenarios=("handheld",),
        required_seeds=FORMAL_SEEDS,
        bootstrap_samples=20,
    )
    gate = success_gate_payload(summary, strongest_baseline="linear")
    ledger = pd.DataFrame(
        [
            {
                "seed": seed,
                "recording_id": recording_id,
                "topology": "point",
                "requested_fraction": 0.2,
                "realized_fraction": 0.2,
                "mask_sha256": hashlib.sha256(
                    f"{seed}:{recording_id}:point:0.2".encode("ascii")
                ).hexdigest(),
            }
            for seed in FORMAL_SEEDS
            for recording_id in ("test-a", "test-b")
        ]
    )
    write_formal_artifacts(
        root,
        pd.concat((metrics, primary), ignore_index=True),
        summary,
        gate,
        ledger,
        coverage,
    )
    return root


def test_validate_completed_formal_root_binds_every_artifact_family(tmp_path):
    root = _make_formal_root(tmp_path)

    report = validate_artifacts(root)

    assert report["kind"] == "formal_root"
    assert report["run_ids"] == []
    assert report["checks"] == {
        "artifact_hashes": True,
        "canonical_manifests": True,
        "checkpoint_hashes": True,
        "config_hashes": True,
        "mask_hashes": True,
        "metrics_hashes": True,
        "scaler_hashes": True,
        "split_hashes": True,
        "window_identities": True,
    }
    assert report["formal"]["split_hash"]
    assert report["formal"]["scaler_hash"]
    assert report["formal"]["strongest_baseline"] == "linear"


@pytest.mark.parametrize(
    "corruption", ("checkpoint", "noncanonical", "extra", "run_id")
)
def test_validate_smoke_rejects_corruption_and_unexpected_artifacts(tmp_path, corruption):
    root, run_dir, _ = _make_smoke_root(tmp_path)
    if corruption == "checkpoint":
        (run_dir / "best.pt").write_bytes(b"tampered")
    elif corruption == "noncanonical":
        value = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        (run_dir / "run.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    elif corruption == "run_id":
        value = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        value["run_id"] = "0" * 16
        _write_canonical(run_dir / "run.json", value)
    else:
        (run_dir / "unexpected.txt").write_text("unexpected", encoding="ascii")

    with pytest.raises(ValueError):
        validate_artifacts(root)


def test_validate_smoke_rejects_symlinked_checkpoint(tmp_path):
    root, run_dir, _ = _make_smoke_root(tmp_path)
    outside = tmp_path / "outside.pt"
    outside.write_bytes((run_dir / "best.pt").read_bytes())
    (run_dir / "best.pt").unlink()
    try:
        (run_dir / "best.pt").symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable for this Windows account: {error}")

    with pytest.raises(ValueError, match="symlink"):
        validate_artifacts(root)


@pytest.mark.parametrize("corruption", ("metric_hash", "completion", "extra", "path_escape"))
def test_validate_formal_rejects_incomplete_tampered_or_escaping_artifacts(
    tmp_path, corruption
):
    root = _make_formal_root(tmp_path)
    if corruption == "metric_hash":
        with (root / "per_record_metrics.csv").open("ab") as handle:
            handle.write(b"tampered\n")
    elif corruption == "completion":
        (root / "artifact_hashes.json").unlink()
    elif corruption == "extra":
        (root / "unregistered.json").write_text("{}\n", encoding="ascii")
    else:
        frozen_path = root / "frozen_models.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"outside")
        frozen["checkpoints"][0]["checkpoint_path"] = str(outside.resolve())
        frozen["checkpoints"][0]["checkpoint_sha256"] = hashlib.sha256(
            outside.read_bytes()
        ).hexdigest()
        _write_canonical(frozen_path, frozen)

    with pytest.raises(ValueError):
        validate_artifacts(root)


def test_validate_artifacts_cli_prints_canonical_json_and_returns_two_on_failure(
    tmp_path, capsys
):
    import imputation_v3.cli as cli

    root, _, manifest = _make_smoke_root(tmp_path)
    assert cli.main(["validate-artifacts", "--output", str(root)]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["run_ids"] == [manifest["run_id"]]
    assert captured.out == canonical_json(payload) + "\n"
    assert captured.err == ""

    (root / manifest["run_id"] / "best.pt").write_bytes(b"tampered")
    assert cli.main(["validate-artifacts", "--output", str(root)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "checkpoint.json" in captured.err
