from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import csv

import pytest
import torch
import numpy as np
import pandas as pd

from imputation_v3.experiments.validate_artifacts import validate_artifacts
from imputation_v3.experiments.runner import (
    FORMAL_SEEDS,
    formal_mask_seed,
    formal_matrix_plan,
    make_primary_rows,
    paired_formal_summaries,
    success_gate_payload,
    write_formal_artifacts,
)
from imputation_v3.config import TeacherConfig
from imputation_v3.experiments.training import run_teacher_smoke
from validation_v2.data.masking import point_missing
from validation_v2.data.splits import MANIFEST_COLUMNS
from validation_v2.evaluation.statistics import PER_RECORD_COLUMNS
from validation_v2.experiments.provenance import (
    canonical_json,
    run_id as provenance_run_id,
)


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes((canonical_json(value) + "\n").encode("utf-8"))


def _make_smoke_root(tmp_path: Path) -> tuple[Path, Path, dict]:
    root = tmp_path / "smoke"
    data_root = tmp_path / "dataset"
    scenario = data_root / "handheld-1"
    scenario.mkdir(parents=True)
    base_s = 1_496_760_699.22
    for recording_index in range(1, 5):
        with (scenario / f"imu{recording_index}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            for sample in range(12):
                value = recording_index + sample / 100.0
                writer.writerow(
                    [
                        base_s + sample * 0.01,
                        0.1, 0.2, 0.3,
                        value, value + 1, value + 2,
                        0.0, 0.0, -1.0,
                        value + 3, value + 4, value + 5,
                        10.0, 20.0, 30.0,
                    ]
                )
        with (scenario / f"vi{recording_index}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            for sample in range(12):
                writer.writerow(
                    [
                        int((base_s + sample * 0.01) * 1e9),
                        sample, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0,
                    ]
                )
    config = TeacherConfig(
        data_root=data_root,
        output_root=root,
        selection_split="validation",
        seeds=(2026,),
        window_seconds=(0.08,),
        nominal_dt_s=0.01,
        batch_size=2,
        epochs=1,
        hidden_size=4,
        tcn_width=4,
        tcn_dilations=(1,),
        learning_rate=0.001,
        training_rates=(0.2,),
        training_topologies=("point", "block"),
        models=("linear", "teacher"),
    )
    repository_root = Path(__file__).resolve().parents[2]
    report = run_teacher_smoke(
        config, repository_root=repository_root, requested_device="cpu"
    )
    run_dir = Path(report["run_dir"])
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
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
    assert root_report["checks"]["metrics_hashes"] is True
    assert root_report["checks"]["history_integrity"] is True
    assert run_report["kind"] == "smoke_run"
    assert run_report["run_ids"] == [manifest["run_id"]]
    after = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


def _formal_metric_row(
    *, seed: int, recording_id: str, model: str, checkpoint_sha256: str, value: float,
    realized_fraction: float = 0.2,
) -> dict:
    return {
        "run_id": f"formal-{seed}-{model}",
        "seed": seed,
        "recording_id": recording_id,
        "scenario": "handheld",
        "protocol": "overall",
        "topology": "point",
        "requested_fraction": 0.2,
        "realized_fraction": realized_fraction,
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

    formal_config = TeacherConfig(
        data_root=Path("Oxford Dataset"),
        output_root=Path("results/imputation_v3/formal"),
        selection_split="validation",
        seeds=FORMAL_SEEDS,
        window_seconds=(1.28,),
        nominal_dt_s=0.01,
        batch_size=2,
        epochs=1,
        hidden_size=16,
        tcn_width=16,
        tcn_dilations=(1, 2),
        learning_rate=0.001,
        training_rates=(0.2,),
        training_topologies=("point",),
        models=("linear", "teacher"),
    )
    config_payload = json.loads(canonical_json(formal_config))
    resolved_payload = {
        "config": config_payload,
        "device": "cpu",
        "output_root": str(root.resolve()),
    }
    config_sha = hashlib.sha256(
        canonical_json(resolved_payload).encode("utf-8")
    ).hexdigest()
    matrix_sha = hashlib.sha256(
        canonical_json(formal_matrix_plan(formal_config)).encode("utf-8")
    ).hexdigest()
    _write_canonical(
        root / "resolved_config.json",
        {
            "schema": "imputation-v3-formal-resolved-config-v1",
            "resolved": resolved_payload,
            "resolved_config_sha256": config_sha,
            "matrix_plan_sha256": matrix_sha,
        },
    )
    window_entries = [
        {
            "seed": seed,
            "context_samples": 128,
            "split": split_name,
            "window_ids_sha256": hashlib.sha256(
                f"{seed}:128:{split_name}".encode("ascii")
            ).hexdigest(),
            "window_count": 4,
        }
        for seed in FORMAL_SEEDS
        for split_name in ("train", "validation")
    ]
    _write_canonical(
        root / "window_identity_ledger.json",
        {
            "schema": "imputation-v3-formal-window-identities-v1",
            "entries": window_entries,
            "entries_sha256": hashlib.sha256(
                canonical_json(window_entries).encode("utf-8")
            ).hexdigest(),
        },
    )
    window_lookup = {
        (entry["seed"], entry["split"]): entry for entry in window_entries
    }
    window_ledger_sha = hashlib.sha256(
        (root / "window_identity_ledger.json").read_bytes()
    ).hexdigest()

    candidates = root / "candidates"
    candidates.mkdir()
    checkpoints = []
    checkpoint_hashes = {}
    conditions = (
        "linear",
        "teacher_actual_residual",
        "teacher_constant_residual",
        "teacher_dt_feature_only_residual",
        "teacher_no_dt_residual",
        "teacher_actual_raw",
    )
    for seed in FORMAL_SEEDS:
        for condition in conditions:
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
                    "train_window_ids_sha256": window_lookup[(seed, "train")][
                        "window_ids_sha256"
                    ],
                    "train_window_count": 4,
                    "validation_window_ids_sha256": window_lookup[
                        (seed, "validation")
                    ]["window_ids_sha256"],
                    "validation_window_count": 4,
                }
            )
    frozen = {
        "selection_split": "validation",
        "split_hash": split_hash,
        "scaler_hash": scaler_hash,
        "git_commit": "c" * 40,
        "dirty_state_digest": "",
        "resolved_config_sha256": config_sha,
        "matrix_plan_sha256": matrix_sha,
        "window_identity_ledger_sha256": window_ledger_sha,
        "strongest_baseline": "linear",
        "checkpoints": checkpoints,
    }
    _write_canonical(root / "frozen_models.json", frozen)

    rows = []
    for seed in FORMAL_SEEDS:
        for recording_id in ("test-a", "test-b"):
            mask_seed = formal_mask_seed(recording_id, seed, "point", 0.2)
            realized_fraction = float(
                (point_missing(torch.zeros((12, 6)), 0.2, mask_seed).mask == 0)
                .double()
                .mean()
            )
            for condition in conditions:
                rows.append(
                    _formal_metric_row(
                        seed=seed,
                        recording_id=recording_id,
                        model=condition,
                        checkpoint_sha256=checkpoint_hashes[(seed, condition)],
                        value=1.0 if condition == "teacher_actual_residual" else 2.0,
                        realized_fraction=realized_fraction,
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
    test_source_hashes = dict(
        zip(
            split.loc[split["split"] == "test", "recording_id"],
            split.loc[split["split"] == "test", "imu_sha256"],
        )
    )
    ledger_rows = []
    for seed in FORMAL_SEEDS:
        for recording_id in ("test-a", "test-b"):
            condition_seed = formal_mask_seed(recording_id, seed, "point", 0.2)
            mask = point_missing(torch.zeros((12, 6)), 0.2, condition_seed).mask
            ledger_rows.append({
                "seed": seed,
                "recording_id": recording_id,
                "topology": "point",
                "requested_fraction": 0.2,
                "realized_fraction": float((mask == 0).double().mean()),
                "mask_sha256": hashlib.sha256(
                    np.ascontiguousarray(mask.numpy()).tobytes()
                ).hexdigest(),
                "generator": "formal-test-mask-v1",
                "condition_seed": condition_seed,
                "target_source_sha256": test_source_hashes[recording_id],
                "target_length": 12,
                "channels": 6,
            })
    ledger = pd.DataFrame(ledger_rows)
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


def test_validate_artifacts_cli_rejects_excessive_json_depth(tmp_path, capsys):
    import imputation_v3.cli as cli

    root, run_dir, _ = _make_smoke_root(tmp_path)
    (run_dir / "run.json").write_text("[" * 80 + "]" * 80, encoding="ascii")

    assert cli.main(["validate-artifacts", "--output", str(root)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "nesting depth" in captured.err


def test_smoke_replay_rejects_source_changed_during_load(tmp_path, monkeypatch):
    import imputation_v3.experiments.validate_artifacts as validator

    root, run_dir, _ = _make_smoke_root(tmp_path)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    selected_id = manifest["config"]["selected_recording_ids"]["train"][0]
    selected = next(
        row for row in manifest["config"]["split_manifest"]
        if row["recording_id"] == selected_id
    )
    selected_imu = Path(selected["imu_path"])
    original = validator.load_recording
    mutated = False

    def mutate_after_load(imu_path, vicon_path):
        nonlocal mutated
        recording = original(imu_path, vicon_path)
        if Path(imu_path) == selected_imu and not mutated:
            mutated = True
            with selected_imu.open("ab") as handle:
                handle.write(b"\n")
        return recording

    monkeypatch.setattr(validator, "load_recording", mutate_after_load)
    with pytest.raises(ValueError, match="changed while|source changed"):
        validator.validate_artifacts(root)


def _reseal_formal_hash(root: Path, name: str) -> None:
    hashes_path = root / "artifact_hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    _write_canonical(hashes_path, hashes)


def _reseal_smoke_after_config_change(run_dir: Path, config: dict) -> Path:
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    manifest["config"] = config
    manifest["config_sha256"] = hashlib.sha256(
        canonical_json(config).encode("utf-8")
    ).hexdigest()
    manifest["run_id"] = provenance_run_id(
        config,
        manifest["seed"],
        manifest["split_hash"],
        manifest["scaler_hash"],
        manifest["git_commit"],
        manifest["dirty_state_digest"],
    )
    new_dir = run_dir.parent / manifest["run_id"]
    run_dir.rename(new_dir)
    _write_canonical(new_dir / "run.json", manifest)
    checkpoint = json.loads((new_dir / "checkpoint.json").read_text(encoding="utf-8"))
    checkpoint["run_id"] = manifest["run_id"]
    _write_canonical(new_dir / "checkpoint.json", checkpoint)
    evidence = json.loads((new_dir / "evidence.json").read_text(encoding="utf-8"))
    evidence.update(
        run_id=manifest["run_id"],
        run_manifest_sha256=hashlib.sha256((new_dir / "run.json").read_bytes()).hexdigest(),
        checkpoint_metadata_sha256=hashlib.sha256(
            (new_dir / "checkpoint.json").read_bytes()
        ).hexdigest(),
    )
    _write_canonical(new_dir / "evidence.json", evidence)
    return new_dir


def test_smoke_mask_cannot_be_forged_with_all_internal_hashes_updated(tmp_path):
    root, run_dir, _ = _make_smoke_root(tmp_path)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    row = config["window_evidence"]["train"][0]
    count = row["samples"] * row["channels"]
    values = list(
        struct.unpack(f"<{count}f", bytes.fromhex(row["mask_bytes_hex"]))
    )
    missing = values.index(0.0)
    observed = values.index(1.0)
    values[missing], values[observed] = values[observed], values[missing]
    forged = struct.pack(f"<{count}f", *values)
    row["mask_bytes_hex"] = forged.hex()
    row["mask_sha256"] = hashlib.sha256(forged).hexdigest()
    _reseal_smoke_after_config_change(run_dir, config)

    with pytest.raises(ValueError, match="replay|window|mask"):
        validate_artifacts(root)


def test_smoke_rmse_cannot_be_forged_with_all_internal_hashes_updated(tmp_path):
    root, run_dir, _ = _make_smoke_root(tmp_path)
    evidence_path = run_dir / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["final_checkpoint_metrics"]["validation"]["missing_rmse"] += 1.0
    evidence["final_checkpoint_metrics_sha256"] = hashlib.sha256(
        canonical_json(evidence["final_checkpoint_metrics"]).encode("utf-8")
    ).hexdigest()
    _write_canonical(evidence_path, evidence)

    with pytest.raises(ValueError, match="RMSE|metric|replay"):
        validate_artifacts(root)


def test_formal_primary_checkpoint_cannot_be_forged_with_internal_rehash(tmp_path):
    root = _make_formal_root(tmp_path)
    metrics_path = root / "per_record_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    target = (metrics["protocol"] == "teacher_primary") & (metrics["model"] == "teacher")
    metrics.loc[target, "checkpoint_sha256"] = "f" * 64
    metrics_path.write_bytes(metrics.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    _reseal_formal_hash(root, "per_record_metrics.csv")

    with pytest.raises(ValueError, match="primary|checkpoint"):
        validate_artifacts(root)


def test_formal_mask_cannot_be_forged_with_internal_rehash(tmp_path):
    root = _make_formal_root(tmp_path)
    ledger_path = root / "mask_ledger.csv"
    ledger = pd.read_csv(ledger_path)
    ledger.loc[0, "mask_sha256"] = "f" * 64
    ledger_path.write_bytes(ledger.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    _reseal_formal_hash(root, "mask_ledger.csv")

    with pytest.raises(ValueError, match="mask"):
        validate_artifacts(root)
