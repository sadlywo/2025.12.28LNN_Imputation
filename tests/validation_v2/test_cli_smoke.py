import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import torch
import yaml
import numpy as np

from validation_v2.experiments.runner import (
    build_execution_model,
    discover_oxiod_pairs,
    resolve_protocol_records,
)
from validation_v2.experiments.provenance import collect_provenance
from validation_v2.types import Recording


REPO_ROOT = Path(__file__).resolve().parents[2]


def _cli(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "validation_v2.cli", *arguments],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )


def test_server_matrix_dry_run_is_byte_stable_and_complete():
    arguments = ("matrix", "--config", "server_full.yaml", "--dry-run")

    first = _cli(*arguments)
    second = _cli(*arguments)

    assert first.returncode == 0, first.stderr.decode()
    assert first.stdout == second.stdout
    lines = first.stdout.decode("utf-8").splitlines()
    header = json.loads(lines[0])
    combinations = [json.loads(line) for line in lines[1:]]
    assert header == {
        "command": "matrix",
        "combination_count": len(combinations),
        "dry_run": True,
    }
    assert {item["seed"] for item in combinations} == {
        2026,
        2027,
        2028,
        2029,
        2030,
    }
    irregular = [item for item in combinations if item["case_type"] == "irregular"]
    assert irregular
    assert all(item["irregular_method"] == "interval_jitter" for item in irregular)
    assert all(item["value_topology"] == "point" for item in irregular)
    assert all(item["value_requested_fraction"] == 0.3 for item in irregular)


def test_server_config_declares_real_scenarios_and_bounded_execution_inputs():
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "validation_v2" / "server_full.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["protocols"] == [
        "strict_file",
        "scenario_holdout:handbag",
        "scenario_holdout:handheld",
        "scenario_holdout:running",
        "scenario_holdout:slow_walking",
        "scenario_holdout:trolley",
        "scenario_holdout:user-2",
    ]
    assert config["irregular_cases"] == [
        {
            "method": "interval_jitter",
            "requested_irregularity": 0.2,
            "value_topology": "point",
            "value_requested_fraction": 0.3,
        }
    ]
    assert config["max_train_windows"] > 0
    assert config["max_eval_samples"] is None
    assert config["split_seed"] == 2026


def test_resolved_provenance_changes_with_hidden_size_or_learning_rate(tmp_path: Path):
    from validation_v2.experiments.runner import resolved_execution_config
    from validation_v2.experiments.train import resume_run, train_one_run

    source = yaml.safe_load(
        (REPO_ROOT / "configs" / "validation_v2" / "smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    source["output_root"] = "ignored/location"
    source["_execution_conditions"] = ["internal"]
    source["_skip_descriptive_summary"] = True
    conditions = [{"topology": "point", "requested_fraction": 0.3}]

    base = resolved_execution_config(
        source, model="linear", seed=2026, protocol="strict_file",
        conditions=conditions, resolved_device="cpu",
    )
    hidden_source = {**source, "hidden_size": source["hidden_size"] + 1}
    learning_source = {**source, "learning_rate": source["learning_rate"] * 2}
    hidden = resolved_execution_config(
        hidden_source, model="linear", seed=2026, protocol="strict_file",
        conditions=conditions, resolved_device="cpu",
    )
    learning = resolved_execution_config(
        learning_source, model="linear", seed=2026, protocol="strict_file",
        conditions=conditions, resolved_device="cpu",
    )

    manifests = [collect_provenance(item, seed=2026) for item in (base, hidden, learning)]
    run_ids = {manifest["run_id"] for manifest in manifests}
    assert len(run_ids) == 3
    assert "output_root" not in base["source_config"]
    assert not any(key.startswith("_") for key in base["source_config"])
    assert base["source_config"]["epochs"] == source["epochs"]
    assert base["source_config"]["batch_size"] == source["batch_size"]
    assert base["evaluation_scope"] == "bounded_overlap_slice"
    server_source = {**source, "max_eval_samples": None}
    server = resolved_execution_config(
        server_source, model="linear", seed=2026, protocol="strict_file",
        conditions=conditions, resolved_device="cpu",
    )
    assert server["evaluation_scope"] == "full_overlap_record"

    model = torch.nn.Linear(1, 1)
    metadata = train_one_run(
        tmp_path / manifests[0]["run_id"],
        manifests[0],
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        train_loader=[],
        validation_loader=[],
        epochs=1,
        train_epoch=lambda *_: {"missing_rmse": 1.0},
        evaluate_epoch=lambda *_: {"missing_rmse": 1.0},
    )
    with pytest.raises(ValueError, match="config hash|run_id"):
        resume_run(
            tmp_path / manifests[0]["run_id"],
            manifests[1],
            metadata["checkpoint_sha256"],
        )


def test_split_seed_freezes_protocol_across_all_training_seeds():
    from validation_v2.experiments.runner import resolve_configured_records

    config = {"split_seed": 2026}
    pairs = discover_oxiod_pairs(REPO_ROOT / "Oxford Dataset")
    manifests = [
        resolve_configured_records(
            config,
            data_root=REPO_ROOT / "Oxford Dataset",
            protocol="strict_file",
            training_seed=training_seed,
        )
        for training_seed in range(2026, 2031)
    ]

    split_views = [
        [(item["recording_id"], item["split"]) for item in manifest]
        for manifest in manifests
    ]
    test_ids = [
        [item["recording_id"] for item in manifest if item["split"] == "test"]
        for manifest in manifests
    ]
    split_hashes = [
        hashlib.sha256(
            json.dumps(view, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for view in split_views
    ]
    assert all(view == split_views[0] for view in split_views[1:])
    assert all(ids == test_ids[0] for ids in test_ids[1:])
    assert len(set(split_hashes)) == 1


def test_reverse_dt_aligns_intervals_to_reversed_input():
    from validation_v2.experiments.runner import reverse_aligned_dt

    dt = torch.tensor([[0.25, 0.1, 0.7, 0.2]], dtype=torch.float64)

    reversed_dt = reverse_aligned_dt(dt)

    torch.testing.assert_close(
        reversed_dt,
        torch.tensor([[0.2, 0.2, 0.7, 0.1]], dtype=torch.float64),
    )


def test_unlimited_overlap_slice_keeps_every_available_sample():
    from validation_v2.experiments.runner import _slice_recording

    imu_time = np.arange(150, dtype=np.float64) * 0.01
    recording = Recording(
        id="synthetic/imu1",
        imu_time_s=imu_time,
        imu_six=np.zeros((150, 6), dtype=np.float64),
        vicon_time_s=np.array([0.2, 1.2], dtype=np.float64),
        vicon_position_m=np.zeros((2, 3), dtype=np.float64),
        vicon_quaternion_xyzw=np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        overlap_s=(0.2, 1.2),
        metadata={},
    )

    unlimited_time, unlimited_values = _slice_recording(recording, None)
    zero_time, zero_values = _slice_recording(recording, 0)

    assert len(unlimited_time) == len(zero_time) == 101
    assert unlimited_values.shape == zero_values.shape == (101, 6)
    assert unlimited_time[0] == pytest.approx(0.2)
    assert unlimited_time[-1] == pytest.approx(1.2)


def _synthetic_training_recording(
    recording_id: str = "synthetic/train", *, length: int = 240
) -> Recording:
    intervals = 0.01 + (np.arange(length, dtype=np.float64) % 5) * 0.001
    time = np.cumsum(intervals)
    values = np.column_stack(
        [np.arange(length, dtype=np.float64) + channel for channel in range(6)]
    )
    return Recording(
        id=recording_id,
        imu_time_s=time,
        imu_six=values,
        vicon_time_s=np.array([time[0], time[-1]], dtype=np.float64),
        vicon_position_m=np.zeros((2, 3), dtype=np.float64),
        vicon_quaternion_xyzw=np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64
        ),
        overlap_s=(float(time[0]), float(time[-1])),
        metadata={},
    )


@pytest.mark.parametrize("topology", ["point", "block", "channel"])
@pytest.mark.parametrize("rate", [0.1, 0.2, 0.3, 0.4])
def test_every_training_window_has_requested_missingness(topology: str, rate: float):
    from validation_v2.data.normalization import RobustTrainScaler
    from validation_v2.experiments.runner import _windows

    recording = _synthetic_training_recording()
    scaler = RobustTrainScaler(
        center_=np.zeros(6), scale_=np.ones(6), training_ids=(recording.id,)
    )

    windows = _windows(
        [recording], scaler, seq_len=30, maximum_windows=6,
        rate=rate, seed=2026, topology=topology,
    )

    assert len(windows) == 6
    realized = [float((window.mask == 0).float().mean()) for window in windows]
    assert all(value > 0 for value in realized)
    if topology in {"point", "block"}:
        expected = round(30 * (6 if topology == "point" else 1) * rate) / (
            30 * (6 if topology == "point" else 1)
        )
    else:
        expected = max(1, int(6 * rate)) / 6
    assert realized == pytest.approx([expected] * 6)


def test_training_windows_preserve_timing_boundaries_and_hidden_target_invariance():
    from validation_v2.data.normalization import RobustTrainScaler
    from validation_v2.experiments.runner import _windows

    first = _synthetic_training_recording("synthetic/first", length=60)
    second = _synthetic_training_recording("synthetic/second", length=60)
    second = Recording(
        **{**second.__dict__, "imu_six": second.imu_six + 1000.0}
    )
    scaler = RobustTrainScaler(
        center_=np.zeros(6), scale_=np.ones(6),
        training_ids=tuple(sorted((first.id, second.id))),
    )
    windows = _windows(
        [first, second], scaler, seq_len=30, maximum_windows=4,
        rate=0.1, seed=2026, topology="block",
    )

    expected_dt = []
    for recording in (first, second):
        dt = np.empty(60, dtype=np.float32)
        dt[1:] = np.diff(recording.imu_time_s).astype(np.float32)
        dt[0] = dt[1]
        expected_dt.extend((dt[:30], dt[30:]))
    for window, dt in zip(windows, expected_dt):
        np.testing.assert_allclose(window.dt.numpy(), dt)
    assert all(float(window.target.mean()) < 500 for window in windows[:2])
    assert all(float(window.target.mean()) > 500 for window in windows[2:])

    changed_values = first.imu_six.copy()
    first_masks = torch.cat([window.mask for window in windows[:2]]).numpy()
    changed_values[first_masks == 0] += 10_000.0
    changed = Recording(**{**first.__dict__, "imu_six": changed_values})
    changed_windows = _windows(
        [changed], scaler, seq_len=30, maximum_windows=2,
        rate=0.1, seed=2026, topology="block",
    )
    for original, hidden_changed in zip(windows[:2], changed_windows):
        assert torch.equal(original.mask, hidden_changed.mask)
        assert torch.equal(original.features, hidden_changed.features)


def test_formal_like_block_training_callback_has_missing_targets_in_every_batch():
    from validation_v2.data.normalization import RobustTrainScaler
    from validation_v2.experiments.runner import _batches, _epoch_callbacks, _windows

    recording = _synthetic_training_recording()
    scaler = RobustTrainScaler(
        center_=np.zeros(6), scale_=np.ones(6), training_ids=(recording.id,)
    )
    batches = _batches(
        _windows(
            [recording], scaler, seq_len=30, maximum_windows=6,
            rate=0.1, seed=2026, topology="block",
        ),
        batch_size=2,
    )
    model = build_execution_model("bilnn", hidden_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_epoch, _ = _epoch_callbacks("bilnn", torch.device("cpu"))

    assert len(batches) == 3
    assert all(torch.any(batch.mask == 0) for batch in batches)
    metrics = train_epoch(model, optimizer, batches, epoch=1)
    assert np.isfinite(metrics["missing_rmse"])


def test_stitched_neural_prediction_windows_cover_and_average_every_sample():
    from validation_v2.experiments.runner import predict_stitched_sequence

    class WindowPositionSpy:
        name = "bilstm"

        def __init__(self):
            self.lengths = []

        def predict(self, features, mask, dt, reported_model=None):
            del mask, dt, reported_model
            self.lengths.append(features.shape[1])
            local = torch.arange(
                features.shape[1], dtype=features.dtype, device=features.device
            )
            return local[None, :, None].expand(features.shape[0], -1, 6)

    spy = WindowPositionSpy()
    length, seq_len = 13, 4
    features = torch.zeros(length, 25)
    mask = torch.zeros(length, 6)
    dt = torch.full((length,), 0.1)

    prediction, coverage = predict_stitched_sequence(
        spy, features, mask, dt, seq_len=seq_len, batch_size=2,
        return_coverage=True,
    )

    starts = [0, 2, 4, 6, 8, 9]
    expected_sum = torch.zeros(length)
    expected_count = torch.zeros(length)
    for start in starts:
        expected_sum[start : start + seq_len] += torch.arange(seq_len)
        expected_count[start : start + seq_len] += 1
    assert prediction.shape == (length, 6)
    torch.testing.assert_close(prediction[:, 0], expected_sum / expected_count)
    torch.testing.assert_close(coverage, expected_count)
    assert torch.all(coverage > 0)
    assert spy.lengths and max(spy.lengths) == seq_len


def test_irregular_linear_signal_is_resampled_at_jittered_timestamps():
    from validation_v2.experiments.runner import resample_physical_time

    source_time = np.array([0.0, 0.2, 0.5, 0.7, 1.0])
    physical = np.column_stack(
        [(axis + 1) * source_time for axis in range(6)]
    )
    query_time = np.array([0.0, 0.1, 0.4, 0.85, 1.0])

    resampled = resample_physical_time(source_time, physical, query_time)

    np.testing.assert_allclose(
        resampled,
        np.column_stack([(axis + 1) * query_time for axis in range(6)]),
    )
    assert query_time[0] == source_time[0]
    assert query_time[-1] == source_time[-1]


def test_matrix_dry_run_ignores_mapping_and_axis_list_order(tmp_path: Path):
    first_config = tmp_path / "first.yaml"
    second_config = tmp_path / "second.yaml"
    first_config.write_text(
        "models: [hybrid, linear]\n"
        "seeds: [2027, 2026]\n"
        "topologies: [block, point]\n"
        "rates: [0.3, 0.1]\n"
        "protocols: [strict_file]\n",
        encoding="utf-8",
    )
    second_config.write_text(
        "protocols: [strict_file]\n"
        "rates: [0.1, 0.3]\n"
        "topologies: [point, block]\n"
        "seeds: [2026, 2027]\n"
        "models: [linear, hybrid]\n",
        encoding="utf-8",
    )

    first = _cli("matrix", "--config", str(first_config), "--dry-run")
    second = _cli("matrix", "--config", str(second_config), "--dry-run")

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout


def test_matrix_bad_config_has_clear_stderr_and_nonzero_exit(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: [linear]\n", encoding="utf-8")

    result = _cli("matrix", "--config", str(bad), "--dry-run")

    assert result.returncode != 0
    assert b"missing matrix axes" in result.stderr
    assert result.stdout == b""


def test_smoke_config_declares_the_bounded_real_oxiod_protocol():
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "validation_v2" / "smoke.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["data_root"] == "Oxford Dataset"
    assert config["output_root"] == "results/validation_v2/smoke"
    assert config["seeds"] == [2026]
    assert (config["epochs"], config["batch_size"], config["seq_len"]) == (1, 4, 30)
    assert config["models"] == ["linear", "bilstm", "hybrid"]
    assert config["topologies"] == ["point"]
    assert config["rates"] == [0.3]
    assert config["max_train_windows"] > 0
    assert config["max_eval_samples"] > 0
    splits = config["recordings"]
    assert [record["split"] for record in splits].count("train") == 2
    assert [record["split"] for record in splits].count("validation") == 1
    assert [record["split"] for record in splits].count("test") == 1
    assert [(record["imu"], record["vicon"]) for record in splits] == [
        (f"handbag-1/imu{index}.csv", f"handbag-1/vi{index}.csv")
        for index in range(1, 5)
    ]


def test_real_smoke_writes_frozen_runs_and_descriptive_summary(tmp_path: Path):
    data_root = REPO_ROOT / "Oxford Dataset"
    if not (data_root / "handbag-1" / "imu1.csv").is_file():
        pytest.skip("real OxIOD files are not available")
    output_root = tmp_path / "real-smoke"

    result = _cli(
        "smoke",
        "--config",
        "smoke.yaml",
        "--output-root",
        str(output_root),
        "--device",
        "cpu",
    )

    assert result.returncode == 0, result.stderr.decode()
    report = json.loads(result.stdout)
    assert report["status"] == "completed"
    assert report["real_data"] is True
    assert report["descriptive_only"] is True
    assert report["n_recordings"] == 1
    manifest = pd.read_csv(output_root / "split_manifest.csv")
    assert manifest["split"].value_counts().to_dict() == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }
    scaler = json.loads((output_root / "scaler.json").read_text(encoding="utf-8"))
    assert scaler["training_ids"] == ["handbag-1/imu1", "handbag-1/imu2"]
    assert scaler["channel_order"] == [
        "rotation_rate_x",
        "rotation_rate_y",
        "rotation_rate_z",
        "user_acc_x",
        "user_acc_y",
        "user_acc_z",
    ]
    assert scaler["split_hash"] == report["split_hash"]
    run_dirs = sorted(path.parent for path in output_root.glob("*/run.json"))
    assert len(run_dirs) == 3
    all_models = set()
    for run_dir in run_dirs:
        for name in (
            "run.json",
            "history.json",
            "best.pt",
            "checkpoint.json",
            "test_evaluation.json",
            "per_record_metrics.csv",
        ):
            assert (run_dir / name).is_file(), f"missing {name} in {run_dir.name}"
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        all_models.add(run["config"]["model"])
        metrics = pd.read_csv(run_dir / "per_record_metrics.csv")
        assert set(metrics["recording_id"]) == {"handbag-1/imu4"}
        assert {
            "reconstruction_normalized",
            "reconstruction_physical",
            "ate_rmse_m",
            "rpe_rmse_m",
            "endpoint_drift_m",
            "velocity_rmse_mps",
        }.issubset(set(metrics["metric"]))
        assert any(metrics["metric"].str.startswith("delta_"))
    assert all_models == {"linear", "bilstm", "hybrid"}
    smoke_summary = json.loads(
        (output_root / "smoke_summary.json").read_text(encoding="utf-8")
    )
    assert smoke_summary["descriptive_only"] is True
    assert smoke_summary["n_recordings"] == 1
    assert (output_root / "summary.csv").is_file()
    assert (output_root / "summary.json").is_file()


def test_matrix_explicit_limit_runs_real_cell_and_marks_partial(tmp_path: Path):
    if not (REPO_ROOT / "Oxford Dataset" / "handbag-1" / "imu1.csv").is_file():
        pytest.skip("real OxIOD files are not available")
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "validation_v2" / "smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["models"] = ["linear"]
    config["rates"] = [0.3, 0.4]
    config_path = tmp_path / "mini-matrix.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_root = tmp_path / "matrix-output"

    result = _cli(
        "matrix",
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
        "--device",
        "cpu",
        "--max-combinations",
        "1",
    )

    assert result.returncode == 0, result.stderr.decode()
    report = json.loads(result.stdout)
    assert report["partial"] is True
    assert report["selected_cells"] == 1
    assert report["total_cells"] == 2
    marker = json.loads(
        (output_root / "matrix_execution.json").read_text(encoding="utf-8")
    )
    assert marker["partial"] is True
    assert len(list(output_root.glob("*/per_record_metrics.csv"))) == 1

    summary = _cli("summarize", "--root", str(output_root), "--baseline", "linear")
    assert summary.returncode != 0
    assert b"partial" in summary.stderr


def test_server_scan_finds_all_real_pairs_and_normalizes_scenarios():
    pairs = discover_oxiod_pairs(REPO_ROOT / "Oxford Dataset")

    assert len(pairs) == 45
    assert len({pair["recording_id"] for pair in pairs}) == 45
    assert {pair["scenario"] for pair in pairs} == {
        "handbag",
        "handheld",
        "running",
        "slow_walking",
        "trolley",
        "user-2",
    }
    assert all(Path(pair["imu_path"]).is_file() for pair in pairs)
    assert all(Path(pair["vicon_path"]).is_file() for pair in pairs)


def test_server_protocol_splits_are_disjoint_and_hold_out_complete_scenario():
    pairs = discover_oxiod_pairs(REPO_ROOT / "Oxford Dataset")

    strict = resolve_protocol_records(pairs, "strict_file", seed=2026)
    holdout = resolve_protocol_records(
        pairs, "scenario_holdout:handbag", seed=2026
    )

    assert len(strict) == len(holdout) == 45
    assert set(item["split"] for item in strict) == {"train", "validation", "test"}
    assert len({item["recording_id"] for item in strict}) == 45
    assert all(
        (item["split"] == "test") == (item["scenario"] == "handbag")
        for item in holdout
    )
    assert {item["split"] for item in holdout if item["scenario"] != "handbag"} == {
        "train",
        "validation",
    }


@pytest.mark.parametrize(
    "model_name",
    [
        "linear",
        "locf",
        "bilstm",
        "bilnn",
        "hybrid",
        "equal_average",
        "fixed_gate_0",
        "fixed_gate_0.5",
        "fixed_gate_1",
    ],
)
def test_every_server_model_constructs_and_forwards(model_name: str):
    model = build_execution_model(model_name, hidden_size=2)
    features = torch.zeros(1, 6, 25)
    target = torch.zeros(1, 6, 6)
    mask = torch.ones_like(target)
    mask[:, 2, :] = 0
    dt = torch.full((1, 6), 0.01)

    prediction = model.predict(features, mask, dt)

    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()


def test_gate_labels_share_identical_hybrid_branch_predictions():
    from validation_v2.models.hybrid import HybridComponents

    class BranchSpy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward_components(self, features, forward_dt, reverse_dt, observed, mask):
            self.inputs.append(
                (features.clone(), forward_dt.clone(), reverse_dt.clone(), mask.clone())
            )
            lnn = torch.full_like(observed, 2.0)
            lstm = torch.full_like(observed, 4.0)
            gate = torch.full_like(observed, 0.25)
            raw = gate * lnn + (1.0 - gate) * lstm
            return HybridComponents(lnn, lstm, gate, raw, raw)

    model = build_execution_model("hybrid", hidden_size=2)
    spy = BranchSpy()
    model.core = spy
    features = torch.zeros(1, 6, 25)
    mask = torch.zeros(1, 6, 6)
    dt = torch.full((1, 6), 0.1)
    labels = [
        "hybrid", "equal_average", "fixed_gate_0", "fixed_gate_0.5", "fixed_gate_1"
    ]

    predictions = {
        label: model.predict(features, mask, dt, reported_model=label)
        for label in labels
    }

    assert len(spy.inputs) == 5
    for inputs in spy.inputs[1:]:
        for actual, expected in zip(inputs, spy.inputs[0]):
            torch.testing.assert_close(actual, expected)
    assert predictions["hybrid"][0, 0, 0].item() == pytest.approx(3.5)
    assert predictions["equal_average"][0, 0, 0].item() == pytest.approx(3.0)
    assert predictions["fixed_gate_0"][0, 0, 0].item() == pytest.approx(4.0)
    assert predictions["fixed_gate_0.5"][0, 0, 0].item() == pytest.approx(3.0)
    assert predictions["fixed_gate_1"][0, 0, 0].item() == pytest.approx(2.0)


def test_complete_mini_matrix_groups_gate_family_under_one_checkpoint(tmp_path: Path):
    if not (REPO_ROOT / "Oxford Dataset" / "handbag-1" / "imu1.csv").is_file():
        pytest.skip("real OxIOD files are not available")
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "validation_v2" / "smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    gate_models = [
        "hybrid", "equal_average", "fixed_gate_0", "fixed_gate_0.5", "fixed_gate_1"
    ]
    config["models"] = gate_models
    config["rates"] = [0.3]
    config["irregular_cases"] = [
        {
            "method": "interval_jitter",
            "requested_irregularity": 0.2,
            "value_topology": "point",
            "value_requested_fraction": 0.3,
        }
    ]
    config_path = tmp_path / "complete-mini.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_root = tmp_path / "complete-output"

    result = _cli(
        "matrix", "--config", str(config_path), "--output-root", str(output_root),
        "--device", "cpu",
    )

    assert result.returncode == 0, result.stderr.decode()
    marker = json.loads(result.stdout)
    assert marker["partial"] is False
    assert marker["selected_cells"] == 10
    assert marker["training_groups"] == 1
    assert marker["grouping_key"] == [
        "training_family", "seed", "protocol", "objective"
    ]
    metrics_paths = list(output_root.glob("*/per_record_metrics.csv"))
    assert len(metrics_paths) == 1
    metrics = pd.read_csv(metrics_paths[0])
    assert set(metrics["model"]) == set(gate_models)
    assert metrics["checkpoint_sha256"].nunique() == 1
    assert set(metrics["requested_fraction"]) == {0.3}
    assert "irregular:interval_jitter+point" in set(metrics["topology"])
    irregular_metrics = metrics.loc[
        metrics["topology"] == "irregular:interval_jitter+point"
    ]
    assert set(irregular_metrics["metric"]).issuperset(
        {"irregularity_requested", "irregularity_realized"}
    )
    requested = irregular_metrics.loc[
        irregular_metrics["metric"] == "irregularity_requested", "value"
    ]
    realized = irregular_metrics.loc[
        irregular_metrics["metric"] == "irregularity_realized", "value"
    ]
    assert set(requested) == {0.2}
    assert set(realized) != {0.2}
