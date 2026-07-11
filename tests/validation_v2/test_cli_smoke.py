import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import torch
import yaml

from validation_v2.experiments.runner import (
    build_execution_model,
    discover_oxiod_pairs,
    resolve_protocol_records,
)


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
    assert config["max_eval_samples"] > 0


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


def test_complete_mini_matrix_groups_two_cells_under_one_checkpoint(tmp_path: Path):
    if not (REPO_ROOT / "Oxford Dataset" / "handbag-1" / "imu1.csv").is_file():
        pytest.skip("real OxIOD files are not available")
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "validation_v2" / "smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["models"] = ["linear"]
    config["rates"] = [0.3, 0.4]
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
    assert marker["selected_cells"] == 3
    assert marker["training_groups"] == 1
    metrics_paths = list(output_root.glob("*/per_record_metrics.csv"))
    assert len(metrics_paths) == 1
    metrics = pd.read_csv(metrics_paths[0])
    assert set(metrics["requested_fraction"]) == {0.3, 0.4}
    assert "irregular:interval_jitter+point" in set(metrics["topology"])
