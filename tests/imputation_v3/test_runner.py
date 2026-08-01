from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from imputation_v3.config import load_teacher_config
from imputation_v3.experiments.evaluate import (
    aggregate_raw_windows,
    diagnostic_masks,
    evaluate_record,
    evaluate_record_diagnostics,
    physical_record_metrics,
)
from imputation_v3.experiments.runner import (
    FORMAL_SEEDS,
    build_native_model,
    evaluate_record_rows,
    formal_matrix_plan,
    make_primary_rows,
    run_formal_protocol,
    success_gate_payload,
    teacher_success,
    write_formal_artifacts,
)
from imputation_v3.models.native_controls import (
    BiCfCControl,
    BiLSTMControl,
    FeatureMLPControl,
    TCNControl,
    count_parameters,
)
from imputation_v3.models.teacher import OfflineTeacher
from validation_v2.evaluation.statistics import PER_RECORD_COLUMNS


ROOT = Path(__file__).resolve().parents[2]


def _metric_row(**changes):
    row = {
        "run_id": "run",
        "seed": 2026,
        "recording_id": "r1",
        "scenario": "handheld",
        "protocol": "overall",
        "topology": "block",
        "requested_fraction": 0.2,
        "realized_fraction": 0.25,
        "model": "teacher_actual_residual",
        "metric": "rmse_physical",
        "value": 2.0,
        "checkpoint_sha256": "a" * 64,
    }
    row.update(changes)
    return row


def test_overlap_aggregation_precedes_inverse_scaling_and_physical_metrics():
    windows = [
        np.array([[0.0], [2.0], [4.0]]),
        np.array([[6.0], [8.0], [10.0]]),
    ]
    stitched = aggregate_raw_windows(windows, starts=[0, 1], recording_length=4)
    np.testing.assert_allclose(stitched[:, 0], [0.0, 4.0, 6.0, 10.0])

    scaler = SimpleNamespace(
        inverse_transform=lambda value: np.asarray(value) * 2.0 + 10.0
    )
    rows = evaluate_record(
        raw_windows=windows,
        starts=[0, 1],
        target_normalized=np.array([[0.0], [3.0], [7.0], [10.0]]),
        observed_mask=np.array([[1], [0], [0], [1]], dtype=np.uint8),
        scaler=scaler,
        recording_id="r1",
    )
    assert rows == {"recording_id": "r1", "rmse_physical": 2.0, "mae_physical": 2.0}


@pytest.mark.parametrize(
    ("windows", "starts", "length", "error"),
    (
        ([], [], 4, "non-empty"),
        ([np.ones((2, 1))], [], 4, "aligned"),
        ([np.ones((2, 1))], [-1], 4, "outside"),
        ([np.ones((2, 1))], [3], 4, "outside"),
        ([np.ones((2, 1)), np.ones((2, 2))], [0, 2], 4, "channels"),
        ([np.ones((2, 1))], [0], 4, "coverage"),
        ([np.array([[math.nan], [1.0]])], [0], 2, "finite"),
    ),
)
def test_overlap_aggregation_rejects_malformed_or_incomplete_inputs(
    windows, starts, length, error
):
    with pytest.raises((TypeError, ValueError), match=error):
        aggregate_raw_windows(windows, starts=starts, recording_length=length)


def test_diagnostic_masks_use_irregular_timestamps_for_gap_duration():
    missing = np.zeros((4, 6), dtype=bool)
    missing[2, 0] = True
    missing[1:3, 3] = True
    masks = diagnostic_masks(missing, np.array([0.0, 0.01, 0.02, 0.20]))

    assert set(masks) >= {
        "overall",
        "sensor/gyro",
        "sensor/accelerometer",
        "axis/gx",
        "axis/ax",
        "gap/50-200ms",
    }
    assert masks["gap/50-200ms"][2, 0]
    assert masks["gap/50-200ms"][1:3, 3].all()
    assert not masks.get("gap/0-50ms", np.zeros_like(missing))[2, 0]


@pytest.mark.parametrize(
    ("missing", "time", "error"),
    (
        (np.zeros((2, 5)), np.array([0.0, 0.1]), "six"),
        (np.zeros((2, 6)), np.array([0.0]), "length"),
        (np.zeros((2, 6)), np.array([0.1, 0.0]), "increasing"),
        (np.zeros((2, 6)), np.array([0.0, math.nan]), "finite"),
        (np.full((2, 6), 0.5), np.array([0.0, 0.1]), "binary"),
    ),
)
def test_diagnostic_masks_validate_shapes_values_and_time(missing, time, error):
    with pytest.raises((TypeError, ValueError), match=error):
        diagnostic_masks(missing, time)


def test_physical_metrics_validate_finite_selected_values_and_mask():
    prediction = np.array([[0.0], [2.0], [4.0]])
    target = np.array([[0.0], [1.0], [2.0]])
    assert physical_record_metrics(
        prediction=prediction,
        target=target,
        missing=np.array([[False], [True], [True]]),
        recording_id="r1",
    ) == pytest.approx(
        {"recording_id": "r1", "rmse_physical": math.sqrt(2.5), "mae_physical": 1.5}
    )
    bad = prediction.copy()
    bad[1, 0] = math.inf
    with pytest.raises(ValueError, match="finite"):
        physical_record_metrics(
            prediction=bad,
            target=target,
            missing=np.array([[0], [1], [1]]),
            recording_id="r1",
        )
    with pytest.raises(ValueError, match="binary"):
        physical_record_metrics(
            prediction=prediction,
            target=target,
            missing=np.array([[0.0], [0.5], [1.0]]),
            recording_id="r1",
        )


def test_physical_metrics_use_overflow_stable_rmse():
    rows = physical_record_metrics(
        prediction=np.array([[1e308], [1e308]]),
        target=np.zeros((2, 1)),
        missing=np.ones((2, 1), dtype=bool),
        recording_id="r1",
    )
    assert rows["rmse_physical"] == pytest.approx(1e308)


def test_evaluate_record_diagnostics_inverse_scales_once_and_scores_all_groups():
    class CountingScaler:
        def __init__(self):
            self.calls = 0

        def inverse_transform(self, value):
            self.calls += 1
            return np.asarray(value) * 2

    scaler = CountingScaler()
    target = np.zeros((4, 6))
    missing = np.zeros_like(target, dtype=bool)
    missing[1:3] = True
    metrics = evaluate_record_diagnostics(
        raw_windows=[np.ones((4, 6))],
        starts=[0],
        target_normalized=target,
        observed_mask=~missing,
        time=np.array([0.0, 0.03, 0.07, 0.10]),
        scaler=scaler,
        recording_id="r1",
    )
    assert scaler.calls == 2
    assert set(metrics) >= {
        "overall",
        "sensor/gyro",
        "sensor/accelerometer",
        "axis/gx",
        "gap/50-200ms",
    }
    assert metrics["overall"] == {
        "recording_id": "r1",
        "rmse_physical": 2.0,
        "mae_physical": 2.0,
    }


def test_runner_converts_diagnostics_to_exact_validation_v2_long_schema():
    target = np.zeros((4, 6))
    observed = np.ones_like(target, dtype=bool)
    observed[1:3] = False
    rows = evaluate_record_rows(
        raw_windows=[np.ones((4, 6))],
        starts=[0],
        target_normalized=target,
        observed_mask=observed,
        time=np.array([0.0, 0.03, 0.07, 0.10]),
        scaler=SimpleNamespace(inverse_transform=lambda value: np.asarray(value) * 2),
        run_id="run",
        seed=2026,
        recording_id="r1",
        scenario="handheld",
        topology="block",
        requested_fraction=0.2,
        realized_fraction=0.5,
        model="teacher_actual_residual",
        checkpoint_sha256="a" * 64,
    )
    assert list(rows.columns) == list(PER_RECORD_COLUMNS)
    assert set(rows["metric"]) == {"rmse_physical", "mae_physical"}
    assert {"overall", "sensor/gyro", "axis/gx", "gap/50-200ms"} <= set(
        rows["protocol"]
    )
    assert (rows.loc[rows.metric == "rmse_physical", "value"] == 2.0).all()


def test_teacher_success_requires_exact_finite_preregistered_primary_row():
    summary = pd.DataFrame(
        [
            {
                "scenario": "all",
                "protocol": "teacher_primary",
                "topology": "all",
                "requested_fraction": 0.2,
                "model": "teacher",
                "baseline": "saits",
                "metric": "rmse_physical",
                "ci95_low": -0.3,
                "ci95_high": -0.1,
            },
            {
                "scenario": "all",
                "protocol": "teacher_primary",
                "topology": "all",
                "requested_fraction": 0.2,
                "model": "teacher",
                "baseline": "linear",
                "metric": "rmse_physical",
                "ci95_low": -0.4,
                "ci95_high": -0.2,
            },
        ]
    )
    assert teacher_success(summary, strongest_baseline="saits") is True
    summary.loc[0, "ci95_high"] = 0.0
    assert teacher_success(summary, strongest_baseline="saits") is False
    summary.loc[0, "ci95_high"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        teacher_success(summary, strongest_baseline="saits")
    duplicate = pd.concat((summary.iloc[[1]], summary.iloc[[1]]), ignore_index=True)
    with pytest.raises(ValueError, match="exactly one"):
        teacher_success(duplicate, strongest_baseline="linear")
    malformed = summary.iloc[[1]].copy()
    malformed["ci95_low"] = ["-0.4"]
    with pytest.raises((TypeError, ValueError), match="numeric"):
        teacher_success(malformed, strongest_baseline="linear")


def test_success_gate_payload_is_exact_and_boolean():
    summary = pd.DataFrame(
        [{"model": "teacher", "baseline": "saits", "metric": "rmse_physical", "ci95_low": -1.0, "ci95_high": -0.1}]
    )
    assert success_gate_payload(summary, strongest_baseline="saits") == {
        "candidate": "teacher",
        "strongest_baseline": "saits",
        "metric": "rmse_physical",
        "criterion": "paired_ci95_high_below_zero",
        "passed": True,
        "next_stage": "plan_fixed_lag_students",
    }


@pytest.mark.parametrize(
    ("condition", "model_type"),
    (
        ("bilstm", BiLSTMControl),
        ("bilnn", BiCfCControl),
        ("tcn", TCNControl),
        ("teacher_actual_residual", OfflineTeacher),
        ("teacher_constant_residual", OfflineTeacher),
        ("teacher_dt_feature_only_residual", OfflineTeacher),
        ("teacher_no_dt_residual", OfflineTeacher),
        ("teacher_actual_raw", OfflineTeacher),
    ),
)
def test_native_model_factory_uses_frozen_explicit_conditions(condition, model_type):
    config = load_teacher_config(ROOT / "configs/imputation_v3/teacher_smoke.yaml")
    model = build_native_model(condition, config)
    assert isinstance(model, model_type)
    if condition.startswith("teacher_"):
        expected = {
            "teacher_actual_residual": ("actual", "residual"),
            "teacher_constant_residual": ("constant", "residual"),
            "teacher_dt_feature_only_residual": ("dt_feature_only", "residual"),
            "teacher_no_dt_residual": ("no_dt", "residual"),
            "teacher_actual_raw": ("actual", "raw"),
        }[condition]
        assert (model.time_mode, model.residual_mode) == expected


def test_feature_mlp_factory_selects_closest_frozen_capacity():
    config = load_teacher_config(ROOT / "configs/imputation_v3/teacher_smoke.yaml")
    model = build_native_model("feature_mlp", config)
    teacher = OfflineTeacher(31, config.hidden_size, config.tcn_width, config.tcn_dilations)
    distances = {
        width: abs(
            count_parameters(FeatureMLPControl(31, width))
            - count_parameters(teacher)
        )
        for width in (32, 48, 64, 96, 128, 192)
    }
    assert isinstance(model, FeatureMLPControl)
    assert model.representation_size == min(distances, key=distances.get)
    with pytest.raises(ValueError, match="unsupported native condition"):
        build_native_model("teacher", config)


def test_formal_matrix_plan_expands_aliases_and_counts_without_data_access():
    smoke = load_teacher_config(ROOT / "configs/imputation_v3/teacher_smoke.yaml")
    report = formal_matrix_plan(smoke)
    assert report["counts"] == {
        "seeds": 1,
        "contexts": 1,
        "models": 2,
        "conditions_per_seed_context": 6,
        "matrix_cells": 6,
    }
    assert report["test_data_accessed"] is False
    assert [cell["condition"] for cell in report["cells"]] == [
        "linear",
        "teacher_actual_residual",
        "teacher_constant_residual",
        "teacher_dt_feature_only_residual",
        "teacher_no_dt_residual",
        "teacher_actual_raw",
    ]
    full = load_teacher_config(ROOT / "configs/imputation_v3/teacher_full.yaml")
    assert formal_matrix_plan(full)["counts"]["matrix_cells"] == 240


def test_matrix_rejects_duplicate_aliases_and_contexts():
    config = load_teacher_config(ROOT / "configs/imputation_v3/teacher_smoke.yaml")
    with pytest.raises(ValueError, match="models.*unique"):
        formal_matrix_plan(replace(config, models=("teacher", "teacher")))
    with pytest.raises(ValueError, match="contexts.*unique"):
        formal_matrix_plan(replace(config, window_seconds=(1.28, 1.28)))


def test_primary_rows_root_mean_condition_squared_errors_and_exact_schema():
    frame = pd.DataFrame(
        [
            _metric_row(value=3.0),
            _metric_row(model="teacher_constant_residual", value=4.0),
            _metric_row(model="saits", value=2.0),
        ],
        columns=PER_RECORD_COLUMNS,
    )
    primary = make_primary_rows(
        frame,
        teacher_conditions=("teacher_actual_residual", "teacher_constant_residual"),
        strongest_baseline="saits",
    )
    assert list(primary.columns) == list(PER_RECORD_COLUMNS)
    assert primary["model"].tolist() == ["saits", "teacher"]
    assert primary.loc[primary.model == "teacher", "value"].item() == pytest.approx(
        math.sqrt(12.5)
    )
    assert set(primary["scenario"]) == {"all"}
    assert set(primary["protocol"]) == {"teacher_primary"}
    assert set(primary["topology"]) == {"all"}


def test_primary_rows_use_overflow_stable_condition_rms():
    frame = pd.DataFrame(
        [
            _metric_row(value=1e308),
            _metric_row(model="teacher_constant_residual", value=1e308),
            _metric_row(model="saits", value=1e308),
        ],
        columns=PER_RECORD_COLUMNS,
    )
    primary = make_primary_rows(
        frame,
        teacher_conditions=("teacher_actual_residual", "teacher_constant_residual"),
        strongest_baseline="saits",
    )
    assert np.isfinite(primary["value"]).all()
    np.testing.assert_allclose(primary["value"].to_numpy(), 1e308)


class _SpyBackend:
    def __init__(self, metrics):
        self.events = []
        self.metrics = metrics

    def load_frozen_manifest_and_scaler(self):
        self.events.append("load_assets")
        return "assets"

    def materialize_train_validation(self, assets, plan):
        assert assets == "assets"
        self.events.append("materialize_train_validation")
        return "windows"

    def train_select_validation(self, windows, plan):
        assert windows == "windows"
        self.events.append("train_select_validation")
        return "selected"

    def freeze_checkpoints(self, selected, plan):
        assert selected == "selected"
        self.events.append("freeze_checkpoints")
        return "frozen"

    def load_test_data(self, assets):
        assert self.events[-1] == "freeze_checkpoints"
        self.events.append("load_test_data")
        return "test"

    def evaluate_test_once(self, frozen, test_data, plan):
        assert (frozen, test_data) == ("frozen", "test")
        self.events.append("evaluate_test_once")
        return {
            "per_record_metrics": self.metrics,
            "mask_ledger": pd.DataFrame([{"mask_id": "m1"}]),
            "strongest_baseline": "saits",
        }


def test_formal_protocol_order_never_loads_test_before_freeze(tmp_path, monkeypatch):
    config = load_teacher_config(ROOT / "configs/imputation_v3/teacher_full.yaml")
    rows = []
    for seed in FORMAL_SEEDS:
        for recording, teacher_value, baseline_value in (("r1", 1.0, 2.0), ("r2", 1.5, 2.5)):
            rows.extend(
                [
                    _metric_row(seed=seed, recording_id=recording, value=teacher_value),
                    _metric_row(seed=seed, recording_id=recording, model="saits", value=baseline_value),
                ]
            )
    backend = _SpyBackend(pd.DataFrame(rows, columns=PER_RECORD_COLUMNS))

    import imputation_v3.experiments.runner as runner

    real_summary = runner.paired_model_summary

    def summary_spy(metrics, **kwargs):
        backend.events.append("statistics")
        assert kwargs["required_seeds"] == FORMAL_SEEDS
        return real_summary(metrics, bootstrap_samples=50, **kwargs)

    monkeypatch.setattr(runner, "paired_model_summary", summary_spy)
    report = run_formal_protocol(config, backend=backend, output_root=tmp_path / "formal")
    assert backend.events == [
        "load_assets",
        "materialize_train_validation",
        "train_select_validation",
        "freeze_checkpoints",
        "load_test_data",
        "evaluate_test_once",
        "statistics",
    ]
    assert report["gate"]["passed"] is True
    assert (tmp_path / "formal" / "success_gate.json").is_file()


def test_formal_protocol_does_not_touch_test_when_selection_or_freeze_fails(tmp_path):
    class Failing(_SpyBackend):
        def freeze_checkpoints(self, selected, plan):
            self.events.append("freeze_checkpoints")
            raise RuntimeError("freeze failed")

        def load_test_data(self, assets):
            raise AssertionError("test data touched before successful freeze")

    config = load_teacher_config(ROOT / "configs/imputation_v3/teacher_full.yaml")
    backend = Failing(pd.DataFrame())
    with pytest.raises(RuntimeError, match="freeze failed"):
        run_formal_protocol(config, backend=backend, output_root=tmp_path / "formal")


def test_formal_protocol_validates_complete_backend_before_any_phase(tmp_path):
    config = load_teacher_config(ROOT / "configs/imputation_v3/teacher_full.yaml")
    events = []

    class Incomplete:
        def load_frozen_manifest_and_scaler(self):
            events.append("loaded")

    with pytest.raises(TypeError, match="materialize_train_validation"):
        run_formal_protocol(config, backend=Incomplete(), output_root=tmp_path)
    assert events == []


def test_formal_protocol_requires_exact_preregistered_seeds_before_backend_use(tmp_path):
    config = load_teacher_config(ROOT / "configs/imputation_v3/teacher_smoke.yaml")
    backend = _SpyBackend(pd.DataFrame())
    with pytest.raises(ValueError, match="2026.*2030"):
        run_formal_protocol(config, backend=backend, output_root=tmp_path / "formal")
    assert backend.events == []


def test_formal_artifacts_are_immutable_and_hash_bound(tmp_path):
    metrics = pd.DataFrame(
        [_metric_row(model="teacher"), _metric_row(model="saits", value=3.0)],
        columns=PER_RECORD_COLUMNS,
    )
    summary = pd.DataFrame(
        [{"model": "teacher", "baseline": "saits", "metric": "rmse_physical", "ci95_low": -1.0, "ci95_high": -0.1}]
    )
    gate = success_gate_payload(summary, strongest_baseline="saits")
    ledger = pd.DataFrame([{"mask_id": "m1", "sha256": "b" * 64}])
    output = tmp_path / "formal"
    first = write_formal_artifacts(output, metrics, summary, gate, ledger)
    second = write_formal_artifacts(output, metrics, summary, gate, ledger)
    assert first == second
    hashes = json.loads((output / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert set(hashes) == {
        "per_record_metrics.csv",
        "summary.csv",
        "success_gate.json",
        "mask_ledger.csv",
    }
    changed = metrics.copy()
    changed.loc[0, "value"] = 99.0
    before = (output / "per_record_metrics.csv").read_bytes()
    with pytest.raises(FileExistsError, match="inconsistent"):
        write_formal_artifacts(output, changed, summary, gate, ledger)
    assert (output / "per_record_metrics.csv").read_bytes() == before

    nonfinite_summary = summary.copy()
    nonfinite_summary.loc[0, "ci95_high"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        write_formal_artifacts(
            tmp_path / "bad", metrics, nonfinite_summary, gate, ledger
        )


def test_teacher_matrix_cli_dry_run_is_canonical_and_does_not_discover_data(monkeypatch, capsys):
    import imputation_v3.cli as cli

    monkeypatch.setattr(
        cli,
        "run_teacher_smoke",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("data touched")),
    )
    assert cli.main(
        [
            "teacher-matrix",
            "--config",
            "configs/imputation_v3/teacher_smoke.yaml",
            "--dry-run",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"]["matrix_cells"] == 6
    assert payload["test_data_accessed"] is False
