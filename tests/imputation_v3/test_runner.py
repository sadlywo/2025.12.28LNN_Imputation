from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import imputation_v3.experiments.runner as runner_module
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
    OXIODFormalBackend,
    RTS_PROCESS_VARIANCES,
    build_native_model,
    capacity_candidates,
    estimate_rts_observation_variance,
    evaluate_record_rows,
    formal_matrix_plan,
    make_primary_rows,
    paired_formal_summaries,
    freeze_pypots_predictor,
    reload_pypots_predictor,
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
from validation_v2.data.normalization import RobustTrainScaler
from validation_v2.types import Recording


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
        [{"scenario": "all", "protocol": "teacher_primary", "topology": "all", "model": "teacher", "baseline": "saits", "metric": "rmse_physical", "ci95_low": -1.0, "ci95_high": -0.1}]
    )
    assert success_gate_payload(summary, strongest_baseline="saits") == {
        "candidate": "teacher",
        "strongest_baseline": "saits",
        "metric": "rmse_physical",
        "criterion": "paired_ci95_high_below_zero",
        "passed": True,
        "next_stage": "plan_fixed_lag_students",
    }
    bare = summary.drop(columns=["scenario"])
    with pytest.raises(ValueError, match="scenario"):
        teacher_success(bare, strongest_baseline="saits")


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
            _metric_row(value=3.0, topology="point", requested_fraction=0.1),
            _metric_row(value=4.0, topology="block", requested_fraction=0.1),
            _metric_row(model="teacher_constant_residual", value=1000.0),
            _metric_row(model="saits", value=2.0, topology="point", requested_fraction=0.1),
            _metric_row(model="saits", value=2.0, topology="block", requested_fraction=0.1),
            _metric_row(protocol="axis/gx", value=2000.0),
        ],
        columns=PER_RECORD_COLUMNS,
    )
    primary = make_primary_rows(
        frame,
        candidate_model="teacher_actual_residual",
        strongest_baseline="saits",
        required_topologies=("point", "block"),
        required_rates=(0.1,),
        required_scenarios=("handheld",),
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
            _metric_row(model="saits", value=1e308),
        ],
        columns=PER_RECORD_COLUMNS,
    )
    primary = make_primary_rows(
        frame,
        candidate_model="teacher_actual_residual",
        strongest_baseline="saits",
        required_topologies=("block",),
        required_rates=(0.2,),
        required_scenarios=("handheld",),
    )
    assert np.isfinite(primary["value"]).all()
    np.testing.assert_allclose(primary["value"].to_numpy(), 1e308)


def test_primary_rows_reject_missing_or_duplicate_preregistered_cells():
    rows = [
        _metric_row(topology="point", requested_fraction=0.1),
        _metric_row(topology="block", requested_fraction=0.1),
        _metric_row(model="saits", topology="point", requested_fraction=0.1),
        _metric_row(model="saits", topology="block", requested_fraction=0.1),
    ]
    kwargs = dict(
        candidate_model="teacher_actual_residual",
        strongest_baseline="saits",
        required_topologies=("point", "block"),
        required_rates=(0.1,),
        required_scenarios=("handheld",),
    )
    with pytest.raises(ValueError, match="complete"):
        make_primary_rows(pd.DataFrame(rows[:-1], columns=PER_RECORD_COLUMNS), **kwargs)
    duplicate = pd.DataFrame([*rows, rows[0]], columns=PER_RECORD_COLUMNS)
    with pytest.raises(ValueError, match="duplicate"):
        make_primary_rows(duplicate, **kwargs)
    with pytest.raises(ValueError, match="scenario"):
        make_primary_rows(
            pd.DataFrame(rows, columns=PER_RECORD_COLUMNS),
            **{**kwargs, "required_scenarios": ("handheld", "running")},
        )


def test_formal_summaries_append_secondary_diagnostics_but_gate_only_primary(monkeypatch):
    rows = []
    for seed in FORMAL_SEEDS:
        for recording in ("r1", "r2"):
            for protocol in ("overall", "sensor/gyro", "axis/gx", "gap/50-200ms"):
                rows.append(_metric_row(seed=seed, recording_id=recording, protocol=protocol, value=1.0))
                rows.append(_metric_row(seed=seed, recording_id=recording, protocol=protocol, model="saits", value=2.0))
                rows.append(_metric_row(seed=seed, recording_id=recording, protocol=protocol, model="teacher_constant_residual", value=1.5))
    metrics = pd.DataFrame(rows, columns=PER_RECORD_COLUMNS)
    summary, primary, coverage = paired_formal_summaries(
        metrics,
        candidate_model="teacher_actual_residual",
        strongest_baseline="saits",
        required_topologies=("block",),
        required_rates=(0.2,),
        required_scenarios=("handheld",),
        required_seeds=FORMAL_SEEDS,
        bootstrap_samples=50,
    )
    assert set(summary["protocol"]) >= {"teacher_primary", "sensor/gyro", "axis/gx", "gap/50-200ms"}
    assert len(summary.loc[(summary.model == "teacher") & (summary.protocol == "teacher_primary")]) == 1
    assert "teacher_constant_residual" in set(summary["model"])
    assert set(primary["protocol"]) == {"teacher_primary"}
    assert "teacher_constant_residual" not in set(primary["model"])
    assert coverage["included"].all()


def test_sparse_diagnostic_pairs_are_excluded_with_explicit_coverage_not_fabricated():
    rows = []
    for seed in FORMAL_SEEDS:
        for recording in ("r1", "r2"):
            for model, value in (("teacher_actual_residual", 1.0), ("saits", 2.0)):
                rows.append(_metric_row(seed=seed, recording_id=recording, model=model, value=value))
                if (seed + (recording == "r2")) % 2 == 0:
                    rows.append(_metric_row(seed=seed, recording_id=recording, model=model, protocol="axis/gx", value=value))
                if seed != 2028:
                    rows.append(_metric_row(seed=seed, recording_id=recording, model=model, protocol="gap/50-200ms", value=value))
    metrics = pd.DataFrame(rows, columns=PER_RECORD_COLUMNS)
    summary, primary, coverage = paired_formal_summaries(
        metrics, candidate_model="teacher_actual_residual", strongest_baseline="saits",
        required_topologies=("block",), required_rates=(0.2,),
        required_scenarios=("handheld",), required_seeds=FORMAL_SEEDS,
        bootstrap_samples=50,
    )
    assert set(summary["protocol"]) == {"teacher_primary", "overall"}
    excluded = coverage.loc[~coverage["included"]]
    assert set(excluded["protocol"]) == {"axis/gx", "gap/50-200ms"}
    assert excluded["reason"].str.contains("incomplete").all()
    assert {"axis/gx", "gap/50-200ms"} <= set(metrics["protocol"])
    assert not ({"axis/gx", "gap/50-200ms"} & set(summary["protocol"]))
    assert len(primary) == 20


def test_nested_pypots_predictor_freezes_inner_module_reloads_and_rejects_drift(tmp_path):
    class Outer:
        def __init__(self, weight):
            self.model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.model.weight.fill_(weight)

    class Adapter:
        def __init__(self, weight):
            self.model = Outer(weight)

        def predict(self, value):
            return self.model.model(torch.tensor([[value]])).item()

    adapter = Adapter(3.0)
    checkpoint = tmp_path / "pypots.pt"
    identity = {"name": "brits", "n_steps": 8, "pypots": "1.5.0"}
    digest = freeze_pypots_predictor(adapter, checkpoint, constructor_identity=identity)
    expected = adapter.predict(2.0)
    with torch.no_grad():
        adapter.model.model.weight.fill_(99.0)
    reload_pypots_predictor(adapter, checkpoint, expected_sha256=digest, constructor_identity=identity)
    assert adapter.predict(2.0) == pytest.approx(expected)
    with pytest.raises(FileExistsError, match="fitted predictor"):
        freeze_pypots_predictor(Adapter(4.0), checkpoint, constructor_identity=identity)


def test_rts_train_only_variance_and_frozen_process_grid():
    train = (_recording("train-a"), _recording("train-b"))
    first = estimate_rts_observation_variance(train)
    changed_test = _recording("test")
    changed_test.imu_six[:] = 1e9
    assert estimate_rts_observation_variance(train) == first
    assert first > 0 and math.isfinite(first)
    assert RTS_PROCESS_VARIANCES == (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def test_rts_full_record_prediction_uses_frozen_selected_variances(monkeypatch):
    captured = []

    def rts_spy(observed, mask, time, **kwargs):
        captured.append((time.clone(), dict(kwargs)))
        return observed.clone()

    monkeypatch.setattr(
        "imputation_v3.experiments.runner.constant_velocity_rts", rts_spy
    )
    backend = object.__new__(OXIODFormalBackend)
    backend.device = torch.device("cpu")
    candidate = SimpleNamespace(
        condition="rts",
        context_samples=4,
    )
    predictor = {"process_var": 1e-4, "observation_var": 0.321}
    target = torch.arange(18, dtype=torch.float64).reshape(6, 3)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[2:4] = False
    time = torch.arange(6, dtype=torch.float64) * 0.01

    predicted = backend._predict_full(candidate, predictor, target, mask, time)

    assert predicted.shape == (6, 3)
    assert len(captured) == 2
    for local_time, kwargs in captured:
        assert local_time.shape == (4,)
        assert local_time[0] == 0.0
        assert kwargs["process_var"] == pytest.approx(1e-4)
        assert kwargs["observation_var"] == pytest.approx(0.321)
        assert kwargs["empty_fill"] == 0.0


def test_classical_loader_reconstructs_rts_parameters_from_frozen_checkpoint(tmp_path):
    descriptor = {
        "kind": "classical",
        "condition": "rts",
        "seed": 2026,
        "context_samples": 128,
        "capacity": {"process_var": 1e-4},
        "process_var": 1e-4,
        "observation_var": 0.321,
    }
    frozen = (runner_module.canonical_json(descriptor) + "\n").encode("utf-8")
    checkpoint = tmp_path / "rts.json"
    checkpoint.write_bytes(frozen)
    candidate = runner_module._Candidate(
        seed=2026,
        model_alias="rts",
        condition="rts",
        context_samples=128,
        validation_rmse=1.0,
        checkpoint_sha256=hashlib.sha256(frozen).hexdigest(),
        checkpoint_path=checkpoint,
        capacity={"process_var": 1e-4},
        inference_config={"process_var": 999.0, "observation_var": 888.0},
    )
    backend = object.__new__(OXIODFormalBackend)

    predictor = backend._load_candidate_predictor(candidate)

    assert predictor == {"process_var": 1e-4, "observation_var": 0.321}


@pytest.mark.parametrize("condition", ("locf", "linear", "pchip"))
def test_classical_loader_reconstructs_parameter_free_predictors_from_checkpoint(
    condition, tmp_path
):
    descriptor = {
        "kind": "classical",
        "condition": condition,
        "seed": 2026,
        "context_samples": 128,
        "capacity": {"capacity": "fixed"},
    }
    frozen = (runner_module.canonical_json(descriptor) + "\n").encode("utf-8")
    checkpoint = tmp_path / f"{condition}.json"
    checkpoint.write_bytes(frozen)
    candidate = runner_module._Candidate(
        seed=2026,
        model_alias=condition,
        condition=condition,
        context_samples=128,
        validation_rmse=1.0,
        checkpoint_sha256=hashlib.sha256(frozen).hexdigest(),
        checkpoint_path=checkpoint,
        capacity={"capacity": "fixed"},
        inference_config={"mutable": "must-not-survive"},
    )
    backend = object.__new__(OXIODFormalBackend)

    assert backend._load_candidate_predictor(candidate) == {}


@pytest.mark.parametrize(
    ("field", "changed", "message"),
    (
        ("kind", "native", "kind"),
        ("condition", "linear", "condition"),
        ("seed", 2027, "seed"),
        ("context_samples", 256, "context"),
        ("capacity", {"process_var": 1e-3}, "capacity"),
    ),
)
def test_classical_loader_rejects_descriptor_identity_mismatch(
    field, changed, message, tmp_path
):
    descriptor = {
        "kind": "classical",
        "condition": "rts",
        "seed": 2026,
        "context_samples": 128,
        "capacity": {"process_var": 1e-4},
        "process_var": 1e-4,
        "observation_var": 0.321,
    }
    descriptor[field] = changed
    frozen = (runner_module.canonical_json(descriptor) + "\n").encode("utf-8")
    checkpoint = tmp_path / f"mismatch-{field}.json"
    checkpoint.write_bytes(frozen)
    candidate = runner_module._Candidate(
        seed=2026,
        model_alias="rts",
        condition="rts",
        context_samples=128,
        validation_rmse=1.0,
        checkpoint_sha256=hashlib.sha256(frozen).hexdigest(),
        checkpoint_path=checkpoint,
        capacity={"process_var": 1e-4},
        inference_config={"process_var": 999.0, "observation_var": 888.0},
    )
    backend = object.__new__(OXIODFormalBackend)

    with pytest.raises(ValueError, match=message):
        backend._load_candidate_predictor(candidate)


def test_capacity_candidates_are_inner_selection_not_matrix_multiplication():
    config = load_teacher_config(ROOT / "configs/imputation_v3/teacher_full.yaml")
    candidates = capacity_candidates("teacher_actual_residual", config)
    assert len(candidates) >= 2
    assert {item["hidden_size"] for item in candidates} >= {config.hidden_size}
    plan = formal_matrix_plan(config)
    assert plan["counts"]["matrix_cells"] == 240
    cell = next(item for item in plan["cells"] if item["condition"] == "teacher_actual_residual")
    assert cell["capacity_candidates"] == candidates


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
            for topology in ("point", "block", "channel"):
                for rate in (0.1, 0.2, 0.3, 0.4):
                    rows.extend(
                        [
                            _metric_row(seed=seed, recording_id=recording, topology=topology, requested_fraction=rate, value=teacher_value),
                            _metric_row(seed=seed, recording_id=recording, topology=topology, requested_fraction=rate, model="saits", value=baseline_value),
                        ]
                    )
    backend = _SpyBackend(pd.DataFrame(rows, columns=PER_RECORD_COLUMNS))

    import imputation_v3.experiments.runner as runner

    real_summary = runner.paired_model_summary

    def summary_spy(metrics, **kwargs):
        backend.events.append("statistics")
        assert kwargs["required_seeds"] == FORMAL_SEEDS
        kwargs.pop("bootstrap_samples", None)
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


def test_formal_artifacts_are_immutable_hash_bound_and_crash_recoverable(
    tmp_path, monkeypatch
):
    metrics = pd.DataFrame(
        [_metric_row(model="teacher"), _metric_row(model="saits", value=3.0)],
        columns=PER_RECORD_COLUMNS,
    )
    summary = pd.DataFrame(
        [{"scenario": "all", "protocol": "teacher_primary", "topology": "all", "model": "teacher", "baseline": "saits", "metric": "rmse_physical", "ci95_low": -1.0, "ci95_high": -0.1}]
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

    partial = tmp_path / "partial"
    stable_write = runner_module._write_stable

    def crash_on_summary(path, content):
        if Path(path).name == "summary.csv":
            raise OSError("simulated artifact crash")
        stable_write(path, content)

    with monkeypatch.context() as scoped:
        scoped.setattr(runner_module, "_write_stable", crash_on_summary)
        with pytest.raises(OSError, match="simulated artifact crash"):
            write_formal_artifacts(partial, metrics, summary, gate, ledger)
    assert (partial / "per_record_metrics.csv").is_file()
    assert not (partial / "artifact_hashes.json").exists()
    recovered = write_formal_artifacts(partial, metrics, summary, gate, ledger)
    assert recovered == first
    assert (partial / "artifact_hashes.json").is_file()


def test_atomic_stable_write_leaves_no_partial_target_when_publish_crashes(
    tmp_path, monkeypatch
):
    target = tmp_path / "checkpoint.bin"

    def crash_before_publish(source, destination):
        del source, destination
        raise OSError("simulated publish crash")

    monkeypatch.setattr(runner_module.os, "link", crash_before_publish)
    with pytest.raises(OSError, match="simulated publish crash"):
        runner_module._write_stable(target, b"complete-checkpoint")

    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_pypots_reload_detects_checkpoint_mutation_during_deserialization(
    tmp_path, monkeypatch
):
    class Outer:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1, bias=False)

    adapter = SimpleNamespace(model=Outer())
    checkpoint = tmp_path / "pypots.pt"
    identity = {"name": "brits", "pypots_version": "1.5.0"}
    digest = freeze_pypots_predictor(
        adapter, checkpoint, constructor_identity=identity
    )
    original_load = runner_module.torch.load

    def mutate_during_load(source, *args, **kwargs):
        frozen_bytes = source.read() if hasattr(source, "read") else Path(source).read_bytes()
        checkpoint.write_bytes(b"mutated-during-load")
        return original_load(io.BytesIO(frozen_bytes), *args, **kwargs)

    monkeypatch.setattr(runner_module.torch, "load", mutate_during_load)
    with pytest.raises(ValueError, match="changed during|hash"):
        reload_pypots_predictor(
            adapter,
            checkpoint,
            expected_sha256=digest,
            constructor_identity=identity,
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


def _recording(recording_id, *, epoch_s=0.0, rows=8):
    time = float(epoch_s) + np.arange(rows, dtype=np.float64) * 0.01
    return Recording(
        id=recording_id,
        imu_time_s=time,
        imu_six=np.arange(rows * 6, dtype=np.float64).reshape(rows, 6),
        vicon_time_s=time,
        vicon_position_m=np.zeros((rows, 3)),
        vicon_quaternion_xyzw=np.tile([0.0, 0.0, 0.0, 1.0], (rows, 1)),
        overlap_s=(float(time[0]), float(time[-1])),
        metadata={},
    )


def test_concrete_backend_loads_only_train_validation_before_freeze(tmp_path):
    config = replace(
        load_teacher_config(ROOT / "configs/imputation_v3/teacher_full.yaml"),
        window_seconds=(0.04,),
        training_topologies=("point",),
        training_rates=(0.2,),
        models=("linear", "teacher"),
    )
    manifest = pd.DataFrame(
        [
            ["train", "s", "train-imu", "train-vi", "train", "a" * 64, "b" * 64],
            ["validation", "s", "validation-imu", "validation-vi", "validation", "c" * 64, "d" * 64],
            ["test", "s", "test-imu", "test-vi", "test", "e" * 64, "f" * 64],
        ],
        columns=("recording_id", "scenario", "imu_path", "vicon_path", "split", "imu_sha256", "vicon_sha256"),
    )
    calls = []

    def loader(imu, vicon):
        del vicon
        recording_id = str(imu).removesuffix("-imu")
        calls.append(recording_id)
        return _recording(recording_id)

    backend = OXIODFormalBackend(
        config,
        repository_root=ROOT,
        output_root=tmp_path,
        requested_device="cpu",
        discover_pairs=lambda root: [{"unused": str(root)}],
        splitter=lambda pairs, seed: manifest.copy(),
        recording_loader=loader,
        source_verifier=lambda path, digest: None,
    )
    assets = backend.load_frozen_manifest_and_scaler()
    assert calls == ["train", "validation"]
    assert assets.scaler.training_ids == ("train",)
    assert list(tmp_path.glob("split_manifest-*.csv"))
    assert list(tmp_path.glob("scaler-*.json"))
    repository = backend.materialize_train_validation(assets, formal_matrix_plan(config))
    first = tuple(repository.iter("train", 2026, 4))
    second = tuple(repository.iter("train", 2026, 4))
    assert first and [window.window_id for window in first] == [window.window_id for window in second]
    assert not hasattr(repository, "_cache")
    assert repository.cached_tensor_windows == 0
    assert repository.identity_cache_cardinality <= 1
    with pytest.raises(RuntimeError, match="freeze"):
        backend.load_test_data(assets)
    assert calls == ["train", "validation"]


def test_simulated_full_matrix_repository_caches_only_bounded_identities():
    config = replace(
        load_teacher_config(ROOT / "configs/imputation_v3/teacher_full.yaml"),
        training_topologies=("point",),
        training_rates=(0.2,),
    )
    train = _recording("train", rows=600)
    validation = _recording("validation", rows=600)
    scaler = RobustTrainScaler.fit((train,), allowed_ids={"train"})
    assets = SimpleNamespace(
        recordings={"train": (train,), "validation": (validation,)},
        scaler=scaler,
    )
    repository = runner_module._SharedWindowRepository(assets, config)
    plan = formal_matrix_plan(config)
    scopes = {
        (int(cell["seed"]), int(cell["context_samples"]))
        for cell in plan["cells"]
    }
    for split in ("train", "validation"):
        for seed, samples in sorted(scopes):
            digest, count = repository.identity(split, seed, samples)
            assert len(digest) == 64
            assert count > 0

    assert plan["counts"]["matrix_cells"] == 240
    assert repository.cached_tensor_windows == 0
    assert repository.identity_cache_cardinality == 2 * len(scopes) == 30
    assert "predictor" not in {field.name for field in fields(runner_module._Candidate)}


def test_teacher_matrix_cli_non_dry_builds_real_backend_and_runs_protocol(monkeypatch, tmp_path, capsys):
    import imputation_v3.cli as cli

    captured = {}
    sentinel = object()

    def backend_factory(config, **kwargs):
        captured["backend"] = (config, kwargs)
        return sentinel

    def protocol(config, *, backend, output_root):
        captured["protocol"] = (config, backend, output_root)
        return {"status": "completed", "matrix_cells": 240}

    monkeypatch.setattr(cli, "OXIODFormalBackend", backend_factory)
    monkeypatch.setattr(cli, "run_formal_protocol", protocol)
    monkeypatch.setattr(cli, "installed_pypots_version", lambda: "1.5.0")
    package = ModuleType("pypots")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "pypots", package)
    monkeypatch.setitem(sys.modules, "pypots.imputation", ModuleType("pypots.imputation"))
    assert cli.main([
        "teacher-matrix", "--config", "configs/imputation_v3/teacher_full.yaml",
        "--device", "cpu", "--output-root", str(tmp_path),
    ]) == 0
    assert captured["backend"][1]["requested_device"] == "cpu"
    assert captured["protocol"][1] is sentinel
    assert captured["protocol"][2] == tmp_path.resolve()
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_teacher_matrix_cli_rejects_pypots_version_before_backend(
    monkeypatch, tmp_path, capsys
):
    import imputation_v3.cli as cli

    def reject_version():
        raise RuntimeError(
            "formal PyPOTS execution requires exactly pypots==1.5.0; "
            "installed version is 1.5.1"
        )

    monkeypatch.setattr(cli, "installed_pypots_version", reject_version)
    monkeypatch.setattr(
        cli,
        "OXIODFormalBackend",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("backend must not start")
        ),
    )

    assert cli.main([
        "teacher-matrix",
        "--config", "configs/imputation_v3/teacher_full.yaml",
        "--device", "cpu",
        "--output-root", str(tmp_path),
    ]) == 2
    assert "requires exactly pypots==1.5.0" in capsys.readouterr().err


def test_concrete_backend_executes_tiny_native_and_classical_protocol(tmp_path):
    config = replace(
        load_teacher_config(ROOT / "configs/imputation_v3/teacher_smoke.yaml"),
        seeds=(2026,), window_seconds=(0.04,), batch_size=2, epochs=1,
        hidden_size=8, tcn_width=8, tcn_dilations=(1,),
        training_topologies=("point",), training_rates=(0.2,),
        models=("linear", "teacher"),
    )
    manifest = pd.DataFrame(
        [
            ["train", "s", "train-imu", "train-vi", "train", "a" * 64, "b" * 64],
            ["validation", "s", "validation-imu", "validation-vi", "validation", "c" * 64, "d" * 64],
            ["test", "s", "test-imu", "test-vi", "test", "e" * 64, "f" * 64],
            ["test2", "s", "test2-imu", "test2-vi", "test", "1" * 64, "2" * 64],
        ],
        columns=("recording_id", "scenario", "imu_path", "vicon_path", "split", "imu_sha256", "vicon_sha256"),
    )
    backend = OXIODFormalBackend(
        config, repository_root=ROOT, output_root=tmp_path, requested_device="cpu",
        discover_pairs=lambda root: [{"root": str(root)}],
        splitter=lambda pairs, seed: manifest.copy(),
        recording_loader=lambda imu, vicon: _recording(
            str(imu).removesuffix("-imu"), epoch_s=1.496e9
        ),
        source_verifier=lambda path, digest: None,
    )
    plan = formal_matrix_plan(config)
    assets = backend.load_frozen_manifest_and_scaler()
    windows = backend.materialize_train_validation(assets, plan)
    selected = backend.train_select_validation(windows, plan)
    assert len(selected) == 6
    assert all(not hasattr(candidate, "predictor") for candidate in selected.values())
    native_manifest = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "candidates").glob("*/run.json")
    )
    assert native_manifest["config"]["hyperparameters"]["hidden_size"] == 8
    assert len(native_manifest["config"]["train_window_ids_sha256"]) == 64
    assert len(native_manifest["config"]["validation_window_ids_sha256"]) == 64
    candidate = next(iter(selected.values()))
    original_checkpoint = candidate.checkpoint_path.read_bytes()
    candidate.checkpoint_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint.*identity"):
        backend.freeze_checkpoints(selected, plan)
    candidate.checkpoint_path.write_bytes(original_checkpoint)
    frozen = backend.freeze_checkpoints(selected, plan)
    frozen_manifest = json.loads((tmp_path / "frozen_models.json").read_text(encoding="utf-8"))
    for item in frozen_manifest["checkpoints"]:
        assert item["validation_rmse"] == min(
            score["validation_rmse"] for score in item["validation_scores"]
        )
        if item["condition"].startswith("teacher_"):
            assert len(item["validation_scores"]) == 2
    candidate.checkpoint_path.write_bytes(b"tampered-after-freeze")
    with pytest.raises(ValueError, match="checkpoint.*identity"):
        backend.load_test_data(assets)
    candidate.checkpoint_path.write_bytes(original_checkpoint)
    test_data = backend.load_test_data(assets)
    evaluation = backend.evaluate_test_once(frozen, test_data, plan)
    assert set(evaluation["per_record_metrics"]["model"]) == {
        "linear", "teacher_actual_residual", "teacher_constant_residual",
        "teacher_dt_feature_only_residual", "teacher_no_dt_residual", "teacher_actual_raw",
    }
    assert len(evaluation["mask_ledger"]) == 2
    assert (tmp_path / "frozen_models.json").is_file()
    summary, primary, coverage = paired_formal_summaries(
        evaluation["per_record_metrics"],
        candidate_model="teacher_actual_residual", strongest_baseline="linear",
        required_topologies=("point",), required_rates=(0.2,),
        required_scenarios=("s",), required_seeds=(2026,), bootstrap_samples=50,
    )
    gate = success_gate_payload(summary, strongest_baseline="linear")
    hashes = write_formal_artifacts(
        tmp_path / "published",
        pd.concat((evaluation["per_record_metrics"], primary), ignore_index=True),
        summary, gate, evaluation["mask_ledger"], coverage,
    )
    assert "coverage_ledger.csv" in hashes


def test_concrete_backend_rehashes_deferred_test_sources_before_parsing(tmp_path):
    config = replace(
        load_teacher_config(ROOT / "configs/imputation_v3/teacher_smoke.yaml"),
        seeds=(2026,), window_seconds=(0.04,), training_topologies=("point",),
        training_rates=(0.2,), models=("rts",),
    )
    rows = []
    for split in ("train", "validation", "test"):
        paths = []
        digests = []
        for kind in ("imu", "vi"):
            path = tmp_path / f"{split}-{kind}.csv"
            path.write_text(f"{split}-{kind}", encoding="utf-8")
            paths.append(path)
            digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append([split, "s", str(paths[0]), str(paths[1]), split, *digests])
    manifest = pd.DataFrame(rows, columns=(
        "recording_id", "scenario", "imu_path", "vicon_path", "split",
        "imu_sha256", "vicon_sha256",
    ))
    calls = []

    def loader(imu, vicon):
        del vicon
        recording_id = Path(imu).stem.removesuffix("-imu")
        calls.append(recording_id)
        return _recording(recording_id)

    backend = OXIODFormalBackend(
        config, repository_root=ROOT, output_root=tmp_path / "out", requested_device="cpu",
        discover_pairs=lambda root: [], splitter=lambda pairs, seed: manifest.copy(),
        recording_loader=loader,
    )
    plan = formal_matrix_plan(config)
    assets = backend.load_frozen_manifest_and_scaler()
    repository = backend.materialize_train_validation(assets, plan)
    selected = backend.train_select_validation(repository, plan)
    backend.freeze_checkpoints(selected, plan)
    frozen_manifest = json.loads(
        (tmp_path / "out" / "frozen_models.json").read_text(encoding="utf-8")
    )
    rts = frozen_manifest["checkpoints"][0]
    assert len(rts["validation_scores"]) == len(RTS_PROCESS_VARIANCES)
    assert rts["capacity"]["process_var"] in RTS_PROCESS_VARIANCES
    assert rts["validation_rmse"] == min(
        item["validation_rmse"] for item in rts["validation_scores"]
    )
    selected_rts = selected[(2026, "rts")]
    selected_rts.inference_config = {
        "process_var": 999.0,
        "observation_var": 888.0,
    }
    predictor = backend._load_candidate_predictor(selected_rts)
    assert predictor == {
        "process_var": selected_rts.capacity["process_var"],
        "observation_var": assets.rts_observation_var,
    }
    Path(manifest.loc[manifest.split == "test", "imu_path"].item()).write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source SHA256"):
        backend.load_test_data(assets)
    assert calls == ["train", "validation"]


def test_concrete_backend_rejects_source_mutation_during_recording_load(tmp_path):
    config = replace(
        load_teacher_config(ROOT / "configs/imputation_v3/teacher_smoke.yaml"),
        models=("linear",),
    )
    rows = []
    source_paths = {}
    for split in ("train", "validation", "test"):
        paths = []
        digests = []
        for kind in ("imu", "vi"):
            path = tmp_path / f"{split}-{kind}.csv"
            path.write_text(f"{split}-{kind}", encoding="utf-8")
            source_paths[(split, kind)] = path
            paths.append(path)
            digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append([split, "s", str(paths[0]), str(paths[1]), split, *digests])
    manifest = pd.DataFrame(rows, columns=(
        "recording_id", "scenario", "imu_path", "vicon_path", "split",
        "imu_sha256", "vicon_sha256",
    ))

    def mutating_loader(imu, vicon):
        del vicon
        recording_id = Path(imu).stem.removesuffix("-imu")
        if recording_id == "train":
            Path(imu).write_text("changed while parsing", encoding="utf-8")
        return _recording(recording_id)

    backend = OXIODFormalBackend(
        config,
        repository_root=ROOT,
        output_root=tmp_path / "output",
        requested_device="cpu",
        discover_pairs=lambda root: [{"root": str(root)}],
        splitter=lambda pairs, seed: manifest.copy(),
        recording_loader=mutating_loader,
    )

    with pytest.raises(ValueError, match="source SHA256 mismatch"):
        backend.load_frozen_manifest_and_scaler()
