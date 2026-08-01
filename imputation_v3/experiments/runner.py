"""Leakage-ordered formal matrix orchestration and the teacher accuracy gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import gc
import hashlib
import io
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import random
import tempfile
from types import SimpleNamespace
from typing import Any, Final, Protocol

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset

from imputation_v3.config import TeacherConfig
from imputation_v3.data.features import build_features
from imputation_v3.data.windows import collate_prepared_windows, iter_teacher_windows
from imputation_v3.experiments.evaluate import evaluate_record_diagnostics
from imputation_v3.experiments.pypots import (
    build_pypots_model,
    installed_pypots_version,
    to_pypots_sets,
)
from imputation_v3.experiments.training import make_teacher_callbacks
from imputation_v3.models.baselines import (
    constant_velocity_rts,
    timestamp_linear,
    timestamp_locf,
    timestamp_pchip,
)
from imputation_v3.models.native_controls import (
    CONTROL_CONDITIONS,
    TEACHER_CONDITION_MODES,
    BiCfCControl,
    BiLSTMControl,
    FeatureMLPControl,
    TCNControl,
    count_parameters,
)
from imputation_v3.models.teacher import OfflineTeacher
from imputation_v3.types import PreparedWindow
from validation_v2.data.masking import channel_outage, contiguous_block, point_missing
from validation_v2.data.normalization import RobustTrainScaler
from validation_v2.data.oxiod import load_recording
from validation_v2.data.splits import MANIFEST_COLUMNS, stratified_file_split
from validation_v2.evaluation.statistics import (
    GROUP_COLUMNS,
    PER_RECORD_COLUMNS,
    paired_model_summary,
    validate_per_record_metrics,
)
from validation_v2.experiments.provenance import canonical_json
from validation_v2.experiments.provenance import collect_provenance, git_worktree_identity
from validation_v2.experiments.runner import discover_oxiod_pairs
from validation_v2.experiments.train import train_one_run


FORMAL_SEEDS: Final[tuple[int, ...]] = (2026, 2027, 2028, 2029, 2030)
RTS_PROCESS_VARIANCES: Final[tuple[float, ...]] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
_CLASSICAL_MODELS = ("locf", "linear", "pchip", "rts")
_PYPOTS_MODELS = ("brits", "saits", "csdi")
_FEATURE_MLP_WIDTHS = (32, 48, 64, 96, 128, 192)
_GATE_KEYS = (
    "candidate",
    "strongest_baseline",
    "metric",
    "criterion",
    "passed",
    "next_stage",
)


def capacity_candidates(condition: str, config: TeacherConfig) -> list[dict[str, Any]]:
    """Return condition-appropriate inner validation-selection capacities."""
    config = _require_config(config)
    if condition in _CLASSICAL_MODELS or condition in _PYPOTS_MODELS:
        if condition == "rts":
            return [{"process_var": value} for value in RTS_PROCESS_VARIANCES]
        return [{"capacity": "fixed"}]
    if condition in {"bilstm", "bilnn"}:
        values = sorted({max(4, config.hidden_size // 2), config.hidden_size})
        return [{"hidden_size": value} for value in values]
    if condition == "tcn":
        values = sorted({max(4, config.tcn_width // 2), config.tcn_width})
        return [{"tcn_width": value} for value in values]
    if condition == "feature_mlp":
        return [{"mlp_width": value} for value in _FEATURE_MLP_WIDTHS]
    if condition in TEACHER_CONDITION_MODES:
        pairs = {
            (max(4, config.hidden_size // 2), max(4, config.tcn_width // 2)),
            (config.hidden_size, config.tcn_width),
        }
        return [
            {"hidden_size": hidden, "tcn_width": width}
            for hidden, width in sorted(pairs)
        ]
    raise ValueError(f"unsupported capacity condition: {condition}")


def estimate_rts_observation_variance(
    train_recordings: Sequence[Any], scaler: RobustTrainScaler | None = None
) -> float:
    """Estimate normalized observation noise from training residuals only."""
    if not train_recordings:
        raise ValueError("RTS observation variance requires training recordings")
    residuals = []
    for recording in train_recordings:
        values = np.asarray(recording.imu_six, dtype=np.float64)
        if scaler is not None:
            values = scaler.transform(values)
        if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != 6:
            raise ValueError("RTS training recordings require at least three six-axis rows")
        if not np.isfinite(values).all():
            raise ValueError("RTS training values must be finite")
        residuals.append(np.diff(values, n=2, axis=0) / math.sqrt(6.0))
    joined = np.concatenate(residuals, axis=0)
    estimate = max(float(np.mean(np.square(joined))), 1e-8)
    if not math.isfinite(estimate):
        raise ValueError("RTS observation variance must be finite")
    return estimate


def _pypots_state_target(adapter: Any) -> torch.nn.Module:
    outer = getattr(adapter, "model", None)
    inner = getattr(outer, "model", None)
    target = inner if isinstance(inner, torch.nn.Module) else outer
    if not isinstance(target, torch.nn.Module):
        raise TypeError("PyPOTS predictor must expose its actual inner nn.Module")
    return target


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in module.state_dict().items()
    }


def _state_equal(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> bool:
    return set(first) == set(second) and all(
        first[name].dtype == second[name].dtype
        and first[name].shape == second[name].shape
        and torch.equal(first[name], second[name])
        for name in first
    )


def reload_pypots_predictor(
    adapter: Any,
    checkpoint_path: Path,
    *,
    expected_sha256: str,
    constructor_identity: Mapping[str, Any],
) -> None:
    """Hash-verify and load exact frozen inner PyPOTS predictor bytes."""
    path = Path(checkpoint_path)
    frozen_bytes = _read_verified_bytes(
        path, expected_sha256, label="PyPOTS checkpoint"
    )
    payload = torch.load(io.BytesIO(frozen_bytes), map_location="cpu", weights_only=True)
    if not path.is_file() or _sha256_path(path) != expected_sha256:
        raise ValueError("PyPOTS checkpoint changed during deserialization")
    expected_identity = canonical_json(dict(constructor_identity))
    if not isinstance(payload, dict) or payload.get("constructor_identity") != expected_identity:
        raise ValueError("PyPOTS constructor identity does not match checkpoint")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("PyPOTS checkpoint state_dict is malformed")
    _pypots_state_target(adapter).load_state_dict(state, strict=True)


def freeze_pypots_predictor(
    adapter: Any,
    checkpoint_path: Path,
    *,
    constructor_identity: Mapping[str, Any],
) -> str:
    """Freeze the fitted inner predictor, compare prior state, then reload it."""
    path = Path(checkpoint_path)
    identity_json = canonical_json(dict(constructor_identity))
    fitted_state = _cpu_state_dict(_pypots_state_target(adapter))
    if path.exists():
        existing_bytes = path.read_bytes()
        existing_digest = _sha256_bytes(existing_bytes)
        existing = torch.load(
            io.BytesIO(existing_bytes), map_location="cpu", weights_only=True
        )
        if not path.is_file() or _sha256_path(path) != existing_digest:
            raise ValueError("PyPOTS checkpoint changed during comparison")
        if (
            not isinstance(existing, dict)
            or existing.get("constructor_identity") != identity_json
            or not isinstance(existing.get("state_dict"), dict)
            or not _state_equal(fitted_state, existing["state_dict"])
        ):
            raise FileExistsError("existing PyPOTS checkpoint differs from fitted predictor")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".pypots-", suffix=".pt", delete=False
            ) as handle:
                temporary = Path(handle.name)
            torch.save(
                {"constructor_identity": identity_json, "state_dict": fitted_state},
                temporary,
            )
            _write_stable(path, temporary.read_bytes())
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    digest = _sha256_path(path)
    reload_pypots_predictor(
        adapter,
        path,
        expected_sha256=digest,
        constructor_identity=constructor_identity,
    )
    return digest


def _require_config(config: object) -> TeacherConfig:
    if not isinstance(config, TeacherConfig):
        raise TypeError("config must be TeacherConfig")
    if config.selection_split != "validation":
        raise ValueError("formal selection_split must be validation")
    return config


def build_native_model(
    condition: str,
    config: TeacherConfig,
    *,
    capacity: Mapping[str, Any] | None = None,
):
    """Build one native control from Task 9's frozen explicit condition names."""
    config = _require_config(config)
    if not isinstance(condition, str):
        raise TypeError("native condition must be a string")
    selected_capacity = dict(capacity or {})
    hidden_size = int(selected_capacity.get("hidden_size", config.hidden_size))
    tcn_width = int(selected_capacity.get("tcn_width", config.tcn_width))
    if condition == "bilstm":
        return BiLSTMControl(31, hidden_size)
    if condition == "bilnn":
        return BiCfCControl(31, hidden_size)
    if condition == "tcn":
        return TCNControl(31, tcn_width, config.tcn_dilations)
    if condition == "feature_mlp":
        if "mlp_width" in selected_capacity:
            return FeatureMLPControl(31, int(selected_capacity["mlp_width"]))
        teacher = OfflineTeacher(
            31, config.hidden_size, config.tcn_width, config.tcn_dilations
        )
        target_parameters = count_parameters(teacher)
        candidates = [
            FeatureMLPControl(31, width) for width in _FEATURE_MLP_WIDTHS
        ]
        return min(
            candidates,
            key=lambda model: abs(count_parameters(model) - target_parameters),
        )
    try:
        time_mode, residual_mode = TEACHER_CONDITION_MODES[condition]
    except KeyError as exc:
        raise ValueError(f"unsupported native condition: {condition}") from exc
    return OfflineTeacher(
        31,
        hidden_size,
        tcn_width,
        config.tcn_dilations,
        time_mode=time_mode,
        residual_mode=residual_mode,
    )


def _model_conditions(model: str) -> tuple[str, ...]:
    if model == "teacher":
        return tuple(TEACHER_CONDITION_MODES)
    if model in (*_CLASSICAL_MODELS, *CONTROL_CONDITIONS, *_PYPOTS_MODELS):
        return (model,)
    raise ValueError(f"unsupported formal model alias: {model}")


def formal_matrix_plan(config: TeacherConfig) -> dict[str, Any]:
    """Expand the deterministic seed/context/model/condition matrix without I/O."""
    config = _require_config(config)
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be unique")
    if len(set(config.models)) != len(config.models):
        raise ValueError("models must be unique")
    contexts = tuple(zip(config.window_seconds, config.window_samples))
    if len({samples for _, samples in contexts}) != len(contexts):
        raise ValueError("contexts must be unique after conversion to samples")

    cells: list[dict[str, Any]] = []
    conditions_per_context = sum(len(_model_conditions(model)) for model in config.models)
    for seed in config.seeds:
        for seconds, samples in contexts:
            for model in config.models:
                for condition in _model_conditions(model):
                    cells.append(
                        {
                            "seed": seed,
                            "context_seconds": seconds,
                            "context_samples": samples,
                            "model": model,
                            "condition": condition,
                            "capacity_candidates": capacity_candidates(
                                condition, config
                            ),
                        }
                    )
    return {
        "mode": "imputation_v3_teacher_matrix",
        "selection_split": "validation",
        "dry_run": True,
        "test_data_accessed": False,
        "seeds": list(config.seeds),
        "contexts": [
            {"seconds": seconds, "samples": samples}
            for seconds, samples in contexts
        ],
        "models": list(config.models),
        "counts": {
            "seeds": len(config.seeds),
            "contexts": len(contexts),
            "models": len(config.models),
            "conditions_per_seed_context": conditions_per_context,
            "matrix_cells": len(cells),
        },
        "cells": cells,
    }


def teacher_success(summary: pd.DataFrame, *, strongest_baseline: str) -> bool:
    """Apply the preregistered strict paired-CI accuracy gate."""
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("summary must be a pandas DataFrame")
    if not isinstance(strongest_baseline, str) or not strongest_baseline:
        raise ValueError("strongest_baseline must be a non-empty string")
    required = {
        "scenario",
        "protocol",
        "topology",
        "model",
        "baseline",
        "metric",
        "ci95_low",
        "ci95_high",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError("summary is missing required columns: " + ", ".join(missing))
    selected = summary.loc[
        (summary["model"] == "teacher")
        & (summary["baseline"] == strongest_baseline)
        & (summary["metric"] == "rmse_physical")
    ]
    selected = selected.loc[
        (selected["scenario"] == "all")
        & (selected["protocol"] == "teacher_primary")
        & (selected["topology"] == "all")
    ]
    if len(selected) != 1:
        raise ValueError(
            "teacher success requires exactly one preregistered strongest-baseline row"
        )
    low_value = selected.iloc[0]["ci95_low"]
    high_value = selected.iloc[0]["ci95_high"]
    if (
        isinstance(low_value, (bool, np.bool_))
        or isinstance(high_value, (bool, np.bool_))
        or not isinstance(low_value, Real)
        or not isinstance(high_value, Real)
    ):
        raise TypeError("teacher success CI bounds must be real numeric values")
    low = float(low_value)
    high = float(high_value)
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("teacher success CI bounds must be finite")
    if low > high:
        raise ValueError("teacher success CI lower bound must not exceed upper bound")
    return high < 0.0


def success_gate_payload(
    summary: pd.DataFrame, *, strongest_baseline: str
) -> dict[str, Any]:
    """Return the exact immutable handoff decision payload."""
    passed = teacher_success(summary, strongest_baseline=strongest_baseline)
    return {
        "candidate": "teacher",
        "strongest_baseline": strongest_baseline,
        "metric": "rmse_physical",
        "criterion": "paired_ci95_high_below_zero",
        "passed": passed,
        "next_stage": (
            "plan_fixed_lag_students"
            if passed
            else "stop_and_analyze_teacher_failure"
        ),
    }


def evaluate_record_rows(
    *,
    raw_windows,
    starts,
    target_normalized,
    observed_mask,
    time,
    scaler,
    run_id: str,
    seed: int,
    recording_id: str,
    scenario: str,
    topology: str,
    requested_fraction: float,
    realized_fraction: float,
    model: str,
    checkpoint_sha256: str,
) -> pd.DataFrame:
    """Evaluate one recording once and emit every diagnostic in v2 long form."""
    strings = {
        "run_id": run_id,
        "recording_id": recording_id,
        "scenario": scenario,
        "topology": topology,
        "model": model,
    }
    if any(not isinstance(value, str) or not value for value in strings.values()):
        raise ValueError("record metric identity fields must be non-empty strings")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if not isinstance(checkpoint_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", checkpoint_sha256
    ) is None:
        raise ValueError("checkpoint_sha256 must be 64 lowercase hex characters")
    fractions: dict[str, float] = {}
    for name, value in (
        ("requested_fraction", requested_fraction),
        ("realized_fraction", realized_fraction),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real fraction")
        converted = float(value)
        if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1")
        fractions[name] = converted

    diagnostics = evaluate_record_diagnostics(
        raw_windows=raw_windows,
        starts=starts,
        target_normalized=target_normalized,
        observed_mask=observed_mask,
        time=time,
        scaler=scaler,
        recording_id=recording_id,
    )
    rows = []
    for protocol, metric_values in diagnostics.items():
        for metric in ("rmse_physical", "mae_physical"):
            rows.append(
                {
                    "run_id": run_id,
                    "seed": int(seed),
                    "recording_id": recording_id,
                    "scenario": scenario,
                    "protocol": protocol,
                    "topology": topology,
                    "requested_fraction": fractions["requested_fraction"],
                    "realized_fraction": fractions["realized_fraction"],
                    "model": model,
                    "metric": metric,
                    "value": metric_values[metric],
                    "checkpoint_sha256": checkpoint_sha256,
                }
            )
    return validate_per_record_metrics(pd.DataFrame(rows, columns=PER_RECORD_COLUMNS))


def _digest_strings(values: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json(sorted(values)).encode("utf-8")).hexdigest()


def _stable_rms(values: np.ndarray) -> float:
    absolute = np.abs(values)
    maximum = float(absolute.max())
    if maximum == 0.0:
        return 0.0
    return float(maximum * np.sqrt(np.mean(np.square(absolute / maximum))))


def make_primary_rows(
    metrics: pd.DataFrame,
    *,
    candidate_model: str,
    strongest_baseline: str,
    required_topologies: Sequence[str],
    required_rates: Sequence[float],
    required_scenarios: Sequence[str],
) -> pd.DataFrame:
    """Macro-average only the exact preregistered overall condition matrix."""
    checked = validate_per_record_metrics(metrics)
    if not isinstance(candidate_model, str) or not candidate_model:
        raise ValueError("candidate_model must be a non-empty explicit condition")
    if candidate_model == "teacher":
        raise ValueError("candidate_model must be explicit, not the teacher alias")
    if not isinstance(strongest_baseline, str) or not strongest_baseline:
        raise ValueError("strongest_baseline must be a non-empty string")
    topologies = tuple(required_topologies)
    rates = tuple(float(value) for value in required_rates)
    scenarios = tuple(required_scenarios)
    if not topologies or len(set(topologies)) != len(topologies):
        raise ValueError("required_topologies must be non-empty and unique")
    if not rates or len(set(rates)) != len(rates):
        raise ValueError("required_rates must be non-empty and unique")
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("required_scenarios must be non-empty and unique")
    selected = checked.loc[
        (checked["metric"] == "rmse_physical")
        & (checked["protocol"] == "overall")
        & checked["model"].isin((candidate_model, strongest_baseline))
    ].copy()
    if selected.empty:
        raise ValueError("no primary RMSE rows match teacher and strongest baseline")

    unexpected_topologies = sorted(set(selected["topology"]) - set(topologies))
    unexpected_rates = sorted(
        set(selected["requested_fraction"].astype(float)) - set(rates)
    )
    if unexpected_topologies or unexpected_rates:
        raise ValueError("primary rows contain conditions outside preregistration")
    for (seed, model), model_rows in selected.groupby(["seed", "model"]):
        if set(model_rows["scenario"]) != set(scenarios):
            raise ValueError(
                f"primary scenario matrix is incomplete for seed={seed}, model={model}"
            )

    rows: list[dict[str, Any]] = []
    for (seed, recording_id), group in selected.groupby(
        ["seed", "recording_id"], sort=True, dropna=False
    ):
        teacher = group.loc[group["model"] == candidate_model]
        baseline = group.loc[group["model"] == strongest_baseline]
        expected_cells = {(topology, rate) for topology in topologies for rate in rates}
        for model_rows in (teacher, baseline):
            cells = list(
                zip(
                    model_rows["topology"],
                    model_rows["requested_fraction"].astype(float),
                )
            )
            if len(cells) != len(set(cells)):
                raise ValueError("primary matrix contains duplicate condition cells")
            if set(cells) != expected_cells:
                raise ValueError("primary matrix must be an exact complete condition matrix")
        for model_name, model_rows in (
            (strongest_baseline, baseline),
            ("teacher", teacher),
        ):
            values = model_rows["value"].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("primary RMSE values must be finite")
            source_run_ids = [str(value) for value in model_rows["run_id"]]
            checkpoint_hashes = [str(value) for value in model_rows["checkpoint_sha256"]]
            unique_checkpoint_hashes = set(checkpoint_hashes)
            if len(unique_checkpoint_hashes) != 1:
                raise ValueError("primary condition rows must share one frozen checkpoint")
            rows.append(
                {
                    "run_id": "primary-" + _digest_strings(source_run_ids)[:16],
                    "seed": int(seed),
                    "recording_id": str(recording_id),
                    "scenario": "all",
                    "protocol": "teacher_primary",
                    "topology": "all",
                    "requested_fraction": float(
                        model_rows["requested_fraction"].astype(float).mean()
                    ),
                    "realized_fraction": float(
                        model_rows["realized_fraction"].astype(float).mean()
                    ),
                    "model": model_name,
                    "metric": "rmse_physical",
                    "value": _stable_rms(values),
                    "checkpoint_sha256": next(iter(unique_checkpoint_hashes)),
                }
            )
    result = pd.DataFrame(rows, columns=PER_RECORD_COLUMNS).sort_values(
        ["seed", "recording_id", "model"], ignore_index=True
    )
    return validate_per_record_metrics(result)


def paired_formal_summaries(
    metrics: pd.DataFrame,
    *,
    candidate_model: str,
    strongest_baseline: str,
    required_topologies: Sequence[str],
    required_rates: Sequence[float],
    required_scenarios: Sequence[str],
    required_seeds: Sequence[int],
    bootstrap_samples: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return appended primary and diagnostic paired summaries."""
    checked = validate_per_record_metrics(metrics)
    primary = make_primary_rows(
        checked,
        candidate_model=candidate_model,
        strongest_baseline=strongest_baseline,
        required_topologies=required_topologies,
        required_rates=required_rates,
        required_scenarios=required_scenarios,
    )
    primary_summary = paired_model_summary(
        primary,
        baseline=strongest_baseline,
        required_seeds=required_seeds,
        bootstrap_samples=bootstrap_samples,
    )
    secondary = checked.copy()
    secondary.loc[secondary["model"] == candidate_model, "model"] = "teacher"
    expected_seeds = set(int(value) for value in required_seeds)
    expected_models = set(secondary.loc[secondary["protocol"] == "overall", "model"])
    included_groups: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    for group_key, group in secondary.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        group_values = dict(zip(GROUP_COLUMNS, group_key))
        seeds = set(int(value) for value in group["seed"])
        models = set(group["model"])
        cell_recordings = {
            (int(seed), str(model)): set(values["recording_id"].astype(str))
            for (seed, model), values in group.groupby(["seed", "model"])
        }
        expected_cells = {(seed, model) for seed in expected_seeds for model in expected_models}
        recording_sets = list(cell_recordings.values())
        common = set.intersection(*recording_sets) if recording_sets else set()
        union = set.union(*recording_sets) if recording_sets else set()
        complete = (
            seeds == expected_seeds
            and models == expected_models
            and set(cell_recordings) == expected_cells
            and len(common) >= 2
            and all(value == common for value in recording_sets)
        )
        reasons = []
        if seeds != expected_seeds:
            reasons.append("incomplete_seeds")
        if models != expected_models or set(cell_recordings) != expected_cells:
            reasons.append("incomplete_models")
        if len(common) < 2:
            reasons.append("incomplete_recordings")
        if recording_sets and any(value != common for value in recording_sets):
            reasons.append("incomplete_pair_coverage")
        coverage_rows.append(
            {
                **group_values,
                "included": complete,
                "reason": "complete" if complete else "+".join(reasons),
                "present_seeds": len(seeds),
                "required_seeds": len(expected_seeds),
                "present_models": len(models),
                "required_models": len(expected_models),
                "common_recordings": len(common),
                "union_recordings": len(union),
            }
        )
        if complete:
            included_groups.append(group)
    if not included_groups:
        raise ValueError("no secondary strata have complete paired coverage")
    complete_secondary = pd.concat(included_groups, ignore_index=True)
    secondary_summary = paired_model_summary(
        complete_secondary,
        baseline=strongest_baseline,
        required_seeds=required_seeds,
        bootstrap_samples=bootstrap_samples,
    )
    summary = pd.concat((primary_summary, secondary_summary), ignore_index=True)
    coverage = pd.DataFrame(coverage_rows)
    return summary, primary, coverage


def _dataframe_bytes(frame: pd.DataFrame, name: str) -> bytes:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    text = frame.to_csv(index=False, lineterminator="\n")
    return text.encode("utf-8")


def _require_finite_numeric_columns(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        raise ValueError(f"{name} must not be empty")
    if frame.isna().any().any():
        raise ValueError(f"{name} must not contain missing or non-finite values")
    for column in frame.select_dtypes(include=[np.number]).columns:
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} numeric values must be finite")


def _validate_gate(gate: Mapping[str, Any]) -> None:
    if list(gate.keys()) != list(_GATE_KEYS):
        raise ValueError("success gate must use the exact fixed schema")
    if type(gate["passed"]) is not bool:
        raise TypeError("success gate passed must be bool")
    expected_next = (
        "plan_fixed_lag_students"
        if gate["passed"]
        else "stop_and_analyze_teacher_failure"
    )
    if (
        gate["candidate"] != "teacher"
        or gate["metric"] != "rmse_physical"
        or gate["criterion"] != "paired_ci95_high_below_zero"
        or gate["next_stage"] != expected_next
    ):
        raise ValueError("success gate content does not match the preregistered contract")


def write_formal_artifacts(
    output_root: Path,
    per_record_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    gate: Mapping[str, Any],
    mask_ledger: pd.DataFrame,
    coverage_ledger: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Seal formal outputs without replacing inconsistent existing content."""
    root = Path(output_root)
    if root.exists() and not root.is_dir():
        raise ValueError("formal output_root must be a directory")
    checked_metrics = validate_per_record_metrics(per_record_metrics)
    _validate_gate(gate)
    _require_finite_numeric_columns(summary, "summary")
    _require_finite_numeric_columns(mask_ledger, "mask_ledger")
    if coverage_ledger is not None:
        _require_finite_numeric_columns(coverage_ledger, "coverage_ledger")
    payloads = {
        "per_record_metrics.csv": _dataframe_bytes(checked_metrics, "per_record_metrics"),
        "summary.csv": _dataframe_bytes(summary, "summary"),
        "success_gate.json": (canonical_json(dict(gate)) + "\n").encode("utf-8"),
        "mask_ledger.csv": _dataframe_bytes(mask_ledger, "mask_ledger"),
    }
    if coverage_ledger is not None:
        payloads["coverage_ledger.csv"] = _dataframe_bytes(
            coverage_ledger, "coverage_ledger"
        )
    upstream_fixed = (
        "resolved_config.json", "window_identity_ledger.json", "frozen_models.json"
    )
    upstream_present = [name for name in upstream_fixed if (root / name).exists()]
    split_paths = sorted(root.glob("split_manifest-*.csv")) if root.exists() else []
    scaler_paths = sorted(root.glob("scaler-*.json")) if root.exists() else []
    if upstream_present or split_paths or scaler_paths:
        if len(upstream_present) != len(upstream_fixed) or len(split_paths) != 1 or len(scaler_paths) != 1:
            raise ValueError("formal provenance evidence is incomplete")
        for name in upstream_fixed:
            payloads[name] = (root / name).read_bytes()
        payloads[split_paths[0].name] = split_paths[0].read_bytes()
        payloads[scaler_paths[0].name] = scaler_paths[0].read_bytes()
    hashes = {
        name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()
    }
    payloads["artifact_hashes.json"] = (canonical_json(hashes) + "\n").encode("utf-8")

    # Preflight every existing artifact before creating anything new.
    for name, content in payloads.items():
        path = root / name
        if path.exists():
            if not path.is_file() or path.read_bytes() != content:
                raise FileExistsError(f"inconsistent prior formal artifact: {name}")
    root.mkdir(parents=True, exist_ok=True)
    # The hash manifest is the completion marker and is deliberately published
    # last. A crash can leave complete individual files, never a partial target;
    # rerun safely completes the set after the same-content preflight.
    ordered_names = [
        name for name in payloads if name != "artifact_hashes.json"
    ] + ["artifact_hashes.json"]
    for name in ordered_names:
        _write_stable(root / name, payloads[name])
    return hashes


@dataclass
class _FormalAssets:
    manifest: pd.DataFrame
    recordings: dict[str, tuple[Any, ...]]
    scaler: RobustTrainScaler
    split_hash: str
    scaler_hash: str
    git_commit: str
    dirty_digest: str
    rts_observation_var: float


class _SharedWindowRepository:
    def __init__(self, assets: _FormalAssets, config: TeacherConfig) -> None:
        self.assets = assets
        self.config = config
        self._identity_cache: dict[tuple[str, int, int], tuple[str, int]] = {}

    def iter(self, split: str, seed: int, samples: int):
        if split not in {"train", "validation"}:
            raise ValueError("shared training windows may only use train or validation")
        return iter_teacher_windows(
            self.assets.recordings[split],
            self.assets.scaler,
            window_samples=samples,
            stride=max(1, samples // 2),
            seed=seed,
            topologies=self.config.training_topologies,
            rates=self.config.training_rates,
            exhaustive=True,
        )

    def identity(self, split: str, seed: int, samples: int) -> tuple[str, int]:
        key = (split, seed, samples)
        if key not in self._identity_cache:
            identifiers = [window.window_id for window in self.iter(*key)]
            self._identity_cache[key] = (
                _sha256_bytes(canonical_json(identifiers).encode("utf-8")),
                len(identifiers),
            )
        return self._identity_cache[key]

    @property
    def cached_tensor_windows(self) -> int:
        return 0

    @property
    def identity_cache_cardinality(self) -> int:
        return len(self._identity_cache)


class _WindowIterableDataset(IterableDataset):
    def __init__(
        self, repository: _SharedWindowRepository, split: str, seed: int, samples: int
    ) -> None:
        super().__init__()
        self.repository = repository
        self.split = split
        self.seed = seed
        self.samples = samples

    def __iter__(self):
        return self.repository.iter(self.split, self.seed, self.samples)


@dataclass
class _Candidate:
    seed: int
    model_alias: str
    condition: str
    context_samples: int
    validation_rmse: float
    checkpoint_sha256: str
    checkpoint_path: Path
    capacity: dict[str, Any] | None = None
    validation_scores: tuple[dict[str, Any], ...] = ()
    inference_config: dict[str, Any] | None = None
    constructor_identity: dict[str, Any] | None = None
    train_window_ids_sha256: str = ""
    train_window_count: int = 0
    validation_window_ids_sha256: str = ""
    validation_window_count: int = 0


@dataclass
class _SelectedModels:
    candidates: dict[tuple[int, str], _Candidate]
    strongest_baseline: str


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formal_mask_seed(recording_id: str, seed: int, topology: str, rate: float) -> int:
    digest = hashlib.sha256(
        canonical_json(
            ["formal-test-mask-v1", recording_id, seed, topology, rate]
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _verify_source_sha(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or _sha256_path(path) != expected_sha256:
        raise ValueError(f"frozen source SHA256 mismatch: {path}")


def _read_verified_bytes(path: Path, expected_sha256: str, *, label: str) -> bytes:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    content = path.read_bytes()
    if _sha256_bytes(content) != expected_sha256:
        raise ValueError(f"{label} hash does not match frozen identity")
    return content


def _write_stable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise FileExistsError(f"inconsistent frozen asset: {path.name}")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if _sha256_path(temporary) != _sha256_bytes(content):
            raise OSError(f"temporary publish verification failed: {path.name}")
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != content:
                raise FileExistsError(f"inconsistent frozen asset: {path.name}")
        if not path.is_file() or _sha256_path(path) != _sha256_bytes(content):
            raise OSError(f"published asset verification failed: {path.name}")
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _relative_time_tensor(raw_time: Any) -> torch.Tensor:
    time64 = np.asarray(raw_time, dtype=np.float64)
    if time64.ndim != 1 or time64.size < 2 or not np.isfinite(time64).all():
        raise ValueError("recording time must be a finite one-dimensional sequence")
    if not np.all(np.diff(time64) > 0):
        raise ValueError("recording time must be strictly increasing")
    relative64 = time64 - time64[0]
    relative32 = relative64.astype(np.float32)
    if not np.isfinite(relative32).all() or not np.all(np.diff(relative32) > 0):
        raise ValueError("relative recording time must remain strictly increasing")
    return torch.from_numpy(np.array(relative32, copy=True))


def _release_predictor(predictor: Any) -> None:
    module = predictor if isinstance(predictor, torch.nn.Module) else None
    if module is None:
        outer = getattr(predictor, "model", None)
        inner = getattr(outer, "model", None)
        module = inner if isinstance(inner, torch.nn.Module) else outer
    if isinstance(module, torch.nn.Module):
        module.to(device="cpu")
    del predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _window_raw_predictions(
    condition: str,
    predictor: Any,
    windows: Sequence[PreparedWindow],
    device: torch.device,
) -> list[np.ndarray]:
    if condition in _CLASSICAL_MODELS:
        values = []
        for window in windows:
            kwargs = {"empty_fill": 0.0}
            if condition == "locf":
                raw = timestamp_locf(window.observed, window.mask, window.time, **kwargs)
            elif condition == "linear":
                raw = timestamp_linear(window.observed, window.mask, window.time, **kwargs)
            elif condition == "pchip":
                raw = timestamp_pchip(window.observed, window.mask, window.time, **kwargs)
            else:
                rts_parameters = predictor if isinstance(predictor, Mapping) else {}
                raw = constant_velocity_rts(
                    window.observed,
                    window.mask,
                    window.time,
                    empty_fill=0.0,
                    process_var=float(rts_parameters.get("process_var", 1.0)),
                    observation_var=float(
                        rts_parameters.get("observation_var", 1e-2)
                    ),
                )
            values.append(raw.detach().cpu().numpy())
        return values
    if condition in _PYPOTS_MODELS:
        target = np.stack([window.target.numpy() for window in windows])
        mask = np.stack([window.mask.numpy() for window in windows])
        completed = predictor.impute({"X": np.where(mask.astype(bool), target, np.nan)})
        return [completed[index] for index in range(len(completed))]

    loader = DataLoader(
        list(windows),
        batch_size=min(64, max(1, len(windows))),
        shuffle=False,
        collate_fn=collate_prepared_windows,
    )
    values: list[np.ndarray] = []
    predictor.eval()
    with torch.inference_mode():
        for batch in loader:
            output = predictor(
                batch.features.to(device),
                batch.dt.to(device),
                batch.observed.to(device),
                batch.mask.to(device),
                batch.baseline.to(device),
            )
            values.extend(output.raw.detach().cpu().numpy())
    return values


def _missing_rmse(
    predictions: Sequence[np.ndarray], windows: Sequence[PreparedWindow]
) -> float:
    if len(predictions) != len(windows) or not windows:
        raise ValueError("validation predictions must align with non-empty windows")
    squared = np.zeros(6, dtype=np.float64)
    counts = np.zeros(6, dtype=np.int64)
    for prediction, window in zip(predictions, windows):
        target = window.target.numpy()
        missing = window.mask.numpy() == 0
        error = np.asarray(prediction, dtype=np.float64) - target
        if not np.isfinite(error[missing]).all():
            raise ValueError("validation predictions must be finite at missing values")
        squared += np.where(missing, error * error, 0.0).sum(axis=0)
        counts += missing.sum(axis=0)
    represented = counts > 0
    if not represented.any():
        raise ValueError("validation windows must contain missing values")
    result = float(np.sqrt(np.mean(squared[represented] / counts[represented])))
    if not math.isfinite(result):
        raise ValueError("validation RMSE must be finite")
    return result


def _stream_missing_rmse(
    condition: str,
    predictor: Any,
    windows: Any,
    device: torch.device,
) -> float:
    squared = np.zeros(6, dtype=np.float64)
    counts = np.zeros(6, dtype=np.int64)
    seen = 0
    for window in windows:
        prediction = _window_raw_predictions(
            condition, predictor, [window], device
        )[0]
        target = window.target.numpy()
        missing = window.mask.numpy() == 0
        error = np.asarray(prediction, dtype=np.float64) - target
        if not np.isfinite(error[missing]).all():
            raise ValueError("validation predictions must be finite at missing values")
        squared += np.where(missing, error * error, 0.0).sum(axis=0)
        counts += missing.sum(axis=0)
        seen += 1
    represented = counts > 0
    if seen == 0 or not represented.any():
        raise ValueError("validation windows must contain missing values")
    return float(np.sqrt(np.mean(squared[represented] / counts[represented])))


class OXIODFormalBackend:
    """Concrete OXIOD implementation of the frozen formal protocol.

    Train/validation recordings are loaded and scaled before selection. Test
    file contents are not loaded until :meth:`load_test_data`, which the outer
    protocol calls only after all selected checkpoint identities are frozen.
    """

    def __init__(
        self,
        config: TeacherConfig,
        *,
        repository_root: Path,
        output_root: Path | None,
        requested_device: str,
        discover_pairs=discover_oxiod_pairs,
        splitter=stratified_file_split,
        recording_loader=load_recording,
        source_verifier=_verify_source_sha,
    ) -> None:
        self.config = _require_config(config)
        self.repository_root = Path(repository_root).resolve()
        selected_output = output_root or config.output_root
        self.output_root = (
            selected_output
            if Path(selected_output).is_absolute()
            else self.repository_root / selected_output
        ).resolve()
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        resolved = "cuda" if requested_device == "cuda" or (
            requested_device == "auto" and torch.cuda.is_available()
        ) else "cpu"
        self.device = torch.device(resolved)
        self._discover_pairs = discover_pairs
        self._splitter = splitter
        self._recording_loader = recording_loader
        self._source_verifier = source_verifier
        self._assets: _FormalAssets | None = None
        self._frozen: _SelectedModels | None = None
        self._test_loaded = False
        self._evaluated_cells: set[tuple[int, str, str]] = set()
        self._resolved_config_sha256 = ""
        self._matrix_plan_sha256 = ""
        self._window_identity_ledger_sha256 = ""

    def _seal_resolved_config(self) -> None:
        if self._resolved_config_sha256:
            return
        plan = formal_matrix_plan(self.config)
        config_payload = json.loads(canonical_json(self.config))
        resolved_payload = {
            "config": config_payload,
            "device": str(self.device),
            "output_root": str(self.output_root),
        }
        config_sha256 = _sha256_bytes(
            canonical_json(resolved_payload).encode("utf-8")
        )
        matrix_sha256 = _sha256_bytes(canonical_json(plan).encode("utf-8"))
        document = {
            "schema": "imputation-v3-formal-resolved-config-v1",
            "resolved": resolved_payload,
            "resolved_config_sha256": config_sha256,
            "matrix_plan_sha256": matrix_sha256,
        }
        _write_stable(
            self.output_root / "resolved_config.json",
            (canonical_json(document) + "\n").encode("utf-8"),
        )
        self._resolved_config_sha256 = config_sha256
        self._matrix_plan_sha256 = matrix_sha256

    def _verify_split_sources(self, manifest: pd.DataFrame, split: str) -> None:
        rows = manifest.loc[manifest["split"] == split]
        for row in rows.itertuples(index=False):
            self._source_verifier(Path(row.imu_path), str(row.imu_sha256))
            self._source_verifier(Path(row.vicon_path), str(row.vicon_sha256))

    def _load_verified_recording(self, row: Any) -> Any:
        imu_path = Path(row.imu_path)
        vicon_path = Path(row.vicon_path)
        self._source_verifier(imu_path, str(row.imu_sha256))
        self._source_verifier(vicon_path, str(row.vicon_sha256))
        recording = self._recording_loader(imu_path, vicon_path)
        # The legacy OXIOD loader is path-based.  A second verification closes
        # the mutation-during-parse window and rejects any mixed snapshot.
        self._source_verifier(imu_path, str(row.imu_sha256))
        self._source_verifier(vicon_path, str(row.vicon_sha256))
        return recording

    def load_frozen_manifest_and_scaler(self) -> _FormalAssets:
        if self._assets is not None:
            return self._assets
        self._seal_resolved_config()
        data_root = self.config.data_root
        if not data_root.is_absolute():
            data_root = self.repository_root / data_root
        manifest = self._splitter(self._discover_pairs(data_root), seed=2026)
        if list(manifest.columns) != list(MANIFEST_COLUMNS):
            raise ValueError("formal split manifest must use the frozen v2 schema")
        if set(manifest["split"]) != {"train", "validation", "test"}:
            raise ValueError("formal split requires non-empty train, validation, and test")
        manifest = manifest.sort_values("recording_id", ignore_index=True)
        split_content = manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
        split_hash = _sha256_bytes(split_content)
        _write_stable(
            self.output_root / f"split_manifest-{split_hash}.csv", split_content
        )

        loaded: dict[str, tuple[Any, ...]] = {}
        for split in ("train", "validation"):
            rows = manifest.loc[manifest["split"] == split]
            self._verify_split_sources(manifest, split)
            recordings = tuple(
                self._load_verified_recording(row)
                for row in rows.itertuples(index=False)
            )
            if tuple(recording.id for recording in recordings) != tuple(rows["recording_id"]):
                raise ValueError(f"loaded {split} recording identities do not match manifest")
            loaded[split] = recordings
        train_ids = {recording.id for recording in loaded["train"]}
        scaler = RobustTrainScaler.fit(loaded["train"], allowed_ids=train_ids)
        scaler_payload = {
            "center": scaler.center_.tolist(),
            "scale": scaler.scale_.tolist(),
            "training_ids": list(scaler.training_ids),
            "split_hash": split_hash,
        }
        scaler_content = (canonical_json(scaler_payload) + "\n").encode("utf-8")
        scaler_hash = _sha256_bytes(scaler_content)
        _write_stable(self.output_root / f"scaler-{scaler_hash}.json", scaler_content)
        identity = git_worktree_identity(self.repository_root)
        self._assets = _FormalAssets(
            manifest=manifest,
            recordings=loaded,
            scaler=scaler,
            split_hash=split_hash,
            scaler_hash=scaler_hash,
            git_commit=identity["git_commit"],
            dirty_digest=identity["dirty_state_digest"],
            rts_observation_var=estimate_rts_observation_variance(
                loaded["train"], scaler
            ),
        )
        return self._assets

    def materialize_train_validation(
        self, assets: _FormalAssets, plan: Mapping
    ) -> _SharedWindowRepository:
        if assets is not self._assets or plan.get("selection_split") != "validation":
            raise ValueError("formal assets or selection plan identity changed")
        self._verify_split_sources(assets.manifest, "train")
        self._verify_split_sources(assets.manifest, "validation")
        return _SharedWindowRepository(assets, self.config)

    def _classical_checkpoint(self, candidate_config: Mapping[str, Any]) -> tuple[Path, str]:
        content = (canonical_json(candidate_config) + "\n").encode("utf-8")
        digest = _sha256_bytes(content)
        path = self.output_root / "candidates" / f"classical-{digest}.json"
        _write_stable(path, content)
        return path, digest

    def _native_candidate(
        self,
        condition: str,
        seed: int,
        samples: int,
        windows: _SharedWindowRepository,
        capacity: Mapping[str, Any],
    ) -> _Candidate:
        _seed_everything(seed)
        model = build_native_model(
            condition, self.config, capacity=capacity
        ).to(self.device)
        train_loader = DataLoader(
            _WindowIterableDataset(windows, "train", seed, samples),
            batch_size=self.config.batch_size,
            collate_fn=collate_prepared_windows,
        )
        validation_loader = DataLoader(
            _WindowIterableDataset(windows, "validation", seed, samples),
            batch_size=self.config.batch_size,
            collate_fn=collate_prepared_windows,
        )
        train_identity, train_count = windows.identity("train", seed, samples)
        validation_identity, validation_count = windows.identity(
            "validation", seed, samples
        )
        resolved = {
            "mode": "imputation_v3_formal_candidate",
            "selection_split": "validation",
            "condition": condition,
            "seed": seed,
            "context_samples": samples,
            "device": str(self.device),
            "capacity": dict(capacity),
            "hyperparameters": {
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "hidden_size": self.config.hidden_size,
                "tcn_width": self.config.tcn_width,
                "tcn_dilations": list(self.config.tcn_dilations),
                "learning_rate": self.config.learning_rate,
                "training_topologies": list(self.config.training_topologies),
                "training_rates": list(self.config.training_rates),
            },
            "train_window_ids_sha256": train_identity,
            "train_window_count": train_count,
            "validation_window_ids_sha256": validation_identity,
            "validation_window_count": validation_count,
        }
        provenance = collect_provenance(
            resolved, seed, split_hash=windows.assets.split_hash,
            scaler_hash=windows.assets.scaler_hash,
            git_commit=windows.assets.git_commit,
            dirty_digest=windows.assets.dirty_digest,
        )
        run_dir = self.output_root / "candidates" / provenance["run_id"]
        metadata_path = run_dir / "checkpoint.json"
        expected = None
        if metadata_path.is_file():
            expected = json.loads(metadata_path.read_text(encoding="utf-8")).get(
                "checkpoint_sha256"
            )
        train_epoch, eval_epoch = make_teacher_callbacks(self.device)
        metadata = train_one_run(
            run_dir, provenance, train_loader=train_loader,
            validation_loader=validation_loader, epochs=self.config.epochs,
            train_epoch=train_epoch, evaluate_epoch=eval_epoch, model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=self.config.learning_rate),
            expected_checkpoint_sha256=expected,
        )
        checkpoint = run_dir / "best.pt"
        frozen_bytes = _read_verified_bytes(
            checkpoint, metadata["checkpoint_sha256"], label="native checkpoint"
        )
        state = torch.load(
            io.BytesIO(frozen_bytes), map_location=self.device, weights_only=True
        )
        if _sha256_path(checkpoint) != metadata["checkpoint_sha256"]:
            raise ValueError("native checkpoint changed during deserialization")
        model.load_state_dict(state)
        score = float(eval_epoch(model, validation_loader, 0)["missing_rmse"])
        candidate = _Candidate(
            seed, condition, condition, samples, score,
            metadata["checkpoint_sha256"], checkpoint,
            capacity=dict(capacity),
            inference_config={"kind": "native"},
        )
        _release_predictor(model)
        return candidate

    def _pypots_candidate(
        self,
        condition: str,
        seed: int,
        samples: int,
        windows: _SharedWindowRepository,
    ) -> _Candidate:
        _seed_everything(seed)
        train_windows = tuple(windows.iter("train", seed, samples))
        validation_windows = tuple(windows.iter("validation", seed, samples))
        train_data = SimpleNamespace(
            target=np.stack([window.target.numpy() for window in train_windows]),
            mask=np.stack([window.mask.numpy() for window in train_windows]),
        )
        validation_data = SimpleNamespace(
            target=np.stack([window.target.numpy() for window in validation_windows]),
            mask=np.stack([window.mask.numpy() for window in validation_windows]),
        )
        train_set, validation_set = to_pypots_sets(train_data, validation_data)
        saving_path = self.output_root / "candidates" / f"{condition}-{seed}-{samples}"
        adapter = build_pypots_model(
            condition, n_steps=samples, epochs=self.config.epochs,
            batch_size=self.config.batch_size, device=str(self.device),
            saving_path=saving_path,
        )
        adapter.fit(train_set, validation_set)
        completed = adapter.impute({"X": validation_set["X"]})
        score = _missing_rmse(
            [completed[index] for index in range(len(completed))], validation_windows
        )
        checkpoint = saving_path / "formal-best.pt"
        outer = adapter.model
        inner = _pypots_state_target(adapter)
        actual_version = installed_pypots_version()
        constructor_identity = {
            "pypots_version": actual_version,
            "condition": condition,
            "n_steps": samples,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "outer_class": f"{type(outer).__module__}.{type(outer).__qualname__}",
            "inner_class": f"{type(inner).__module__}.{type(inner).__qualname__}",
        }
        digest = freeze_pypots_predictor(
            adapter,
            checkpoint,
            constructor_identity=constructor_identity,
        )
        candidate = _Candidate(
            seed, condition, condition, samples, score, digest, checkpoint,
            capacity={"capacity": "fixed"},
            inference_config={"kind": "pypots", "pypots_version": actual_version},
            constructor_identity=constructor_identity,
        )
        _release_predictor(adapter)
        del train_windows, validation_windows, train_data, validation_data
        return candidate

    def train_select_validation(
        self, windows: _SharedWindowRepository, plan: Mapping
    ) -> dict[tuple[int, str], _Candidate]:
        if not isinstance(windows, _SharedWindowRepository):
            raise TypeError("formal training requires the shared Task7 window repository")
        # Re-hash the immutable training inputs at the last boundary before any
        # candidate is fitted or selected.  This catches edits made after window
        # materialisation instead of silently training against stale provenance.
        self._verify_split_sources(windows.assets.manifest, "train")
        self._verify_split_sources(windows.assets.manifest, "validation")
        identity_entries = []
        for seed in FORMAL_SEEDS:
            for context in plan["contexts"]:
                samples = int(context["samples"])
                for split in ("train", "validation"):
                    digest, count = windows.identity(split, seed, samples)
                    identity_entries.append(
                        {
                            "seed": seed,
                            "context_samples": samples,
                            "split": split,
                            "window_ids_sha256": digest,
                            "window_count": count,
                        }
                    )
        ledger_document = {
            "schema": "imputation-v3-formal-window-identities-v1",
            "entries": identity_entries,
            "entries_sha256": _sha256_bytes(
                canonical_json(identity_entries).encode("utf-8")
            ),
        }
        ledger_path = self.output_root / "window_identity_ledger.json"
        _write_stable(
            ledger_path, (canonical_json(ledger_document) + "\n").encode("utf-8")
        )
        self._window_identity_ledger_sha256 = _sha256_path(ledger_path)
        best: dict[tuple[int, str], _Candidate] = {}
        score_ledger: dict[tuple[int, str], list[dict[str, Any]]] = {}
        seen: set[tuple[int, str, int, str]] = set()
        for cell in plan["cells"]:
            seed = int(cell["seed"])
            condition = str(cell["condition"])
            alias = str(cell["model"])
            samples = int(cell["context_samples"])
            for raw_capacity in cell["capacity_candidates"]:
                capacity = dict(raw_capacity)
                capacity_json = canonical_json(capacity)
                identity = (seed, condition, samples, capacity_json)
                if identity in seen:
                    raise ValueError("formal candidate matrix contains duplicates")
                seen.add(identity)
                if condition in _CLASSICAL_MODELS:
                    predictor = (
                        {
                            "process_var": float(capacity["process_var"]),
                            "observation_var": windows.assets.rts_observation_var,
                        }
                        if condition == "rts"
                        else None
                    )
                    score = _stream_missing_rmse(
                        condition,
                        predictor,
                        windows.iter("validation", seed, samples),
                        self.device,
                    )
                    train_identity, train_count = windows.identity(
                        "train", seed, samples
                    )
                    validation_identity, validation_count = windows.identity(
                        "validation", seed, samples
                    )
                    descriptor = {
                        "kind": "classical", "condition": condition, "seed": seed,
                        "context_samples": samples, "capacity": capacity,
                        "selection_split": "validation", "validation_rmse": score,
                        "split_hash": windows.assets.split_hash,
                        "scaler_hash": windows.assets.scaler_hash,
                        "train_window_ids_sha256": train_identity,
                        "train_window_count": train_count,
                        "validation_window_ids_sha256": validation_identity,
                        "validation_window_count": validation_count,
                    }
                    if condition == "rts":
                        descriptor.update(
                            observation_var=windows.assets.rts_observation_var,
                            process_var=float(capacity["process_var"]),
                        )
                    path, digest = self._classical_checkpoint(descriptor)
                    candidate = _Candidate(
                        seed, alias, condition, samples, score, digest, path,
                        capacity=capacity,
                        inference_config=dict(predictor or {}),
                    )
                elif condition in _PYPOTS_MODELS:
                    candidate = self._pypots_candidate(
                        condition, seed, samples, windows
                    )
                    candidate.model_alias = alias
                else:
                    candidate = self._native_candidate(
                        condition, seed, samples, windows, capacity,
                    )
                    candidate.model_alias = alias
                train_identity, train_count = windows.identity("train", seed, samples)
                validation_identity, validation_count = windows.identity(
                    "validation", seed, samples
                )
                candidate.train_window_ids_sha256 = train_identity
                candidate.train_window_count = train_count
                candidate.validation_window_ids_sha256 = validation_identity
                candidate.validation_window_count = validation_count
                key = (seed, condition)
                score_ledger.setdefault(key, []).append(
                    {
                        "context_samples": samples,
                        "capacity": capacity,
                        "validation_rmse": candidate.validation_rmse,
                    }
                )
                current = best.get(key)
                candidate_key = (
                    candidate.validation_rmse,
                    samples,
                    capacity_json,
                )
                current_key = (
                    current.validation_rmse,
                    current.context_samples,
                    canonical_json(current.capacity or {}),
                ) if current is not None else None
                if current is None or candidate_key < current_key:
                    best[key] = candidate
        for key, candidate in best.items():
            candidate.validation_scores = tuple(
                sorted(
                    score_ledger[key],
                    key=lambda item: (
                        item["context_samples"], canonical_json(item["capacity"])
                    ),
                )
            )
            if candidate.condition == "rts":
                selected_descriptor = {
                    "kind": "classical",
                    "condition": candidate.condition,
                    "seed": candidate.seed,
                    "context_samples": candidate.context_samples,
                    "capacity": dict(candidate.capacity or {}),
                    "selection_split": "validation",
                    "process_var": candidate.capacity["process_var"],
                    "observation_var": windows.assets.rts_observation_var,
                    "validation_scores": list(candidate.validation_scores),
                    "split_hash": windows.assets.split_hash,
                    "scaler_hash": windows.assets.scaler_hash,
                }
                path, digest = self._classical_checkpoint(selected_descriptor)
                candidate.checkpoint_path = path
                candidate.checkpoint_sha256 = digest
        return best

    def freeze_checkpoints(
        self, selected: dict[tuple[int, str], _Candidate], plan: Mapping
    ) -> _SelectedModels:
        expected = {(int(cell["seed"]), str(cell["condition"])) for cell in plan["cells"]}
        if set(selected) != expected:
            raise ValueError("selected checkpoint matrix is incomplete")
        for candidate in selected.values():
            if (
                not candidate.checkpoint_path.is_file()
                or _sha256_path(candidate.checkpoint_path)
                != candidate.checkpoint_sha256
            ):
                raise ValueError("selected checkpoint identity does not match frozen bytes")
        baselines = [
            candidate for candidate in selected.values()
            if candidate.condition not in TEACHER_CONDITION_MODES
        ]
        if not baselines:
            raise ValueError("formal gate requires at least one eligible baseline")
        means: dict[str, list[float]] = {}
        for candidate in baselines:
            means.setdefault(candidate.condition, []).append(candidate.validation_rmse)
        strongest = min(means, key=lambda name: (float(np.mean(means[name])), name))
        manifest = {
            "selection_split": "validation",
            "split_hash": self._assets.split_hash,
            "scaler_hash": self._assets.scaler_hash,
            "git_commit": self._assets.git_commit,
            "dirty_state_digest": self._assets.dirty_digest,
            "resolved_config_sha256": self._resolved_config_sha256,
            "matrix_plan_sha256": self._matrix_plan_sha256,
            "window_identity_ledger_sha256": self._window_identity_ledger_sha256,
            "strongest_baseline": strongest,
            "checkpoints": [
                {
                    "seed": seed,
                    "condition": condition,
                    "context_samples": candidate.context_samples,
                    "capacity": candidate.capacity,
                    "validation_rmse": candidate.validation_rmse,
                    "validation_scores": list(candidate.validation_scores),
                    "checkpoint_sha256": candidate.checkpoint_sha256,
                    "checkpoint_path": str(candidate.checkpoint_path),
                    "inference_config": candidate.inference_config,
                    "constructor_identity": candidate.constructor_identity,
                    "train_window_ids_sha256": candidate.train_window_ids_sha256,
                    "train_window_count": candidate.train_window_count,
                    "validation_window_ids_sha256": candidate.validation_window_ids_sha256,
                    "validation_window_count": candidate.validation_window_count,
                }
                for (seed, condition), candidate in sorted(selected.items())
            ],
        }
        _write_stable(
            self.output_root / "frozen_models.json",
            (canonical_json(manifest) + "\n").encode("utf-8"),
        )
        self._frozen = _SelectedModels(selected, strongest)
        return self._frozen

    def load_test_data(self, assets: _FormalAssets) -> tuple[Any, ...]:
        if assets is not self._assets:
            raise ValueError("test loader received different frozen assets")
        if self._frozen is None:
            raise RuntimeError("test data cannot be loaded before checkpoint freeze")
        for candidate in self._frozen.candidates.values():
            if (
                not candidate.checkpoint_path.is_file()
                or _sha256_path(candidate.checkpoint_path)
                != candidate.checkpoint_sha256
            ):
                raise ValueError("frozen checkpoint identity changed before test load")
        if self._test_loaded:
            raise RuntimeError("formal test data may be loaded exactly once")
        self._verify_split_sources(assets.manifest, "test")
        rows = assets.manifest.loc[assets.manifest["split"] == "test"]
        recordings = tuple(
            self._load_verified_recording(row)
            for row in rows.itertuples(index=False)
        )
        if tuple(recording.id for recording in recordings) != tuple(rows["recording_id"]):
            raise ValueError("loaded test recording identities do not match manifest")
        self._test_loaded = True
        return recordings

    @staticmethod
    def _full_mask(target: torch.Tensor, recording_id: str, seed: int, topology: str, rate: float):
        condition_seed = formal_mask_seed(recording_id, seed, topology, rate)
        generator = {"point": point_missing, "block": contiguous_block, "channel": channel_outage}[topology]
        return generator(target, rate, condition_seed).mask

    def _predict_full(
        self,
        candidate: _Candidate,
        predictor: Any,
        target: torch.Tensor,
        mask: torch.Tensor,
        time: torch.Tensor,
    ) -> np.ndarray:
        observed = torch.where(mask.bool(), target, torch.zeros_like(target))
        length = len(target)
        samples = candidate.context_samples
        if length < samples:
            raise ValueError("test recording is shorter than selected context")
        stride = max(1, samples // 2)
        starts = list(range(0, length - samples + 1, stride))
        if starts[-1] != length - samples:
            starts.append(length - samples)
        if candidate.condition in _CLASSICAL_MODELS:
            windows = [
                SimpleNamespace(
                    observed=observed[start:start + samples],
                    mask=mask[start:start + samples],
                    time=time[start:start + samples] - time[start],
                )
                for start in starts
            ]
            predicted = _window_raw_predictions(
                candidate.condition, predictor, windows, self.device
            )
            return self._overlap_predictions(predicted, starts, target.shape)
        delta = torch.empty_like(time)
        delta[1:] = time[1:] - time[:-1]
        delta[0] = delta[1] if len(delta) > 1 else torch.tensor(
            self.config.nominal_dt_s, dtype=time.dtype
        )
        features = build_features(target, mask, delta).values
        baseline = timestamp_linear(observed, mask, time, empty_fill=0.0)
        if candidate.condition in _PYPOTS_MODELS:
            x = np.stack([
                np.where(mask[start:start + samples].numpy().astype(bool),
                         target[start:start + samples].numpy(), np.nan)
                for start in starts
            ])
            predicted = predictor.impute({"X": x})
        else:
            predictor.eval()
            predicted_parts = []
            with torch.inference_mode():
                for offset in range(0, len(starts), self.config.batch_size):
                    group = starts[offset:offset + self.config.batch_size]
                    output = predictor(
                        torch.stack([features[s:s + samples] for s in group]).to(self.device),
                        torch.stack([delta[s:s + samples] for s in group]).to(self.device),
                        torch.stack([observed[s:s + samples] for s in group]).to(self.device),
                        torch.stack([mask[s:s + samples] for s in group]).to(self.device),
                        torch.stack([baseline[s:s + samples] for s in group]).to(self.device),
                    )
                    predicted_parts.extend(output.raw.detach().cpu().numpy())
            predicted = np.asarray(predicted_parts)
        return self._overlap_predictions(predicted, starts, target.shape)

    def _load_candidate_predictor(self, candidate: _Candidate) -> Any:
        frozen_bytes = _read_verified_bytes(
            candidate.checkpoint_path,
            candidate.checkpoint_sha256,
            label=f"{candidate.condition} checkpoint",
        )
        if candidate.condition in _CLASSICAL_MODELS:
            descriptor = json.loads(frozen_bytes.decode("utf-8"))
            if not isinstance(descriptor, dict) or descriptor.get("kind") != "classical":
                raise ValueError("frozen classical checkpoint kind changed")
            if descriptor.get("condition") != candidate.condition:
                raise ValueError("frozen classical checkpoint condition changed")
            if (
                type(descriptor.get("seed")) is not int
                or descriptor["seed"] != candidate.seed
            ):
                raise ValueError("frozen classical checkpoint seed changed")
            if (
                type(descriptor.get("context_samples")) is not int
                or descriptor["context_samples"] != candidate.context_samples
            ):
                raise ValueError("frozen classical checkpoint context changed")
            if (
                not isinstance(descriptor.get("capacity"), dict)
                or descriptor["capacity"] != candidate.capacity
            ):
                raise ValueError("frozen classical checkpoint capacity changed")
            if candidate.condition != "rts":
                return {}
            predictor = {}
            for name in ("process_var", "observation_var"):
                value = descriptor.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                ):
                    raise ValueError(f"frozen RTS {name} must be finite and positive")
                predictor[name] = float(value)
            if predictor["process_var"] != descriptor["capacity"].get("process_var"):
                raise ValueError("frozen RTS process variance does not match capacity")
            return predictor
        if candidate.condition in _PYPOTS_MODELS:
            actual_version = installed_pypots_version()
            if (
                candidate.inference_config is None
                or candidate.inference_config.get("pypots_version") != actual_version
                or candidate.constructor_identity is None
            ):
                raise ValueError("frozen PyPOTS version or constructor identity changed")
            adapter = build_pypots_model(
                candidate.condition,
                n_steps=candidate.context_samples,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                device=str(self.device),
                saving_path=self.output_root / "evaluation" / (
                    f"{candidate.condition}-{candidate.seed}"
                ),
            )
            reload_pypots_predictor(
                adapter,
                candidate.checkpoint_path,
                expected_sha256=candidate.checkpoint_sha256,
                constructor_identity=candidate.constructor_identity,
            )
            return adapter
        model = build_native_model(
            candidate.condition, self.config, capacity=candidate.capacity
        ).to(self.device)
        state = torch.load(
            io.BytesIO(frozen_bytes), map_location=self.device, weights_only=True
        )
        if _sha256_path(candidate.checkpoint_path) != candidate.checkpoint_sha256:
            raise ValueError("native checkpoint changed during deserialization")
        model.load_state_dict(state, strict=True)
        return model


    @staticmethod
    def _overlap_predictions(
        predicted: Sequence[np.ndarray],
        starts: Sequence[int],
        target_shape: Sequence[int],
    ) -> np.ndarray:
        total = np.zeros(target_shape, dtype=np.float64)
        length = int(target_shape[0])
        count = np.zeros((length, 1), dtype=np.int64)
        for raw, start in zip(predicted, starts):
            stop = start + len(raw)
            total[start:stop] += raw
            count[start:stop] += 1
        if np.any(count == 0):
            raise RuntimeError("formal stitched prediction left uncovered samples")
        return total / count

    def evaluate_test_once(
        self, frozen: _SelectedModels, test_data: tuple[Any, ...], plan: Mapping
    ) -> Mapping[str, Any]:
        if frozen is not self._frozen or not self._test_loaded:
            raise ValueError("test evaluation requires frozen models and one loaded test set")
        scenarios = dict(zip(self._assets.manifest["recording_id"], self._assets.manifest["scenario"]))
        source_hashes = dict(
            zip(self._assets.manifest["recording_id"], self._assets.manifest["imu_sha256"])
        )
        rows: list[pd.DataFrame] = []
        ledger: dict[tuple[int, str, str, float], dict[str, Any]] = {}
        for (seed, condition), candidate in sorted(frozen.candidates.items()):
            predictor = self._load_candidate_predictor(candidate)
            try:
                for recording in test_data:
                    cell = (seed, condition, recording.id)
                    if cell in self._evaluated_cells:
                        raise RuntimeError("each recording/model/seed may be evaluated exactly once")
                    self._evaluated_cells.add(cell)
                    target_np = self._assets.scaler.transform(recording.imu_six)
                    target = torch.as_tensor(target_np, dtype=torch.float32)
                    time = _relative_time_tensor(recording.imu_time_s)
                    for topology in self.config.training_topologies:
                        for rate in self.config.training_rates:
                            key = (seed, recording.id, topology, rate)
                            mask = self._full_mask(target, recording.id, seed, topology, rate)
                            mask_bytes = np.ascontiguousarray(mask.numpy()).tobytes()
                            mask_sha = _sha256_bytes(mask_bytes)
                            prior = ledger.get(key)
                            entry = {
                                "seed": seed, "recording_id": recording.id,
                                "topology": topology, "requested_fraction": rate,
                                "realized_fraction": float((mask == 0).double().mean()),
                                "mask_sha256": mask_sha,
                                "generator": "formal-test-mask-v1",
                                "condition_seed": formal_mask_seed(
                                    recording.id, seed, topology, rate
                                ),
                                "target_source_sha256": str(source_hashes[recording.id]),
                                "target_length": int(target.shape[0]),
                                "channels": int(target.shape[1]),
                            }
                            if prior is not None and prior != entry:
                                raise RuntimeError("formal masks changed across models")
                            ledger[key] = entry
                            raw = self._predict_full(
                                candidate, predictor, target, mask, time
                            )
                            rows.append(evaluate_record_rows(
                                raw_windows=[raw], starts=[0], target_normalized=target_np,
                                observed_mask=mask.numpy(), time=recording.imu_time_s,
                                scaler=self._assets.scaler, run_id=f"formal-{seed}-{condition}",
                                seed=seed, recording_id=recording.id,
                                scenario=str(scenarios[recording.id]), topology=topology,
                                requested_fraction=rate,
                                realized_fraction=entry["realized_fraction"], model=condition,
                                checkpoint_sha256=candidate.checkpoint_sha256,
                            ))
            finally:
                _release_predictor(predictor)
                predictor = None
        metrics = pd.concat(rows, ignore_index=True)
        return {
            "per_record_metrics": metrics,
            "mask_ledger": pd.DataFrame(list(ledger.values())),
            "strongest_baseline": frozen.strongest_baseline,
            "primary_candidate_model": "teacher_actual_residual",
            "required_scenarios": sorted(set(scenarios.values())),
        }


class FormalBackend(Protocol):
    """Phase-separated backend that makes pre-freeze test access impossible by API."""

    def load_frozen_manifest_and_scaler(self) -> object: ...
    def materialize_train_validation(self, assets: object, plan: Mapping) -> object: ...
    def train_select_validation(self, windows: object, plan: Mapping) -> object: ...
    def freeze_checkpoints(self, selected: object, plan: Mapping) -> object: ...
    def load_test_data(self, assets: object) -> object: ...
    def evaluate_test_once(
        self, frozen: object, test_data: object, plan: Mapping
    ) -> Mapping[str, Any]: ...


def _backend_method(backend: object, name: str):
    method = getattr(backend, name, None)
    if not callable(method):
        raise TypeError(f"formal backend must provide callable {name}")
    return method


def run_formal_protocol(
    config: TeacherConfig,
    *,
    backend: FormalBackend,
    output_root: Path,
) -> dict[str, Any]:
    """Execute the preregistered phases in the only permitted leakage-safe order."""
    config = _require_config(config)
    if config.seeds != FORMAL_SEEDS:
        raise ValueError("formal protocol requires seeds 2026 through 2030 in order")
    plan = formal_matrix_plan(config)

    methods = {
        name: _backend_method(backend, name)
        for name in (
            "load_frozen_manifest_and_scaler",
            "materialize_train_validation",
            "train_select_validation",
            "freeze_checkpoints",
            "load_test_data",
            "evaluate_test_once",
        )
    }

    # Deliberately keep test loading out of every pre-freeze method signature.
    assets = methods["load_frozen_manifest_and_scaler"]()
    windows = methods["materialize_train_validation"](assets, plan)
    selected = methods["train_select_validation"](windows, plan)
    frozen = methods["freeze_checkpoints"](selected, plan)
    if frozen is None:
        raise ValueError("freeze_checkpoints must return frozen checkpoint identities")
    test_data = methods["load_test_data"](assets)
    evaluation = methods["evaluate_test_once"](frozen, test_data, plan)
    if not isinstance(evaluation, Mapping):
        raise TypeError("formal evaluation must return a mapping")
    required = {"per_record_metrics", "mask_ledger", "strongest_baseline"}
    missing = sorted(required - set(evaluation))
    if missing:
        raise ValueError("formal evaluation is missing: " + ", ".join(missing))
    strongest = evaluation["strongest_baseline"]
    if not isinstance(strongest, str) or not strongest:
        raise ValueError("strongest_baseline must be a non-empty string")
    condition_metrics = validate_per_record_metrics(evaluation["per_record_metrics"])
    candidate_model = evaluation.get(
        "primary_candidate_model", "teacher_actual_residual"
    )
    if not isinstance(candidate_model, str):
        raise TypeError("primary_candidate_model must be a string")
    summary, primary, coverage = paired_formal_summaries(
        condition_metrics,
        candidate_model=candidate_model,
        strongest_baseline=strongest,
        required_topologies=config.training_topologies,
        required_rates=config.training_rates,
        required_scenarios=tuple(
            evaluation.get(
                "required_scenarios",
                sorted(set(condition_metrics["scenario"].astype(str))),
            )
        ),
        required_seeds=FORMAL_SEEDS,
    )
    gate = success_gate_payload(summary, strongest_baseline=strongest)
    all_metrics = pd.concat((condition_metrics, primary), ignore_index=True)
    hashes = write_formal_artifacts(
        output_root,
        all_metrics,
        summary,
        gate,
        evaluation["mask_ledger"],
        coverage,
    )
    return {
        "status": "completed",
        "matrix_cells": plan["counts"]["matrix_cells"],
        "strongest_baseline": strongest,
        "gate": gate,
        "artifact_hashes": hashes,
    }


__all__ = [
    "FORMAL_SEEDS",
    "RTS_PROCESS_VARIANCES",
    "FormalBackend",
    "OXIODFormalBackend",
    "build_native_model",
    "capacity_candidates",
    "estimate_rts_observation_variance",
    "evaluate_record_rows",
    "formal_matrix_plan",
    "freeze_pypots_predictor",
    "make_primary_rows",
    "paired_formal_summaries",
    "run_formal_protocol",
    "reload_pypots_predictor",
    "success_gate_payload",
    "teacher_success",
    "write_formal_artifacts",
]
