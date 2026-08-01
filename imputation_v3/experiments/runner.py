"""Leakage-ordered formal matrix orchestration and the teacher accuracy gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from numbers import Real
import os
from pathlib import Path
import re
from typing import Any, Final, Protocol

import numpy as np
import pandas as pd

from imputation_v3.config import TeacherConfig
from imputation_v3.experiments.evaluate import evaluate_record_diagnostics
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
from validation_v2.evaluation.statistics import (
    PER_RECORD_COLUMNS,
    paired_model_summary,
    validate_per_record_metrics,
)
from validation_v2.experiments.provenance import canonical_json


FORMAL_SEEDS: Final[tuple[int, ...]] = (2026, 2027, 2028, 2029, 2030)
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


def _require_config(config: object) -> TeacherConfig:
    if not isinstance(config, TeacherConfig):
        raise TypeError("config must be TeacherConfig")
    if config.selection_split != "validation":
        raise ValueError("formal selection_split must be validation")
    return config


def build_native_model(condition: str, config: TeacherConfig):
    """Build one native control from Task 9's frozen explicit condition names."""
    config = _require_config(config)
    if not isinstance(condition, str):
        raise TypeError("native condition must be a string")
    if condition == "bilstm":
        return BiLSTMControl(31, config.hidden_size)
    if condition == "bilnn":
        return BiCfCControl(31, config.hidden_size)
    if condition == "tcn":
        return TCNControl(31, config.tcn_width, config.tcn_dilations)
    if condition == "feature_mlp":
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
        config.hidden_size,
        config.tcn_width,
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
    required = {"model", "baseline", "metric", "ci95_low", "ci95_high"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError("summary is missing required columns: " + ", ".join(missing))
    selected = summary.loc[
        (summary["model"] == "teacher")
        & (summary["baseline"] == strongest_baseline)
        & (summary["metric"] == "rmse_physical")
    ]
    preregistered = {
        "scenario": "all",
        "protocol": "teacher_primary",
        "topology": "all",
    }
    for column, expected in preregistered.items():
        if column in selected.columns:
            selected = selected.loc[selected[column] == expected]
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
    teacher_conditions: Sequence[str],
    strongest_baseline: str,
) -> pd.DataFrame:
    """Collapse condition RMSE squared errors into the preregistered stratum."""
    checked = validate_per_record_metrics(metrics)
    conditions = tuple(teacher_conditions)
    if not conditions or any(not isinstance(value, str) or not value for value in conditions):
        raise ValueError("teacher_conditions must contain non-empty strings")
    if len(set(conditions)) != len(conditions):
        raise ValueError("teacher_conditions must be unique")
    if not isinstance(strongest_baseline, str) or not strongest_baseline:
        raise ValueError("strongest_baseline must be a non-empty string")
    selected = checked.loc[
        (checked["metric"] == "rmse_physical")
        & checked["model"].isin((*conditions, strongest_baseline))
    ].copy()
    if selected.empty:
        raise ValueError("no primary RMSE rows match teacher and strongest baseline")

    rows: list[dict[str, Any]] = []
    for (seed, recording_id), group in selected.groupby(
        ["seed", "recording_id"], sort=True, dropna=False
    ):
        teacher = group.loc[group["model"].isin(conditions)]
        baseline = group.loc[group["model"] == strongest_baseline]
        present = set(teacher["model"])
        if present != set(conditions) or baseline.empty:
            raise ValueError("primary matrix has missing teacher condition or baseline cells")
        for model_name, model_rows in (
            (strongest_baseline, baseline),
            ("teacher", teacher),
        ):
            values = model_rows["value"].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("primary RMSE values must be finite")
            source_run_ids = [str(value) for value in model_rows["run_id"]]
            checkpoint_hashes = [str(value) for value in model_rows["checkpoint_sha256"]]
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
                    "checkpoint_sha256": _digest_strings(checkpoint_hashes),
                }
            )
    result = pd.DataFrame(rows, columns=PER_RECORD_COLUMNS).sort_values(
        ["seed", "recording_id", "model"], ignore_index=True
    )
    return validate_per_record_metrics(result)


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
) -> dict[str, str]:
    """Seal formal outputs without replacing inconsistent existing content."""
    root = Path(output_root)
    if root.exists() and not root.is_dir():
        raise ValueError("formal output_root must be a directory")
    checked_metrics = validate_per_record_metrics(per_record_metrics)
    _validate_gate(gate)
    _require_finite_numeric_columns(summary, "summary")
    _require_finite_numeric_columns(mask_ledger, "mask_ledger")
    payloads = {
        "per_record_metrics.csv": _dataframe_bytes(checked_metrics, "per_record_metrics"),
        "summary.csv": _dataframe_bytes(summary, "summary"),
        "success_gate.json": (canonical_json(dict(gate)) + "\n").encode("utf-8"),
        "mask_ledger.csv": _dataframe_bytes(mask_ledger, "mask_ledger"),
    }
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
    for name, content in payloads.items():
        path = root / name
        if path.exists():
            continue
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(f"inconsistent prior formal artifact: {name}")
            continue
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    return hashes


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
    primary_conditions = tuple(
        evaluation.get("primary_teacher_conditions", ("teacher_actual_residual",))
    )
    primary = make_primary_rows(
        condition_metrics,
        teacher_conditions=primary_conditions,
        strongest_baseline=strongest,
    )
    summary = paired_model_summary(
        primary,
        baseline=strongest,
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
    "FormalBackend",
    "build_native_model",
    "evaluate_record_rows",
    "formal_matrix_plan",
    "make_primary_rows",
    "run_formal_protocol",
    "success_gate_payload",
    "teacher_success",
    "write_formal_artifacts",
]
