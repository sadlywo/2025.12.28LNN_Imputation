"""Bounded real-data execution used by the validation-v2 CLI."""

from __future__ import annotations

import os

_CUBLAS_WORKSPACE_CONFIG = os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
)
if _CUBLAS_WORKSPACE_CONFIG not in {":4096:8", ":16:8"}:
    raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8")

import csv
import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from validation_v2.data.features import build_features
from validation_v2.data.masking import (
    channel_outage,
    contiguous_block,
    generate_interval_jittered_time,
    point_missing,
)
from validation_v2.data.normalization import RobustTrainScaler
from validation_v2.data.oxiod import IMU_CHANNEL_NAMES, load_recording
from validation_v2.data.windows import make_windows
from validation_v2.evaluation.reconstruction import reconstruction_metrics
from validation_v2.evaluation.trajectory import measured_attitude_full_record_diagnostic
from validation_v2.experiments.evaluate import evaluate_test_once
from validation_v2.experiments.matrix import enumerate_matrix
from validation_v2.experiments.provenance import canonical_json, collect_provenance
from validation_v2.experiments.train import resume_run, train_one_run
from validation_v2.models.baselines import (
    equal_average,
    fixed_gate,
    linear_interpolation,
    locf,
)
from validation_v2.models.bilnn import BidirectionalCfC
from validation_v2.models.bilstm import BiLSTMImputer
from validation_v2.models.hybrid import HybridImputer, complete_signal
from validation_v2.objectives.reconstruction import missing_mse, missing_rmse
from validation_v2.types import Recording


@dataclass(frozen=True)
class _Window:
    features: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    dt: torch.Tensor


class _BaselineCheckpoint(nn.Module):
    """A state-bearing marker for a parameter-free observed-only baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("format_version", torch.tensor(1, dtype=torch.int64))
        # train_one_run has an optimizer contract; this inert scalar is never used.
        self._optimizer_anchor = nn.Parameter(torch.zeros(()), requires_grad=True)


class ExecutionModel(nn.Module):
    """Uniform observed-only prediction interface for every declared model."""

    def __init__(self, name: str, hidden_size: int) -> None:
        super().__init__()
        self.name = name
        if name in {"linear", "locf"}:
            self.core = _BaselineCheckpoint()
        elif name == "bilstm":
            self.core = BiLSTMImputer(25, hidden_size, output_size=6)
        elif name == "bilnn":
            self.core = BidirectionalCfC(25, hidden_size, output_size=6)
        elif name in {
            "hybrid", "equal_average", "fixed_gate_0", "fixed_gate_0.5", "fixed_gate_1"
        }:
            self.core = HybridImputer(
                25, lnn_hidden_size=hidden_size, lstm_hidden_size=hidden_size
            )
        else:
            raise ValueError(f"unsupported model: {name}")

    def predict(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        dt: torch.Tensor,
        reported_model: str | None = None,
    ) -> torch.Tensor:
        selected_model = reported_model or self.name
        observed = features[..., :6]
        if selected_model == "linear":
            return linear_interpolation(observed, mask)
        if selected_model == "locf":
            return locf(observed, mask)
        reverse_dt = reverse_aligned_dt(dt)
        if selected_model == "bilstm":
            return complete_signal(observed, mask, self.core(features))
        if selected_model == "bilnn":
            return complete_signal(
                observed, mask, self.core(features, dt, reverse_dt)
            )
        components = self.core.forward_components(
            features, dt, reverse_dt, observed, mask
        )
        if selected_model == "hybrid":
            return components.completed
        if selected_model == "equal_average":
            return equal_average(
                observed, mask, components.lnn, components.lstm
            )
        gate = float(selected_model.removeprefix("fixed_gate_"))
        return fixed_gate(
            observed, mask, components.lnn, components.lstm, gate
        )


def build_execution_model(model_name: str, hidden_size: int) -> ExecutionModel:
    """Construct a server model under the common observed-only contract."""

    return ExecutionModel(model_name, hidden_size)


def reverse_aligned_dt(dt: torch.Tensor) -> torch.Tensor:
    """Align elapsed intervals with a time-reversed feature sequence."""

    if not isinstance(dt, torch.Tensor) or dt.ndim < 1 or dt.shape[-1] < 2:
        raise ValueError("dt must have a final time axis with at least two values")
    if not dt.is_floating_point() or not torch.isfinite(dt).all() or not torch.all(dt > 0):
        raise ValueError("dt must be finite, positive, and floating point")
    return torch.cat((dt[..., -1:], dt[..., 1:].flip(-1)), dim=-1)


def predict_stitched_sequence(
    model: Any,
    features: torch.Tensor,
    mask: torch.Tensor,
    dt: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    reported_model: str | None = None,
    return_coverage: bool = False,
):
    """Predict a full sequence via overlapping training-length neural windows."""

    if features.ndim != 2 or mask.shape != (features.shape[0], 6):
        raise ValueError("features and mask must have shapes (N, F) and (N, 6)")
    if dt.shape != (features.shape[0],):
        raise ValueError("dt must have shape (N,)")
    if seq_len <= 0 or batch_size <= 0:
        raise ValueError("seq_len and batch_size must be positive")
    length = features.shape[0]
    if getattr(model, "name", None) in {"linear", "locf"} or length <= seq_len:
        prediction = model.predict(
            features.unsqueeze(0), mask.unsqueeze(0), dt.unsqueeze(0),
            reported_model=reported_model,
        )[0]
        coverage = torch.ones(length, dtype=features.dtype, device=features.device)
    else:
        stride = max(1, seq_len // 2)
        starts = list(range(0, length - seq_len + 1, stride))
        tail = length - seq_len
        if starts[-1] != tail:
            starts.append(tail)
        total = torch.zeros((length, 6), dtype=features.dtype, device=features.device)
        coverage = torch.zeros(length, dtype=features.dtype, device=features.device)
        for offset in range(0, len(starts), batch_size):
            group = starts[offset : offset + batch_size]
            feature_batch = torch.stack(
                [features[start : start + seq_len] for start in group]
            )
            mask_batch = torch.stack([mask[start : start + seq_len] for start in group])
            dt_batch = torch.stack([dt[start : start + seq_len] for start in group])
            predicted = model.predict(
                feature_batch, mask_batch, dt_batch, reported_model=reported_model
            )
            for index, start in enumerate(group):
                total[start : start + seq_len] += predicted[index]
                coverage[start : start + seq_len] += 1
        if not torch.all(coverage > 0):
            raise RuntimeError("stitched inference left uncovered samples")
        prediction = total / coverage[:, None]
        prediction = complete_signal(features[:, :6], mask, prediction)
    return (prediction, coverage) if return_coverage else prediction


def resample_physical_time(
    source_time_s: np.ndarray,
    physical: np.ndarray,
    query_time_s: np.ndarray,
) -> np.ndarray:
    """Linearly resample all six physical IMU channels without extrapolation."""

    source_time = np.asarray(source_time_s, dtype=np.float64)
    query_time = np.asarray(query_time_s, dtype=np.float64)
    values = np.asarray(physical, dtype=np.float64)
    if source_time.ndim != 1 or query_time.ndim != 1:
        raise ValueError("source and query time must be one-dimensional")
    if values.shape != (source_time.size, 6):
        raise ValueError("physical must have shape (len(source_time), 6)")
    if np.any(np.diff(source_time) <= 0) or np.any(np.diff(query_time) <= 0):
        raise ValueError("source and query time must be strictly increasing")
    if query_time[0] < source_time[0] or query_time[-1] > source_time[-1]:
        raise ValueError("query time must remain within source endpoints")
    return np.column_stack(
        [np.interp(query_time, source_time, values[:, channel]) for channel in range(6)]
    )


def _scenario_name(directory_name: str) -> str:
    if directory_name.startswith("handbag-"):
        return "handbag"
    if directory_name.startswith("handheld-"):
        return "handheld"
    if directory_name == "slow walking":
        return "slow_walking"
    return directory_name


def discover_oxiod_pairs(data_root: Path | str) -> list[dict[str, str]]:
    """Discover exact imuN/viN pairs in the real Oxford Dataset tree."""

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise ValueError(f"data_root is not a directory: {root}")
    pairs: list[dict[str, str]] = []
    for directory in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda p: p.name):
        imu_by_index = {
            int(match.group(1)): path
            for path in directory.glob("imu*.csv")
            if (match := re.fullmatch(r"imu(\d+)\.csv", path.name))
        }
        vicon_by_index = {
            int(match.group(1)): path
            for path in directory.glob("vi*.csv")
            if (match := re.fullmatch(r"vi(\d+)\.csv", path.name))
        }
        if set(imu_by_index) != set(vicon_by_index):
            raise ValueError(f"unpaired IMU/Vicon files in {directory}")
        for index in sorted(imu_by_index):
            pairs.append(
                {
                    "recording_id": f"{directory.name}/imu{index}",
                    "scenario": _scenario_name(directory.name),
                    "imu_path": str(imu_by_index[index].resolve()),
                    "vicon_path": str(vicon_by_index[index].resolve()),
                }
            )
    if not pairs:
        raise ValueError(f"no imuN.csv/viN.csv pairs found under {root}")
    return pairs


def resolve_protocol_records(
    pairs: Sequence[Mapping[str, str]], protocol: str, *, seed: int
) -> list[dict[str, str]]:
    """Assign every discovered file to one deterministic recording-level split."""

    if protocol != "strict_file" and not protocol.startswith("scenario_holdout:"):
        raise ValueError(f"unknown protocol: {protocol}")
    held_out = protocol.partition(":")[2] if ":" in protocol else None
    scenarios = sorted({str(item["scenario"]) for item in pairs})
    if held_out and held_out not in scenarios:
        raise ValueError(f"held-out scenario not found: {held_out}")
    split_by_id: dict[str, str] = {}
    for scenario in scenarios:
        group = sorted(
            (dict(item) for item in pairs if item["scenario"] == scenario),
            key=lambda item: item["recording_id"],
        )
        if held_out == scenario:
            for item in group:
                split_by_id[item["recording_id"]] = "test"
            continue
        rng = np.random.default_rng(_record_seed(seed, f"split:{scenario}"))
        order = rng.permutation(len(group)).tolist()
        if held_out:
            validation_count = max(1, int(round(0.15 * len(group))))
            validation_indices = set(order[:validation_count])
            for index, item in enumerate(group):
                split_by_id[item["recording_id"]] = (
                    "validation" if index in validation_indices else "train"
                )
        else:
            validation_count = max(1, int(round(0.15 * len(group))))
            test_count = max(1, int(round(0.15 * len(group))))
            if validation_count + test_count >= len(group):
                validation_count = test_count = 1
            validation_indices = set(order[:validation_count])
            test_indices = set(order[validation_count : validation_count + test_count])
            for index, item in enumerate(group):
                split_by_id[item["recording_id"]] = (
                    "validation"
                    if index in validation_indices
                    else "test"
                    if index in test_indices
                    else "train"
                )
    return [
        {
            **dict(item),
            "imu": str(item["imu_path"]),
            "vicon": str(item["vicon_path"]),
            "split": split_by_id[str(item["recording_id"])],
        }
        for item in sorted(pairs, key=lambda value: value["recording_id"])
    ]


def resolve_configured_records(
    config: Mapping[str, Any],
    *,
    data_root: Path,
    protocol: str,
    training_seed: int,
) -> list[dict[str, Any]]:
    """Resolve explicit records or a scan using the frozen split seed."""

    del training_seed
    configured = config.get("recordings")
    if configured is not None:
        if not isinstance(configured, list):
            raise ValueError("recordings must be a list when supplied")
        return [dict(item) for item in configured]
    split_seed = config.get("split_seed", 2026)
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ValueError("split_seed must be an integer")
    return resolve_protocol_records(
        discover_oxiod_pairs(data_root), protocol, seed=split_seed
    )


def resolved_execution_config(
    source_config: Mapping[str, Any],
    *,
    model: str,
    seed: int,
    protocol: str,
    conditions: Sequence[Mapping[str, Any]],
    resolved_device: str,
    recording_splits: Sequence[Mapping[str, str]] = (),
    training_family: str | None = None,
    reported_models: Sequence[str] = (),
) -> dict[str, Any]:
    """Return every behavior-affecting input used for run provenance."""

    filtered_source = {
        key: value
        for key, value in source_config.items()
        if key != "output_root" and not key.startswith("_")
    }
    return {
        "mode": "validation_v2",
        "source_config": filtered_source,
        "model": model,
        "training_family": training_family or model,
        "reported_models": list(reported_models) or [model],
        "seed": seed,
        "protocol": protocol,
        "objective": str(source_config.get("objective", "reconstruction_only")),
        "condition_list": list(conditions),
        "resolved_device": resolved_device,
        "evaluation_scope": (
            "full_overlap_record"
            if source_config.get("max_eval_samples") in (None, 0)
            else "bounded_overlap_slice"
        ),
        "recording_splits": [dict(item) for item in recording_splits],
    }


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_stable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"existing {path.name} does not match resolved inputs")
        return
    path.write_bytes(content)


def _require(config: Mapping[str, Any], key: str, expected: type) -> Any:
    value = config.get(key)
    if not isinstance(value, expected):
        raise ValueError(f"{key} must be {expected.__name__}")
    return value


def _positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of auto, cpu, cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("device cuda was requested but CUDA is unavailable")
    return torch.device(
        "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _git_value(arguments: Sequence[str], default: str = "") -> str:
    try:
        result = subprocess.run(
            ["git", *arguments], capture_output=True, check=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return default
    return result.stdout.strip()


def _record_seed(seed: int, recording_id: str) -> int:
    suffix = int(hashlib.sha256(recording_id.encode("utf-8")).hexdigest()[:8], 16)
    return (seed + suffix) % (2**31)


def _slice_recording(
    recording: Recording, maximum: int | None
) -> tuple[np.ndarray, np.ndarray]:
    start, end = recording.overlap_s
    indices = np.flatnonzero(
        (recording.imu_time_s >= start) & (recording.imu_time_s <= end)
    )
    if maximum is not None and maximum > 0:
        indices = indices[:maximum]
    if indices.size < 2:
        raise ValueError(f"recording {recording.id} has fewer than two overlap samples")
    return recording.imu_time_s[indices], recording.imu_six[indices]


def _dt(time_s: np.ndarray) -> torch.Tensor:
    values = np.empty(time_s.shape, dtype=np.float32)
    values[1:] = np.diff(time_s).astype(np.float32)
    values[0] = values[1]
    return torch.from_numpy(values)


def _prepared_sequence(
    recording: Recording,
    scaler: RobustTrainScaler,
    *,
    maximum: int | None,
    rate: float,
    seed: int,
    topology: str = "point",
    requested_irregularity: float | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
    Any | None,
]:
    time_s, physical = _slice_recording(recording, maximum)
    irregular_result = None
    if requested_irregularity is not None:
        irregular_result = generate_interval_jittered_time(
            torch.from_numpy(time_s),
            requested_irregularity,
            _record_seed(seed, f"irregular:{recording.id}"),
        )
        query_time = irregular_result.time.numpy()
        physical = resample_physical_time(time_s, physical, query_time)
        time_s = query_time
        dt = irregular_result.dt.to(torch.float32)
    else:
        dt = _dt(time_s)
    target = torch.from_numpy(scaler.transform(physical).astype(np.float32))
    generators = {
        "point": point_missing,
        "block": contiguous_block,
        "channel": channel_outage,
    }
    if topology not in generators:
        raise ValueError(f"unsupported missingness topology: {topology}")
    result = generators[topology](target, rate, _record_seed(seed, recording.id))
    if not torch.any(result.mask == 0):
        raise ValueError(
            f"{topology} at requested_fraction={rate} realizes no missing values"
        )
    features = build_features(target, result.mask, dt).values
    return features, target, result.mask, dt, time_s, irregular_result


def _windows(
    recordings: Sequence[Recording],
    scaler: RobustTrainScaler,
    *,
    seq_len: int,
    maximum_windows: int,
    rate: float,
    seed: int,
    topology: str = "point",
) -> list[_Window]:
    prepared: list[_Window] = []
    per_record = max(1, (maximum_windows + len(recordings) - 1) // len(recordings))
    for recording in recordings:
        if len(prepared) >= maximum_windows:
            break
        maximum = seq_len * per_record
        features, target, mask, dt, time_s, _ = _prepared_sequence(
            recording, scaler, maximum=maximum, rate=rate, seed=seed,
            topology=topology,
        )
        batches = make_windows(
            target,
            mask,
            dt,
            torch.arange(len(time_s)),
            torch.from_numpy(time_s),
            recording.id,
            window_size=seq_len,
        )
        for batch in batches:
            feature = build_features(batch.target, batch.mask, batch.dt)
            prepared.append(
                _Window(feature.values, batch.target, batch.mask, batch.dt)
            )
    result = prepared[:maximum_windows]
    if not result:
        raise ValueError("bounded recording slices produced no complete windows")
    return result


def _batches(windows: Sequence[_Window], batch_size: int) -> list[_Window]:
    result: list[_Window] = []
    for start in range(0, len(windows), batch_size):
        group = windows[start : start + batch_size]
        result.append(
            _Window(
                torch.stack([item.features for item in group]),
                torch.stack([item.target for item in group]),
                torch.stack([item.mask for item in group]),
                torch.stack([item.dt for item in group]),
            )
        )
    return result


def _model(name: str, hidden_size: int) -> ExecutionModel:
    return build_execution_model(name, hidden_size)


def _prediction(
    model_name: str,
    model: nn.Module,
    batch: _Window,
    *,
    reported_model: str | None = None,
) -> torch.Tensor:
    del model_name
    return model.predict(
        batch.features, batch.mask, batch.dt, reported_model=reported_model
    )


def _epoch_callbacks(model_name: str, device: torch.device):
    def train_epoch(
        model: nn.Module, optimizer: torch.optim.Optimizer, loader: Sequence[_Window], epoch: int
    ) -> Mapping[str, float]:
        del epoch
        model.train()
        errors: list[float] = []
        for source in loader:
            batch = _Window(
                source.features.to(device),
                source.target.to(device),
                source.mask.to(device),
                source.dt.to(device),
            )
            if model_name in {"linear", "locf"}:
                with torch.no_grad():
                    value = missing_rmse(_prediction(model_name, model, batch), batch.target, batch.mask)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = missing_mse(_prediction(model_name, model, batch), batch.target, batch.mask)
                loss.backward()
                optimizer.step()
                value = torch.sqrt(loss.detach())
            errors.append(float(value.detach().cpu()))
        return {"missing_rmse": float(np.mean(errors))}

    def evaluate_epoch(
        model: nn.Module, loader: Sequence[_Window], epoch: int
    ) -> Mapping[str, float]:
        del epoch
        model.eval()
        errors: list[float] = []
        with torch.no_grad():
            for source in loader:
                batch = _Window(
                    source.features.to(device),
                    source.target.to(device),
                    source.mask.to(device),
                    source.dt.to(device),
                )
                value = missing_rmse(
                    _prediction(model_name, model, batch), batch.target, batch.mask
                )
                errors.append(float(value.cpu()))
        return {"missing_rmse": float(np.mean(errors))}

    return train_epoch, evaluate_epoch


def _manifest_rows(
    records: Sequence[Mapping[str, Any]], data_root: Path
) -> tuple[list[dict[str, str]], dict[str, Recording]]:
    rows: list[dict[str, str]] = []
    loaded: dict[str, Recording] = {}
    seen_sources: set[Path] = set()
    seen_ids: set[str] = set()
    for item in records:
        if not isinstance(item, Mapping):
            raise ValueError("recordings entries must be mappings")
        split = str(item.get("split"))
        if split not in {"train", "validation", "test"}:
            raise ValueError("recording split must be train, validation, or test")
        imu_path = (data_root / str(item.get("imu"))).resolve()
        vicon_path = (data_root / str(item.get("vicon"))).resolve()
        if imu_path in seen_sources or vicon_path in seen_sources:
            raise ValueError("train/validation/test source files must be disjoint")
        seen_sources.update((imu_path, vicon_path))
        recording = load_recording(imu_path, vicon_path)
        if recording.id in seen_ids:
            raise ValueError("train/validation/test recording ids must be disjoint")
        seen_ids.add(recording.id)
        loaded[recording.id] = recording
        rows.append(
            {
                "recording_id": recording.id,
                "scenario": str(item.get("scenario")),
                "imu_path": str(imu_path),
                "vicon_path": str(vicon_path),
                "split": split,
                "imu_sha256": _sha256_file(imu_path),
                "vicon_sha256": _sha256_file(vicon_path),
            }
        )
    counts = {name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "test")}
    if any(counts[name] == 0 for name in counts):
        raise ValueError("execution requires non-empty train, validation, and test splits")
    return rows, loaded


def _split_content(rows: Sequence[Mapping[str, str]]) -> bytes:
    columns = (
        "recording_id", "scenario", "imu_path", "vicon_path", "split", "imu_sha256", "vicon_sha256"
    )
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _scaler_content(scaler: RobustTrainScaler, *, split_hash: str) -> bytes:
    value = {
        "center": scaler.center_.tolist(),
        "scale": scaler.scale_.tolist(),
        "channel_order": list(IMU_CHANNEL_NAMES),
        "training_ids": list(scaler.training_ids),
        "split_hash": split_hash,
    }
    return (canonical_json(value) + "\n").encode("utf-8")


def _trajectory_rows(
    *,
    prediction_physical: np.ndarray,
    complete_physical: np.ndarray,
    time_s: np.ndarray,
    recording: Recording,
) -> Mapping[str, float]:
    diagnostic = measured_attitude_full_record_diagnostic(
        complete_physical,
        prediction_physical,
        time_s,
        recording.vicon_time_s,
        recording.vicon_position_m,
        recording.vicon_quaternion_xyzw,
        frame_metadata={
            "quaternion_order": "xyzw",
            "quaternion_frame": "body_to_reference",
            "euler_order": "xyz",
            "imu_acceleration_unit": "G",
            "user_acceleration_semantics": "gravity_removed",
            "position_unit": "m",
            "time_unit": "s",
        },
    )
    names = ("ate_rmse_m", "rpe_rmse_m", "endpoint_drift_m", "velocity_rmse_mps")
    result = {name: float(diagnostic.imputed_metrics[name]) for name in names}
    result.update(
        {f"delta_{name}": float(value) for name, value in diagnostic.delta_vs_complete.items()}
    )
    return result


def _evaluate_record(
    *,
    run_dir: Path,
    model_name: str,
    model: nn.Module,
    device: torch.device,
    scenarios: Mapping[str, str],
    scaler: RobustTrainScaler,
    maximum: int | None,
    conditions: Sequence[Mapping[str, Any]],
    seed: int,
    trajectory_enabled: bool,
    seq_len: int,
    batch_size: int,
):
    metadata = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))

    def callback(recording: Recording, checkpoint: Path) -> list[dict[str, Any]]:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        rows: list[dict[str, Any]] = []
        for condition in conditions:
            reported_model = str(condition.get("model", model_name))
            is_irregular = condition.get("case_type", "missingness") == "irregular"
            if is_irregular:
                if condition.get("irregular_method") != "interval_jitter":
                    raise ValueError("only interval_jitter irregular cases are supported")
                topology = str(condition.get("value_topology"))
                rate = float(condition.get("value_requested_fraction"))
                irregularity = float(condition.get("requested_irregularity"))
                topology_label = f"irregular:interval_jitter+{topology}"
            else:
                topology = str(condition["topology"])
                rate = float(condition["requested_fraction"])
                irregularity = None
                topology_label = topology
            features, target, mask, dt, time_s, irregular_result = _prepared_sequence(
                recording, scaler, maximum=maximum, rate=rate, seed=seed,
                topology=topology,
                requested_irregularity=irregularity,
            )
            with torch.no_grad():
                prediction = predict_stitched_sequence(
                    model,
                    features.to(device),
                    mask.to(device),
                    dt.to(device),
                    seq_len=seq_len,
                    batch_size=batch_size,
                    reported_model=reported_model,
                ).cpu().numpy()
            target_values = target.numpy()
            mask_values = mask.numpy()
            reports = reconstruction_metrics(
                prediction, target_values, mask_values, scaler=scaler
            )
            metric_values: dict[str, float] = {
                "reconstruction_normalized": float(reports["normalized"]["rmse"]),
                "reconstruction_physical": float(reports["physical"]["rmse"]),
            }
            if irregular_result is not None:
                metric_values.update(
                    irregularity_requested=float(
                        irregular_result.requested_irregularity
                    ),
                    irregularity_realized=float(
                        irregular_result.realized_irregularity
                    ),
                )
            if trajectory_enabled:
                metric_values.update(
                    _trajectory_rows(
                        prediction_physical=scaler.inverse_transform(prediction),
                        complete_physical=scaler.inverse_transform(target_values),
                        time_s=time_s,
                        recording=recording,
                    )
                )
            base = {
                "run_id": run_dir.name,
                "seed": seed,
                "recording_id": recording.id,
                "scenario": scenarios[recording.id],
                "protocol": str(condition.get("protocol", "strict_file")),
                "topology": topology_label,
                "requested_fraction": rate,
                "realized_fraction": float((mask_values == 0).mean()),
                "model": reported_model,
                "checkpoint_sha256": metadata["checkpoint_sha256"],
            }
            rows.extend(
                {**base, "metric": name, "value": value}
                for name, value in metric_values.items()
            )
        return rows

    return callback


def _descriptive_summary(output_root: Path, run_dirs: Sequence[Path]) -> None:
    frame = pd.concat(
        [pd.read_csv(run_dir / "per_record_metrics.csv") for run_dir in run_dirs],
        ignore_index=True,
    )
    summary = (
        frame.groupby(["model", "metric"], sort=True)["value"]
        .mean()
        .rename("mean")
        .reset_index()
    )
    summary.insert(0, "n_recordings", 1)
    summary.insert(0, "descriptive_only", True)
    summary.to_csv(output_root / "summary.csv", index=False, lineterminator="\n")
    records = summary.to_dict(orient="records")
    _write_stable(
        output_root / "summary.json", (canonical_json(records) + "\n").encode("utf-8")
    )
    smoke_summary = {
        "descriptive_only": True,
        "n_recordings": 1,
        "reason": "single test recording; no confidence interval was computed",
        "models": sorted(frame["model"].unique().tolist()),
    }
    _write_stable(
        output_root / "smoke_summary.json",
        (canonical_json(smoke_summary) + "\n").encode("utf-8"),
    )


def run_smoke(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
    output_root: Path | None = None,
    requested_device: str | None = None,
) -> Mapping[str, Any]:
    """Run the bounded 2/1/1 OxIOD smoke protocol on real CSV pairs."""

    seed_values = _require(config, "seeds", list)
    if len(seed_values) != 1 or isinstance(seed_values[0], bool):
        raise ValueError("one integer seed is required per execution group")
    seed = int(seed_values[0])
    epochs = _positive_int(config, "epochs")
    batch_size = _positive_int(config, "batch_size")
    seq_len = _positive_int(config, "seq_len")
    max_train_windows = _positive_int(config, "max_train_windows")
    raw_max_eval_samples = config.get("max_eval_samples")
    if isinstance(raw_max_eval_samples, bool):
        raise ValueError("max_eval_samples must be null, zero, or a positive integer")
    if raw_max_eval_samples in (None, 0):
        max_eval_samples: int | None = None
    elif (
        not isinstance(raw_max_eval_samples, int)
        or raw_max_eval_samples < 0
    ):
        raise ValueError("max_eval_samples must be null, zero, or a positive integer")
    else:
        max_eval_samples = raw_max_eval_samples
    hidden_size = _positive_int(config, "hidden_size")
    models = _require(config, "models", list)
    supported_models = {
        "linear", "locf", "bilstm", "bilnn", "hybrid", "equal_average",
        "fixed_gate_0", "fixed_gate_0.5", "fixed_gate_1",
    }
    if not models or any(model not in supported_models for model in models):
        raise ValueError("config contains an unsupported model")
    default_rate = float(_require(config, "rates", list)[0])
    default_topology = str(_require(config, "topologies", list)[0])
    conditions = config.get("_execution_conditions") or [
        {
            "case_type": "missingness",
            "protocol": str(_require(config, "protocols", list)[0]),
            "topology": default_topology,
            "requested_fraction": default_rate,
        }
    ]
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("execution condition list must be non-empty")
    training_condition = next(
        (condition for condition in conditions if condition.get("case_type") == "missingness"),
        None,
    )
    if training_condition is None:
        raise ValueError("each training group requires at least one missingness condition")
    rate = float(training_condition["requested_fraction"])
    topology = str(training_condition["topology"])
    if config.get("objective") != "reconstruction_only":
        raise ValueError("smoke primary objective must be reconstruction_only")
    if bool(config.get("kinematic_ablation", {}).get("enabled", False)):
        raise ValueError("kinematic_ablation must be disabled for smoke")
    trajectory_enabled = bool(config.get("trajectory_enabled", False))
    device = _device(requested_device or str(config.get("device", "auto")))
    _set_seed(seed)

    data_root = Path(str(config.get("data_root")))
    if not data_root.is_absolute():
        data_root = repository_root / data_root
    destination = output_root or Path(str(config.get("output_root")))
    if not destination.is_absolute():
        destination = repository_root / destination
    destination.mkdir(parents=True, exist_ok=True)

    records_config = resolve_configured_records(
        config,
        data_root=data_root,
        protocol=str(training_condition.get("protocol", "strict_file")),
        training_seed=seed,
    )
    manifest_rows, loaded = _manifest_rows(records_config, data_root)
    split_content = _split_content(manifest_rows)
    split_hash = _sha256_bytes(split_content)
    grouped_execution = bool(config.get("_skip_descriptive_summary", False))
    split_name = (
        f"split_manifest-{split_hash}.csv" if grouped_execution else "split_manifest.csv"
    )
    _write_stable(destination / split_name, split_content)
    by_split = {
        split: [loaded[row["recording_id"]] for row in manifest_rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    scaler = RobustTrainScaler.fit(
        by_split["train"], allowed_ids={recording.id for recording in by_split["train"]}
    )
    scaler_content = _scaler_content(scaler, split_hash=split_hash)
    scaler_hash = _sha256_bytes(scaler_content)
    scaler_name = f"scaler-{scaler_hash}.json" if grouped_execution else "scaler.json"
    _write_stable(destination / scaler_name, scaler_content)

    baseline_group = all(model in {"linear", "locf"} for model in models)
    execution_train_windows = 1 if baseline_group else max_train_windows
    execution_validation_windows = 1 if baseline_group else max_train_windows
    train_batches = _batches(
        _windows(
            by_split["train"], scaler, seq_len=seq_len,
            maximum_windows=execution_train_windows, rate=rate, seed=seed,
            topology=topology,
        ),
        batch_size,
    )
    validation_batches = _batches(
        _windows(
            by_split["validation"], scaler, seq_len=seq_len,
            maximum_windows=execution_validation_windows, rate=rate, seed=seed,
            topology=topology,
        ),
        batch_size,
    )
    commit = _git_value(("rev-parse", "HEAD"))
    dirty_text = _git_value(("status", "--porcelain=v1", "--untracked-files=no"))
    dirty_digest = _sha256_bytes(dirty_text.encode("utf-8")) if dirty_text else ""
    run_dirs: list[Path] = []
    scenarios = {row["recording_id"]: row["scenario"] for row in manifest_rows}
    for model_name in models:
        _set_seed(seed)
        resolved = resolved_execution_config(
            config,
            model=model_name,
            seed=seed,
            protocol=str(training_condition.get("protocol", "strict_file")),
            conditions=conditions,
            resolved_device=str(device),
            training_family=str(config.get("_training_family", model_name)),
            reported_models=tuple(config.get("_reported_models", (model_name,))),
            recording_splits=[
                {"recording_id": row["recording_id"], "split": row["split"]}
                for row in manifest_rows
            ],
        )
        provenance = collect_provenance(
            resolved,
            seed,
            split_hash=split_hash,
            scaler_hash=scaler_hash,
            git_commit=commit,
            dirty_digest=dirty_digest,
        )
        run_dir = destination / provenance["run_id"]
        model = _model(model_name, hidden_size).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config.get("learning_rate", 0.001))
        )
        train_epoch, validation_epoch = _epoch_callbacks(model_name, device)
        if (run_dir / "checkpoint.json").is_file():
            metadata = json.loads(
                (run_dir / "checkpoint.json").read_text(encoding="utf-8")
            )
            resume_run(run_dir, provenance, metadata["checkpoint_sha256"])
        else:
            train_one_run(
                run_dir,
                provenance,
                model=model,
                optimizer=optimizer,
                train_loader=train_batches,
                validation_loader=validation_batches,
                epochs=1 if model_name in {"linear", "locf"} else epochs,
                train_epoch=train_epoch,
                evaluate_epoch=validation_epoch,
            )
        ledger = run_dir / "test_evaluation.json"
        if ledger.is_file():
            status = json.loads(ledger.read_text(encoding="utf-8")).get("status")
            if status != "completed" or not (run_dir / "per_record_metrics.csv").is_file():
                raise ValueError("partial or failed test evaluation cannot be resumed")
        else:
            evaluate_test_once(
                run_dir,
                lambda: list(by_split["test"]),
                _evaluate_record(
                    run_dir=run_dir,
                    model_name=model_name,
                    model=model,
                    device=device,
                    scenarios=scenarios,
                    scaler=scaler,
                    maximum=max_eval_samples,
                    conditions=conditions,
                    seed=seed,
                    trajectory_enabled=trajectory_enabled,
                    seq_len=seq_len,
                    batch_size=batch_size,
                ),
                trajectory_enabled=trajectory_enabled,
            )
        run_dirs.append(run_dir)
    if not bool(config.get("_skip_descriptive_summary", False)):
        _descriptive_summary(destination, run_dirs)
    return {
        "status": "completed",
        "real_data": True,
        "descriptive_only": True,
        "n_recordings": len(by_split["test"]),
        "output_root": str(destination),
        "run_ids": [path.name for path in run_dirs],
        "split_hash": split_hash,
        "scaler_hash": scaler_hash,
        "device": str(device),
        "max_eval_samples": max_eval_samples,
        "evaluation_scope": (
            "full_overlap_record"
            if max_eval_samples is None
            else "bounded_overlap_slice"
        ),
    }


def run_matrix(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
    output_root: Path | None = None,
    requested_device: str | None = None,
    max_combinations: int | None = None,
) -> Mapping[str, Any]:
    """Execute selected cells, sharing one checkpoint per training-key group."""

    combinations = enumerate_matrix(config)
    if max_combinations is not None and (
        isinstance(max_combinations, bool) or max_combinations <= 0
    ):
        raise ValueError("max_combinations must be a positive integer")
    execution_order = sorted(
        combinations,
        key=lambda cell: (
            cell["model"] != "linear",
            cell["protocol"] != "strict_file",
            cell["case_type"] != "missingness",
            cell.get("topology") != "point",
            canonical_json(cell),
        ),
    )
    selected = execution_order[:max_combinations] if max_combinations is not None else combinations
    partial = len(selected) < len(combinations)
    destination = output_root or Path(str(config.get("output_root")))
    if not destination.is_absolute():
        destination = repository_root / destination
    destination.mkdir(parents=True, exist_ok=True)
    objective = str(config.get("objective", "reconstruction_only"))
    gate_models = {
        "hybrid", "equal_average", "fixed_gate_0", "fixed_gate_0.5", "fixed_gate_1"
    }
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for cell in selected:
        model_name = str(cell["model"])
        training_family = "hybrid_shared" if model_name in gate_models else model_name
        key = (
            training_family,
            int(cell["seed"]),
            str(cell["protocol"]),
            objective,
        )
        grouped.setdefault(key, []).append(cell)
    marker_path = destination / "matrix_execution.json"
    marker: dict[str, Any] = {
        "partial": partial,
        "selected_cells": len(selected),
        "total_cells": len(combinations),
        "training_groups": len(grouped),
        "grouping_key": ["training_family", "seed", "protocol", "objective"],
        "selected_combination_ids": [cell["combination_id"] for cell in selected],
        "status": "started",
    }
    marker_path.write_text(canonical_json(marker) + "\n", encoding="utf-8")
    reports: list[Mapping[str, Any]] = []
    try:
        for (training_family, seed, protocol, _), cells in sorted(grouped.items()):
            training_model = (
                "hybrid" if training_family == "hybrid_shared" else training_family
            )
            reported_models = sorted({str(cell["model"]) for cell in cells})
            group_config = dict(config)
            group_config.update(
                models=[training_model],
                seeds=[seed],
                protocols=[protocol],
                _execution_conditions=cells,
                _skip_descriptive_summary=True,
                _training_family=training_family,
                _reported_models=reported_models,
            )
            reports.append(
                run_smoke(
                    group_config,
                    repository_root=repository_root,
                    output_root=destination,
                    requested_device=requested_device,
                )
            )
    except BaseException as error:
        marker.update(status="failed", error=type(error).__name__)
        marker_path.write_text(canonical_json(marker) + "\n", encoding="utf-8")
        raise
    marker.update(
        status="completed",
        run_ids=sorted(
            run_id for report in reports for run_id in report.get("run_ids", [])
        ),
    )
    marker_path.write_text(canonical_json(marker) + "\n", encoding="utf-8")
    return marker


__all__ = [
    "ExecutionModel",
    "build_execution_model",
    "discover_oxiod_pairs",
    "resolve_protocol_records",
    "run_matrix",
    "run_smoke",
]
