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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from validation_v2.data.adapters import get_dataset_adapter
from validation_v2.data.features import build_features
from validation_v2.data.masking import (
    channel_outage,
    contiguous_block,
    generate_interval_jittered_time,
    point_missing,
)
from validation_v2.data.normalization import RobustTrainScaler
from validation_v2.data.oxiod import IMU_CHANNEL_NAMES
from validation_v2.data.windows import make_windows
from validation_v2.evaluation.reconstruction import reconstruction_metrics
from validation_v2.evaluation.physics import physics_endpoint_diagnostics
from validation_v2.evaluation.synchronization import synchronize_vicon_to_imu
from validation_v2.evaluation.trajectory import measured_attitude_full_record_diagnostic
from validation_v2.experiments.evaluate import evaluate_test_once
from validation_v2.experiments.groups import (
    enumerate_training_groups,
    group_execution_config,
)
from validation_v2.experiments.matrix import enumerate_matrix
from validation_v2.experiments.provenance import (
    canonical_json,
    collect_provenance,
    git_worktree_identity,
)
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
from validation_v2.objectives.physics_informed import (
    IMUPhysicsInformedLoss,
    PhysicsLossConfig,
)
from validation_v2.objectives.reconstruction import missing_mse, missing_rmse
from validation_v2.types import Recording


@dataclass(frozen=True)
class _Window:
    features: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    dt: torch.Tensor
    recording_index: int = -1
    start: int = -1
    vicon_position_m: torch.Tensor | None = None
    vicon_rotation_body_to_world: torch.Tensor | None = None
    vicon_velocity_mps: torch.Tensor | None = None
    normalization_center: torch.Tensor | None = None
    normalization_scale: torch.Tensor | None = None


@dataclass(frozen=True)
class ExternalDataPreparation:
    """V2-owned split and normalization state for external model workers."""

    manifest_rows: tuple[dict[str, str], ...]
    recordings_by_split: dict[str, tuple[Recording, ...]]
    scalers: dict[str, RobustTrainScaler]
    split_content: bytes
    split_hash: str
    scaler_content: bytes
    scaler_hash: str

    @property
    def scaler(self) -> RobustTrainScaler:
        """Keep the legacy single-dataset API while rejecting ambiguous use."""

        if len(self.scalers) != 1:
            raise ValueError(
                "joint data has one train-only scaler per dataset; use scalers"
            )
        return next(iter(self.scalers.values()))


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

    def predict_raw(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        dt: torch.Tensor,
        reported_model: str | None = None,
    ) -> torch.Tensor:
        selected_model = reported_model or self.name
        observed = features[..., :6]
        # RobustTrainScaler maps the train-only channel median to zero.  That
        # leakage-safe statistic is the declared fallback for channel outage,
        # where interpolation and LOCF have no within-series observation.
        if selected_model == "linear":
            return linear_interpolation(observed, mask, empty_series_fill=0.0)
        if selected_model == "locf":
            return locf(observed, mask, empty_series_fill=0.0)
        reverse_dt = reverse_aligned_dt(dt)
        if selected_model == "bilstm":
            return self.core(features)
        if selected_model == "bilnn":
            return self.core(features, dt, reverse_dt)
        components = self.core.forward_components(
            features, dt, reverse_dt, observed, mask
        )
        if selected_model == "hybrid":
            return components.raw
        if selected_model == "equal_average":
            return 0.5 * (components.lnn + components.lstm)
        gate = float(selected_model.removeprefix("fixed_gate_"))
        return gate * components.lnn + (1.0 - gate) * components.lstm

    def predict(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        dt: torch.Tensor,
        reported_model: str | None = None,
    ) -> torch.Tensor:
        """Return a completed signal while keeping the raw head deployable."""

        raw = self.predict_raw(features, mask, dt, reported_model=reported_model)
        return complete_signal(features[..., :6], mask, raw)


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

    return [dict(item) for item in get_dataset_adapter("oxiod").discover(Path(data_root))]


def discover_dataset_pairs(
    data_root: Path | str, *, dataset_name: str = "oxiod"
) -> list[dict[str, str]]:
    """Discover recordings through the configured dataset adapter."""

    return [
        dict(item)
        for item in get_dataset_adapter(dataset_name).discover(Path(data_root))
    ]


def _split_ratios(config: Mapping[str, Any]) -> tuple[float, float, float]:
    raw = config.get("split_ratios", (0.7, 0.15, 0.15))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
        raise ValueError("split_ratios must contain train, validation, and test")
    ratios = tuple(float(value) for value in raw)
    if any(value <= 0.0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("split_ratios must be positive and sum to one")
    return ratios  # type: ignore[return-value]


def _dataset_sources(
    config: Mapping[str, Any], repository_root: Path
) -> tuple[tuple[str, Path], ...]:
    configured = config.get("datasets")
    if configured is None:
        if config.get("data_root") is None:
            raise ValueError("configure either data_root or datasets")
        root = Path(str(config["data_root"]))
        if not root.is_absolute():
            root = repository_root / root
        return ((str(config.get("dataset_name", "oxiod")), root.resolve()),)
    if config.get("data_root") is not None or config.get("dataset_name") is not None:
        raise ValueError("datasets cannot be combined with data_root or dataset_name")
    if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
        raise ValueError("datasets must be a non-empty list")
    sources: list[tuple[str, Path]] = []
    for item in configured:
        if not isinstance(item, Mapping) or set(item) != {"name", "data_root"}:
            raise ValueError("each dataset requires exactly name and data_root")
        name = str(item["name"])
        get_dataset_adapter(name)
        root = Path(str(item["data_root"]))
        if not root.is_absolute():
            root = repository_root / root
        sources.append((name, root.resolve()))
    if not sources or len({name for name, _ in sources}) != len(sources):
        raise ValueError("datasets must be non-empty and have unique names")
    return tuple(sources)


def resolve_protocol_records(
    pairs: Sequence[Mapping[str, str]], protocol: str, *, seed: int,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
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
            validation_fraction = split_ratios[1] / sum(split_ratios[:2])
            validation_count = max(1, int(round(validation_fraction * len(group))))
            validation_indices = set(order[:validation_count])
            for index, item in enumerate(group):
                split_by_id[item["recording_id"]] = (
                    "validation" if index in validation_indices else "train"
                )
        else:
            validation_count = max(1, int(round(split_ratios[1] * len(group))))
            test_count = max(1, int(round(split_ratios[2] * len(group))))
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
    data_root: Path | None = None,
    repository_root: Path | None = None,
    protocol: str,
    training_seed: int,
) -> list[dict[str, Any]]:
    """Resolve explicit records or a scan using the frozen split seed."""

    del training_seed
    configured = config.get("recordings")
    if configured is not None:
        if config.get("datasets") is not None:
            raise ValueError("joint datasets use adapter discovery, not recordings")
        if not isinstance(configured, list):
            raise ValueError("recordings must be a list when supplied")
        dataset_name = str(config.get("dataset_name", "oxiod"))
        if data_root is not None:
            source_root = data_root.resolve()
        else:
            source_root = _dataset_sources(
                config, Path(repository_root or Path.cwd()).resolve()
            )[0][1]
        return [
            {
                "dataset": dataset_name,
                "data_root": str(source_root),
                **dict(item),
            }
            for item in configured
        ]
    split_seed = config.get("split_seed", 2026)
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ValueError("split_seed must be an integer")
    base = Path(repository_root or Path.cwd()).resolve()
    if config.get("datasets") is None and data_root is not None:
        sources = ((str(config.get("dataset_name", "oxiod")), data_root.resolve()),)
    else:
        sources = _dataset_sources(config, base)
    ratios = _split_ratios(config)
    resolved: list[dict[str, Any]] = []
    for dataset_name, source_root in sources:
        dataset_records = resolve_protocol_records(
            discover_dataset_pairs(source_root, dataset_name=dataset_name),
            protocol,
            seed=split_seed,
            split_ratios=ratios,
        )
        resolved.extend(
            {"dataset": dataset_name, "data_root": str(source_root), **item}
            for item in dataset_records
        )
    return resolved


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


def _recording_dataset(recording: Recording) -> str:
    dataset = recording.metadata.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError(f"recording {recording.id} does not declare its dataset")
    return dataset


def _scaler_for(
    recording: Recording,
    scalers: RobustTrainScaler | Mapping[str, RobustTrainScaler],
) -> RobustTrainScaler:
    if isinstance(scalers, RobustTrainScaler):
        return scalers
    dataset = _recording_dataset(recording)
    try:
        return scalers[dataset]
    except KeyError as error:
        raise ValueError(f"no train-only scaler for dataset {dataset}") from error


def _prepared_sequence(
    recording: Recording,
    scalers: RobustTrainScaler | Mapping[str, RobustTrainScaler],
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
    scaler = _scaler_for(recording, scalers)
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
    scalers: RobustTrainScaler | Mapping[str, RobustTrainScaler],
    *,
    seq_len: int,
    maximum_windows: int,
    rate: float,
    seed: int,
    topology: str = "point",
) -> list[_Window]:
    if not recordings:
        raise ValueError("recordings must not be empty")
    indexed = list(enumerate(recordings))
    if isinstance(scalers, Mapping) and len(scalers) > 1:
        grouped: dict[str, list[tuple[int, Recording]]] = {}
        for pair in indexed:
            grouped.setdefault(_recording_dataset(pair[1]), []).append(pair)
        names = sorted(grouped)
        quotient, remainder = divmod(maximum_windows, len(names))
        groups = [
            (grouped[name], quotient + (index < remainder))
            for index, name in enumerate(names)
        ]
    else:
        groups = [(indexed, maximum_windows)]
    generators = {
        "point": point_missing,
        "block": contiguous_block,
        "channel": channel_outage,
    }
    if topology not in generators:
        raise ValueError(f"unsupported missingness topology: {topology}")
    prepared_groups: list[list[_Window]] = []
    for group, budget in groups:
        prepared: list[_Window] = []
        per_record = max(1, (budget + len(group) - 1) // len(group))
        for recording_index, recording in group:
            if len(prepared) >= budget:
                break
            scaler = _scaler_for(recording, scalers)
            maximum = seq_len * per_record
            time_s, physical = _slice_recording(recording, maximum)
            synchronized = synchronize_vicon_to_imu(
                recording.vicon_time_s,
                recording.vicon_position_m,
                recording.vicon_quaternion_xyzw,
                time_s,
                frame_metadata={
                    "quaternion_order": "xyzw",
                    "quaternion_frame": "body_to_reference",
                    "euler_order": "xyz",
                },
            )
            vicon_position = torch.from_numpy(
                synchronized.position_m.astype(np.float32, copy=False)
            )
            vicon_rotation = torch.from_numpy(
                synchronized.rotation_body_to_world.astype(np.float32, copy=False)
            )
            vicon_velocity = torch.from_numpy(
                synchronized.velocity_world_mps.astype(np.float32, copy=False)
            )
            normalization_center = torch.from_numpy(
                scaler.center_.astype(np.float32, copy=True)
            )
            normalization_scale = torch.from_numpy(
                scaler.scale_.astype(np.float32, copy=True)
            )
            target = torch.from_numpy(scaler.transform(physical).astype(np.float32))
            dt = _dt(time_s)
            batches = make_windows(
                target,
                torch.ones_like(target),
                dt,
                torch.arange(len(time_s)),
                torch.from_numpy(time_s),
                recording.id,
                window_size=seq_len,
            )
            for window_number, batch in enumerate(batches):
                window_seed = _record_seed(
                    seed,
                    ":".join(
                        (
                            "training-window",
                            recording.id,
                            str(int(batch.index[0].item())),
                            str(window_number),
                            topology,
                            format(float(rate), ".17g"),
                        )
                    ),
                )
                mask = generators[topology](batch.target, rate, window_seed).mask
                feature = build_features(batch.target, mask, batch.dt)
                positions = batch.index.to(dtype=torch.long)
                prepared.append(
                    _Window(
                        feature.values,
                        batch.target,
                        mask,
                        batch.dt,
                        recording_index=recording_index,
                        start=int(batch.index[0].item()),
                        vicon_position_m=vicon_position[positions],
                        vicon_rotation_body_to_world=vicon_rotation[positions],
                        vicon_velocity_mps=vicon_velocity[positions],
                        normalization_center=normalization_center,
                        normalization_scale=normalization_scale,
                    )
                )
        prepared_groups.append(prepared[:budget])
    result = [
        group[index]
        for index in range(max(map(len, prepared_groups), default=0))
        for group in prepared_groups
        if index < len(group)
    ][:maximum_windows]
    if not result:
        raise ValueError("bounded recording slices produced no complete windows")
    return result


def _batches(windows: Sequence[_Window], batch_size: int) -> list[_Window]:
    result: list[_Window] = []
    for start in range(0, len(windows), batch_size):
        group = windows[start : start + batch_size]
        optional_names = (
            "vicon_position_m",
            "vicon_rotation_body_to_world",
            "vicon_velocity_mps",
            "normalization_center",
            "normalization_scale",
        )
        if any(
            any(getattr(item, name) is None for item in group)
            for name in optional_names
        ):
            raise ValueError("training windows require complete physics metadata")
        result.append(
            _Window(
                torch.stack([item.features for item in group]),
                torch.stack([item.target for item in group]),
                torch.stack([item.mask for item in group]),
                torch.stack([item.dt for item in group]),
                vicon_position_m=torch.stack(
                    [item.vicon_position_m for item in group]  # type: ignore[list-item]
                ),
                vicon_rotation_body_to_world=torch.stack(
                    [item.vicon_rotation_body_to_world for item in group]  # type: ignore[list-item]
                ),
                vicon_velocity_mps=torch.stack(
                    [item.vicon_velocity_mps for item in group]  # type: ignore[list-item]
                ),
                normalization_center=torch.stack(
                    [item.normalization_center for item in group]  # type: ignore[list-item]
                ),
                normalization_scale=torch.stack(
                    [item.normalization_scale for item in group]  # type: ignore[list-item]
                ),
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
    raw: bool = False,
) -> torch.Tensor:
    del model_name
    method = model.predict_raw if raw else model.predict
    return method(batch.features, batch.mask, batch.dt, reported_model=reported_model)


def _device_window(source: _Window, device: torch.device) -> _Window:
    def move(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value.to(device)

    return _Window(
        source.features.to(device),
        source.target.to(device),
        source.mask.to(device),
        source.dt.to(device),
        source.recording_index,
        source.start,
        move(source.vicon_position_m),
        move(source.vicon_rotation_body_to_world),
        move(source.vicon_velocity_mps),
        move(source.normalization_center),
        move(source.normalization_scale),
    )


def _physics_loss_config(value: Mapping[str, Any] | None) -> PhysicsLossConfig:
    if not isinstance(value, Mapping):
        raise ValueError("physics_informed objective requires a physics mapping")
    required = {
        "lambda_physics",
        "sigma_rotation_rad",
        "sigma_velocity_mps",
        "sigma_position_m",
        "acceleration_mode",
        "acceleration_unit",
        "frame_validation_status",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"physics config missing keys: {', '.join(missing)}")
    lambda_physics = float(value["lambda_physics"])
    if lambda_physics > 0.0 and value["frame_validation_status"] != "validated":
        raise ValueError(
            "non-zero physics loss is gated until clean-IMU/Vicon frame validation is marked validated"
        )
    return PhysicsLossConfig(
        lambda_physics=lambda_physics,
        sigma_rotation_rad=float(value["sigma_rotation_rad"]),
        sigma_velocity_mps=float(value["sigma_velocity_mps"]),
        sigma_position_m=float(value["sigma_position_m"]),
        acceleration_mode=str(value["acceleration_mode"]),
        acceleration_unit=str(value["acceleration_unit"]),
    )


def _require_physics_window(batch: _Window) -> None:
    names = (
        "vicon_position_m",
        "vicon_rotation_body_to_world",
        "vicon_velocity_mps",
        "normalization_center",
        "normalization_scale",
    )
    missing = [name for name in names if getattr(batch, name) is None]
    if missing:
        raise ValueError(f"physics batch missing fields: {', '.join(missing)}")


def _endpoint_components(components: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {
        name: float(value.detach().cpu())
        for name, value in components.items()
        if name != "total"
    }


def _epoch_callbacks(
    model_name: str,
    device: torch.device,
    *,
    objective: str = "reconstruction_only",
    physics: Mapping[str, Any] | None = None,
):
    if objective not in {"reconstruction_only", "physics_informed"}:
        raise ValueError(f"unsupported training objective: {objective}")
    criterion = (
        IMUPhysicsInformedLoss(_physics_loss_config(physics))
        if objective == "physics_informed"
        else None
    )

    def loss_and_metrics(model: nn.Module, batch: _Window):
        raw = _prediction(model_name, model, batch, raw=True)
        if criterion is None:
            loss = missing_mse(raw, batch.target, batch.mask)
            return loss, {"signal": float(loss.detach().cpu()), "physics": 0.0}, raw
        _require_physics_window(batch)
        completed = complete_signal(batch.target, batch.mask, raw)
        loss, components = criterion(
            prediction=raw,
            target=batch.target,
            mask=batch.mask,
            completed=completed,
            dt=batch.dt,
            normalization_center=batch.normalization_center,  # type: ignore[arg-type]
            normalization_scale=batch.normalization_scale,  # type: ignore[arg-type]
            vicon_position_m=batch.vicon_position_m,  # type: ignore[arg-type]
            vicon_rotation_body_to_world=batch.vicon_rotation_body_to_world,  # type: ignore[arg-type]
            vicon_velocity_mps=batch.vicon_velocity_mps,  # type: ignore[arg-type]
        )
        return loss, _endpoint_components(components), raw

    def train_epoch(
        model: nn.Module, optimizer: torch.optim.Optimizer, loader: Sequence[_Window], epoch: int
    ) -> Mapping[str, float]:
        del epoch
        model.train()
        errors: list[float] = []
        logged: dict[str, list[float]] = {}
        gradient_norms: list[float] = []
        for source in loader:
            batch = _device_window(source, device)
            if model_name in {"linear", "locf"}:
                with torch.no_grad():
                    value = missing_rmse(_prediction(model_name, model, batch), batch.target, batch.mask)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss, components, raw = loss_and_metrics(model, batch)
                loss.backward()
                squared = torch.zeros((), device=device)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        squared = squared + parameter.grad.detach().square().sum()
                gradient_norms.append(float(torch.sqrt(squared).cpu()))
                optimizer.step()
                value = missing_rmse(raw.detach(), batch.target, batch.mask)
                for name, component in components.items():
                    logged.setdefault(name, []).append(component)
            errors.append(float(value.detach().cpu()))
        result = {"missing_rmse": float(np.mean(errors))}
        result.update({name: float(np.mean(values)) for name, values in logged.items()})
        if gradient_norms:
            result["gradient_norm"] = float(np.mean(gradient_norms))
        return result

    def evaluate_epoch(
        model: nn.Module, loader: Sequence[_Window], epoch: int
    ) -> Mapping[str, float]:
        del epoch
        model.eval()
        errors: list[float] = []
        logged: dict[str, list[float]] = {}
        with torch.no_grad():
            for source in loader:
                batch = _device_window(source, device)
                raw = _prediction(model_name, model, batch, raw=True)
                value = missing_rmse(raw, batch.target, batch.mask)
                if criterion is not None:
                    _, components, _ = loss_and_metrics(model, batch)
                    for name, component in components.items():
                        logged.setdefault(name, []).append(component)
                errors.append(float(value.cpu()))
        result = {"missing_rmse": float(np.mean(errors))}
        result.update({name: float(np.mean(values)) for name, values in logged.items()})
        return result

    return train_epoch, evaluate_epoch


def _manifest_rows(
    records: Sequence[Mapping[str, Any]], data_root: Path | None = None, *,
    dataset_name: str = "oxiod",
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
        item_dataset = str(item.get("dataset", dataset_name))
        adapter = get_dataset_adapter(item_dataset)
        item_root = Path(str(item.get("data_root", data_root or ".")))
        imu_path = (item_root / str(item.get("imu"))).resolve()
        vicon_path = (item_root / str(item.get("vicon"))).resolve()
        if imu_path in seen_sources or vicon_path in seen_sources:
            raise ValueError("train/validation/test source files must be disjoint")
        seen_sources.update((imu_path, vicon_path))
        recording = adapter.load(imu_path, vicon_path)
        if recording.id in seen_ids:
            raise ValueError("train/validation/test recording ids must be disjoint")
        seen_ids.add(recording.id)
        loaded[recording.id] = recording
        rows.append(
            {
                "dataset": item_dataset,
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
    for item_dataset in sorted({row["dataset"] for row in rows}):
        dataset_counts = {
            name: sum(
                row["dataset"] == item_dataset and row["split"] == name
                for row in rows
            )
            for name in ("train", "validation", "test")
        }
        if any(value == 0 for value in dataset_counts.values()):
            raise ValueError(
                f"dataset {item_dataset} requires non-empty train, validation, and test splits"
            )
    return rows, loaded


def _split_content(rows: Sequence[Mapping[str, str]]) -> bytes:
    columns = (
        "dataset", "recording_id", "scenario", "imu_path", "vicon_path", "split",
        "imu_sha256", "vicon_sha256"
    )
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _scaler_content(
    scaler: RobustTrainScaler,
    *,
    split_hash: str,
    channel_order: Sequence[str] = IMU_CHANNEL_NAMES,
) -> bytes:
    value = {
        "center": scaler.center_.tolist(),
        "scale": scaler.scale_.tolist(),
        "channel_order": list(channel_order),
        "training_ids": list(scaler.training_ids),
        "split_hash": split_hash,
    }
    return (canonical_json(value) + "\n").encode("utf-8")


def _scalers_content(
    scalers: Mapping[str, RobustTrainScaler], *, split_hash: str
) -> bytes:
    if len(scalers) == 1:
        name, scaler = next(iter(scalers.items()))
        return _scaler_content(
            scaler,
            split_hash=split_hash,
            channel_order=get_dataset_adapter(name).channel_names,
        )
    value = {
        "schema_version": 2,
        "normalization_scope": "per_dataset_train_only",
        "joint_sampling": "dataset_balanced",
        "split_hash": split_hash,
        "datasets": {
            name: {
                "center": scaler.center_.tolist(),
                "scale": scaler.scale_.tolist(),
                "channel_order": list(get_dataset_adapter(name).channel_names),
                "training_ids": list(scaler.training_ids),
                "acceleration_unit": get_dataset_adapter(name).semantics.acceleration_unit,
                "acceleration_mode": get_dataset_adapter(name).semantics.acceleration_mode,
            }
            for name, scaler in sorted(scalers.items())
        },
    }
    return (canonical_json(value) + "\n").encode("utf-8")


def _fit_scalers(
    recordings: Sequence[Recording],
) -> dict[str, RobustTrainScaler]:
    grouped: dict[str, list[Recording]] = {}
    for recording in recordings:
        grouped.setdefault(_recording_dataset(recording), []).append(recording)
    return {
        name: RobustTrainScaler.fit(
            group, allowed_ids={recording.id for recording in group}
        )
        for name, group in sorted(grouped.items())
    }


def prepare_external_data(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
    protocol: str,
    seed: int,
) -> ExternalDataPreparation:
    """Resolve splits and fit the train-only scaler once for external workers."""

    records_config = resolve_configured_records(
        config,
        repository_root=repository_root,
        protocol=protocol,
        training_seed=seed,
    )
    manifest_rows, loaded = _manifest_rows(records_config)
    split_content = _split_content(manifest_rows)
    split_hash = _sha256_bytes(split_content)
    recordings_by_split = {
        split: tuple(
            loaded[row["recording_id"]]
            for row in manifest_rows
            if row["split"] == split
        )
        for split in ("train", "validation", "test")
    }
    train_recordings = recordings_by_split["train"]
    scalers = _fit_scalers(train_recordings)
    scaler_content = _scalers_content(scalers, split_hash=split_hash)
    return ExternalDataPreparation(
        manifest_rows=tuple(dict(row) for row in manifest_rows),
        recordings_by_split=recordings_by_split,
        scalers=scalers,
        split_content=split_content,
        split_hash=split_hash,
        scaler_content=scaler_content,
        scaler_hash=_sha256_bytes(scaler_content),
    )


def prepare_external_windows(
    recordings: Sequence[Recording],
    scaler: RobustTrainScaler | Mapping[str, RobustTrainScaler],
    *,
    seq_len: int,
    maximum_windows: int,
    rate: float,
    seed: int,
    topology: str = "point",
) -> tuple[dict[str, Any], ...]:
    """Expose the exact V2 training windows without duplicating mask logic."""

    return tuple(
        {
            "target": window.target,
            "mask": window.mask,
            "dt": window.dt,
            "recording_index": window.recording_index,
            "start": window.start,
        }
        for window in _windows(
            recordings,
            scaler,
            seq_len=seq_len,
            maximum_windows=maximum_windows,
            rate=rate,
            seed=seed,
            topology=topology,
        )
    )


def prepare_external_sequence(
    recording: Recording,
    scaler: RobustTrainScaler | Mapping[str, RobustTrainScaler],
    *,
    maximum: int | None,
    rate: float,
    seed: int,
    topology: str = "point",
    requested_irregularity: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Expose V2's canonical normalized target, mask, dt, and physical time."""

    _, target, mask, dt, time_s, _ = _prepared_sequence(
        recording,
        scaler,
        maximum=maximum,
        rate=rate,
        seed=seed,
        topology=topology,
        requested_irregularity=requested_irregularity,
    )
    return target, mask, dt, time_s


def _trajectory_rows(
    *,
    prediction_physical: np.ndarray,
    complete_physical: np.ndarray,
    time_s: np.ndarray,
    recording: Recording,
    acceleration_unit: str,
    acceleration_mode: str,
) -> Mapping[str, float]:
    diagnostic_mode = (
        "gravity_removed"
        if acceleration_mode == "gravity_compensated"
        else acceleration_mode
    )
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
            "imu_acceleration_unit": acceleration_unit,
            "user_acceleration_semantics": diagnostic_mode,
            "position_unit": "m",
            "time_unit": "s",
        },
        acceleration_unit=acceleration_unit,
        acceleration_mode=diagnostic_mode,
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
    scalers: Mapping[str, RobustTrainScaler],
    maximum: int | None,
    conditions: Sequence[Mapping[str, Any]],
    seed: int,
    trajectory_enabled: bool,
    physics_diagnostics_enabled: bool,
    seq_len: int,
    batch_size: int,
):
    metadata = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))

    def callback(recording: Recording, checkpoint: Path) -> list[dict[str, Any]]:
        dataset = _recording_dataset(recording)
        scaler = _scaler_for(recording, scalers)
        semantics = get_dataset_adapter(dataset).semantics
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        rows: list[dict[str, Any]] = []
        irregular_condition_count = len(
            {
                (
                    condition.get("irregular_method"),
                    condition.get("requested_irregularity"),
                    condition.get("value_topology"),
                    condition.get("value_requested_fraction"),
                )
                for condition in conditions
                if condition.get("case_type", "missingness") == "irregular"
            }
        )
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
                if irregular_condition_count > 1:
                    topology_label = (
                        f"irregular:interval_jitter@{irregularity:g}+{topology}"
                    )
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
                        acceleration_unit=semantics.acceleration_unit,
                        acceleration_mode=semantics.acceleration_mode,
                    )
                )
            if physics_diagnostics_enabled:
                synchronized = synchronize_vicon_to_imu(
                    recording.vicon_time_s,
                    recording.vicon_position_m,
                    recording.vicon_quaternion_xyzw,
                    time_s,
                    frame_metadata={
                        "quaternion_order": "xyzw",
                        "quaternion_frame": "body_to_reference",
                        "euler_order": "xyz",
                    },
                )
                dt_values = np.empty_like(time_s, dtype=np.float64)
                dt_values[1:] = np.diff(time_s)
                dt_values[0] = dt_values[1]
                metric_values.update(
                    physics_endpoint_diagnostics(
                        scaler.inverse_transform(prediction),
                        scaler.inverse_transform(target_values),
                        mask_values,
                        dt_values,
                        synchronized.position_m,
                        synchronized.rotation_body_to_world,
                        synchronized.velocity_world_mps,
                        acceleration_unit=semantics.acceleration_unit,
                        acceleration_mode=semantics.acceleration_mode,
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
    """Run a bounded real-data protocol for one or more dataset adapters."""

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
        (
            condition
            for condition in conditions
            if condition.get("case_type") == "missingness"
            and condition.get("topology") == "point"
            and float(condition.get("requested_fraction", -1.0)) == 0.3
        ),
        next(
            (
                condition
                for condition in conditions
                if condition.get("case_type") == "missingness"
            ),
            None,
        ),
    )
    if training_condition is None:
        raise ValueError("each training group requires at least one missingness condition")
    rate = float(training_condition["requested_fraction"])
    topology = str(training_condition["topology"])
    objective = str(config.get("objective", "reconstruction_only"))
    if objective not in {"reconstruction_only", "physics_informed"}:
        raise ValueError("unsupported smoke objective")
    if objective == "physics_informed" and any(
        model in {"linear", "locf"} for model in models
    ):
        raise ValueError("parameter-free baselines cannot optimize physics_informed loss")
    if bool(config.get("kinematic_ablation", {}).get("enabled", False)):
        raise ValueError("kinematic_ablation must be disabled for smoke")
    trajectory_enabled = bool(config.get("trajectory_enabled", False))
    physics_mapping = config.get("physics")
    physics_diagnostics_enabled = objective == "physics_informed" or bool(
        config.get("physics_diagnostics_enabled", False)
    )
    sources = _dataset_sources(config, repository_root)
    if len(sources) > 1 and objective == "physics_informed":
        raise ValueError(
            "joint physics-informed batches are disabled because dataset acceleration "
            "semantics differ; use reconstruction_only for the joint benchmark"
        )
    if len(sources) == 1:
        dataset_name = sources[0][0]
        adapter = get_dataset_adapter(dataset_name)
        acceleration_unit = (
            str(physics_mapping.get("acceleration_unit", adapter.semantics.acceleration_unit))
            if isinstance(physics_mapping, Mapping)
            else adapter.semantics.acceleration_unit
        )
        acceleration_mode = (
            str(physics_mapping.get("acceleration_mode", adapter.semantics.acceleration_mode))
            if isinstance(physics_mapping, Mapping)
            else adapter.semantics.acceleration_mode
        )
        if acceleration_unit != adapter.semantics.acceleration_unit:
            raise ValueError(
                f"physics acceleration_unit={acceleration_unit!r} conflicts with "
                f"{dataset_name} adapter semantics {adapter.semantics.acceleration_unit!r}"
            )
        if acceleration_mode != adapter.semantics.acceleration_mode:
            raise ValueError(
                f"physics acceleration_mode={acceleration_mode!r} conflicts with "
                f"{dataset_name} adapter semantics {adapter.semantics.acceleration_mode!r}"
            )
    device = _device(requested_device or str(config.get("device", "auto")))
    _set_seed(seed)

    destination = output_root or Path(str(config.get("output_root")))
    if not destination.is_absolute():
        destination = repository_root / destination
    destination.mkdir(parents=True, exist_ok=True)

    records_config = resolve_configured_records(
        config,
        repository_root=repository_root,
        protocol=str(training_condition.get("protocol", "strict_file")),
        training_seed=seed,
    )
    manifest_rows, loaded = _manifest_rows(records_config)
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
    scalers = _fit_scalers(by_split["train"])
    scaler_content = _scalers_content(scalers, split_hash=split_hash)
    scaler_hash = _sha256_bytes(scaler_content)
    scaler_name = f"scaler-{scaler_hash}.json" if grouped_execution else "scaler.json"
    _write_stable(destination / scaler_name, scaler_content)

    baseline_group = all(model in {"linear", "locf"} for model in models)
    execution_train_windows = 1 if baseline_group else max_train_windows
    execution_validation_windows = 1 if baseline_group else max_train_windows
    train_batches = _batches(
        _windows(
            by_split["train"], scalers, seq_len=seq_len,
            maximum_windows=execution_train_windows, rate=rate, seed=seed,
            topology=topology,
        ),
        batch_size,
    )
    validation_batches = _batches(
        _windows(
            by_split["validation"], scalers, seq_len=seq_len,
            maximum_windows=execution_validation_windows, rate=rate, seed=seed,
            topology=topology,
        ),
        batch_size,
    )
    worktree_identity = git_worktree_identity(repository_root)
    commit = worktree_identity["git_commit"]
    dirty_digest = worktree_identity["dirty_state_digest"]
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
        train_epoch, validation_epoch = _epoch_callbacks(
            model_name,
            device,
            objective=objective,
            physics=config.get("physics"),
        )
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
                    scalers=scalers,
                    maximum=max_eval_samples,
                    conditions=conditions,
                    seed=seed,
                    trajectory_enabled=trajectory_enabled,
                    physics_diagnostics_enabled=physics_diagnostics_enabled,
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
    groups = enumerate_training_groups(config, combinations=selected)
    marker_path = destination / "matrix_execution.json"
    marker: dict[str, Any] = {
        "partial": partial,
        "selected_cells": len(selected),
        "total_cells": len(combinations),
        "training_groups": len(groups),
        "grouping_key": ["training_family", "seed", "protocol", "objective"],
        "selected_combination_ids": [cell["combination_id"] for cell in selected],
        "status": "started",
    }
    marker_path.write_text(canonical_json(marker) + "\n", encoding="utf-8")
    reports: list[Mapping[str, Any]] = []
    try:
        for group in groups:
            group_config = group_execution_config(config, group)
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
    "ExternalDataPreparation",
    "build_execution_model",
    "discover_oxiod_pairs",
    "discover_dataset_pairs",
    "prepare_external_data",
    "prepare_external_sequence",
    "prepare_external_windows",
    "resolve_protocol_records",
    "run_matrix",
    "run_smoke",
]
