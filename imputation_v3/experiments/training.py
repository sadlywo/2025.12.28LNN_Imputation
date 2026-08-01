"""Bounded offline-teacher training and smoke orchestration."""

from __future__ import annotations

import dataclasses
import hashlib
from itertools import islice
import json
from pathlib import Path
import math
import os
import random
import tempfile
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from imputation_v3.config import TeacherConfig
from imputation_v3.data.windows import (
    collate_prepared_windows,
    iter_teacher_windows,
)
from imputation_v3.models.teacher import OfflineTeacher
from imputation_v3.objectives.reconstruction import channel_balanced_missing_mse
from validation_v2.data.normalization import RobustTrainScaler
from validation_v2.data.oxiod import load_recording
from validation_v2.data.splits import stratified_file_split
from validation_v2.experiments.provenance import (
    canonical_json,
    collect_provenance,
    git_worktree_identity,
)
from validation_v2.experiments.runner import discover_oxiod_pairs
from validation_v2.experiments.train import train_one_run


_BATCH_TENSORS = ("features", "target", "observed", "mask", "dt", "baseline")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_stable(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"{path.name} already has inconsistent content")
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
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if path.read_bytes() != content:
                raise ValueError(f"{path.name} already has inconsistent content") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _window_evidence(window: Any) -> dict[str, Any]:
    mask = np.ascontiguousarray(window.mask.numpy(), dtype=np.float32)
    return {
        "window_id": window.window_id,
        "recording_id": window.recording_id,
        "topology": window.topology,
        "requested_fraction": window.requested_fraction,
        "realized_fraction": window.realized_fraction,
        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "mask_bytes_hex": mask.tobytes().hex(),
        "mask_dtype": "float32-le",
        "samples": int(mask.shape[0]),
        "channels": int(mask.shape[1]),
    }


def _replayed_checkpoint_metrics(
    checkpoint_path: Path,
    config: TeacherConfig,
    prepared: dict[str, list[Any]],
) -> dict[str, dict[str, float]]:
    device = torch.device("cpu")
    model = OfflineTeacher(
        31,
        config.hidden_size,
        config.tcn_width,
        config.tcn_dilations,
        residual_mode="residual",
        time_mode="actual",
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    _, evaluate_epoch = make_teacher_callbacks(device)
    metrics: dict[str, dict[str, float]] = {}
    for split in ("train", "validation"):
        loader = DataLoader(
            prepared[split],
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_prepared_windows,
        )
        measured = evaluate_epoch(model, loader, 0)
        metrics[split] = {"missing_rmse": float(measured["missing_rmse"])}
    return metrics


def _device_batch(batch: Any, device: torch.device) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for name in _BATCH_TENSORS:
        try:
            value = getattr(batch, name)
        except AttributeError as error:
            raise TypeError(
                "batch must be a PreparedBatch-compatible collator result"
            ) from error
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch.{name} must be a torch tensor")
        if value.device.type == "meta":
            raise ValueError(f"batch.{name} must be materialized")
        non_blocking = device.type == "cuda" and value.device.type == "cpu" and value.is_pinned()
        values[name] = value.to(device=device, non_blocking=non_blocking)

    features = values["features"]
    target = values["target"]
    if target.ndim != 3 or target.shape[0] == 0 or target.shape[-1] != 6:
        raise ValueError("batch.target must have non-empty shape (B, T, 6)")
    if features.shape != (*target.shape[:2], 31):
        raise ValueError("batch.features must have shape (B, T, 31)")
    for name in ("observed", "mask", "baseline"):
        if values[name].shape != target.shape:
            raise ValueError(f"batch.{name} must have shape (B, T, 6)")
    if values["dt"].shape != target.shape[:2]:
        raise ValueError("batch.dt must have shape (B, T)")
    for name, value in values.items():
        if not value.is_floating_point():
            raise TypeError(f"batch.{name} must be floating point")
        if value.dtype != target.dtype:
            raise TypeError(f"batch.{name} dtype must match batch.target")
    if not torch.all((values["mask"] == 0) | (values["mask"] == 1)).item():
        raise ValueError("batch.mask must contain exact binary values")
    return values


def _accumulate_missing(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sse: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    missing = mask == 0
    error = torch.where(missing, prediction - target, torch.zeros_like(prediction))
    if not torch.isfinite(error[missing]).all().item():
        raise ValueError("prediction and target values used by the metric must be finite")
    sse += error.detach().to(device="cpu", dtype=torch.float64).square().sum(dim=(0, 1))
    counts += missing.detach().to(device="cpu", dtype=torch.float64).sum(dim=(0, 1))


def _metric(sse: torch.Tensor, counts: torch.Tensor, batches: int) -> dict[str, float]:
    if batches == 0:
        raise ValueError("loader must not be empty")
    represented = counts > 0
    if not represented.any().item():
        raise ValueError("loader must contain at least one missing value")
    value = torch.sqrt((sse[represented] / counts[represented]).mean()).item()
    if not math.isfinite(value):
        raise ValueError("missing_rmse must be finite")
    return {"missing_rmse": float(value)}


def make_teacher_callbacks(device: torch.device):
    """Build Validation v2-compatible teacher epoch callbacks."""

    if not isinstance(device, torch.device):
        raise TypeError("device must be torch.device")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    def train_epoch(model: Any, optimizer: Any, loader: Any, epoch: int):
        del epoch
        model.train()
        sse = torch.zeros(6, dtype=torch.float64)
        counts = torch.zeros(6, dtype=torch.float64)
        batches = 0
        for source_batch in loader:
            batch = _device_batch(source_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["features"],
                batch["dt"],
                batch["observed"],
                batch["mask"],
                batch["baseline"],
            )
            if not hasattr(output, "raw") or not isinstance(output.raw, torch.Tensor):
                raise TypeError("teacher output.raw must be a torch tensor")
            loss = channel_balanced_missing_mse(
                output.raw, batch["target"], batch["mask"]
            )
            if not torch.isfinite(loss).item():
                raise ValueError("training loss must be finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm).item():
                raise ValueError("gradient norm must be finite")
            _accumulate_missing(
                output.raw, batch["target"], batch["mask"], sse, counts
            )
            optimizer.step()
            batches += 1
        return _metric(sse, counts, batches)

    def evaluate_epoch(model: Any, loader: Any, epoch: int):
        del epoch
        model.eval()
        sse = torch.zeros(6, dtype=torch.float64)
        counts = torch.zeros(6, dtype=torch.float64)
        batches = 0
        with torch.inference_mode():
            for source_batch in loader:
                batch = _device_batch(source_batch, device)
                output = model(
                    batch["features"],
                    batch["dt"],
                    batch["observed"],
                    batch["mask"],
                    batch["baseline"],
                )
                if not hasattr(output, "raw") or not isinstance(output.raw, torch.Tensor):
                    raise TypeError("teacher output.raw must be a torch tensor")
                _accumulate_missing(
                    output.raw, batch["target"], batch["mask"], sse, counts
                )
                batches += 1
        return _metric(sse, counts, batches)

    return train_epoch, evaluate_epoch


def run_teacher_smoke(
    config: TeacherConfig,
    *,
    repository_root: Path,
    requested_device: str,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Train or resume one tightly bounded teacher smoke run."""

    if not isinstance(config, TeacherConfig):
        raise TypeError("config must be TeacherConfig")
    if len(config.seeds) != 1:
        raise ValueError("teacher smoke requires exactly one seed")
    if "teacher" not in config.models:
        raise ValueError("teacher smoke config must include teacher")
    if config.selection_split != "validation":
        raise ValueError("teacher smoke selection_split must be validation")

    requested = str(requested_device)
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    resolved_device = "cuda" if requested == "cuda" or (
        requested == "auto" and torch.cuda.is_available()
    ) else "cpu"
    device = torch.device(resolved_device)

    root = Path(repository_root).resolve()
    data_root = config.data_root if config.data_root.is_absolute() else root / config.data_root
    selected_output = output_root if output_root is not None else config.output_root
    effective_output = (
        selected_output if selected_output.is_absolute() else root / selected_output
    )

    seed = config.seeds[0]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    pairs = discover_oxiod_pairs(data_root)
    split_frame = stratified_file_split(pairs, seed=2026)
    split_rows = [
        {
            "recording_id": str(row["recording_id"]),
            "scenario": str(row["scenario"]),
            "imu_path": str(row["imu_path"]),
            "vicon_path": str(row["vicon_path"]),
            "split": str(row["split"]),
            "imu_sha256": str(row["imu_sha256"]),
            "vicon_sha256": str(row["vicon_sha256"]),
        }
        for row in split_frame.to_dict(orient="records")
    ]
    split_rows.sort(key=lambda row: row["recording_id"])
    split_counts = {
        name: sum(row["split"] == name for row in split_rows)
        for name in ("train", "validation", "test")
    }
    if any(split_counts[name] == 0 for name in split_counts):
        raise ValueError("teacher smoke requires non-empty train, validation, and test splits")

    selected_rows = {
        split: [row for row in split_rows if row["split"] == split][:2]
        for split in ("train", "validation")
    }
    recordings = {
        split: [
            load_recording(Path(row["imu_path"]), Path(row["vicon_path"]))
            for row in selected_rows[split]
        ]
        for split in ("train", "validation")
    }
    for split in ("train", "validation"):
        expected_ids = [row["recording_id"] for row in selected_rows[split]]
        actual_ids = [recording.id for recording in recordings[split]]
        if actual_ids != expected_ids:
            raise ValueError(f"loaded {split} recording ids do not match split assignment")

    train_ids = tuple(row["recording_id"] for row in selected_rows["train"])
    scaler = RobustTrainScaler.fit(recordings["train"], allowed_ids=set(train_ids))
    split_hash = hashlib.sha256(canonical_json(split_rows).encode("utf-8")).hexdigest()
    scaler_state = {
        "center": scaler.center_.tolist(),
        "scale": scaler.scale_.tolist(),
        "training_ids": list(scaler.training_ids),
        "split_hash": split_hash,
    }
    scaler_hash = hashlib.sha256(
        canonical_json(scaler_state).encode("utf-8")
    ).hexdigest()

    window_samples = config.window_samples[0]
    stride = window_samples // 2
    if stride < 1:
        raise ValueError("teacher smoke window length must yield a positive half-window stride")
    prepared = {
        split: list(
            islice(
                iter_teacher_windows(
                    recordings[split],
                    scaler,
                    window_samples=window_samples,
                    stride=stride,
                    seed=seed,
                    topologies=config.training_topologies,
                    rates=config.training_rates,
                    exhaustive=False,
                ),
                4,
            )
        )
        for split in ("train", "validation")
    }
    if not prepared["train"] or not prepared["validation"]:
        raise ValueError("teacher smoke requires prepared train and validation windows")

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        prepared["train"],
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_prepared_windows,
    )
    validation_loader = DataLoader(
        prepared["validation"],
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_prepared_windows,
    )

    source_config = dataclasses.asdict(config)
    source_config.pop("output_root", None)
    resolved_config = {
        "mode": "imputation_v3_teacher_smoke",
        "selection_split": "validation",
        "test_evaluation": False,
        "model": "teacher",
        "seed": seed,
        "device": resolved_device,
        "data_root": data_root.resolve(),
        "source_config": source_config,
        "split_seed": 2026,
        "split_counts": split_counts,
        "selected_recording_ids": {
            split: [row["recording_id"] for row in selected_rows[split]]
            for split in ("train", "validation")
        },
        "scaler_training_ids": list(scaler.training_ids),
        "selected_window_ids": {
            split: [window.window_id for window in prepared[split]]
            for split in ("train", "validation")
        },
        "split_manifest": split_rows,
        "scaler_state": scaler_state,
        "window_evidence": {
            split: [_window_evidence(window) for window in prepared[split]]
            for split in ("train", "validation")
        },
        "bounds": {
            "max_recordings_per_split": 2,
            "max_windows_per_split": 4,
        },
        "hyperparameters": {
            "window_samples": window_samples,
            "stride": stride,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "hidden_size": config.hidden_size,
            "tcn_width": config.tcn_width,
            "tcn_dilations": list(config.tcn_dilations),
            "learning_rate": config.learning_rate,
            "training_rates": list(config.training_rates),
            "training_topologies": list(config.training_topologies),
            "residual_mode": "residual",
            "time_mode": "actual",
        },
    }
    identity = git_worktree_identity(root)
    manifest = collect_provenance(
        resolved_config,
        seed,
        split_hash=split_hash,
        scaler_hash=scaler_hash,
        git_commit=identity["git_commit"],
        dirty_digest=identity["dirty_state_digest"],
    )
    run_dir = effective_output.resolve() / manifest["run_id"]
    artifact_names = (
        "run.json", "history.json", "best.pt", "checkpoint.json", "evidence.json"
    )
    present_artifacts = {name for name in artifact_names if (run_dir / name).exists()}
    if present_artifacts and present_artifacts != set(artifact_names):
        raise ValueError("partial or inconsistent smoke evidence cannot be resumed")
    completed_before = present_artifacts == set(artifact_names)
    expected_checkpoint_sha256 = None
    metadata_path = run_dir / "checkpoint.json"
    if metadata_path.is_file():
        try:
            stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_checkpoint_sha256 = stored_metadata["checkpoint_sha256"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError("partial or inconsistent checkpoint metadata") from error
        if not isinstance(expected_checkpoint_sha256, str):
            raise ValueError("partial or inconsistent checkpoint metadata")

    model = OfflineTeacher(
        31,
        config.hidden_size,
        config.tcn_width,
        config.tcn_dilations,
        residual_mode="residual",
        time_mode="actual",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_epoch, evaluate_epoch = make_teacher_callbacks(device)
    checkpoint = train_one_run(
        run_dir,
        manifest,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=config.epochs,
        train_epoch=train_epoch,
        evaluate_epoch=evaluate_epoch,
        model=model,
        optimizer=optimizer,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    replayed_metrics = _replayed_checkpoint_metrics(
        run_dir / "best.pt", config, prepared
    )
    evidence = {
        "schema": "imputation-v3-smoke-evidence-v2",
        "run_id": manifest["run_id"],
        "run_manifest_sha256": _sha256_path(run_dir / "run.json"),
        "history_sha256": _sha256_path(run_dir / "history.json"),
        "checkpoint_metadata_sha256": _sha256_path(run_dir / "checkpoint.json"),
        "checkpoint_sha256": _sha256_path(run_dir / "best.pt"),
        "final_checkpoint_metrics": replayed_metrics,
        "final_checkpoint_metrics_sha256": hashlib.sha256(
            canonical_json(replayed_metrics).encode("utf-8")
        ).hexdigest(),
    }
    _write_stable(
        run_dir / "evidence.json",
        (canonical_json(evidence) + "\n").encode("utf-8"),
    )
    return {
        "status": "resumed" if completed_before else "completed",
        "run_id": manifest["run_id"],
        "run_dir": str(run_dir),
        "checkpoint": dict(checkpoint),
        "counts": {
            "train_recordings": len(recordings["train"]),
            "validation_recordings": len(recordings["validation"]),
            "train_windows": len(prepared["train"]),
            "validation_windows": len(prepared["validation"]),
            "test_recordings_loaded": 0,
        },
    }


__all__ = ["make_teacher_callbacks", "run_teacher_smoke"]
