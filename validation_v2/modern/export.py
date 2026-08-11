from __future__ import annotations

from dataclasses import asdict
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from validation_v2.experiments.runner import (
    prepare_external_data,
    prepare_external_sequence,
    prepare_external_windows,
)

from .artifacts import canonical_json, write_array_artifact
from .config import ModernConfig


def build_observed_arrays(
    target: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor
) -> dict[str, np.ndarray]:
    if target.shape != mask.shape or target.ndim < 2:
        raise ValueError("target and mask must have the same sequence/feature shape")
    if dt.shape != target.shape[:-1]:
        raise ValueError("dt shape must match target without its feature dimension")
    observed = torch.where(
        mask.bool(), target, torch.full_like(target, float("nan"))
    )
    return {
        "X": observed.detach().cpu().numpy().astype(np.float32, copy=False),
        "mask": mask.detach().cpu().numpy().astype(np.uint8, copy=False),
        "dt": dt.detach().cpu().numpy().astype(np.float32, copy=False),
    }


def window_starts(length: int, *, seq_len: int) -> tuple[int, ...]:
    if length <= 0 or seq_len <= 0:
        raise ValueError("length and seq_len must be positive")
    if length < seq_len:
        return (0,)
    stride = max(1, seq_len // 2)
    starts = list(range(0, length - seq_len + 1, stride))
    tail = length - seq_len
    if starts[-1] != tail:
        starts.append(tail)
    return tuple(starts)


def _write_stable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(f"existing {path.name} does not match exported inputs")


def _training_arrays(windows: tuple[dict[str, Any], ...]) -> dict[str, np.ndarray]:
    targets = torch.stack([window["target"] for window in windows])
    masks = torch.stack([window["mask"] for window in windows])
    dts = torch.stack([window["dt"] for window in windows])
    observed = build_observed_arrays(targets, masks, dts)
    return {
        **observed,
        "X_ori": targets.numpy().astype(np.float32, copy=False),
        "recording_index": np.asarray(
            [window["recording_index"] for window in windows], dtype=np.int32
        ),
        "starts": np.asarray([window["start"] for window in windows], dtype=np.int64),
    }


def _evaluation_arrays(
    target: torch.Tensor,
    mask: torch.Tensor,
    dt: torch.Tensor,
    time_s: np.ndarray,
    *,
    seq_len: int,
) -> dict[str, np.ndarray]:
    if target.shape[0] < seq_len:
        raise ValueError("evaluation recording is shorter than seq_len")
    starts = window_starts(target.shape[0], seq_len=seq_len)
    target_windows = torch.stack([target[start : start + seq_len] for start in starts])
    mask_windows = torch.stack([mask[start : start + seq_len] for start in starts])
    dt_windows = torch.stack([dt[start : start + seq_len] for start in starts])
    observed = build_observed_arrays(target_windows, mask_windows, dt_windows)
    return {
        **observed,
        "X_ori": target_windows.numpy().astype(np.float32, copy=False),
        "time": np.asarray(time_s, dtype=np.float64),
        "starts": np.asarray(starts, dtype=np.int64),
    }


def _condition_id(topology: str, rate: float) -> str:
    return f"{topology}-{int(round(rate * 100)):02d}pct"


def export_modern_dataset(
    config: ModernConfig,
    seed: int,
    repository_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Export one seed's V2-owned splits, masks, and normalized targets."""

    if seed not in config.seeds:
        raise ValueError("export seed is not declared by the modern config")
    repository_root = Path(repository_root).resolve()
    output_dir = Path(output_dir)
    control_config = asdict(config)
    prepared = prepare_external_data(
        control_config,
        repository_root=repository_root,
        protocol=config.protocol,
        seed=seed,
    )
    _write_stable(output_dir / "split_manifest.csv", prepared.split_content)
    _write_stable(output_dir / "scaler.json", prepared.scaler_content)

    artifact_entries: list[dict[str, str]] = []
    for split in ("train", "validation"):
        windows = prepare_external_windows(
            prepared.recordings_by_split[split],
            prepared.scaler,
            seq_len=config.seq_len,
            maximum_windows=config.max_train_windows,
            rate=0.3,
            seed=seed,
            topology="point",
        )
        base = output_dir / split
        manifest = write_array_artifact(
            base,
            "dataset",
            _training_arrays(windows),
            {
                "seed": seed,
                "split": split,
                "split_hash": prepared.split_hash,
                "scaler_hash": prepared.scaler_hash,
                "recording_ids": [
                    recording.id for recording in prepared.recordings_by_split[split]
                ],
                "training_topology": "point",
                "training_requested_fraction": 0.3,
            },
        )
        artifact_entries.append(
            {"path": base.relative_to(output_dir).as_posix(), "artifact_id": str(manifest["artifact_id"])}
        )

    conditions: list[dict[str, object]] = [
        {
            "condition_id": _condition_id(topology, rate),
            "case_type": "missingness",
            "topology": topology,
            "requested_fraction": rate,
            "requested_irregularity": None,
        }
        for topology in config.topologies
        for rate in config.rates
    ]
    for case in config.irregular_case_specs:
        irregularity = float(case["requested_irregularity"])
        value_topology = str(case["value_topology"])
        value_fraction = float(case["value_requested_fraction"])
        conditions.append(
            {
                "condition_id": (
                    f"irregular-interval-jitter-{int(round(irregularity * 100)):02d}pct-"
                    f"{value_topology}-{int(round(value_fraction * 100)):02d}pct"
                ),
                "case_type": "irregular",
                "topology": value_topology,
                "requested_fraction": value_fraction,
                "requested_irregularity": irregularity,
            }
        )

    for condition in conditions:
        condition_id = str(condition["condition_id"])
        for recording in prepared.recordings_by_split["test"]:
            target, mask, dt, time_s = prepare_external_sequence(
                recording,
                prepared.scaler,
                maximum=config.max_eval_samples,
                rate=float(condition["requested_fraction"]),
                seed=seed,
                topology=str(condition["topology"]),
                requested_irregularity=(
                    None
                    if condition["requested_irregularity"] is None
                    else float(condition["requested_irregularity"])
                ),
            )
            base = output_dir / "test" / condition_id / recording.id
            manifest = write_array_artifact(
                base,
                "dataset",
                _evaluation_arrays(
                    target, mask, dt, time_s, seq_len=config.seq_len
                ),
                {
                    "seed": seed,
                    "split": "test",
                    "recording_id": recording.id,
                    "condition": condition,
                    "split_hash": prepared.split_hash,
                    "scaler_hash": prepared.scaler_hash,
                },
            )
            artifact_entries.append(
                {
                    "path": base.relative_to(output_dir).as_posix(),
                    "artifact_id": str(manifest["artifact_id"]),
                }
            )

    payload: dict[str, object] = {
        "schema_version": 1,
        "seed": seed,
        "protocol": config.protocol,
        "split_hash": prepared.split_hash,
        "scaler_hash": prepared.scaler_hash,
        "artifacts": artifact_entries,
    }
    dataset_manifest = {
        **payload,
        "dataset_id": hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest(),
    }
    _write_stable(
        output_dir / "dataset_manifest.json",
        (canonical_json(dataset_manifest) + "\n").encode("utf-8"),
    )
    return dataset_manifest


__all__ = [
    "build_observed_arrays",
    "export_modern_dataset",
    "window_starts",
]
