from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

from .artifacts import (
    canonical_json,
    read_array_artifact,
    sha256_file,
    write_array_artifact,
)


def _lazy_classes() -> tuple[dict[str, type], type]:
    from pypots.imputation import BRITS, CSDI, SAITS
    from pypots.optim import Adam

    return {"brits": BRITS, "saits": SAITS, "csdi": CSDI}, Adam


def build_model(
    model: str,
    config: dict[str, object],
    *,
    n_steps: int,
    n_features: int,
    batch_size: int,
    epochs: int,
    patience: int,
    device: str,
    saving_path: str | None = None,
    classes: dict[str, type] | None = None,
) -> object:
    if classes is None:
        classes, optimizer_class = _lazy_classes()
        optimizer: object = optimizer_class(lr=float(config["learning_rate"]))
    else:
        optimizer = {"name": "Adam", "lr": float(config["learning_rate"])}
    if model not in classes:
        raise ValueError(f"unsupported PyPOTS model: {model}")
    common: dict[str, object] = {
        "n_steps": n_steps,
        "n_features": n_features,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "optimizer": optimizer,
        "num_workers": 0,
        "device": device,
        "saving_path": saving_path,
        "model_saving_strategy": "best",
        "verbose": True,
    }
    if model == "brits":
        kwargs = {
            **common,
            "rnn_hidden_size": int(config["hidden_size"]),
        }
    elif model == "saits":
        d_model = int(config["d_model"])
        kwargs = {
            **common,
            "n_layers": int(config["n_layers"]),
            "d_model": d_model,
            "n_heads": 4,
            "d_k": d_model // 4,
            "d_v": d_model // 4,
            "d_ffn": 2 * d_model,
            "dropout": 0.1,
            "attn_dropout": 0.1,
        }
    else:
        channels = config.get("n_channels", config.get("channels"))
        if channels is None:
            raise ValueError("CSDI configuration requires channels")
        kwargs = {
            **common,
            "n_layers": 4,
            "n_heads": 4,
            "n_channels": int(channels),
            "d_time_embedding": 128,
            "d_feature_embedding": 16,
            "d_diffusion_embedding": 128,
            "n_diffusion_steps": 50,
            "target_strategy": "random",
            "schedule": "quad",
            "beta_start": 0.0001,
            "beta_end": 0.5,
        }
    return classes[model](**kwargs)


def normalize_imputation(
    raw: np.ndarray,
    *,
    windows: int,
    samples: int,
    steps: int,
    features: int,
) -> np.ndarray:
    values = np.asarray(raw)
    expected = (windows, samples, steps, features)
    if values.shape == expected:
        normalized = values
    elif samples == 1 and values.shape == (windows, steps, features):
        normalized = values[:, None, :, :]
    elif values.shape == (windows, steps, features, samples):
        normalized = np.transpose(values, (0, 3, 1, 2))
    elif values.shape == (samples, windows, steps, features):
        normalized = np.transpose(values, (1, 0, 2, 3))
    else:
        raise ValueError(
            f"unexpected imputation shape {values.shape}; expected {expected}"
        )
    return normalized.astype(np.float32, copy=False)


def _force_observed(samples: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    observed_inputs = np.asarray(inputs, dtype=np.float32)
    if samples.shape[0] != observed_inputs.shape[0] or samples.shape[2:] != observed_inputs.shape[1:]:
        raise ValueError("prediction and input shapes do not align")
    observed = np.isfinite(observed_inputs)
    completed = np.where(
        observed[:, None, :, :], observed_inputs[:, None, :, :], samples
    )
    if not np.all(np.isfinite(completed)):
        raise ValueError("model produced nonfinite values at missing positions")
    return completed.astype(np.float32, copy=False)


def _write_json_no_clobber(path: Path, value: object) -> None:
    content = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _task(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("worker task must be a JSON object")
    return value


def train_task(task: dict[str, Any]) -> dict[str, object]:
    model_name = str(task["model"])
    train_arrays, train_manifest = read_array_artifact(
        Path(task["train_artifact"]), expected_kind="dataset"
    )
    validation_arrays, validation_manifest = read_array_artifact(
        Path(task["validation_artifact"]), expected_kind="dataset"
    )
    output_dir = Path(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(
        model_name,
        dict(task["configuration"]),
        n_steps=int(task.get("n_steps", train_arrays["X"].shape[1])),
        n_features=int(task.get("n_features", train_arrays["X"].shape[2])),
        batch_size=int(task["batch_size"]),
        epochs=int(task["epochs"]),
        patience=int(task["patience"]),
        device=str(task["device"]),
        saving_path=str(output_dir / "training"),
    )
    model.fit(
        train_set={"X": train_arrays["X"]},
        val_set={
            "X": validation_arrays["X"],
            "X_ori": validation_arrays["X_ori"],
        },
    )
    checkpoint = output_dir / "best.pypots"
    model.save(str(checkpoint))
    manifest: dict[str, object] = {
        "schema_version": 1,
        "model": model_name,
        "configuration": dict(task["configuration"]),
        "train_artifact_id": train_manifest["artifact_id"],
        "validation_artifact_id": validation_manifest["artifact_id"],
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    _write_json_no_clobber(output_dir / "checkpoint.json", manifest)
    return manifest


def predict_task(task: dict[str, Any]) -> dict[str, object]:
    model_name = str(task["model"])
    arrays, dataset_manifest = read_array_artifact(
        Path(task["dataset_artifact"]), expected_kind="dataset"
    )
    checkpoint = Path(task["checkpoint"])
    model = build_model(
        model_name,
        dict(task["configuration"]),
        n_steps=int(arrays["X"].shape[1]),
        n_features=int(arrays["X"].shape[2]),
        batch_size=int(task["batch_size"]),
        epochs=1,
        patience=1,
        device=str(task["device"]),
    )
    model.load(str(checkpoint))
    sampling_times = int(task.get("n_sampling_times", 1 if model_name != "csdi" else 50))
    if model_name == "csdi":
        prediction = model.predict(
            {"X": arrays["X"]}, n_sampling_times=sampling_times
        )
    else:
        if sampling_times != 1:
            raise ValueError("deterministic PyPOTS models require one sample")
        prediction = model.predict({"X": arrays["X"]})
    raw = prediction["imputation"] if isinstance(prediction, dict) else prediction
    samples = normalize_imputation(
        raw,
        windows=arrays["X"].shape[0],
        samples=sampling_times,
        steps=arrays["X"].shape[1],
        features=arrays["X"].shape[2],
    )
    samples = _force_observed(samples, arrays["X"])
    output = Path(task["output_artifact"])
    manifest = write_array_artifact(
        output,
        "prediction",
        {"samples": samples, "mean": samples.mean(axis=1, dtype=np.float64).astype(np.float32)},
        {
            "model": model_name,
            "configuration": dict(task["configuration"]),
            "dataset_artifact_id": dataset_manifest["artifact_id"],
            "checkpoint_sha256": sha256_file(checkpoint),
            "n_sampling_times": sampling_times,
        },
    )
    return manifest


def preflight(output: Path) -> dict[str, object]:
    import pypots
    import torch

    report: dict[str, object] = {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "pypots": getattr(pypots, "__version__", "unknown"),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    _write_json_no_clobber(Path(output), report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m validation_v2.modern.pypots_worker")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--task", type=Path, required=True)
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "train":
            result = train_task(_task(arguments.task))
        elif arguments.command == "predict":
            result = predict_task(_task(arguments.task))
        else:
            result = preflight(arguments.output)
    except (ImportError, OSError, TypeError, ValueError, KeyError) as error:
        print(f"modern-pypots: {error}", file=sys.stderr)
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_model",
    "main",
    "normalize_imputation",
    "predict_task",
    "preflight",
    "train_task",
]
