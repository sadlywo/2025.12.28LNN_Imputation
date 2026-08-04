from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .artifacts import canonical_json, read_array_artifact, sha256_file, write_array_artifact


def sssd_parameters(*, residual_width: int) -> dict[str, object]:
    if residual_width not in {32, 64}:
        raise ValueError("SSSD residual width must be 32 or 64")
    return {
        "T": 200,
        "beta_0": 0.0001,
        "beta_T": 0.02,
        "in_channels": 6,
        "out_channels": 6,
        "res_channels": residual_width,
        "skip_channels": residual_width,
        "num_res_layers": 36,
        "diffusion_step_embed_dim_in": 128,
        "diffusion_step_embed_dim_mid": 512,
        "diffusion_step_embed_dim_out": 512,
        "s4_lmax": 30,
        "s4_d_state": 64,
        "s4_dropout": 0.0,
        "s4_bidirectional": True,
        "s4_layernorm": True,
    }


def diffusion_loss(
    predicted_noise: torch.Tensor, true_noise: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if predicted_noise.shape != true_noise.shape or mask.shape != true_noise.shape:
        raise ValueError("noise and mask tensors must have identical shapes")
    missing = mask == 0
    if not torch.any(missing):
        raise ValueError("diffusion loss requires artificially missing positions")
    return torch.mean((predicted_noise[missing] - true_noise[missing]) ** 2)


def diffusion_schedule(device: torch.device) -> dict[str, torch.Tensor | int]:
    params = sssd_parameters(residual_width=32)
    beta = torch.linspace(
        float(params["beta_0"]), float(params["beta_T"]), int(params["T"]), device=device
    )
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    beta_tilde = beta.clone()
    beta_tilde[1:] *= (1.0 - alpha_bar[:-1]) / (1.0 - alpha_bar[1:])
    beta_tilde[0] = 0.0
    return {
        "T": int(params["T"]),
        "Beta": beta,
        "Alpha": alpha,
        "Alpha_bar": alpha_bar,
        "Sigma": torch.sqrt(beta_tilde),
    }


def build_denoiser(source: Path, *, residual_width: int, device: torch.device) -> nn.Module:
    source = Path(source).resolve()
    source_root = source / "src" if (source / "src").is_dir() else source
    if not (source_root / "imputers" / "SSSDS4Imputer.py").is_file():
        raise ValueError(f"official SSSD source is incomplete: {source_root}")
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from imputers.SSSDS4Imputer import SSSDS4Imputer

    params = sssd_parameters(residual_width=residual_width)
    constructor = {key: value for key, value in params.items() if key not in {"T", "beta_0", "beta_T"}}
    return SSSDS4Imputer(**constructor).to(device)


def _normal(shape: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
    return torch.randn(shape, device=reference.device, dtype=reference.dtype)


def training_step(
    denoiser: nn.Module,
    target: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    schedule: dict[str, torch.Tensor | int],
) -> torch.Tensor:
    batch = target.shape[0]
    steps = torch.randint(int(schedule["T"]), (batch, 1), device=target.device)
    noise = _normal(tuple(target.shape), target)
    alpha_bar = schedule["Alpha_bar"][steps].reshape(batch, 1, 1)  # type: ignore[index]
    noisy = torch.sqrt(alpha_bar) * target + torch.sqrt(1.0 - alpha_bar) * noise
    prediction = denoiser((noisy, observed, mask, steps.to(target.dtype)))
    return diffusion_loss(prediction, noise, mask)


@torch.no_grad()
def sample_sssd(
    denoiser: nn.Module,
    observed: torch.Tensor,
    mask: torch.Tensor,
    schedule: dict[str, torch.Tensor | int],
    *,
    n_samples: int,
) -> torch.Tensor:
    if n_samples < 2:
        raise ValueError("SSSD probabilistic inference requires at least two samples")
    batch, channels, length = observed.shape
    expanded_observed = observed[:, None].expand(-1, n_samples, -1, -1).reshape(
        batch * n_samples, channels, length
    )
    expanded_mask = mask[:, None].expand_as(observed[:, None].expand(-1, n_samples, -1, -1)).reshape(
        batch * n_samples, channels, length
    )
    x = _normal(tuple(expanded_observed.shape), expanded_observed)
    alpha = schedule["Alpha"]
    alpha_bar = schedule["Alpha_bar"]
    sigma = schedule["Sigma"]
    for step in range(int(schedule["T"]) - 1, -1, -1):
        x = torch.where(expanded_mask.bool(), expanded_observed, x)
        steps = torch.full(
            (x.shape[0], 1), float(step), device=x.device, dtype=x.dtype
        )
        epsilon = denoiser((x, expanded_observed, expanded_mask, steps))
        x = (x - (1.0 - alpha[step]) / torch.sqrt(1.0 - alpha_bar[step]) * epsilon) / torch.sqrt(alpha[step])  # type: ignore[index]
        if step > 0:
            x = x + sigma[step] * _normal(tuple(x.shape), x)  # type: ignore[index]
        x = torch.where(expanded_mask.bool(), expanded_observed, x)
    return x.reshape(batch, n_samples, channels, length)


def _tensors(arrays: dict[str, np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.as_tensor(arrays["X_ori"], dtype=torch.float32, device=device).permute(0, 2, 1)
    mask = torch.as_tensor(arrays["mask"], dtype=torch.float32, device=device).permute(0, 2, 1)
    observed = torch.nan_to_num(
        torch.as_tensor(arrays["X"], dtype=torch.float32, device=device), nan=0.0
    ).permute(0, 2, 1)
    return target, observed, mask


def _atomic_checkpoint(path: Path, payload: dict[str, object]) -> None:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".sssd-", suffix=".pt")
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: object) -> None:
    content = (canonical_json(value) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def train_task(task: dict[str, Any]) -> dict[str, object]:
    device = torch.device(str(task["device"]))
    train_arrays, train_manifest = read_array_artifact(Path(task["train_artifact"]), expected_kind="dataset")
    validation_arrays, validation_manifest = read_array_artifact(Path(task["validation_artifact"]), expected_kind="dataset")
    width = int(task["configuration"]["residual_channels"])
    model = build_denoiser(Path(task["source"]), residual_width=width, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(task["configuration"]["learning_rate"]))
    schedule = diffusion_schedule(device)
    target, observed, mask = _tensors(train_arrays, device)
    loader = DataLoader(TensorDataset(target, observed, mask), batch_size=int(task["batch_size"]), shuffle=True)
    val_target, val_observed, val_mask = _tensors(validation_arrays, device)
    output = Path(task["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best.pt"
    best = float("inf")
    stale = 0
    best_epoch = 0
    for epoch in range(1, int(task["epochs"]) + 1):
        model.train()
        for batch_target, batch_observed, batch_mask in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = training_step(model, batch_target, batch_observed, batch_mask, schedule)
            if not torch.isfinite(loss):
                raise ValueError("SSSD training produced nonfinite loss")
            loss.backward()
            optimizer.step()
        model.eval()
        samples = sample_sssd(model, val_observed, val_mask, schedule, n_samples=2).mean(dim=1)
        missing = val_mask == 0
        score = float(torch.sqrt(torch.mean((samples[missing] - val_target[missing]) ** 2)).cpu())
        if score < best:
            best, stale, best_epoch = score, 0, epoch
            _atomic_checkpoint(checkpoint, {"state_dict": model.state_dict(), "parameters": sssd_parameters(residual_width=width)})
        else:
            stale += 1
        if stale >= int(task["patience"]):
            break
    manifest = {
        "schema_version": 1, "model": "sssd", "best_epoch": best_epoch,
        "validation_missing_rmse": best, "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_artifact_id": train_manifest["artifact_id"],
        "validation_artifact_id": validation_manifest["artifact_id"],
        "configuration": dict(task["configuration"]),
    }
    _write_json(output / "checkpoint.json", manifest)
    return manifest


def predict_task(task: dict[str, Any]) -> dict[str, object]:
    device = torch.device(str(task["device"]))
    arrays, dataset_manifest = read_array_artifact(Path(task["dataset_artifact"]), expected_kind="dataset")
    width = int(task["configuration"]["residual_channels"])
    model = build_denoiser(Path(task["source"]), residual_width=width, device=device)
    checkpoint = Path(task["checkpoint"])
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    _, observed, mask = _tensors(arrays, device)
    generated = sample_sssd(model, observed, mask, diffusion_schedule(device), n_samples=int(task["n_sampling_times"]))
    samples = generated.permute(0, 1, 3, 2).cpu().numpy().astype(np.float32)
    return write_array_artifact(
        Path(task["output_artifact"]), "prediction",
        {"samples": samples, "mean": samples.mean(axis=1, dtype=np.float64).astype(np.float32)},
        {"model": "sssd", "configuration": dict(task["configuration"]),
         "dataset_artifact_id": dataset_manifest["artifact_id"],
         "checkpoint_sha256": sha256_file(checkpoint), "n_sampling_times": int(task["n_sampling_times"])},
    )


def preflight(source: Path, output: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise ValueError("SSSD preflight requires CUDA")
    device = torch.device("cuda")
    model = build_denoiser(source, residual_width=32, device=device)
    schedule = diffusion_schedule(device)
    target = torch.randn(1, 6, 30, device=device)
    mask = torch.ones_like(target); mask[:, :, 10:15] = 0
    observed = target * mask
    loss = training_step(model, target, observed, mask, schedule)
    loss.backward()
    model.eval()
    samples = sample_sssd(model, observed, mask, schedule, n_samples=2)
    if not torch.isfinite(samples).all() or torch.equal(samples[:, 0], samples[:, 1]):
        raise ValueError("SSSD preflight sampling contract failed")
    report = {"schema_version": 1, "python": sys.version, "platform": platform.platform(),
              "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(0),
              "loss": float(loss.detach().cpu()), "parameters": sum(p.numel() for p in model.parameters())}
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m validation_v2.modern.sssd_worker")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "predict"):
        item = commands.add_parser(name); item.add_argument("--task", type=Path, required=True)
    item = commands.add_parser("preflight")
    item.add_argument("--source", type=Path, required=True); item.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "preflight":
            result = preflight(arguments.source, arguments.output)
        else:
            task = json.loads(arguments.task.read_text(encoding="utf-8"))
            result = train_task(task) if arguments.command == "train" else predict_task(task)
    except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"modern-sssd: {error}", file=sys.stderr); return 2
    print(canonical_json(result)); return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_denoiser", "diffusion_loss", "diffusion_schedule", "main", "sample_sssd", "sssd_parameters"]
