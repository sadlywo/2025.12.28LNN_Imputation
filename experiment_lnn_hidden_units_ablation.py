"""Experiment: LNN hidden_units ablation under reconstruction-only loss."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from dataset import CfCIMUDataset
from models import build_model, count_parameters, ReconstructionOnlyLoss
from train import train_one_epoch, evaluate


def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _state_dict_size_mb(state_dict: Dict[str, torch.Tensor]) -> float:
    total_bytes = 0
    for v in state_dict.values():
        if torch.is_tensor(v):
            total_bytes += v.numel() * v.element_size()
    return float(total_bytes) / (1024.0 * 1024.0)


def _measure_inference_time(
    model,
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> Tuple[float, int, int]:
    model.eval()
    total_time = 0.0
    total_batches = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, _targets, _mask in loader:
            if max_batches is not None and total_batches >= max_batches:
                break
            inputs = inputs.to(device)
            _sync(device)
            t0 = time.time()
            out = model(inputs)
            _ = out[0] if isinstance(out, (tuple, list)) else out
            _sync(device)
            total_time += time.time() - t0
            total_batches += 1
            total_samples += int(inputs.shape[0])
    return float(total_time), int(total_batches), int(total_samples)


def run_hidden_units_ablation():
    config = {
        "root_dir": "Oxford Dataset",
        "seq_len": 50,
        "mask_rate": 0.3,
        "missing_mode": "random",
        "batch_size": 16,
        "epochs": 30,
        "lr": 1e-3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 4,
        "seed": 42,
        "output_dir": "results/lnn_hidden_units_ablation",
        "hidden_units_list": [32, 64, 128, 256],
        "inference_max_val_batches": None,
        "drift_scale": 0.01,
    }

    _seed_all(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("LNN HIDDEN_UNITS ABLATION (RECONSTRUCTION-ONLY)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Output: {output_path}")
    print(f"Hidden units: {config['hidden_units_list']}")
    print(f"Epochs: {config['epochs']}")
    print("=" * 80)

    train_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=config["mask_rate"],
        missing_mode=config["missing_mode"],
        split="train",
        eval_mode=False,
        drift_scale=config["drift_scale"],
    )
    val_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=config["mask_rate"],
        missing_mode=config["missing_mode"],
        split="val",
        eval_mode=True,
        drift_scale=0.0,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
    )

    criterion = ReconstructionOnlyLoss(w_recon=1.0)

    summary_rows: List[dict] = []
    history_rows: List[dict] = []
    weight_rows: List[dict] = []

    for hidden_units in config["hidden_units_list"]:
        exp_name = f"lnn_h{hidden_units}"
        print(f"\n{'=' * 80}")
        print(f"Training: {exp_name}")
        print(f"{'=' * 80}")

        model = build_model(
            model_name="lnn",
            input_dim=13,
            hidden_dim=hidden_units,
            output_dim=6,
        ).to(device)

        num_params = int(count_parameters(model))

        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["lr"],
            epochs=config["epochs"],
            steps_per_epoch=len(train_loader),
        )

        best_val_loss = float("inf")
        best_epoch = 0
        best_weight_path = output_path / f"best_model_{exp_name}_{timestamp}.pt"

        for epoch in range(1, config["epochs"] + 1):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, epoch, use_physics=False
            )
            val_metrics = evaluate(model, val_loader, criterion, device, use_physics=False)

            history_rows.append(
                {
                    "hidden_units": hidden_units,
                    "epoch": epoch,
                    "train_loss": train_metrics["total"],
                    "val_loss": val_metrics["total"],
                    "val_mse_all": val_metrics["mse_all"],
                    "val_mse_masked": val_metrics["mse_masked"],
                }
            )

            if val_metrics["total"] < best_val_loss:
                best_val_loss = float(val_metrics["total"])
                best_epoch = int(epoch)
                torch.save(model.state_dict(), best_weight_path)

        model.load_state_dict(torch.load(best_weight_path, map_location=device))
        best_eval = evaluate(model, val_loader, criterion, device, use_physics=False)
        infer_time, infer_batches, infer_samples = _measure_inference_time(
            model, val_loader, device=device, max_batches=config["inference_max_val_batches"]
        )
        state_dict = model.state_dict()
        state_size_mb = _state_dict_size_mb(state_dict)

        summary_rows.append(
            {
                "hidden_units": hidden_units,
                "best_epoch": best_epoch,
                "best_val_loss": float(best_val_loss),
                "best_val_mse_masked": float(best_eval["mse_masked"]),
                "best_val_mse_all": float(best_eval["mse_all"]),
                "num_params": num_params,
                "state_dict_size_mb": state_size_mb,
                "inference_time_sec": infer_time,
                "inference_batches": infer_batches,
                "inference_samples": infer_samples,
                "inference_ms_per_batch": 1000.0 * infer_time / max(infer_batches, 1),
                "inference_ms_per_sample": 1000.0 * infer_time / max(infer_samples, 1),
                "weight_path": str(best_weight_path),
            }
        )

        weight_rows.append(
            {
                "hidden_units": hidden_units,
                "weight_path": str(best_weight_path),
                "num_params": num_params,
                "state_dict_size_mb": state_size_mb,
                "inference_time_sec": infer_time,
                "inference_batches": infer_batches,
                "inference_samples": infer_samples,
                "inference_ms_per_batch": 1000.0 * infer_time / max(infer_batches, 1),
                "inference_ms_per_sample": 1000.0 * infer_time / max(infer_samples, 1),
            }
        )

        print(f"Best epoch: {best_epoch}")
        print(f"Best val MSE(masked): {best_eval['mse_masked']:.6f}")
        print(f"Params: {num_params:,}")
        print(f"Inference: {1000.0 * infer_time / max(infer_batches, 1):.3f} ms/batch")

    df_summary = pd.DataFrame(summary_rows).sort_values(by=["best_val_mse_masked", "best_val_loss"], ascending=[True, True])
    df_history = pd.DataFrame(history_rows)
    df_weights = pd.DataFrame(weight_rows).sort_values(by=["hidden_units"])

    summary_csv = output_path / f"summary_hidden_units_{timestamp}.csv"
    history_csv = output_path / f"history_hidden_units_{timestamp}.csv"
    weights_csv = output_path / f"weights_params_inference_{timestamp}.csv"
    df_summary.to_csv(summary_csv, index=False)
    df_history.to_csv(history_csv, index=False)
    df_weights.to_csv(weights_csv, index=False)

    excel_path = output_path / f"hidden_units_ablation_{timestamp}.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_summary.to_excel(writer, index=False, sheet_name="summary")
            df_weights.to_excel(writer, index=False, sheet_name="weights_speed")
            df_history.to_excel(writer, index=False, sheet_name="history")
            pd.DataFrame([{"key": k, "value": v} for k, v in config.items()]).to_excel(writer, index=False, sheet_name="config")
    except Exception as e:
        print(f"[Warning] Excel export failed: {e}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(df_weights["hidden_units"], df_summary.sort_values("hidden_units")["best_val_mse_masked"], marker="o", linewidth=2)
        plt.title("LNN Hidden Units Ablation (Reconstruction-Only)")
        plt.xlabel("hidden_units")
        plt.ylabel("Best Val MSE (masked)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = output_path / f"mse_vs_hidden_units_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[Warning] Plotting failed: {e}")

    print("\n" + "=" * 80)
    print(f"[Saved] {summary_csv}")
    print(f"[Saved] {weights_csv}")
    print(f"[Saved] {history_csv}")
    print(f"[Saved] {excel_path}")
    print("=" * 80)

    return df_summary


if __name__ == "__main__":
    run_hidden_units_ablation()

