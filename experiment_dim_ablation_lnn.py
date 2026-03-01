"""Experiment: Input dimension ablation with LNN only.

Compares:
1) 13-dim input: base IMU (gyro+acc) + mask + dt
2) 19-dim input: base IMU + gravity + mask + dt

Notes:
- 13-dim input => feature_dim=6, input_dim=6*2+1=13
- 19-dim input => feature_dim=9 (gyro+acc+gravity), input_dim=9*2+1=19
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset import CfCIMUDataset
from models import LNNImputer, ReconstructionOnlyLoss
from visualization import plot_training_curves


def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    losses = {"total": []}

    for inputs, targets, mask in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        dt = inputs[:, :, -1:]

        optimizer.zero_grad()
        pred, uncertainty = model(inputs)
        loss, _ = criterion(pred, targets, mask, uncertainty, dt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses["total"].append(loss.item())

    return {k: float(np.mean(v)) for k, v in losses.items()}


def _evaluate(model, loader, criterion, device):
    model.eval()
    losses = {"total": []}
    mse_all, mse_masked = [], []

    with torch.no_grad():
        for inputs, targets, mask in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            dt = inputs[:, :, -1:]

            pred, uncertainty = model(inputs)
            loss, _ = criterion(pred, targets, mask, uncertainty, dt)

            losses["total"].append(loss.item())
            mse_all.append(F.mse_loss(pred, targets).item())
            missing_err = ((pred - targets) ** 2 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
            mse_masked.append(missing_err.item())

    metrics = {k: float(np.mean(v)) for k, v in losses.items()}
    metrics["mse_all"] = float(np.mean(mse_all))
    metrics["mse_masked"] = float(np.mean(mse_masked))
    return metrics


def run_dim_ablation():
    config = {
        "root_dir": "Oxford Dataset",
        "seq_len": 50,
        "mask_rate": 0.3,
        "missing_mode": "random",
        "batch_size": 16,
        "epochs": 30,
        "lr": 1e-3,
        "hidden_units": 64,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": "results/dim_ablation_lnn",
        "num_workers": 4,
        "drift_scale": 0.00,
        "seed": 42,
    }

    _seed_all(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("INPUT DIM ABLATION (LNN ONLY)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Output: {output_path}")
    print(f"Missing: {config['missing_mode']} @ {config['mask_rate'] * 100:.0f}%")
    print(f"Epochs: {config['epochs']}")
    print("=" * 80)

    experiments = {
        "LNN_dim13": {"use_gravity": False, "use_attitude": False},
        "LNN_dim19": {"use_gravity": True, "use_attitude": False},
    }

    summary_rows: List[dict] = []
    history_rows: List[dict] = []

    for name, flags in experiments.items():
        print(f"\n{'=' * 80}")
        print(f"Training: {name} (gravity={flags['use_gravity']}, attitude={flags['use_attitude']})")
        print(f"{'=' * 80}")

        train_ds = CfCIMUDataset(
            root_dir=config["root_dir"],
            seq_len=config["seq_len"],
            mask_rate=config["mask_rate"],
            missing_mode=config["missing_mode"],
            split="train",
            eval_mode=False,
            drift_scale=config["drift_scale"],
            use_gravity=flags["use_gravity"],
            use_attitude=flags["use_attitude"],
        )
        val_ds = CfCIMUDataset(
            root_dir=config["root_dir"],
            seq_len=config["seq_len"],
            mask_rate=config["mask_rate"],
            missing_mode=config["missing_mode"],
            split="val",
            eval_mode=True,
            drift_scale=0.0,
            use_gravity=flags["use_gravity"],
            use_attitude=flags["use_attitude"],
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

        model = LNNImputer(
            input_dim=train_ds.input_dim,
            hidden_units=config["hidden_units"],
            output_dim=train_ds.feature_dim,
        ).to(device)
        criterion = ReconstructionOnlyLoss(w_recon=1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["lr"],
            epochs=config["epochs"],
            steps_per_epoch=len(train_loader),
        )

        history = {"train_loss": [], "val_loss": [], "val_mse_all": [], "val_mse_masked": []}
        best_val_loss = float("inf")
        best_epoch = 0
        best_weight_path = output_path / f"best_model_{name}_{timestamp}.pt"

        start_time = time.time()
        for epoch in range(1, config["epochs"] + 1):
            train_metrics = _train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, device
            )
            val_metrics = _evaluate(model, val_loader, criterion, device)

            history["train_loss"].append(float(train_metrics["total"]))
            history["val_loss"].append(float(val_metrics["total"]))
            history["val_mse_all"].append(float(val_metrics["mse_all"]))
            history["val_mse_masked"].append(float(val_metrics["mse_masked"]))

            if float(val_metrics["total"]) < best_val_loss:
                best_val_loss = float(val_metrics["total"])
                best_epoch = int(epoch)
                torch.save(model.state_dict(), best_weight_path)

            if epoch % 10 == 0 or epoch == config["epochs"]:
                print(
                    f"  [{name}] Epoch {epoch:3d}/{config['epochs']}  "
                    f"train_loss={train_metrics['total']:.6f}  "
                    f"val_MSE_all={val_metrics['mse_all']:.6f}  "
                    f"val_MSE_masked={val_metrics['mse_masked']:.6f}"
                )

        train_time = time.time() - start_time

        plot_training_curves(
            history,
            save_path=output_path / f"training_curves_{name}_{timestamp}.png",
        )

        for epoch_idx in range(len(history["train_loss"])):
            history_rows.append({
                "model": name,
                "epoch": epoch_idx + 1,
                "train_loss": history["train_loss"][epoch_idx],
                "val_loss": history["val_loss"][epoch_idx],
                "val_mse_all": history["val_mse_all"][epoch_idx],
                "val_mse_masked": history["val_mse_masked"][epoch_idx],
            })

        summary_rows.append({
            "model": name,
            "input_dim": train_ds.input_dim,
            "feature_dim": train_ds.feature_dim,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "best_val_mse_all": float(min(history["val_mse_all"])),
            "best_val_mse_masked": float(min(history["val_mse_masked"])),
            "train_time_sec": round(train_time, 2),
        })

    try:
        summary_df = pd.DataFrame(summary_rows)
        history_df = pd.DataFrame(history_rows)

        summary_csv = output_path / f"summary_dim_ablation_{timestamp}.csv"
        history_csv = output_path / f"history_dim_ablation_{timestamp}.csv"
        summary_df.to_csv(summary_csv, index=False)
        history_df.to_csv(history_csv, index=False)
        print(f"\n[Saved] {summary_csv}")
        print(f"[Saved] {history_csv}")
    except Exception as e:
        print(f"[Warning] Failed to save CSV: {e}")

    print("\n" + "=" * 80)
    print("DONE: INPUT DIM ABLATION")
    print("=" * 80)


if __name__ == "__main__":
    run_dim_ablation()
