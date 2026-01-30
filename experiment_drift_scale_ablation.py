"""Experiment: drift_scale ablation for train-time random-walk drift augmentation."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from dataset import CfCIMUDataset
from models import AdaptiveLoss, ReconstructionOnlyLoss, build_model, count_parameters
from train import evaluate, train_one_epoch
from visualization import plot_training_curves


def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_float_list(s: str) -> List[float]:
    items: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            items.append(float(part))
    return items


def run_drift_scale_ablation(
    model_name: str = "lnn",
    drift_scale_values: List[float] | None = None,
    epochs: int = 30,
    output_dir: str = "results/drift_scale_ablation",
):
    config = {
        "root_dir": "Oxford Dataset",
        "seq_len": 50,
        "mask_rate": 0.3,
        "missing_mode": "random",
        "batch_size": 16,
        "epochs": int(epochs),
        "lr": 1e-3,
        "hidden_units": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 4,
        "seed": 42,
        "output_dir": str(output_dir),
        "model_name": str(model_name),
        "drift_scale_values": drift_scale_values or [0.0, 0.001, 0.005, 0.01, 0.02],
        "loss": "adaptive" if str(model_name).lower() in ["cfc", "lnn"] else "recon_only",
        "w_recon": 1.0,
        "w_consistency": 0.0,
        "w_smooth": 0.00,
    }

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("DRIFT_SCALE ABLATION")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {config['model_name']}")
    print(f"Loss: {config['loss']}")
    print(f"Missing: {config['missing_mode']} @ {config['mask_rate'] * 100:.0f}%")
    print(f"Epochs: {config['epochs']}")
    print(f"Drift scales: {config['drift_scale_values']}")
    print(f"Output: {output_path}")
    print("=" * 80)

    summary_rows: List[dict] = []
    history_rows: List[dict] = []

    for drift_scale in config["drift_scale_values"]:
        drift_scale = float(drift_scale)
        exp_name = f"{config['model_name']}_drift{drift_scale:g}"
        exp_label = f"drift_scale={drift_scale:g}"

        print(f"\n{'=' * 80}")
        print(f"Training: {exp_name}")
        print(f"{'=' * 80}")

        _seed_all(config["seed"])

        train_ds = CfCIMUDataset(
            root_dir=config["root_dir"],
            seq_len=config["seq_len"],
            mask_rate=config["mask_rate"],
            missing_mode=config["missing_mode"],
            split="train",
            eval_mode=False,
            drift_scale=drift_scale,
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

        model = build_model(
            model_name=config["model_name"],
            input_dim=13,
            hidden_dim=config["hidden_units"],
            output_dim=6,
        ).to(device)
        num_params = int(count_parameters(model))

        if config["loss"] == "adaptive":
            criterion = AdaptiveLoss(
                w_recon=config["w_recon"],
                w_consistency=config["w_consistency"],
                w_smooth=config["w_smooth"],
            )
        else:
            criterion = ReconstructionOnlyLoss(w_recon=config["w_recon"])

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

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_mse_all": [],
            "val_mse_masked": [],
        }

        for epoch in range(1, config["epochs"] + 1):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, epoch, use_physics=False
            )
            val_metrics = evaluate(model, val_loader, criterion, device, use_physics=False)

            history["train_loss"].append(float(train_metrics["total"]))
            history["val_loss"].append(float(val_metrics["total"]))
            history["val_mse_all"].append(float(val_metrics["mse_all"]))
            history["val_mse_masked"].append(float(val_metrics["mse_masked"]))

            history_rows.append(
                {
                    "drift_scale": drift_scale,
                    "epoch": epoch,
                    "train_loss": float(train_metrics["total"]),
                    "val_loss": float(val_metrics["total"]),
                    "val_mse_all": float(val_metrics["mse_all"]),
                    "val_mse_masked": float(val_metrics["mse_masked"]),
                }
            )

            if float(val_metrics["total"]) < best_val_loss:
                best_val_loss = float(val_metrics["total"])
                best_epoch = int(epoch)
                torch.save(model.state_dict(), best_weight_path)

        model.load_state_dict(torch.load(best_weight_path, map_location=device))
        best_eval = evaluate(model, val_loader, criterion, device, use_physics=False)

        plot_training_curves(
            history,
            save_path=output_path / f"training_curves_{exp_name}_{timestamp}.png",
        )

        summary_rows.append(
            {
                "model": config["model_name"],
                "loss": config["loss"],
                "drift_scale": drift_scale,
                "best_epoch": best_epoch,
                "best_val_loss": float(best_val_loss),
                "best_val_mse_masked": float(best_eval["mse_masked"]),
                "best_val_mse_all": float(best_eval["mse_all"]),
                "num_params": num_params,
                "weight_path": str(best_weight_path),
            }
        )

        print(f"Best Val Loss: {best_val_loss:.6f} @ epoch {best_epoch}")
        print(f"Best Val MSE(masked): {float(best_eval['mse_masked']):.6f}")
        print(f"Best Val MSE(all): {float(best_eval['mse_all']):.6f}")

    df_summary = pd.DataFrame(summary_rows).sort_values(
        by=["best_val_mse_masked", "best_val_loss"], ascending=[True, True]
    )
    summary_path = output_path / f"summary_drift_scale_{config['model_name']}_{timestamp}.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[Saved] Summary CSV: {summary_path}")

    df_history = pd.DataFrame(history_rows)
    history_path = output_path / f"history_drift_scale_{config['model_name']}_{timestamp}.csv"
    df_history.to_csv(history_path, index=False)
    print(f"[Saved] History CSV: {history_path}")

    pivot_path = output_path / f"val_mse_masked_by_epoch_{config['model_name']}_{timestamp}.csv"
    df_pivot = df_history.pivot(index="epoch", columns="drift_scale", values="val_mse_masked").reset_index()
    df_pivot.to_csv(pivot_path, index=False)
    print(f"[Saved] Val MSE(masked) by epoch CSV: {pivot_path}")

    return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--drift_scales", type=str, default="0,0.001,0.005,0.01,0.02")
    parser.add_argument("--output_dir", type=str, default="results/drift_scale_ablation")
    args = parser.parse_args()

    run_drift_scale_ablation(
        model_name=args.model,
        drift_scale_values=_parse_float_list(args.drift_scales),
        epochs=args.epochs,
        output_dir=args.output_dir,
    )

