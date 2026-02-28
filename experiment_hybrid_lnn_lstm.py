"""
Experiment: Hybrid LNN + LSTM imputation.

Compares:
1. LNN-only (short-term CfC)
2. LSTM-only (long-term bidirectional LSTM)
3. Hybrid (learned gating fusion)
4. Hybrid + RMSE reweight (post-hoc RMSE-based fusion on validation)
5. Light full-sequence loss variants

Outputs:
- CSV summary with MSE metrics, parameter counts, and timing
- Per-epoch training history
- Training curves and sample visualization
- Gate analysis (LNN vs LSTM contribution per channel/timestep)
- Excel workbook with all results
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CfCIMUDataset, compute_ate, compute_relative_trajectory_error
from models import ReconstructionOnlyLoss
from models_hybrid import (
    ShortTermLNN,
    LongTermLSTM,
    HybridLNNLSTM,
    count_parameters,
    rmse_based_reweight,
)
from train import train_one_epoch, evaluate
from visualization import plot_training_curves, plot_imputation_samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _state_dict_size_mb(model: nn.Module) -> float:
    total = 0
    for v in model.state_dict().values():
        total += v.nelement() * v.element_size()
    return total / (1024 * 1024)


class LightAllLoss(nn.Module):
    """Lightweight full-sequence loss: missing MSE + lambda * full MSE."""

    def __init__(self, w_recon: float = 1.0, w_all: float = 0.1):
        super().__init__()
        self.w_recon = w_recon
        self.w_all = w_all

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        uncertainty: torch.Tensor = None,
        dt: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        missing = ((pred - target) ** 2 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
        full = F.mse_loss(pred, target)
        total = self.w_recon * missing + self.w_all * full
        components = {"recon": missing.item(), "all": full.item()}
        return total, components


def _measure_inference_time(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: Optional[int] = None,
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


# ---------------------------------------------------------------------------
# Training helpers for single models
# ---------------------------------------------------------------------------

def _train_model(
    model: nn.Module,
    model_name: str,
    train_loader,
    val_loader,
    device: torch.device,
    config: dict,
    output_path: Path,
    timestamp: str,
    criterion: nn.Module | None = None,
) -> Tuple[dict, dict, float]:
    """
    Train a model with standard loop, returning history, best eval metrics, and train time.
    """
    criterion = criterion or ReconstructionOnlyLoss(w_recon=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        epochs=config["epochs"],
        steps_per_epoch=len(train_loader),
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_mse_all": [],
        "val_mse_masked": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    best_weight_path = output_path / f"best_model_{model_name}_{timestamp}.pt"

    start_time = time.time()
    for epoch in range(1, config["epochs"] + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, use_physics=False
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_physics=False)

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
                f"  [{model_name}] Epoch {epoch:3d}/{config['epochs']}  "
                f"train_loss={train_metrics['total']:.6f}  "
                f"val_MSE_all={val_metrics['mse_all']:.6f}  "
                f"val_MSE_masked={val_metrics['mse_masked']:.6f}"
            )

    train_time = time.time() - start_time

    # Load best weights
    model.load_state_dict(torch.load(best_weight_path, map_location=device))
    best_eval = evaluate(model, val_loader, criterion, device, use_physics=False)

    # Plot training curves
    plot_training_curves(
        history,
        save_path=output_path / f"training_curves_{model_name}_{timestamp}.png",
    )

    return history, {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_mse_all": float(best_eval["mse_all"]),
        "best_val_mse_masked": float(best_eval["mse_masked"]),
    }, train_time


# ---------------------------------------------------------------------------
# RMSE-based post-hoc evaluation
# ---------------------------------------------------------------------------

def _evaluate_rmse_fusion(
    lnn_model: nn.Module,
    lstm_model: nn.Module,
    val_loader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate RMSE-based post-hoc fusion of standalone LNN and LSTM.
    """
    lnn_model.eval()
    lstm_model.eval()

    total_mse_all = 0.0
    total_mse_masked = 0.0
    total_samples = 0
    all_w_lnn = []
    all_w_lstm = []

    with torch.no_grad():
        for inputs, targets, mask in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            lnn_pred, _ = lnn_model(inputs)
            lstm_pred, _ = lstm_model(inputs)

            fused, w_lnn, w_lstm = rmse_based_reweight(lnn_pred, lstm_pred, targets, mask)

            mse_all = F.mse_loss(fused, targets).item()
            missing_err = ((fused - targets) ** 2 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)

            batch_size = inputs.shape[0]
            total_mse_all += mse_all * batch_size
            total_mse_masked += missing_err.item() * batch_size
            total_samples += batch_size
            all_w_lnn.append(w_lnn)
            all_w_lstm.append(w_lstm)

    return {
        "mse_all": total_mse_all / max(total_samples, 1),
        "mse_masked": total_mse_masked / max(total_samples, 1),
        "avg_w_lnn": float(np.mean(all_w_lnn)),
        "avg_w_lstm": float(np.mean(all_w_lstm)),
    }


def _evaluate_ate_rmse_fusion(
    lnn_model: nn.Module,
    lstm_model: nn.Module,
    ate_loader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate ATE for RMSE-based post-hoc fusion.
    """
    lnn_model.eval()
    lstm_model.eval()
    
    all_ate = []
    all_max_drift = []
    all_axis_errors = []
    
    with torch.no_grad():
        for batch in ate_loader:
            if len(batch) < 5:
                continue
                
            inputs, targets, mask, stats, vicon = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            stats = stats.to(device)
            vicon = vicon.to(device)
            
            # Get predictions from both models
            lnn_pred, _ = lnn_model(inputs)
            lstm_pred, _ = lstm_model(inputs)
            
            # RMSE-based fusion
            fused, _, _ = rmse_based_reweight(lnn_pred, lstm_pred, targets, mask)
            
            # Extract dt
            dt = inputs[:, :, -1]
            
            try:
                ate_result = compute_ate(
                    pred_acc=fused,
                    gt_pos=vicon,
                    dt=dt,
                    stats=stats,
                    acc_indices=(3, 4, 5),
                )
                all_ate.append(ate_result["ate"])
                all_max_drift.append(ate_result["max_drift"])
                all_axis_errors.append(ate_result["ate_per_axis"])
            except Exception as e:
                print(f"[Warning] RMSE fusion ATE computation failed: {e}")
                continue
    
    if len(all_ate) == 0:
        return {"ate": float("nan"), "max_drift": float("nan"), "ate_x": float("nan"), "ate_y": float("nan"), "ate_z": float("nan")}
    
    return {
        "ate": float(np.mean(all_ate)),
        "max_drift": float(np.mean(all_max_drift)),
        "ate_x": float(np.mean([e[0] for e in all_axis_errors])),
        "ate_y": float(np.mean([e[1] for e in all_axis_errors])),
        "ate_z": float(np.mean([e[2] for e in all_axis_errors])),
    }


# ---------------------------------------------------------------------------
# ATE (Absolute Trajectory Error) evaluation
# ---------------------------------------------------------------------------

def _evaluate_ate(
    model: nn.Module,
    ate_loader,
    device: torch.device,
    use_denorm: bool = True,
) -> Dict[str, float]:
    """
    Evaluate Absolute Trajectory Error by integrating predicted acceleration.
    
    Args:
        model: Trained imputation model
        ate_loader: DataLoader with return_vicon=True, return_stats=True
        device: Computation device
        use_denorm: Whether to denormalize acceleration before integration
    
    Returns:
        dict with ATE metrics (mean, per_axis, max_drift)
    """
    model.eval()
    
    all_ate = []
    all_max_drift = []
    all_axis_errors = []
    
    with torch.no_grad():
        for batch in ate_loader:
            # batch: (inputs, targets, mask, stats, vicon)
            if len(batch) < 5:
                print("[Warning] ATE loader missing vicon data, skipping batch")
                continue
                
            inputs, targets, mask, stats, vicon = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            stats = stats.to(device) if use_denorm else None
            vicon = vicon.to(device)
            
            # Get model predictions
            out = model(inputs)
            pred = out[0] if isinstance(out, (tuple, list)) else out
            
            # Extract dt from inputs (last channel)
            dt = inputs[:, :, -1]  # (batch, seq_len)
            
            # Compute ATE
            try:
                ate_result = compute_ate(
                    pred_acc=pred,
                    gt_pos=vicon,
                    dt=dt,
                    stats=stats,
                    acc_indices=(3, 4, 5),  # user_acc indices
                )
                all_ate.append(ate_result["ate"])
                all_max_drift.append(ate_result["max_drift"])
                all_axis_errors.append(ate_result["ate_per_axis"])
            except Exception as e:
                print(f"[Warning] ATE computation failed: {e}")
                continue
    
    if len(all_ate) == 0:
        return {"ate": float("nan"), "max_drift": float("nan"), "ate_x": float("nan"), "ate_y": float("nan"), "ate_z": float("nan")}
    
    mean_ate = float(np.mean(all_ate))
    mean_max_drift = float(np.mean(all_max_drift))
    mean_axis = np.mean(all_axis_errors, axis=0)
    
    return {
        "ate": mean_ate,
        "max_drift": mean_max_drift,
        "ate_x": float(mean_axis[0]),
        "ate_y": float(mean_axis[1]),
        "ate_z": float(mean_axis[2]),
    }


# ---------------------------------------------------------------------------
# Gate analysis
# ---------------------------------------------------------------------------

def _analyze_gate(
    hybrid_model: HybridLNNLSTM,
    val_loader,
    device: torch.device,
    output_path: Path,
    timestamp: str,
):
    """
    Analyze the learned gating weights from the hybrid model.
    """
    hybrid_model.eval()
    all_gates = []

    with torch.no_grad():
        for inputs, targets, mask in val_loader:
            inputs = inputs.to(device)
            components = hybrid_model.forward_with_components(inputs)
            all_gates.append(components["gate"].cpu())

    gates = torch.cat(all_gates, dim=0)  # (N, T, 6)
    mean_gate = gates.mean(dim=0).numpy()  # (T, 6)

    channel_names = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 1. Gate over time (averaged across samples)
    ax = axes[0]
    for c in range(6):
        ax.plot(mean_gate[:, c], label=channel_names[c], linewidth=1.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("LNN weight (gate)")
    ax.set_title("Learned Gate: LNN Weight over Time (1=LNN, 0=LSTM)")
    ax.legend(fontsize=8, ncol=3)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 2. Gate distribution per channel
    ax = axes[1]
    gate_flat = gates.numpy().reshape(-1, 6)
    bp = ax.boxplot(
        [gate_flat[:, c] for c in range(6)],
        labels=channel_names,
        patch_artist=True,
    )
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("LNN weight (gate)")
    ax.set_title("Gate Distribution per Channel")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = output_path / f"gate_analysis_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Gate analysis: {plot_path}")

    # Per-channel stats
    stats = {}
    for c in range(6):
        stats[channel_names[c]] = {
            "mean": float(gate_flat[:, c].mean()),
            "std": float(gate_flat[:, c].std()),
            "median": float(np.median(gate_flat[:, c])),
        }
    return stats


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def _plot_comparison(
    summary_df: pd.DataFrame,
    output_path: Path,
    timestamp: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    models = summary_df["model"].tolist()
    x = np.arange(len(models))
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(models))))

    # MSE masked
    ax = axes[0]
    bars = ax.bar(x, summary_df["best_val_mse_masked"], color=colors[:len(models)])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("MSE (masked)")
    ax.set_title("MSE on Missing Positions")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, summary_df["best_val_mse_masked"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.6f}", ha="center", va="bottom", fontsize=8)

    # MSE all
    ax = axes[1]
    bars = ax.bar(x, summary_df["best_val_mse_all"], color=colors[:len(models)])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("MSE (all)")
    ax.set_title("MSE on All Positions")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, summary_df["best_val_mse_all"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.6f}", ha="center", va="bottom", fontsize=8)

    # Parameters
    ax = axes[2]
    params = summary_df["num_params"].tolist()
    bars = ax.bar(x, params, color=colors[:len(models)])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Parameters")
    ax.set_title("Model Size")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("Hybrid LNN-LSTM vs Single Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = output_path / f"comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Comparison plot: {plot_path}")


def _plot_all_histories(
    all_histories: Dict[str, dict],
    output_path: Path,
    timestamp: str,
):
    """Plot overlaid training curves for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for model_name, history in all_histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=model_name, linewidth=1.5)
        axes[1].plot(epochs, history["val_mse_all"], label=model_name, linewidth=1.5)
        axes[2].plot(epochs, history["val_mse_masked"], label=model_name, linewidth=1.5)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation MSE (All)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Validation MSE (Masked)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MSE")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Training History Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = output_path / f"all_training_curves_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Combined training curves: {plot_path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_hybrid_experiment(
    epochs: int = 30,
    lnn_hidden: int = 64,
    lstm_hidden: int = 128,
    lstm_layers: int = 2,
    seq_len: int = 50,
    mask_rate: float = 0.3,
    missing_mode: str = "random",
    drift_scale: float = 0.01,
    output_dir: str = "results/hybrid_lnn_lstm",
):
    config = {
        "root_dir": "Oxford Dataset",
        "seq_len": int(seq_len),
        "mask_rate": float(mask_rate),
        "missing_mode": str(missing_mode),
        "batch_size": 16,
        "epochs": int(epochs),
        "lr": 1e-3,
        "lnn_hidden": int(lnn_hidden),
        "lstm_hidden": int(lstm_hidden),
        "lstm_layers": int(lstm_layers),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 4,
        "seed": 42,
        "output_dir": str(output_dir),
        "drift_scale": float(drift_scale),
    }

    _seed_all(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("HYBRID LNN-LSTM IMPUTATION EXPERIMENT")
    print("=" * 80)
    print(f"Device:       {device}")
    print(f"Seq len:      {config['seq_len']}")
    print(f"Missing:      {config['missing_mode']} @ {config['mask_rate'] * 100:.0f}%")
    print(f"Epochs:       {config['epochs']}")
    print(f"LNN hidden:   {config['lnn_hidden']}")
    print(f"LSTM hidden:  {config['lstm_hidden']} x {config['lstm_layers']} layers (BiLSTM)")
    print(f"Drift scale:  {config['drift_scale']}")
    print(f"Output:       {output_path}")
    print("=" * 80)

    # ----- Data -----
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
    
    # ATE evaluation dataset (with Vicon ground truth)
    ate_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=config["mask_rate"],
        missing_mode=config["missing_mode"],
        split="val",
        eval_mode=True,
        drift_scale=0.0,
        return_stats=True,
        return_vicon=True,
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
    
    ate_loader = torch.utils.data.DataLoader(
        ate_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"\nTrain samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"ATE eval samples: {len(ate_ds)}")

    # ----- Models -----
    models_to_train = {
        "LNN_only": {
            "model": ShortTermLNN(
                input_dim=13,
                hidden_units=config["lnn_hidden"],
                output_dim=6,
            ),
            "criterion": ReconstructionOnlyLoss(w_recon=1.0),
        },
        "LSTM_only": {
            "model": LongTermLSTM(
                input_dim=13,
                hidden_dim=config["lstm_hidden"],
                output_dim=6,
                num_layers=config["lstm_layers"],
            ),
            "criterion": ReconstructionOnlyLoss(w_recon=1.0),
        },
        "Hybrid_LNN_LSTM": {
            "model": HybridLNNLSTM(
                input_dim=13,
                lnn_hidden=config["lnn_hidden"],
                lstm_hidden=config["lstm_hidden"],
                output_dim=6,
                lstm_layers=config["lstm_layers"],
                fusion_mode="learned",
            ),
            "criterion": ReconstructionOnlyLoss(w_recon=1.0),
        },
        "Hybrid_LNN_LSTM_Uncertainty": {
            "model": HybridLNNLSTM(
                input_dim=13,
                lnn_hidden=config["lnn_hidden"],
                lstm_hidden=config["lstm_hidden"],
                output_dim=6,
                lstm_layers=config["lstm_layers"],
                fusion_mode="uncertainty",
            ),
            "criterion": ReconstructionOnlyLoss(w_recon=1.0),
        },
        "LNN_only_LightAll": {
            "model": ShortTermLNN(
                input_dim=13,
                hidden_units=config["lnn_hidden"],
                output_dim=6,
            ),
            "criterion": LightAllLoss(w_recon=1.0, w_all=0.1),
        },
        "LSTM_only_LightAll": {
            "model": LongTermLSTM(
                input_dim=13,
                hidden_dim=config["lstm_hidden"],
                output_dim=6,
                num_layers=config["lstm_layers"],
            ),
            "criterion": LightAllLoss(w_recon=1.0, w_all=0.1),
        },
        "Hybrid_LNN_LSTM_LightAll": {
            "model": HybridLNNLSTM(
                input_dim=13,
                lnn_hidden=config["lnn_hidden"],
                lstm_hidden=config["lstm_hidden"],
                output_dim=6,
                lstm_layers=config["lstm_layers"],
                fusion_mode="learned",
            ),
            "criterion": LightAllLoss(w_recon=1.0, w_all=0.1),
        },
    }

    summary_rows: List[dict] = []
    history_rows: List[dict] = []
    all_histories: Dict[str, dict] = {}

    for model_name, payload in models_to_train.items():
        model = payload["model"].to(device)
        criterion = payload["criterion"]
        num_params = count_parameters(model)
        size_mb = _state_dict_size_mb(model)

        print(f"\n{'=' * 80}")
        print(f"Training: {model_name}  (params={num_params:,}, size={size_mb:.2f} MB)")
        print(f"{'=' * 80}")

        _seed_all(config["seed"])

        history, best_metrics, train_time = _train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            output_path=output_path,
            timestamp=timestamp,
            criterion=criterion,
        )
        all_histories[model_name] = history

        # Inference timing
        infer_time, infer_batches, infer_samples = _measure_inference_time(
            model, val_loader, device
        )
        
        # ATE evaluation
        ate_metrics = _evaluate_ate(model, ate_loader, device, use_denorm=True)

        # Save history rows
        for epoch_idx in range(len(history["train_loss"])):
            history_rows.append({
                "model": model_name,
                "epoch": epoch_idx + 1,
                "train_loss": history["train_loss"][epoch_idx],
                "val_loss": history["val_loss"][epoch_idx],
                "val_mse_all": history["val_mse_all"][epoch_idx],
                "val_mse_masked": history["val_mse_masked"][epoch_idx],
            })

        row = {
            "model": model_name,
            "num_params": num_params,
            "param_size_mb": round(size_mb, 4),
            "best_epoch": best_metrics["best_epoch"],
            "best_val_loss": best_metrics["best_val_loss"],
            "best_val_mse_all": best_metrics["best_val_mse_all"],
            "best_val_mse_masked": best_metrics["best_val_mse_masked"],
            "ate": ate_metrics["ate"],
            "ate_max_drift": ate_metrics["max_drift"],
            "ate_x": ate_metrics["ate_x"],
            "ate_y": ate_metrics["ate_y"],
            "ate_z": ate_metrics["ate_z"],
            "train_time_sec": round(train_time, 2),
            "inference_time_sec": round(infer_time, 4),
            "inference_ms_per_sample": round(1000.0 * infer_time / max(infer_samples, 1), 4),
        }
        summary_rows.append(row)

        print(f"  Best epoch:          {best_metrics['best_epoch']}")
        print(f"  Best val MSE(masked): {best_metrics['best_val_mse_masked']:.6f}")
        print(f"  Best val MSE(all):    {best_metrics['best_val_mse_all']:.6f}")
        print(f"  ATE (trajectory):     {ate_metrics['ate']:.4f} m")
        print(f"  ATE max drift:        {ate_metrics['max_drift']:.4f} m")
        print(f"  Train time:          {train_time:.1f}s")
        print(f"  Inference:           {1000.0 * infer_time / max(infer_batches, 1):.3f} ms/batch")

    # ----- RMSE-based post-hoc fusion -----
    print(f"\n{'=' * 80}")
    print("Evaluating RMSE-based post-hoc fusion (LNN + LSTM)")
    print(f"{'=' * 80}")

    lnn_model = models_to_train["LNN_only"]["model"].to(device)
    lstm_model = models_to_train["LSTM_only"]["model"].to(device)

    # Load best weights
    lnn_model.load_state_dict(
        torch.load(output_path / f"best_model_LNN_only_{timestamp}.pt", map_location=device)
    )
    lstm_model.load_state_dict(
        torch.load(output_path / f"best_model_LSTM_only_{timestamp}.pt", map_location=device)
    )

    rmse_results = _evaluate_rmse_fusion(lnn_model, lstm_model, val_loader, device)
    
    # ATE for RMSE fusion
    rmse_ate_metrics = _evaluate_ate_rmse_fusion(lnn_model, lstm_model, ate_loader, device)

    summary_rows.append({
        "model": "RMSE_Fusion(LNN+LSTM)",
        "num_params": count_parameters(lnn_model) + count_parameters(lstm_model),
        "param_size_mb": round(_state_dict_size_mb(lnn_model) + _state_dict_size_mb(lstm_model), 4),
        "best_epoch": "-",
        "best_val_loss": "-",
        "best_val_mse_all": rmse_results["mse_all"],
        "best_val_mse_masked": rmse_results["mse_masked"],
        "ate": rmse_ate_metrics["ate"],
        "ate_max_drift": rmse_ate_metrics["max_drift"],
        "ate_x": rmse_ate_metrics["ate_x"],
        "ate_y": rmse_ate_metrics["ate_y"],
        "ate_z": rmse_ate_metrics["ate_z"],
        "train_time_sec": "-",
        "inference_time_sec": "-",
        "inference_ms_per_sample": "-",
    })

    print(f"  RMSE Fusion MSE(masked): {rmse_results['mse_masked']:.6f}")
    print(f"  RMSE Fusion MSE(all):    {rmse_results['mse_all']:.6f}")
    print(f"  RMSE Fusion ATE:         {rmse_ate_metrics['ate']:.4f} m")
    print(f"  Avg w_LNN:               {rmse_results['avg_w_lnn']:.4f}")
    print(f"  Avg w_LSTM:              {rmse_results['avg_w_lstm']:.4f}")

    # ----- Gate Analysis -----
    print(f"\n{'=' * 80}")
    print("Analyzing learned gate (Hybrid model)")
    print(f"{'=' * 80}")

    hybrid_model = models_to_train["Hybrid_LNN_LSTM"]["model"].to(device)
    hybrid_model.load_state_dict(
        torch.load(output_path / f"best_model_Hybrid_LNN_LSTM_{timestamp}.pt", map_location=device)
    )
    gate_stats = _analyze_gate(hybrid_model, val_loader, device, output_path, timestamp)
    for ch, st in gate_stats.items():
        print(f"  {ch}: mean={st['mean']:.4f}, std={st['std']:.4f}, median={st['median']:.4f}")

    # ----- Save results -----
    df_summary = pd.DataFrame(summary_rows)
    df_history = pd.DataFrame(history_rows)

    summary_csv = output_path / f"summary_{timestamp}.csv"
    history_csv = output_path / f"history_{timestamp}.csv"
    df_summary.to_csv(summary_csv, index=False)
    df_history.to_csv(history_csv, index=False)

    # Excel workbook
    excel_path = output_path / f"hybrid_experiment_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_history.to_excel(writer, sheet_name="History", index=False)

        # Config
        config_df = pd.DataFrame([config])
        config_df.to_excel(writer, sheet_name="Config", index=False)

        # Gate stats
        gate_df = pd.DataFrame(gate_stats).T
        gate_df.index.name = "channel"
        gate_df.to_excel(writer, sheet_name="Gate_Stats")

        # RMSE fusion weights
        rmse_df = pd.DataFrame([rmse_results])
        rmse_df.to_excel(writer, sheet_name="RMSE_Fusion", index=False)

    # ----- Plots -----
    _plot_comparison(
        df_summary[df_summary["best_val_mse_masked"] != "-"].copy(),
        output_path,
        timestamp,
    )
    _plot_all_histories(all_histories, output_path, timestamp)

    # ----- Print final summary -----
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print()
    print(f"[Saved] {summary_csv}")
    print(f"[Saved] {history_csv}")
    print(f"[Saved] {excel_path}")
    print("=" * 80)

    return df_summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid LNN-LSTM imputation experiment"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lnn_hidden", type=int, default=64)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--missing_mode", type=str, default="random",
                        choices=["random", "block", "channel"])
    parser.add_argument("--drift_scale", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="results/hybrid_lnn_lstm")
    args = parser.parse_args()

    run_hybrid_experiment(
        epochs=args.epochs,
        lnn_hidden=args.lnn_hidden,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        seq_len=args.seq_len,
        mask_rate=args.mask_rate,
        missing_mode=args.missing_mode,
        drift_scale=args.drift_scale,
        output_dir=args.output_dir,
    )