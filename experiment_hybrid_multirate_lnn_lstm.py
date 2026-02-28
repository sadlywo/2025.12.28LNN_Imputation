"""
Experiment: Multi-Rate Hybrid LNN + LSTM Imputation.

Core idea (our proposed method):
- LSTM branch receives **downsampled** input (every K points),
  capturing long-term motion patterns (turns, periodic gait, etc.)
  at a coarser temporal resolution.
- LNN (CfC) branch receives **full-rate** original input,
  performing smooth continuous-time ODE integration for high-frequency
  kinematic imputation.
- At inference, the LSTM output is **upsampled** back to the original
  time resolution via linear interpolation, then fused with the LNN
  output using RMSE-based adaptive weighting on observed positions.

Comparison groups:
1. LNN_only           – full-rate CfC
2. LSTM_only          – full-rate BiLSTM (baseline, no downsampling)
3. Hybrid_Normal      – standard learned-gate hybrid (same input to both)
4. MultiRate_Fusion   – **proposed** (downsampled LSTM + full-rate LNN + RMSE fusion)

Outputs:
- CSV summary & history
- Per-model training curves
- Combined comparison bar chart
- Fusion weight analysis
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

from dataset import CfCIMUDataset
from models import ReconstructionOnlyLoss
from models_hybrid import (
    ShortTermLNN,
    LongTermLSTM,
    HybridLNNLSTM,
    count_parameters,
    rmse_based_reweight,
)
from visualization import plot_training_curves


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Rate Hybrid Model  (proposed)
# ═══════════════════════════════════════════════════════════════════════════

class MultiRateHybridLNNLSTM(nn.Module):
    """
    Multi-rate hybrid model.

    - LNN receives the original full-rate sequence  (B, T, D).
    - LSTM receives a downsampled sequence           (B, T//K, D),
      capturing long-range dependencies more efficiently.
    - LSTM output is linearly interpolated back to T  steps and fused
      with the LNN output via a **learned gate** during training.
    - At evaluation time an RMSE-based post-hoc fusion can replace the gate.

    Parameters
    ----------
    downsample_factor : int
        Keep every K-th frame for the LSTM branch (default 5).
    """

    def __init__(
        self,
        input_dim: int = 13,
        lnn_hidden: int = 64,
        lstm_hidden: int = 128,
        output_dim: int = 6,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        downsample_factor: int = 5,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.downsample_factor = downsample_factor

        # ── Sub-models ──────────────────────────────────────────────────
        self.lnn = ShortTermLNN(
            input_dim=input_dim,
            hidden_units=lnn_hidden,
            output_dim=output_dim,
        )
        self.lstm = LongTermLSTM(
            input_dim=input_dim,
            hidden_dim=lstm_hidden,
            output_dim=output_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )

        # ── Learned gate ────────────────────────────────────────────────
        # Input: lnn_pred(6) + lstm_pred(6) + lnn_unc(6) + lstm_unc(6) + x(13) = 37
        gate_input_dim = output_dim * 4 + input_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid(),
        )

        self.combined_uncertainty = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Softplus(),
        )

    # -----------------------------------------------------------------
    @staticmethod
    def _downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
        """Take every `factor`-th timestep: (B, T, D) → (B, T//K, D)."""
        return x[:, ::factor, :]

    @staticmethod
    def _upsample_linear(x_ds: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Linearly interpolate (B, T_ds, D) → (B, target_len, D).
        Uses torch.nn.functional.interpolate on the channel-last layout.
        """
        # interpolate expects (B, C, L)
        x_t = x_ds.permute(0, 2, 1)                           # (B, D, T_ds)
        x_up = F.interpolate(x_t, size=target_len, mode="linear", align_corners=True)
        return x_up.permute(0, 2, 1)                           # (B, target_len, D)

    # -----------------------------------------------------------------
    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, 13) full-rate input
        Returns:
            pred: (B, T, 6)
            uncertainty: (B, T, 6)
        """
        B, T, D = x.shape
        K = self.downsample_factor

        # ── LNN branch: full-rate ───────────────────────────────────────
        lnn_pred, lnn_unc = self.lnn(x)           # (B, T, 6)

        # ── LSTM branch: downsampled ────────────────────────────────────
        x_ds = self._downsample(x, K)             # (B, T//K, D)
        lstm_pred_ds, lstm_unc_ds = self.lstm(x_ds)  # (B, T//K, 6)

        # Upsample back to T
        lstm_pred = self._upsample_linear(lstm_pred_ds, T)  # (B, T, 6)
        lstm_unc = self._upsample_linear(lstm_unc_ds, T)

        # ── Learned gate fusion ─────────────────────────────────────────
        gate_in = torch.cat([lnn_pred, lstm_pred, lnn_unc, lstm_unc, x], dim=-1)
        gate = self.gate_net(gate_in)              # (B, T, 6)  ∈ [0,1]

        pred = gate * lnn_pred + (1.0 - gate) * lstm_pred
        unc = self.combined_uncertainty(torch.cat([lnn_unc, lstm_unc], dim=-1))

        return pred, unc

    def forward_with_components(
        self, x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, T, D = x.shape
        K = self.downsample_factor

        lnn_pred, lnn_unc = self.lnn(x)
        x_ds = self._downsample(x, K)
        lstm_pred_ds, lstm_unc_ds = self.lstm(x_ds)
        lstm_pred = self._upsample_linear(lstm_pred_ds, T)
        lstm_unc = self._upsample_linear(lstm_unc_ds, T)

        gate_in = torch.cat([lnn_pred, lstm_pred, lnn_unc, lstm_unc, x], dim=-1)
        gate = self.gate_net(gate_in)

        pred = gate * lnn_pred + (1.0 - gate) * lstm_pred
        unc = self.combined_uncertainty(torch.cat([lnn_unc, lstm_unc], dim=-1))

        return {
            "pred": pred,
            "uncertainty": unc,
            "lnn_pred": lnn_pred,
            "lstm_pred": lstm_pred,
            "lnn_uncertainty": lnn_unc,
            "lstm_uncertainty": lstm_unc,
            "gate": gate,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Training / Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    losses = []
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
        losses.append(loss.item())
    return {"total": float(np.mean(losses))}


def _evaluate(model, loader, criterion, device):
    model.eval()
    losses, mse_all_list, mse_masked_list = [], [], []
    with torch.no_grad():
        for inputs, targets, mask in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            dt = inputs[:, :, -1:]

            pred, uncertainty = model(inputs)
            loss, _ = criterion(pred, targets, mask, uncertainty, dt)

            losses.append(loss.item())
            mse_all_list.append(F.mse_loss(pred, targets).item())
            missing_err = ((pred - targets) ** 2 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
            mse_masked_list.append(missing_err.item())

    return {
        "total": float(np.mean(losses)),
        "mse_all": float(np.mean(mse_all_list)),
        "mse_masked": float(np.mean(mse_masked_list)),
    }


def _train_model(
    model: nn.Module,
    model_name: str,
    train_loader,
    val_loader,
    device: torch.device,
    config: dict,
    output_path: Path,
    timestamp: str,
    criterion: nn.Module,
) -> Tuple[dict, dict, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        epochs=config["epochs"],
        steps_per_epoch=len(train_loader),
    )

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_mse_all": [], "val_mse_masked": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    best_path = output_path / f"best_model_{model_name}_{timestamp}.pt"

    start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        train_m = _train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_m = _evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_m["total"])
        history["val_loss"].append(val_m["total"])
        history["val_mse_all"].append(val_m["mse_all"])
        history["val_mse_masked"].append(val_m["mse_masked"])

        if val_m["total"] < best_val_loss:
            best_val_loss = val_m["total"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0 or epoch == config["epochs"]:
            print(
                f"  [{model_name}] Epoch {epoch:3d}/{config['epochs']}  "
                f"train={train_m['total']:.6f}  "
                f"val_all={val_m['mse_all']:.6f}  "
                f"val_masked={val_m['mse_masked']:.6f}"
            )

    train_time = time.time() - start

    model.load_state_dict(torch.load(best_path, map_location=device))
    best_eval = _evaluate(model, val_loader, criterion, device)

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


# ═══════════════════════════════════════════════════════════════════════════
# Post-hoc multi-rate RMSE fusion (inference only, no extra training)
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate_multirate_rmse_fusion(
    lnn_model: ShortTermLNN,
    lstm_model: LongTermLSTM,
    val_loader,
    device: torch.device,
    downsample_factor: int = 5,
) -> Dict[str, float]:
    """
    LNN runs on full-rate input; LSTM runs on downsampled input then
    upsampled back; predictions are combined via RMSE reweighting.
    """
    lnn_model.eval()
    lstm_model.eval()

    total_mse_all = 0.0
    total_mse_masked = 0.0
    total_samples = 0
    all_w_lnn, all_w_lstm = [], []

    with torch.no_grad():
        for inputs, targets, mask in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            B, T, D = inputs.shape

            # LNN: full-rate
            lnn_pred, _ = lnn_model(inputs)

            # LSTM: downsampled → run → upsample
            x_ds = inputs[:, ::downsample_factor, :]           # (B, T//K, D)
            lstm_pred_ds, _ = lstm_model(x_ds)                 # (B, T//K, 6)
            lstm_pred = MultiRateHybridLNNLSTM._upsample_linear(lstm_pred_ds, T)

            # RMSE fusion on observed positions
            fused, w_lnn, w_lstm = rmse_based_reweight(lnn_pred, lstm_pred, targets, mask)

            bs = inputs.shape[0]
            total_mse_all += F.mse_loss(fused, targets).item() * bs
            missing_err = ((fused - targets) ** 2 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
            total_mse_masked += missing_err.item() * bs
            total_samples += bs
            all_w_lnn.append(w_lnn)
            all_w_lstm.append(w_lstm)

    return {
        "mse_all": total_mse_all / max(total_samples, 1),
        "mse_masked": total_mse_masked / max(total_samples, 1),
        "avg_w_lnn": float(np.mean(all_w_lnn)),
        "avg_w_lstm": float(np.mean(all_w_lstm)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def _plot_comparison(summary_df: pd.DataFrame, output_path: Path, timestamp: str):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    models = summary_df["model"].tolist()
    x = np.arange(len(models))
    colors = plt.cm.Set2(np.linspace(0, 1, max(3, len(models))))

    for ax_idx, (col, label) in enumerate([
        ("best_val_mse_masked", "MSE (masked)"),
        ("best_val_mse_all", "MSE (all)"),
        ("num_params", "Parameters"),
    ]):
        ax = axes[ax_idx]
        vals = summary_df[col].tolist()
        bars = ax.bar(x, vals, color=colors[:len(models)])
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis="y")
        fmt = ".6f" if ax_idx < 2 else ","
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:{fmt}}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("Multi-Rate Hybrid vs Baselines", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / f"comparison_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_all_histories(all_histories: Dict[str, dict], output_path: Path, timestamp: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name, h in all_histories.items():
        ep = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], label=name, linewidth=1.5)
        axes[1].plot(ep, h["val_mse_all"], label=name, linewidth=1.5)
        axes[2].plot(ep, h["val_mse_masked"], label=name, linewidth=1.5)

    for ax, title in zip(axes, ["Training Loss", "Val MSE (All)", "Val MSE (Masked)"]):
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training History Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / f"all_training_curves_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_fusion_weights(
    all_weights: Dict[str, Dict[str, float]],
    output_path: Path,
    timestamp: str,
):
    """Bar chart comparing fusion weights across methods."""
    methods = list(all_weights.keys())
    w_lnn = [all_weights[m]["avg_w_lnn"] for m in methods]
    w_lstm = [all_weights[m]["avg_w_lstm"] for m in methods]
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, w_lnn, width, label="LNN weight", color="#2196F3")
    ax.bar(x + width / 2, w_lstm, width, label="LSTM weight", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Weight")
    ax.set_title("RMSE Fusion Weights: LNN vs LSTM")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for i, (l, s) in enumerate(zip(w_lnn, w_lstm)):
        ax.text(i - width / 2, l + 0.01, f"{l:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, s + 0.01, f"{s:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path / f"fusion_weights_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_multirate_experiment(
    epochs: int = 30,
    lnn_hidden: int = 64,
    lstm_hidden: int = 128,
    lstm_layers: int = 2,
    seq_len: int = 50,
    mask_rate: float = 0.3,
    missing_mode: str = "random",
    drift_scale: float = 0.01,
    downsample_factor: int = 5,
    output_dir: str = "results/hybrid_multirate",
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
        "downsample_factor": int(downsample_factor),
    }

    _seed_all(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    K = config["downsample_factor"]
    lstm_seq_len = config["seq_len"] // K  # effective LSTM seq len after downsampling

    print("=" * 80)
    print("MULTI-RATE HYBRID LNN-LSTM EXPERIMENT")
    print("=" * 80)
    print(f"Device:            {device}")
    print(f"Seq len:           {config['seq_len']}  (LSTM sees {lstm_seq_len} after ↓{K})")
    print(f"Downsample factor: {K}")
    print(f"Missing:           {config['missing_mode']} @ {config['mask_rate'] * 100:.0f}%")
    print(f"Epochs:            {config['epochs']}")
    print(f"LNN hidden:        {config['lnn_hidden']}")
    print(f"LSTM hidden:       {config['lstm_hidden']} x {config['lstm_layers']} layers (BiLSTM)")
    print(f"Drift scale:       {config['drift_scale']}")
    print(f"Output:            {output_path}")
    print("=" * 80)

    # ── Data ────────────────────────────────────────────────────────────
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
        pin_memory=device.type == "cuda",
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )

    print(f"\nTrain: {len(train_ds)} samples ({len(train_loader)} batches)")
    print(f"Val:   {len(val_ds)} samples ({len(val_loader)} batches)")

    # ── Models to train ─────────────────────────────────────────────────
    criterion = ReconstructionOnlyLoss(w_recon=1.0)

    models_to_train = {
        # 1. LNN only (full-rate CfC)
        "LNN_only": ShortTermLNN(
            input_dim=13,
            hidden_units=config["lnn_hidden"],
            output_dim=6,
        ),
        # 2. LSTM only (full-rate BiLSTM, baseline)
        "LSTM_only": LongTermLSTM(
            input_dim=13,
            hidden_dim=config["lstm_hidden"],
            output_dim=6,
            num_layers=config["lstm_layers"],
        ),
        # 3. Normal Hybrid (learned gate, same input to both branches)
        "Hybrid_Normal": HybridLNNLSTM(
            input_dim=13,
            lnn_hidden=config["lnn_hidden"],
            lstm_hidden=config["lstm_hidden"],
            output_dim=6,
            lstm_layers=config["lstm_layers"],
            fusion_mode="learned",
        ),
        # 4. Multi-Rate Hybrid (proposed: downsampled LSTM + full-rate LNN)
        f"MultiRate_Hybrid_K{K}": MultiRateHybridLNNLSTM(
            input_dim=13,
            lnn_hidden=config["lnn_hidden"],
            lstm_hidden=config["lstm_hidden"],
            output_dim=6,
            lstm_layers=config["lstm_layers"],
            downsample_factor=K,
        ),
    }

    summary_rows: List[dict] = []
    history_rows: List[dict] = []
    all_histories: Dict[str, dict] = {}

    for model_name, model in models_to_train.items():
        model = model.to(device)
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
        model.eval()
        t0 = time.time()
        n_samples = 0
        with torch.no_grad():
            for inp, _, _ in val_loader:
                inp = inp.to(device)
                _sync(device)
                _ = model(inp)
                _sync(device)
                n_samples += inp.shape[0]
        infer_time = time.time() - t0

        for ei in range(len(history["train_loss"])):
            history_rows.append({
                "model": model_name, "epoch": ei + 1,
                "train_loss": history["train_loss"][ei],
                "val_loss": history["val_loss"][ei],
                "val_mse_all": history["val_mse_all"][ei],
                "val_mse_masked": history["val_mse_masked"][ei],
            })

        row = {
            "model": model_name,
            "num_params": num_params,
            "param_size_mb": round(size_mb, 4),
            "best_epoch": best_metrics["best_epoch"],
            "best_val_loss": best_metrics["best_val_loss"],
            "best_val_mse_all": best_metrics["best_val_mse_all"],
            "best_val_mse_masked": best_metrics["best_val_mse_masked"],
            "train_time_sec": round(train_time, 2),
            "inference_ms_per_sample": round(1000 * infer_time / max(n_samples, 1), 4),
        }
        summary_rows.append(row)

        print(f"  Best epoch:           {best_metrics['best_epoch']}")
        print(f"  Best val MSE(masked): {best_metrics['best_val_mse_masked']:.6f}")
        print(f"  Best val MSE(all):    {best_metrics['best_val_mse_all']:.6f}")
        print(f"  Train time:           {train_time:.1f}s")
        print(f"  Inference:            {1000 * infer_time / max(n_samples, 1):.4f} ms/sample")

    # ── Post-hoc RMSE fusion variants ───────────────────────────────────
    print(f"\n{'=' * 80}")
    print("Post-hoc RMSE Fusion Evaluation")
    print(f"{'=' * 80}")

    # Load best weights for standalone models
    lnn_model = models_to_train["LNN_only"].to(device)
    lstm_model = models_to_train["LSTM_only"].to(device)
    lnn_model.load_state_dict(
        torch.load(output_path / f"best_model_LNN_only_{timestamp}.pt", map_location=device)
    )
    lstm_model.load_state_dict(
        torch.load(output_path / f"best_model_LSTM_only_{timestamp}.pt", map_location=device)
    )

    fusion_weights: Dict[str, Dict[str, float]] = {}

    # ─ Normal RMSE fusion (both full-rate) ──────────────────────────────
    normal_fusion = _evaluate_multirate_rmse_fusion(
        lnn_model, lstm_model, val_loader, device, downsample_factor=1,  # no downsample
    )
    fusion_weights["RMSE_Normal"] = normal_fusion

    summary_rows.append({
        "model": "RMSE_Fusion_Normal",
        "num_params": count_parameters(lnn_model) + count_parameters(lstm_model),
        "param_size_mb": round(_state_dict_size_mb(lnn_model) + _state_dict_size_mb(lstm_model), 4),
        "best_epoch": "-", "best_val_loss": "-",
        "best_val_mse_all": normal_fusion["mse_all"],
        "best_val_mse_masked": normal_fusion["mse_masked"],
        "train_time_sec": "-", "inference_ms_per_sample": "-",
    })
    print(f"  [RMSE_Normal]   MSE_masked={normal_fusion['mse_masked']:.6f}  "
          f"MSE_all={normal_fusion['mse_all']:.6f}  "
          f"w_LNN={normal_fusion['avg_w_lnn']:.4f}  w_LSTM={normal_fusion['avg_w_lstm']:.4f}")

    # ─ Multi-rate RMSE fusion (proposed, LSTM downsampled) ──────────────
    multirate_fusion = _evaluate_multirate_rmse_fusion(
        lnn_model, lstm_model, val_loader, device, downsample_factor=K,
    )
    fusion_weights[f"RMSE_MultiRate_K{K}"] = multirate_fusion

    summary_rows.append({
        "model": f"RMSE_MultiRate_K{K}",
        "num_params": count_parameters(lnn_model) + count_parameters(lstm_model),
        "param_size_mb": round(_state_dict_size_mb(lnn_model) + _state_dict_size_mb(lstm_model), 4),
        "best_epoch": "-", "best_val_loss": "-",
        "best_val_mse_all": multirate_fusion["mse_all"],
        "best_val_mse_masked": multirate_fusion["mse_masked"],
        "train_time_sec": "-", "inference_ms_per_sample": "-",
    })
    print(f"  [RMSE_MultiRate_K{K}]  MSE_masked={multirate_fusion['mse_masked']:.6f}  "
          f"MSE_all={multirate_fusion['mse_all']:.6f}  "
          f"w_LNN={multirate_fusion['avg_w_lnn']:.4f}  w_LSTM={multirate_fusion['avg_w_lstm']:.4f}")

    # ── Save results ────────────────────────────────────────────────────
    df_summary = pd.DataFrame(summary_rows)
    df_history = pd.DataFrame(history_rows)

    summary_csv = output_path / f"summary_{timestamp}.csv"
    history_csv = output_path / f"history_{timestamp}.csv"
    df_summary.to_csv(summary_csv, index=False)
    df_history.to_csv(history_csv, index=False)

    # Excel
    excel_path = output_path / f"multirate_experiment_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_history.to_excel(writer, sheet_name="History", index=False)
        pd.DataFrame([config]).to_excel(writer, sheet_name="Config", index=False)
        pd.DataFrame(fusion_weights).T.to_excel(writer, sheet_name="Fusion_Weights")

    # ── Plots ───────────────────────────────────────────────────────────
    # Filter numeric rows for bar chart
    numeric_df = df_summary[df_summary["best_val_mse_masked"].apply(
        lambda v: isinstance(v, (int, float))
    )].copy()
    _plot_comparison(numeric_df, output_path, timestamp)
    _plot_all_histories(all_histories, output_path, timestamp)
    _plot_fusion_weights(fusion_weights, output_path, timestamp)

    # ── Final summary ───────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Rate Hybrid LNN-LSTM imputation experiment"
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
    parser.add_argument("--downsample_factor", type=int, default=5,
                        help="LSTM branch downsampling factor K (default=5)")
    parser.add_argument("--output_dir", type=str, default="results/hybrid_multirate")
    args = parser.parse_args()

    run_multirate_experiment(
        epochs=args.epochs,
        lnn_hidden=args.lnn_hidden,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        seq_len=args.seq_len,
        mask_rate=args.mask_rate,
        missing_mode=args.missing_mode,
        drift_scale=args.drift_scale,
        downsample_factor=args.downsample_factor,
        output_dir=args.output_dir,
    )
