"""
Experiment: Multi-Rate Hybrid LNN + LSTM Imputation.

Core idea (our proposed method):
- LSTM branch receives **downsampled** input (every K points),
    capturing long-term motion patterns (turns, periodic gait, etc.)
    at a coarser temporal resolution.
- LNN (CfC) branch receives **full-rate** input without mask channels,
    focusing on short-term kinematic continuity.
- The LSTM output is **upsampled** back to the original time resolution
    via linear interpolation, then fused with the LNN output using a
    learned gate (and RMSE fusion for the baseline row).

Comparison groups:
1. LNN_only           – full-rate CfC (maskless, 25-dim with window stats)
2. LSTM_only          – full-rate BiLSTM
3. Hybrid_Normal      – standard learned-gate hybrid (maskless LNN + masked LSTM)

Outputs:
- CSV summary & history
- Per-model training curves
- Combined comparison bar chart
- Fusion weight analysis
- Excel workbook with all results
"""
from __future__ import annotations

import argparse
import inspect
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
from visualization import plot_training_curves


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _state_dict_size_mb(model: nn.Module) -> float:
    total = 0
    for v in model.state_dict().values():
        total += v.nelement() * v.element_size()
    return total / (1024 * 1024)


def _build_recon_loss(observed_weight: float) -> ReconstructionOnlyLoss:
    sig = inspect.signature(ReconstructionOnlyLoss.__init__)
    if "w_observed" in sig.parameters:
        return ReconstructionOnlyLoss(w_recon=1.0, w_observed=observed_weight)
    return ReconstructionOnlyLoss(w_recon=1.0)


def _unpack_batch(batch):
    inputs = batch[0]
    targets = batch[1]
    mask = batch[2]
    stats = batch[3] if len(batch) > 3 else None
    vicon = batch[4] if len(batch) > 4 else None
    return inputs, targets, mask, stats, vicon


def _strip_mask_inputs(x: torch.Tensor, feature_dim: int) -> torch.Tensor:
    imu = x[:, :, :feature_dim]
    dt = x[:, :, feature_dim * 2 : feature_dim * 2 + 1]
    window_feats = x[:, :, feature_dim * 2 + 1 :]
    return torch.cat([imu, dt, window_feats], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Rate Hybrid Model  (proposed)
# ═══════════════════════════════════════════════════════════════════════════

class MultiRateHybridLNNLSTM(nn.Module):
    """
    Multi-rate hybrid model.

    - LNN receives the original full-rate sequence  (B, T, D).
        - LSTM receives a downsampled sequence           (B, T//K, D),
            then is **upsampled back to length T** before entering LSTM,
            keeping the input window length the same while reducing
            high-frequency detail (so it can focus on long-term patterns).
        - LSTM output has length T and is fused with the LNN output via a
            **learned gate** during training.
    - At evaluation time an RMSE-based post-hoc fusion can replace the gate.

    Parameters
    ----------
    downsample_factor : int
        Keep every K-th frame for the LSTM branch (default 5).
    """

    def __init__(
        self,
        input_dim: int = 13,
        feature_dim: int = 6,
        window_feat_dim: int = 0,
        lnn_hidden: int = 64,
        lstm_hidden: int = 64,
        output_dim: int = 6,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        downsample_factor: int = 5,
        fusion_mode: str = "learned",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.downsample_factor = downsample_factor
        self.fusion_mode = fusion_mode
        self.feature_dim = feature_dim
        self.window_feat_dim = window_feat_dim
        self.lnn_input_dim = input_dim - feature_dim

        # ── Sub-models ──────────────────────────────────────────────────
        self.lnn = ShortTermLNN(
            input_dim=self.lnn_input_dim,
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
            nn.Linear(gate_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
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
    def _strip_mask(self, x: torch.Tensor) -> torch.Tensor:
        imu = x[:, :, : self.feature_dim]
        dt = x[:, :, self.feature_dim * 2 : self.feature_dim * 2 + 1]
        window_feats = x[:, :, self.feature_dim * 2 + 1 :]
        return torch.cat([imu, dt, window_feats], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
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

        # ── LNN branch: full-rate (mask removed) ────────────────────────
        lnn_input = self._strip_mask(x)
        lnn_pred, lnn_unc = self.lnn(lnn_input)   # (B, T, 6)

        # ── LSTM branch: downsample input, output low-rate, then upsample ──
        x_ds = self._downsample(x, K)             # (B, T//K, D)
        lstm_pred_ds, lstm_unc_ds = self.lstm(x_ds)   # (B, T//K, 6)
        lstm_pred = self._upsample_linear(lstm_pred_ds, T)
        lstm_unc = self._upsample_linear(lstm_unc_ds, T)

        # ── Learned gate fusion ─────────────────────────────────────────
        gate_in = torch.cat([lnn_pred, lstm_pred, lnn_unc, lstm_unc, x], dim=-1)
        gate = self.gate_net(gate_in)              # (B, T, 6)  ∈ [0,1]
        gate = gate.mean(dim=1, keepdim=True)
        gate = gate.expand(-1, lnn_pred.shape[1], -1)

        pred = gate * lnn_pred + (1.0 - gate) * lstm_pred
        unc = self.combined_uncertainty(torch.cat([lnn_unc, lstm_unc], dim=-1))

        return pred, unc

    def forward_with_components(
        self, x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, T, D = x.shape
        K = self.downsample_factor

        lnn_input = self._strip_mask(x)
        lnn_pred, lnn_unc = self.lnn(lnn_input)
        x_ds = self._downsample(x, K)
        lstm_pred_ds, lstm_unc_ds = self.lstm(x_ds)
        lstm_pred = self._upsample_linear(lstm_pred_ds, T)
        lstm_unc = self._upsample_linear(lstm_unc_ds, T)

        gate_in = torch.cat([lnn_pred, lstm_pred, lnn_unc, lstm_unc, x], dim=-1)
        gate = self.gate_net(gate_in)
        gate = gate.mean(dim=1, keepdim=True)
        gate = gate.expand(-1, lnn_pred.shape[1], -1)

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


class MasklessLNN(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int, hidden_units: int = 64, output_dim: int = 6):
        super().__init__()
        self.feature_dim = feature_dim
        self.lnn = ShortTermLNN(
            input_dim=input_dim,
            hidden_units=hidden_units,
            output_dim=output_dim,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lnn_input = _strip_mask_inputs(x, self.feature_dim)
        return self.lnn(lnn_input)


class MasklessHybridLNNLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        lnn_hidden: int = 64,
        lstm_hidden: int = 64,
        output_dim: int = 6,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        fusion_mode: str = "learned",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.fusion_mode = fusion_mode
        self.lnn = ShortTermLNN(
            input_dim=input_dim - feature_dim,
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
        gate_input_dim = output_dim * 4 + input_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Sigmoid(),
        )
        self.combined_uncertainty = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor, target=None, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        lnn_input = _strip_mask_inputs(x, self.feature_dim)
        lnn_pred, lnn_unc = self.lnn(lnn_input)
        lstm_pred, lstm_unc = self.lstm(x)

        if self.fusion_mode == "rmse" and target is not None and mask is not None:
            pred, w_lnn, w_lstm = rmse_based_reweight(lnn_pred, lstm_pred, target, mask)
            unc = w_lnn * lnn_unc + w_lstm * lstm_unc
            return pred, unc

        gate_input = torch.cat([lnn_pred, lstm_pred, lnn_unc, lstm_unc, x], dim=-1)
        gate = self.gate_net(gate_input)
        gate = gate.mean(dim=1, keepdim=True)
        gate = gate.expand(-1, lnn_pred.shape[1], -1)
        pred = gate * lnn_pred + (1.0 - gate) * lstm_pred
        unc = self.combined_uncertainty(torch.cat([lnn_unc, lstm_unc], dim=-1))
        return pred, unc

    def forward_with_components(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        lnn_input = _strip_mask_inputs(x, self.feature_dim)
        lnn_pred, lnn_unc = self.lnn(lnn_input)
        lstm_pred, lstm_unc = self.lstm(x)

        gate_input = torch.cat([lnn_pred, lstm_pred, lnn_unc, lstm_unc, x], dim=-1)
        gate = self.gate_net(gate_input)
        gate = gate.mean(dim=1, keepdim=True)
        gate = gate.expand(-1, lnn_pred.shape[1], -1)
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

def _train_one_epoch(model, loader, criterion, optimizer, scheduler, device, gate_reg_weight: float = 0.0):
    model.train()
    losses = []
    for inputs, targets, mask in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        dt = inputs[:, :, -1:]

        optimizer.zero_grad()
        gate_reg = torch.tensor(0.0, device=device)
        if gate_reg_weight > 0 and hasattr(model, "forward_with_components"):
            components = model.forward_with_components(inputs)
            pred = components["pred"]
            uncertainty = components["uncertainty"]

            missing = (1 - mask)
            lnn_err = ((components["lnn_pred"] - targets) ** 2 * missing).sum(dim=(1, 2))
            lstm_err = ((components["lstm_pred"] - targets) ** 2 * missing).sum(dim=(1, 2))
            denom = (missing.sum(dim=(1, 2)) + 1e-8)
            lnn_mse = lnn_err / denom
            lstm_mse = lstm_err / denom
            target_w_lnn = (lstm_mse / (lnn_mse + lstm_mse + 1e-8)).view(-1, 1, 1)
            target_w_lnn = target_w_lnn.expand_as(components["gate"])
            gate_reg = F.mse_loss(components["gate"], target_w_lnn)
        else:
            pred, uncertainty = model(inputs)

        loss, _ = criterion(pred, targets, mask, uncertainty, dt)
        loss = loss + gate_reg_weight * gate_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
    return {"total": float(np.mean(losses))}


def _evaluate(model, loader, criterion, device):
    model.eval()
    losses, mse_all_list, mse_observed_list, mse_missing_list = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs, targets, mask, _, _ = _unpack_batch(batch)
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            dt = inputs[:, :, -1:]

            pred, uncertainty = model(inputs)
            loss, _ = criterion(pred, targets, mask, uncertainty, dt)

            losses.append(loss.item())
            mse_all_list.append(F.mse_loss(pred, targets).item())
            mse_observed = ((pred - targets) ** 2 * mask).sum() / (mask.sum() + 1e-8)
            mse_missing = ((pred - targets) ** 2 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
            mse_observed_list.append(mse_observed.item())
            mse_missing_list.append(mse_missing.item())

    return {
        "total": float(np.mean(losses)),
        "mse_all": float(np.mean(mse_all_list)),
        "mse_observed": float(np.mean(mse_observed_list)),
        "mse_missing": float(np.mean(mse_missing_list)),
    }


def _evaluate_trajectory_metrics(model, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_ate = 0.0
    total_rte = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets, _, stats, vicon = _unpack_batch(batch)
            if stats is None or vicon is None:
                continue
            inputs = inputs.to(device)
            targets = targets.to(device)
            stats = stats.to(device)
            vicon = vicon.to(device)
            dt = inputs[:, :, -1]

            pred, _ = model(inputs)
            ate_result = compute_ate(pred, vicon, dt, stats=stats)
            rte_result = compute_relative_trajectory_error(pred, vicon, dt, stats=stats)

            batch_size = inputs.shape[0]
            total_ate += ate_result["ate"] * batch_size
            total_rte += rte_result["rte"] * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return {"ate": float("nan"), "rte": float("nan")}
    return {
        "ate": total_ate / total_samples,
        "rte": total_rte / total_samples,
    }


def _evaluate_trajectory_metrics_rmse_fusion(
    lnn_model: nn.Module,
    lstm_model: nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    lnn_model.eval()
    lstm_model.eval()
    total_ate = 0.0
    total_rte = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets, mask, stats, vicon = _unpack_batch(batch)
            if stats is None or vicon is None:
                continue
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            stats = stats.to(device)
            vicon = vicon.to(device)
            dt = inputs[:, :, -1]

            lnn_pred, _ = lnn_model(inputs)
            lstm_pred, _ = lstm_model(inputs)
            fused, _, _ = rmse_based_reweight(lnn_pred, lstm_pred, targets, mask)

            ate_result = compute_ate(fused, vicon, dt, stats=stats)
            rte_result = compute_relative_trajectory_error(fused, vicon, dt, stats=stats)

            batch_size = inputs.shape[0]
            total_ate += ate_result["ate"] * batch_size
            total_rte += rte_result["rte"] * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return {"ate": float("nan"), "rte": float("nan")}
    return {
        "ate": total_ate / total_samples,
        "rte": total_rte / total_samples,
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
        "train_loss": [],
        "val_loss": [],
        "val_mse_all": [],
        "val_mse_observed": [],
        "val_mse_missing": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    best_path = output_path / f"best_model_{model_name}_{timestamp}.pt"

    start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        train_m = _train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            gate_reg_weight=config.get("gate_reg_weight", 0.0),
        )
        val_m = _evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_m["total"])
        history["val_loss"].append(val_m["total"])
        history["val_mse_all"].append(val_m["mse_all"])
        history["val_mse_observed"].append(val_m["mse_observed"])
        history["val_mse_missing"].append(val_m["mse_missing"])

        if val_m["total"] < best_val_loss:
            best_val_loss = val_m["total"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0 or epoch == config["epochs"]:
            print(
                f"  [{model_name}] Epoch {epoch:3d}/{config['epochs']}  "
                f"train={train_m['total']:.6f}  "
                f"val_all={val_m['mse_all']:.6f}  "
                f"val_missing={val_m['mse_missing']:.6f}"
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
        "best_val_mse_observed": float(best_eval["mse_observed"]),
        "best_val_mse_missing": float(best_eval["mse_missing"]),
    }, train_time


def _evaluate_rmse_fusion(
    lnn_model: nn.Module,
    lstm_model: nn.Module,
    val_loader,
    device: torch.device,
) -> Dict[str, float]:
    lnn_model.eval()
    lstm_model.eval()

    total_mse_all = 0.0
    total_mse_missing = 0.0
    total_mse_observed = 0.0
    total_samples = 0
    all_w_lnn = []
    all_w_lstm = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets, mask, _, _ = _unpack_batch(batch)
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            lnn_pred, _ = lnn_model(inputs)
            lstm_pred, _ = lstm_model(inputs)

            fused, w_lnn, w_lstm = rmse_based_reweight(lnn_pred, lstm_pred, targets, mask)

            mse_all = F.mse_loss(fused, targets).item()
            missing_err = ((fused - targets) ** 2 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
            observed_err = ((fused - targets) ** 2 * mask).sum() / (mask.sum() + 1e-8)

            batch_size = inputs.shape[0]
            total_mse_all += mse_all * batch_size
            total_mse_missing += missing_err.item() * batch_size
            total_mse_observed += observed_err.item() * batch_size
            total_samples += batch_size
            all_w_lnn.append(w_lnn)
            all_w_lstm.append(w_lstm)

    return {
        "mse_all": total_mse_all / max(total_samples, 1),
        "mse_observed": total_mse_observed / max(total_samples, 1),
        "mse_missing": total_mse_missing / max(total_samples, 1),
        "avg_w_lnn": float(np.mean(all_w_lnn)),
        "avg_w_lstm": float(np.mean(all_w_lstm)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Post-hoc multi-rate RMSE fusion (inference only, no extra training)
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def _plot_comparison(summary_df: pd.DataFrame, output_path: Path, timestamp: str):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    models = summary_df["model"].tolist()
    x = np.arange(len(models))
    colors = plt.cm.Set2(np.linspace(0, 1, max(3, len(models))))

    for ax_idx, (col, label) in enumerate([
        ("best_val_mse_missing", "MSE (missing)"),
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
        axes[2].plot(ep, h["val_mse_missing"], label=name, linewidth=1.5)

    for ax, title in zip(axes, ["Training Loss", "Val MSE (All)", "Val MSE (Missing)"]):
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training History Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / f"all_training_curves_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_multirate_experiment(
    epochs: int = 30,
    lnn_hidden: int = 64,
    lstm_hidden: int = 64,
    lstm_layers: int = 2,
    seq_len: int = 30,
    mask_rate: float = 0.3,
    missing_mode: str = "random",
    drift_scale: float = 0.00,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    gate_reg_weight: float = 0.1,
    observed_weight: float = 0.1,
    output_dir: str = "results/hybrid_multirate",
):
    config = {
        "root_dir": "Oxford Dataset",
        "seq_len": int(seq_len),
        "mask_rate": float(mask_rate),
        "missing_mode": str(missing_mode),
        "batch_size": 32,
        "epochs": int(epochs),
        "lr": 1e-3,
        "lnn_hidden": int(lnn_hidden),
        "lstm_hidden": int(lstm_hidden),
        "lstm_layers": int(lstm_layers),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 4,
        "seed": 2026,
        "output_dir": str(output_dir),
        "drift_scale": float(drift_scale),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "gate_reg_weight": float(gate_reg_weight),
        "observed_weight": float(observed_weight),
        "include_window_features": True,
    }

    _seed_all(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("MULTI-RATE HYBRID LNN-LSTM EXPERIMENT")
    print("=" * 80)
    print(f"Device:            {device}")
    print(f"Seq len:           {config['seq_len']}")
    print(f"Missing:           {config['missing_mode']} @ {config['mask_rate'] * 100:.0f}%")
    print(f"Epochs:            {config['epochs']}")
    print(f"LNN hidden:        {config['lnn_hidden']}")
    print(f"LSTM hidden:       {config['lstm_hidden']} x {config['lstm_layers']} layers (BiLSTM)")
    print(f"Drift scale:       {config['drift_scale']}")
    print(f"Observed weight:   {config['observed_weight']}")
    print(f"Output:            {output_path}")
    print("=" * 80)

    # ── Data ────────────────────────────────────────────────────────────
    train_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=config["mask_rate"],
        missing_mode=config["missing_mode"],
        split="train",
        split_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        eval_mode=False,
        drift_scale=config["drift_scale"],
        include_window_features=config["include_window_features"],
    )
    val_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=config["mask_rate"],
        missing_mode=config["missing_mode"],
        split="val",
        split_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        eval_mode=True,
        drift_scale=0.0,
        include_window_features=config["include_window_features"],
    )
    test_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=config["mask_rate"],
        missing_mode=config["missing_mode"],
        split="test",
        split_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        eval_mode=True,
        drift_scale=0.0,
        return_stats=True,
        return_vicon=True,
        include_window_features=config["include_window_features"],
    )

    config["input_dim"] = int(train_ds.input_dim)
    config["feature_dim"] = int(train_ds.feature_dim)
    config["window_feat_dim"] = int(train_ds.window_feat_dim)
    config["lnn_input_dim"] = int(train_ds.input_dim - train_ds.feature_dim)

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
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )

    print(f"\nTrain: {len(train_ds)} samples ({len(train_loader)} batches)")
    print(f"Val:   {len(val_ds)} samples ({len(val_loader)} batches)")
    print(f"Test:  {len(test_ds)} samples ({len(test_loader)} batches)")

    # ── Models to train ─────────────────────────────────────────────────
    criterion = _build_recon_loss(config["observed_weight"])

    models_to_train = {
        # 1. LNN only (full-rate CfC, no mask channels)
        "LNN_only": MasklessLNN(
            input_dim=config["lnn_input_dim"],
            feature_dim=config["feature_dim"],
            hidden_units=config["lnn_hidden"],
            output_dim=6,
        ),
        # 2. LSTM only (full-rate BiLSTM, baseline)
        "LSTM_only": LongTermLSTM(
            input_dim=config["input_dim"],
            hidden_dim=config["lstm_hidden"],
            output_dim=6,
            num_layers=config["lstm_layers"],
        ),
        # 3. Normal Hybrid (dynamic gate, maskless LNN + masked LSTM)
        "Hybrid_Normal": MasklessHybridLNNLSTM(
            input_dim=config["input_dim"],
            feature_dim=config["feature_dim"],
            lnn_hidden=config["lnn_hidden"],
            lstm_hidden=config["lstm_hidden"],
            output_dim=6,
            lstm_layers=config["lstm_layers"],
            fusion_mode="learned",
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
        test_metrics = _evaluate(model, test_loader, criterion, device)
        traj_metrics = _evaluate_trajectory_metrics(model, test_loader, device)

        for ei in range(len(history["train_loss"])):
            history_rows.append({
                "model": model_name, "epoch": ei + 1,
                "train_loss": history["train_loss"][ei],
                "val_loss": history["val_loss"][ei],
                "val_mse_all": history["val_mse_all"][ei],
                "val_mse_observed": history["val_mse_observed"][ei],
                "val_mse_missing": history["val_mse_missing"][ei],
            })

        row = {
            "model": model_name,
            "num_params": num_params,
            "param_size_mb": round(size_mb, 4),
            "best_epoch": best_metrics["best_epoch"],
            "best_val_loss": best_metrics["best_val_loss"],
            "best_val_mse_all": best_metrics["best_val_mse_all"],
            "best_val_mse_observed": best_metrics["best_val_mse_observed"],
            "best_val_mse_missing": best_metrics["best_val_mse_missing"],
            "test_mse_all": float(test_metrics["mse_all"]),
            "test_mse_observed": float(test_metrics["mse_observed"]),
            "test_mse_missing": float(test_metrics["mse_missing"]),
            "test_ate": float(traj_metrics["ate"]),
            "test_rte": float(traj_metrics["rte"]),
            "train_time_sec": round(train_time, 2),
        }
        summary_rows.append(row)

        print(f"  Best epoch:           {best_metrics['best_epoch']}")
        print(f"  Best val MSE(missing): {best_metrics['best_val_mse_missing']:.6f}")
        print(f"  Best val MSE(all):     {best_metrics['best_val_mse_all']:.6f}")
        print(f"  Test MSE(missing):     {test_metrics['mse_missing']:.6f}")
        print(f"  Test MSE(all):         {test_metrics['mse_all']:.6f}")
        print(f"  Train time:           {train_time:.1f}s")


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

    # ── Plots ───────────────────────────────────────────────────────────
    # Filter numeric rows for bar chart
    numeric_df = df_summary[df_summary["best_val_mse_missing"].apply(
        lambda v: isinstance(v, (int, float))
    )].copy()
    _plot_comparison(numeric_df, output_path, timestamp)
    _plot_all_histories(all_histories, output_path, timestamp)

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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lnn_hidden", type=int, default=64)
    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--missing_mode", type=str, default="random",
                        choices=["random", "block", "channel"])
    parser.add_argument("--drift_scale", type=float, default=0.00)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--gate_reg_weight", type=float, default=0.1)
    parser.add_argument("--observed_weight", type=float, default=0.1,
                        help="Weight for observed reconstruction loss")
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
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        gate_reg_weight=args.gate_reg_weight,
        observed_weight=args.observed_weight,
        output_dir=args.output_dir,
    )
