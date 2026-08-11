"""
Experiment: Bidirectional LNN residual imputation vs BiLSTM vs Hybrid.

Requirements implemented:
1. Bidirectional LNN: use forward/backward CfC encoders to leverage past and future.
2. Compare Bidirectional_LNN, BiLSTM, and Hybrid_BiLNN_BiLSTM.
3. Add missing-boundary / position encoding inputs.
4. Use reconstruction error only.
5. Use residual imputation.
6. Evaluate with missing-point RMSE and trajectory errors (ATE / RTE from ground truth).
"""
from __future__ import annotations

import argparse
import json
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
from ncps.torch import CfC

from dataset import CfCIMUDataset, compute_ate, compute_relative_trajectory_error
from models_hybrid import LongTermLSTM, count_parameters
from visualization import plot_training_curves


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


def _find_latest_best_model(output_path: Path, model_name: str) -> Path | None:
    candidates = sorted(output_path.glob(f"best_model_{model_name}_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _unpack_batch(batch):
    inputs = batch[0]
    targets = batch[1]
    mask = batch[2]
    stats = batch[3] if len(batch) > 3 else None
    vicon = batch[4] if len(batch) > 4 else None
    return inputs, targets, mask, stats, vicon


class ReconstructionRMSEOnlyLoss(nn.Module):
    """Reconstruction-only loss on missing positions, reported as MSE for optimization."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        uncertainty: torch.Tensor | None = None,
        dt: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, dict]:
        missing = 1.0 - mask
        mse_missing = ((pred - target) ** 2 * missing).sum() / (missing.sum() + 1e-8)
        return mse_missing, {"recon": float(mse_missing.item())}


class ResidualInputAdapter(nn.Module):
    """Build residual-imputation inputs with boundary / position encodings."""

    def __init__(self, feature_dim: int, input_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.input_dim = input_dim
        self.base_start = 0
        self.mask_start = feature_dim
        self.dt_index = feature_dim * 2
        self.delta_start = self.dt_index + 1
        self.window_start = self.delta_start + 6

    def _dist_to_observed(self, observed_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute normalized distances to nearest observed point on left/right."""
        t = observed_1d.shape[0]
        device = observed_1d.device
        left = torch.full((t,), float(t), device=device)
        right = torch.full((t,), float(t), device=device)

        last = -1
        for i in range(t):
            if observed_1d[i] > 0.5:
                last = i
            left[i] = 0.0 if last == i else (i - last if last >= 0 else t)

        nxt = -1
        for i in range(t - 1, -1, -1):
            if observed_1d[i] > 0.5:
                nxt = i
            right[i] = 0.0 if nxt == i else (nxt - i if nxt >= 0 else t)

        scale = max(t - 1, 1)
        return left / scale, right / scale

    def build(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            residual_input: model input with residual baseline and position features
            baseline: baseline residual anchor of shape (B, T, feature_dim)
        """
        masked_imu = x[:, :, self.base_start:self.base_start + self.feature_dim]
        mask = x[:, :, self.mask_start:self.mask_start + self.feature_dim]
        dt = x[:, :, self.dt_index:self.dt_index + 1]
        delta = x[:, :, self.delta_start:self.delta_start + 6]
        window_feats = x[:, :, self.window_start:] if self.window_start < self.input_dim else x[:, :, 0:0]

        baseline = masked_imu.clone()
        obs_any = (mask.mean(dim=-1) > 0.5).float()
        left_dist_list = []
        right_dist_list = []
        rel_pos_list = []
        missing_indicator_list = []

        for b in range(x.shape[0]):
            obs_mask = obs_any[b]
            left_dist, right_dist = self._dist_to_observed(obs_mask)
            left_dist_list.append(left_dist)
            right_dist_list.append(right_dist)

            missing_indicator = 1.0 - obs_mask
            missing_indicator_list.append(missing_indicator)

            rel_pos = torch.linspace(0.0, 1.0, x.shape[1], device=x.device, dtype=x.dtype)
            rel_pos_list.append(rel_pos)

            for c in range(self.feature_dim):
                obs_idx = torch.nonzero(mask[b, :, c] > 0.5, as_tuple=False).flatten()
                if obs_idx.numel() == 0:
                    baseline[b, :, c] = 0.0
                    continue
                baseline[b, : obs_idx[0], c] = masked_imu[b, obs_idx[0], c]
                baseline[b, obs_idx[-1] + 1 :, c] = masked_imu[b, obs_idx[-1], c]
                for i in range(obs_idx.numel() - 1):
                    s = int(obs_idx[i].item())
                    e = int(obs_idx[i + 1].item())
                    baseline[b, s:e + 1, c] = torch.linspace(
                        masked_imu[b, s, c].item(),
                        masked_imu[b, e, c].item(),
                        e - s + 1,
                        device=x.device,
                        dtype=x.dtype,
                    )

        left_dist = torch.stack(left_dist_list, dim=0).unsqueeze(-1)
        right_dist = torch.stack(right_dist_list, dim=0).unsqueeze(-1)
        rel_pos = torch.stack(rel_pos_list, dim=0).unsqueeze(-1)
        missing_indicator = torch.stack(missing_indicator_list, dim=0).unsqueeze(-1)

        residual_observation = masked_imu - baseline
        enriched = torch.cat(
            [
                residual_observation,
                mask,
                dt,
                delta,
                window_feats,
                baseline,
                left_dist,
                right_dist,
                rel_pos,
                missing_indicator,
            ],
            dim=-1,
        )
        return enriched, baseline


class BidirectionalLNNResidual(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int, hidden_units: int = 128, output_dim: int = 6):
        super().__init__()
        self.feature_dim = feature_dim
        self.adapter = ResidualInputAdapter(feature_dim=feature_dim, input_dim=input_dim)
        residual_input_dim = input_dim + feature_dim + 4
        self.forward_lnn = CfC(residual_input_dim, hidden_units, batch_first=True, mixed_memory=True)
        self.backward_lnn = CfC(residual_input_dim, hidden_units, batch_first=True, mixed_memory=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_units * 2, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_units * 2, hidden_units // 2),
            nn.ReLU(),
            nn.Linear(hidden_units // 2, output_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enriched, baseline = self.adapter.build(x)
        fwd_h, _ = self.forward_lnn(enriched)
        rev_x = torch.flip(enriched, dims=[1])
        bwd_h_rev, _ = self.backward_lnn(rev_x)
        bwd_h = torch.flip(bwd_h_rev, dims=[1])
        h = torch.cat([fwd_h, bwd_h], dim=-1)
        residual = self.head(h)
        pred = baseline + residual
        uncertainty = self.uncertainty_head(h)
        return pred, uncertainty


class ResidualBiLSTM(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int, hidden_dim: int = 128, output_dim: int = 6, num_layers: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.adapter = ResidualInputAdapter(feature_dim=feature_dim, input_dim=input_dim)
        residual_input_dim = input_dim + feature_dim + 4
        self.backbone = LongTermLSTM(
            input_dim=residual_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enriched, baseline = self.adapter.build(x)
        residual, uncertainty = self.backbone(enriched)
        pred = baseline + residual
        return pred, uncertainty


class ResidualHybridBiLNNBiLSTM(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int, lnn_hidden: int = 128, lstm_hidden: int = 64, output_dim: int = 6, lstm_layers: int = 2):
        super().__init__()
        self.bi_lnn = BidirectionalLNNResidual(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_units=lnn_hidden,
            output_dim=output_dim,
        )
        self.bi_lstm = ResidualBiLSTM(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dim=lstm_hidden,
            output_dim=output_dim,
            num_layers=lstm_layers,
        )
        gate_input_dim = output_dim * 4 + input_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid(),
        )
        self.combined_uncertainty = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lnn_pred, lnn_unc = self.bi_lnn(x)
        lstm_pred, lstm_unc = self.bi_lstm(x)
        gate_in = torch.cat([lnn_pred, lstm_pred, lnn_unc, lstm_unc, x], dim=-1)
        gate = self.gate_net(gate_in)
        pred = gate * lnn_pred + (1.0 - gate) * lstm_pred
        uncertainty = self.combined_uncertainty(torch.cat([lnn_unc, lstm_unc], dim=-1))
        return pred, uncertainty

    def forward_with_components(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        lnn_pred, lnn_unc = self.bi_lnn(x)
        lstm_pred, lstm_unc = self.bi_lstm(x)
        gate_in = torch.cat([lnn_pred, lstm_pred, lnn_unc, lstm_unc, x], dim=-1)
        gate = self.gate_net(gate_in)
        pred = gate * lnn_pred + (1.0 - gate) * lstm_pred
        uncertainty = self.combined_uncertainty(torch.cat([lnn_unc, lstm_unc], dim=-1))
        return {
            "pred": pred,
            "uncertainty": uncertainty,
            "lnn_pred": lnn_pred,
            "lstm_pred": lstm_pred,
            "gate": gate,
        }


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


def _compute_missing_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    missing = 1.0 - mask
    mse_missing = ((pred - target) ** 2 * missing).sum() / (missing.sum() + 1e-8)
    return float(torch.sqrt(mse_missing + 1e-12).item())


def _evaluate(model, loader, criterion, device):
    model.eval()
    losses, rmse_missing_list, mse_missing_list, mse_all_list = [], [], [], []
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
            missing = 1.0 - mask
            mse_missing = ((pred - targets) ** 2 * missing).sum() / (missing.sum() + 1e-8)
            rmse_missing_list.append(_compute_missing_rmse(pred, targets, mask))
            mse_missing_list.append(float(mse_missing.item()))
            mse_all_list.append(F.mse_loss(pred, targets).item())

    return {
        "total": float(np.mean(losses)),
        "rmse_missing": float(np.mean(rmse_missing_list)),
        "mse_missing": float(np.mean(mse_missing_list)),
        "mse_all": float(np.mean(mse_all_list)),
    }


def _evaluate_trajectory_metrics(model, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_ate = 0.0
    total_rte = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            inputs, _, _, stats, vicon = _unpack_batch(batch)
            if stats is None or vicon is None:
                continue
            inputs = inputs.to(device)
            stats = stats.to(device)
            vicon = vicon.to(device)
            dt = inputs[:, :, 12] if inputs.shape[-1] > 12 else inputs[:, :, -1]

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


def _evaluate_across_missing_rates(
    model: nn.Module,
    config: dict,
    device: torch.device,
    criterion: nn.Module,
    missing_rates: List[float],
) -> List[dict]:
    rows = []
    for rate in missing_rates:
        test_ds = CfCIMUDataset(
            root_dir=config["root_dir"],
            seq_len=config["seq_len"],
            mask_rate=rate,
            missing_mode=config["missing_mode"],
            split="test",
            split_ratio=config["train_ratio"],
            val_ratio=config["val_ratio"],
            eval_mode=True,
            return_stats=True,
            return_vicon=True,
            include_window_features=config["include_window_features"],
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=device.type == "cuda",
        )
        test_metrics = _evaluate(model, test_loader, criterion, device)
        traj_metrics = _evaluate_trajectory_metrics(model, test_loader, device)
        rows.append({
            "mask_rate": rate,
            "test_rmse_missing": float(test_metrics["rmse_missing"]),
            "test_mse_missing": float(test_metrics["mse_missing"]),
            "test_mse_all": float(test_metrics["mse_all"]),
            "test_ate": float(traj_metrics["ate"]),
            "test_rte": float(traj_metrics["rte"]),
        })
    return rows


def _train_model(model, model_name, train_loader, val_loader, device, config, output_path, timestamp, criterion):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        epochs=config["epochs"],
        steps_per_epoch=len(train_loader),
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse_missing": [],
        "val_mse_missing": [],
        "val_mse_all": [],
    }
    best_val_rmse = float("inf")
    best_epoch = 0
    best_path = output_path / f"best_model_{model_name}_{timestamp}.pt"

    start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        train_m = _train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_m = _evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_m["total"])
        history["val_loss"].append(val_m["total"])
        history["val_rmse_missing"].append(val_m["rmse_missing"])
        history["val_mse_missing"].append(val_m["mse_missing"])
        history["val_mse_all"].append(val_m["mse_all"])

        if val_m["rmse_missing"] < best_val_rmse:
            best_val_rmse = val_m["rmse_missing"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0 or epoch == config["epochs"]:
            print(
                f"  [{model_name}] Epoch {epoch:3d}/{config['epochs']}  "
                f"train={train_m['total']:.6f}  "
                f"val_rmse_missing={val_m['rmse_missing']:.6f}"
            )

    train_time = time.time() - start
    model.load_state_dict(torch.load(best_path, map_location=device))
    best_eval = _evaluate(model, val_loader, criterion, device)

    plot_training_curves(history, save_path=output_path / f"training_curves_{model_name}_{timestamp}.png")

    return history, {
        "best_epoch": best_epoch,
        "best_val_rmse_missing": float(best_eval["rmse_missing"]),
        "best_val_loss": float(best_eval["total"]),
        "best_val_mse_all": float(best_eval["mse_all"]),
    }, train_time


def _load_or_train_model(model, model_name, train_loader, val_loader, device, config, output_path, timestamp, criterion):
    reuse_existing = config.get("reuse_existing", True)
    existing_path = _find_latest_best_model(output_path, model_name) if reuse_existing else None
    if existing_path is not None:
        print(f"  [Reuse] Loading existing checkpoint: {existing_path.name}")
        model.load_state_dict(torch.load(existing_path, map_location=device))
        best_eval = _evaluate(model, val_loader, criterion, device)
        return None, {
            "best_epoch": -1,
            "best_val_rmse_missing": float(best_eval["rmse_missing"]),
            "best_val_loss": float(best_eval["total"]),
            "best_val_mse_all": float(best_eval["mse_all"]),
            "reused_checkpoint": existing_path.name,
        }, 0.0

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
    return history, best_metrics, train_time


def _plot_summary(summary_df: pd.DataFrame, output_path: Path, timestamp: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cols = [
        ("best_val_rmse_missing", "Best Val Missing RMSE"),
        ("test_ate", "Test ATE"),
        ("test_rte", "Test RTE"),
    ]
    x = np.arange(len(summary_df))
    labels = summary_df["model"].tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for ax, (col, title) in zip(axes, cols):
        vals = summary_df[col].tolist()
        bars = ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path / f"comparison_bidirectional_lnn_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_missing_rate_comparison(multi_rate_df: pd.DataFrame, output_path: Path, timestamp: str):
    if multi_rate_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("test_rmse_missing", "Missing RMSE"),
        ("test_ate", "ATE"),
        ("test_rte", "RTE"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for model_name, group in multi_rate_df.groupby("model"):
            g = group.sort_values("mask_rate")
            ax.plot(g["mask_rate"] * 100.0, g[metric], marker="o", linewidth=2, label=model_name)
        ax.set_title(title)
        ax.set_xlabel("Missing Rate (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path / f"missing_rate_comparison_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_experiment(
    epochs: int = 20,
    seq_len: int = 30,
    mask_rate: float = 0.3,
    missing_mode: str = "random",
    lnn_hidden: int = 128,
    lstm_hidden: int = 64,
    lstm_layers: int = 2,
    test_mask_rates: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
    reuse_existing: bool = True,
    output_dir: str = "results/bidirectional_lnn_residual",
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
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "include_window_features": True,
        "test_mask_rates": [float(v) for v in test_mask_rates],
        "reuse_existing": bool(reuse_existing),
    }

    _seed_all(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("BIDIRECTIONAL LNN RESIDUAL IMPUTATION EXPERIMENT")
    print("=" * 80)
    print(f"Device:      {device}")
    print(f"Seq len:     {config['seq_len']}")
    print(f"Missing:     {config['missing_mode']} @ {config['mask_rate'] * 100:.0f}%")
    print(f"Epochs:      {config['epochs']}")
    print(f"Output:      {output_path}")
    print("=" * 80)

    train_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=config["mask_rate"],
        missing_mode=config["missing_mode"],
        split="train",
        split_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        eval_mode=False,
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
        return_stats=True,
        return_vicon=True,
        include_window_features=config["include_window_features"],
    )

    config["input_dim"] = int(train_ds.input_dim)
    config["feature_dim"] = int(train_ds.feature_dim)

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

    criterion = ReconstructionRMSEOnlyLoss()

    models_to_train = {
        "Bidirectional_LNN": BidirectionalLNNResidual(
            input_dim=config["input_dim"],
            feature_dim=config["feature_dim"],
            hidden_units=config["lnn_hidden"],
            output_dim=6,
        ),
        "BiLSTM": ResidualBiLSTM(
            input_dim=config["input_dim"],
            feature_dim=config["feature_dim"],
            hidden_dim=config["lstm_hidden"],
            output_dim=6,
            num_layers=config["lstm_layers"],
        ),
        "Hybrid_BiLNN_BiLSTM": ResidualHybridBiLNNBiLSTM(
            input_dim=config["input_dim"],
            feature_dim=config["feature_dim"],
            lnn_hidden=config["lnn_hidden"],
            lstm_hidden=config["lstm_hidden"],
            output_dim=6,
            lstm_layers=config["lstm_layers"],
        ),
    }

    summary_rows: List[dict] = []
    history_rows: List[dict] = []
    multi_rate_rows: List[dict] = []

    for model_name, model in models_to_train.items():
        model = model.to(device)
        _seed_all(config["seed"])
        num_params = count_parameters(model)
        size_mb = _state_dict_size_mb(model)

        print(f"\n{'=' * 80}")
        print(f"Training: {model_name} (params={num_params:,}, size={size_mb:.2f} MB)")
        print(f"{'=' * 80}")

        history, best_metrics, train_time = _load_or_train_model(
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
        test_metrics = _evaluate(model, test_loader, criterion, device)
        traj_metrics = _evaluate_trajectory_metrics(model, test_loader, device)

        per_rate_rows = _evaluate_across_missing_rates(
            model=model,
            config=config,
            device=device,
            criterion=criterion,
            missing_rates=config["test_mask_rates"],
        )
        for row in per_rate_rows:
            multi_rate_rows.append({"model": model_name, **row})

        if history is not None:
            for ei in range(len(history["train_loss"])):
                history_rows.append({
                    "model": model_name,
                    "epoch": ei + 1,
                    "train_loss": history["train_loss"][ei],
                    "val_loss": history["val_loss"][ei],
                    "val_rmse_missing": history["val_rmse_missing"][ei],
                    "val_mse_missing": history["val_mse_missing"][ei],
                    "val_mse_all": history["val_mse_all"][ei],
                })

        summary_rows.append({
            "model": model_name,
            "num_params": num_params,
            "param_size_mb": round(size_mb, 4),
            "best_epoch": best_metrics["best_epoch"],
            "best_val_loss": best_metrics["best_val_loss"],
            "best_val_rmse_missing": best_metrics["best_val_rmse_missing"],
            "best_val_mse_all": best_metrics["best_val_mse_all"],
            "test_rmse_missing": test_metrics["rmse_missing"],
            "test_mse_all": test_metrics["mse_all"],
            "test_ate": float(traj_metrics["ate"]),
            "test_rte": float(traj_metrics["rte"]),
            "train_time_sec": round(train_time, 2),
            "checkpoint_reused": best_metrics.get("reused_checkpoint", ""),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_history = pd.DataFrame(history_rows)
    df_multi_rate = pd.DataFrame(multi_rate_rows)
    summary_csv = output_path / f"summary_bidirectional_lnn_{timestamp}.csv"
    history_csv = output_path / f"history_bidirectional_lnn_{timestamp}.csv"
    multi_rate_csv = output_path / f"missing_rate_comparison_{timestamp}.csv"
    excel_path = output_path / f"bidirectional_lnn_residual_{timestamp}.xlsx"

    df_summary.to_csv(summary_csv, index=False)
    df_history.to_csv(history_csv, index=False)
    df_multi_rate.to_csv(multi_rate_csv, index=False)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_history.to_excel(writer, sheet_name="History", index=False)
        df_multi_rate.to_excel(writer, sheet_name="MissingRateComparison", index=False)
        pd.DataFrame([config]).to_excel(writer, sheet_name="Config", index=False)

    _plot_summary(df_summary, output_path, timestamp)
    _plot_missing_rate_comparison(df_multi_rate, output_path, timestamp)

    with open(output_path / f"config_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print(f"[Saved] {summary_csv}")
    print(f"[Saved] {history_csv}")
    print(f"[Saved] {multi_rate_csv}")
    print(f"[Saved] {excel_path}")
    print("=" * 80)
    return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bidirectional LNN residual imputation experiment")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--missing_mode", type=str, default="random", choices=["random", "block", "channel"])
    parser.add_argument("--lnn_hidden", type=int, default=128)
    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--test_mask_rates", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--reuse_existing", action="store_true", help="Reuse latest saved best-model checkpoints if present")
    parser.add_argument("--output_dir", type=str, default="results/bidirectional_lnn_residual")
    args = parser.parse_args()

    run_experiment(
        epochs=args.epochs,
        seq_len=args.seq_len,
        mask_rate=args.mask_rate,
        missing_mode=args.missing_mode,
        lnn_hidden=args.lnn_hidden,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        test_mask_rates=tuple(args.test_mask_rates),
        reuse_existing=args.reuse_existing,
        output_dir=args.output_dir,
    )
