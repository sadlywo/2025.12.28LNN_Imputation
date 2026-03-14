"""
Baseline comparison experiment for IMU imputation.

Methods compared:
1. Mean Imputation
2. LOCF Imputation
3. KNN Imputation
4. MICE Imputation
5. GRU Imputer
6. Transformer Imputer

This script follows the same dataset split, window length, missing setting,
and evaluation protocol as experiment_bidirectional_lnn_residual.py.
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

from dataset import CfCIMUDataset, compute_ate, compute_relative_trajectory_error
from models import GRUImputer, TransformerImputer, ReconstructionOnlyLoss
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


def _extract_parts(inputs: torch.Tensor, feature_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked_imu = inputs[:, :, :feature_dim]
    mask = inputs[:, :, feature_dim:feature_dim * 2]
    dt = inputs[:, :, feature_dim * 2:feature_dim * 2 + 1]
    return masked_imu, mask, dt


class DeterministicUncertaintyWrapper(nn.Module):
    def __init__(self, method_name: str, feature_dim: int):
        super().__init__()
        self.method_name = method_name
        self.feature_dim = feature_dim

    def _mean_impute(self, x_masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        obs_sum = (x_masked * mask).sum(dim=1, keepdim=True)
        obs_count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_vals = obs_sum / obs_count
        return x_masked + (1.0 - mask) * mean_vals

    def _locf_impute(self, x_masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = x_masked.clone()
        B, T, C = out.shape
        for b in range(B):
            for c in range(C):
                last = None
                for t in range(T):
                    if mask[b, t, c] > 0.5:
                        last = out[b, t, c].item()
                    elif last is not None:
                        out[b, t, c] = last
                last = None
                for t in range(T - 1, -1, -1):
                    if mask[b, t, c] > 0.5:
                        last = out[b, t, c].item()
                    elif last is not None and (t == T - 1 or mask[b, t, c] <= 0.5):
                        if out[b, t, c].abs().item() < 1e-12:
                            out[b, t, c] = last
        return out

    def _knn_impute(self, x_masked: torch.Tensor, mask: torch.Tensor, k: int = 3) -> torch.Tensor:
        out = x_masked.clone()
        B, T, C = out.shape
        for b in range(B):
            for c in range(C):
                obs_idx = torch.nonzero(mask[b, :, c] > 0.5, as_tuple=False).flatten()
                if obs_idx.numel() == 0:
                    continue
                for t in range(T):
                    if mask[b, t, c] > 0.5:
                        continue
                    distances = torch.abs(obs_idx - t)
                    k_eff = min(k, obs_idx.numel())
                    nn_idx = obs_idx[torch.topk(distances.float(), k=k_eff, largest=False).indices]
                    out[b, t, c] = x_masked[b, nn_idx, c].mean()
        return out

    def _mice_impute(self, x_masked: torch.Tensor, mask: torch.Tensor, iterations: int = 2) -> torch.Tensor:
        out = self._mean_impute(x_masked, mask)
        B, T, C = out.shape
        flat = out.reshape(B * T, C)
        flat_mask = mask.reshape(B * T, C)
        for _ in range(iterations):
            for c in range(C):
                miss = flat_mask[:, c] < 0.5
                obs = ~miss
                if obs.sum() < 2 or miss.sum() == 0:
                    continue
                x_obs = flat[obs][:, torch.arange(C) != c]
                y_obs = flat[obs][:, c:c + 1]
                x_miss = flat[miss][:, torch.arange(C) != c]
                x_obs_aug = torch.cat([x_obs, torch.ones(x_obs.shape[0], 1, device=flat.device)], dim=1)
                x_miss_aug = torch.cat([x_miss, torch.ones(x_miss.shape[0], 1, device=flat.device)], dim=1)
                beta = torch.linalg.pinv(x_obs_aug) @ y_obs
                flat[miss, c:c + 1] = x_miss_aug @ beta
        return flat.reshape(B, T, C)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_masked, mask, _ = _extract_parts(inputs, self.feature_dim)
        if self.method_name == "Mean":
            pred = self._mean_impute(x_masked, mask)
        elif self.method_name == "LOCF":
            pred = self._locf_impute(x_masked, mask)
        elif self.method_name == "KNN":
            pred = self._knn_impute(x_masked, mask)
        elif self.method_name == "MICE":
            pred = self._mice_impute(x_masked, mask)
        else:
            raise ValueError(f"Unsupported deterministic baseline: {self.method_name}")
        uncertainty = torch.ones_like(pred) * 0.1
        return pred, uncertainty


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
    return {"ate": total_ate / total_samples, "rte": total_rte / total_samples}


def _evaluate_across_missing_rates(model, config: dict, device: torch.device, criterion: nn.Module, missing_rates: List[float]) -> List[dict]:
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
    history = {"train_loss": [], "val_loss": [], "val_rmse_missing": [], "val_mse_missing": [], "val_mse_all": []}
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
            print(f"  [{model_name}] Epoch {epoch:3d}/{config['epochs']} train={train_m['total']:.6f} val_rmse_missing={val_m['rmse_missing']:.6f}")
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
    existing_path = _find_latest_best_model(output_path, model_name) if config.get("reuse_existing", True) else None
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
    return _train_model(model, model_name, train_loader, val_loader, device, config, output_path, timestamp, criterion)


def _plot_summary(summary_df: pd.DataFrame, output_path: Path, timestamp: str):
    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    cols = [("best_val_rmse_missing", "Best Val Missing RMSE"), ("test_ate", "Test ATE"), ("test_rte", "Test RTE")]
    x = np.arange(len(summary_df))
    labels = summary_df["model"].tolist()
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for ax, (col, title) in zip(axes, cols):
        vals = summary_df[col].tolist()
        bars = ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path / f"baseline_comparison_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_missing_rate_comparison(multi_rate_df: pd.DataFrame, output_path: Path, timestamp: str):
    if multi_rate_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    metrics = [("test_rmse_missing", "Missing RMSE"), ("test_ate", "ATE"), ("test_rte", "RTE")]
    for ax, (metric, title) in zip(axes, metrics):
        for model_name, group in multi_rate_df.groupby("model"):
            g = group.sort_values("mask_rate")
            ax.plot(g["mask_rate"] * 100.0, g[metric], marker="o", linewidth=2, label=model_name)
        ax.set_title(title)
        ax.set_xlabel("Missing Rate (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path / f"baseline_missing_rate_comparison_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_experiment(
    epochs: int = 20,
    seq_len: int = 30,
    mask_rate: float = 0.3,
    missing_mode: str = "random",
    hidden_units: int = 128,
    transformer_nhead: int = 4,
    transformer_nlayers: int = 2,
    test_mask_rates: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
    reuse_existing: bool = True,
    output_dir: str = "results/baseline_imputation_comparison",
):
    config = {
        "root_dir": "Oxford Dataset",
        "seq_len": int(seq_len),
        "mask_rate": float(mask_rate),
        "missing_mode": str(missing_mode),
        "batch_size": 32,
        "epochs": int(epochs),
        "lr": 1e-3,
        "hidden_units": int(hidden_units),
        "transformer_nhead": int(transformer_nhead),
        "transformer_nlayers": int(transformer_nlayers),
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

    train_ds = CfCIMUDataset(
        root_dir=config["root_dir"], seq_len=config["seq_len"], mask_rate=config["mask_rate"], missing_mode=config["missing_mode"],
        split="train", split_ratio=config["train_ratio"], val_ratio=config["val_ratio"], eval_mode=False,
        include_window_features=config["include_window_features"],
    )
    val_ds = CfCIMUDataset(
        root_dir=config["root_dir"], seq_len=config["seq_len"], mask_rate=config["mask_rate"], missing_mode=config["missing_mode"],
        split="val", split_ratio=config["train_ratio"], val_ratio=config["val_ratio"], eval_mode=True,
        include_window_features=config["include_window_features"],
    )
    test_ds = CfCIMUDataset(
        root_dir=config["root_dir"], seq_len=config["seq_len"], mask_rate=config["mask_rate"], missing_mode=config["missing_mode"],
        split="test", split_ratio=config["train_ratio"], val_ratio=config["val_ratio"], eval_mode=True,
        return_stats=True, return_vicon=True, include_window_features=config["include_window_features"],
    )

    config["input_dim"] = int(train_ds.input_dim)
    config["feature_dim"] = int(train_ds.feature_dim)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=device.type == "cuda")
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=device.type == "cuda")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=device.type == "cuda")

    criterion = ReconstructionOnlyLoss(w_recon=1.0)

    deterministic_models = {
        "Mean": DeterministicUncertaintyWrapper("Mean", feature_dim=config["feature_dim"]),
        "LOCF": DeterministicUncertaintyWrapper("LOCF", feature_dim=config["feature_dim"]),
        "KNN": DeterministicUncertaintyWrapper("KNN", feature_dim=config["feature_dim"]),
        "MICE": DeterministicUncertaintyWrapper("MICE", feature_dim=config["feature_dim"]),
    }
    learned_models = {
        "GRU": GRUImputer(input_dim=config["input_dim"], hidden_dim=config["hidden_units"], output_dim=6),
        "Transformer": TransformerImputer(input_dim=config["input_dim"], hidden_dim=config["hidden_units"], output_dim=6, nhead=config["transformer_nhead"], nlayers=config["transformer_nlayers"]),
    }

    summary_rows, history_rows, multi_rate_rows = [], [], []

    for model_name, model in {**deterministic_models, **learned_models}.items():
        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) if isinstance(model, nn.Module) else 0
        size_mb = _state_dict_size_mb(model) if num_params > 0 else 0.0

        print(f"\n{'=' * 80}")
        print(f"Running: {model_name} (params={num_params:,}, size={size_mb:.2f} MB)")
        print(f"{'=' * 80}")

        if model_name in learned_models:
            history, best_metrics, train_time = _load_or_train_model(model, model_name, train_loader, val_loader, device, config, output_path, timestamp, criterion)
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
        else:
            history = None
            best_eval = _evaluate(model, val_loader, criterion, device)
            best_metrics = {
                "best_epoch": -1,
                "best_val_rmse_missing": float(best_eval["rmse_missing"]),
                "best_val_loss": float(best_eval["total"]),
                "best_val_mse_all": float(best_eval["mse_all"]),
            }
            train_time = 0.0

        test_metrics = _evaluate(model, test_loader, criterion, device)
        traj_metrics = _evaluate_trajectory_metrics(model, test_loader, device)
        for row in _evaluate_across_missing_rates(model, config, device, criterion, config["test_mask_rates"]):
            multi_rate_rows.append({"model": model_name, **row})

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

    summary_csv = output_path / f"summary_baseline_comparison_{timestamp}.csv"
    history_csv = output_path / f"history_baseline_comparison_{timestamp}.csv"
    multi_rate_csv = output_path / f"missing_rate_comparison_{timestamp}.csv"
    excel_path = output_path / f"baseline_imputation_comparison_{timestamp}.xlsx"

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
    parser = argparse.ArgumentParser(description="Baseline imputation comparison experiment")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--missing_mode", type=str, default="random", choices=["random", "block", "channel"])
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--transformer_nhead", type=int, default=4)
    parser.add_argument("--transformer_nlayers", type=int, default=2)
    parser.add_argument("--test_mask_rates", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--reuse_existing", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/baseline_imputation_comparison")
    args = parser.parse_args()

    run_experiment(
        epochs=args.epochs,
        seq_len=args.seq_len,
        mask_rate=args.mask_rate,
        missing_mode=args.missing_mode,
        hidden_units=args.hidden_units,
        transformer_nhead=args.transformer_nhead,
        transformer_nlayers=args.transformer_nlayers,
        test_mask_rates=tuple(args.test_mask_rates),
        reuse_existing=args.reuse_existing,
        output_dir=args.output_dir,
    )
