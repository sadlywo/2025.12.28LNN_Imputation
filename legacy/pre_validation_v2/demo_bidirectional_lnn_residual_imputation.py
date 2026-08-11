from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import CfCIMUDataset, compute_ate
from experiment_bidirectional_lnn_residual import (
    BidirectionalLNNResidual,
    ResidualBiLSTM,
    ResidualHybridBiLNNBiLSTM,
)


def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_latest_checkpoint(output_dir: Path, model_name: str) -> Path | None:
    candidates = sorted(
        output_dir.glob(f"best_model_{model_name}_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _build_models(input_dim: int, feature_dim: int, lnn_hidden: int, lstm_hidden: int, lstm_layers: int):
    return {
        "Bidirectional_LNN": BidirectionalLNNResidual(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_units=lnn_hidden,
            output_dim=6,
        ),
        "BiLSTM": ResidualBiLSTM(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dim=lstm_hidden,
            output_dim=6,
            num_layers=lstm_layers,
        ),
        "Hybrid_BiLNN_BiLSTM": ResidualHybridBiLNNBiLSTM(
            input_dim=input_dim,
            feature_dim=feature_dim,
            lnn_hidden=lnn_hidden,
            lstm_hidden=lstm_hidden,
            output_dim=6,
            lstm_layers=lstm_layers,
        ),
    }


def _extract_demo_batch(
    dataset: CfCIMUDataset,
    index: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    item = dataset[index]
    if len(item) < 5:
        raise ValueError("Dataset item does not contain stats and vicon. Please enable return_stats and return_vicon.")
    inputs, target, mask, stats, vicon = item[:5]
    return (
        inputs.unsqueeze(0).to(device),
        target.unsqueeze(0).to(device),
        mask.unsqueeze(0).to(device),
        stats.unsqueeze(0).to(device),
        vicon.unsqueeze(0).to(device),
    )


def _longest_missing_segment(mask: torch.Tensor) -> Tuple[int, int]:
    missing = (mask.mean(dim=-1) < 0.999).cpu().numpy().astype(np.int32)
    best_len = 0
    best = (0, max(1, len(missing) // 4))
    start = None
    for i, v in enumerate(missing):
        if v == 1 and start is None:
            start = i
        if (v == 0 or i == len(missing) - 1) and start is not None:
            end = i if v == 0 else i + 1
            seg_len = end - start
            if seg_len > best_len:
                best_len = seg_len
                best = (start, end)
            start = None
    return best


def _all_missing_segments(mask: torch.Tensor, threshold: float = 0.999) -> List[Tuple[int, int]]:
    missing = (mask.mean(dim=-1) < threshold).cpu().numpy().astype(np.int32)
    segments: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(missing):
        if v == 1 and start is None:
            start = i
        if (v == 0 or i == len(missing) - 1) and start is not None:
            end = i if v == 0 else i + 1
            segments.append((start, end))
            start = None
    return segments


def _pick_segments_for_demo(segments: List[Tuple[int, int]], max_segments: int = 3) -> List[Tuple[int, int]]:
    if len(segments) == 0:
        return [(0, 1)]
    segments = sorted(segments, key=lambda z: (z[1] - z[0]), reverse=True)
    return segments[:max_segments]


def _choose_xy_axes(traj_xyz: np.ndarray) -> Tuple[int, int]:
    if traj_xyz.ndim != 2 or traj_xyz.shape[1] < 3:
        return 0, min(1, traj_xyz.shape[1] - 1)
    d = np.diff(traj_xyz, axis=0)
    mono = np.abs(np.mean(np.sign(d + 1e-12), axis=0))
    span = np.ptp(traj_xyz, axis=0)
    score = (1.0 - mono) * (span + 1e-12)
    axes = np.argsort(score)[-2:]
    if axes[0] == axes[1]:
        axes = np.argsort(span)[-2:]
    axes = np.sort(axes)
    return int(axes[0]), int(axes[1])


def _choose_xy_axes_by_span(traj_xyz: np.ndarray) -> Tuple[int, int]:
    if traj_xyz.ndim != 2 or traj_xyz.shape[1] < 3:
        return 0, min(1, traj_xyz.shape[1] - 1)
    span = np.ptp(traj_xyz, axis=0)
    axes = np.argsort(span)[-2:]
    axes = np.sort(axes)
    return int(axes[0]), int(axes[1])


def _to_xy(arr: torch.Tensor, axes: Tuple[int, int]) -> np.ndarray:
    arr_np = arr.detach().cpu().numpy()
    return arr_np[:, [axes[0], axes[1]]]


def _compute_limits(arrays: List[np.ndarray], margin_scale: float = 0.45):
    pts = np.concatenate(arrays, axis=0)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    dx = float(x_max - x_min)
    dy = float(y_max - y_min)
    pad_x = max(1e-3, dx * margin_scale)
    pad_y = max(1e-3, dy * margin_scale)
    return (x_min - pad_x, x_max + pad_x), (y_min - pad_y, y_max + pad_y)


def _subsample_one_tenth(xy: np.ndarray) -> np.ndarray:
    n = len(xy)
    if n <= 2:
        return xy
    k = max(2, n // 10)
    idx = np.linspace(0, n - 1, num=k, dtype=int)
    return xy[idx]


def _loop_score(xy: np.ndarray) -> float:
    if len(xy) < 6:
        return -1e9
    d = np.diff(xy, axis=0)
    speed = np.linalg.norm(d, axis=1)
    valid = speed > 1e-8
    if valid.sum() < 4:
        return -1e9
    heading = np.arctan2(d[:, 1], d[:, 0])
    dheading = np.diff(heading)
    dheading = (dheading + np.pi) % (2 * np.pi) - np.pi
    turn_sum = float(np.sum(np.abs(dheading[valid[1:]])))
    span = float(np.linalg.norm(np.ptp(xy, axis=0))) + 1e-8
    closure = float(np.linalg.norm(xy[-1] - xy[0])) / span
    return turn_sum - 3.0 * closure


def _find_loop_like_sample(dataset: CfCIMUDataset, max_scan: int = 500) -> int:
    best_idx = 0
    best_score = -1e9
    scan_n = min(len(dataset), max_scan)
    for i in range(scan_n):
        item = dataset[i]
        vicon = item[4]
        gt = vicon[:, :3].detach().cpu().numpy()
        axes = _choose_xy_axes(gt)
        xy = gt[:, [axes[0], axes[1]]]
        score = _loop_score(xy)
        if score > best_score:
            best_score = score
            best_idx = i
    return int(best_idx)


def run_demo(
    output_dir: str = "results/bidirectional_lnn_residual",
    demo_output_dir: str = "results/bidirectional_lnn_residual/demo_imputation",
    seq_len: int = 30,
    mask_rate: float = 0.3,
    missing_mode: str = "random",
    sample_index: int = 0,
    lnn_hidden: int = 128,
    lstm_hidden: int = 64,
    lstm_layers: int = 2,
):
    _seed_all(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(output_dir)
    save_dir = Path(demo_output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset = CfCIMUDataset(
        root_dir="Oxford Dataset",
        seq_len=seq_len,
        mask_rate=mask_rate,
        missing_mode=missing_mode,
        split="test",
        split_ratio=0.8,
        val_ratio=0.1,
        eval_mode=True,
        return_stats=True,
        return_vicon=True,
        include_window_features=True,
    )
    if len(dataset) == 0:
        raise ValueError("No test sample available for demo.")
    sample_index = int(np.clip(sample_index, 0, len(dataset) - 1))
    inputs, target, mask, stats, vicon = _extract_demo_batch(dataset, sample_index, device)

    feature_dim = dataset.feature_dim
    dt_index = feature_dim * 2
    dt = inputs[:, :, dt_index]
    masked_signal = inputs[:, :, :feature_dim]
    gt_pos = vicon[:, :, :3]

    xy_axes = _choose_xy_axes(gt_pos[0].detach().cpu().numpy())

    models = _build_models(
        input_dim=dataset.input_dim,
        feature_dim=dataset.feature_dim,
        lnn_hidden=lnn_hidden,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
    )

    model_outputs: Dict[str, Dict[str, np.ndarray | float]] = {}
    loaded_models: Dict[str, torch.nn.Module] = {}
    load_rows: List[dict] = []
    for model_name, model in models.items():
        ckpt = _find_latest_checkpoint(ckpt_dir, model_name)
        if ckpt is None:
            load_rows.append({"model": model_name, "checkpoint": "", "loaded": False, "note": "checkpoint not found"})
            continue

        model = model.to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            pred, _ = model(inputs)
        loaded_models[model_name] = model

        before_result = compute_ate(masked_signal, gt_pos, dt, stats=stats)
        after_result = compute_ate(pred, gt_pos, dt, stats=stats)
        traj_before = before_result["pred_trajectory"][0]
        traj_after = after_result["pred_trajectory"][0]
        traj_gt = gt_pos[0]
        missing_mask_3d = (1.0 - mask[:, :, :6]).max(dim=-1).values.unsqueeze(-1)
        pred_missing_err = (pred - target)[:, :, :6] * missing_mask_3d
        missing_count = int(missing_mask_3d.sum().item())
        missing_rmse = float(
            torch.sqrt((pred_missing_err.pow(2).sum() / max(missing_count * 6, 1))).item()
        )
        model_outputs[model_name] = {
            "before_xy": _to_xy(traj_before, xy_axes),
            "after_xy": _to_xy(traj_after, xy_axes),
            "gt_xy": _to_xy(traj_gt, xy_axes),
            "ate_before": float(before_result["ate"]),
            "ate_after": float(after_result["ate"]),
            "missing_rmse": missing_rmse,
        }
        load_rows.append({"model": model_name, "checkpoint": ckpt.name, "loaded": True, "note": "ok"})

    if len(model_outputs) == 0:
        raise FileNotFoundError(f"No trained checkpoints found in {ckpt_dir}.")

    first_name = next(iter(model_outputs.keys()))

    all_segments = _all_missing_segments(mask[0], threshold=0.999)
    selected_segments = _pick_segments_for_demo(all_segments, max_segments=3)
    s0, s1 = selected_segments[0]
    missing_step_idx = np.where((mask[0].mean(dim=-1).detach().cpu().numpy() < 0.999))[0]

    model_names = list(model_outputs.keys())
    all_traj_arrays: List[np.ndarray] = []
    for name in model_names:
        d = model_outputs[name]
        all_traj_arrays.extend([d["gt_xy"], d["before_xy"], d["after_xy"]])
    xlim_full, ylim_full = _compute_limits(all_traj_arrays, margin_scale=0.55)
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.8), squeeze=False)
    for i, name in enumerate(model_names):
        ax = axes[0, i]
        d = model_outputs[name]
        ax.plot(d["before_xy"][:, 0], d["before_xy"][:, 1], color="#d62728", linestyle="--", linewidth=1.25, alpha=0.9, label="Before imputation", zorder=2)
        ax.plot(d["after_xy"][:, 0], d["after_xy"][:, 1], color="#1f77b4", linewidth=1.4, alpha=0.9, label="After imputation", zorder=3)
        ax.plot(
            d["gt_xy"][:, 0],
            d["gt_xy"][:, 1],
            color="black",
            linewidth=2.2,
            alpha=0.95,
            marker="o",
            markersize=2.8,
            markevery=max(1, len(d["gt_xy"]) // 20),
            label="GT trajectory",
            zorder=6,
        )
        if len(missing_step_idx) > 0:
            ax.scatter(
                d["gt_xy"][missing_step_idx, 0],
                d["gt_xy"][missing_step_idx, 1],
                s=20,
                color="#ffbf00",
                edgecolor="black",
                linewidth=0.25,
                alpha=0.95,
                label="Missing points",
                zorder=5,
            )
        ax.set_title(name)
        ax.set_xlabel("X Position (m)")
        if i == 0:
            ax.set_ylabel("Y Position (m)")
        ax.set_xlim(*xlim_full)
        ax.set_ylim(*ylim_full)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="best", frameon=True, framealpha=0.9)
    fig.suptitle(
        f"Trajectory Imputation Demo (mode={missing_mode}, mask_rate={mask_rate:.0%}, sample={sample_index})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig_path_full = save_dir / f"trajectory_imputation_comparison_{timestamp}.png"
    fig.savefig(fig_path_full, dpi=300, bbox_inches="tight")
    plt.close(fig)

    k = len(selected_segments)
    fig2, axes2 = plt.subplots(k, n, figsize=(5.6 * n, 2.9 * k + 1.2), squeeze=False)
    for r, seg in enumerate(selected_segments):
        rs, re = seg
        for i, name in enumerate(model_names):
            ax = axes2[r, i]
            d = model_outputs[name]
            gt_seg = d["gt_xy"][rs:re]
            before_seg = d["before_xy"][rs:re]
            after_seg = d["after_xy"][rs:re]
            ax.plot(before_seg[:, 0], before_seg[:, 1], color="#d62728", linestyle="--", linewidth=1.35, label="Before imputation", zorder=2)
            ax.plot(after_seg[:, 0], after_seg[:, 1], color="#1f77b4", linewidth=1.5, label="After imputation", zorder=3)
            ax.plot(
                gt_seg[:, 0],
                gt_seg[:, 1],
                color="black",
                linewidth=2.1,
                marker="o",
                markersize=3.0,
                markevery=max(1, len(gt_seg) // 6),
                label="GT segment",
                zorder=6,
            )
            ax.scatter(gt_seg[0, 0], gt_seg[0, 1], color="#2ca02c", s=30, zorder=5, label="Segment start")
            ax.scatter(gt_seg[-1, 0], gt_seg[-1, 1], color="#ff7f0e", marker="X", s=40, zorder=5, label="Segment end")
            ax.set_title(f"{name} | missing segment [{rs}, {re})")
            ax.set_xlabel("X Position (m)")
            if i == 0:
                ax.set_ylabel("Y Position (m)")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            if r == 0 and i == 0:
                ax.legend(loc="best", frameon=True, framealpha=0.9)
    fig2.suptitle(
        f"Zoomed Multi-Segment Trajectory Comparison (mode={missing_mode})",
        fontsize=13,
        fontweight="bold",
    )
    fig2.tight_layout()
    fig_path_seg = save_dir / f"trajectory_missing_segment_comparison_{timestamp}.png"
    fig2.savefig(fig_path_seg, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    fig3, axes3 = plt.subplots(1, n, figsize=(5.6 * n, 4.4), squeeze=False, sharey=True)
    if len(missing_step_idx) > 0:
        miss_bool = np.zeros(mask.shape[1], dtype=bool)
        miss_bool[missing_step_idx] = True
    else:
        miss_bool = np.zeros(mask.shape[1], dtype=bool)
        miss_bool[max(0, s0):min(mask.shape[1], s1)] = True
    obs_bool = ~miss_bool
    for i, name in enumerate(model_names):
        ax = axes3[0, i]
        d = model_outputs[name]
        err_before = np.linalg.norm(d["before_xy"] - d["gt_xy"], axis=1)
        err_after = np.linalg.norm(d["after_xy"] - d["gt_xy"], axis=1)
        before_missing = float(np.mean(err_before[miss_bool])) if miss_bool.any() else float(np.mean(err_before))
        after_missing = float(np.mean(err_after[miss_bool])) if miss_bool.any() else float(np.mean(err_after))
        before_observed = float(np.mean(err_before[obs_bool])) if obs_bool.any() else float(np.mean(err_before))
        after_observed = float(np.mean(err_after[obs_bool])) if obs_bool.any() else float(np.mean(err_after))
        labels = ["Missing", "Observed"]
        x = np.arange(2)
        bw = 0.34
        ax.bar(x - bw / 2, [before_missing, before_observed], width=bw, color="#d62728", alpha=0.86, label="Before")
        ax.bar(x + bw / 2, [after_missing, after_observed], width=bw, color="#1f77b4", alpha=0.86, label="After")
        ax.set_title(name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if i == 0:
            ax.set_ylabel("Mean 2D Position Error (m)")
            ax.legend(loc="best", frameon=True, framealpha=0.9)
        ax.grid(True, axis="y", alpha=0.25)
    fig3.suptitle("Error Breakdown by Missing/Observed Steps", fontsize=13, fontweight="bold")
    fig3.tight_layout()
    fig_path_breakdown = save_dir / f"trajectory_missing_observed_breakdown_{timestamp}.png"
    fig3.savefig(fig_path_breakdown, dpi=300, bbox_inches="tight")
    plt.close(fig3)

    fig4, axes4 = plt.subplots(2, n, figsize=(5.9 * n, 7.2), squeeze=False, sharex=True)
    t = np.arange(mask.shape[1])
    for i, name in enumerate(model_names):
        d = model_outputs[name]
        for r, coord_name in enumerate(["X", "Y"]):
            ax = axes4[r, i]
            dim = 0 if r == 0 else 1
            ax.plot(t, d["gt_xy"][:, dim], color="black", linewidth=2.0, label="GT", zorder=4)
            ax.plot(t, d["before_xy"][:, dim], color="#d62728", linestyle="--", linewidth=1.35, label="Before", zorder=2)
            ax.plot(t, d["after_xy"][:, dim], color="#1f77b4", linewidth=1.45, label="After", zorder=3)
            for seg_start, seg_end in all_segments:
                ax.axvspan(seg_start, seg_end, color="#ffbf00", alpha=0.16)
            if r == 0:
                ax.set_title(name)
            if i == 0:
                ax.set_ylabel(f"{coord_name} Position (m)")
            if r == 1:
                ax.set_xlabel("Time Step")
            ax.grid(True, alpha=0.22)
            if r == 0 and i == 0:
                ax.legend(loc="best", frameon=True, framealpha=0.9)
    fig4.suptitle("Trajectory Coordinates Over Time (Missing Intervals Highlighted)", fontsize=13, fontweight="bold")
    fig4.tight_layout()
    fig_path_coords_t = save_dir / f"trajectory_coordinates_timeseries_{timestamp}.png"
    fig4.savefig(fig_path_coords_t, dpi=300, bbox_inches="tight")
    plt.close(fig4)

    fig5, axes5 = plt.subplots(1, n, figsize=(5.8 * n, 4.2), squeeze=False, sharey=True)
    time_steps = np.arange(mask.shape[1])
    for i, name in enumerate(model_names):
        ax = axes5[0, i]
        d = model_outputs[name]
        err_before = np.linalg.norm(d["before_xy"] - d["gt_xy"], axis=1)
        err_after = np.linalg.norm(d["after_xy"] - d["gt_xy"], axis=1)
        ax.plot(time_steps, err_before, color="#d62728", linestyle="--", linewidth=1.45, label="Before")
        ax.plot(time_steps, err_after, color="#1f77b4", linewidth=1.65, label="After")
        for seg_start, seg_end in all_segments:
            ax.axvspan(seg_start, seg_end, color="#ffbf00", alpha=0.18)
        ax.set_title(name)
        ax.set_xlabel("Time Step")
        if i == 0:
            ax.set_ylabel("2D Position Error (m)")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="best", frameon=True, framealpha=0.9)
    fig5.suptitle("Per-Step Trajectory Error (Highlighted Missing Intervals)", fontsize=13, fontweight="bold")
    fig5.tight_layout()
    fig_path_error_t = save_dir / f"trajectory_error_timeseries_{timestamp}.png"
    fig5.savefig(fig_path_error_t, dpi=300, bbox_inches="tight")
    plt.close(fig5)

    pd_rows = []
    for name in model_names:
        d = model_outputs[name]
        before_rmse = float(np.sqrt(np.mean(np.sum((d["before_xy"] - d["gt_xy"]) ** 2, axis=1))))
        after_rmse = float(np.sqrt(np.mean(np.sum((d["after_xy"] - d["gt_xy"]) ** 2, axis=1))))
        pd_rows.append(
            {
                "model": name,
                "trajectory_rmse_before": before_rmse,
                "trajectory_rmse_after": after_rmse,
                "rmse_improvement": before_rmse - after_rmse,
                "ate_before": float(d["ate_before"]),
                "ate_after": float(d["ate_after"]),
                "ate_improvement": float(d["ate_before"]) - float(d["ate_after"]),
                "missing_imputation_rmse": float(d["missing_rmse"]),
                "missing_segment_start": int(s0),
                "missing_segment_end": int(s1),
            }
        )
    df_metrics = pd.DataFrame(pd_rows)

    model_labels = df_metrics["model"].to_list()
    x = np.arange(len(model_labels))
    bar_w = 0.36
    fig6, axes6 = plt.subplots(1, 3, figsize=(17.2, 4.8), squeeze=False)
    ax_rmse = axes6[0, 0]
    ax_rmse.bar(x - bar_w / 2, df_metrics["trajectory_rmse_before"], width=bar_w, color="#d62728", alpha=0.85, label="Before")
    ax_rmse.bar(x + bar_w / 2, df_metrics["trajectory_rmse_after"], width=bar_w, color="#1f77b4", alpha=0.85, label="After")
    ax_rmse.set_title("Trajectory RMSE Comparison")
    ax_rmse.set_ylabel("RMSE (m)")
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(model_labels, rotation=10)
    ax_rmse.grid(True, axis="y", alpha=0.25)
    ax_rmse.legend(loc="best", frameon=True, framealpha=0.9)

    ax_gain = axes6[0, 1]
    ate_gain = df_metrics["ate_improvement"].to_numpy()
    colors = np.where(ate_gain >= 0, "#2ca02c", "#d62728")
    bars = ax_gain.bar(x, ate_gain, color=colors, alpha=0.88)
    ax_gain.axhline(0.0, color="black", linewidth=1.0)
    ax_gain.set_title("ATE Improvement (Before - After)")
    ax_gain.set_ylabel("ATE Gain (m)")
    ax_gain.set_xticks(x)
    ax_gain.set_xticklabels(model_labels, rotation=10)
    ax_gain.grid(True, axis="y", alpha=0.25)
    for j, b in enumerate(bars):
        h = float(b.get_height())
        va = "bottom" if h >= 0 else "top"
        dy = 0.01 if h >= 0 else -0.01
        ax_gain.text(b.get_x() + b.get_width() * 0.5, h + dy, f"{h:.3f}", ha="center", va=va, fontsize=9)

    ax_missing = axes6[0, 2]
    miss_rmse = df_metrics["missing_imputation_rmse"].to_numpy()
    miss_bars = ax_missing.bar(x, miss_rmse, color="#9467bd", alpha=0.9)
    ax_missing.set_title("Imputation RMSE on Missing IMU")
    ax_missing.set_ylabel("Missing-Point RMSE")
    ax_missing.set_xticks(x)
    ax_missing.set_xticklabels(model_labels, rotation=10)
    ax_missing.grid(True, axis="y", alpha=0.25)
    for b in miss_bars:
        h = float(b.get_height())
        ax_missing.text(b.get_x() + b.get_width() * 0.5, h + 0.001, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    fig6.suptitle("Model-Wise Downstream Trajectory Impact Summary", fontsize=13, fontweight="bold")
    fig6.tight_layout()
    fig_path_summary = save_dir / f"trajectory_downstream_impact_summary_{timestamp}.png"
    fig6.savefig(fig_path_summary, dpi=300, bbox_inches="tight")
    plt.close(fig6)
    df_load = pd.DataFrame(load_rows)
    metrics_csv = save_dir / f"demo_metrics_{timestamp}.csv"
    load_csv = save_dir / f"checkpoint_load_status_{timestamp}.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    df_load.to_csv(load_csv, index=False)

    print(f"[Saved] {fig_path_full}")
    print(f"[Saved] {fig_path_seg}")
    print(f"[Saved] {fig_path_breakdown}")
    print(f"[Saved] {fig_path_coords_t}")
    print(f"[Saved] {fig_path_error_t}")
    print(f"[Saved] {fig_path_summary}")
    print(f"[Saved] {metrics_csv}")
    print(f"[Saved] {load_csv}")
    print(f"[Info] XY axes selected from GT trajectory: {xy_axes}")
    print(f"[Info] missing segments found: {len(all_segments)}, selected: {selected_segments}")
    print(df_metrics.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo: load trained BiLNN/BiLSTM/Hybrid models for trajectory imputation under random missing")
    parser.add_argument("--output_dir", type=str, default="results/bidirectional_lnn_residual")
    parser.add_argument("--demo_output_dir", type=str, default="results/bidirectional_lnn_residual/demo_imputation")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--missing_mode", type=str, default="random", choices=["random", "block", "channel"])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--lnn_hidden", type=int, default=128)
    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=2)
    args = parser.parse_args()
    run_demo(
        output_dir=args.output_dir,
        demo_output_dir=args.demo_output_dir,
        seq_len=args.seq_len,
        mask_rate=args.mask_rate,
        missing_mode=args.missing_mode,
        sample_index=args.sample_index,
        lnn_hidden=args.lnn_hidden,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
    )
