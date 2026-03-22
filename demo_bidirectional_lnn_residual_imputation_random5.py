from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
        "BiLNN": BidirectionalLNNResidual(
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
        "Hybrid": ResidualHybridBiLNNBiLSTM(
            input_dim=input_dim,
            feature_dim=feature_dim,
            lnn_hidden=lnn_hidden,
            lstm_hidden=lstm_hidden,
            output_dim=6,
            lstm_layers=lstm_layers,
        ),
    }


def _extract_demo_batch(dataset: CfCIMUDataset, index: int, device: torch.device):
    item = dataset[index]
    if len(item) < 5:
        raise ValueError("Dataset item does not contain stats and vicon.")
    inputs, target, mask, stats, vicon = item[:5]
    return (
        inputs.unsqueeze(0).to(device),
        target.unsqueeze(0).to(device),
        mask.unsqueeze(0).to(device),
        stats.unsqueeze(0).to(device),
        vicon.unsqueeze(0).to(device),
    )


def _choose_xy_axes_by_span(traj_xyz: np.ndarray):
    if traj_xyz.ndim != 2 or traj_xyz.shape[1] < 3:
        return 0, min(1, traj_xyz.shape[1] - 1)
    span = np.ptp(traj_xyz, axis=0)
    axes = np.argsort(span)[-2:]
    axes = np.sort(axes)
    return int(axes[0]), int(axes[1])


def _to_xy(arr: torch.Tensor, axes):
    arr_np = arr.detach().cpu().numpy()
    return arr_np[:, [axes[0], axes[1]]]


def _to_xyz(arr: torch.Tensor):
    arr_np = arr.detach().cpu().numpy()
    return arr_np[:, :3]


def _select_five_indices(total: int) -> List[int]:
    if total <= 5:
        return list(range(total))
    idx = np.linspace(0, total - 1, 5)
    idx = np.round(idx).astype(int).tolist()
    dedup = []
    for i in idx:
        if i not in dedup:
            dedup.append(i)
    cur = 0
    while len(dedup) < 5 and cur < total:
        if cur not in dedup:
            dedup.append(cur)
        cur += 1
    return sorted(dedup[:5])


def run_demo(
    output_dir: str = "results/bidirectional_lnn_residual",
    demo_output_dir: str = "results/bidirectional_lnn_residual/demo_imputation",
    seq_len: int = 30,
    mask_rate: float = 0.3,
    missing_mode: str = "random",
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

    models = _build_models(
        input_dim=dataset.input_dim,
        feature_dim=dataset.feature_dim,
        lnn_hidden=lnn_hidden,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
    )

    checkpoint_aliases = {
        "BiLNN": ["Bidirectional_LNN", "BiLNN"],
        "BiLSTM": ["BiLSTM"],
        "Hybrid": ["Hybrid_BiLNN_BiLSTM", "Hybrid"],
    }
    loaded_models: Dict[str, torch.nn.Module] = {}
    missing_models: List[str] = []
    for model_name, model in models.items():
        ckpt = None
        for alias in checkpoint_aliases[model_name]:
            ckpt = _find_latest_checkpoint(ckpt_dir, alias)
            if ckpt is not None:
                break
        if ckpt is None:
            missing_models.append(model_name)
            continue
        model = model.to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()
        loaded_models[model_name] = model

    if missing_models:
        found = sorted([p.name for p in ckpt_dir.glob("best_model_*.pt")])
        raise FileNotFoundError(
            f"Missing checkpoints for {missing_models} in {ckpt_dir}. Found: {found}"
        )

    sample_indices = _select_five_indices(len(dataset))
    plot_rows: List[dict] = []
    panel_data: List[dict] = []
    curve_data_dir = save_dir / f"five_segments_curve_data_{timestamp}"
    curve_data_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in sample_indices:
        inputs, target, mask, stats, vicon = _extract_demo_batch(dataset, sample_idx, device)
        gt_pos = vicon[:, :, :3]
        dt = inputs[:, :, dataset.feature_dim * 2]
        xy_axes = _choose_xy_axes_by_span(gt_pos[0].detach().cpu().numpy())

        full_result = compute_ate(target, gt_pos, dt, stats=stats)
        full_xyz = _to_xyz(full_result["pred_trajectory"][0])
        gt_xyz = _to_xyz(gt_pos[0])
        full_xy = full_xyz[:, [xy_axes[0], xy_axes[1]]]
        gt_xy = gt_xyz[:, [xy_axes[0], xy_axes[1]]]

        model_xy = {}
        model_xyz = {}
        for model_name, model in loaded_models.items():
            with torch.no_grad():
                pred, _ = model(inputs)
            out = compute_ate(pred, gt_pos, dt, stats=stats)
            pred_xyz = _to_xyz(out["pred_trajectory"][0])
            model_xyz[model_name] = pred_xyz
            model_xy[model_name] = pred_xyz[:, [xy_axes[0], xy_axes[1]]]
            plot_rows.append(
                {
                    "sample_index": sample_idx,
                    "model": model_name,
                    "ate": float(out["ate"]),
                    "full_imu_ate": float(full_result["ate"]),
                }
            )

        panel_data.append(
            {
                "sample_index": sample_idx,
                "full_xy": full_xy,
                "gt_xy": gt_xy,
                "full_xyz": full_xyz,
                "gt_xyz": gt_xyz,
                "model_xy": model_xy,
                "model_xyz": model_xyz,
                "miss_steps": np.where((mask[0].mean(dim=-1).detach().cpu().numpy() < 0.999))[0],
            }
        )

    all_long_rows: List[dict] = []
    for seg_id, panel in enumerate(panel_data, start=1):
        n_steps = panel["full_xy"].shape[0]
        steps = np.arange(n_steps, dtype=int)
        missing_flag = np.zeros(n_steps, dtype=np.int32)
        if len(panel["miss_steps"]) > 0:
            missing_flag[panel["miss_steps"]] = 1

        wide_df = pd.DataFrame(
            {
                "segment_id": seg_id,
                "sample_index": int(panel["sample_index"]),
                "time_step": steps,
                "is_missing": missing_flag,
                "full_x": panel["full_xy"][:, 0],
                "full_y": panel["full_xy"][:, 1],
                "full_z": panel["full_xyz"][:, 2],
                "gt_x": panel["gt_xy"][:, 0],
                "gt_y": panel["gt_xy"][:, 1],
                "gt_z": panel["gt_xyz"][:, 2],
                "bilnn_x": panel["model_xy"]["BiLNN"][:, 0],
                "bilnn_y": panel["model_xy"]["BiLNN"][:, 1],
                "bilnn_z": panel["model_xyz"]["BiLNN"][:, 2],
                "bilstm_x": panel["model_xy"]["BiLSTM"][:, 0],
                "bilstm_y": panel["model_xy"]["BiLSTM"][:, 1],
                "bilstm_z": panel["model_xyz"]["BiLSTM"][:, 2],
                "hybrid_x": panel["model_xy"]["Hybrid"][:, 0],
                "hybrid_y": panel["model_xy"]["Hybrid"][:, 1],
                "hybrid_z": panel["model_xyz"]["Hybrid"][:, 2],
            }
        )
        wide_path = curve_data_dir / f"segment_{seg_id}_sample_{int(panel['sample_index'])}_wide.csv"
        wide_df.to_csv(wide_path, index=False)

        for curve_name, xy in [
            ("Full", panel["full_xyz"]),
            ("BiLNN", panel["model_xyz"]["BiLNN"]),
            ("BiLSTM", panel["model_xyz"]["BiLSTM"]),
            ("Hybrid", panel["model_xyz"]["Hybrid"]),
            ("GT", panel["gt_xyz"]),
        ]:
            long_df = pd.DataFrame(
                {
                    "segment_id": seg_id,
                    "sample_index": int(panel["sample_index"]),
                    "curve": curve_name,
                    "time_step": steps,
                    "is_missing": missing_flag,
                    "x": xy[:, 0],
                    "y": xy[:, 1],
                    "z": xy[:, 2],
                }
            )
            all_long_rows.extend(long_df.to_dict(orient="records"))

    all_segments_long_df = pd.DataFrame(all_long_rows)
    all_segments_long_path = curve_data_dir / "all_segments_curves_long.csv"
    all_segments_long_df.to_csv(all_segments_long_path, index=False)

    fig, axes = plt.subplots(1, 5, figsize=(27, 5.2), squeeze=False)
    for i, panel in enumerate(panel_data):
        ax = axes[0, i]
        ax.plot(panel["full_xy"][:, 0], panel["full_xy"][:, 1], color="black", linewidth=2.0, label="Full-IMU trajectory")
        ax.plot(panel["model_xy"]["BiLNN"][:, 0], panel["model_xy"]["BiLNN"][:, 1], color="#1f77b4", linewidth=1.55, label="BiLNN")
        ax.plot(panel["model_xy"]["BiLSTM"][:, 0], panel["model_xy"]["BiLSTM"][:, 1], color="#ff7f0e", linewidth=1.55, label="BiLSTM")
        ax.plot(panel["model_xy"]["Hybrid"][:, 0], panel["model_xy"]["Hybrid"][:, 1], color="#2ca02c", linewidth=1.55, label="Hybrid")
        miss_idx = panel["miss_steps"]
        if len(miss_idx) > 0:
            ax.scatter(
                panel["gt_xy"][miss_idx, 0],
                panel["gt_xy"][miss_idx, 1],
                color="#d62728",
                s=12,
                alpha=0.65,
                label="Missing steps",
            )
        ax.set_title(f"Segment {i+1} | sample={panel['sample_index']}")
        ax.set_xlabel("X Position (m)")
        if i == 0:
            ax.set_ylabel("Y Position (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="best", frameon=True, framealpha=0.9)

    fig.suptitle("5 Trajectory Segments (30% Random Missing -> Imputation -> Trajectory)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig_path = save_dir / f"five_segments_random30_comparison_{timestamp}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig3d = plt.figure(figsize=(27.5, 5.8))
    for i, panel in enumerate(panel_data):
        ax3d = fig3d.add_subplot(1, 5, i + 1, projection="3d")
        ax3d.plot(panel["full_xyz"][:, 0], panel["full_xyz"][:, 1], panel["full_xyz"][:, 2], color="black", linewidth=2.0, label="Full-IMU trajectory")
        ax3d.plot(panel["model_xyz"]["BiLNN"][:, 0], panel["model_xyz"]["BiLNN"][:, 1], panel["model_xyz"]["BiLNN"][:, 2], color="#1f77b4", linewidth=1.45, label="BiLNN")
        ax3d.plot(panel["model_xyz"]["BiLSTM"][:, 0], panel["model_xyz"]["BiLSTM"][:, 1], panel["model_xyz"]["BiLSTM"][:, 2], color="#ff7f0e", linewidth=1.45, label="BiLSTM")
        ax3d.plot(panel["model_xyz"]["Hybrid"][:, 0], panel["model_xyz"]["Hybrid"][:, 1], panel["model_xyz"]["Hybrid"][:, 2], color="#2ca02c", linewidth=1.45, label="Hybrid")
        miss_idx = panel["miss_steps"]
        if len(miss_idx) > 0:
            ax3d.scatter(
                panel["gt_xyz"][miss_idx, 0],
                panel["gt_xyz"][miss_idx, 1],
                panel["gt_xyz"][miss_idx, 2],
                color="#d62728",
                s=10,
                alpha=0.65,
                label="Missing steps",
            )
        ax3d.set_title(f"Segment {i+1} | sample={panel['sample_index']}")
        ax3d.set_xlabel("X (m)")
        if i == 0:
            ax3d.set_ylabel("Y (m)")
            ax3d.set_zlabel("Z (m)")
        pts = np.concatenate(
            [
                panel["full_xyz"],
                panel["model_xyz"]["BiLNN"],
                panel["model_xyz"]["BiLSTM"],
                panel["model_xyz"]["Hybrid"],
                panel["gt_xyz"],
            ],
            axis=0,
        )
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        centers = (mins + maxs) * 0.5
        radius = max(float(np.max(maxs - mins)) * 0.5, 1e-6)
        ax3d.set_xlim(centers[0] - radius, centers[0] + radius)
        ax3d.set_ylim(centers[1] - radius, centers[1] + radius)
        ax3d.set_zlim(centers[2] - radius, centers[2] + radius)
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.grid(True, alpha=0.25)
        ax3d.view_init(elev=22, azim=-55)
        if i == 0:
            ax3d.legend(loc="best", frameon=True, framealpha=0.9)

    fig3d.suptitle("5 Trajectory Segments 3D (30% Random Missing -> Imputation -> Trajectory)", fontsize=13, fontweight="bold")
    fig3d.tight_layout()
    fig3d_path = save_dir / f"five_segments_random30_comparison_3d_{timestamp}.png"
    fig3d.savefig(fig3d_path, dpi=300, bbox_inches="tight")
    plt.close(fig3d)

    metrics_df = pd.DataFrame(plot_rows)
    metrics_path = save_dir / f"five_segments_random30_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"[Saved] {fig_path}")
    print(f"[Saved] {fig3d_path}")
    print(f"[Saved] {metrics_path}")
    print(f"[Saved] {curve_data_dir}")
    print(f"[Saved] {all_segments_long_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5 segments demo under 30% random missing for BiLNN/BiLSTM/Hybrid")
    parser.add_argument("--output_dir", type=str, default="results/bidirectional_lnn_residual")
    parser.add_argument("--demo_output_dir", type=str, default="results/bidirectional_lnn_residual/demo_imputation")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--missing_mode", type=str, default="random", choices=["random", "block", "channel"])
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
        lnn_hidden=args.lnn_hidden,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
    )
