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
import torch.nn as nn

from dataset import CfCIMUDataset, compute_ate


def _extract_parts(inputs: torch.Tensor, feature_dim: int):
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

    def forward(self, inputs: torch.Tensor):
        x_masked, mask, _ = _extract_parts(inputs, self.feature_dim)
        if self.method_name == "LOCF":
            pred = self._locf_impute(x_masked, mask)
        elif self.method_name == "MICE":
            pred = self._mice_impute(x_masked, mask)
        else:
            raise ValueError(f"Unsupported deterministic baseline: {self.method_name}")
        uncertainty = torch.ones_like(pred) * 0.1
        return pred, uncertainty


class GRUImputer(nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 128, output_dim: int = 6):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor):
        h, _ = self.rnn(x)
        pred = self.head(h)
        uncert = self.uncertainty_head(h)
        return pred, uncert


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x[:, :, -1]
        t = torch.cumsum(dt, dim=1)
        B, T = t.shape
        device = t.device
        pe = torch.zeros(B, T, self.d_model, device=device)
        position = t.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device) * (-np.log(10000.0) / self.d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class TransformerImputer(nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 128, output_dim: int = 6, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.posenc = PositionalEncoding(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor):
        z = self.input_proj(x)
        z = z + self.posenc(x)
        h = self.encoder(z)
        pred = self.head(h)
        uncert = self.uncertainty_head(h)
        return pred, uncert


def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_latest_checkpoint(output_dir: Path, model_alias: str) -> Path | None:
    candidates = sorted(
        output_dir.glob(f"best_model_{model_alias}*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


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


def _build_models(input_dim: int, feature_dim: int, hidden_dim: int, nhead: int, nlayers: int):
    return {
        "GRU": GRUImputer(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=6),
        "Transformer": TransformerImputer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=6,
            nhead=nhead,
            nlayers=nlayers,
        ),
        "LOCF": DeterministicUncertaintyWrapper("LOCF", feature_dim=feature_dim),
        "MICE": DeterministicUncertaintyWrapper("MICE", feature_dim=feature_dim),
    }


def _infer_transformer_layers(state: dict, default_layers: int) -> int:
    idxs = []
    for k in state.keys():
        if k.startswith("encoder.layers."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                idxs.append(int(parts[2]))
    return (max(idxs) + 1) if idxs else default_layers


def run_demo(
    output_dir: str = "results/Downstream_Tra_Plot",
    demo_output_dir: str = "results/Downstream_Tra_Plot/demo_downstream_baselines",
    seq_len: int = 30,
    mask_rate: float = 0.3,
    missing_mode: str = "random",
    hidden_dim: int = 128,
    transformer_nhead: int = 4,
    transformer_nlayers: int = 2,
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
        hidden_dim=hidden_dim,
        nhead=transformer_nhead,
        nlayers=transformer_nlayers,
    )

    ckpt_aliases = {
        "GRU": ["GRU", "gru_train_block", "gru"],
        "Transformer": ["Transformer", "transformer_train_block", "transformer"],
    }
    loaded_models: Dict[str, torch.nn.Module] = {}
    model_input_dims: Dict[str, int] = {}
    for model_name, model in models.items():
        if model_name in ["LOCF", "MICE"]:
            loaded_models[model_name] = model.to(device).eval()
            model_input_dims[model_name] = dataset.input_dim
            continue
        ckpt = None
        for alias in ckpt_aliases[model_name]:
            ckpt = _find_latest_checkpoint(ckpt_dir, alias)
            if ckpt is not None:
                break
        if ckpt is None:
            found = sorted([p.name for p in ckpt_dir.glob("best_model_*.pt")])
            raise FileNotFoundError(f"Missing checkpoint for {model_name} in {ckpt_dir}. Found: {found}")
        state = torch.load(ckpt, map_location=device)
        if model_name == "GRU":
            in_dim = int(state["rnn.weight_ih_l0"].shape[1])
            h_dim = int(state["rnn.weight_hh_l0"].shape[1])
            model = GRUImputer(input_dim=in_dim, hidden_dim=h_dim, output_dim=6)
        else:
            in_dim = int(state["input_proj.weight"].shape[1])
            h_dim = int(state["input_proj.weight"].shape[0])
            n_layers = _infer_transformer_layers(state, transformer_nlayers)
            model = TransformerImputer(
                input_dim=in_dim,
                hidden_dim=h_dim,
                output_dim=6,
                nhead=transformer_nhead,
                nlayers=n_layers,
            )
        model = model.to(device)
        model.load_state_dict(state)
        model.eval()
        loaded_models[model_name] = model
        model_input_dims[model_name] = in_dim

    sample_indices = _select_five_indices(len(dataset))
    curve_data_dir = save_dir / f"five_segments_curve_data_{timestamp}"
    curve_data_dir.mkdir(parents=True, exist_ok=True)
    panel_data: List[dict] = []
    metric_rows: List[dict] = []

    for sample_idx in sample_indices:
        inputs, target, mask, stats, vicon = _extract_demo_batch(dataset, sample_idx, device)
        gt_pos = vicon[:, :, :3]
        dt = inputs[:, :, dataset.feature_dim * 2]
        xy_axes = _choose_xy_axes_by_span(gt_pos[0].detach().cpu().numpy())

        full_result = compute_ate(target, gt_pos, dt, stats=stats)
        full_xyz = _to_xyz(full_result["pred_trajectory"][0])
        method_xyz: Dict[str, np.ndarray] = {}
        method_xy: Dict[str, np.ndarray] = {}
        for method_name, model in loaded_models.items():
            cur_in_dim = int(model_input_dims.get(method_name, inputs.shape[-1]))
            model_inputs = inputs[:, :, :cur_in_dim] if cur_in_dim < inputs.shape[-1] else inputs
            with torch.no_grad():
                pred, _ = model(model_inputs)
            ate = compute_ate(pred, gt_pos, dt, stats=stats)
            pred_xyz = _to_xyz(ate["pred_trajectory"][0])
            method_xyz[method_name] = pred_xyz
            method_xy[method_name] = pred_xyz[:, [xy_axes[0], xy_axes[1]]]
            metric_rows.append(
                {
                    "sample_index": sample_idx,
                    "method": method_name,
                    "ate": float(ate["ate"]),
                    "full_imu_ate": float(full_result["ate"]),
                    "ate_delta_vs_full": float(ate["ate"]) - float(full_result["ate"]),
                }
            )

        panel_data.append(
            {
                "sample_index": sample_idx,
                "full_xyz": full_xyz,
                "full_xy": full_xyz[:, [xy_axes[0], xy_axes[1]]],
                "method_xyz": method_xyz,
                "method_xy": method_xy,
                "miss_steps": np.where((mask[0].mean(dim=-1).detach().cpu().numpy() < 0.999))[0],
            }
        )

    all_long_rows: List[dict] = []
    for seg_id, panel in enumerate(panel_data, start=1):
        n_steps = panel["full_xyz"].shape[0]
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
                "full_x": panel["full_xyz"][:, 0],
                "full_y": panel["full_xyz"][:, 1],
                "full_z": panel["full_xyz"][:, 2],
                "gru_x": panel["method_xyz"]["GRU"][:, 0],
                "gru_y": panel["method_xyz"]["GRU"][:, 1],
                "gru_z": panel["method_xyz"]["GRU"][:, 2],
                "transformer_x": panel["method_xyz"]["Transformer"][:, 0],
                "transformer_y": panel["method_xyz"]["Transformer"][:, 1],
                "transformer_z": panel["method_xyz"]["Transformer"][:, 2],
                "locf_x": panel["method_xyz"]["LOCF"][:, 0],
                "locf_y": panel["method_xyz"]["LOCF"][:, 1],
                "locf_z": panel["method_xyz"]["LOCF"][:, 2],
                "mice_x": panel["method_xyz"]["MICE"][:, 0],
                "mice_y": panel["method_xyz"]["MICE"][:, 1],
                "mice_z": panel["method_xyz"]["MICE"][:, 2],
            }
        )
        wide_path = curve_data_dir / f"segment_{seg_id}_sample_{int(panel['sample_index'])}_wide.csv"
        wide_df.to_csv(wide_path, index=False)

        for curve_name, xyz in [
            ("Full", panel["full_xyz"]),
            ("GRU", panel["method_xyz"]["GRU"]),
            ("Transformer", panel["method_xyz"]["Transformer"]),
            ("LOCF", panel["method_xyz"]["LOCF"]),
            ("MICE", panel["method_xyz"]["MICE"]),
        ]:
            long_df = pd.DataFrame(
                {
                    "segment_id": seg_id,
                    "sample_index": int(panel["sample_index"]),
                    "curve": curve_name,
                    "time_step": steps,
                    "is_missing": missing_flag,
                    "x": xyz[:, 0],
                    "y": xyz[:, 1],
                    "z": xyz[:, 2],
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
        ax.plot(panel["method_xy"]["GRU"][:, 0], panel["method_xy"]["GRU"][:, 1], color="#1f77b4", linewidth=1.4, label="GRU")
        ax.plot(panel["method_xy"]["Transformer"][:, 0], panel["method_xy"]["Transformer"][:, 1], color="#ff7f0e", linewidth=1.4, label="Transformer")
        ax.plot(panel["method_xy"]["LOCF"][:, 0], panel["method_xy"]["LOCF"][:, 1], color="#2ca02c", linewidth=1.4, label="LOCF")
        ax.plot(panel["method_xy"]["MICE"][:, 0], panel["method_xy"]["MICE"][:, 1], color="#9467bd", linewidth=1.4, label="MICE")
        if len(panel["miss_steps"]) > 0:
            ax.scatter(
                panel["full_xy"][panel["miss_steps"], 0],
                panel["full_xy"][panel["miss_steps"], 1],
                color="#d62728",
                s=10,
                alpha=0.6,
                label="Missing steps",
            )
        ax.set_title(f"Segment {i + 1} | sample={panel['sample_index']}")
        ax.set_xlabel("X Position (m)")
        if i == 0:
            ax.set_ylabel("Y Position (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="best", frameon=True, framealpha=0.9)
    fig.suptitle("5 Segments: Downstream Trajectory Impact (GRU/Transformer/LOCF/MICE)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig_path = save_dir / f"downstream_baselines_2d_{timestamp}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig3d = plt.figure(figsize=(27.5, 5.8))
    for i, panel in enumerate(panel_data):
        ax3d = fig3d.add_subplot(1, 5, i + 1, projection="3d")
        ax3d.plot(panel["full_xyz"][:, 0], panel["full_xyz"][:, 1], panel["full_xyz"][:, 2], color="black", linewidth=2.0, label="Full-IMU trajectory")
        ax3d.plot(panel["method_xyz"]["GRU"][:, 0], panel["method_xyz"]["GRU"][:, 1], panel["method_xyz"]["GRU"][:, 2], color="#1f77b4", linewidth=1.3, label="GRU")
        ax3d.plot(panel["method_xyz"]["Transformer"][:, 0], panel["method_xyz"]["Transformer"][:, 1], panel["method_xyz"]["Transformer"][:, 2], color="#ff7f0e", linewidth=1.3, label="Transformer")
        ax3d.plot(panel["method_xyz"]["LOCF"][:, 0], panel["method_xyz"]["LOCF"][:, 1], panel["method_xyz"]["LOCF"][:, 2], color="#2ca02c", linewidth=1.3, label="LOCF")
        ax3d.plot(panel["method_xyz"]["MICE"][:, 0], panel["method_xyz"]["MICE"][:, 1], panel["method_xyz"]["MICE"][:, 2], color="#9467bd", linewidth=1.3, label="MICE")
        pts = np.concatenate(
            [
                panel["full_xyz"],
                panel["method_xyz"]["GRU"],
                panel["method_xyz"]["Transformer"],
                panel["method_xyz"]["LOCF"],
                panel["method_xyz"]["MICE"],
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
        ax3d.set_title(f"Segment {i + 1} | sample={panel['sample_index']}")
        ax3d.set_xlabel("X (m)")
        if i == 0:
            ax3d.set_ylabel("Y (m)")
            ax3d.set_zlabel("Z (m)")
            ax3d.legend(loc="best", frameon=True, framealpha=0.9)
    fig3d.suptitle("5 Segments 3D: Downstream Trajectory Impact (GRU/Transformer/LOCF/MICE)", fontsize=13, fontweight="bold")
    fig3d.tight_layout()
    fig3d_path = save_dir / f"downstream_baselines_3d_{timestamp}.png"
    fig3d.savefig(fig3d_path, dpi=300, bbox_inches="tight")
    plt.close(fig3d)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = save_dir / f"downstream_baselines_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"[Saved] {fig_path}")
    print(f"[Saved] {fig3d_path}")
    print(f"[Saved] {metrics_path}")
    print(f"[Saved] {curve_data_dir}")
    print(f"[Saved] {all_segments_long_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5 segments downstream trajectory demo for GRU/Transformer/LOCF/MICE")
    parser.add_argument("--output_dir", type=str, default="results/Downstream_Tra_Plot")
    parser.add_argument("--demo_output_dir", type=str, default="results/Downstream_Tra_Plot/demo_downstream_baselines")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--missing_mode", type=str, default="random", choices=["random", "block", "channel"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--transformer_nhead", type=int, default=4)
    parser.add_argument("--transformer_nlayers", type=int, default=2)
    args = parser.parse_args()
    run_demo(
        output_dir=args.output_dir,
        demo_output_dir=args.demo_output_dir,
        seq_len=args.seq_len,
        mask_rate=args.mask_rate,
        missing_mode=args.missing_mode,
        hidden_dim=args.hidden_dim,
        transformer_nhead=args.transformer_nhead,
        transformer_nlayers=args.transformer_nlayers,
    )
