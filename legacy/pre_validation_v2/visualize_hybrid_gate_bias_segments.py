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

from dataset import CfCIMUDataset
from experiment_bidirectional_lnn_residual import ResidualHybridBiLNNBiLSTM


def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = sorted(
        output_dir.glob("best_model_Hybrid_BiLNN_BiLSTM*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _infer_lstm_layers_from_state(state: Dict[str, torch.Tensor], prefix: str) -> int:
    layer_ids = set()
    key_prefix = f"{prefix}.lstm.weight_ih_l"
    for k in state.keys():
        if k.startswith(key_prefix):
            tail = k[len(key_prefix):]
            num = ""
            for ch in tail:
                if ch.isdigit():
                    num += ch
                else:
                    break
            if num:
                layer_ids.add(int(num))
    return max(layer_ids) + 1 if layer_ids else 2


def _build_hybrid_from_checkpoint(ckpt_path: Path, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    lnn_hidden = int(state["bi_lnn.head.0.weight"].shape[0])
    out_dim = int(state["bi_lnn.head.2.weight"].shape[0])
    feature_dim = out_dim
    if "bi_lnn.forward_lnn.lstm.input_map.weight" in state:
        residual_input_dim = int(state["bi_lnn.forward_lnn.lstm.input_map.weight"].shape[1])
    else:
        residual_input_dim = int(state["bi_lnn.forward_lnn.rnn_cell.ff1.weight"].shape[1])
    input_dim = residual_input_dim - feature_dim - 4
    lstm_hidden = int(state["bi_lstm.backbone.lstm.weight_hh_l0"].shape[1])
    lstm_layers = _infer_lstm_layers_from_state(state, prefix="bi_lstm.backbone")
    model = ResidualHybridBiLNNBiLSTM(
        input_dim=input_dim,
        feature_dim=feature_dim,
        lnn_hidden=lnn_hidden,
        lstm_hidden=lstm_hidden,
        output_dim=out_dim,
        lstm_layers=lstm_layers,
    )
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def _extract_demo_batch(dataset: CfCIMUDataset, index: int, device: torch.device):
    item = dataset[index]
    if len(item) < 3:
        raise ValueError("Dataset item does not contain enough fields.")
    inputs, target, mask = item[:3]
    return (
        inputs.unsqueeze(0).to(device),
        target.unsqueeze(0).to(device),
        mask.unsqueeze(0).to(device),
    )


def _select_sample_indices(total: int, n_segments: int) -> List[int]:
    if total <= n_segments:
        return list(range(total))
    idx = np.linspace(0, total - 1, n_segments)
    idx = np.round(idx).astype(int).tolist()
    dedup = []
    for i in idx:
        if i not in dedup:
            dedup.append(i)
    cur = 0
    while len(dedup) < n_segments and cur < total:
        if cur not in dedup:
            dedup.append(cur)
        cur += 1
    return sorted(dedup[:n_segments])


def run_visualization(
    checkpoint_dir: str = "results/Downstream_Tra_Plot",
    output_dir: str = "results/Downstream_Tra_Plot/gate_bias_visualization",
    seq_len: int = 30,
    mask_rate: float = 0.3,
    missing_mode: str = "random",
    n_segments: int = 5,
):
    _seed_all(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(checkpoint_dir)
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ckpt = _find_latest_checkpoint(ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No Hybrid checkpoint found in {ckpt_dir}")
    model = _build_hybrid_from_checkpoint(ckpt, device=device)

    dataset = CfCIMUDataset(
        root_dir="Oxford Dataset",
        seq_len=seq_len,
        mask_rate=mask_rate,
        missing_mode=missing_mode,
        split="test",
        split_ratio=0.8,
        val_ratio=0.1,
        eval_mode=True,
        include_window_features=True,
    )
    if len(dataset) == 0:
        raise ValueError("No test sample available for visualization.")

    sample_indices = _select_sample_indices(len(dataset), n_segments=n_segments)
    gate_rows: List[dict] = []
    segment_summary_rows: List[dict] = []
    segment_panels: List[dict] = []

    for seg_id, sample_idx in enumerate(sample_indices, start=1):
        inputs, _, mask = _extract_demo_batch(dataset, sample_idx, device)
        with torch.no_grad():
            comp = model.forward_with_components(inputs)
        gate = comp["gate"][0].detach().cpu().numpy()
        gate_mean_t = gate.mean(axis=1)
        gate_min_t = gate.min(axis=1)
        gate_max_t = gate.max(axis=1)
        miss_steps = np.where((mask[0].mean(dim=-1).detach().cpu().numpy() < 0.999))[0]
        miss_flag = np.zeros(gate.shape[0], dtype=np.int32)
        if len(miss_steps) > 0:
            miss_flag[miss_steps] = 1

        for t in range(gate.shape[0]):
            row = {
                "segment_id": seg_id,
                "sample_index": int(sample_idx),
                "time_step": int(t),
                "is_missing": int(miss_flag[t]),
                "gate_mean": float(gate_mean_t[t]),
                "lnn_bias": float(gate_mean_t[t]),
                "lstm_bias": float(1.0 - gate_mean_t[t]),
            }
            for ch in range(gate.shape[1]):
                row[f"gate_ch{ch + 1}"] = float(gate[t, ch])
            gate_rows.append(row)

        segment_summary_rows.append(
            {
                "segment_id": seg_id,
                "sample_index": int(sample_idx),
                "gate_mean": float(np.mean(gate_mean_t)),
                "lnn_bias": float(np.mean(gate_mean_t)),
                "lstm_bias": float(1.0 - np.mean(gate_mean_t)),
                "gate_std": float(np.std(gate_mean_t)),
                "missing_ratio": float(np.mean(miss_flag)),
            }
        )
        segment_panels.append(
            {
                "segment_id": seg_id,
                "sample_index": int(sample_idx),
                "gate": gate,
                "gate_mean_t": gate_mean_t,
                "gate_min_t": gate_min_t,
                "gate_max_t": gate_max_t,
                "miss_steps": miss_steps,
            }
        )

    gate_df = pd.DataFrame(gate_rows)
    summary_df = pd.DataFrame(segment_summary_rows)
    gate_csv = save_dir / f"hybrid_gate_timeseries_{timestamp}.csv"
    summary_csv = save_dir / f"hybrid_gate_segment_summary_{timestamp}.csv"
    gate_df.to_csv(gate_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    n = len(segment_panels)
    fig1, axes1 = plt.subplots(1, n, figsize=(5.5 * n, 4.2), squeeze=False, sharey=True)
    for i, p in enumerate(segment_panels):
        ax = axes1[0, i]
        t = np.arange(len(p["gate_mean_t"]))
        ax.plot(t, p["gate_mean_t"], color="#1f77b4", linewidth=1.6, label="Gate mean (LNN weight)")
        ax.fill_between(t, p["gate_min_t"], p["gate_max_t"], color="#1f77b4", alpha=0.18, label="Channel range")
        for m in p["miss_steps"]:
            ax.axvline(int(m), color="#d62728", alpha=0.08, linewidth=1.0)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Segment {p['segment_id']} | sample={p['sample_index']}")
        ax.set_xlabel("Time Step")
        if i == 0:
            ax.set_ylabel("Gate Value")
            ax.legend(loc="best", frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.25)
    fig1.suptitle("Hybrid Gate Branch Bias Over Time", fontsize=13, fontweight="bold")
    fig1.tight_layout()
    fig1_path = save_dir / f"hybrid_gate_bias_timeseries_{timestamp}.png"
    fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, axes2 = plt.subplots(n, 1, figsize=(11.0, 1.8 * n + 1.0), squeeze=False)
    for i, p in enumerate(segment_panels):
        ax = axes2[i, 0]
        im = ax.imshow(p["gate"].T, aspect="auto", cmap="coolwarm", vmin=0.0, vmax=1.0)
        ax.set_title(f"Segment {p['segment_id']} | sample={p['sample_index']} | Channel-wise Gate Heatmap")
        ax.set_ylabel("IMU Channel")
        if i == n - 1:
            ax.set_xlabel("Time Step")
        ax.set_yticks(np.arange(p["gate"].shape[1]))
        ax.set_yticklabels([f"ch{j + 1}" for j in range(p["gate"].shape[1])])
    cbar = fig2.colorbar(im, ax=axes2[:, 0], fraction=0.02, pad=0.02)
    cbar.set_label("LNN branch weight")
    fig2.tight_layout()
    fig2_path = save_dir / f"hybrid_gate_bias_heatmap_{timestamp}.png"
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(1, 1, figsize=(9.2, 4.6))
    x = np.arange(len(summary_df))
    bw = 0.38
    ax3.bar(x - bw / 2, summary_df["lnn_bias"], width=bw, color="#1f77b4", alpha=0.9, label="LNN branch")
    ax3.bar(x + bw / 2, summary_df["lstm_bias"], width=bw, color="#ff7f0e", alpha=0.9, label="BiLSTM branch")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"S{sid}" for sid in summary_df["segment_id"].tolist()])
    ax3.set_ylim(0.0, 1.0)
    ax3.set_ylabel("Average Branch Weight")
    ax3.set_xlabel("Segment")
    ax3.set_title("Average Branch Bias per Segment")
    ax3.grid(True, axis="y", alpha=0.25)
    ax3.legend(loc="best", frameon=True, framealpha=0.9)
    fig3.tight_layout()
    fig3_path = save_dir / f"hybrid_gate_bias_segment_bar_{timestamp}.png"
    fig3.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close(fig3)

    print(f"[Checkpoint] {ckpt}")
    print(f"[Saved] {gate_csv}")
    print(f"[Saved] {summary_csv}")
    print(f"[Saved] {fig1_path}")
    print(f"[Saved] {fig2_path}")
    print(f"[Saved] {fig3_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize branch bias changes of Hybrid gate unit over multiple segments")
    parser.add_argument("--checkpoint_dir", type=str, default="results/Downstream_Tra_Plot")
    parser.add_argument("--output_dir", type=str, default="results/Downstream_Tra_Plot/gate_bias_visualization")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--missing_mode", type=str, default="random", choices=["random", "block", "channel"])
    parser.add_argument("--n_segments", type=int, default=5)
    args = parser.parse_args()
    run_visualization(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        mask_rate=args.mask_rate,
        missing_mode=args.missing_mode,
        n_segments=args.n_segments,
    )
