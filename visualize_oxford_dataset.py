"""Visualize first-trial XY trajectories for all Oxford Dataset subfolders."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


def _style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "axes.titlesize": 9.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.4,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _format_name(folder_name: str) -> str:
    return folder_name.replace("-", " ").title()


def _scan_first_vi_files(root_dir: Path) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    for folder in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        vi_files = sorted(folder.glob("vi*.csv"))
        if vi_files:
            items.append((_format_name(folder.name), vi_files[0]))
    return items


def _read_vi_xy(vi_path: Path):
    # Preferred path: read with header and use named translation columns if present.
    try:
        df_h = pd.read_csv(vi_path)
        cols_lower = {str(c).strip().lower(): c for c in df_h.columns}
        if "translation.x" in cols_lower and "translation.y" in cols_lower:
            x = pd.to_numeric(df_h[cols_lower["translation.x"]], errors="coerce").to_numpy(dtype=np.float64)
            y = pd.to_numeric(df_h[cols_lower["translation.y"]], errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(x) & np.isfinite(y)
            return x[valid], y[valid]
    except Exception:
        pass

    # Fallback path for headerless files:
    # vi format from README: [Time, Header, translation.x, translation.y, translation.z, ...]
    df = pd.read_csv(vi_path, header=None)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.shape[1] >= 4:
        x = numeric.iloc[:, 2].to_numpy(dtype=np.float64)
        y = numeric.iloc[:, 3].to_numpy(dtype=np.float64)
    elif numeric.shape[1] >= 3:
        x = numeric.iloc[:, 1].to_numpy(dtype=np.float64)
        y = numeric.iloc[:, 2].to_numpy(dtype=np.float64)
    else:
        x = np.zeros(len(df), dtype=np.float64)
        y = np.zeros(len(df), dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    return x, y


def _read_vi_txy(vi_path: Path):
    try:
        df_h = pd.read_csv(vi_path)
        cols_lower = {str(c).strip().lower(): c for c in df_h.columns}
        if "translation.x" in cols_lower and "translation.y" in cols_lower:
            t_col = cols_lower["time"] if "time" in cols_lower else df_h.columns[0]
            t = pd.to_numeric(df_h[t_col], errors="coerce").to_numpy(dtype=np.float64)
            x = pd.to_numeric(df_h[cols_lower["translation.x"]], errors="coerce").to_numpy(dtype=np.float64)
            y = pd.to_numeric(df_h[cols_lower["translation.y"]], errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
            return t[valid], x[valid], y[valid]
    except Exception:
        pass

    df = pd.read_csv(vi_path, header=None)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    t = numeric.iloc[:, 0].to_numpy(dtype=np.float64) if numeric.shape[1] >= 1 else np.arange(len(df), dtype=np.float64)
    x = numeric.iloc[:, 2].to_numpy(dtype=np.float64) if numeric.shape[1] >= 3 else np.zeros(len(df), dtype=np.float64)
    y = numeric.iloc[:, 3].to_numpy(dtype=np.float64) if numeric.shape[1] >= 4 else np.zeros(len(df), dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    return t[valid], x[valid], y[valid]


def _compute_trajectory_features(t: np.ndarray, x: np.ndarray, y: np.ndarray):
    if len(x) < 2:
        return {
            "duration_sec": 0.0,
            "path_length_m": 0.0,
            "speed_mean_mps": 0.0,
            "speed_p95_mps": 0.0,
            "speed_max_mps": 0.0,
            "time_norm": np.array([0.0]),
            "cum_dist": np.array([0.0]),
            "speed": np.array([0.0]),
        }

    dt = np.diff(t)
    if np.nanmedian(np.abs(dt)) > 1e6:
        dt = dt / 1e9
    dt = np.clip(dt, 1e-4, None)

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx * dx + dy * dy)
    speed = ds / dt
    speed = np.clip(speed, 0.0, np.percentile(speed, 99.5) if len(speed) > 10 else np.max(speed))
    cum_dist = np.concatenate([[0.0], np.cumsum(ds)])
    duration_sec = float(np.sum(dt))
    time_norm = np.linspace(0.0, 1.0, len(cum_dist))

    return {
        "duration_sec": duration_sec,
        "path_length_m": float(np.sum(ds)),
        "speed_mean_mps": float(np.mean(speed)),
        "speed_p95_mps": float(np.percentile(speed, 95)),
        "speed_max_mps": float(np.max(speed)),
        "time_norm": time_norm,
        "cum_dist": cum_dist,
        "speed": speed,
    }


def _prepare_xy(x: np.ndarray, y: np.ndarray, max_points: int = 1500):
    if len(x) <= max_points:
        return x, y
    stride = max(1, int(np.ceil(len(x) / max_points)))
    return x[::stride], y[::stride]


def _prepare_txy(t: np.ndarray, x: np.ndarray, y: np.ndarray, max_points: int = 1500):
    if len(x) <= max_points:
        return t, x, y
    stride = max(1, int(np.ceil(len(x) / max_points)))
    return t[::stride], x[::stride], y[::stride]


def plot_first_vi_trajectories(items: List[Tuple[str, Path]], save_path: Path):
    n = len(items)
    n_cols = min(4, max(1, n))
    n_rows = int(math.ceil(n / n_cols))
    fig_width = 7.2
    fig_height = max(3.6, 1.75 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False, constrained_layout=False)

    prepared = []
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    for scenario_name, vi_path in items:
        x, y = _read_vi_xy(vi_path)
        x, y = _prepare_xy(x, y, max_points=1500)
        prepared.append((scenario_name, x, y))
        if len(x) > 0:
            x_min = min(x_min, float(np.nanmin(x)))
            x_max = max(x_max, float(np.nanmax(x)))
            y_min = min(y_min, float(np.nanmin(y)))
            y_max = max(y_max, float(np.nanmax(y)))

    if not np.isfinite([x_min, x_max, y_min, y_max]).all():
        x_min, x_max, y_min, y_max = -1.0, 1.0, -1.0, 1.0

    x_span = float(x_max - x_min)
    y_span = float(y_max - y_min)
    span = max(x_span, y_span, 1e-6)
    pad = max(0.03, 0.025 * span)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    half = 0.5 * span + pad
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, n)))

    for i, (scenario_name, x, y) in enumerate(prepared):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]
        ax.set_facecolor("#fcfcfc")
        if len(x) > 0:
            ax.plot(x, y, color=colors[i], alpha=0.92, linewidth=1.5)
            ax.scatter(x[0], y[0], color="#2ca02c", s=10, zorder=3)
            ax.scatter(x[-1], y[-1], color="#d62728", s=12, marker="x", zorder=3)
        ax.set_title(scenario_name)
        ax.set_xlim(x_center - half, x_center + half)
        ax.set_ylim(y_center - half, y_center + half)
        if r < n_rows - 1:
            ax.set_xticklabels([])
        if c > 0:
            ax.set_yticklabels([])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.20, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n, n_rows * n_cols):
        r = j // n_cols
        c = j % n_cols
        axes[r, c].axis("off")

    fig.suptitle("Oxford Dataset First-Trial Trajectories", fontsize=11.2, fontweight="bold", y=0.965)
    fig.supxlabel("X Position (m)", fontsize=10)
    fig.supylabel("Y Position (m)", fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.998, bottom=0.09, top=0.86, wspace=0.01, hspace=0.18)
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_speed_colored_trajectories(items: List[Tuple[str, Path]], save_path: Path):
    n = len(items)
    n_cols = min(4, max(1, n))
    n_rows = int(math.ceil(n / n_cols))
    fig_width = 7.2
    fig_height = max(3.6, 1.75 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False, constrained_layout=False)

    vmin = np.inf
    vmax = -np.inf
    prepared = []
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    for scenario_name, vi_path in items:
        t, x, y = _read_vi_txy(vi_path)
        t, x, y = _prepare_txy(t, x, y, max_points=1800)
        feat = _compute_trajectory_features(t, x, y)
        speed = feat["speed"]
        prepared.append((scenario_name, x, y, speed))
        if len(speed) > 0:
            vmin = min(vmin, float(np.min(speed)))
            vmax = max(vmax, float(np.max(speed)))
        if len(x) > 0:
            x_min = min(x_min, float(np.min(x)))
            x_max = max(x_max, float(np.max(x)))
            y_min = min(y_min, float(np.min(y)))
            y_max = max(y_max, float(np.max(y)))

    if not np.isfinite([x_min, x_max, y_min, y_max]).all():
        x_min, x_max, y_min, y_max = -1.0, 1.0, -1.0, 1.0
    if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
        vmin, vmax = 0.0, 1.0

    span = max(x_max - x_min, y_max - y_min, 1e-6)
    pad = max(0.03, 0.025 * span)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    half = 0.5 * span + pad

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    for i, (scenario_name, x, y, speed) in enumerate(prepared):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]
        ax.set_facecolor("#fcfcfc")
        if len(x) > 1:
            pts = np.column_stack([x, y]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap="viridis", norm=norm, linewidth=1.35, alpha=0.95)
            lc.set_array(speed[: len(segs)])
            ax.add_collection(lc)
            ax.scatter(
                x[0], y[0],
                marker="o", s=46,
                facecolor="#39d353", edgecolor="black", linewidth=0.65,
                zorder=5,
            )
            ax.scatter(
                x[-1], y[-1],
                marker="X", s=64,
                facecolor="#ff3b30", edgecolor="white", linewidth=0.9,
                zorder=6,
            )
        ax.set_title(scenario_name)
        ax.set_xlim(x_center - half, x_center + half)
        ax.set_ylim(y_center - half, y_center + half)
        if r < n_rows - 1:
            ax.set_xticklabels([])
        if c > 0:
            ax.set_yticklabels([])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            handles = [
                Line2D([0], [0], marker="o", color="none", markerfacecolor="#39d353", markeredgecolor="black", markersize=6, label="Start"),
                Line2D([0], [0], marker="X", color="none", markerfacecolor="#ff3b30", markeredgecolor="white", markersize=7, label="End"),
            ]

    for j in range(n, n_rows * n_cols):
        r = j // n_cols
        c = j % n_cols
        axes[r, c].axis("off")

    fig.suptitle("Speed-Colored Trajectories by Scenario", fontsize=11.2, fontweight="bold", y=0.965)
    fig.supxlabel("X Position (m)", fontsize=10)
    fig.supylabel("Y Position (m)", fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.935, bottom=0.09, top=0.86, wspace=0.01, hspace=0.18)
    cax = fig.add_axes([0.945, 0.18, 0.012, 0.62])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("Speed (m/s)")
    fig.legend(
        handles=handles,
        labels=["Start", "End"],
        loc="upper left",
        bbox_to_anchor=(0.012, 0.975),
        frameon=True,
        framealpha=0.92,
        borderpad=0.3,
        handletextpad=0.4,
    )
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_scenario_feature_panels(summary_df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=False)
    ax0, ax1, ax2 = axes

    order = summary_df.sort_values("speed_mean_mps", ascending=False)["scenario"].tolist()
    df = summary_df.set_index("scenario").loc[order].reset_index()
    x = np.arange(len(df))

    ax0.bar(x, df["speed_mean_mps"], color="#4C72B0", alpha=0.88)
    ax0.set_title("Mean Speed")
    ax0.set_ylabel("m/s")
    ax0.set_xticks(x)
    ax0.set_xticklabels(df["scenario"], rotation=35, ha="right")
    ax0.grid(True, axis="y", alpha=0.2)

    ax1.bar(x, df["path_length_m"], color="#55A868", alpha=0.88)
    ax1.set_title("Path Length")
    ax1.set_ylabel("m")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["scenario"], rotation=35, ha="right")
    ax1.grid(True, axis="y", alpha=0.2)

    ax2.bar(x, df["duration_sec"], color="#C44E52", alpha=0.88)
    ax2.set_title("Duration")
    ax2.set_ylabel("s")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["scenario"], rotation=35, ha="right")
    ax2.grid(True, axis="y", alpha=0.2)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Scenario-Level Motion Statistics", fontsize=11.2, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.30, top=0.84, wspace=0.30)
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def main():
    _style()
    root_dir = Path("Oxford Dataset")
    output_dir = Path("results/oxford_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)

    first_vi_items = _scan_first_vi_files(root_dir)
    if not first_vi_items:
        raise ValueError(f"No valid vi files found under {root_dir}")

    summary = pd.DataFrame(
        [
            {"scenario": scenario, "vi_file": str(vi_path)}
            for scenario, vi_path in first_vi_items
        ]
    )
    feature_rows = []
    for scenario, vi_path in first_vi_items:
        t, x, y = _read_vi_txy(vi_path)
        feat = _compute_trajectory_features(t, x, y)
        feature_rows.append(
            {
                "scenario": scenario,
                "vi_file": str(vi_path),
                "duration_sec": feat["duration_sec"],
                "path_length_m": feat["path_length_m"],
                "speed_mean_mps": feat["speed_mean_mps"],
                "speed_p95_mps": feat["speed_p95_mps"],
                "speed_max_mps": feat["speed_max_mps"],
            }
        )
    summary_features = pd.DataFrame(feature_rows)
    summary_all = summary.merge(summary_features, on=["scenario", "vi_file"], how="left")

    summary_path = output_dir / "first_vi_summary.csv"
    fig_path = output_dir / "oxford_first_vi_trajectories.png"
    speed_fig_path = output_dir / "oxford_speed_colored_trajectories.png"
    stats_fig_path = output_dir / "oxford_scenario_motion_stats.png"

    plot_first_vi_trajectories(first_vi_items, fig_path)
    plot_speed_colored_trajectories(first_vi_items, speed_fig_path)
    plot_scenario_feature_panels(summary_features, stats_fig_path)

    summary_all.to_csv(summary_path, index=False)

    print(f"[Saved] Trajectory figure: {fig_path}")
    print(f"[Saved] Speed-colored figure: {speed_fig_path}")
    print(f"[Saved] Motion stats figure: {stats_fig_path}")
    print(f"[Saved] Scenario summary: {summary_path}")


if __name__ == "__main__":
    main()
