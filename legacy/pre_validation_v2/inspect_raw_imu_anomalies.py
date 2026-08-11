"""
Inspect raw IMU CSV files for jumps / spikes / timestamp anomalies.

Outputs:
- Per-file visualization PNG with anomaly markers
- CSV summary of anomaly counts and severity
- CSV of detailed anomaly events
- Console summary of suspicious files
"""
from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IMU_COLS = [
    "rotation_rate_x", "rotation_rate_y", "rotation_rate_z",
    "user_acc_x", "user_acc_y", "user_acc_z",
]
EXPECTED_COLS = ["Time", "att_roll", "att_pitch", "att_yaw",
                 "rotation_rate_x", "rotation_rate_y", "rotation_rate_z",
                 "grav_x", "grav_y", "grav_z",
                 "user_acc_x", "user_acc_y", "user_acc_z",
                 "mag_x", "mag_y", "mag_z"]


@dataclass
class FileAnomalySummary:
    file: str
    folder: str
    n_rows: int
    dt_median: float
    dt_max: float
    dt_anomaly_count: int
    jump_count: int
    spike_count: int
    flatline_count: int
    severity_score: float
    flagged: bool
    notes: str


def robust_zscore(x: np.ndarray) -> np.ndarray:
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median)) + 1e-9
    return (x - median) / (1.4826 * mad)


def load_imu_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] == len(EXPECTED_COLS):
        df.columns = EXPECTED_COLS
    else:
        df.columns = ["Time"] + [f"col_{i}" for i in range(df.shape[1] - 1)]
    return df


def detect_anomalies(df: pd.DataFrame, file_path: Path) -> tuple[FileAnomalySummary, pd.DataFrame]:
    if all(col in df.columns for col in IMU_COLS):
        imu = df[IMU_COLS].to_numpy(dtype=np.float64)
    else:
        imu = df.iloc[:, [4, 5, 6, 10, 11, 12]].to_numpy(dtype=np.float64)

    time = df["Time"].to_numpy(dtype=np.float64)
    dt = np.diff(time, prepend=time[0])
    dt_pos = dt[1:][np.isfinite(dt[1:])]
    dt_median = float(np.median(dt_pos)) if dt_pos.size else 0.0
    dt_mad = float(np.median(np.abs(dt_pos - dt_median))) + 1e-9 if dt_pos.size else 1e-9

    dt_anomaly_mask = np.zeros_like(dt, dtype=bool)
    if dt_pos.size:
        dt_anomaly_mask[1:] = (dt[1:] <= 0) | (np.abs(dt[1:] - dt_median) > max(10 * dt_mad, 5 * dt_median))

    diff = np.diff(imu, axis=0, prepend=imu[[0]])
    jump_mask = np.zeros(diff.shape[0], dtype=bool)
    spike_mask = np.zeros(diff.shape[0], dtype=bool)
    flatline_mask = np.zeros(diff.shape[0], dtype=bool)

    for ch in range(diff.shape[1]):
        dz = np.abs(robust_zscore(diff[:, ch]))
        xz = np.abs(robust_zscore(imu[:, ch]))
        jump_mask |= dz > 8.0
        spike_mask |= (xz > 10.0) & (dz > 5.0)

        eps = max(np.nanstd(imu[:, ch]) * 1e-4, 1e-8)
        const_run = np.abs(diff[:, ch]) < eps
        run_len = 0
        for i, v in enumerate(const_run):
            run_len = run_len + 1 if v else 0
            if run_len >= 20:
                flatline_mask[i - run_len + 1:i + 1] = True

    events: List[Dict] = []
    for idx in np.where(dt_anomaly_mask)[0]:
        events.append({
            "file": str(file_path),
            "index": int(idx),
            "time": float(time[idx]),
            "type": "dt_anomaly",
            "detail": float(dt[idx]),
        })
    for idx in np.where(jump_mask)[0]:
        events.append({
            "file": str(file_path),
            "index": int(idx),
            "time": float(time[idx]),
            "type": "jump",
            "detail": float(np.linalg.norm(diff[idx])),
        })
    for idx in np.where(spike_mask)[0]:
        events.append({
            "file": str(file_path),
            "index": int(idx),
            "time": float(time[idx]),
            "type": "spike",
            "detail": float(np.linalg.norm(imu[idx])),
        })
    for idx in np.where(flatline_mask)[0]:
        events.append({
            "file": str(file_path),
            "index": int(idx),
            "time": float(time[idx]),
            "type": "flatline",
            "detail": 1.0,
        })

    jump_count = int(jump_mask.sum())
    spike_count = int(spike_mask.sum())
    flatline_count = int(flatline_mask.sum())
    dt_anomaly_count = int(dt_anomaly_mask.sum())
    severity_score = float(jump_count + 1.5 * spike_count + 0.5 * flatline_count + 2.0 * dt_anomaly_count)
    flagged = severity_score > 0

    notes = []
    if dt_anomaly_count > 0:
        notes.append("timestamp异常")
    if jump_count > 0:
        notes.append("跳变")
    if spike_count > 0:
        notes.append("尖峰")
    if flatline_count > 0:
        notes.append("平直卡死")

    summary = FileAnomalySummary(
        file=str(file_path),
        folder=file_path.parent.name,
        n_rows=int(len(df)),
        dt_median=float(dt_median),
        dt_max=float(np.nanmax(dt)) if len(dt) else 0.0,
        dt_anomaly_count=dt_anomaly_count,
        jump_count=jump_count,
        spike_count=spike_count,
        flatline_count=flatline_count,
        severity_score=severity_score,
        flagged=flagged,
        notes=";".join(notes) if notes else "正常",
    )
    return summary, pd.DataFrame(events)


def plot_file(df: pd.DataFrame, events: pd.DataFrame, file_path: Path, output_dir: Path):
    if all(col in df.columns for col in IMU_COLS):
        imu = df[IMU_COLS].to_numpy(dtype=np.float64)
    else:
        imu = df.iloc[:, [4, 5, 6, 10, 11, 12]].to_numpy(dtype=np.float64)
    time = df["Time"].to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(3, 2, figsize=(18, 10), sharex=True)
    axes = axes.reshape(-1)
    jump_idx = events.loc[events["type"].isin(["jump", "spike"]), "index"].to_numpy(dtype=int) if not events.empty else np.array([], dtype=int)

    for i, col in enumerate(IMU_COLS):
        ax = axes[i]
        ax.plot(time, imu[:, i], linewidth=0.8, color="#1f77b4")
        if jump_idx.size > 0:
            valid = jump_idx[(jump_idx >= 0) & (jump_idx < len(time))]
            ax.scatter(time[valid], imu[valid, i], s=14, color="red", alpha=0.7, label="jump/spike")
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
        if i == 0 and jump_idx.size > 0:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Raw IMU inspection: {file_path.parent.name}/{file_path.name}", fontsize=14, fontweight="bold")
    fig.supxlabel("Time")
    plt.tight_layout()
    save_path = output_dir / f"{file_path.parent.name}_{file_path.stem}.png"
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def inspect_dataset(root_dir: Path, output_dir: Path, top_k_plots: int = 30):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    imu_files = sorted(Path(p) for p in glob.glob(str(root_dir / "**" / "imu*.csv"), recursive=True))
    if not imu_files:
        raise ValueError(f"No imu*.csv files found under {root_dir}")

    summaries: List[Dict] = []
    detail_frames: List[pd.DataFrame] = []
    plot_cache = []

    for file_path in imu_files:
        try:
            df = load_imu_csv(file_path)
            summary, detail_df = detect_anomalies(df, file_path)
            summaries.append(summary.__dict__)
            if not detail_df.empty:
                detail_frames.append(detail_df)
            plot_cache.append((summary.severity_score, df, detail_df, file_path))
            print(f"[Scanned] {file_path} | severity={summary.severity_score:.1f} | notes={summary.notes}")
        except Exception as exc:
            summaries.append(FileAnomalySummary(
                file=str(file_path),
                folder=file_path.parent.name,
                n_rows=0,
                dt_median=0.0,
                dt_max=0.0,
                dt_anomaly_count=0,
                jump_count=0,
                spike_count=0,
                flatline_count=0,
                severity_score=999.0,
                flagged=True,
                notes=f"读取失败:{exc}",
            ).__dict__)
            print(f"[Failed] {file_path}: {exc}")

    summary_df = pd.DataFrame(summaries).sort_values(by=["flagged", "severity_score"], ascending=[False, False])
    details_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame(columns=["file", "index", "time", "type", "detail"])

    summary_path = output_dir / "imu_anomaly_summary.csv"
    detail_path = output_dir / "imu_anomaly_details.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    details_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    for _, df, detail_df, file_path in sorted(plot_cache, key=lambda x: x[0], reverse=True)[:top_k_plots]:
        plot_file(df, detail_df, file_path, plot_dir)

    flagged_df = summary_df[summary_df["flagged"]].copy()
    report_path = output_dir / "imu_anomaly_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("IMU Raw Data Anomaly Inspection Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Scanned files: {len(summary_df)}\n")
        f.write(f"Flagged files: {len(flagged_df)}\n\n")
        if flagged_df.empty:
            f.write("No obvious anomaly files detected.\n")
        else:
            for _, row in flagged_df.iterrows():
                f.write(
                    f"{row['file']} | severity={row['severity_score']:.1f} | "
                    f"dt={row['dt_anomaly_count']} jump={row['jump_count']} "
                    f"spike={row['spike_count']} flatline={row['flatline_count']} | {row['notes']}\n"
                )

    print("\n" + "=" * 80)
    print("RAW IMU INSPECTION DONE")
    print(f"Scanned files: {len(summary_df)}")
    print(f"Flagged files: {len(flagged_df)}")
    print(f"[Saved] {summary_path}")
    print(f"[Saved] {detail_path}")
    print(f"[Saved] {report_path}")
    print(f"[Saved] plot dir: {plot_dir}")
    print("=" * 80)

    return summary_df, details_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect raw IMU CSV files for jumps and anomalies")
    parser.add_argument("--root_dir", type=str, default="Oxford Dataset")
    parser.add_argument("--output_dir", type=str, default="results/raw_imu_inspection")
    parser.add_argument("--top_k_plots", type=int, default=30)
    args = parser.parse_args()

    inspect_dataset(Path(args.root_dir), Path(args.output_dir), top_k_plots=args.top_k_plots)
