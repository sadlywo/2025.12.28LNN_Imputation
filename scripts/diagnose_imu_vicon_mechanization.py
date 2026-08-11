"""Diagnose OxIOD IMU/Vicon conventions before enabling physics loss.

This script never trains a model.  It propagates complete, unmasked IMU
windows from Vicon initial state and compares endpoint rotation, velocity, and
position under explicit convention hypotheses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation_v2.data.adapters import get_dataset_adapter
from validation_v2.evaluation.synchronization import synchronize_vicon_to_imu
from validation_v2.physics import propagate_imu, so3_log


FRAME_METADATA = {
    "quaternion_order": "xyzw",
    "quaternion_frame": "body_to_reference",
    "euler_order": "xyz",
}


def _window_metrics(
    gyro: np.ndarray,
    acc_g: np.ndarray,
    time_s: np.ndarray,
    position_m: np.ndarray,
    rotation: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    invert_rotation: bool,
    gyro_sign: float,
    acceleration_mode: str,
) -> dict[str, float]:
    dtype = torch.float64
    rotations = np.swapaxes(rotation, -1, -2) if invert_rotation else rotation
    dt = np.empty(len(time_s), dtype=np.float64)
    dt[1:] = np.diff(time_s)
    dt[0] = dt[1]
    propagated = propagate_imu(
        torch.as_tensor(gyro * gyro_sign, dtype=dtype)[None],
        torch.as_tensor(acc_g * 9.80665, dtype=dtype)[None],
        torch.as_tensor(dt, dtype=dtype)[None],
        torch.as_tensor(rotations[0], dtype=dtype)[None],
        torch.as_tensor(velocity_mps[0], dtype=dtype)[None],
        torch.as_tensor(position_m[0], dtype=dtype)[None],
        acceleration_mode=acceleration_mode,
    )
    rotation_residual = so3_log(
        torch.as_tensor(rotations[-1], dtype=dtype).transpose(-1, -2)
        @ propagated.rotation_body_to_world[0, -1]
    )
    return {
        "rotation_rad": float(torch.linalg.vector_norm(rotation_residual)),
        "velocity_mps": float(
            torch.linalg.vector_norm(
                propagated.velocity_world_mps[0, -1]
                - torch.as_tensor(velocity_mps[-1], dtype=dtype)
            )
        ),
        "position_m": float(
            torch.linalg.vector_norm(
                propagated.position_world_m[0, -1]
                - torch.as_tensor(position_m[-1], dtype=dtype)
            )
        ),
    }


def diagnose(
    data_root: Path,
    *,
    seq_len: int,
    max_recordings: int,
    windows_per_recording: int,
    max_median_dt_s: float,
) -> dict[str, object]:
    adapter = get_dataset_adapter("oxiod")
    pairs = list(adapter.discover(data_root))[:max_recordings]
    hypotheses = [
        {
            "name": f"{'world_to_body' if inverse else 'body_to_world'}__gyro_{sign:+.0f}__{mode}",
            "invert_rotation": inverse,
            "gyro_sign": sign,
            "acceleration_mode": mode,
        }
        for inverse in (False, True)
        for sign in (1.0, -1.0)
        for mode in ("gravity_compensated", "specific_force")
    ]
    rows: dict[str, list[dict[str, float]]] = {str(item["name"]): [] for item in hypotheses}
    recording_ids: list[str] = []
    skipped_recordings: list[dict[str, object]] = []
    for pair in pairs:
        recording = adapter.load(Path(pair["imu_path"]), Path(pair["vicon_path"]))
        median_dt_s = float(np.median(np.diff(recording.imu_time_s)))
        if median_dt_s > max_median_dt_s:
            skipped_recordings.append(
                {
                    "recording_id": recording.id,
                    "median_dt_s": median_dt_s,
                    "reason": "clean frame diagnostic excludes low-rate/large-interval streams",
                }
            )
            continue
        recording_ids.append(recording.id)
        overlap = (
            (recording.imu_time_s >= recording.overlap_s[0])
            & (recording.imu_time_s <= recording.overlap_s[1])
        )
        indices = np.flatnonzero(overlap)
        if len(indices) < seq_len:
            continue
        starts = np.linspace(
            0,
            len(indices) - seq_len,
            num=min(windows_per_recording, max(1, len(indices) // seq_len)),
            dtype=int,
        )
        for offset in np.unique(starts):
            selected = indices[offset : offset + seq_len]
            time_s = recording.imu_time_s[selected]
            synced = synchronize_vicon_to_imu(
                recording.vicon_time_s,
                recording.vicon_position_m,
                recording.vicon_quaternion_xyzw,
                time_s,
                frame_metadata=FRAME_METADATA,
            )
            for hypothesis in hypotheses:
                rows[str(hypothesis["name"])].append(
                    _window_metrics(
                        recording.imu_six[selected, :3],
                        recording.imu_six[selected, 3:],
                        time_s,
                        synced.position_m,
                        synced.rotation_body_to_world,
                        synced.velocity_world_mps,
                        invert_rotation=bool(hypothesis["invert_rotation"]),
                        gyro_sign=float(hypothesis["gyro_sign"]),
                        acceleration_mode=str(hypothesis["acceleration_mode"]),
                    )
                )
    summary = []
    for hypothesis in hypotheses:
        values = rows[str(hypothesis["name"])]
        if not values:
            continue
        summary.append(
            {
                **hypothesis,
                "windows": len(values),
                "mean_rotation_rad": float(np.mean([item["rotation_rad"] for item in values])),
                "mean_rotation_deg": float(np.degrees(np.mean([item["rotation_rad"] for item in values]))),
                "mean_velocity_mps": float(np.mean([item["velocity_mps"] for item in values])),
                "mean_position_m": float(np.mean([item["position_m"] for item in values])),
                "max_rotation_rad": float(np.max([item["rotation_rad"] for item in values])),
                "max_velocity_mps": float(np.max([item["velocity_mps"] for item in values])),
                "max_position_m": float(np.max([item["position_m"] for item in values])),
            }
        )
    summary.sort(
        key=lambda item: (
            item["mean_rotation_rad"], item["mean_velocity_mps"], item["mean_position_m"]
        )
    )
    return {
        "schema_version": 1,
        "dataset": "OxIOD",
        "recordings": recording_ids,
        "skipped_recordings": skipped_recordings,
        "seq_len": seq_len,
        "max_median_dt_s": max_median_dt_s,
        "semantics_from_schema": {
            "gyro": "rad/s body frame",
            "acceleration": "user_acc in G; documented as gravity removed",
            "quaternion_order": "xyzw",
            "time": "IMU seconds; Vicon nanoseconds converted to seconds",
        },
        "hypotheses": summary,
        "best_hypothesis": summary[0] if summary else None,
        "validation_status": "diagnostic_only_not_automatic_approval",
    }


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# Clean IMU/Vicon mechanization diagnostic",
        "",
        "This report is diagnostic evidence, not automatic frame validation.",
        "",
        "| Hypothesis | Windows | Rot deg | Vel m/s | Pos m |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["hypotheses"]:  # type: ignore[index]
        lines.append(
            f"| {item['name']} | {item['windows']} | {item['mean_rotation_deg']:.6f} | "
            f"{item['mean_velocity_mps']:.6f} | {item['mean_position_m']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Remaining checks before non-zero physics training: fixed IMU/Vicon extrinsic, "
            "axis signs, and whether Vicon rotation is the device body frame used by gyro.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "Oxford Dataset")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--max-recordings", type=int, default=3)
    parser.add_argument("--windows-per-recording", type=int, default=4)
    parser.add_argument("--max-median-dt-s", type=float, default=0.05)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "physics_loss_refactor" / "v1" / "diagnostics" / "clean_mechanization.json",
    )
    args = parser.parse_args()
    report = diagnose(
        args.data_root,
        seq_len=args.seq_len,
        max_recordings=args.max_recordings,
        windows_per_recording=args.windows_per_recording,
        max_median_dt_s=args.max_median_dt_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["best_hypothesis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
