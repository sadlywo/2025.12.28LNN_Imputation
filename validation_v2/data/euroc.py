"""EuRoC MAV six-axis IMU and state-ground-truth loader."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from validation_v2.types import Recording

from .oxiod import overlapping_interval


EUROC_IMU_CHANNEL_NAMES = (
    "w_RS_S_x [rad s^-1]",
    "w_RS_S_y [rad s^-1]",
    "w_RS_S_z [rad s^-1]",
    "a_RS_S_x [m s^-2]",
    "a_RS_S_y [m s^-2]",
    "a_RS_S_z [m s^-2]",
)
EUROC_POSITION_NAMES = (
    "p_RS_R_x [m]",
    "p_RS_R_y [m]",
    "p_RS_R_z [m]",
)
EUROC_QUATERNION_WXYZ_NAMES = (
    "q_RS_w []",
    "q_RS_x []",
    "q_RS_y []",
    "q_RS_z []",
)


def _read_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"EuRoC CSV does not exist: {path}")
    frame = pd.read_csv(path)
    frame.columns = [str(name).strip().lstrip("#").strip() for name in frame.columns]
    return frame


def _ordered_unique(
    frame: pd.DataFrame, timestamp_name: str, *, source: Path
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    try:
        raw_time = frame[timestamp_name].to_numpy(dtype=np.int64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"EuRoC CSV {source} requires integer {timestamp_name!r}"
        ) from exc
    if raw_time.size < 2:
        raise ValueError(f"EuRoC CSV requires at least two rows: {source}")
    order = np.argsort(raw_time, kind="stable")
    ordered = raw_time[order]
    keep = np.concatenate(([True], np.diff(ordered) != 0))
    prepared = frame.iloc[order].loc[keep].reset_index(drop=True)
    return prepared, ordered[keep], {
        "source_rows": int(len(frame)),
        "rows_reordered": int(np.count_nonzero(order != np.arange(len(order)))),
        "rows_deduplicated": int(np.count_nonzero(~keep)),
    }


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...], *, source: Path) -> np.ndarray:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"EuRoC CSV {source} is missing columns: {missing}")
    try:
        values = frame[list(columns)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"EuRoC CSV {source} contains non-numeric values") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"EuRoC CSV {source} contains non-finite values")
    return values


def _unit_quaternion_xyzw(values_wxyz: np.ndarray, *, source: Path) -> np.ndarray:
    norms = np.linalg.norm(values_wxyz, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-12):
        raise ValueError(f"EuRoC ground-truth quaternion is invalid: {source}")
    if np.max(np.abs(norms - 1.0)) > 1e-3:
        raise ValueError(f"EuRoC ground-truth quaternion is not unit length: {source}")
    normalized = values_wxyz / norms[:, None]
    return normalized[:, (1, 2, 3, 0)]


def _read_only(values: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values)
    values.setflags(write=False)
    return values


def load_euroc_recording(imu_path: Path, reference_path: Path) -> Recording:
    """Load one extracted EuRoC ``imu0``/ground-truth CSV pair.

    EuRoC stores nanosecond epoch timestamps and wxyz quaternions. The returned
    recording uses a shared relative-seconds clock and canonical xyzw order.
    """

    imu_path = Path(imu_path).resolve()
    reference_path = Path(reference_path).resolve()
    imu_frame, imu_time_ns, imu_rows = _ordered_unique(
        _read_csv(imu_path), "timestamp [ns]", source=imu_path
    )
    reference_frame, reference_time_ns, reference_rows = _ordered_unique(
        _read_csv(reference_path), "timestamp", source=reference_path
    )
    origin_ns = int(min(imu_time_ns[0], reference_time_ns[0]))
    imu_time_s = (imu_time_ns - origin_ns).astype(np.float64) / 1e9
    reference_time_s = (reference_time_ns - origin_ns).astype(np.float64) / 1e9
    if np.any(np.diff(imu_time_s) <= 0) or np.any(np.diff(reference_time_s) <= 0):
        raise ValueError("EuRoC timestamps must be strictly increasing after conversion")

    imu_six = _numeric(imu_frame, EUROC_IMU_CHANNEL_NAMES, source=imu_path)
    position_m = _numeric(
        reference_frame, EUROC_POSITION_NAMES, source=reference_path
    )
    quaternion_xyzw = _unit_quaternion_xyzw(
        _numeric(
            reference_frame,
            EUROC_QUATERNION_WXYZ_NAMES,
            source=reference_path,
        ),
        source=reference_path,
    )
    overlap_s = overlapping_interval(imu_time_s, reference_time_s)
    sequence = imu_path.parents[2].name if len(imu_path.parents) >= 3 else imu_path.stem
    metadata = MappingProxyType(
        {
            "dataset": "euroc_mav",
            "sequence": sequence,
            "imu_path": str(imu_path),
            "reference_path": str(reference_path),
            "time_origin_ns": origin_ns,
            "imu_source_time_unit": "ns",
            "reference_source_time_unit": "ns",
            "time_unit": "s",
            "imu_channel_names": EUROC_IMU_CHANNEL_NAMES,
            "imu_channel_units": (
                "rad/s", "rad/s", "rad/s", "m/s^2", "m/s^2", "m/s^2"
            ),
            "acceleration_semantics": "specific_force",
            "position_unit": "m",
            "quaternion_source_order": "wxyz",
            "quaternion_order": "xyzw",
            "rotation_mapping": "sensor_to_reference_documented_not_validated",
            "reference_is_target_only": True,
            **{f"imu_{key}": value for key, value in imu_rows.items()},
            **{f"reference_{key}": value for key, value in reference_rows.items()},
        }
    )
    return Recording(
        id=f"euroc_mav/{sequence}",
        imu_time_s=_read_only(imu_time_s),
        imu_six=_read_only(imu_six),
        vicon_time_s=_read_only(reference_time_s),
        vicon_position_m=_read_only(position_m),
        vicon_quaternion_xyzw=_read_only(quaternion_xyzw),
        overlap_s=overlap_s,
        metadata=metadata,
    )


__all__ = [
    "EUROC_IMU_CHANNEL_NAMES",
    "EUROC_POSITION_NAMES",
    "EUROC_QUATERNION_WXYZ_NAMES",
    "load_euroc_recording",
]
