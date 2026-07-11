"""Load raw OxIOD IMU and Vicon CSV recordings without implicit alignment."""

from __future__ import annotations

import csv
from pathlib import Path
from types import MappingProxyType
from typing import Sequence

import numpy as np
import pandas as pd

from validation_v2.types import Recording


IMU_COLUMNS = (
    "Time",
    "attitude_roll",
    "attitude_pitch",
    "attitude_yaw",
    "rotation_rate_x",
    "rotation_rate_y",
    "rotation_rate_z",
    "gravity_x",
    "gravity_y",
    "gravity_z",
    "user_acc_x",
    "user_acc_y",
    "user_acc_z",
    "magnetic_field_x",
    "magnetic_field_y",
    "magnetic_field_z",
)
VICON_COLUMNS = (
    "Time",
    "Header",
    "translation.x",
    "translation.y",
    "translation.z",
    "rotation.x",
    "rotation.y",
    "rotation.z",
    "rotation.w",
)
IMU_CHANNEL_NAMES = (
    "rotation_rate_x",
    "rotation_rate_y",
    "rotation_rate_z",
    "user_acc_x",
    "user_acc_y",
    "user_acc_z",
)
IMU_CHANNEL_UNITS = ("rad/s", "rad/s", "rad/s", "G", "G", "G")


def _time_to_seconds(
    values: np.ndarray,
    stream: str,
    source_unit: str | None = None,
) -> np.ndarray:
    """Convert timestamps using explicit units and validate their order.

    When ``source_unit`` is omitted, the raw OxIOD schema is used: IMU time is
    seconds and Vicon time is nanoseconds.
    """
    times = np.asarray(values, dtype=np.float64)
    if not times.size:
        raise ValueError(f"{stream} timestamps must not be empty")
    if not np.all(np.isfinite(times)):
        raise ValueError(f"{stream} timestamps must contain only finite values")
    if source_unit is None:
        try:
            source_unit = {"imu": "s", "vicon": "ns"}[stream]
        except KeyError as exc:
            raise ValueError(f"unknown timestamp stream: {stream}") from exc
    if source_unit not in {"s", "ns"}:
        raise ValueError(f"unknown timestamp source unit: {source_unit}")
    if source_unit == "ns":
        times = times / 1e9
    if not np.all(np.diff(times) > 0):
        raise ValueError(f"{stream} timestamps must be strictly increasing")
    return times.astype(np.float64, copy=False)


def overlapping_interval(
    imu_t: np.ndarray,
    vicon_t: np.ndarray,
) -> tuple[float, float]:
    """Return the closed time bounds shared by both streams."""
    start = max(float(imu_t[0]), float(vicon_t[0]))
    end = min(float(imu_t[-1]), float(vicon_t[-1]))
    if start >= end:
        raise ValueError("no IMU/Vicon overlap")
    return start, end


def _first_row_has_header(path: Path) -> bool:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        first_row = next(csv.reader(handle), None)
    if not first_row:
        raise ValueError(f"CSV is empty: {path}")
    try:
        float(first_row[0])
    except ValueError:
        return True
    return False


def _read_columns(
    path: Path,
    *,
    stream_label: str,
    positional_names: Sequence[str],
    required_names: Sequence[str],
) -> pd.DataFrame:
    has_header = _first_row_has_header(path)
    frame = pd.read_csv(path, header=0 if has_header else None)
    if not has_header:
        found = frame.shape[1]
        expected = len(positional_names)
        if found != expected:
            raise ValueError(
                f"{stream_label} CSV {path}: expected {expected} columns from "
                f"Oxford Dataset/ReadMe.txt, found {found}"
            )
        frame.columns = list(positional_names)
    else:
        frame.columns = [str(name).strip() for name in frame.columns]
        missing = [name for name in required_names if name not in frame.columns]
        if missing:
            raise ValueError(
                f"{stream_label} CSV {path}: missing required columns {missing}; "
                f"found {list(frame.columns)}"
            )
    return frame


def _numeric_array(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    stream_label: str,
    path: Path,
) -> np.ndarray:
    try:
        return frame[list(columns)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{stream_label} CSV {path}: columns {list(columns)} must be numeric"
        ) from exc


def _read_only(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


def _order_and_deduplicate(
    frame: pd.DataFrame,
    *,
    stream: str,
    stream_label: str,
    source_unit: str,
    path: Path,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    raw_times = _numeric_array(
        frame,
        ("Time",),
        stream_label=stream_label,
        path=path,
    )[:, 0]
    order = np.argsort(raw_times, kind="stable")
    reordered = int(np.count_nonzero(order != np.arange(len(order))))
    ordered_times = raw_times[order]
    keep = (
        np.concatenate((np.array([True]), np.diff(ordered_times) != 0))
        if ordered_times.size
        else np.array([], dtype=bool)
    )
    prepared = frame.iloc[order].loc[keep].reset_index(drop=True)
    times_s = _time_to_seconds(
        ordered_times[keep],
        stream,
        source_unit=source_unit,
    )
    return prepared, times_s, {
        f"{stream}_source_rows": int(len(frame)),
        f"{stream}_rows_deduplicated": int(np.count_nonzero(~keep)),
        f"{stream}_rows_reordered": reordered,
    }


def load_recording(imu_path: Path, vicon_path: Path) -> Recording:
    """Load one raw OxIOD file pair while preserving both source timelines."""
    imu_path = Path(imu_path)
    vicon_path = Path(vicon_path)
    imu_frame = _read_columns(
        imu_path,
        stream_label="IMU",
        positional_names=IMU_COLUMNS,
        required_names=("Time", *IMU_CHANNEL_NAMES),
    )
    vicon_frame = _read_columns(
        vicon_path,
        stream_label="Vicon",
        positional_names=VICON_COLUMNS,
        required_names=(
            "Time",
            "translation.x",
            "translation.y",
            "translation.z",
            "rotation.x",
            "rotation.y",
            "rotation.z",
            "rotation.w",
        ),
    )

    imu_frame, imu_time_s, imu_row_metadata = _order_and_deduplicate(
        imu_frame,
        stream="imu",
        stream_label="IMU",
        source_unit="s",
        path=imu_path,
    )
    vicon_frame, vicon_time_s, vicon_row_metadata = _order_and_deduplicate(
        vicon_frame,
        stream="vicon",
        stream_label="Vicon",
        source_unit="ns",
        path=vicon_path,
    )
    imu_six = _numeric_array(
        imu_frame,
        IMU_CHANNEL_NAMES,
        stream_label="IMU",
        path=imu_path,
    )
    vicon_position_m = _numeric_array(
        vicon_frame,
        ("translation.x", "translation.y", "translation.z"),
        stream_label="Vicon",
        path=vicon_path,
    )
    vicon_quaternion_xyzw = _numeric_array(
        vicon_frame,
        ("rotation.x", "rotation.y", "rotation.z", "rotation.w"),
        stream_label="Vicon",
        path=vicon_path,
    )
    overlap_s = overlapping_interval(imu_time_s, vicon_time_s)

    metadata = MappingProxyType(
        {
            "imu_path": str(imu_path),
            "vicon_path": str(vicon_path),
            "imu_time_unit": "s",
            "vicon_source_time_unit": "ns",
            "vicon_time_unit": "s",
            "imu_channel_names": IMU_CHANNEL_NAMES,
            "imu_channel_units": IMU_CHANNEL_UNITS,
            "vicon_position_unit": "m",
            "vicon_quaternion_order": "xyzw",
            "vicon_is_target_only": True,
            **imu_row_metadata,
            **vicon_row_metadata,
        }
    )
    recording_id = f"{imu_path.parent.name}/{imu_path.stem}"
    return Recording(
        id=recording_id,
        imu_time_s=_read_only(imu_time_s),
        imu_six=_read_only(imu_six),
        vicon_time_s=_read_only(vicon_time_s),
        vicon_position_m=_read_only(vicon_position_m),
        vicon_quaternion_xyzw=_read_only(vicon_quaternion_xyzw),
        overlap_s=overlap_s,
        metadata=metadata,
    )
