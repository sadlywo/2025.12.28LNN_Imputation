"""IDOL Feather trajectory loader using the Stencil IMU frame."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from validation_v2.types import Recording


IDOL_IMU_CHANNEL_NAMES = (
    "stencilGyroX",
    "stencilGyroY",
    "stencilGyroZ",
    "stencilAccX",
    "stencilAccY",
    "stencilAccZ",
)
IDOL_POSITION_NAMES = ("processedPosX", "processedPosY", "processedPosZ")
IDOL_QUATERNION_WXYZ_NAMES = ("orientW", "orientX", "orientY", "orientZ")
IDOL_REQUIRED_COLUMNS = (
    "timestamp",
    *IDOL_IMU_CHANNEL_NAMES,
    *IDOL_POSITION_NAMES,
    *IDOL_QUATERNION_WXYZ_NAMES,
)


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...], *, source: Path) -> np.ndarray:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"IDOL Feather {source} is missing columns: {missing}")
    try:
        values = frame[list(columns)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"IDOL Feather {source} contains non-numeric values") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"IDOL Feather {source} contains non-finite values")
    return values


def _trajectory_metadata(path: Path) -> dict[str, object]:
    metadata_path = path.parent / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"IDOL split metadata is missing: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        row = metadata[path.stem]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"IDOL metadata has no valid entry for {path.name}") from exc
    if not isinstance(row, dict):
        raise ValueError(f"IDOL metadata entry must be an object: {path}")
    return dict(row)


def _read_only(values: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values)
    values.setflags(write=False)
    return values


def load_idol_recording(imu_path: Path, reference_path: Path) -> Recording:
    """Load one IDOL trajectory.

    A Feather file contains both streams on one synchronized 100 Hz clock. The
    Stencil gyro/accelerometer is selected because IDOL documents that its IMU
    frame and ground-truth frame are the same. Source wxyz quaternions are
    returned in canonical xyzw order.
    """

    imu_path = Path(imu_path).resolve()
    reference_path = Path(reference_path).resolve()
    if imu_path != reference_path:
        raise ValueError("IDOL IMU and reference must point to the same Feather file")
    if not imu_path.is_file():
        raise ValueError(f"IDOL Feather does not exist: {imu_path}")
    try:
        frame = pd.read_feather(imu_path, columns=list(IDOL_REQUIRED_COLUMNS))
    except (ImportError, OSError, ValueError) as exc:
        raise ValueError(f"unable to read IDOL Feather {imu_path}: {exc}") from exc

    timestamp = _numeric(frame, ("timestamp",), source=imu_path)[:, 0]
    if timestamp.size < 2:
        raise ValueError(f"IDOL trajectory requires at least two rows: {imu_path}")
    order = np.argsort(timestamp, kind="stable")
    ordered_time = timestamp[order]
    keep = np.concatenate(([True], np.diff(ordered_time) != 0))
    prepared = frame.iloc[order].loc[keep].reset_index(drop=True)
    source_rows = len(frame)
    timestamp = ordered_time[keep]
    time_s = timestamp - timestamp[0]
    if np.any(np.diff(time_s) <= 0):
        raise ValueError(f"IDOL timestamps must be strictly increasing: {imu_path}")

    imu_six = _numeric(prepared, IDOL_IMU_CHANNEL_NAMES, source=imu_path)
    position_m = _numeric(prepared, IDOL_POSITION_NAMES, source=imu_path)
    quaternion_wxyz = _numeric(
        prepared, IDOL_QUATERNION_WXYZ_NAMES, source=imu_path
    )
    quaternion_norm = np.linalg.norm(quaternion_wxyz, axis=1)
    if np.any(quaternion_norm < 1e-12) or np.max(np.abs(quaternion_norm - 1.0)) > 1e-3:
        raise ValueError(f"IDOL ground-truth quaternion is invalid: {imu_path}")
    quaternion_wxyz = quaternion_wxyz / quaternion_norm[:, None]
    quaternion_xyzw = quaternion_wxyz[:, (1, 2, 3, 0)]

    split_name = imu_path.parent.name
    building = imu_path.parent.parent.name
    source_metadata = _trajectory_metadata(imu_path)
    building_rotation_offsets = {"building1": 0.0, "building2": 1.8510, "building3": 0.2822}
    metadata = MappingProxyType(
        {
            "dataset": "idol",
            "building": building,
            "source_subset": split_name,
            "subject_id": source_metadata.get("subjectID"),
            "calibration_segment": source_metadata.get("calibration", "unknown"),
            "source_path": str(imu_path),
            "source_time_origin_s": float(timestamp[0]),
            "time_unit": "s",
            "nominal_sample_rate_hz": 100.0,
            "imu_source": "stencil",
            "imu_channel_names": IDOL_IMU_CHANNEL_NAMES,
            "imu_channel_units": (
                "rad/s", "rad/s", "rad/s", "m/s^2", "m/s^2", "m/s^2"
            ),
            "acceleration_semantics": "specific_force",
            "position_unit": "m",
            "quaternion_source_order": "wxyz",
            "quaternion_order": "xyzw",
            "rotation_mapping": "stencil_to_global_documented_not_validated",
            "provided_global_alignment": True,
            "published_building_rotation_offset_rad": building_rotation_offsets.get(building),
            "reference_is_target_only": True,
            "source_rows": int(source_rows),
            "rows_reordered": int(np.count_nonzero(order != np.arange(source_rows))),
            "rows_deduplicated": int(np.count_nonzero(~keep)),
        }
    )
    return Recording(
        id=f"idol/{building}/{split_name}/{imu_path.stem}",
        imu_time_s=_read_only(time_s),
        imu_six=_read_only(imu_six),
        vicon_time_s=_read_only(time_s.copy()),
        vicon_position_m=_read_only(position_m),
        vicon_quaternion_xyzw=_read_only(quaternion_xyzw),
        overlap_s=(float(time_s[0]), float(time_s[-1])),
        metadata=metadata,
    )


__all__ = [
    "IDOL_IMU_CHANNEL_NAMES",
    "IDOL_POSITION_NAMES",
    "IDOL_QUATERNION_WXYZ_NAMES",
    "IDOL_REQUIRED_COLUMNS",
    "load_idol_recording",
]
