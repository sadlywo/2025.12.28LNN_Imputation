"""Timestamp-exact Vicon interpolation onto an IMU query clock."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _frozen_array(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True, init=False, repr=False, eq=False)
class SynchronizedVicon:
    """Vicon pose sampled exactly at the returned query timestamps."""

    _time_s: np.ndarray = field(repr=False)
    _position_m: np.ndarray = field(repr=False)
    _quaternion_xyzw: np.ndarray = field(repr=False)

    def __init__(self, time_s: np.ndarray, position_m: np.ndarray, quaternion_xyzw: np.ndarray) -> None:
        object.__setattr__(self, "_time_s", _frozen_array(time_s))
        object.__setattr__(self, "_position_m", _frozen_array(position_m))
        object.__setattr__(self, "_quaternion_xyzw", _frozen_array(quaternion_xyzw))

    @property
    def time_s(self) -> np.ndarray:
        return self._time_s.copy()

    @property
    def position_m(self) -> np.ndarray:
        return self._position_m.copy()

    @property
    def quaternion_xyzw(self) -> np.ndarray:
        return self._quaternion_xyzw.copy()


def validate_attitude_metadata(frame_metadata: Mapping[str, object]) -> None:
    """Require the declared quaternion convention used by SciPy Rotation."""

    if not isinstance(frame_metadata, Mapping):
        raise ValueError("frame_metadata must be provided")
    if frame_metadata.get("quaternion_order") != "xyzw":
        raise ValueError("quaternion metadata must declare quaternion_order='xyzw'")
    if frame_metadata.get("quaternion_frame") != "body_to_reference":
        raise ValueError(
            "quaternion metadata must declare quaternion_frame='body_to_reference'"
        )
    if frame_metadata.get("euler_order") != "xyz":
        raise ValueError("frame_metadata must declare euler_order='xyz'")


def _validated_time(value: np.ndarray, name: str, *, minimum_length: int) -> np.ndarray:
    try:
        time = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric seconds") from exc
    if time.ndim != 1 or time.size < minimum_length:
        raise ValueError(f"{name} must be one-dimensional with at least {minimum_length} values")
    if not np.all(np.isfinite(time)) or (time.size > 1 and np.any(np.diff(time) <= 0)):
        raise ValueError(f"{name} must contain finite, strictly increasing seconds")
    return time


def synchronize_vicon_to_imu(
    source_time_s: np.ndarray,
    source_position_m: np.ndarray,
    source_quaternion_xyzw: np.ndarray,
    query_time_s: np.ndarray,
    *,
    frame_metadata: Mapping[str, object],
) -> SynchronizedVicon:
    """Linearly interpolate position and SLERP body-to-reference attitude.

    Query times outside the source domain are rejected rather than extrapolated.
    Euler order is validated as framework metadata, but no unnecessary Euler
    conversion is performed: interpolation operates directly on xyzw quaternions.
    """

    validate_attitude_metadata(frame_metadata)
    source_time = _validated_time(source_time_s, "source_time_s", minimum_length=2)
    query_time = _validated_time(query_time_s, "query_time_s", minimum_length=1)
    if query_time[0] < source_time[0] or query_time[-1] > source_time[-1]:
        raise ValueError("query_time_s lies outside the source time range")

    try:
        position = np.asarray(source_position_m, dtype=np.float64)
        quaternion = np.asarray(source_quaternion_xyzw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("source position and quaternion must be numeric arrays") from exc
    if position.shape != (source_time.size, 3):
        raise ValueError("source_position_m must have shape (N, 3)")
    if quaternion.shape != (source_time.size, 4):
        raise ValueError("source_quaternion_xyzw must have shape (N, 4)")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        raise ValueError("source pose must contain only finite values")
    if not np.allclose(np.linalg.norm(quaternion, axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("source_quaternion_xyzw must contain unit quaternions")

    interpolated_position = np.column_stack(
        [np.interp(query_time, source_time, position[:, axis]) for axis in range(3)]
    )
    rotations = Slerp(source_time, Rotation.from_quat(quaternion))(query_time)
    interpolated_quaternion = rotations.as_quat()
    interpolated_quaternion /= np.linalg.norm(interpolated_quaternion, axis=1, keepdims=True)
    return SynchronizedVicon(query_time, interpolated_position, interpolated_quaternion)
