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
    """Vicon pose/velocity sampled exactly at the returned IMU timestamps."""

    _time_s: np.ndarray = field(repr=False)
    _position_m: np.ndarray = field(repr=False)
    _quaternion_xyzw: np.ndarray = field(repr=False)
    _rotation_body_to_world: np.ndarray = field(repr=False)
    _velocity_world_mps: np.ndarray = field(repr=False)

    def __init__(
        self,
        time_s: np.ndarray,
        position_m: np.ndarray,
        quaternion_xyzw: np.ndarray,
        rotation_body_to_world: np.ndarray,
        velocity_world_mps: np.ndarray,
    ) -> None:
        object.__setattr__(self, "_time_s", _frozen_array(time_s))
        object.__setattr__(self, "_position_m", _frozen_array(position_m))
        object.__setattr__(self, "_quaternion_xyzw", _frozen_array(quaternion_xyzw))
        object.__setattr__(
            self, "_rotation_body_to_world", _frozen_array(rotation_body_to_world)
        )
        object.__setattr__(self, "_velocity_world_mps", _frozen_array(velocity_world_mps))

    @property
    def time_s(self) -> np.ndarray:
        return self._time_s.copy()

    @property
    def position_m(self) -> np.ndarray:
        return self._position_m.copy()

    @property
    def quaternion_xyzw(self) -> np.ndarray:
        return self._quaternion_xyzw.copy()

    @property
    def rotation_body_to_world(self) -> np.ndarray:
        return self._rotation_body_to_world.copy()

    @property
    def velocity_world_mps(self) -> np.ndarray:
        return self._velocity_world_mps.copy()


def position_velocity(position_m: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Differentiate aligned position with central interior differences."""

    position = np.asarray(position_m, dtype=np.float64)
    time = np.asarray(time_s, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3 or time.shape != (len(position),):
        raise ValueError("position/time must have shapes (T,3) and (T,)")
    if len(position) < 1 or not np.all(np.isfinite(position)):
        raise ValueError("position requires at least one finite sample")
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("time must be finite, increasing seconds")
    if len(position) == 1:
        return np.zeros_like(position)
    velocity = np.empty_like(position)
    velocity[0] = (position[1] - position[0]) / (time[1] - time[0])
    velocity[-1] = (position[-1] - position[-2]) / (time[-1] - time[-2])
    if len(position) > 2:
        velocity[1:-1] = (position[2:] - position[:-2]) / (
            time[2:] - time[:-2]
        )[:, None]
    return velocity


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
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("source_quaternion_xyzw must have positive norms")
    quaternion = quaternion / norms
    # q and -q are the same attitude.  Enforce a continuous shortest-path
    # representation before SLERP so source sign jumps cannot create artifacts.
    quaternion = quaternion.copy()
    for index in range(1, len(quaternion)):
        if np.dot(quaternion[index - 1], quaternion[index]) < 0:
            quaternion[index] *= -1.0

    interpolated_position = np.column_stack(
        [np.interp(query_time, source_time, position[:, axis]) for axis in range(3)]
    )
    rotations = Slerp(source_time, Rotation.from_quat(quaternion))(query_time)
    interpolated_quaternion = rotations.as_quat()
    interpolated_quaternion /= np.linalg.norm(interpolated_quaternion, axis=1, keepdims=True)
    velocity = position_velocity(interpolated_position, query_time)
    return SynchronizedVicon(
        query_time,
        interpolated_position,
        interpolated_quaternion,
        rotations.as_matrix(),
        velocity,
    )


__all__ = [
    "SynchronizedVicon",
    "position_velocity",
    "synchronize_vicon_to_imu",
    "validate_attitude_metadata",
]
