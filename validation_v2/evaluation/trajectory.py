"""Measured-attitude, full-record inertial trajectory diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .synchronization import synchronize_vicon_to_imu, validate_attitude_metadata


def _frozen_array(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True, init=False, repr=False, eq=False)
class Trajectory:
    """Position and velocity whose internal NumPy storage is truly immutable."""

    _position_m: np.ndarray = field(repr=False)
    _velocity_mps: np.ndarray = field(repr=False)

    def __init__(self, position_m: np.ndarray, velocity_mps: np.ndarray) -> None:
        object.__setattr__(self, "_position_m", _frozen_array(position_m))
        object.__setattr__(self, "_velocity_mps", _frozen_array(velocity_mps))

    @property
    def position_m(self) -> np.ndarray:
        return self._position_m.copy()

    @property
    def velocity_mps(self) -> np.ndarray:
        return self._velocity_mps.copy()


@dataclass(frozen=True)
class DiagnosticResult:
    """Two full-record trajectories, metrics, and imputed-minus-complete deltas."""

    complete_trajectory: Trajectory
    imputed_trajectory: Trajectory
    reference_trajectory: Trajectory
    complete_metrics: Mapping[str, float | str]
    imputed_metrics: Mapping[str, float | str]
    delta_vs_complete: Mapping[str, float]
    time_s: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "complete_metrics", MappingProxyType(dict(self.complete_metrics))
        )
        object.__setattr__(
            self, "imputed_metrics", MappingProxyType(dict(self.imputed_metrics))
        )
        object.__setattr__(
            self, "delta_vs_complete", MappingProxyType(dict(self.delta_vs_complete))
        )
        object.__setattr__(self, "time_s", _frozen_array(self.time_s))


def _vector_array(value: np.ndarray, name: str, *, length: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    expected = (length, 3) if length is not None else None
    if array.ndim != 2 or array.shape[1] != 3 or (expected is not None and array.shape != expected):
        raise ValueError(f"{name} must have shape (N, 3)")
    if array.shape[0] == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and finite")
    return array


def rotate_body_to_world(
    vectors_body: np.ndarray,
    quaternion_xyzw: np.ndarray,
    *,
    mapping: str,
    frame_metadata: Mapping[str, object],
) -> np.ndarray:
    """Apply measured body-to-reference rotations to body-frame vectors."""

    validate_attitude_metadata(frame_metadata)
    if mapping != "body_to_reference":
        raise ValueError("mapping must be exactly 'body_to_reference'")
    vectors = _vector_array(vectors_body, "vectors_body")
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (vectors.shape[0], 4) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion_xyzw must be a finite array with shape (N, 4)")
    if not np.allclose(np.linalg.norm(quaternion, axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("quaternion_xyzw must contain unit quaternions")
    return Rotation.from_quat(quaternion).apply(vectors)


def _validate_integration_metadata(frame_metadata: Mapping[str, object]) -> None:
    if not isinstance(frame_metadata, Mapping):
        raise ValueError("frame_metadata must be supplied")
    required = {
        "acceleration_unit": "m/s^2",
        "time_unit": "s",
        "acceleration_frame": "world",
    }
    for key, expected in required.items():
        if frame_metadata.get(key) != expected:
            raise ValueError(f"physical unit/frame metadata must declare {key}={expected!r}")


def integrate_acceleration(
    acceleration_world_mps2: np.ndarray,
    dt_s: np.ndarray,
    *,
    initial_position_m: np.ndarray,
    initial_velocity_mps: np.ndarray,
    frame_metadata: Mapping[str, object],
) -> Trajectory:
    """Integrate one full record, initializing position and velocity once.

    ``dt_s[i]`` is the elapsed time from sample ``i-1`` to sample ``i``.
    ``dt_s[0]`` is a finite placeholder and is never integrated.  Trapezoidal
    acceleration and velocity integration is exact for constant acceleration.
    """

    _validate_integration_metadata(frame_metadata)
    acceleration = _vector_array(acceleration_world_mps2, "acceleration_world_mps2")
    dt = np.asarray(dt_s, dtype=np.float64)
    if dt.shape != (acceleration.shape[0],) or not np.all(np.isfinite(dt)):
        raise ValueError("dt_s must be a finite length-N array")
    if acceleration.shape[0] > 1 and np.any(dt[1:] <= 0):
        raise ValueError("dt_s[1:] must be strictly positive")
    p0 = np.asarray(initial_position_m, dtype=np.float64)
    v0 = np.asarray(initial_velocity_mps, dtype=np.float64)
    if p0.shape != (3,) or v0.shape != (3,) or not np.all(np.isfinite(p0)) or not np.all(np.isfinite(v0)):
        raise ValueError("initial_position_m and initial_velocity_mps must be finite 3-vectors")

    position = np.empty_like(acceleration)
    velocity = np.empty_like(acceleration)
    position[0] = p0
    velocity[0] = v0
    for index in range(1, acceleration.shape[0]):
        velocity[index] = velocity[index - 1] + 0.5 * (
            acceleration[index - 1] + acceleration[index]
        ) * dt[index]
        position[index] = position[index - 1] + 0.5 * (
            velocity[index - 1] + velocity[index]
        ) * dt[index]
    return Trajectory(position, velocity)


def trajectory_metrics(
    predicted_position_m: np.ndarray,
    reference_position_m: np.ndarray,
    *,
    predicted_velocity_mps: np.ndarray | None = None,
    reference_velocity_mps: np.ndarray | None = None,
    interval: int = 1,
) -> dict[str, float | str]:
    """Compute coordinate-aligned full-record metrics without similarity fit.

    ATE-RMSE is the RMS Euclidean position error.  Fixed-interval RPE/RTE are
    the RMS error between predicted and reference translation increments over
    exactly ``interval`` samples.  Both names are emitted for reporting
    compatibility and intentionally denote the same translational quantity.
    """

    predicted = _vector_array(predicted_position_m, "predicted_position_m")
    reference = _vector_array(reference_position_m, "reference_position_m", length=predicted.shape[0])
    if not isinstance(interval, (int, np.integer)) or isinstance(interval, bool):
        raise ValueError("interval must be an integer sample count")
    if interval < 1 or interval >= predicted.shape[0]:
        raise ValueError("interval must satisfy 1 <= interval < record length")
    error = predicted - reference
    ate = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
    predicted_delta = predicted[interval:] - predicted[:-interval]
    reference_delta = reference[interval:] - reference[:-interval]
    relative_error = predicted_delta - reference_delta
    relative_rmse = float(np.sqrt(np.mean(np.sum(relative_error * relative_error, axis=1))))
    result: dict[str, float | str] = {
        "ate_rmse_m": ate,
        "rpe_rmse_m": relative_rmse,
        "rte_rmse_m": relative_rmse,
        "endpoint_drift_m": float(np.linalg.norm(error[-1])),
        "alignment": "coordinate_aligned_no_similarity_transform",
        "interval_samples": float(interval),
    }
    if (predicted_velocity_mps is None) != (reference_velocity_mps is None):
        raise ValueError("predicted and reference velocity must be supplied together")
    if predicted_velocity_mps is not None:
        predicted_velocity = _vector_array(
            predicted_velocity_mps, "predicted_velocity_mps", length=predicted.shape[0]
        )
        reference_velocity = _vector_array(
            reference_velocity_mps, "reference_velocity_mps", length=predicted.shape[0]
        )
        velocity_error = predicted_velocity - reference_velocity
        result["velocity_rmse_mps"] = float(
            np.sqrt(np.mean(np.sum(velocity_error * velocity_error, axis=1)))
        )
    return result


def _diagnostic_metadata(frame_metadata: Mapping[str, object]) -> None:
    validate_attitude_metadata(frame_metadata)
    required = {
        "imu_acceleration_unit": "G",
        "user_acceleration_semantics": "gravity_removed",
        "position_unit": "m",
        "time_unit": "s",
    }
    for key, expected in required.items():
        if frame_metadata.get(key) != expected:
            raise ValueError(f"diagnostic metadata must declare {key}={expected!r}")


def measured_attitude_full_record_diagnostic(
    complete_imu_six: np.ndarray,
    imputed_imu_six: np.ndarray,
    imu_time_s: np.ndarray,
    vicon_time_s: np.ndarray,
    vicon_position_m: np.ndarray,
    vicon_quaternion_xyzw: np.ndarray,
    *,
    frame_metadata: Mapping[str, object],
    initial_velocity_mps: np.ndarray | None = None,
    rpe_interval: int = 1,
) -> DiagnosticResult:
    """Evaluate complete and imputed IMU over the entire Vicon-overlap record.

    Only columns 3:6 (gravity-removed user acceleration in G) are used.  Vicon
    attitude and position are evaluation-only metadata and never model inputs.
    Measured xyzw attitude is SLERPed to the IMU overlap clock and applied
    directly; ``euler_order='xyz'`` is a validated framework declaration, not a
    request for an unnecessary quaternion-to-Euler conversion.  If ``v0`` is
    omitted it is estimated once from the first synchronized GT position pair.
    """

    _diagnostic_metadata(frame_metadata)
    complete = np.asarray(complete_imu_six, dtype=np.float64)
    imputed = np.asarray(imputed_imu_six, dtype=np.float64)
    imu_time = np.asarray(imu_time_s, dtype=np.float64)
    if complete.ndim != 2 or complete.shape[1] != 6 or imputed.shape != complete.shape:
        raise ValueError("complete and imputed IMU must have matching shape (N, 6)")
    if imu_time.shape != (complete.shape[0],):
        raise ValueError("imu_time_s length must match IMU rows")
    if not np.all(np.isfinite(complete)) or not np.all(np.isfinite(imputed)):
        raise ValueError("complete and imputed IMU must contain only finite values")
    if not np.all(np.isfinite(imu_time)) or np.any(np.diff(imu_time) <= 0):
        raise ValueError("imu_time_s must contain finite, strictly increasing seconds")
    vicon_time = np.asarray(vicon_time_s, dtype=np.float64)
    if vicon_time.ndim != 1 or vicon_time.size < 2 or not np.all(np.isfinite(vicon_time)) or np.any(np.diff(vicon_time) <= 0):
        raise ValueError("vicon_time_s must contain finite, strictly increasing seconds")

    overlap = (imu_time >= vicon_time[0]) & (imu_time <= vicon_time[-1])
    query_time = imu_time[overlap]
    if query_time.size < 2:
        raise ValueError("IMU/Vicon overlap must contain at least two IMU samples")
    synced = synchronize_vicon_to_imu(
        vicon_time,
        vicon_position_m,
        vicon_quaternion_xyzw,
        query_time,
        frame_metadata=frame_metadata,
    )
    complete_acceleration_body = complete[overlap, 3:6] * 9.81
    imputed_acceleration_body = imputed[overlap, 3:6] * 9.81
    complete_acceleration_world = rotate_body_to_world(
        complete_acceleration_body,
        synced.quaternion_xyzw,
        mapping="body_to_reference",
        frame_metadata=frame_metadata,
    )
    imputed_acceleration_world = rotate_body_to_world(
        imputed_acceleration_body,
        synced.quaternion_xyzw,
        mapping="body_to_reference",
        frame_metadata=frame_metadata,
    )
    dt = np.empty_like(query_time)
    dt[0] = 0.0
    dt[1:] = np.diff(query_time)
    edge_order = 2 if query_time.size >= 3 else 1
    reference_velocity = np.gradient(
        synced.position_m, query_time, axis=0, edge_order=edge_order
    )
    if initial_velocity_mps is None:
        v0 = reference_velocity[0]
    else:
        v0 = np.asarray(initial_velocity_mps, dtype=np.float64)
    integration_metadata = {
        "acceleration_unit": "m/s^2",
        "time_unit": "s",
        "acceleration_frame": "world",
    }
    p0 = synced.position_m[0]
    complete_trajectory = integrate_acceleration(
        complete_acceleration_world,
        dt,
        initial_position_m=p0,
        initial_velocity_mps=v0,
        frame_metadata=integration_metadata,
    )
    imputed_trajectory = integrate_acceleration(
        imputed_acceleration_world,
        dt,
        initial_position_m=p0,
        initial_velocity_mps=v0,
        frame_metadata=integration_metadata,
    )
    reference_trajectory = Trajectory(synced.position_m, reference_velocity)
    complete_metrics = trajectory_metrics(
        complete_trajectory.position_m,
        synced.position_m,
        predicted_velocity_mps=complete_trajectory.velocity_mps,
        reference_velocity_mps=reference_velocity,
        interval=rpe_interval,
    )
    imputed_metrics = trajectory_metrics(
        imputed_trajectory.position_m,
        synced.position_m,
        predicted_velocity_mps=imputed_trajectory.velocity_mps,
        reference_velocity_mps=reference_velocity,
        interval=rpe_interval,
    )
    delta = {
        key: float(imputed_metrics[key]) - float(complete_metrics[key])
        for key in complete_metrics
        if isinstance(complete_metrics[key], (int, float, np.integer, np.floating))
        and key != "interval_samples"
    }
    return DiagnosticResult(
        complete_trajectory=complete_trajectory,
        imputed_trajectory=imputed_trajectory,
        reference_trajectory=reference_trajectory,
        complete_metrics=complete_metrics,
        imputed_metrics=imputed_metrics,
        delta_vs_complete=delta,
        time_s=query_time,
    )


# Descriptive alias for callers that prefer an imperative evaluation name.
evaluate_measured_attitude_reconstruction = measured_attitude_full_record_diagnostic
