import numpy as np
import pytest
from scipy.spatial.transform import Rotation


FRAME_METADATA = {
    "quaternion_order": "xyzw",
    "quaternion_frame": "body_to_reference",
    "euler_order": "xyz",
}
INTEGRATION_METADATA = {
    "acceleration_unit": "m/s^2",
    "time_unit": "s",
    "acceleration_frame": "world",
}


def test_synchronization_interpolates_position_and_quaternion_at_query_times():
    from validation_v2.evaluation.synchronization import synchronize_vicon_to_imu

    source_time = np.array([0.0, 2.0])
    source_position = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    source_quaternion = Rotation.from_euler("z", [0.0, 90.0], degrees=True).as_quat()

    synced = synchronize_vicon_to_imu(
        source_time,
        source_position,
        source_quaternion,
        np.array([0.0, 1.0, 2.0]),
        frame_metadata=FRAME_METADATA,
    )

    np.testing.assert_allclose(synced.time_s, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(synced.position_m[:, 0], [0.0, 1.0, 2.0])
    midpoint_degrees = Rotation.from_quat(synced.quaternion_xyzw[1]).as_euler(
        "xyz", degrees=True
    )[2]
    assert midpoint_degrees == pytest.approx(45.0)
    np.testing.assert_allclose(
        np.linalg.norm(synced.quaternion_xyzw, axis=1), 1.0, atol=1e-12
    )


@pytest.mark.parametrize(
    "source_time,query_time",
    [
        (np.array([0.0, 1.0]), np.array([-0.1, 0.5])),
        (np.array([0.0, 1.0]), np.array([0.5, 1.1])),
        (np.array([0.0, 0.0]), np.array([0.0])),
        (np.array([0.0, 1.0]), np.array([0.5, 0.4])),
        (np.array([0.0, np.nan]), np.array([0.5])),
    ],
)
def test_synchronization_rejects_bad_time_domains(source_time, query_time):
    from validation_v2.evaluation.synchronization import synchronize_vicon_to_imu

    with pytest.raises(ValueError):
        synchronize_vicon_to_imu(
            source_time,
            np.zeros((2, 3)),
            np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
            query_time,
            frame_metadata=FRAME_METADATA,
        )


def test_synchronization_rejects_missing_quaternion_metadata():
    from validation_v2.evaluation.synchronization import synchronize_vicon_to_imu

    with pytest.raises(ValueError, match="quaternion"):
        synchronize_vicon_to_imu(
            np.array([0.0, 1.0]),
            np.zeros((2, 3)),
            np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
            np.array([0.5]),
            frame_metadata={"euler_order": "xyz"},
        )


def test_body_to_reference_rotation_maps_body_x_to_world_y():
    from validation_v2.evaluation.trajectory import rotate_body_to_world

    quaternion = Rotation.from_euler("z", 90.0, degrees=True).as_quat()[None, :]
    result = rotate_body_to_world(
        np.array([[1.0, 0.0, 0.0]]),
        quaternion,
        mapping="body_to_reference",
        frame_metadata=FRAME_METADATA,
    )

    np.testing.assert_allclose(result, [[0.0, 1.0, 0.0]], atol=1e-12)
    with pytest.raises(ValueError, match="body_to_reference"):
        rotate_body_to_world(
            np.array([[1.0, 0.0, 0.0]]),
            quaternion,
            mapping="reference_to_body",
            frame_metadata=FRAME_METADATA,
        )


def test_full_record_integration_uses_dt_i_for_previous_to_current_interval_once():
    from validation_v2.evaluation.trajectory import integrate_acceleration

    acceleration = np.zeros((101, 3))
    acceleration[:, 0] = 2.0
    dt_s = np.full(101, 0.01)
    dt_s[0] = 123.0  # explicit placeholder: never integrated

    trajectory = integrate_acceleration(
        acceleration,
        dt_s,
        initial_position_m=np.zeros(3),
        initial_velocity_mps=np.zeros(3),
        frame_metadata=INTEGRATION_METADATA,
    )

    assert trajectory.position_m[-1, 0] == pytest.approx(1.0, abs=1e-12)
    assert trajectory.velocity_mps[-1, 0] == pytest.approx(2.0, abs=1e-12)
    trajectory.position_m[0, 0] = 99.0
    assert trajectory.position_m[0, 0] == 0.0


def test_integration_rejects_invalid_dt_and_missing_units():
    from validation_v2.evaluation.trajectory import integrate_acceleration

    acceleration = np.zeros((3, 3))
    with pytest.raises(ValueError, match="dt"):
        integrate_acceleration(
            acceleration,
            np.array([0.0, 0.1, 0.0]),
            initial_position_m=np.zeros(3),
            initial_velocity_mps=np.zeros(3),
            frame_metadata=INTEGRATION_METADATA,
        )
    with pytest.raises(ValueError, match="unit"):
        integrate_acceleration(
            acceleration,
            np.array([0.0, 0.1, 0.1]),
            initial_position_m=np.zeros(3),
            initial_velocity_mps=np.zeros(3),
            frame_metadata={"acceleration_frame": "world"},
        )


def test_trajectory_metrics_are_full_record_coordinate_aligned_values():
    from validation_v2.evaluation.trajectory import trajectory_metrics

    reference_position = np.column_stack([np.arange(5.0), np.zeros((5, 2))])
    predicted_position = reference_position + np.array([1.0, 0.0, 0.0])
    reference_velocity = np.tile([1.0, 0.0, 0.0], (5, 1))
    predicted_velocity = reference_velocity + np.array([0.5, 0.0, 0.0])

    metrics = trajectory_metrics(
        predicted_position,
        reference_position,
        predicted_velocity_mps=predicted_velocity,
        reference_velocity_mps=reference_velocity,
        interval=2,
    )

    assert metrics["ate_rmse_m"] == pytest.approx(1.0)
    assert metrics["endpoint_drift_m"] == pytest.approx(1.0)
    assert metrics["rpe_rmse_m"] == pytest.approx(0.0)
    assert metrics["rte_rmse_m"] == pytest.approx(0.0)
    assert metrics["velocity_rmse_mps"] == pytest.approx(0.5)
    assert metrics["alignment"] == "coordinate_aligned_no_similarity_transform"


def test_full_record_diagnostic_uses_only_user_acceleration_and_reports_delta():
    from validation_v2.evaluation.trajectory import (
        measured_attitude_full_record_diagnostic,
    )

    imu_time = np.linspace(0.0, 1.0, 101)
    complete = np.zeros((101, 6))
    imputed = complete.copy()
    complete[:, 3] = 2.0 / 9.81
    imputed[:, 3] = 1.0 / 9.81
    # Huge gyro values must not enter translational integration.
    complete[:, :3] = 1e6
    imputed[:, :3] = -1e6
    vicon_position = np.column_stack([np.square(imu_time), np.zeros((101, 2))])
    quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (101, 1))
    metadata = {
        **FRAME_METADATA,
        "imu_acceleration_unit": "G",
        "user_acceleration_semantics": "gravity_removed",
        "position_unit": "m",
        "time_unit": "s",
    }

    result = measured_attitude_full_record_diagnostic(
        complete,
        imputed,
        imu_time,
        imu_time,
        vicon_position,
        quaternions,
        frame_metadata=metadata,
        initial_velocity_mps=np.zeros(3),
        rpe_interval=10,
    )

    assert result.complete_metrics["endpoint_drift_m"] == pytest.approx(0.0, abs=1e-12)
    assert result.imputed_metrics["endpoint_drift_m"] == pytest.approx(0.5, abs=1e-12)
    assert result.delta_vs_complete["endpoint_drift_m"] == pytest.approx(0.5)
    assert result.complete_trajectory.position_m.shape == (101, 3)
    assert result.imputed_trajectory.position_m.shape == (101, 3)


def test_full_record_diagnostic_default_velocity_matches_analytic_ground_truth():
    from validation_v2.evaluation.trajectory import (
        measured_attitude_full_record_diagnostic,
    )

    time_s = np.linspace(0.0, 1.0, 101)
    imu = np.zeros((101, 6))
    imu[:, 3] = 2.0 / 9.81
    position_m = np.column_stack([np.square(time_s), np.zeros((101, 2))])
    quaternion_xyzw = np.tile([0.0, 0.0, 0.0, 1.0], (101, 1))
    metadata = {
        **FRAME_METADATA,
        "imu_acceleration_unit": "G",
        "user_acceleration_semantics": "gravity_removed",
        "position_unit": "m",
        "time_unit": "s",
    }

    result = measured_attitude_full_record_diagnostic(
        imu,
        imu.copy(),
        time_s,
        time_s,
        position_m,
        quaternion_xyzw,
        frame_metadata=metadata,
        rpe_interval=10,
    )

    for metrics in (result.complete_metrics, result.imputed_metrics):
        assert metrics["ate_rmse_m"] == pytest.approx(0.0, abs=1e-12)
        assert metrics["endpoint_drift_m"] == pytest.approx(0.0, abs=1e-12)
        assert metrics["velocity_rmse_mps"] == pytest.approx(0.0, abs=1e-12)


def test_diagnostic_result_defensively_freezes_metric_mappings():
    from validation_v2.evaluation.trajectory import DiagnosticResult, Trajectory

    trajectory = Trajectory(np.zeros((2, 3)), np.zeros((2, 3)))
    complete = {"ate_rmse_m": 1.0}
    imputed = {"ate_rmse_m": 2.0}
    delta = {"ate_rmse_m": 1.0}
    result = DiagnosticResult(
        complete_trajectory=trajectory,
        imputed_trajectory=trajectory,
        reference_trajectory=trajectory,
        complete_metrics=complete,
        imputed_metrics=imputed,
        delta_vs_complete=delta,
        time_s=np.array([0.0, 1.0]),
    )
    complete["ate_rmse_m"] = 99.0
    imputed["ate_rmse_m"] = 99.0
    delta["ate_rmse_m"] = 99.0

    assert result.complete_metrics["ate_rmse_m"] == 1.0
    assert result.imputed_metrics["ate_rmse_m"] == 2.0
    assert result.delta_vs_complete["ate_rmse_m"] == 1.0
    with pytest.raises(TypeError):
        result.complete_metrics["ate_rmse_m"] = 3.0
    with pytest.raises(TypeError):
        result.imputed_metrics["ate_rmse_m"] = 3.0
    with pytest.raises(TypeError):
        result.delta_vs_complete["ate_rmse_m"] = 3.0
