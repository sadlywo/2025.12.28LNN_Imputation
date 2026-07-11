import numpy as np
import pytest
import torch


def test_missing_metrics_ignore_observed_values_and_use_current_timestep():
    from validation_v2.objectives.reconstruction import (
        missing_mae,
        missing_mse,
        missing_rmse,
    )

    target = torch.tensor([[0.0, 2.0], [3.0, 4.0]])
    prediction = torch.tensor([[99.0, 4.0], [2.0, 99.0]])
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    assert missing_mse(prediction, target, mask).item() == pytest.approx(2.5)
    assert missing_rmse(prediction, target, mask).item() == pytest.approx(np.sqrt(2.5))
    assert missing_mae(prediction, target, mask).item() == pytest.approx(1.5)


@pytest.mark.parametrize(
    "prediction,target,mask",
    [
        (torch.zeros(2), torch.zeros(3), torch.zeros(2)),
        (torch.zeros(2), torch.zeros(2), torch.ones(2)),
        (torch.zeros(2), torch.zeros(2), torch.tensor([0.0, 0.5])),
        (torch.tensor([0.0, float("nan")]), torch.zeros(2), torch.zeros(2)),
        (torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=torch.float64), torch.zeros(2)),
    ],
)
def test_missing_mse_rejects_invalid_inputs(prediction, target, mask):
    from validation_v2.objectives.reconstruction import missing_mse

    with pytest.raises(ValueError):
        missing_mse(prediction, target, mask)


def test_kinematic_consistency_requires_physical_metadata_and_labels():
    from validation_v2.objectives.kinematic import kinematic_consistency_loss

    acceleration = torch.zeros((4, 3), requires_grad=True)
    time_s = torch.arange(4, dtype=torch.float32)

    with pytest.raises(ValueError, match="metadata"):
        kinematic_consistency_loss(
            acceleration,
            time_s,
            frame_metadata={},
            velocity_mps=torch.zeros((4, 3)),
        )
    with pytest.raises(ValueError, match="label"):
        kinematic_consistency_loss(
            acceleration,
            time_s,
            frame_metadata={
                "acceleration_unit": "m/s^2",
                "time_unit": "s",
                "acceleration_frame": "world",
            },
        )
    with pytest.raises(ValueError, match="label"):
        kinematic_consistency_loss(
            acceleration,
            time_s,
            frame_metadata={
                "acceleration_unit": "m/s^2",
                "time_unit": "s",
                "acceleration_frame": "world",
            },
            velocity_mps=torch.zeros((4, 3)),
        )


def test_kinematic_consistency_constant_acceleration_has_zero_loss_and_gradient():
    from validation_v2.objectives.kinematic import kinematic_consistency_loss

    time_s = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    acceleration = torch.zeros((11, 3), dtype=torch.float64)
    acceleration[:, 0] = 2.0
    acceleration.requires_grad_()
    velocity = torch.zeros_like(acceleration)
    displacement = torch.zeros_like(acceleration)
    velocity[:, 0] = 2.0 * time_s
    displacement[:, 0] = time_s.square()

    result = kinematic_consistency_loss(
        acceleration,
        time_s,
        frame_metadata={
            "acceleration_unit": "m/s^2",
            "time_unit": "s",
            "acceleration_frame": "world",
        },
        velocity_mps=velocity,
        displacement_m=displacement,
        velocity_weight=2.0,
        displacement_weight=3.0,
    )

    assert result.total.item() == pytest.approx(0.0, abs=1e-14)
    assert result.velocity.item() == pytest.approx(0.0, abs=1e-14)
    assert result.displacement.item() == pytest.approx(0.0, abs=1e-14)
    result.total.backward()
    assert acceleration.grad is not None
    assert torch.all(torch.isfinite(acceleration.grad))


def test_reconstruction_report_never_labels_normalized_values_as_physical():
    from validation_v2.evaluation.reconstruction import reconstruction_metrics

    prediction = np.array([[1.0, 2.0]])
    target = np.array([[0.0, 0.0]])
    mask = np.zeros_like(prediction)

    with pytest.raises(ValueError, match="physical"):
        reconstruction_metrics(prediction, target, mask)

    report = reconstruction_metrics(
        prediction,
        target,
        mask,
        physical_prediction=np.array([[10.0, 20.0]]),
        physical_target=np.array([[0.0, 0.0]]),
    )
    assert report["normalized"]["mse"] == pytest.approx(2.5)
    assert report["physical"]["mse"] == pytest.approx(250.0)
