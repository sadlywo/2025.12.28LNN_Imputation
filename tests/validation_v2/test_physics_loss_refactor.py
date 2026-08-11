import math

import pytest
import torch

from validation_v2.data.features import build_features
from validation_v2.data.normalization import RobustTrainScaler
from validation_v2.experiments.runner import _physics_loss_config
from validation_v2.models.hybrid import HybridImputer, complete_signal
from validation_v2.objectives.physics_informed import (
    IMUPhysicsInformedLoss,
    PhysicsLossConfig,
)
from validation_v2.physics import propagate_imu, so3_exp, so3_log


def _consistent_case(device="cpu", dtype=torch.float64):
    batch, steps = 2, 7
    gyro = torch.zeros(batch, steps, 3, device=device, dtype=dtype)
    gyro[..., 2] = 0.2
    acceleration = torch.zeros(batch, steps, 3, device=device, dtype=dtype)
    acceleration[..., 0] = 0.4
    target = torch.cat((gyro, acceleration), dim=-1)
    dt = torch.full((batch, steps), 0.05, device=device, dtype=dtype)
    rotation0 = torch.eye(3, device=device, dtype=dtype).expand(batch, 3, 3).clone()
    velocity0 = torch.zeros(batch, 3, device=device, dtype=dtype)
    position0 = torch.zeros(batch, 3, device=device, dtype=dtype)
    truth = propagate_imu(
        gyro,
        acceleration,
        dt,
        rotation0,
        velocity0,
        position0,
        acceleration_mode="gravity_compensated",
    )
    mask = torch.ones_like(target)
    mask[:, 2, 0] = 0
    mask[:, 3, 3] = 0
    return target, mask, dt, truth


def _criterion(lambda_physics=1.0):
    return IMUPhysicsInformedLoss(
        PhysicsLossConfig(
            lambda_physics=lambda_physics,
            sigma_rotation_rad=1.0,
            sigma_velocity_mps=1.0,
            sigma_position_m=1.0,
            acceleration_mode="gravity_compensated",
            acceleration_unit="m/s^2",
        )
    )


def _loss(criterion, prediction, target, mask, dt, truth):
    batch = prediction.shape[0]
    return criterion(
        prediction=prediction,
        target=target,
        mask=mask,
        completed=complete_signal(target, mask, prediction),
        dt=dt,
        normalization_center=torch.zeros(batch, 6, **_tensor_options(prediction)),
        normalization_scale=torch.ones(batch, 6, **_tensor_options(prediction)),
        vicon_position_m=truth.position_world_m,
        vicon_rotation_body_to_world=truth.rotation_body_to_world,
        vicon_velocity_mps=truth.velocity_world_mps,
    )


def _tensor_options(reference):
    return {"device": reference.device, "dtype": reference.dtype}


def test_signal_and_physics_loss_are_zero_for_exact_synthetic_solution():
    target, mask, dt, truth = _consistent_case()
    total, components = _loss(_criterion(), target.clone(), target, mask, dt, truth)

    assert total.item() == pytest.approx(0.0, abs=1e-12)
    assert components["signal"].item() == pytest.approx(0.0, abs=1e-12)
    assert components["physics"].item() == pytest.approx(0.0, abs=1e-12)


def test_complete_signal_and_physics_gradients_respect_mask():
    target, mask, dt, truth = _consistent_case()
    prediction = target.clone()
    prediction[:, 2, 0] += 0.3
    prediction[:, 3, 3] -= 0.2
    prediction.requires_grad_()
    completed = complete_signal(target, mask, prediction)

    assert torch.equal(completed[mask.bool()], target[mask.bool()])
    assert torch.equal(completed[~mask.bool()], prediction[~mask.bool()])
    total, _ = _loss(_criterion(), prediction, target, mask, dt, truth)
    total.backward()

    assert torch.count_nonzero(prediction.grad[mask.bool()]) == 0
    assert prediction.grad[:, 2, 0].abs().sum() > 0
    assert prediction.grad[:, 3, 3].abs().sum() > 0


@pytest.mark.parametrize(
    "phi",
    [
        torch.tensor([[1e-7, -2e-7, 3e-7]], dtype=torch.float64),
        torch.tensor([[0.2, -0.3, 0.1]], dtype=torch.float64),
    ],
)
def test_so3_log_exp_round_trip(phi):
    torch.testing.assert_close(so3_log(so3_exp(phi)), phi, atol=1e-9, rtol=1e-7)


def test_zero_gyro_preserves_orientation():
    gyro = torch.zeros(1, 6, 3, dtype=torch.float64)
    acceleration = torch.zeros_like(gyro)
    dt = torch.full((1, 6), 0.1, dtype=torch.float64)
    rotation0 = so3_exp(torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64))
    result = propagate_imu(
        gyro, acceleration, dt, rotation0, torch.zeros(1, 3, dtype=torch.float64),
        torch.zeros(1, 3, dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.rotation_body_to_world,
        rotation0[:, None].expand(-1, 6, -1, -1),
        atol=1e-12,
        rtol=1e-12,
    )


def test_constant_world_acceleration_matches_analytic_solution():
    steps, interval, acceleration_value = 11, 0.1, 2.0
    gyro = torch.zeros(1, steps, 3, dtype=torch.float64)
    acceleration = torch.zeros_like(gyro)
    acceleration[..., 0] = acceleration_value
    result = propagate_imu(
        gyro,
        acceleration,
        torch.full((1, steps), interval, dtype=torch.float64),
        torch.eye(3, dtype=torch.float64)[None],
        torch.zeros(1, 3, dtype=torch.float64),
        torch.zeros(1, 3, dtype=torch.float64),
    )
    elapsed = (steps - 1) * interval
    assert result.velocity_world_mps[0, -1, 0] == pytest.approx(
        acceleration_value * elapsed, abs=1e-12
    )
    assert result.position_world_m[0, -1, 0] == pytest.approx(
        0.5 * acceleration_value * elapsed**2, abs=1e-12
    )


def test_tensor_normalization_round_trip_preserves_values_and_gradient():
    scaler = RobustTrainScaler(
        center_=torch.arange(6, dtype=torch.float64).numpy(),
        scale_=torch.linspace(0.5, 1.5, 6, dtype=torch.float64).numpy(),
        training_ids=("train",),
    )
    values = torch.randn(3, 5, 6, dtype=torch.float64, requires_grad=True)
    restored = scaler.inverse_transform_tensor(scaler.transform_tensor(values))
    torch.testing.assert_close(restored, values, atol=1e-12, rtol=1e-12)
    restored.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_physics_loss_is_finite_on_available_devices(device):
    target, mask, dt, truth = _consistent_case(device=device, dtype=torch.float32)
    prediction = (target + 0.01 * torch.randn_like(target)).requires_grad_()
    total, components = _loss(_criterion(), prediction, target, mask, dt, truth)
    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in components.values())
    total.backward()
    assert torch.isfinite(prediction.grad).all()


def test_hybrid_mini_step_gives_both_branches_and_gate_gradients():
    pytest.importorskip("ncps")
    dtype = torch.float32
    target, mask, dt, truth = _consistent_case(dtype=dtype)
    features = torch.stack(
        [build_features(target[index], mask[index], dt[index]).values for index in range(2)]
    )
    model = HybridImputer(25, lnn_hidden_size=4, lstm_hidden_size=4)
    components = model.forward_components(
        features,
        dt,
        torch.cat((dt[:, -1:], dt[:, 1:].flip(-1)), dim=-1),
        features[..., :6],
        mask,
    )
    total, _ = _loss(_criterion(0.1), components.raw, target, mask, dt, truth)
    total.backward()

    groups = (model.lnn_branch, model.lstm_branch, model.gate_network)
    for group in groups:
        gradients = [parameter.grad for parameter in group.parameters()]
        assert any(gradient is not None and torch.any(gradient != 0) for gradient in gradients)


def test_real_nonzero_physics_is_fail_closed_until_frame_validation():
    config = {
        "lambda_physics": 0.1,
        "sigma_rotation_rad": 0.1,
        "sigma_velocity_mps": 0.5,
        "sigma_position_m": 0.5,
        "acceleration_mode": "gravity_compensated",
        "acceleration_unit": "G",
        "frame_validation_status": "diagnostic_only_not_validated",
    }
    with pytest.raises(ValueError, match="non-zero physics loss is gated"):
        _physics_loss_config(config)

    config["lambda_physics"] = 0.0
    assert _physics_loss_config(config).lambda_physics == 0.0
