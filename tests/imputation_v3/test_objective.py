import math

import pytest
import torch

from imputation_v3.objectives.reconstruction import channel_balanced_missing_mse


def test_channel_means_are_equally_weighted_despite_unequal_missing_counts():
    prediction = torch.zeros(1, 3, 6, dtype=torch.float64)
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[0, 0, 0] = False
    target[0, 0, 0] = 2.0  # channel mean squared error: 4
    mask[0, :, 1] = False
    target[0, :, 1] = 1.0  # channel mean squared error: 1

    loss = channel_balanced_missing_mse(prediction, target, mask)

    torch.testing.assert_close(loss, torch.tensor(2.5, dtype=torch.float64))
    assert loss.dtype == torch.float64
    assert loss.ndim == 0


@pytest.mark.parametrize(
    ("count", "error", "expected"),
    ((70_000, 1.0, 1.0), (1_000, 100.0, 10_000.0)),
)
def test_float16_reduction_stays_finite_when_mean_is_representable(
    count, error, expected
):
    prediction = torch.full((count, 6), error, dtype=torch.float16)
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[:, 0] = False

    loss = channel_balanced_missing_mse(prediction, target, mask)

    assert loss.dtype == torch.float16
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.tensor(expected, dtype=torch.float16))


def test_float32_normalizes_before_summing_large_squared_errors():
    prediction = torch.full((4_096, 6), 1e18, dtype=torch.float32)
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[:, 2] = False

    loss = channel_balanced_missing_mse(prediction, target, mask)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.tensor(1e36), rtol=2e-6, atol=0)


def test_float32_channel_average_does_not_overflow_before_division():
    prediction = torch.full((6, 6), 1e19, dtype=torch.float32)
    target = torch.zeros_like(prediction)
    mask = torch.zeros_like(prediction, dtype=torch.bool)

    loss = channel_balanced_missing_mse(prediction, target, mask)

    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.tensor(1e38), rtol=2e-6, atol=0)


def test_stable_scaling_has_finite_gradients_and_exact_observed_zeros():
    prediction = torch.full(
        (1_000, 6), 100.0, dtype=torch.float16, requires_grad=True
    )
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[:, 4] = False

    channel_balanced_missing_mse(prediction, target, mask).backward()

    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[mask]).item() == 0
    assert torch.all(prediction.grad[~mask] != 0)


def test_nonfinite_result_after_original_dtype_cast_is_rejected():
    prediction = torch.full((32, 6), 300.0, dtype=torch.float16)
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[:, 0] = False

    with pytest.raises(ValueError, match="finite"):
        channel_balanced_missing_mse(prediction, target, mask)


@pytest.mark.parametrize("dtype", (torch.float16, torch.float32, torch.float64))
def test_loss_scalar_uses_original_floating_dtype(dtype):
    prediction = torch.zeros(2, 6, dtype=dtype)
    target = torch.ones_like(prediction)
    mask = torch.zeros_like(prediction, dtype=torch.bool)

    loss = channel_balanced_missing_mse(prediction, target, mask)

    assert loss.dtype == dtype
    torch.testing.assert_close(loss, torch.tensor(1.0, dtype=dtype))


def test_observed_gradients_are_exactly_zero_and_missing_errors_have_gradients():
    prediction = torch.zeros(1, 2, 6, requires_grad=True)
    target = torch.ones_like(prediction)
    mask = torch.tensor(
        [[[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]]], dtype=torch.bool
    )
    loss = channel_balanced_missing_mse(prediction, target, mask)
    loss.backward()

    assert torch.count_nonzero(prediction.grad[mask]).item() == 0
    assert torch.all(prediction.grad[~mask] != 0)


def test_channels_with_no_missing_entries_are_excluded():
    prediction = torch.zeros(1, 2, 6)
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[0, 0, 3] = False
    target[0, 0, 3] = 4.0
    target[..., 0] = 1000.0

    torch.testing.assert_close(
        channel_balanced_missing_mse(prediction, target, mask), torch.tensor(16.0)
    )


def test_no_missing_values_are_rejected():
    values = torch.zeros(2, 3, 6)
    with pytest.raises(ValueError, match="missing"):
        channel_balanced_missing_mse(values, values, torch.ones_like(values))


def test_nonfinite_values_are_rejected_only_when_consumed_by_loss():
    prediction = torch.zeros(1, 2, 6)
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[0, 1, 0] = False
    target[0, 0, 0] = math.nan

    loss = channel_balanced_missing_mse(prediction, target, mask)
    torch.testing.assert_close(loss, torch.tensor(0.0))

    target[0, 1, 0] = math.inf
    with pytest.raises(ValueError, match="finite"):
        channel_balanced_missing_mse(prediction, target, mask)


@pytest.mark.parametrize(
    ("prediction", "target", "mask", "error", "message"),
    (
        (None, torch.zeros(1, 1, 6), torch.zeros(1, 1, 6), TypeError, "tensor"),
        (torch.zeros(1, 1, 6), None, torch.zeros(1, 1, 6), TypeError, "tensor"),
        (torch.zeros(1, 1, 6), torch.zeros(1, 1, 6), None, TypeError, "tensor"),
        (
            torch.zeros(1, 1, 5),
            torch.zeros(1, 1, 5),
            torch.zeros(1, 1, 5),
            ValueError,
            "6",
        ),
        (
            torch.empty(1, 0, 6),
            torch.empty(1, 0, 6),
            torch.empty(1, 0, 6),
            ValueError,
            "nonempty",
        ),
        (
            torch.zeros(1, 1, 6),
            torch.zeros(2, 1, 6),
            torch.zeros(1, 1, 6),
            ValueError,
            "shape",
        ),
        (
            torch.zeros(1, 1, 6),
            torch.zeros(1, 1, 6),
            torch.zeros(2, 1, 6),
            ValueError,
            "shape",
        ),
        (
            torch.zeros(1, 1, 6, dtype=torch.int64),
            torch.zeros(1, 1, 6),
            torch.zeros(1, 1, 6),
            TypeError,
            "floating",
        ),
        (
            torch.zeros(1, 1, 6),
            torch.zeros(1, 1, 6, dtype=torch.float64),
            torch.zeros(1, 1, 6),
            TypeError,
            "dtype",
        ),
        (
            torch.zeros(1, 1, 6),
            torch.zeros(1, 1, 6),
            torch.zeros(1, 1, 6, dtype=torch.int64),
            TypeError,
            "bool|dtype",
        ),
        (
            torch.zeros(1, 1, 6),
            torch.zeros(1, 1, 6),
            torch.full((1, 1, 6), 0.5),
            ValueError,
            "binary|0 or 1",
        ),
    ),
)
def test_invalid_inputs_are_rejected(prediction, target, mask, error, message):
    with pytest.raises(error, match=message):
        channel_balanced_missing_mse(prediction, target, mask)


def test_float_mask_must_use_prediction_dtype_and_exact_binary_values():
    prediction = torch.zeros(1, 1, 6, dtype=torch.float64)
    target = torch.ones_like(prediction)
    mask = torch.zeros_like(prediction)
    assert torch.isfinite(channel_balanced_missing_mse(prediction, target, mask))

    with pytest.raises(TypeError, match="dtype"):
        channel_balanced_missing_mse(prediction, target, mask.float())


def test_different_devices_are_rejected():
    prediction = torch.zeros(1, 1, 6)
    target = torch.zeros(1, 1, 6, device="meta")
    mask = torch.zeros(1, 1, 6)
    with pytest.raises(ValueError, match="device"):
        channel_balanced_missing_mse(prediction, target, mask)

    with pytest.raises(ValueError, match="device"):
        channel_balanced_missing_mse(prediction, prediction, mask.to("meta"))
