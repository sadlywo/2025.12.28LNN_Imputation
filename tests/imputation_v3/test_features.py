from dataclasses import FrozenInstanceError

import pytest
import torch

from imputation_v3.data.features import build_features
from imputation_v3.types import FeatureBatch


def _valid_inputs(
    *, samples: int = 3, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.arange(samples * 6, dtype=dtype).reshape(samples, 6)
    mask = torch.ones_like(target)
    dt = torch.full((samples,), 0.1, dtype=dtype)
    return target, mask, dt


def test_hidden_target_perturbations_cannot_change_any_feature_bit():
    target, mask, dt = _valid_inputs(samples=5, dtype=torch.float64)
    mask[1:4, 1::2] = 0
    perturbed = target.clone()
    perturbed[mask == 0] = perturbed[mask == 0] * -1000 + 17

    original_features = build_features(target, mask, dt).values
    perturbed_features = build_features(perturbed, mask, dt).values

    torch.testing.assert_close(
        original_features, perturbed_features, rtol=0, atol=0
    )


def test_features_have_exact_31_column_temporal_contract():
    target = torch.tensor([1.0, 2.0, -999.0, 4.0])[:, None].repeat(1, 6)
    mask = torch.ones_like(target)
    mask[2] = 0
    dt = torch.full((4,), 0.1)

    values = build_features(target, mask, dt).values

    assert values.shape == (4, 31)
    torch.testing.assert_close(values[:, 0:6], target * mask)
    torch.testing.assert_close(values[:, 6:12], mask)
    torch.testing.assert_close(values[:, 12], dt)
    torch.testing.assert_close(values[2, 13:19], torch.full((6,), 0.1))
    torch.testing.assert_close(values[2, 19:25], torch.full((6,), 10.0))
    torch.testing.assert_close(values[2, 25:31], torch.ones(6))


def test_age_and_slope_use_real_time_and_only_prior_observations():
    target = torch.tensor(
        [
            [123.0] * 6,
            [2.0] * 6,
            [999.0] * 6,
            [8.0] * 6,
            [-999.0] * 6,
        ],
        dtype=torch.float64,
    )
    mask = torch.ones_like(target)
    mask[0] = 0
    mask[2] = 0
    mask[4] = 0
    dt = torch.tensor([0.4, 0.2, 0.3, 0.5, 0.7], dtype=torch.float64)

    values = build_features(target, mask, dt).values

    expected_age = torch.tensor([0.0, 0.0, 0.3, 0.0, 0.7], dtype=torch.float64)
    expected_slope = torch.tensor([0.0, 0.0, 0.0, 7.5, 7.5], dtype=torch.float64)
    expected_valid = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(values[:, 13:19], expected_age[:, None].repeat(1, 6))
    torch.testing.assert_close(values[:, 19:25], expected_slope[:, None].repeat(1, 6))
    torch.testing.assert_close(
        values[:, 25:31], expected_valid[:, None].repeat(1, 6)
    )


def test_feature_batch_clones_constructor_inputs_and_every_property_read():
    values = torch.arange(62, dtype=torch.float32).reshape(2, 31)
    dt = torch.tensor([0.1, 0.2])
    mask = torch.ones(2, 6)
    expected = (values.clone(), dt.clone(), mask.clone())

    batch = FeatureBatch(values=values, dt=dt, mask=mask)
    values.zero_()
    dt.zero_()
    mask.zero_()

    for name, wanted in zip(("values", "dt", "mask"), expected):
        first_read = getattr(batch, name)
        torch.testing.assert_close(first_read, wanted)
        first_read.zero_()
        torch.testing.assert_close(getattr(batch, name), wanted)
        with pytest.raises(FrozenInstanceError):
            setattr(batch, name, wanted)


@pytest.mark.parametrize(
    ("target", "mask", "dt", "error", "message"),
    [
        (None, torch.ones(2, 6), torch.ones(2), TypeError, "torch tensors"),
        (torch.ones(2, 6), None, torch.ones(2), TypeError, "torch tensors"),
        (torch.ones(2, 6), torch.ones(2, 6), None, TypeError, "torch tensors"),
        (torch.ones(2, 5), torch.ones(2, 5), torch.ones(2), ValueError, "shape"),
        (torch.ones(2, 6, 1), torch.ones(2, 6, 1), torch.ones(2), ValueError, "shape"),
        (torch.empty(0, 6), torch.empty(0, 6), torch.empty(0), ValueError, "sample"),
        (
            torch.ones(2, 6, dtype=torch.int64),
            torch.ones(2, 6),
            torch.ones(2),
            TypeError,
            "target",
        ),
        (torch.ones(2, 6), torch.ones(3, 6), torch.ones(2), ValueError, "mask"),
        (torch.ones(2, 6), torch.full((2, 6), 0.5), torch.ones(2), ValueError, "0 or 1"),
        (
            torch.ones(2, 6),
            torch.ones(2, 6, dtype=torch.complex64),
            torch.ones(2),
            TypeError,
            "mask",
        ),
        (torch.ones(2, 6), torch.ones(2, 6), torch.ones(2, 1), ValueError, "dt"),
        (torch.ones(2, 6), torch.ones(2, 6), torch.ones(3), ValueError, "dt"),
        (
            torch.ones(2, 6),
            torch.ones(2, 6),
            torch.ones(2, dtype=torch.int64),
            TypeError,
            "dt",
        ),
        (
            torch.ones(2, 6),
            torch.ones(2, 6),
            torch.tensor([0.1, torch.nan]),
            ValueError,
            "finite",
        ),
        (
            torch.ones(2, 6),
            torch.ones(2, 6),
            torch.tensor([0.1, torch.inf]),
            ValueError,
            "finite",
        ),
        (
            torch.ones(2, 6),
            torch.ones(2, 6),
            torch.tensor([0.1, 0.0]),
            ValueError,
            "positive",
        ),
        (
            torch.ones(2, 6),
            torch.ones(2, 6),
            torch.tensor([0.1, -0.1]),
            ValueError,
            "positive",
        ),
    ],
)
def test_features_reject_invalid_inputs(target, mask, dt, error, message):
    with pytest.raises(error, match=message):
        build_features(target, mask, dt)


@pytest.mark.parametrize("invalid", [torch.nan, torch.inf, -torch.inf])
def test_hidden_non_finite_values_are_ignored_but_observed_values_are_rejected(invalid):
    target, mask, dt = _valid_inputs(samples=3, dtype=torch.float64)
    mask[1, 2] = 0
    hidden_non_finite = target.clone()
    hidden_non_finite[1, 2] = invalid
    hidden_finite = target.clone()
    hidden_finite[1, 2] = -123456.0

    torch.testing.assert_close(
        build_features(hidden_non_finite, mask, dt).values,
        build_features(hidden_finite, mask, dt).values,
        rtol=0,
        atol=0,
    )

    observed_mask = mask.clone()
    observed_mask[1, 2] = 1
    with pytest.raises(ValueError, match="observed target"):
        build_features(hidden_non_finite, observed_mask, dt)


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.int64, torch.float64])
def test_mask_is_binary_converted_to_target_dtype_without_mutating_inputs(mask_dtype):
    target, _, dt = _valid_inputs(samples=3, dtype=torch.float32)
    mask = torch.tensor(
        [[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1]],
        dtype=mask_dtype,
    )
    originals = (target.clone(), mask.clone(), dt.clone())

    batch = build_features(target, mask, dt)

    assert batch.values.dtype == target.dtype
    assert batch.dt.dtype == target.dtype
    assert batch.mask.dtype == target.dtype
    for actual, expected in zip((target, mask, dt), originals):
        torch.testing.assert_close(actual, expected)


def test_dt_is_converted_to_target_dtype_for_deterministic_concatenation():
    target, mask, _ = _valid_inputs(samples=2, dtype=torch.float64)
    dt = torch.tensor([0.1, 0.2], dtype=torch.float32)

    batch = build_features(target, mask.bool(), dt)

    assert batch.values.dtype == torch.float64
    assert batch.dt.dtype == torch.float64
    torch.testing.assert_close(batch.values[:, 12], dt.to(torch.float64))


def test_inputs_must_share_a_device():
    target, mask, dt = _valid_inputs(samples=2)
    meta_mask = mask.to(device="meta")
    meta_dt = dt.to(device="meta")

    with pytest.raises(ValueError, match="device"):
        build_features(target, meta_mask, dt)
    with pytest.raises(ValueError, match="device"):
        build_features(target, mask, meta_dt)
