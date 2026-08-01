import math

import pytest
import torch

import imputation_v3.models.baselines as baselines_module
from imputation_v3.models.baselines import (
    complete_signal,
    constant_velocity_rts,
    timestamp_linear,
    timestamp_locf,
    timestamp_pchip,
)


BASELINES = (timestamp_locf, timestamp_linear, timestamp_pchip)


def _run(baseline, source, mask, time):
    kwargs = {"empty_fill": -7.0}
    if baseline is constant_velocity_rts:
        kwargs.update(process_var=1e-2, observation_var=1e-3)
    return baseline(source, mask, time, **kwargs)


def test_linear_uses_real_timestamps_and_not_hidden_source_values():
    source = torch.tensor([[0.0], [12345.0], [4.0]], dtype=torch.float64)
    mask = torch.tensor([[1], [0], [1]])
    time = torch.tensor([0.0, 0.25, 1.0], dtype=torch.float64)

    result = timestamp_linear(source, mask, time, empty_fill=-9.0)

    torch.testing.assert_close(
        result, torch.tensor([[0.0], [1.0], [4.0]], dtype=torch.float64)
    )


@pytest.mark.parametrize(
    "path",
    (
        complete_signal,
        timestamp_locf,
        timestamp_linear,
        timestamp_pchip,
        constant_velocity_rts,
    ),
)
@pytest.mark.parametrize("hidden_value", (-123456.0, math.nan, math.inf, -math.inf))
def test_every_completion_path_preserves_observed_and_ignores_hidden_values(
    path, hidden_value
):
    source = torch.tensor(
        [[-0.0, 10.25], [hidden_value, hidden_value], [4.5, -3.75]],
        dtype=torch.float64,
    )
    reference = source.clone()
    reference[1] = 777.0
    mask = torch.tensor([[1, 1], [0, 0], [1, 1]], dtype=torch.bool)
    time = torch.tensor([0.0, 0.4, 1.0], dtype=torch.float64)

    if path is complete_signal:
        prediction = torch.tensor(
            [[math.nan, math.inf], [1.5, -2.0], [math.nan, -math.inf]],
            dtype=torch.float64,
        )
        result = path(source, mask, prediction)
        expected = path(reference, mask, prediction)
    else:
        result = _run(path, source, mask, time)
        expected = _run(path, reference, mask, time)

    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(result[mask], source[mask], rtol=0, atol=0)
    assert torch.equal(
        result[mask].contiguous().view(torch.int64),
        source[mask].contiguous().view(torch.int64),
    )
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("baseline", (*BASELINES, constant_velocity_rts))
def test_empty_channel_uses_empty_fill(baseline):
    source = torch.full((4, 2), math.nan, dtype=torch.float32)
    mask = torch.zeros_like(source, dtype=torch.bool)
    time = torch.tensor([0.0, 0.2, 0.7, 1.9])

    result = _run(baseline, source, mask, time)

    torch.testing.assert_close(result, torch.full_like(source, -7.0))


def test_single_observation_fill_policies():
    source = torch.tensor([[99.0], [5.0], [99.0], [99.0]])
    mask = torch.tensor([[0], [1], [0], [0]], dtype=torch.bool)
    time = torch.tensor([0.0, 1.0, 2.0, 3.0])

    expected_constant = torch.full_like(source, 5.0)
    torch.testing.assert_close(
        timestamp_linear(source, mask, time, empty_fill=-7.0), expected_constant
    )
    torch.testing.assert_close(
        timestamp_pchip(source, mask, time, empty_fill=-7.0), expected_constant
    )
    torch.testing.assert_close(
        timestamp_locf(source, mask, time, empty_fill=-7.0),
        torch.tensor([[-7.0], [5.0], [5.0], [5.0]]),
    )
    rts = _run(constant_velocity_rts, source, mask, time)
    assert torch.isfinite(rts).all()
    assert rts[1, 0].item() == source[1, 0].item()


@pytest.mark.parametrize("baseline", (timestamp_linear, timestamp_pchip))
def test_interpolation_holds_nearest_observation_outside_support(baseline):
    source = torch.tensor([[99.0], [2.0], [99.0], [8.0], [99.0]])
    mask = torch.tensor([[0], [1], [0], [1], [0]], dtype=torch.bool)
    time = torch.tensor([-1.0, 0.0, 1.0, 3.0, 10.0])

    result = baseline(source, mask, time, empty_fill=-7.0)

    torch.testing.assert_close(result[[0, 4], 0], torch.tensor([2.0, 8.0]))
    assert 2.0 < result[2, 0] < 8.0


def test_pchip_is_shape_preserving_for_a_monotone_nonlinear_signal():
    source = torch.tensor([[0.0], [99.0], [1.0], [99.0], [4.0]], dtype=torch.float64)
    mask = torch.tensor([[1], [0], [1], [0], [1]], dtype=torch.bool)
    time = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], dtype=torch.float64)

    result = timestamp_pchip(source, mask, time, empty_fill=-7.0)

    torch.testing.assert_close(
        result[:, 0],
        torch.tensor([0.0, 0.3125, 1.0, 2.1875, 4.0], dtype=torch.float64),
    )


def test_baselines_allow_a_one_sample_sequence():
    source = torch.tensor([[3.25]], dtype=torch.float32)
    mask = torch.ones_like(source, dtype=torch.bool)
    time = torch.tensor([4.0])

    for baseline in (*BASELINES, constant_velocity_rts):
        torch.testing.assert_close(_run(baseline, source, mask, time), source)


def test_rts_reconstructs_constant_velocity_interior_points_sensibly():
    time = torch.tensor([0.0, 0.3, 0.9, 1.7, 2.4], dtype=torch.float64)
    truth = (1.5 + 2.25 * time)[:, None]
    source = truth.clone()
    source[1:4] = torch.tensor([[math.nan], [math.inf], [-math.inf]])
    mask = torch.tensor([[1], [0], [0], [0], [1]], dtype=torch.bool)

    result = constant_velocity_rts(
        source,
        mask,
        time,
        empty_fill=0.0,
        process_var=1e-4,
        observation_var=1e-5,
    )

    torch.testing.assert_close(result[1:4], truth[1:4], rtol=0.02, atol=0.05)
    torch.testing.assert_close(result[mask], truth[mask], rtol=0, atol=0)
    assert torch.isfinite(result).all()


def test_rts_diffuse_prior_reconstructs_a_substantial_leading_gap():
    time = torch.tensor([0.0, 0.2, 0.21], dtype=torch.float64)
    truth = (1.0 + 2.0 * time)[:, None]
    mask = torch.tensor([[0], [1], [1]], dtype=torch.bool)
    source = torch.where(mask, truth, torch.full_like(truth, math.nan))

    result = constant_velocity_rts(
        source,
        mask,
        time,
        empty_fill=0.0,
        process_var=1e-4,
        observation_var=1e-5,
    )

    # Two distinct, exact analytic measurements identify the line. The 2e-2
    # allowance covers model weighting but is far below the old 7e-2 bias.
    torch.testing.assert_close(result[0], truth[0], rtol=0, atol=2e-2)


def test_rts_processes_boundary_measurements_once_before_a_trailing_gap():
    time = torch.tensor([0.0, 0.01, 0.8], dtype=torch.float64)
    truth = (1.0 + 2.0 * time)[:, None]
    mask = torch.tensor([[1], [1], [0]], dtype=torch.bool)
    source = torch.where(mask, truth, torch.full_like(truth, math.inf))

    result = constant_velocity_rts(
        source,
        mask,
        time,
        empty_fill=0.0,
        process_var=1e-4,
        observation_var=1e-5,
    )

    # The diffuse prior lets two exact measurements identify velocity. The
    # allowance is far below the old 2.65e-1 bias from reusing row zero.
    torch.testing.assert_close(result[2], truth[2], rtol=0, atol=2e-2)


def test_rts_solve_failure_reports_channel_and_time_index(monkeypatch):
    source = torch.tensor(
        [[math.nan, 1.0], [math.nan, math.nan], [math.nan, 3.0]],
        dtype=torch.float64,
    )
    mask = torch.tensor([[0, 1], [0, 0], [0, 1]], dtype=torch.bool)
    time = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)

    def fail_solve(*args, **kwargs):
        raise RuntimeError("singular test system")

    monkeypatch.setattr(torch.linalg, "solve", fail_solve)

    with pytest.raises(ValueError, match="channel 1 at time index 1"):
        constant_velocity_rts(
            source,
            mask,
            time,
            empty_fill=0.0,
            process_var=1e-4,
            observation_var=1e-5,
        )


def test_rts_does_not_translate_unrelated_runtime_errors(monkeypatch):
    source = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    mask = torch.ones_like(source, dtype=torch.bool)
    time = torch.tensor([0.0, 1.0], dtype=torch.float64)

    def fail_transition(*args, **kwargs):
        raise RuntimeError("unrelated allocation failure")

    monkeypatch.setattr(baselines_module, "_transition_and_noise", fail_transition)

    with pytest.raises(RuntimeError, match="unrelated allocation failure"):
        constant_velocity_rts(
            source,
            mask,
            time,
            empty_fill=0.0,
            process_var=1e-4,
            observation_var=1e-5,
        )


def test_complete_signal_rejects_only_non_finite_predictions_that_are_used():
    observed = torch.tensor([[1.0], [2.0]])
    mask = torch.tensor([[1], [0]], dtype=torch.bool)

    okay = complete_signal(observed, mask, torch.tensor([[math.nan], [3.0]]))
    torch.testing.assert_close(okay, torch.tensor([[1.0], [3.0]]))

    with pytest.raises(ValueError, match="prediction"):
        complete_signal(observed, mask, torch.tensor([[0.0], [math.nan]]))


@pytest.mark.parametrize("dtype", (torch.int64, torch.bool))
def test_complete_signal_rejects_non_floating_predictions(dtype):
    observed = torch.ones(2, 1, dtype=torch.float32)
    mask = torch.tensor([[1], [0]], dtype=torch.bool)
    prediction = torch.ones(2, 1, dtype=dtype)

    with pytest.raises(TypeError, match="floating"):
        complete_signal(observed, mask, prediction)


def test_complete_signal_rejects_mixed_floating_prediction_dtype():
    observed = torch.ones(2, 1, dtype=torch.float32)
    mask = torch.tensor([[1], [0]], dtype=torch.bool)
    prediction = torch.ones(2, 1, dtype=torch.float64)

    with pytest.raises(ValueError, match="dtype"):
        complete_signal(observed, mask, prediction)


def test_complete_signal_preserves_observed_dtype_without_promotion():
    observed = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    mask = torch.tensor([[1], [0]], dtype=torch.bool)
    prediction = torch.tensor([[99.0], [3.0]], dtype=torch.float64)

    result = complete_signal(observed, mask, prediction)

    assert result.dtype == observed.dtype
    torch.testing.assert_close(
        result, torch.tensor([[1.0], [3.0]], dtype=torch.float64)
    )


def test_complete_signal_rejects_prediction_on_a_different_device():
    observed = torch.ones(2, 1)
    mask = torch.ones_like(observed, dtype=torch.bool)
    prediction = torch.ones(2, 1, device="meta")

    with pytest.raises(ValueError, match="device"):
        complete_signal(observed, mask, prediction)


@pytest.mark.parametrize(
    ("source", "mask", "prediction", "error", "message"),
    [
        (None, torch.ones(2, 1), torch.ones(2, 1), TypeError, "tensor"),
        (torch.ones(2), torch.ones(2), torch.ones(2), ValueError, "2-D"),
        (
            torch.empty(0, 1),
            torch.empty(0, 1),
            torch.empty(0, 1),
            ValueError,
            "non-empty",
        ),
        (
            torch.ones(2, 1, dtype=torch.int64),
            torch.ones(2, 1),
            torch.ones(2, 1),
            TypeError,
            "floating",
        ),
        (torch.ones(2, 1), torch.ones(3, 1), torch.ones(2, 1), ValueError, "shape"),
        (torch.ones(2, 1), torch.ones(2, 1), torch.ones(2, 2), ValueError, "shape"),
        (
            torch.ones(2, 1),
            torch.full((2, 1), 0.99999),
            torch.ones(2, 1),
            ValueError,
            "0 or 1",
        ),
        (
            torch.ones(2, 1),
            torch.ones(2, 1, dtype=torch.complex64),
            torch.ones(2, 1),
            TypeError,
            "mask",
        ),
    ],
)
def test_complete_signal_rejects_malformed_inputs(
    source, mask, prediction, error, message
):
    with pytest.raises(error, match=message):
        complete_signal(source, mask, prediction)


@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf))
def test_observed_non_finite_source_is_rejected(invalid):
    source = torch.tensor([[1.0], [invalid], [3.0]])
    time = torch.tensor([0.0, 1.0, 2.0])
    observed_mask = torch.ones_like(source)

    with pytest.raises(ValueError, match="observed"):
        timestamp_linear(source, observed_mask, time, empty_fill=0.0)
    with pytest.raises(ValueError, match="observed"):
        complete_signal(source, observed_mask, torch.zeros_like(source))


@pytest.mark.parametrize(
    ("source", "mask", "time", "error", "message"),
    [
        (torch.ones(2), torch.ones(2), torch.arange(2.0), ValueError, "2-D"),
        (
            torch.empty(0, 1),
            torch.empty(0, 1),
            torch.empty(0),
            ValueError,
            "non-empty",
        ),
        (
            torch.ones(2, 1, dtype=torch.int64),
            torch.ones(2, 1),
            torch.arange(2.0),
            TypeError,
            "floating",
        ),
        (torch.ones(2, 1), torch.ones(3, 1), torch.arange(2.0), ValueError, "shape"),
        (
            torch.ones(2, 1),
            torch.full((2, 1), 0.5),
            torch.arange(2.0),
            ValueError,
            "0 or 1",
        ),
        (torch.ones(2, 1), torch.ones(2, 1), torch.ones(2, 1), ValueError, "time"),
        (torch.ones(2, 1), torch.ones(2, 1), torch.ones(3), ValueError, "time"),
        (torch.ones(2, 1), torch.ones(2, 1), torch.arange(2), TypeError, "floating"),
        (
            torch.ones(2, 1),
            torch.ones(2, 1),
            torch.tensor([0.0, math.nan]),
            ValueError,
            "finite",
        ),
        (
            torch.ones(2, 1),
            torch.ones(2, 1),
            torch.tensor([0.0, math.inf]),
            ValueError,
            "finite",
        ),
        (
            torch.ones(2, 1),
            torch.ones(2, 1),
            torch.tensor([0.0, 0.0]),
            ValueError,
            "increasing",
        ),
        (
            torch.ones(2, 1),
            torch.ones(2, 1),
            torch.tensor([1.0, 0.0]),
            ValueError,
            "increasing",
        ),
    ],
)
def test_baselines_reject_malformed_shared_inputs(source, mask, time, error, message):
    with pytest.raises(error, match=message):
        timestamp_locf(source, mask, time, empty_fill=0.0)


@pytest.mark.parametrize("fill", (True, "0", math.nan, math.inf, -math.inf))
def test_baselines_reject_invalid_empty_fill(fill):
    source = torch.ones(2, 1)
    mask = torch.ones_like(source)
    time = torch.arange(2.0)

    for baseline in (*BASELINES, constant_velocity_rts):
        kwargs = {"empty_fill": fill}
        if baseline is constant_velocity_rts:
            kwargs.update(process_var=1.0, observation_var=1.0)
        with pytest.raises((TypeError, ValueError), match="empty_fill"):
            baseline(source, mask, time, **kwargs)


@pytest.mark.parametrize("name", ("process_var", "observation_var"))
@pytest.mark.parametrize("value", (True, "1", 0.0, -1.0, math.nan, math.inf))
def test_rts_rejects_invalid_variances(name, value):
    source = torch.ones(2, 1)
    mask = torch.ones_like(source)
    time = torch.arange(2.0)
    kwargs = dict(empty_fill=0.0, process_var=1.0, observation_var=1.0)
    kwargs[name] = value

    with pytest.raises((TypeError, ValueError), match=name):
        constant_velocity_rts(source, mask, time, **kwargs)


def test_inputs_are_not_mutated_and_outputs_restore_source_dtype_and_device():
    source = torch.tensor([[1.0], [99.0], [4.0]], dtype=torch.float32)
    mask = torch.tensor([[1], [0], [1]], dtype=torch.int64)
    time = torch.tensor([0.0, 0.2, 1.0], dtype=torch.float64)
    originals = source.clone(), mask.clone(), time.clone()

    for baseline in (*BASELINES, constant_velocity_rts):
        result = _run(baseline, source, mask, time)
        assert result.dtype == source.dtype
        assert result.device == source.device

    for actual, expected in zip((source, mask, time), originals):
        torch.testing.assert_close(actual, expected)


def test_source_mask_and_time_must_share_a_device():
    source = torch.ones(2, 1)
    mask = torch.ones_like(source)
    time = torch.arange(2.0)

    with pytest.raises(ValueError, match="device"):
        timestamp_linear(source, mask.to("meta"), time, empty_fill=0.0)
    with pytest.raises(ValueError, match="device"):
        timestamp_linear(source, mask, time.to("meta"), empty_fill=0.0)
