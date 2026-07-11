from dataclasses import FrozenInstanceError

import pytest
import torch

from validation_v2.data.features import build_features
from validation_v2.data.masking import (
    channel_outage,
    contiguous_block,
    generate_interval_jittered_time,
    point_missing,
)
from validation_v2.data.windows import make_windows
from validation_v2.types import (
    FeatureBatch,
    IrregularTimeResult,
    MaskResult,
    WindowBatch,
)


def _feature_batch(source):
    return FeatureBatch(values=source, dt=torch.ones(2), mask=torch.ones(2, 6))


def _mask_result(source):
    return MaskResult(
        mask=source,
        requested_fraction=0.5,
        realized_fraction=0.5,
        topology="point_missing",
        seed=1,
    )


def _irregular_time_result(source):
    return IrregularTimeResult(
        time=source,
        dt=torch.ones(2),
        retained_indices=torch.arange(2),
        requested_irregularity=0.5,
        realized_irregularity=0.5,
        method="interval_jitter",
        seed=1,
    )


def _window_batch(source):
    return WindowBatch(
        target=source,
        mask=torch.ones(2, 6),
        dt=torch.ones(2),
        index=torch.arange(2),
        time=torch.arange(2, dtype=torch.float32),
        recording_id="recording",
    )


@pytest.mark.parametrize(
    ("factory", "attribute", "source"),
    [
        (_feature_batch, "values", torch.arange(12, dtype=torch.float32).reshape(2, 6)),
        (_mask_result, "mask", torch.ones(2, 6)),
        (_irregular_time_result, "time", torch.tensor([0.0, 1.0])),
        (_window_batch, "target", torch.arange(12, dtype=torch.float32).reshape(2, 6)),
    ],
)
def test_public_tensor_state_is_isolated_from_callers_and_returned_copies(
    factory, attribute, source
):
    expected = source.clone()
    result = factory(source)

    source.zero_()
    first_read = getattr(result, attribute)
    torch.testing.assert_close(first_read, expected)
    first_read.zero_()
    second_read = getattr(result, attribute)
    torch.testing.assert_close(second_read, expected)
    second_read.reshape(-1)[0] = -999
    torch.testing.assert_close(getattr(result, attribute), expected)
    with pytest.raises(FrozenInstanceError):
        setattr(result, attribute, expected)


def test_hidden_targets_cannot_change_model_input():
    target_a = torch.arange(60, dtype=torch.float32).reshape(10, 6)
    target_b = target_a.clone()
    mask = torch.ones_like(target_a)
    mask[3:7, 2:5] = 0
    target_b[mask == 0] += 10_000

    input_a = build_features(target_a, mask, torch.full((10,), 0.01))
    input_b = build_features(target_b, mask, torch.full((10,), 0.01))

    torch.testing.assert_close(input_a.values, input_b.values, rtol=0, atol=0)


def test_features_are_observed_only_and_have_the_declared_25_column_order():
    target = torch.arange(18, dtype=torch.float64).reshape(3, 6)
    mask = torch.ones_like(target)
    mask[1, 2] = 0
    target[1, 2] = torch.nan
    dt = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

    batch = build_features(target, mask, dt)
    observed = torch.where(mask.bool(), target, torch.zeros_like(target))
    valid_delta = torch.zeros_like(mask)
    valid_delta[1:] = mask[1:] * mask[:-1]
    delta = torch.zeros_like(target)
    delta[1:] = torch.where(
        valid_delta[1:].bool(), observed[1:] - observed[:-1], 0.0
    )

    assert batch.values.shape == (3, 25)
    torch.testing.assert_close(batch.values[:, 0:6], observed)
    torch.testing.assert_close(batch.values[:, 6:12], mask)
    torch.testing.assert_close(batch.values[:, 12], dt)
    torch.testing.assert_close(batch.values[:, 13:19], delta)
    torch.testing.assert_close(batch.values[:, 19:25], valid_delta)
    assert torch.isfinite(batch.values).all()


def test_feature_inputs_are_not_modified_and_batch_fields_are_frozen():
    target = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    mask = torch.ones_like(target)
    dt = torch.tensor([0.1, 0.1])
    originals = (target.clone(), mask.clone(), dt.clone())

    batch = build_features(target, mask, dt)

    for value, original in zip((target, mask, dt), originals):
        torch.testing.assert_close(value, original)
    with pytest.raises(FrozenInstanceError):
        batch.values = torch.empty(0)


@pytest.mark.parametrize(
    ("target", "mask", "dt"),
    [
        (torch.ones(2, 5), torch.ones(2, 5), torch.ones(2)),
        (torch.ones(2, 6), torch.ones(3, 6), torch.ones(2)),
        (torch.ones(2, 6), torch.full((2, 6), 0.5), torch.ones(2)),
        (torch.ones(2, 6), torch.ones(2, 6), torch.ones(3)),
        (torch.ones(2, 6), torch.ones(2, 6), torch.tensor([0.1, 0.0])),
        (torch.ones(2, 6), torch.ones(2, 6), torch.tensor([0.1, torch.nan])),
    ],
)
def test_features_reject_bad_shape_mask_values_and_dt(target, mask, dt):
    with pytest.raises((TypeError, ValueError)):
        build_features(target, mask, dt)


def test_features_allow_only_hidden_nan_targets():
    hidden_nan = torch.ones(2, 6)
    hidden_nan[0, 0] = torch.nan
    mask = torch.ones_like(hidden_nan)
    mask[0, 0] = 0
    build_features(hidden_nan, mask, torch.ones(2))

    observed_nan = hidden_nan.clone()
    observed_mask = mask.clone()
    observed_mask[0, 0] = 1
    with pytest.raises(ValueError, match="observed target"):
        build_features(observed_nan, observed_mask, torch.ones(2))
    hidden_inf = hidden_nan.nan_to_num(0.0)
    hidden_inf[0, 0] = torch.inf
    with pytest.raises(ValueError, match="infinite"):
        build_features(hidden_inf, mask, torch.ones(2))


@pytest.mark.parametrize(
    ("generator", "topology"),
    [
        (point_missing, "point_missing"),
        (contiguous_block, "contiguous_block"),
        (channel_outage, "channel_outage"),
    ],
)
def test_masks_are_deterministic_binary_and_report_realized_fraction(generator, topology):
    template = torch.arange(600, dtype=torch.float32).reshape(100, 6)
    original = template.clone()

    first = generator(template, requested_fraction=0.30, seed=7)
    second = generator(template, requested_fraction=0.30, seed=7)

    assert torch.equal(first.mask, second.mask)
    assert first.mask.shape == template.shape
    assert first.mask.dtype == template.dtype
    assert set(first.mask.unique().tolist()) <= {0.0, 1.0}
    assert first.requested_fraction == 0.30
    assert first.realized_fraction == pytest.approx((first.mask == 0).float().mean().item())
    assert first.topology == topology
    assert first.seed == 7
    torch.testing.assert_close(template, original)


def test_point_mask_uses_half_up_rounding_for_an_exact_flat_missing_count():
    result = point_missing(torch.ones(3, 3), requested_fraction=0.50, seed=1)
    assert int((result.mask == 0).sum()) == 5
    assert result.realized_fraction == pytest.approx(5 / 9)


def test_block_mask_is_one_in_bounds_contiguous_time_interval():
    result = contiguous_block(torch.ones(10, 6), requested_fraction=0.30, seed=4)
    missing_rows = torch.where((result.mask == 0).all(dim=1))[0]
    assert len(missing_rows) == 3
    assert torch.equal(missing_rows, torch.arange(missing_rows[0], missing_rows[0] + 3))
    assert (result.mask[~torch.isin(torch.arange(10), missing_rows)] == 1).all()


def test_channel_mask_reports_floor_discrete_realized_rate():
    result = channel_outage(torch.ones(100, 6), requested_fraction=0.30, seed=7)
    assert result.masked_channels == 1
    assert result.realized_fraction == pytest.approx(1 / 6)
    assert int((result.mask == 0).all(dim=0).sum()) == 1


def test_positive_channel_rate_masks_at_least_one_channel():
    result = channel_outage(torch.ones(100, 6), requested_fraction=0.10, seed=7)

    assert result.masked_channels == 1
    assert result.realized_fraction == pytest.approx(1 / 6)
    assert int((result.mask == 0).all(dim=0).sum()) == 1


@pytest.mark.parametrize("fraction", [0.0, 1.0])
@pytest.mark.parametrize("generator", [point_missing, contiguous_block, channel_outage])
def test_mask_fraction_boundaries(generator, fraction):
    result = generator(torch.ones(5, 6), requested_fraction=fraction, seed=2)
    assert result.realized_fraction == fraction


@pytest.mark.parametrize("fraction", [-0.01, 1.01, torch.nan, torch.inf])
@pytest.mark.parametrize("generator", [point_missing, contiguous_block, channel_outage])
def test_mask_rejects_invalid_requested_fraction(generator, fraction):
    with pytest.raises(ValueError, match="requested_fraction"):
        generator(torch.ones(5, 6), requested_fraction=fraction, seed=2)


def test_interval_jitter_is_deterministic_strictly_increasing_and_not_a_value_mask():
    original_time = torch.arange(10, dtype=torch.float64) * 0.1

    first = generate_interval_jittered_time(
        original_time, requested_irregularity=0.4, seed=9, jitter_fraction=0.25
    )
    second = generate_interval_jittered_time(
        original_time, requested_irregularity=0.4, seed=9, jitter_fraction=0.25
    )

    torch.testing.assert_close(first.time, second.time, rtol=0, atol=0)
    assert (first.time[1:] > first.time[:-1]).all()
    assert (first.dt > 0).all()
    assert torch.equal(first.retained_indices, torch.arange(10))
    assert first.requested_irregularity == 0.4
    assert first.realized_irregularity == pytest.approx(4 / 9)
    assert first.method == "interval_jitter"
    assert not hasattr(first, "mask")
    torch.testing.assert_close(original_time, torch.arange(10, dtype=torch.float64) * 0.1)


def test_interval_jitter_preserves_endpoints_duration_and_changes_two_intervals():
    original_time = torch.tensor(
        [0.0, 0.07, 0.21, 0.30, 0.55, 0.8], dtype=torch.float64
    )

    result = generate_interval_jittered_time(
        original_time, requested_irregularity=0.01, seed=19
    )

    assert result.time[0].item() == original_time[0].item()
    assert result.time[-1].item() == original_time[-1].item()
    assert torch.all(torch.diff(result.time) > 0)
    assert torch.sum(result.dt[1:]).item() == pytest.approx(
        (original_time[-1] - original_time[0]).item()
    )
    changed = ~torch.isclose(
        torch.diff(result.time),
        torch.diff(original_time),
        rtol=0.0,
        atol=torch.finfo(original_time.dtype).eps * 8,
    )
    assert int(changed.sum()) >= 2
    assert result.realized_irregularity == pytest.approx(
        changed.to(torch.float64).mean().item()
    )


def test_windows_do_not_cross_recording_or_index_discontinuities_and_preserve_fields():
    target = torch.arange(72, dtype=torch.float32).reshape(12, 6)
    mask = torch.ones_like(target)
    dt = torch.full((12,), 0.1)
    index = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 10, 11, 12, 13])
    time = torch.arange(12, dtype=torch.float32) * 0.1
    recording_id = ["a"] * 5 + ["b"] * 7

    windows = make_windows(
        target, mask, dt, index, time, recording_id, window_size=3
    )

    assert [(window.recording_id, window.index.tolist()) for window in windows] == [
        ("a", [0, 1, 2]),
        ("b", [0, 1, 2]),
        ("b", [10, 11, 12]),
    ]
    for window in windows:
        assert window.target.shape == (3, 6)
        assert window.mask.shape == (3, 6)
        assert window.dt.shape == (3,)
        assert window.time.shape == (3,)


def test_windows_default_to_nonoverlap_and_partial_window_behavior_is_explicit():
    target = torch.arange(30, dtype=torch.float32).reshape(5, 6)
    mask = torch.ones_like(target)
    dt = torch.ones(5)
    index = torch.arange(5)
    time = torch.arange(5, dtype=torch.float32)

    dropped = make_windows(target, mask, dt, index, time, "r", window_size=3)
    kept = make_windows(
        target, mask, dt, index, time, "r", window_size=3, drop_last=False
    )
    overlapped = make_windows(
        target, mask, dt, index, time, "r", window_size=3, stride=1
    )

    assert [window.index.tolist() for window in dropped] == [[0, 1, 2]]
    assert [window.index.tolist() for window in kept] == [[0, 1, 2], [3, 4]]
    assert [window.index.tolist() for window in overlapped] == [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
    ]
