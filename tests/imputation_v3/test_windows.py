from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from imputation_v3.data.features import build_features
from imputation_v3.data.windows import (
    TOPOLOGY_GENERATOR_NAMES,
    collate_prepared_windows,
    iter_teacher_windows,
    materialize_teacher_windows,
)
from imputation_v3.types import PreparedBatch, PreparedWindow
from validation_v2.data.normalization import RobustTrainScaler
from validation_v2.types import Recording


def _recording(
    recording_id: str = "recording",
    *,
    rows: int = 40,
    time: np.ndarray | None = None,
    values: np.ndarray | None = None,
) -> Recording:
    time_values = (
        np.arange(rows, dtype=np.float64) * 0.01
        if time is None
        else np.asarray(time)
    )
    imu_values = (
        np.arange(rows * 6, dtype=np.float64).reshape(rows, 6) / 10.0
        if values is None
        else np.asarray(values)
    )
    return Recording(
        id=recording_id,
        imu_time_s=time_values,
        imu_six=imu_values,
        vicon_time_s=np.arange(max(rows, 1), dtype=np.float64),
        vicon_position_m=np.zeros((max(rows, 1), 3)),
        vicon_quaternion_xyzw=np.zeros((max(rows, 1), 4)),
        overlap_s=(0.0, float(max(rows - 1, 0))),
        metadata={},
    )


def _scaler() -> RobustTrainScaler:
    return RobustTrainScaler(
        center_=np.zeros(6), scale_=np.ones(6), training_ids=("training",)
    )


def _materialize(recordings, **overrides):
    arguments = dict(
        window_samples=16,
        stride=8,
        seed=2026,
        topologies=("point", "block", "channel"),
        rates=(0.25, 0.5),
        exhaustive=False,
    )
    arguments.update(overrides)
    return materialize_teacher_windows(recordings, _scaler(), **arguments)


def _prepared_arguments() -> dict[str, object]:
    target = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    mask = torch.ones_like(target)
    mask[1, 2] = 0
    observed = torch.where(mask.bool(), target, torch.zeros_like(target))
    dt = torch.tensor([0.1, 0.1, 0.2, 0.1])
    time = torch.tensor([0.0, 0.1, 0.3, 0.4])
    return {
        "features": build_features(target, mask, dt).values,
        "target": target,
        "observed": observed,
        "mask": mask,
        "dt": dt,
        "time": time,
        "baseline": observed.clone(),
        "window_id": "window",
        "recording_id": "recording",
        "topology": "point",
        "requested_fraction": 0.25,
        "realized_fraction": 1.0 / 24.0,
    }


def test_training_materialization_builds_complete_fixed_shape_windows():
    windows = _materialize([_recording()])

    assert len(windows) == 4
    assert len({window.window_id for window in windows}) == len(windows)
    for window in windows:
        assert window.target.shape == (16, 6)
        assert window.features.shape == (16, 31)
        assert window.observed.shape == window.mask.shape == window.baseline.shape


def test_exhaustive_materialization_emits_every_condition_for_each_window():
    windows = _materialize(
        [_recording()], stride=16, exhaustive=True
    )

    assert len(windows) == 12
    for start in (0, 16):
        target = torch.from_numpy(_recording().imu_six[start : start + 16]).float()
        physical = [window for window in windows if torch.equal(window.target, target)]
        assert {(window.topology, window.requested_fraction) for window in physical} == {
            (topology, rate)
            for topology in ("point", "block", "channel")
            for rate in (0.25, 0.5)
        }


def test_materialization_is_exactly_repeatable_and_input_order_independent():
    recordings = [_recording("z-recording"), _recording("a-recording")]
    first = _materialize(recordings)
    repeated = _materialize(recordings)
    reversed_input = _materialize(list(reversed(recordings)))

    for left_collection, right_collection in (
        (first, repeated),
        (first, reversed_input),
    ):
        assert [item.window_id for item in left_collection] == [
            item.window_id for item in right_collection
        ]
        for left, right in zip(left_collection, right_collection):
            assert (
                left.recording_id,
                left.topology,
                left.requested_fraction,
                left.realized_fraction,
            ) == (
                right.recording_id,
                right.topology,
                right.requested_fraction,
                right.realized_fraction,
            )
            for field in (
                "features",
                "target",
                "observed",
                "mask",
                "dt",
                "time",
                "baseline",
            ):
                assert torch.equal(getattr(left, field), getattr(right, field))


def test_exhaustive_condition_identity_is_independent_of_argument_order():
    first = _materialize([_recording()], stride=16, exhaustive=True)
    reordered = _materialize(
        [_recording()],
        stride=16,
        exhaustive=True,
        topologies=("channel", "point", "block"),
        rates=(0.5, 0.25),
    )

    def keyed(windows):
        return {
            (
                window.recording_id,
                tuple(window.time.tolist()),
                window.topology,
                window.requested_fraction,
            ): window
            for window in windows
        }

    first_by_condition = keyed(first)
    reordered_by_condition = keyed(reordered)
    assert first_by_condition.keys() == reordered_by_condition.keys()
    for key, window in first_by_condition.items():
        other = reordered_by_condition[key]
        assert window.window_id == other.window_id
        assert torch.equal(window.mask, other.mask)


def test_non_exhaustive_selection_is_independent_of_candidate_order_and_type():
    first = _materialize([_recording("b"), _recording("a")])
    reordered = _materialize(
        [_recording("a"), _recording("b")],
        topologies=("channel", "point", "block"),
        rates=(0.5, 0.25),
    )
    unordered = _materialize(
        (_recording("b"), _recording("a")),
        topologies={"block", "channel", "point"},
        rates={0.25, 0.5},
    )

    for other in (reordered, unordered):
        assert [window.window_id for window in first] == [
            window.window_id for window in other
        ]
        assert [window.topology for window in first] == [
            window.topology for window in other
        ]
        assert [window.requested_fraction for window in first] == [
            window.requested_fraction for window in other
        ]
        assert all(torch.equal(left.mask, right.mask) for left, right in zip(first, other))


@pytest.mark.parametrize("rate", [0.0, -0.0])
def test_zero_rates_are_rejected(rate):
    with pytest.raises(ValueError, match="positive|zero|missing"):
        _materialize([_recording()], rates=(rate,))


def test_experiment_seed_changes_identity_and_mask_but_not_physical_window():
    first = _materialize(
        [_recording()],
        stride=16,
        exhaustive=True,
        topologies=("point",),
        rates=(0.5,),
        seed=1,
    )
    second = _materialize(
        [_recording()],
        stride=16,
        exhaustive=True,
        topologies=("point",),
        rates=(0.5,),
        seed=2,
    )

    assert {window.window_id for window in first}.isdisjoint(
        {window.window_id for window in second}
    )
    assert all(torch.equal(a.target, b.target) for a, b in zip(first, second))
    assert all(torch.equal(a.time, b.time) for a, b in zip(first, second))
    assert any(not torch.equal(a.mask, b.mask) for a, b in zip(first, second))


def test_window_ids_bind_recording_content_and_scaler_provenance():
    recording = _recording(rows=16)
    base = _materialize([recording], stride=16, topologies=("point",), rates=(0.5,))[0]

    changed_values = recording.imu_six.copy()
    changed_values[0, 0] += 1.0
    data_changed = _materialize(
        [_recording(rows=16, values=changed_values)],
        stride=16,
        topologies=("point",),
        rates=(0.5,),
    )[0]
    changed_time = recording.imu_time_s.copy()
    changed_time[1:] += np.linspace(0.0, 0.001, 15)
    time_changed = _materialize(
        [_recording(rows=16, time=changed_time)],
        stride=16,
        topologies=("point",),
        rates=(0.5,),
    )[0]
    changed_scaler = RobustTrainScaler(
        center_=np.zeros(6),
        scale_=np.ones(6),
        training_ids=("different-training-provenance",),
    )
    scaler_changed = materialize_teacher_windows(
        [recording],
        changed_scaler,
        window_samples=16,
        stride=16,
        seed=2026,
        topologies=("point",),
        rates=(0.5,),
    )[0]

    assert len(
        {
            base.window_id,
            data_changed.window_id,
            time_changed.window_id,
            scaler_changed.window_id,
        }
    ) == 4
    assert torch.equal(base.mask, data_changed.mask)
    assert torch.equal(base.mask, time_changed.mask)
    assert torch.equal(base.mask, scaler_changed.mask)


def test_public_topology_mapping_documents_validation_v2_generators():
    assert dict(TOPOLOGY_GENERATOR_NAMES) == {
        "point": "point_missing",
        "block": "contiguous_block",
        "channel": "channel_outage",
    }


def test_prepared_window_clones_every_tensor_on_construction_and_read():
    arguments = _prepared_arguments()
    expected = {
        name: value.clone()
        for name, value in arguments.items()
        if isinstance(value, torch.Tensor)
    }
    window = PreparedWindow(**arguments)

    for value in arguments.values():
        if isinstance(value, torch.Tensor):
            value.zero_()
    for name, value in expected.items():
        first_read = getattr(window, name)
        assert torch.equal(first_read, value)
        first_read.fill_(-999)
        assert torch.equal(getattr(window, name), value)
    with pytest.raises(FrozenInstanceError):
        window.window_id = "changed"


def test_materialized_windows_are_shared_ready_clone_on_read_records():
    window = _materialize([_recording()])[0]
    expected = {
        name: getattr(window, name)
        for name in ("features", "target", "observed", "mask", "dt", "time", "baseline")
    }

    for name, value in expected.items():
        getattr(window, name).zero_()
        assert torch.equal(getattr(window, name), value)


def test_collate_prepared_windows_is_dataloader_ready_and_teacher_shaped():
    windows = _materialize([_recording()], stride=16, exhaustive=True)
    loader = DataLoader(
        windows,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_prepared_windows,
    )

    batch = next(iter(loader))
    assert isinstance(batch, PreparedBatch)
    assert batch.features.shape == (2, 16, 31)
    assert batch.target.shape == batch.observed.shape == batch.mask.shape == (2, 16, 6)
    assert batch.dt.shape == batch.time.shape == (2, 16)
    assert batch.baseline.shape == (2, 16, 6)
    assert batch.window_ids == tuple(window.window_id for window in windows[:2])
    assert batch.recording_ids == tuple(window.recording_id for window in windows[:2])
    assert batch.topologies == tuple(window.topology for window in windows[:2])
    assert batch.requested_fractions == tuple(
        window.requested_fraction for window in windows[:2]
    )
    assert batch.realized_fractions == tuple(
        window.realized_fraction for window in windows[:2]
    )
    with pytest.raises(FrozenInstanceError):
        batch.features = torch.empty(0)


def test_prepared_batch_clones_tensor_inputs_once_on_construction():
    window = PreparedWindow(**_prepared_arguments())
    tensor_fields = {
        name: torch.stack([getattr(window, name), getattr(window, name)])
        for name in ("features", "target", "observed", "mask", "dt", "time", "baseline")
    }
    expected_features = tensor_fields["features"].clone()
    batch = PreparedBatch(
        **tensor_fields,
        window_ids=("one", "two"),
        recording_ids=("recording", "recording"),
        topologies=("point", "point"),
        requested_fractions=(0.25, 0.25),
        realized_fractions=(1.0 / 24.0, 1.0 / 24.0),
    )

    for value in tensor_fields.values():
        value.zero_()
    assert torch.equal(batch.features, expected_features)
    assert batch.features is batch.features


def test_collate_rejects_empty_nonwindow_and_incompatible_items():
    with pytest.raises(ValueError, match="non-empty"):
        collate_prepared_windows([])
    with pytest.raises(TypeError, match="PreparedWindow"):
        collate_prepared_windows([object()])

    different_lengths = [
        _materialize([_recording(rows=16)], window_samples=16, stride=16)[0],
        _materialize([_recording(rows=8)], window_samples=8, stride=8)[0],
    ]
    with pytest.raises(ValueError, match="same T|shape"):
        collate_prepared_windows(different_lengths)

    double_arguments = _prepared_arguments()
    for name, value in tuple(double_arguments.items()):
        if isinstance(value, torch.Tensor):
            double_arguments[name] = value.double()
    double_arguments["dt"][1:] = torch.diff(double_arguments["time"])
    double_arguments["features"] = build_features(
        double_arguments["target"],
        double_arguments["mask"],
        double_arguments["dt"],
    ).values
    with pytest.raises(ValueError, match="dtype"):
        collate_prepared_windows(
            [PreparedWindow(**_prepared_arguments()), PreparedWindow(**double_arguments)]
        )


def test_iter_teacher_windows_is_lazy_and_matches_materialization(monkeypatch):
    calls = []
    original_transform = RobustTrainScaler.transform

    def tracked_transform(self, values):
        calls.append(len(values))
        return original_transform(self, values)

    monkeypatch.setattr(RobustTrainScaler, "transform", tracked_transform)
    arguments = dict(
        window_samples=16,
        stride=16,
        seed=7,
        topologies=("point", "block", "channel"),
        rates=(0.25, 0.5),
        exhaustive=True,
    )
    iterator = iter_teacher_windows([_recording()], _scaler(), **arguments)
    assert calls == []

    first = next(iterator)
    assert calls == [40]
    iterated = [first, *iterator]
    materialized = materialize_teacher_windows(
        [_recording()], _scaler(), **arguments
    )
    assert calls == [40, 40]
    assert [item.window_id for item in iterated] == [
        item.window_id for item in materialized
    ]
    for left, right in zip(iterated, materialized):
        assert torch.equal(left.mask, right.mask)
        assert torch.equal(left.features, right.features)


def test_materialize_documents_iterator_for_formal_exhaustive_evaluation():
    documentation = materialize_teacher_windows.__doc__ or ""
    assert "iter_teacher_windows" in documentation
    assert "exhaustive" in documentation


def test_each_window_dt_uses_only_internal_intervals_and_relative_time():
    intervals = np.ones(39, dtype=np.float64)
    intervals[7] = 1000.0
    intervals[8:15] = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    raw_time = np.concatenate(([123456.0], 123456.0 + np.cumsum(intervals)))
    windows = _materialize(
        [_recording(time=raw_time)],
        stride=8,
        exhaustive=True,
        topologies=("point",),
        rates=(0.25,),
    )
    second = windows[1]
    internal = np.diff(raw_time[8:24])

    assert second.time[0].item() == 0.0
    assert torch.all(second.time[1:] > second.time[:-1])
    assert second.dt[0].item() == pytest.approx(float(np.median(internal)))
    assert second.dt[0].item() != pytest.approx(raw_time[8] - raw_time[7])
    np.testing.assert_array_equal(
        second.time.numpy(), (raw_time[8:24] - raw_time[8]).astype(np.float32)
    )
    np.testing.assert_array_equal(second.dt[1:].numpy(), internal.astype(np.float32))


def test_features_and_baseline_do_not_depend_on_hidden_targets():
    recording = _recording(rows=16)
    first = _materialize(
        [recording],
        stride=16,
        exhaustive=True,
        topologies=("point",),
        rates=(0.5,),
    )[0]
    perturbed_values = recording.imu_six.copy()
    perturbed_values[first.mask.numpy() == 0] += 1_000_000.0
    perturbed = _materialize(
        [_recording(rows=16, values=perturbed_values)],
        stride=16,
        exhaustive=True,
        topologies=("point",),
        rates=(0.5,),
    )[0]

    assert torch.equal(first.mask, perturbed.mask)
    assert not torch.equal(first.target, perturbed.target)
    assert torch.equal(first.features, perturbed.features)
    assert torch.equal(first.baseline, perturbed.baseline)


def test_mask_metadata_observed_values_and_baseline_completion_are_consistent():
    windows = _materialize([_recording()], stride=16, exhaustive=True)

    assert {window.topology for window in windows} == {"point", "block", "channel"}
    for window in windows:
        realized = float((window.mask == 0).to(torch.float64).mean().item())
        assert window.realized_fraction == pytest.approx(realized)
        assert window.requested_fraction in (0.25, 0.5)
        assert torch.equal(
            window.observed,
            torch.where(window.mask.bool(), window.target, torch.zeros_like(window.target)),
        )
        assert torch.equal(window.baseline[window.mask.bool()], window.target[window.mask.bool()])
        assert torch.isfinite(window.baseline).all()

    fully_missing = _materialize(
        [_recording(rows=16)],
        stride=16,
        exhaustive=True,
        topologies=("channel",),
        rates=(1.0,),
    )[0]
    assert torch.count_nonzero(fully_missing.observed) == 0
    assert torch.count_nonzero(fully_missing.baseline) == 0


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"window_samples": 1}, (TypeError, ValueError)),
        ({"window_samples": True}, (TypeError, ValueError)),
        ({"stride": 0}, (TypeError, ValueError)),
        ({"stride": 1.5}, (TypeError, ValueError)),
        ({"seed": True}, TypeError),
        ({"seed": 1.5}, TypeError),
        ({"topologies": ()}, ValueError),
        ({"topologies": ("point", "point")}, ValueError),
        ({"topologies": ("unknown",)}, ValueError),
        ({"topologies": (1,)}, (TypeError, ValueError)),
        ({"rates": ()}, ValueError),
        ({"rates": (0.25, 0.25)}, ValueError),
        ({"rates": (-0.1,)}, ValueError),
        ({"rates": (1.1,)}, ValueError),
        ({"rates": (np.nan,)}, ValueError),
        ({"rates": (True,)}, (TypeError, ValueError)),
        ({"exhaustive": 1}, TypeError),
    ],
)
def test_materialization_rejects_invalid_arguments(override, error):
    with pytest.raises(error):
        _materialize([_recording()], **override)


@pytest.mark.parametrize("recordings", [None, 1, "recording", b"recording"])
def test_materialization_requires_a_recording_iterable(recordings):
    with pytest.raises((TypeError, ValueError)):
        _materialize(recordings)


def test_materialization_accepts_one_shot_iterables_and_rejects_duplicate_ids():
    windows = _materialize(iter([_recording("b"), _recording("a")]))
    assert windows[0].recording_id == "a"

    with pytest.raises(ValueError, match="unique"):
        _materialize([_recording("same"), _recording("same")])


@pytest.mark.parametrize("recording_id", ["", None, 1, True])
def test_materialization_rejects_invalid_recording_ids(recording_id):
    with pytest.raises((TypeError, ValueError), match="recording.*id|ID"):
        _materialize([_recording(recording_id)])


@pytest.mark.parametrize(
    ("time", "values", "message"),
    [
        (np.arange(40).reshape(20, 2), None, "time"),
        (np.r_[np.arange(39), np.nan], None, "finite"),
        (np.r_[np.arange(39), 38.0], None, "strictly increasing"),
        (None, np.zeros((40, 5)), "shape"),
        (None, np.zeros((39, 6)), "aligned"),
        (None, np.full((40, 6), np.nan), "finite"),
    ],
)
def test_materialization_rejects_malformed_recordings(time, values, message):
    with pytest.raises(ValueError, match=message):
        _materialize([_recording(time=time, values=values)])


@pytest.mark.parametrize(
    ("time", "values"),
    [
        (np.arange(40, dtype=np.complex128) + 1j, None),
        (None, np.ones((40, 6), dtype=np.complex128) * (1.0 + 1j)),
    ],
)
def test_materialization_rejects_complex_recording_arrays_before_cast(time, values):
    with pytest.raises((TypeError, ValueError), match="complex"):
        _materialize([_recording(time=time, values=values)])


def test_materialization_rejects_float32_unrepresentable_local_time():
    time = np.arange(40, dtype=np.float64) * 1e-50
    with pytest.raises(ValueError, match="float32|represent"):
        _materialize([_recording(time=time)])


def test_materialization_rejects_empty_output_instead_of_training_empty():
    with pytest.raises(ValueError, match="no complete windows|no windows"):
        _materialize([_recording(rows=15)])


def test_positive_rate_that_realizes_no_missing_values_is_rejected():
    with pytest.raises(ValueError, match="realizes no missing|missing entries"):
        _materialize(
            [_recording(rows=16)],
            stride=16,
            topologies=("point",),
            rates=(1e-12,),
        )


def test_materialization_requires_exact_frozen_robust_scaler_type():
    class AdaptiveScaler:
        def transform(self, values):
            return values

    with pytest.raises(TypeError, match="RobustTrainScaler|frozen scaler"):
        materialize_teacher_windows(
            [_recording()],
            AdaptiveScaler(),
            window_samples=16,
            stride=8,
            seed=1,
            topologies=("point",),
            rates=(0.25,),
        )


def test_materialization_rejects_complex_scaler_arrays_before_cast():
    scaler = _scaler()
    object.__setattr__(scaler, "center_", np.ones(6, dtype=np.complex128))

    with pytest.raises((TypeError, ValueError), match="complex"):
        materialize_teacher_windows(
            [_recording()],
            scaler,
            window_samples=16,
            stride=8,
            seed=1,
            topologies=("point",),
            rates=(0.25,),
        )


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((40, 5)),
        np.zeros((39, 6)),
        np.full((40, 6), np.nan),
        "not-an-array",
    ],
)
def test_materialization_rejects_malformed_scaler_output(output, monkeypatch):
    monkeypatch.setattr(RobustTrainScaler, "transform", lambda self, values: output)

    with pytest.raises((TypeError, ValueError), match="scaler|transform|shape|finite"):
        materialize_teacher_windows(
            [_recording()],
            _scaler(),
            window_samples=16,
            stride=8,
            seed=1,
            topologies=("point",),
            rates=(0.25,),
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("window_id", "", ValueError),
        ("recording_id", 1, TypeError),
        ("topology", True, TypeError),
        ("topology", "unknown", ValueError),
        ("requested_fraction", "0.5", TypeError),
        ("requested_fraction", np.nan, ValueError),
        ("requested_fraction", 1.1, ValueError),
        ("realized_fraction", True, TypeError),
        ("realized_fraction", -0.1, ValueError),
    ],
)
def test_prepared_window_rejects_invalid_scalar_metadata(field, value, error):
    arguments = _prepared_arguments()
    arguments[field] = value
    with pytest.raises(error):
        PreparedWindow(**arguments)


def _replace_tensor_value(arguments, field, index, value):
    changed = arguments[field].clone()
    changed[index] = value
    arguments[field] = changed


@pytest.mark.parametrize(
    "mutate",
    [
        lambda values: _replace_tensor_value(values, "time", 0, 0.01),
        lambda values: _replace_tensor_value(values, "dt", 2, 0.25),
        lambda values: _replace_tensor_value(values, "observed", (0, 0), -1.0),
        lambda values: _replace_tensor_value(values, "observed", (1, 2), 99.0),
        lambda values: _replace_tensor_value(values, "baseline", (0, 0), -1.0),
        lambda values: values.update(realized_fraction=0.5),
        lambda values: _replace_tensor_value(values, "features", (0, 0), -1.0),
        lambda values: _replace_tensor_value(values, "features", (0, 6), 0.0),
        lambda values: _replace_tensor_value(values, "features", (0, 12), 9.0),
    ],
)
def test_prepared_window_rejects_cross_field_inconsistency(mutate):
    arguments = _prepared_arguments()
    mutate(arguments)
    with pytest.raises(ValueError):
        PreparedWindow(**arguments)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target", torch.ones(4, 5), "target"),
        ("observed", torch.ones(5, 6), "observed"),
        ("mask", torch.full((4, 6), 0.5), "binary|0 or 1"),
        ("mask", torch.ones(4, 6, dtype=torch.int64), "floating|dtype"),
        ("baseline", torch.ones(4, 5), "baseline"),
        ("features", torch.ones(4, 30), "features"),
        ("dt", torch.ones(5), "dt"),
        ("time", torch.ones(5), "time"),
        ("dt", torch.tensor([0.1, 0.0, 0.1, 0.1]), "positive"),
        ("time", torch.tensor([0.0, 0.1, 0.1, 0.2]), "increasing"),
        ("features", torch.full((4, 31), torch.nan), "finite"),
        ("target", torch.full((4, 6), torch.inf), "finite"),
    ],
)
def test_prepared_window_validates_tensor_boundary(field, value, message):
    arguments = _prepared_arguments()
    arguments[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        PreparedWindow(**arguments)


def test_prepared_window_requires_compatible_tensor_dtypes_and_devices():
    arguments = _prepared_arguments()
    arguments["time"] = arguments["time"].double()
    with pytest.raises(ValueError, match="dtype"):
        PreparedWindow(**arguments)

    arguments = _prepared_arguments()
    arguments["mask"] = arguments["mask"].to("meta")
    with pytest.raises(ValueError, match="device"):
        PreparedWindow(**arguments)


def test_prepared_window_rejects_unmaterialized_meta_tensors():
    arguments = _prepared_arguments()
    for name, value in tuple(arguments.items()):
        if isinstance(value, torch.Tensor):
            arguments[name] = value.to("meta")

    with pytest.raises(ValueError, match="materialized|meta"):
        PreparedWindow(**arguments)
