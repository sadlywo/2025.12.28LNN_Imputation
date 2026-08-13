import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validation_v2.data.normalization import RobustTrainScaler
from validation_v2.data.splits import (
    leave_one_scenario_out,
    stratified_file_split,
)
from validation_v2.types import Recording


def _recording_index(tmp_path: Path, counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    for scenario, count in counts.items():
        for index in range(count):
            recording_id = f"{scenario}-{index}"
            imu_path = tmp_path / f"{recording_id}-imu.csv"
            vicon_path = tmp_path / f"{recording_id}-vicon.csv"
            imu_path.write_bytes(f"imu:{recording_id}".encode())
            vicon_path.write_bytes(f"vicon:{recording_id}".encode())
            rows.append(
                {
                    "recording_id": recording_id,
                    "scenario": scenario,
                    "imu_path": imu_path,
                    "vicon_path": vicon_path,
                }
            )
    return pd.DataFrame(rows)


def _recording(recording_id: str, imu_six: np.ndarray) -> Recording:
    rows = len(imu_six)
    return Recording(
        id=recording_id,
        imu_time_s=np.arange(rows, dtype=float),
        imu_six=np.asarray(imu_six, dtype=float),
        vicon_time_s=np.arange(max(rows, 1), dtype=float),
        vicon_position_m=np.zeros((max(rows, 1), 3)),
        vicon_quaternion_xyzw=np.zeros((max(rows, 1), 4)),
        overlap_s=(0.0, float(max(rows - 1, 0))),
        metadata={},
    )


def test_stratified_split_is_disjoint_deterministic_and_preserves_scenarios(tmp_path):
    index = _recording_index(tmp_path, {"handbag": 9, "corridor": 9})

    first = stratified_file_split(index, seed=2026)
    second = stratified_file_split(index, seed=2026)

    pd.testing.assert_frame_equal(first, second)
    split_ids = {
        split: set(first.loc[first["split"] == split, "recording_id"])
        for split in ("train", "validation", "test")
    }
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])
    for scenario in index["scenario"].unique():
        scenario_splits = set(first.loc[first["scenario"] == scenario, "split"])
        assert scenario_splits == {"train", "validation", "test"}


def test_manifest_has_one_traceable_row_per_source_pair(tmp_path):
    index = _recording_index(tmp_path, {"handbag": 4, "corridor": 4})

    manifest = stratified_file_split(index, seed=7)

    assert list(manifest.columns) == [
        "recording_id",
        "scenario",
        "imu_path",
        "vicon_path",
        "split",
        "imu_sha256",
        "vicon_sha256",
    ]
    assert manifest["recording_id"].is_unique
    assert manifest["imu_path"].is_unique
    assert manifest["vicon_path"].is_unique
    assert len(manifest) == len(index)
    for row in manifest.itertuples(index=False):
        assert Path(row.imu_path).is_absolute()
        assert Path(row.vicon_path).is_absolute()
        assert row.imu_sha256 == hashlib.sha256(Path(row.imu_path).read_bytes()).hexdigest()
        assert row.vicon_sha256 == hashlib.sha256(Path(row.vicon_path).read_bytes()).hexdigest()


def test_leave_one_scenario_out_holds_out_all_and_splits_only_remaining(tmp_path):
    index = _recording_index(tmp_path, {"handbag": 5, "corridor": 5, "office": 5})

    manifest = leave_one_scenario_out(
        index,
        held_out_scenario="office",
        seed=2026,
    )

    held_out = manifest[manifest["scenario"] == "office"]
    assert set(held_out["split"]) == {"test"}
    assert "office" not in set(manifest.loc[manifest["split"] != "test", "scenario"])
    assert not manifest.loc[manifest["split"] == "validation"].empty
    assert manifest["recording_id"].is_unique
    split_ids = [set(group["recording_id"]) for _, group in manifest.groupby("split")]
    assert all(left.isdisjoint(right) for i, left in enumerate(split_ids) for right in split_ids[i + 1 :])


def test_small_scenarios_use_deterministic_nonduplicating_fallback(tmp_path):
    index = _recording_index(tmp_path, {"one": 1, "two": 2})

    first = stratified_file_split(index, seed=11)
    second = stratified_file_split(index, seed=11)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 3
    assert first["recording_id"].is_unique
    assert set(first["split"]).issubset({"train", "validation", "test"})


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.drop(columns="scenario"), "missing required columns"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "duplicate recording_id"),
        (lambda frame: frame.assign(imu_path=frame["imu_path"].where(frame.index != 0, frame.iloc[1]["imu_path"])), "source file"),
    ],
)
def test_split_rejects_invalid_recording_indexes(tmp_path, mutator, message):
    index = _recording_index(tmp_path, {"scenario": 2})

    with pytest.raises(ValueError, match=message):
        stratified_file_split(mutator(index))


def test_split_accepts_a_sequence_of_mapping_records_and_rejects_missing_files(tmp_path):
    index = _recording_index(tmp_path, {"scenario": 2})
    records = index.to_dict(orient="records")
    manifest = stratified_file_split(records)
    assert len(manifest) == 2

    Path(records[0]["imu_path"]).unlink()
    with pytest.raises(ValueError, match=r"does not exist|not a readable file"):
        stratified_file_split(records)


def test_split_accepts_one_synchronized_source_for_both_roles(tmp_path):
    index = _recording_index(tmp_path, {"scenario": 3})
    index.loc[0, "vicon_path"] = index.loc[0, "imu_path"]

    manifest = stratified_file_split(index)

    row = manifest.loc[manifest["recording_id"] == index.loc[0, "recording_id"]].iloc[0]
    assert row["imu_path"] == row["vicon_path"]
    assert row["imu_sha256"] == row["vicon_sha256"]


def test_scaler_rejects_any_recording_outside_the_allowed_training_ids():
    train = _recording("train", np.zeros((2, 6)))
    test = _recording("test", np.ones((2, 6)))

    with pytest.raises(ValueError, match="fit accepts train recordings only"):
        RobustTrainScaler.fit([train, test], allowed_ids={train.id})


def test_scaler_uses_train_values_median_mad_floor_and_sorted_ids():
    train_b = _recording("b", np.array([[0.0, 10.0], [2.0, 10.0]]))
    train_a = _recording("a", np.array([[4.0, 10.0]]))

    scaler = RobustTrainScaler.fit([train_b, train_a], allowed_ids={"a", "b"})

    np.testing.assert_allclose(scaler.center_, [2.0, 10.0])
    np.testing.assert_allclose(scaler.scale_, [1.4826 * 2.0, 1e-6])
    assert scaler.training_ids == ("a", "b")


@pytest.mark.parametrize("attribute", ["center_", "scale_"])
def test_fit_returns_arrays_that_cannot_be_made_writable(attribute):
    training = _recording("train", np.array([[0.0, 1.0], [2.0, 3.0]]))
    scaler = RobustTrainScaler.fit([training], allowed_ids={"train"})

    with pytest.raises(ValueError):
        getattr(scaler, attribute).setflags(write=True)


def test_direct_constructor_defensively_freezes_scaler_arrays():
    center = np.array([1.0, 2.0])
    scale = np.array([3.0, 4.0])
    scaler = RobustTrainScaler(
        center_=center,
        scale_=scale,
        training_ids=("train",),
    )

    center[:] = 99.0
    scale[:] = 88.0

    np.testing.assert_array_equal(scaler.center_, [1.0, 2.0])
    np.testing.assert_array_equal(scaler.scale_, [3.0, 4.0])
    with pytest.raises(ValueError):
        scaler.center_.setflags(write=True)
    with pytest.raises(ValueError):
        scaler.scale_.setflags(write=True)


@pytest.mark.parametrize(
    ("center", "scale", "training_ids", "message"),
    [
        (np.zeros((1, 2)), np.ones(2), ("a",), "one-dimensional"),
        (np.zeros(2), np.ones(3), ("a",), "same shape"),
        (np.array([np.nan]), np.ones(1), ("a",), "finite"),
        (np.zeros(1), np.array([np.inf]), ("a",), "finite"),
        (np.zeros(1), np.zeros(1), ("a",), "strictly positive"),
        (np.zeros(1), np.ones(1), (), "non-empty"),
        (np.zeros(1), np.ones(1), ("a", "a"), "unique"),
        (np.zeros(1), np.ones(1), ("b", "a"), "sorted"),
    ],
)
def test_direct_constructor_rejects_invalid_scaler_state(
    center,
    scale,
    training_ids,
    message,
):
    with pytest.raises(ValueError, match=message):
        RobustTrainScaler(
            center_=center,
            scale_=scale,
            training_ids=training_ids,
        )


def test_scaler_transform_round_trips_without_mutating_input():
    training = _recording("train", np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))
    scaler = RobustTrainScaler.fit([training], allowed_ids={"train"})
    values = np.array([[8.0, 9.0], [10.0, 11.0]])
    original = values.copy()

    transformed = scaler.transform(values)
    restored = scaler.inverse_transform(transformed)

    np.testing.assert_allclose(restored, values)
    np.testing.assert_array_equal(values, original)
    assert transformed is not values
    assert restored is not transformed


@pytest.mark.parametrize(
    ("recordings", "allowed_ids", "message"),
    [
        ([], set(), "at least one"),
        ([_recording("empty", np.empty((0, 2)))], {"empty"}, "empty"),
        (
            [_recording("a", np.zeros((1, 2))), _recording("b", np.zeros((1, 3)))],
            {"a", "b"},
            "feature dimension",
        ),
        ([_recording("nan", np.array([[np.nan, 0.0]]))], {"nan"}, "finite"),
        ([_recording("inf", np.array([[np.inf, 0.0]]))], {"inf"}, "finite"),
    ],
)
def test_scaler_fit_rejects_invalid_training_data(recordings, allowed_ids, message):
    with pytest.raises(ValueError, match=message):
        RobustTrainScaler.fit(recordings, allowed_ids=allowed_ids)


@pytest.mark.parametrize(
    "values",
    [
        np.array([1.0, 2.0]),
        np.zeros((1, 3)),
        np.array([[np.nan, 0.0]]),
        np.array([[np.inf, 0.0]]),
        np.empty((0, 2)),
    ],
)
def test_scaler_transform_rejects_bad_shape_empty_and_nonfinite(values):
    training = _recording("train", np.array([[0.0, 1.0], [2.0, 3.0]]))
    scaler = RobustTrainScaler.fit([training], allowed_ids={"train"})

    with pytest.raises(ValueError):
        scaler.transform(values)
