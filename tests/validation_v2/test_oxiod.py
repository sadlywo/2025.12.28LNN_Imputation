import csv
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from validation_v2.data.oxiod import (
    _time_to_seconds,
    load_recording,
    overlapping_interval,
)


def _write_synthetic_pair(
    root: Path,
    *,
    imu_times: tuple[float, ...] = (1_496_760_699.22, 1_496_760_699.23),
    vicon_times_ns: tuple[int, ...] = (
        1_496_760_699_220_000_000,
        1_496_760_699_230_000_000,
    ),
) -> tuple[Path, Path]:
    imu_path = root / "imu1.csv"
    vicon_path = root / "vi1.csv"

    with imu_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for index, time_s in enumerate(imu_times):
            writer.writerow(
                [
                    time_s,
                    0.1,
                    0.2,
                    0.3,
                    1.0 + index,
                    2.0 + index,
                    3.0 + index,
                    0.0,
                    0.0,
                    -1.0,
                    0.01 + index,
                    0.02 + index,
                    0.03 + index,
                    10.0,
                    20.0,
                    30.0,
                ]
            )

    with vicon_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for index, time_ns in enumerate(vicon_times_ns):
            writer.writerow(
                [
                    time_ns,
                    100 + index,
                    0.1 + index,
                    0.2 + index,
                    0.3 + index,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                ]
            )
    return imu_path, vicon_path


def test_vicon_nanoseconds_are_converted_to_seconds(tmp_path: Path):
    imu_path, vicon_path = _write_synthetic_pair(tmp_path)

    recording = load_recording(imu_path, vicon_path)

    assert recording.vicon_time_s.dtype == np.float64
    assert recording.vicon_time_s[0] == pytest.approx(recording.imu_time_s[0])
    assert np.all(np.diff(recording.imu_time_s) > 0)
    assert np.all(np.diff(recording.vicon_time_s) > 0)


def test_short_epoch_vicon_nanoseconds_are_explicitly_converted(tmp_path: Path):
    imu_path, vicon_path = _write_synthetic_pair(
        tmp_path,
        imu_times=(20.0, 21.0),
        vicon_times_ns=(20_000_000_000, 21_000_000_000),
    )

    recording = load_recording(imu_path, vicon_path)

    np.testing.assert_allclose(recording.vicon_time_s, [20.0, 21.0])
    assert recording.metadata["vicon_source_time_unit"] == "ns"


@pytest.mark.parametrize("stream", ["imu", "vicon"])
def test_time_conversion_rejects_nonincreasing_stream(stream: str):
    with pytest.raises(ValueError, match=f"{stream} timestamps must be strictly increasing"):
        _time_to_seconds(np.array([2.0, 2.0]), stream)


def test_time_conversion_rejects_empty_timestamps():
    with pytest.raises(ValueError, match=r"empty|timestamps"):
        _time_to_seconds(np.array([]), "imu")


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_time_conversion_rejects_nonfinite_timestamps(invalid: float):
    with pytest.raises(ValueError, match=r"finite.*timestamps|timestamps.*finite"):
        _time_to_seconds(np.array([invalid]), "imu")


def test_time_conversion_rejects_unknown_source_unit():
    with pytest.raises(ValueError, match="unknown timestamp source unit"):
        _time_to_seconds(np.array([1.0, 2.0]), "vicon", source_unit="ms")


def test_loader_rejects_empty_time_overlap(tmp_path: Path):
    imu_path, vicon_path = _write_synthetic_pair(
        tmp_path,
        imu_times=(10.0, 11.0),
        vicon_times_ns=(30_000_000_000, 31_000_000_000),
    )

    with pytest.raises(ValueError, match="no IMU/Vicon overlap"):
        load_recording(imu_path, vicon_path)


def test_overlap_rejects_contact_without_duration():
    with pytest.raises(ValueError, match="no IMU/Vicon overlap"):
        overlapping_interval(np.array([0.0, 1.0]), np.array([1.0, 2.0]))


def test_loader_uses_readme_channel_and_quaternion_order(tmp_path: Path):
    imu_path, vicon_path = _write_synthetic_pair(tmp_path)

    recording = load_recording(imu_path, vicon_path)

    np.testing.assert_allclose(recording.imu_six[0], [1.0, 2.0, 3.0, 0.01, 0.02, 0.03])
    np.testing.assert_allclose(recording.vicon_quaternion_xyzw[0], [0.4, 0.5, 0.6, 0.7])
    assert recording.metadata["imu_channel_names"] == (
        "rotation_rate_x",
        "rotation_rate_y",
        "rotation_rate_z",
        "user_acc_x",
        "user_acc_y",
        "user_acc_z",
    )
    assert recording.metadata["imu_channel_units"] == (
        "rad/s",
        "rad/s",
        "rad/s",
        "G",
        "G",
        "G",
    )
    assert recording.metadata["vicon_quaternion_order"] == "xyzw"
    assert recording.overlap_s == pytest.approx(
        (recording.imu_time_s[0], recording.imu_time_s[-1])
    )
    with pytest.raises(FrozenInstanceError):
        recording.id = "changed"  # type: ignore[misc]


def test_loader_reports_missing_columns(tmp_path: Path):
    imu_path, vicon_path = _write_synthetic_pair(tmp_path)
    imu_path.write_text("1,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"IMU CSV .* expected 16 columns .* found 3"):
        load_recording(imu_path, vicon_path)


@pytest.mark.parametrize(
    ("empty_stream", "header"),
    [
        (
            "imu",
            [
                "Time",
                "rotation_rate_x",
                "rotation_rate_y",
                "rotation_rate_z",
                "user_acc_x",
                "user_acc_y",
                "user_acc_z",
            ],
        ),
        (
            "vicon",
            [
                "Time",
                "translation.x",
                "translation.y",
                "translation.z",
                "rotation.x",
                "rotation.y",
                "rotation.z",
                "rotation.w",
            ],
        ),
    ],
)
def test_loader_rejects_header_only_stream(
    tmp_path: Path,
    empty_stream: str,
    header: list[str],
):
    imu_path, vicon_path = _write_synthetic_pair(tmp_path)
    empty_path = imu_path if empty_stream == "imu" else vicon_path
    with empty_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(header)

    with pytest.raises(ValueError, match=f"{empty_stream} timestamps.*empty"):
        load_recording(imu_path, vicon_path)


def test_loader_explicitly_repairs_raw_row_order_and_duplicate_times(tmp_path: Path):
    t0 = 1_496_760_699.0
    imu_path, vicon_path = _write_synthetic_pair(
        tmp_path,
        imu_times=(t0, t0 + 0.02, t0 + 0.01, t0 + 0.02, t0 + 0.03),
        vicon_times_ns=(1_496_760_699_000_000_000, 1_496_760_700_000_000_000),
    )

    recording = load_recording(imu_path, vicon_path)

    assert np.all(np.diff(recording.imu_time_s) > 0)
    assert len(recording.imu_time_s) == len(recording.imu_six) == 4
    assert recording.metadata["imu_source_rows"] == 5
    assert recording.metadata["imu_rows_deduplicated"] == 1
    assert recording.metadata["imu_rows_reordered"] > 0


def test_real_handbag_recording_has_valid_association_and_motion():
    data_dir = Path(__file__).resolve().parents[2] / "Oxford Dataset" / "handbag-1"
    imu_path = data_dir / "imu1.csv"
    vicon_path = data_dir / "vi1.csv"
    if not (imu_path.is_file() and vicon_path.is_file()):
        pytest.skip(f"OxIOD handbag-1 pair not found under {data_dir}")

    recording = load_recording(imu_path, vicon_path)
    start, end = recording.overlap_s
    query_times = recording.imu_time_s[
        (recording.imu_time_s >= start) & (recording.imu_time_s <= end)
    ]

    assert query_times.size > 1
    assert np.all(np.diff(recording.imu_time_s) > 0)
    assert np.all(np.diff(recording.vicon_time_s) > 0)
    assert len(recording.imu_time_s) == len(recording.imu_six)
    assert len(recording.vicon_time_s) == len(recording.vicon_position_m)
    assert recording.metadata["imu_rows_deduplicated"] == 5
    assert recording.metadata["imu_rows_reordered"] > 0
    assert np.all(query_times >= recording.vicon_time_s[0])
    assert np.all(query_times <= recording.vicon_time_s[-1])
    interpolated_position = np.column_stack(
        [
            np.interp(query_times, recording.vicon_time_s, recording.vicon_position_m[:, axis])
            for axis in range(3)
        ]
    )
    assert np.unique(interpolated_position, axis=0).shape[0] > 1
