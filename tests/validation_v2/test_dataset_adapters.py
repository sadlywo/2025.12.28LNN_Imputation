import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validation_v2.data import get_dataset_adapter


def _write_euroc(root: Path, sequence: str = "V1_01_easy") -> tuple[Path, Path]:
    mav0 = root / sequence / "mav0"
    imu_path = mav0 / "imu0" / "data.csv"
    reference_path = mav0 / "state_groundtruth_estimate0" / "data.csv"
    imu_path.parent.mkdir(parents=True)
    reference_path.parent.mkdir(parents=True)
    timestamps = [1_400_000_000_000_000_000 + index * 5_000_000 for index in range(4)]
    pd.DataFrame(
        {
            "#timestamp [ns]": timestamps,
            "w_RS_S_x [rad s^-1]": [0.1, 0.2, 0.3, 0.4],
            "w_RS_S_y [rad s^-1]": [0.0] * 4,
            "w_RS_S_z [rad s^-1]": [0.0] * 4,
            "a_RS_S_x [m s^-2]": [0.0] * 4,
            "a_RS_S_y [m s^-2]": [0.0] * 4,
            "a_RS_S_z [m s^-2]": [9.80665] * 4,
        }
    ).to_csv(imu_path, index=False)
    pd.DataFrame(
        {
            "#timestamp": timestamps,
            " p_RS_R_x [m]": [0.0, 0.1, 0.2, 0.3],
            " p_RS_R_y [m]": [0.0] * 4,
            " p_RS_R_z [m]": [0.0] * 4,
            " q_RS_w []": [1.0] * 4,
            " q_RS_x []": [0.0] * 4,
            " q_RS_y []": [0.0] * 4,
            " q_RS_z []": [0.0] * 4,
        }
    ).to_csv(reference_path, index=False)
    return imu_path, reference_path


def test_euroc_adapter_discovers_and_converts_source_semantics(tmp_path):
    _write_euroc(tmp_path)
    adapter = get_dataset_adapter("euroc_mav")
    pairs = adapter.discover(tmp_path)

    assert len(pairs) == 1
    assert pairs[0]["scenario"] == "vicon_room1"
    recording = adapter.load(
        Path(pairs[0]["imu_path"]), Path(pairs[0]["vicon_path"])
    )
    assert recording.id == "euroc_mav/V1_01_easy"
    np.testing.assert_allclose(recording.imu_time_s, [0.0, 0.005, 0.01, 0.015])
    np.testing.assert_allclose(
        recording.vicon_quaternion_xyzw,
        np.tile([0.0, 0.0, 0.0, 1.0], (4, 1)),
    )
    assert recording.metadata["acceleration_semantics"] == "specific_force"
    assert adapter.semantics.acceleration_unit == "m/s^2"


def _write_idol(root: Path) -> Path:
    pytest.importorskip("pyarrow")
    directory = root / "building1" / "known"
    directory.mkdir(parents=True)
    path = directory / "0.feather"
    count = 4
    frame = pd.DataFrame(
        {
            "timestamp": 1000.0 + np.arange(count) * 0.01,
            "stencilGyroX": np.arange(count, dtype=float),
            "stencilGyroY": np.zeros(count),
            "stencilGyroZ": np.zeros(count),
            "stencilAccX": np.zeros(count),
            "stencilAccY": np.zeros(count),
            "stencilAccZ": np.full(count, 9.80665),
            "processedPosX": np.arange(count, dtype=float),
            "processedPosY": np.zeros(count),
            "processedPosZ": np.zeros(count),
            "orientW": np.ones(count),
            "orientX": np.zeros(count),
            "orientY": np.zeros(count),
            "orientZ": np.zeros(count),
        }
    )
    frame.to_feather(path)
    (directory / "metadata.json").write_text(
        json.dumps({"0": {"subjectID": 7, "calibration": "start"}}),
        encoding="utf-8",
    )
    return path


def test_idol_adapter_uses_stencil_imu_and_single_synchronized_file(tmp_path):
    _write_idol(tmp_path)
    adapter = get_dataset_adapter("idol")
    pair = adapter.discover(tmp_path)[0]

    assert pair["imu_path"] == pair["vicon_path"]
    assert pair["scenario"] == "building1"
    recording = adapter.load(Path(pair["imu_path"]), Path(pair["vicon_path"]))
    assert recording.id == "idol/building1/known/0"
    np.testing.assert_allclose(recording.imu_time_s, [0.0, 0.01, 0.02, 0.03])
    np.testing.assert_allclose(recording.imu_six[:, 0], np.arange(4))
    np.testing.assert_allclose(recording.imu_six[:, 5], 9.80665)
    assert recording.metadata["imu_source"] == "stencil"
    assert recording.metadata["subject_id"] == 7


def test_registered_adapter_names_are_explicit():
    assert get_dataset_adapter("oxiod").name == "oxiod"
    assert get_dataset_adapter("euroc_mav").name == "euroc_mav"
    assert get_dataset_adapter("idol").name == "idol"
