from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path

import numpy as np

from validation_v2.experiments.runner import (
    _recording_dataset,
    _scalers_content,
    _split_content,
    resolve_protocol_records,
)
from validation_v2.types import Recording
from validation_v2.data.normalization import RobustTrainScaler


def _recording(dataset: str, recording_id: str, offset: float) -> Recording:
    time = np.arange(40, dtype=np.float64) * 0.01
    values = np.full((40, 6), offset, dtype=np.float64)
    position = np.zeros((40, 3), dtype=np.float64)
    quaternion = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (40, 1))
    return Recording(
        id=recording_id,
        imu_time_s=time,
        imu_six=values,
        vicon_time_s=time,
        vicon_position_m=position,
        vicon_quaternion_xyzw=quaternion,
        overlap_s=(0.0, float(time[-1])),
        metadata={"dataset": dataset},
    )


def test_joint_split_is_deterministic_and_keeps_all_three_partitions():
    pairs = [
        {
            "recording_id": f"demo/recording-{index}",
            "scenario": "room",
            "imu_path": f"imu-{index}",
            "vicon_path": f"reference-{index}",
        }
        for index in range(10)
    ]
    first = resolve_protocol_records(
        pairs, "strict_file", seed=2026, split_ratios=(0.8, 0.1, 0.1)
    )
    second = resolve_protocol_records(
        pairs, "strict_file", seed=2026, split_ratios=(0.8, 0.1, 0.1)
    )
    assert first == second
    assert {split: sum(row["split"] == split for row in first) for split in
            ("train", "validation", "test")} == {
        "train": 8, "validation": 1, "test": 1
    }


def test_joint_scaler_manifest_is_per_dataset_train_only():
    scalers = {}
    for dataset, offset in (("oxiod", 1.0), ("euroc_mav", 10.0), ("idol", 100.0)):
        recording = _recording(dataset, f"{dataset}/train", offset)
        scalers[dataset] = RobustTrainScaler.fit(
            [recording], allowed_ids={recording.id}
        )
        assert _recording_dataset(recording) == dataset
    payload = json.loads(_scalers_content(scalers, split_hash="a" * 64))
    assert payload["normalization_scope"] == "per_dataset_train_only"
    assert payload["joint_sampling"] == "dataset_balanced"
    assert set(payload["datasets"]) == {"oxiod", "euroc_mav", "idol"}
    assert payload["datasets"]["oxiod"]["center"] != payload["datasets"]["idol"]["center"]


def test_split_manifest_namespaces_rows_by_dataset():
    content = _split_content(
        [
            {
                "dataset": "oxiod",
                "recording_id": "handbag/imu1",
                "scenario": "handbag",
                "imu_path": "imu.csv",
                "vicon_path": "vi.csv",
                "split": "train",
                "imu_sha256": "a",
                "vicon_sha256": "b",
            }
        ]
    )
    rows = list(csv.DictReader(StringIO(content.decode("utf-8"))))
    assert rows[0]["dataset"] == "oxiod"
