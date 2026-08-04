from pathlib import Path

import numpy as np
import pytest

from validation_v2.modern.artifacts import read_array_artifact, write_array_artifact


def test_array_artifact_round_trip_and_hashes(tmp_path: Path):
    arrays = {"target": np.arange(12, dtype=np.float32).reshape(2, 3, 2)}
    manifest = write_array_artifact(
        tmp_path / "bundle", "dataset", arrays, {"seed": 2026}
    )
    loaded, metadata = read_array_artifact(
        tmp_path / "bundle", expected_kind="dataset"
    )
    np.testing.assert_array_equal(loaded["target"], arrays["target"])
    assert metadata["artifact_id"] == manifest["artifact_id"]


def test_array_artifact_rejects_tampered_npz(tmp_path: Path):
    write_array_artifact(
        tmp_path / "bundle", "prediction", {"x": np.ones(2)}, {}
    )
    (tmp_path / "bundle.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_array_artifact(tmp_path / "bundle", expected_kind="prediction")


def test_array_artifact_refuses_overwrite(tmp_path: Path):
    write_array_artifact(tmp_path / "bundle", "dataset", {"x": np.ones(2)}, {})
    with pytest.raises(FileExistsError):
        write_array_artifact(
            tmp_path / "bundle", "dataset", {"x": np.ones(2)}, {}
        )
