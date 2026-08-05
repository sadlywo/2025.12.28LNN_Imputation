from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import visualize_oxford_dataset as visualization


def test_speed_colored_trajectory_preserves_speed_when_downsampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sample_count = 3601
    time_seconds = np.arange(sample_count, dtype=np.float64) * 0.01
    trajectory_path = tmp_path / "vi1.csv"
    pd.DataFrame(
        {
            "Time": time_seconds * 1e9,
            "translation.x": time_seconds,
            "translation.y": np.zeros(sample_count),
        }
    ).to_csv(trajectory_path, index=False)

    captured_norm = {}
    original_normalize = plt.Normalize

    def record_normalize(*args, **kwargs):
        norm = original_normalize(*args, **kwargs)
        captured_norm["value"] = norm
        return norm

    monkeypatch.setattr(visualization.plt, "Normalize", record_normalize)

    visualization.plot_speed_colored_trajectories(
        [("Constant Speed", trajectory_path)], tmp_path / "trajectory.png"
    )

    assert captured_norm["value"].vmax == pytest.approx(1.0)
