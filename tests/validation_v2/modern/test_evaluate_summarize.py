import numpy as np
import pandas as pd
import pytest

from validation_v2.modern.evaluate import complete_samples
from validation_v2.modern.summarize import validate_stage_a_coverage


def test_complete_samples_preserves_observed_values():
    observed = np.array([[1.0], [np.nan], [3.0]], dtype=np.float32)
    mask = np.array([[1], [0], [1]], dtype=np.uint8)
    samples = np.full((2, 3, 1), 9.0, dtype=np.float32)
    result = complete_samples(observed, mask, samples)
    np.testing.assert_array_equal(result[:, [0, 2], 0], [[1, 3], [1, 3]])


def test_summary_rejects_missing_seed():
    frame = pd.DataFrame(
        {"model": ["hybrid"], "seed": [2026], "condition_id": ["point-0.1"]}
    )
    with pytest.raises(ValueError, match="incomplete stage A coverage"):
        validate_stage_a_coverage(frame, expected_recordings=("rec-a",))
