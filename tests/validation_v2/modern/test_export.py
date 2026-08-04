import numpy as np
import torch

from validation_v2.modern.export import build_observed_arrays, window_starts


def test_hidden_target_changes_do_not_change_external_input():
    target = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    mask = torch.tensor([[1, 0, 1, 1, 0, 1]] * 4, dtype=torch.float32)
    dt = torch.full((4,), 0.01)
    changed = torch.where(mask.bool(), target, target + 10000)
    first = build_observed_arrays(target, mask, dt)
    second = build_observed_arrays(changed, mask, dt)
    np.testing.assert_array_equal(first["X"], second["X"])
    np.testing.assert_array_equal(first["mask"], second["mask"])
    np.testing.assert_array_equal(first["dt"], second["dt"])


def test_window_starts_match_v2_half_overlap_and_tail():
    assert window_starts(73, seq_len=30) == (0, 15, 30, 43)
