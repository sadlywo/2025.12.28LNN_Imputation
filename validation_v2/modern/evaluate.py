from __future__ import annotations

import numpy as np


def complete_samples(observed: np.ndarray, mask: np.ndarray, samples: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed)
    mask = np.asarray(mask)
    samples = np.asarray(samples)
    if observed.shape != mask.shape or samples.shape[1:] != observed.shape:
        raise ValueError("observed, mask, and sample shapes do not align")
    completed = np.where(mask[None].astype(bool), observed[None], samples)
    if not np.all(np.isfinite(completed)):
        raise ValueError("completed samples contain nonfinite values")
    return completed


__all__ = ["complete_samples"]
