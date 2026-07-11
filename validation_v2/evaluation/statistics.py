"""Paired, recording-level summaries for per-record experiment metrics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import rankdata


PER_RECORD_COLUMNS: Final[tuple[str, ...]] = (
    "run_id",
    "seed",
    "recording_id",
    "scenario",
    "protocol",
    "topology",
    "requested_fraction",
    "realized_fraction",
    "model",
    "metric",
    "value",
    "checkpoint_sha256",
)
GROUP_COLUMNS: Final[list[str]] = [
    "scenario",
    "protocol",
    "topology",
    "requested_fraction",
    "metric",
]
IDENTITY_COLUMNS: Final[list[str]] = [
    *GROUP_COLUMNS,
    "seed",
    "recording_id",
    "model",
]
SUMMARY_COLUMNS: Final[list[str]] = [
    *GROUP_COLUMNS,
    "model",
    "baseline",
    "mean",
    "sd",
    "median",
    "q1",
    "q3",
    "iqr",
    "mean_difference",
    "ci95_low",
    "ci95_high",
    "rank_biserial",
    "n_recordings",
    "n_seeds",
    "realized_fraction_mean",
    "realized_fraction_min",
    "realized_fraction_max",
]


def validate_per_record_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the fixed long-form per-record schema."""

    if list(frame.columns) != list(PER_RECORD_COLUMNS):
        raise ValueError("per-record metrics must use schema: " + ",".join(PER_RECORD_COLUMNS))
    checked = frame.copy()
    for column in ("seed", "requested_fraction", "realized_fraction", "value"):
        checked[column] = pd.to_numeric(checked[column], errors="raise")
    if not np.isfinite(checked["value"].to_numpy(dtype=float)).all():
        raise ValueError("value must be finite")
    for column in ("requested_fraction", "realized_fraction"):
        values = checked[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"{column} must be finite and between 0 and 1")
    if checked.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("duplicate per-record metric comparison key")
    return checked


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0.0].sum())
    negative = float(ranks[nonzero < 0.0].sum())
    return (positive - negative) / (positive + negative)


def _bootstrap_ci(
    differences: np.ndarray, rng: np.random.Generator, samples: int
) -> tuple[float, float]:
    if differences.size < 2:
        raise ValueError("at least 2 recordings are required for bootstrap CI")
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    indices = rng.integers(0, differences.size, size=(samples, differences.size))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _require_complete_matrix(
    group: pd.DataFrame, required_seeds: frozenset[int] | None
) -> None:
    seeds = sorted(group["seed"].unique())
    if required_seeds is not None and set(seeds) != required_seeds:
        raise ValueError("seed-record-model matrix has missing cells")
    recordings = sorted(group["recording_id"].unique())
    models = sorted(group["model"].unique())
    expected = len(seeds) * len(recordings) * len(models)
    if len(group) != expected:
        raise ValueError("seed-record-model matrix has missing cells")


def paired_model_summary(
    metrics: pd.DataFrame,
    baseline: str = "baseline",
    *,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 10_000,
    required_seeds: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Summarize candidate-minus-baseline differences with recordings as units."""

    metrics = validate_per_record_metrics(metrics)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(bootstrap_seed)
    required_seed_set = (
        None if required_seeds is None else frozenset(int(seed) for seed in required_seeds)
    )
    for group_key, group in metrics.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        _require_complete_matrix(group, required_seed_set)
        models = sorted(group["model"].unique())
        if baseline not in models:
            raise ValueError(f"baseline model not found: {baseline}")
        record_means = group.pivot_table(
            index="recording_id", columns="model", values="value", aggfunc="mean"
        ).sort_index()
        if len(record_means) < 2:
            raise ValueError("at least 2 recordings are required for bootstrap CI")
        baseline_values = record_means[baseline].to_numpy(dtype=float)
        group_values = dict(zip(GROUP_COLUMNS, group_key))
        n_seeds = int(group["seed"].nunique())
        for model in models:
            values = record_means[model].to_numpy(dtype=float)
            differences = values - baseline_values
            q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))
            low, high = _bootstrap_ci(differences, rng, bootstrap_samples)
            model_rows = group.loc[group["model"] == model, "realized_fraction"].to_numpy(float)
            rows.append(
                {
                    **group_values,
                    "model": model,
                    "baseline": baseline,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(median),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(q3 - q1),
                    "mean_difference": float(differences.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "rank_biserial": _rank_biserial(differences),
                    "n_recordings": int(len(record_means)),
                    "n_seeds": n_seeds,
                    "realized_fraction_mean": float(model_rows.mean()),
                    "realized_fraction_min": float(model_rows.min()),
                    "realized_fraction_max": float(model_rows.max()),
                }
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS).sort_values(
        [*GROUP_COLUMNS, "model"], ignore_index=True
    )
