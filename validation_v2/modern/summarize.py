from __future__ import annotations

from collections.abc import Sequence
import itertools

import pandas as pd

from .config import ALL_MODELS


def validate_stage_a_coverage(frame: pd.DataFrame, *, expected_recordings: Sequence[str]) -> None:
    required = {"model", "seed", "condition_id", "recording_id"}
    if not required.issubset(frame.columns):
        raise ValueError("incomplete stage A coverage: missing columns")
    conditions = tuple(sorted(frame["condition_id"].dropna().unique()))
    if len(conditions) != 13:
        raise ValueError("incomplete stage A coverage: expected 13 conditions")
    expected = set(itertools.product(ALL_MODELS, range(2026, 2031), conditions, expected_recordings))
    actual = set(frame.loc[:, ["model", "seed", "condition_id", "recording_id"]].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError("incomplete stage A coverage")


__all__ = ["validate_stage_a_coverage"]
