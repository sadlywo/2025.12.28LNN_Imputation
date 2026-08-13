"""Aggregate per-record experiment metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from validation_v2.evaluation.statistics import (
    PER_RECORD_COLUMNS,
    paired_model_summary,
)


def summarize_runs(
    root: Path | str,
    required_seeds: Iterable[int],
    baseline: str = "baseline",
    *,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 10_000,
) -> pd.DataFrame:
    """Scan immutable runs, validate pairing, and write deterministic summaries."""

    root = Path(root)
    manifests: list[tuple[Path, dict]] = []
    for path in sorted(root.glob("*/run.json")):
        manifests.append((path, json.loads(path.read_text(encoding="utf-8"))))
    run_ids = [str(manifest["run_id"]) for _, manifest in manifests]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate run_id")
    seeds = {int(manifest["seed"]) for _, manifest in manifests}
    required = {int(seed) for seed in required_seeds}
    extra = sorted(seeds - required)
    if extra:
        raise ValueError("unexpected seeds: " + ", ".join(map(str, extra)))
    missing = sorted(required - seeds)
    if missing:
        raise ValueError("missing required seeds: " + ", ".join(map(str, missing)))

    frames: list[pd.DataFrame] = []
    for manifest_path, manifest in manifests:
        csv_path = manifest_path.parent / "per_record_metrics.csv"
        if not csv_path.is_file():
            raise ValueError(f"missing per_record_metrics.csv for run_id {manifest['run_id']}")
        # Keep artifact ingestion in the standard library.  Some cloud images
        # have a binary-incompatible pandas/numpy stack which can terminate the
        # process even when pandas' Python CSV engine performs type inference.
        # The statistical stage below still converts and validates every
        # numeric field explicitly.
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or ())
            frame = pd.DataFrame(reader, columns=columns)
        if list(frame.columns) != list(PER_RECORD_COLUMNS):
            raise ValueError("per-record metrics must use schema: " + ",".join(PER_RECORD_COLUMNS))
        for column in ("seed", "requested_fraction", "realized_fraction", "value"):
            frame[column] = [float(value) for value in frame[column]]
        if not (frame["run_id"].astype(str) == str(manifest["run_id"])).all():
            raise ValueError("CSV run_id does not match run manifest")
        if not (pd.to_numeric(frame["seed"], errors="raise") == int(manifest["seed"])).all():
            raise ValueError("CSV seed does not match run manifest")
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    # strict_file estimates overall file-disjoint performance; scenario-specific
    # inference is supplied by the scenario_holdout protocols.
    strict_file = metrics["protocol"].eq("strict_file")
    if strict_file.any():
        metrics.loc[strict_file, "scenario"] = "overall"
    summary = paired_model_summary(
        metrics,
        baseline=baseline,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
        required_seeds=required,
    )
    summary.to_csv(root / "summary.csv", index=False, lineterminator="\n")
    records = summary.to_dict(orient="records")
    (root / "summary.json").write_text(
        json.dumps(records, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    return summary
