#!/usr/bin/env python3
"""Collect fail-closed rollout metrics for validation-v2 shard stages."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics


def timestamp(value):
    """Parse an RFC3339 timestamp while accepting a trailing ``Z``."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--indices", nargs="+", required=True)
    parser.add_argument("--stage-start", required=True)
    parser.add_argument("--gpu-csv", type=Path, required=True)
    parser.add_argument("--minimum-groups", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    return parser.parse_args()


def main():
    args = parse_arguments()
    stage_start = timestamp(args.stage_start)
    durations = []
    completion_times = []
    marker_statuses = {}
    per_shard_new_groups = {}

    for shard in args.indices:
        marker_path = args.shards_root / shard / "shard_execution.json"
        if not marker_path.is_file():
            raise SystemExit(4)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_statuses[shard] = marker["status"]
        if marker["status"] == "failed":
            print(json.dumps(marker, sort_keys=True))
            raise SystemExit(2)
        assert marker["status"] in {"started", "completed"}, marker
        previous = timestamp(marker["started_at"])
        new_groups = 0
        for binding in marker["group_runs"]:
            run_id = binding["run_ids"][0]
            ledger_path = args.shards_root / shard / run_id / "test_evaluation.json"
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            completed = timestamp(ledger["completed_at"])
            duration = (completed - previous).total_seconds()
            assert duration > 0, (shard, run_id, duration)
            previous = completed
            if completed >= stage_start:
                durations.append(duration)
                completion_times.append(completed)
                new_groups += 1
        per_shard_new_groups[shard] = new_groups

    missing_shard_progress = [
        shard for shard, count in per_shard_new_groups.items() if count < 1
    ]
    completed_without_progress = [
        shard
        for shard in missing_shard_progress
        if marker_statuses[shard] == "completed"
    ]
    if completed_without_progress:
        print(
            "completed active shards cannot contribute after stage_start: "
            "{}; per_shard_new_groups={}".format(
                completed_without_progress, per_shard_new_groups
            )
        )
        raise SystemExit(3)
    if missing_shard_progress:
        print(
            "waiting: every active shard needs one new group after stage_start; "
            "per_shard_new_groups={}".format(per_shard_new_groups)
        )
        raise SystemExit(4)
    if len(durations) < args.minimum_groups:
        print("waiting: {}/{} new groups".format(len(durations), args.minimum_groups))
        raise SystemExit(4)

    with args.gpu_csv.open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    assert samples, "GPU sampler produced no rows"
    used_samples = [float(row["memory_used_mib"]) for row in samples]
    total_samples = [float(row["memory_total_mib"]) for row in samples]
    peak_used = max(used_samples)
    peak_gpu_memory_ratio = max(
        used / total for used, total in zip(used_samples, total_samples)
    )
    elapsed_hours = (max(completion_times) - stage_start).total_seconds() / 3600
    assert elapsed_hours > 0, elapsed_hours
    metrics = {
        "stage_start": stage_start.isoformat(),
        "indices": args.indices,
        "marker_statuses": marker_statuses,
        "per_shard_new_groups": per_shard_new_groups,
        "new_group_count": len(durations),
        "group_durations_seconds": durations,
        "groups_per_hour": len(durations) / elapsed_hours,
        "median_group_seconds": statistics.median(durations),
        "peak_gpu_memory_mib": peak_used,
        "peak_gpu_memory_ratio": peak_gpu_memory_ratio,
    }
    baseline = None
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        metrics["baseline"] = str(args.baseline)
        metrics["gate_passed"] = (
            metrics["groups_per_hour"] >= baseline["groups_per_hour"] * 1.5
            and metrics["median_group_seconds"] < baseline["median_group_seconds"] * 1.8
            and metrics["peak_gpu_memory_ratio"] < 0.8
        )
    else:
        metrics["gate_passed"] = metrics["peak_gpu_memory_ratio"] < 0.8
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))
    try:
        assert all(count >= 1 for count in metrics["per_shard_new_groups"].values())
        if baseline:
            assert metrics["groups_per_hour"] >= baseline["groups_per_hour"] * 1.5
            assert metrics["median_group_seconds"] < baseline["median_group_seconds"] * 1.8
        assert metrics["peak_gpu_memory_ratio"] < 0.8
    except AssertionError as error:
        print("performance/resource gate failed: {}".format(error))
        raise SystemExit(10) from error


if __name__ == "__main__":
    main()
