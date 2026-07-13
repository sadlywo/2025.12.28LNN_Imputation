#!/usr/bin/env python3
"""Collect fail-closed rollout metrics for validation-v2 shard stages."""

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile


SHARD_RE = re.compile(r"^[0-7]{3}$")


class MetricsError(Exception):
    """A deterministic non-gate error with a process exit status."""

    def __init__(self, message, exit_code):
        super().__init__(message)
        self.exit_code = exit_code


def timestamp(value):
    if not isinstance(value, str):
        raise MetricsError("timestamp must be a string", 3)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise MetricsError("invalid timestamp: {}".format(value), 3) from error
    if parsed.tzinfo is None:
        raise MetricsError("timestamp must include a timezone: {}".format(value), 3)
    return parsed


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


def load_json(path, description):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MetricsError("cannot read {}: {}".format(description, path), 3) from error


def require_mapping(value, description):
    if not isinstance(value, dict):
        raise MetricsError("{} must be an object".format(description), 3)
    return value


def collect_group_metrics(args, stage_start):
    durations = []
    completion_times = []
    marker_statuses = {}
    per_shard_new_groups = {}
    for shard in args.indices:
        if not SHARD_RE.fullmatch(shard):
            raise MetricsError("invalid shard index: {}".format(shard), 3)
        marker_path = args.shards_root / shard / "shard_execution.json"
        if not marker_path.is_file() or marker_path.is_symlink():
            raise MetricsError("waiting: missing shard marker: {}".format(shard), 4)
        marker = require_mapping(load_json(marker_path, "shard marker"), "shard marker")
        status = marker.get("status")
        marker_statuses[shard] = status
        if status == "failed":
            print(json.dumps(marker, sort_keys=True))
            raise MetricsError("failed shard marker: {}".format(shard), 2)
        if status not in {"started", "completed"}:
            raise MetricsError("invalid shard marker status: {}".format(status), 3)
        previous = timestamp(marker.get("started_at"))
        group_runs = marker.get("group_runs")
        if not isinstance(group_runs, list):
            raise MetricsError("shard marker group_runs must be a list", 3)
        new_groups = 0
        for binding in group_runs:
            binding = require_mapping(binding, "group run binding")
            run_ids = binding.get("run_ids")
            if not isinstance(run_ids, list) or not run_ids or not isinstance(run_ids[0], str):
                raise MetricsError("group run binding must have run_ids[0]", 3)
            run_id = run_ids[0]
            ledger_path = args.shards_root / shard / run_id / "test_evaluation.json"
            if not ledger_path.is_file() or ledger_path.is_symlink():
                raise MetricsError("missing test ledger: {}".format(ledger_path), 3)
            ledger = require_mapping(load_json(ledger_path, "test ledger"), "test ledger")
            completed = timestamp(ledger.get("completed_at"))
            duration = (completed - previous).total_seconds()
            if not math.isfinite(duration) or duration <= 0:
                raise MetricsError("invalid group duration: {}".format(run_id), 3)
            previous = completed
            if completed >= stage_start:
                durations.append(duration)
                completion_times.append(completed)
                new_groups += 1
        per_shard_new_groups[shard] = new_groups
    return durations, completion_times, marker_statuses, per_shard_new_groups


def collect_gpu_samples(path, stage_start):
    if not path.is_file() or path.is_symlink():
        raise MetricsError("waiting: GPU sampler output is unavailable", 4)
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise MetricsError("cannot read GPU sampler CSV", 3) from error
    accepted = []
    for row in rows:
        if not isinstance(row, dict):
            raise MetricsError("invalid GPU sampler row", 3)
        sample_time = timestamp(row.get("timestamp_utc"))
        if sample_time < stage_start:
            continue
        values = []
        for field in ("memory_used_mib", "memory_total_mib", "utilization_percent"):
            raw = row.get(field)
            if raw is None or not raw.strip():
                raise MetricsError("incomplete GPU sampler row", 3)
            try:
                value = float(raw)
            except ValueError as error:
                raise MetricsError("non-numeric GPU sampler row", 3) from error
            if not math.isfinite(value):
                raise MetricsError("non-finite GPU sampler row", 3)
            values.append(value)
        used, total, _ = values
        if used < 0 or total <= 0:
            raise MetricsError("invalid GPU memory sample", 3)
        accepted.append((used, total))
    if not accepted:
        raise MetricsError("waiting: no valid GPU samples after stage_start", 4)
    return accepted


def load_baseline(path):
    baseline = require_mapping(load_json(path, "baseline"), "baseline")
    for field in ("groups_per_hour", "median_group_seconds"):
        value = baseline.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise MetricsError("invalid baseline {}".format(field), 3)
    return baseline


def write_metrics_atomically(output, metrics):
    parent = output.parent
    if not parent.is_dir() or parent.is_symlink():
        raise MetricsError("output parent must be a real directory", 3)
    if output.exists() or output.is_symlink():
        raise MetricsError("refusing to replace existing output: {}".format(output), 3)
    reservation = parent / ("." + output.name + ".publish.lock")
    try:
        os.mkdir(str(reservation))
    except FileExistsError as error:
        raise MetricsError("output publication already reserved: {}".format(output), 3) from error
    except OSError as error:
        raise MetricsError("cannot reserve output publication: {}".format(output), 3) from error
    temporary = None
    published = False
    try:
        if output.exists() or output.is_symlink():
            raise MetricsError("refusing to replace existing output: {}".format(output), 3)
        descriptor, temporary = tempfile.mkstemp(
            prefix="." + output.name + ".", suffix=".tmp", dir=str(parent)
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(output))
        temporary = None
        published = True
    except MetricsError:
        raise
    except OSError as error:
        raise MetricsError("cannot publish metrics: {}".format(output), 3) from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        if not published:
            try:
                os.rmdir(str(reservation))
            except OSError:
                pass


def main():
    args = parse_arguments()
    try:
        if args.minimum_groups < 1:
            raise MetricsError("minimum groups must be positive", 3)
        stage_start = timestamp(args.stage_start)
        durations, completion_times, marker_statuses, per_shard_new_groups = (
            collect_group_metrics(args, stage_start)
        )
        missing_shard_progress = [
            shard for shard, count in per_shard_new_groups.items() if count < 1
        ]
        completed_without_progress = [
            shard for shard in missing_shard_progress if marker_statuses[shard] == "completed"
        ]
        if completed_without_progress:
            raise MetricsError(
                "completed active shards cannot contribute after stage_start: {}; "
                "per_shard_new_groups={}".format(
                    completed_without_progress, per_shard_new_groups
                ),
                3,
            )
        if missing_shard_progress:
            raise MetricsError(
                "waiting: every active shard needs one new group after stage_start; "
                "per_shard_new_groups={}".format(per_shard_new_groups),
                4,
            )
        if len(durations) < args.minimum_groups:
            raise MetricsError(
                "waiting: {}/{} new groups".format(len(durations), args.minimum_groups), 4
            )
        samples = collect_gpu_samples(args.gpu_csv, stage_start)
        elapsed_hours = (max(completion_times) - stage_start).total_seconds() / 3600
        if not math.isfinite(elapsed_hours) or elapsed_hours <= 0:
            raise MetricsError("invalid stage elapsed time", 3)
        peak_used = max(used for used, _ in samples)
        peak_gpu_memory_ratio = max(used / total for used, total in samples)
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
            baseline = load_baseline(args.baseline)
            metrics["baseline"] = str(args.baseline)
            metrics["gate_passed"] = (
                metrics["groups_per_hour"] >= baseline["groups_per_hour"] * 1.5
                and metrics["median_group_seconds"] < baseline["median_group_seconds"] * 1.8
                and metrics["peak_gpu_memory_ratio"] < 0.8
            )
        else:
            metrics["gate_passed"] = metrics["peak_gpu_memory_ratio"] < 0.8
        write_metrics_atomically(args.output, metrics)
        print(json.dumps(metrics, sort_keys=True))
        if not metrics["gate_passed"]:
            return 10
        return 0
    except MetricsError as error:
        print(str(error), file=os.sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
