"""Command-line entrypoints for validation-v2 experiments."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import subprocess
import sys

import yaml

from validation_v2.experiments.matrix import enumerate_matrix
from validation_v2.experiments.provenance import canonical_json
from validation_v2.experiments.runner import run_matrix, run_smoke
from validation_v2.experiments.sharding import (
    build_shard_plan,
    execute_shard,
    load_shard_plan,
    merge_shards,
    write_shard_plan,
)
from validation_v2.experiments.summarize import summarize_runs


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIRECTORY = _REPOSITORY_ROOT / "configs" / "validation_v2"


def _config_path(value: str) -> Path:
    supplied = Path(value)
    if supplied.is_absolute():
        candidate = supplied
    elif supplied.parent == Path("."):
        candidate = _CONFIG_DIRECTORY / supplied
    else:
        candidate = _REPOSITORY_ROOT / supplied
    if candidate.is_file():
        return candidate
    raise ValueError(f"config file does not exist: {value}")


def _mapping_config(value: str) -> Mapping[str, object]:
    path = _config_path(value)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("config must be a YAML mapping")
    return loaded


def _repository_path(value: Path) -> Path:
    return value if value.is_absolute() else _REPOSITORY_ROOT / value


def _current_git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("unable to determine current git_commit") from error
    commit = completed.stdout.strip()
    if not commit:
        raise ValueError("unable to determine current git_commit")
    return commit


def _write_matrix(config_path: str, *, dry_run: bool) -> None:
    combinations = enumerate_matrix(_mapping_config(config_path))
    print(
        canonical_json(
            {
                "command": "matrix",
                "combination_count": len(combinations),
                "dry_run": dry_run,
            }
        )
    )
    for combination in combinations:
        print(canonical_json(combination))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m validation_v2.cli")
    subcommands = parser.add_subparsers(dest="command", required=True)
    matrix = subcommands.add_parser("matrix")
    matrix.add_argument("--config", required=True)
    matrix.add_argument("--dry-run", action="store_true")
    matrix.add_argument("--output-root", type=Path)
    matrix.add_argument("--device", choices=("auto", "cpu", "cuda"))
    matrix.add_argument("--max-combinations", type=int)
    smoke = subcommands.add_parser("smoke")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--output-root", type=Path)
    smoke.add_argument("--device", choices=("auto", "cpu", "cuda"))
    summarize = subcommands.add_parser("summarize")
    summarize.add_argument("--root", type=Path, required=True)
    summarize.add_argument("--config")
    summarize.add_argument("--required-seeds", nargs="+", type=int)
    summarize.add_argument("--baseline", default="linear")
    shard_plan = subcommands.add_parser("shard-plan")
    shard_plan.add_argument("--config", required=True)
    shard_plan.add_argument("--shard-count", required=True, type=int)
    shard_plan.add_argument("--output", required=True, type=Path)
    shard_plan.add_argument("--device", required=True, choices=("cpu", "cuda"))
    shard = subcommands.add_parser("shard")
    shard.add_argument("--config", required=True)
    shard.add_argument("--plan", required=True, type=Path)
    shard.add_argument("--shard-index", required=True, type=int)
    shard.add_argument("--output-root", required=True, type=Path)
    shard.add_argument("--device", required=True, choices=("cpu", "cuda"))
    merge = subcommands.add_parser("merge-shards")
    merge.add_argument("--config", required=True)
    merge.add_argument("--plan", required=True, type=Path)
    merge.add_argument("--shards-root", required=True, type=Path)
    merge.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "matrix":
            if arguments.dry_run:
                _write_matrix(arguments.config, dry_run=True)
            else:
                report = run_matrix(
                    _mapping_config(arguments.config),
                    repository_root=Path(__file__).resolve().parents[1],
                    output_root=arguments.output_root,
                    requested_device=arguments.device,
                    max_combinations=arguments.max_combinations,
                )
                print(canonical_json(report))
            return 0
        if arguments.command == "smoke":
            report = run_smoke(
                _mapping_config(arguments.config),
                repository_root=Path(__file__).resolve().parents[1],
                output_root=arguments.output_root,
                requested_device=arguments.device,
            )
            print(canonical_json(report))
            return 0
        if arguments.command == "summarize":
            marker_path = arguments.root / "matrix_execution.json"
            if marker_path.is_file():
                marker = yaml.safe_load(marker_path.read_text(encoding="utf-8"))
                if marker.get("partial"):
                    raise ValueError("partial matrix execution cannot be formally summarized")
                if marker.get("status") != "completed":
                    raise ValueError("incomplete matrix execution cannot be formally summarized")
            if arguments.required_seeds:
                seeds = arguments.required_seeds
            elif arguments.config:
                seeds = list(_mapping_config(arguments.config).get("seeds", ()))
            else:
                seeds = sorted(
                    {
                        int(yaml.safe_load(path.read_text(encoding="utf-8"))["seed"])
                        for path in arguments.root.glob("*/run.json")
                    }
                )
            summary = summarize_runs(
                arguments.root, required_seeds=seeds, baseline=arguments.baseline
            )
            print(
                canonical_json(
                    {"status": "completed", "summary_rows": len(summary)}
                )
            )
            return 0
        if arguments.command == "shard-plan":
            config = _mapping_config(arguments.config)
            git_commit = _current_git_commit()
            output = _repository_path(arguments.output)
            if os.path.lexists(output):
                plan = load_shard_plan(
                    output,
                    config=config,
                    git_commit=git_commit,
                    device=arguments.device,
                )
                if plan["shard_count"] != arguments.shard_count:
                    raise ValueError(
                        "shard plan already exists with different shard_count"
                    )
            else:
                plan = build_shard_plan(
                    config,
                    shard_count=arguments.shard_count,
                    git_commit=git_commit,
                    device=arguments.device,
                )
                write_shard_plan(output, plan)
            print(canonical_json(plan))
            return 0
        if arguments.command == "shard":
            config = _mapping_config(arguments.config)
            plan = load_shard_plan(
                _repository_path(arguments.plan),
                config=config,
                git_commit=_current_git_commit(),
                device=arguments.device,
            )
            report = execute_shard(
                config,
                plan=plan,
                shard_index=arguments.shard_index,
                repository_root=_REPOSITORY_ROOT,
                output_root=arguments.output_root,
                requested_device=arguments.device,
            )
            print(canonical_json(report))
            return 0
        if arguments.command == "merge-shards":
            report = merge_shards(
                config_path=_config_path(arguments.config),
                plan_path=_repository_path(arguments.plan),
                shards_root=_repository_path(arguments.shards_root),
                output_root=_repository_path(arguments.output_root),
            )
            print(canonical_json(report))
            return 0
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        print(f"validation-v2: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
