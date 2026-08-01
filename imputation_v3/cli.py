"""Command-line entry point for imputation v3 teacher training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

import yaml

from imputation_v3.config import load_teacher_config
from imputation_v3.experiments.training import run_teacher_smoke
from validation_v2.experiments.provenance import canonical_json


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _config_path(value: str) -> Path:
    supplied = Path(value)
    candidate = supplied if supplied.is_absolute() else _REPOSITORY_ROOT / supplied
    if not candidate.is_file():
        raise ValueError(f"config file does not exist: {value}")
    return candidate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m imputation_v3.cli")
    subcommands = parser.add_subparsers(dest="command", required=True)
    teacher = subcommands.add_parser("teacher")
    teacher.add_argument("--config", required=True)
    teacher.add_argument("--smoke", action="store_true")
    teacher.add_argument("--device", choices=("auto", "cpu", "cuda"), required=True)
    teacher.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "teacher":
            if not arguments.smoke:
                raise ValueError(
                    "--smoke is required; the formal teacher matrix belongs to Task 10"
                )
            config = load_teacher_config(_config_path(arguments.config))
            report = run_teacher_smoke(
                config,
                repository_root=_REPOSITORY_ROOT,
                requested_device=arguments.device,
                output_root=arguments.output_root,
            )
            print(canonical_json(report))
            return 0
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        print(f"imputation-v3: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
