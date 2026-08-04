from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from .artifacts import canonical_json, read_array_artifact
from .config import load_modern_config
from .export import export_modern_dataset


def _stable_json(path: Path, value: object) -> None:
    content = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"existing {path.name} differs from requested campaign")
        return
    with path.open("xb") as handle:
        handle.write(content); handle.flush(); os.fsync(handle.fileno())


def _plan(config_path: Path, output: Path) -> dict[str, object]:
    config = load_modern_config(config_path)
    conditions = [
        {"topology": topology, "requested_fraction": rate}
        for topology in config.topologies for rate in config.rates
    ]
    if config.irregular_cases:
        conditions.append({"topology": "irregular:interval_jitter+point", "requested_fraction": 0.3})
    tasks = []
    for seed in config.seeds:
        for model in config.models:
            payload = {"phase": "formal_training", "model": model, "seed": seed, "conditions": conditions}
            tasks.append({"task_id": hashlib.sha256(canonical_json(payload).encode()).hexdigest(), **payload})
    plan = {"schema_version": 1, "config": str(config_path), "models": list(config.models),
            "seeds": list(config.seeds), "conditions": conditions, "tasks": tasks}
    _stable_json(output / "campaign-plan.json", plan)
    return plan


def _export(config_path: Path, output: Path) -> dict[str, object]:
    config = load_modern_config(config_path)
    root = Path(__file__).resolve().parents[2]
    manifests = []
    for seed in config.seeds:
        manifests.append(export_modern_dataset(config, seed, root, output / "datasets" / str(seed)))
    report = {"status": "complete", "datasets": [item["dataset_id"] for item in manifests]}
    _stable_json(output / "export-report.json", report)
    return report


def _validate(output: Path) -> dict[str, object]:
    manifests = sorted(output.glob("datasets/*/dataset_manifest.json"))
    if not manifests:
        raise ValueError("no exported datasets found")
    count = 0
    for path in manifests:
        value = json.loads(path.read_text(encoding="utf-8"))
        for artifact in value["artifacts"]:
            read_array_artifact(path.parent / artifact["path"], expected_kind="dataset")
            count += 1
    report = {"status": "complete", "datasets": len(manifests), "artifacts": count}
    _stable_json(output / "validation-report.json", report)
    return report


def _package(output: Path, mode: str) -> dict[str, object]:
    destination = output / f"modern-results-{mode}"
    archive = Path(shutil.make_archive(str(destination), "zip", root_dir=output))
    return {"status": "complete", "mode": mode, "archive": str(archive), "bytes": archive.stat().st_size}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m validation_v2.modern.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "export", "tune", "run", "resume", "validate", "summarize"):
        item = commands.add_parser(name); item.add_argument("--config", type=Path, required=True); item.add_argument("--output", type=Path, required=True)
        if name in {"tune", "run", "resume"}:
            item.add_argument("--pypots-python", type=Path, required=True); item.add_argument("--sssd-python", type=Path, required=True)
    item = commands.add_parser("package-results"); item.add_argument("--config", type=Path, required=True); item.add_argument("--output", type=Path, required=True); item.add_argument("--mode", choices=("summary", "full"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "plan": result = _plan(arguments.config, arguments.output)
        elif arguments.command == "export": result = _export(arguments.config, arguments.output)
        elif arguments.command == "validate": result = _validate(arguments.output)
        elif arguments.command == "package-results": result = _package(arguments.output, arguments.mode)
        else:
            raise ValueError(f"{arguments.command} requires the MatPool campaign launcher")
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"modern-imputation: {error}", file=sys.stderr); return 2
    print(canonical_json(result)); return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
