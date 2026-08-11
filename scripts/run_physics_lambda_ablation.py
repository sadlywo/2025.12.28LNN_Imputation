"""Run the formal physics-weight grid without bypassing frame validation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation_v2.experiments.runner import run_matrix


DEFAULT_LAMBDAS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)


def _label(value: float) -> str:
    return "0" if value == 0.0 else f"{value:.0e}".replace("-", "m")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "validation_v2" / "physics_refactor_smoke.yaml",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    args = parser.parse_args()

    loaded = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("physics"), dict):
        raise ValueError("config must contain a physics mapping")
    output_base = ROOT / "results" / "physics_loss_refactor" / "v1" / "runs"
    rows: list[dict[str, object]] = []
    for value in args.lambdas:
        config = deepcopy(loaded)
        config["physics"]["lambda_physics"] = value
        destination = output_base / f"lambda_{_label(value)}_matrix"
        status = str(config["physics"].get("frame_validation_status", "missing"))
        if value > 0.0 and status != "validated":
            rows.append(
                {
                    "lambda_physics": value,
                    "status": "blocked_frame_validation",
                    "output_root": str(destination),
                }
            )
            continue
        if args.dry_run:
            rows.append(
                {
                    "lambda_physics": value,
                    "status": "planned",
                    "output_root": str(destination),
                }
            )
            continue
        report = run_matrix(
            config,
            repository_root=ROOT,
            output_root=destination,
            requested_device=args.device,
        )
        rows.append(
            {
                "lambda_physics": value,
                "status": report["status"],
                "output_root": str(destination),
            }
        )
    print(json.dumps({"lambda_ablation": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
