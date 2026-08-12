from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np

from .artifacts import canonical_json, read_array_artifact
from .config import load_modern_config
from .export import export_modern_dataset
from .probability import empirical_crps, interval_metrics, stitch_samples
from .tuning import candidates, write_selection_lock


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
    for case in config.irregular_case_specs:
        conditions.append(
            {
                "topology": f"irregular:interval_jitter+{case['value_topology']}",
                "requested_fraction": case["value_requested_fraction"],
                "requested_irregularity": case["requested_irregularity"],
            }
        )
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
    campaign_complete = (output / "run-report.json").is_file()
    report = {"status": "complete" if campaign_complete else "datasets_complete",
              "scope": "formal" if len(manifests) == 5 else "smoke",
              "datasets": len(manifests), "artifacts": count,
              "campaign_complete": campaign_complete}
    _stable_json(output / "validation-report.json", report)
    return report


def _run_worker(python: Path, module: str, operation: str, task_path: Path) -> None:
    result = subprocess.run(
        [str(python), "-m", module, operation, "--task", str(task_path)],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    (task_path.parent / f"{operation}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (task_path.parent / f"{operation}.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise ValueError(f"{module} {operation} failed for {task_path}")


def _tune(config_path: Path, output: Path, pypots_python: Path, sssd_python: Path) -> dict[str, object]:
    config = load_modern_config(config_path)
    dataset = output / "datasets" / "2026"
    validation_base = dataset / "validation"
    validation_arrays, validation_manifest = read_array_artifact(validation_base, expected_kind="dataset")
    results: dict[str, list[dict[str, object]]] = {}
    for model in (model for model in config.models if model in {"brits", "saits", "csdi", "sssd"}):
        results[model] = []
        for candidate in candidates(model):
            work = output / "tuning" / model / str(candidate["configuration_id"])
            work.mkdir(parents=True, exist_ok=True)
            module = "validation_v2.modern.sssd_worker" if model == "sssd" else "validation_v2.modern.pypots_worker"
            python = sssd_python if model == "sssd" else pypots_python
            train = {
                "model": model, "configuration": candidate,
                "train_artifact": str(dataset / "train"), "validation_artifact": str(validation_base),
                "output_dir": str(work / "checkpoint"), "batch_size": config.batch_size,
                "epochs": config.epochs, "patience": config.patience, "device": "cuda",
            }
            if model == "sssd": train["source"] = str(Path(__file__).resolve().parents[2] / "third_party/sssd/source")
            train_path = work / "train-task.json"; _stable_json(train_path, train)
            started = time.perf_counter(); _run_worker(python, module, "train", train_path)
            checkpoint = work / "checkpoint" / ("best.pt" if model == "sssd" else "best.pypots")
            predict = {
                "model": model, "configuration": candidate, "dataset_artifact": str(validation_base),
                "checkpoint": str(checkpoint), "output_artifact": str(work / "validation-prediction"),
                "batch_size": config.batch_size, "device": "cuda",
                "n_sampling_times": 5 if model in {"csdi", "sssd"} else 1,
            }
            if model == "sssd": predict["source"] = train["source"]
            predict_path = work / "predict-task.json"; _stable_json(predict_path, predict)
            _run_worker(python, module, "predict", predict_path)
            prediction, _ = read_array_artifact(work / "validation-prediction", expected_kind="prediction")
            missing = validation_arrays["mask"] == 0
            rmse = float(np.sqrt(np.mean((prediction["mean"][missing] - validation_arrays["X_ori"][missing]) ** 2)))
            capacity = int(candidate.get("hidden_size", candidate.get("d_model", candidate.get("channels", candidate.get("residual_channels", 0)))))
            results[model].append({**candidate, "status": "completed", "missing_rmse": rmse,
                "parameters": capacity, "latency_s": time.perf_counter() - started,
                "tuning_dataset_artifact_id": validation_manifest["artifact_id"]})
    lock = write_selection_lock(output / "selected_hyperparameters.json", results)
    return {"status": "complete", "lock_hash": lock["lock_hash"], "selected": lock["selected"]}


def _stitch_static(windows: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    totals = np.zeros((length, windows.shape[-1]), dtype=np.float64); counts = np.zeros(length)
    for window, start in zip(windows, starts):
        stop = int(start) + len(window); totals[int(start):stop] += window; counts[int(start):stop] += 1
    if np.any(counts == 0): raise ValueError("evaluation windows do not cover the recording")
    return totals / counts[:, None]


def _run_modern(config_path: Path, output: Path, pypots_python: Path, sssd_python: Path) -> dict[str, object]:
    config = load_modern_config(config_path)
    existing_metrics = output / "modern_per_record_metrics.csv"
    if existing_metrics.exists():
        with existing_metrics.open("r", encoding="utf-8") as handle:
            return {"status": "complete", "rows": max(0, sum(1 for _ in handle) - 1), "metrics": str(existing_metrics)}
    lock = json.loads((output / "selected_hyperparameters.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for seed in config.seeds:
        dataset = output / "datasets" / str(seed)
        scaler = json.loads((dataset / "scaler.json").read_text(encoding="utf-8"))
        dataset_manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
        for model in (model for model in config.models if model in {"brits", "saits", "csdi", "sssd"}):
            configuration = lock["selected"][model]
            work = output / "formal" / str(seed) / model
            work.mkdir(parents=True, exist_ok=True)
            module = "validation_v2.modern.sssd_worker" if model == "sssd" else "validation_v2.modern.pypots_worker"
            python = sssd_python if model == "sssd" else pypots_python
            checkpoint = work / "checkpoint" / ("best.pt" if model == "sssd" else "best.pypots")
            if not checkpoint.exists():
                train = {"model": model, "configuration": configuration, "train_artifact": str(dataset / "train"),
                    "validation_artifact": str(dataset / "validation"), "output_dir": str(work / "checkpoint"),
                    "batch_size": config.batch_size, "epochs": config.epochs, "patience": config.patience, "device": "cuda"}
                if model == "sssd": train["source"] = str(Path(__file__).resolve().parents[2] / "third_party/sssd/source")
                task_path = work / "train-task.json"; _stable_json(task_path, train); _run_worker(python, module, "train", task_path)
            for artifact in dataset_manifest["artifacts"]:
                relative = str(artifact["path"])
                if not relative.startswith("test/"): continue
                prediction_base = work / "predictions" / relative
                if not prediction_base.with_suffix(".json").exists():
                    predict = {"model": model, "configuration": configuration, "dataset_artifact": str(dataset / relative),
                        "checkpoint": str(checkpoint), "output_artifact": str(prediction_base), "batch_size": config.batch_size,
                        "device": "cuda", "n_sampling_times": config.n_sampling_times if model in {"csdi", "sssd"} else 1}
                    if model == "sssd": predict["source"] = str(Path(__file__).resolve().parents[2] / "third_party/sssd/source")
                    task_path = prediction_base.parent / "predict-task.json"; _stable_json(task_path, predict); _run_worker(python, module, "predict", task_path)
                data, data_meta = read_array_artifact(dataset / relative, expected_kind="dataset")
                prediction, pred_meta = read_array_artifact(prediction_base, expected_kind="prediction")
                length = len(data["time"]); starts = data["starts"]
                samples = stitch_samples(prediction["samples"], starts, length)
                target = _stitch_static(data["X_ori"], starts, length)
                mask = np.rint(_stitch_static(data["mask"], starts, length)).astype(np.uint8)
                condition = data_meta["metadata"]["condition"]; recording = data_meta["metadata"]["recording_id"]
                recording_dataset = data_meta["metadata"]["dataset"]
                scale_payload = (
                    scaler["datasets"][recording_dataset]
                    if scaler.get("schema_version") == 2
                    else scaler
                )
                scale = np.asarray(scale_payload["scale"], dtype=np.float64)
                mean = samples.mean(axis=0); missing = mask == 0
                error = mean - target
                metrics = {
                    "reconstruction_normalized": float(np.sqrt(np.mean(error[missing] ** 2))),
                    "reconstruction_physical": float(np.sqrt(np.mean((error * scale[None, :])[missing] ** 2))),
                }
                if model in {"csdi", "sssd"}:
                    coverage, width = interval_metrics(samples, target, mask, level=0.95)
                    metrics.update(crps=empirical_crps(samples, target, mask), coverage_95=coverage, width_95=width)
                for metric, value in metrics.items():
                    rows.append({"model": model, "seed": seed, "dataset": recording_dataset, "recording_id": recording,
                        "condition_id": condition["condition_id"], "metric": metric, "value": value,
                        "checkpoint_sha256": pred_meta["metadata"]["checkpoint_sha256"],
                        "dataset_artifact_id": data_meta["artifact_id"], "prediction_artifact_id": pred_meta["artifact_id"]})
    metrics_path = output / "modern_per_record_metrics.csv"
    with metrics_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return {"status": "complete", "rows": len(rows), "metrics": str(metrics_path)}


def _run_references(config_path: Path, output: Path) -> dict[str, object]:
    from validation_v2.experiments.runner import run_matrix

    config = load_modern_config(config_path)
    reference_models = [model for model in config.models if model in {"linear", "locf", "bilstm", "bilnn", "hybrid"}]
    if not reference_models:
        return {"status": "skipped"}
    data_config = (
        {"datasets": [dict(item) for item in config.datasets]}
        if config.datasets
        else {"data_root": config.data_root, "dataset_name": config.dataset_name}
    )
    v2_config = {
        **data_config, "split_ratios": list(config.split_ratios),
        "output_root": str(output / "reference"),
        "selection_split": "validation", "seeds": list(config.seeds), "split_seed": config.split_seed,
        "seq_len": config.seq_len, "batch_size": config.batch_size, "epochs": config.epochs,
        "device": config.device, "require_clean_git": False, "models": reference_models,
        "protocols": [config.protocol], "topologies": list(config.topologies), "rates": list(config.rates),
        "objective": "reconstruction_only", "kinematic_ablation": {"name": "kinematic_ablation", "enabled": False},
        "trajectory_enabled": config.trajectory_enabled, "irregular_sampling_is_value_missing": False,
        "irregular_cases": [dict(case) for case in config.irregular_case_specs],
        "max_train_windows": config.max_train_windows, "max_eval_samples": config.max_eval_samples,
        "hidden_size": 32, "learning_rate": 0.001,
    }
    return run_matrix(v2_config, repository_root=Path(__file__).resolve().parents[2], output_root=output / "reference", requested_device=config.device)


def _run_all(config_path: Path, output: Path, pypots_python: Path, sssd_python: Path) -> dict[str, object]:
    reference = _run_references(config_path, output)
    modern = _run_modern(config_path, output, pypots_python, sssd_python)
    report = {"status": "complete", "reference": reference, "modern": modern}
    _stable_json(output / "run-report.json", report)
    return report


def _summarize(output: Path) -> dict[str, object]:
    existing = output / "summary" / "summary.json"
    if existing.is_file():
        return {"status": "complete", "rows": len(json.loads(existing.read_text(encoding="utf-8")))}
    metrics_path = output / "modern_per_record_metrics.csv"
    if not metrics_path.is_file():
        raise ValueError("modern metrics are missing")
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for reference_path in sorted((output / "reference").glob("*/per_record_metrics.csv")):
        with reference_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                topology = row["topology"]
                condition_id = (
                    "irregular-interval-jitter-point-30pct"
                    if topology.startswith("irregular:")
                    else f"{topology}-{int(round(float(row['requested_fraction']) * 100)):02d}pct"
                )
                rows.append({
                    "model": row["model"], "seed": row["seed"],
                    "recording_id": row["recording_id"], "condition_id": condition_id,
                    "metric": row["metric"], "value": row["value"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "dataset_artifact_id": "", "prediction_artifact_id": "",
                })
    if not rows:
        raise ValueError("no reference or modern metric rows were found")
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["condition_id"], row["metric"]), []).append(float(row["value"]))
    summary_dir = output / "summary"; summary_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "seed", "recording_id", "condition_id", "metric", "value",
                  "checkpoint_sha256", "dataset_artifact_id", "prediction_artifact_id"]
    with (summary_dir / "unified_per_record_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    summary_rows = [{"model": key[0], "condition_id": key[1], "metric": key[2],
                     "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                     "n": len(values)} for key, values in sorted(groups.items())]
    with (summary_dir / "summary.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0])); writer.writeheader(); writer.writerows(summary_rows)
    _stable_json(summary_dir / "summary.json", summary_rows)
    return {"status": "complete", "rows": len(summary_rows)}


def package_result_tree(source: Path, destination: Path, *, mode: str) -> dict[str, object]:
    if mode not in {"summary", "full"}:
        raise ValueError("result package mode must be summary or full")
    source = Path(source).resolve()
    excluded = {"checkpoints", "samples", "predictions"}
    files: dict[str, str] = {}
    selected: list[tuple[Path, str]] = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"result packages reject symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        if path.name.startswith("modern-results-"):
            continue
        if mode == "summary" and relative.split("/", 1)[0] in excluded:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[relative] = digest; selected.append((path, relative))
    manifest: dict[str, object] = {"schema_version": 1, "mode": mode, "files": files}
    manifest["manifest_hash"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    archive = Path(destination).with_name(Path(destination).name + f"-{mode}.tar.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "x:gz", format=tarfile.PAX_FORMAT) as handle:
        for path, relative in selected:
            info = handle.gettarinfo(str(path), arcname=relative)
            info.mtime = 0; info.uid = info.gid = 0; info.uname = info.gname = ""
            with path.open("rb") as stream:
                handle.addfile(info, stream)
    manifest["archive"] = str(archive)
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    _stable_json(archive.with_suffix(archive.suffix + ".manifest.json"), manifest)
    return manifest


def _package(output: Path, mode: str) -> dict[str, object]:
    return package_result_tree(output, output / "modern-results", mode=mode)


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
        elif arguments.command == "tune": result = _tune(arguments.config, arguments.output, arguments.pypots_python, arguments.sssd_python)
        elif arguments.command in {"run", "resume"}: result = _run_all(arguments.config, arguments.output, arguments.pypots_python, arguments.sssd_python)
        elif arguments.command == "validate": result = _validate(arguments.output)
        elif arguments.command == "summarize": result = _summarize(arguments.output)
        elif arguments.command == "package-results": result = _package(arguments.output, arguments.mode)
        else:
            raise ValueError(f"{arguments.command} requires the MatPool campaign launcher")
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"modern-imputation: {error}", file=sys.stderr); return 2
    print(canonical_json(result)); return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "package_result_tree"]
