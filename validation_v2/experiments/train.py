"""Validation-only checkpoint selection and training orchestration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from .provenance import _validate_manifest, canonical_json, write_run_manifest


def select_best_checkpoint(history: Sequence[Mapping[str, Any]]) -> int:
    """Return the earliest epoch with minimum validation missing RMSE."""

    if not history:
        raise ValueError("history must not be empty")
    seen: set[int] = set()
    candidates: list[tuple[float, int]] = []
    for row in history:
        if "test" in row or row.get("split") == "test":
            raise ValueError("test metrics are forbidden during checkpoint selection")
        if set(row) - {"epoch", "train", "validation"}:
            raise ValueError("history may contain only epoch, train, and validation")
        epoch = row.get("epoch")
        if type(epoch) is not int:
            raise ValueError("each history row requires an integer epoch")
        if epoch in seen:
            raise ValueError("duplicate epoch")
        seen.add(epoch)
        train = row.get("train")
        validation = row.get("validation")
        if not isinstance(train, Mapping) or not isinstance(validation, Mapping):
            raise ValueError("each epoch requires train and validation metrics")
        value = validation.get("missing_rmse")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("validation missing_rmse must be finite")
        candidates.append((float(value), epoch))
    return min(candidates)[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_no_clobber(path: Path, content: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if path.read_bytes() != content:
                raise ValueError(f"{path.name} already has inconsistent content") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _save_current_best(model: Any, path: Path) -> None:
    """Atomically replace this run's mutable best candidate via a temporary file."""

    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".best-", suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(model.state_dict(), temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resume_run(
    run_dir: Path | str,
    manifest: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any] | None:
    """Return a completed run only when config, run id, and checkpoint all match."""

    _validate_manifest(manifest)
    run_dir = Path(run_dir)
    paths = {
        "manifest": run_dir / "run.json",
        "history": run_dir / "history.json",
        "checkpoint": run_dir / "best.pt",
        "metadata": run_dir / "checkpoint.json",
    }
    present = {name for name, path in paths.items() if path.exists()}
    if not present:
        return None
    if present != set(paths):
        raise ValueError("partial or inconsistent run outputs cannot be resumed")
    try:
        stored_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        history = json.loads(paths["history"].read_text(encoding="utf-8"))
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("partial or inconsistent run outputs cannot be resumed") from error
    _validate_manifest(stored_manifest)
    if (
        stored_manifest["run_id"] != manifest["run_id"]
        or stored_manifest["config_sha256"] != manifest["config_sha256"]
    ):
        raise ValueError("resolved config hash or run_id does not match")
    expected_fields = {
        "run_id", "best_epoch", "selection_split", "selection_metric", "checkpoint_sha256"
    }
    if set(metadata) != expected_fields or metadata["run_id"] != manifest["run_id"]:
        raise ValueError("partial or inconsistent checkpoint metadata")
    actual = _sha256(paths["checkpoint"])
    if actual != metadata["checkpoint_sha256"] or actual != checkpoint_sha256:
        raise ValueError("checkpoint hash does not match")
    if (
        metadata["selection_split"] != "validation"
        or metadata["selection_metric"] != "missing_rmse"
        or select_best_checkpoint(history) != metadata["best_epoch"]
    ):
        raise ValueError("checkpoint selection metadata is inconsistent")
    return metadata


def train_one_run(
    run_dir: Path | str,
    manifest: Mapping[str, Any],
    *,
    train_loader: Any,
    validation_loader: Any,
    epochs: int,
    train_epoch: Any,
    evaluate_epoch: Any,
    model: Any | None = None,
    optimizer: Any | None = None,
    model_factory: Any | None = None,
    optimizer_factory: Any | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Train one run and freeze the best validation-selected state dict."""

    run_dir = Path(run_dir)
    if run_dir.exists() and any(
        (run_dir / name).exists()
        for name in ("run.json", "history.json", "best.pt", "checkpoint.json")
    ):
        if expected_checkpoint_sha256 is None:
            raise ValueError("completed or partial run requires expected checkpoint hash")
        resumed = resume_run(run_dir, manifest, expected_checkpoint_sha256)
        if resumed is None:  # pragma: no cover - guarded by the file check above
            raise ValueError("partial or inconsistent run outputs cannot be resumed")
        return resumed
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if (model is None) == (model_factory is None):
        raise ValueError("provide exactly one of model or model_factory")
    if model is None:
        model = model_factory()
    if (optimizer is None) == (optimizer_factory is None):
        raise ValueError("provide exactly one of optimizer or optimizer_factory")
    if optimizer is None:
        optimizer = optimizer_factory(model)

    write_run_manifest(run_dir, manifest)
    history: list[dict[str, Any]] = []
    best_value = math.inf
    best_epoch: int | None = None
    checkpoint_path = run_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, optimizer, train_loader, epoch)
        validation_metrics = evaluate_epoch(model, validation_loader, epoch)
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        # Strict JSON validation catches unsupported and non-finite callback results.
        canonical_json(row)
        history.append(row)
        selected_epoch = select_best_checkpoint(history)
        if selected_epoch == epoch:
            best_value = float(validation_metrics["missing_rmse"])
            best_epoch = epoch
            _save_current_best(model, checkpoint_path)

    if best_epoch is None:  # pragma: no cover - epochs and metric validation guarantee this
        raise ValueError("no checkpoint was selected")
    if select_best_checkpoint(history) != best_epoch or not math.isfinite(best_value):
        raise ValueError("checkpoint selection is inconsistent")
    history_content = (canonical_json(history) + "\n").encode("utf-8")
    _write_no_clobber(run_dir / "history.json", history_content)
    checkpoint_sha256 = _sha256(checkpoint_path)
    metadata = {
        "run_id": manifest["run_id"],
        "best_epoch": best_epoch,
        "selection_split": "validation",
        "selection_metric": "missing_rmse",
        "checkpoint_sha256": checkpoint_sha256,
    }
    _write_no_clobber(
        run_dir / "checkpoint.json", (canonical_json(metadata) + "\n").encode("utf-8")
    )
    return metadata
