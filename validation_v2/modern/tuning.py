from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .artifacts import canonical_json
from .config import MODERN_MODELS


def _with_identifier(payload: Mapping[str, object]) -> dict[str, object]:
    value = dict(payload)
    value["configuration_id"] = hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()
    return value


def candidates(model: str) -> tuple[dict[str, object], ...]:
    if model == "brits":
        payloads = (
            {"model": model, "hidden_size": hidden_size, "learning_rate": learning_rate}
            for hidden_size in (32, 64)
            for learning_rate in (0.001, 0.0005)
        )
    elif model == "saits":
        payloads = (
            {
                "model": model,
                "n_layers": n_layers,
                "d_model": d_model,
                "learning_rate": learning_rate,
            }
            for n_layers, d_model in ((1, 64), (2, 128))
            for learning_rate in (0.001, 0.0005)
        )
    elif model == "csdi":
        payloads = (
            {
                "model": model,
                "channels": channels,
                "learning_rate": learning_rate,
                "n_diffusion_steps": 50,
                "schedule": "quad",
                "beta_start": 0.0001,
                "beta_end": 0.5,
            }
            for channels in (32, 64)
            for learning_rate in (0.001, 0.0005)
        )
    elif model == "sssd":
        payloads = (
            {
                "model": model,
                "residual_channels": width,
                "skip_channels": width,
                "learning_rate": learning_rate,
                "diffusion_steps": 200,
                "beta_0": 0.0001,
                "beta_T": 0.02,
                "num_res_layers": 36,
                "s4_d_state": 64,
                "s4_lmax": 30,
                "s4_bidirectional": True,
                "s4_layernorm": True,
                "s4_dropout": 0.0,
            }
            for width in (32, 64)
            for learning_rate in (0.001, 0.0005)
        )
    else:
        raise ValueError(f"unsupported tuning model: {model}")
    return tuple(_with_identifier(payload) for payload in payloads)


def select_candidate(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ValueError("candidate results must not be empty")
    normalized: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for source in rows:
        row = dict(source)
        identifier = row.get("configuration_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("candidate result is missing configuration_id")
        if identifier in identifiers:
            raise ValueError("duplicate configuration ID")
        identifiers.add(identifier)
        if row.get("status", "completed") != "completed":
            raise ValueError(f"failed candidate result: {identifier}")
        sort_values: list[float] = []
        for field in ("missing_rmse", "parameters", "latency_s"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"candidate result is missing numeric {field}")
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"candidate result has invalid {field}")
            sort_values.append(number)
        row["_selection_sort"] = (*sort_values, identifier)
        normalized.append(row)
    selected = min(normalized, key=lambda item: item["_selection_sort"])
    selected.pop("_selection_sort")
    return selected


def _plan_hash(models: Sequence[str] = MODERN_MODELS) -> str:
    plan = {model: list(candidates(model)) for model in models}
    return hashlib.sha256(canonical_json(plan).encode("utf-8")).hexdigest()


def write_selection_lock(
    path: Path,
    results: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    selected_models = tuple(model for model in MODERN_MODELS if model in results)
    if not selected_models or set(results) != set(selected_models):
        raise ValueError("tuning results must contain supported modern models")
    normalized_results = {
        model: [dict(row) for row in results[model]] for model in selected_models
    }
    dataset_ids = {
        row.get("tuning_dataset_artifact_id")
        for rows in normalized_results.values()
        for row in rows
    }
    if len(dataset_ids) != 1 or not isinstance(next(iter(dataset_ids), None), str):
        raise ValueError("all tuning rows must share one tuning dataset artifact ID")
    payload: dict[str, object] = {
        "schema_version": 1,
        "plan_hash": _plan_hash(selected_models),
        "tuning_dataset_artifact_id": next(iter(dataset_ids)),
        "seed": 2026,
        "sampling_count": 5,
        "results": normalized_results,
        "selected": {
            model: select_candidate(normalized_results[model])
            for model in selected_models
        },
    }
    lock = {
        **payload,
        "lock_hash": hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest(),
    }
    content = (canonical_json(lock) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return lock


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_selection_lock(
    path: Path, *, expected_plan_hash: str
) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    required = {
        "schema_version",
        "plan_hash",
        "tuning_dataset_artifact_id",
        "seed",
        "sampling_count",
        "results",
        "selected",
        "lock_hash",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("selection lock schema mismatch")
    if value["schema_version"] != 1:
        raise ValueError("unsupported selection lock version")
    if value["plan_hash"] != expected_plan_hash:
        raise ValueError("selection plan hash mismatch")
    payload = {key: item for key, item in value.items() if key != "lock_hash"}
    expected_lock_hash = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    if value["lock_hash"] != expected_lock_hash:
        raise ValueError("selection lock hash mismatch")
    if value["seed"] != 2026 or value["sampling_count"] != 5:
        raise ValueError("selection lock tuning protocol mismatch")
    return value


__all__ = [
    "candidates",
    "read_selection_lock",
    "select_candidate",
    "write_selection_lock",
]
