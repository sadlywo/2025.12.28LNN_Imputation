# Modern Imputation Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a directly uploadable MatPool campaign that retrains the current Hybrid and reference models, adds BRITS, SAITS, CSDI, and SSSD-S4, performs validation-only tuning, evaluates 50-sample probabilistic imputations, and packages auditable results for download.

**Architecture:** `validation_v2` remains the control plane and the only owner of splits, normalization, masks, evaluation, and statistics. BRITS/SAITS/CSDI run through a pinned PyPOTS worker; SSSD-S4 runs through a second pinned Python environment against an exact official source snapshot. Both workers exchange versioned, hash-verified NPZ/JSON artifacts with the control plane, so model environments never make their own split or mask decisions.

**Tech Stack:** Python 3.10, PyTorch 2.3.1 for V2/PyPOTS, PyPOTS 1.5, isolated PyTorch 1.13.0 SSSD runtime, NumPy, pandas, SciPy, PyYAML, pytest, Bash/tmux on MatPool Linux, PowerShell/tar/git on Windows.

---

## File map

### New control-plane modules

- `validation_v2/modern/__init__.py`: public modern-benchmark API.
- `validation_v2/modern/config.py`: strict modern config parsing and model registry.
- `validation_v2/modern/artifacts.py`: canonical JSON, SHA-256, NPZ/JSON schemas, atomic writes.
- `validation_v2/modern/export.py`: export train/validation/test windows from V2-owned data.
- `validation_v2/modern/probability.py`: probabilistic stitching, empirical CRPS, coverage, width.
- `validation_v2/modern/tuning.py`: pre-registered candidate matrix and immutable selection lock.
- `validation_v2/modern/pypots_worker.py`: BRITS/SAITS/CSDI train and predict process.
- `validation_v2/modern/sssd_worker.py`: official SSSD-S4 model construction, training, sampling.
- `validation_v2/modern/campaign.py`: task state machine, subprocess dispatch, resume semantics.
- `validation_v2/modern/evaluate.py`: import predictions and emit V2 metric rows.
- `validation_v2/modern/summarize.py`: nine-model deterministic and probabilistic summaries.
- `validation_v2/modern/cli.py`: prepare, tune, run, validate, summarize, and package commands.

### New configurations and dependency locks

- `configs/validation_v2/modern_smoke.yaml`: bounded local/server smoke.
- `configs/validation_v2/modern_tuning.yaml`: exact four-candidate grids.
- `configs/validation_v2/modern_stage_a.yaml`: strict-file five-seed formal matrix.
- `requirements-modern-pypots.txt`: V2/PyPOTS environment lock.
- `requirements-modern-sssd.txt`: minimal official SSSD runtime lock.
- `third_party/sssd/manifest.json`: source commit, upstream URL, license, archive identity.
- `third_party/sssd/LICENSE`: upstream MIT license.
- `third_party/sssd/compatibility.patch`: minimal import/runtime compatibility patch.

### New operations files

- `scripts/run_modern_imputation_matpool.sh`: tmux launcher and campaign lifecycle.
- `scripts/package_modern_experiment.ps1`: exact-commit Windows upload bundle.
- `docs/modern_imputation_matpool_runbook_zh.md`: upload/run/status/resume/download instructions.

### Tests

- `tests/validation_v2/modern/test_config.py`
- `tests/validation_v2/modern/test_artifacts.py`
- `tests/validation_v2/modern/test_export.py`
- `tests/validation_v2/modern/test_probability.py`
- `tests/validation_v2/modern/test_tuning.py`
- `tests/validation_v2/modern/test_pypots_worker.py`
- `tests/validation_v2/modern/test_sssd_worker.py`
- `tests/validation_v2/modern/test_campaign.py`
- `tests/validation_v2/modern/test_evaluate_summarize.py`
- `tests/validation_v2/modern/test_matpool_launcher.py`
- `tests/validation_v2/modern/test_packaging.py`
- `tests/validation_v2/modern/test_smoke.py`

## Milestone 1: Shared contracts, exported data, and probability math

### Task 1: Strict modern configuration and model registry

**Files:**
- Create: `validation_v2/modern/__init__.py`
- Create: `validation_v2/modern/config.py`
- Create: `configs/validation_v2/modern_smoke.yaml`
- Create: `configs/validation_v2/modern_tuning.yaml`
- Create: `configs/validation_v2/modern_stage_a.yaml`
- Test: `tests/validation_v2/modern/test_config.py`

- [ ] **Step 1: Write failing registry and configuration tests**

```python
from pathlib import Path

import pytest

from validation_v2.modern.config import (
    MODERN_MODELS,
    REFERENCE_MODELS,
    load_modern_config,
)


def test_registry_declares_exact_main_table_models():
    assert REFERENCE_MODELS == ("linear", "locf", "bilstm", "bilnn", "hybrid")
    assert MODERN_MODELS == ("brits", "saits", "csdi", "sssd")


def test_stage_a_is_strict_file_five_seed_thirteen_condition_campaign():
    config = load_modern_config(Path("configs/validation_v2/modern_stage_a.yaml"))
    assert config.protocol == "strict_file"
    assert config.seeds == (2026, 2027, 2028, 2029, 2030)
    assert config.rates == (0.1, 0.2, 0.3, 0.4)
    assert config.topologies == ("point", "block", "channel")
    assert config.irregular_cases == 1
    assert config.n_sampling_times == 50
    assert config.models == REFERENCE_MODELS + MODERN_MODELS


def test_config_rejects_unknown_key(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("data_root: Oxford Dataset\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_modern_config(path)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/validation_v2/modern/test_config.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'validation_v2.modern'`.

- [ ] **Step 3: Implement the frozen config dataclass and exact-key parser**

```python
# validation_v2/modern/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REFERENCE_MODELS = ("linear", "locf", "bilstm", "bilnn", "hybrid")
MODERN_MODELS = ("brits", "saits", "csdi", "sssd")
ALL_MODELS = REFERENCE_MODELS + MODERN_MODELS
_KEYS = {
    "data_root", "output_root", "protocol", "seeds", "split_seed", "seq_len",
    "batch_size", "epochs", "patience", "device", "models", "rates",
    "topologies", "irregular_cases", "n_sampling_times", "tuning_sampling_times",
    "max_train_windows", "max_eval_samples", "trajectory_enabled",
}


@dataclass(frozen=True)
class ModernConfig:
    data_root: str
    output_root: str
    protocol: str
    seeds: tuple[int, ...]
    split_seed: int
    seq_len: int
    batch_size: int
    epochs: int
    patience: int
    device: str
    models: tuple[str, ...]
    rates: tuple[float, ...]
    topologies: tuple[str, ...]
    irregular_cases: int
    n_sampling_times: int
    tuning_sampling_times: int
    max_train_windows: int
    max_eval_samples: int | None
    trajectory_enabled: bool


def load_modern_config(path: Path | str) -> ModernConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    unknown = sorted(set(raw) - _KEYS)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    config = ModernConfig(
        data_root=str(raw["data_root"]), output_root=str(raw["output_root"]),
        protocol=str(raw["protocol"]), seeds=tuple(raw["seeds"]),
        split_seed=int(raw["split_seed"]), seq_len=int(raw["seq_len"]),
        batch_size=int(raw["batch_size"]), epochs=int(raw["epochs"]),
        patience=int(raw["patience"]), device=str(raw["device"]),
        models=tuple(raw["models"]), rates=tuple(float(x) for x in raw["rates"]),
        topologies=tuple(raw["topologies"]),
        irregular_cases=len(raw.get("irregular_cases", [])),
        n_sampling_times=int(raw["n_sampling_times"]),
        tuning_sampling_times=int(raw["tuning_sampling_times"]),
        max_train_windows=int(raw["max_train_windows"]),
        max_eval_samples=raw["max_eval_samples"],
        trajectory_enabled=bool(raw["trajectory_enabled"]),
    )
    if config.protocol != "strict_file" or config.seq_len != 30:
        raise ValueError("modern stage A requires strict_file and seq_len 30")
    if any(model not in ALL_MODELS for model in config.models):
        raise ValueError("unsupported modern benchmark model")
    if config.n_sampling_times != 50 or config.tuning_sampling_times != 5:
        raise ValueError("formal/tuning sampling counts must be 50/5")
    return config
```

Create `validation_v2/modern/__init__.py` exporting `ModernConfig` and `load_modern_config`. Write the three YAML files with the exact values approved in the design; the stage-A file must contain all nine models, five seeds, three topologies, four rates, one interval-jitter case, 100 epochs, patience 20, and 50/5 sampling counts. The smoke file uses seed 2026, one point-30% condition, four windows, one epoch, and two samples.

- [ ] **Step 4: Run tests and config enumeration**

Run: `python -m pytest tests/validation_v2/modern/test_config.py -q`

Expected: all tests pass.

Run: `python -c "from validation_v2.modern.config import load_modern_config; print(load_modern_config('configs/validation_v2/modern_stage_a.yaml').models)"`

Expected: the exact nine-model tuple is printed.

- [ ] **Step 5: Commit**

```bash
git add validation_v2/modern configs/validation_v2/modern_*.yaml tests/validation_v2/modern/test_config.py
git commit -m "feat(modern): define benchmark configurations"
```

### Task 2: Versioned hash-verified artifact contract

**Files:**
- Create: `validation_v2/modern/artifacts.py`
- Test: `tests/validation_v2/modern/test_artifacts.py`

- [ ] **Step 1: Write failing round-trip, tamper, and no-clobber tests**

```python
import json
from pathlib import Path

import numpy as np
import pytest

from validation_v2.modern.artifacts import read_array_artifact, write_array_artifact


def test_array_artifact_round_trip_and_hashes(tmp_path: Path):
    arrays = {"target": np.arange(12, dtype=np.float32).reshape(2, 3, 2)}
    manifest = write_array_artifact(tmp_path / "bundle", "dataset", arrays, {"seed": 2026})
    loaded, metadata = read_array_artifact(tmp_path / "bundle", expected_kind="dataset")
    np.testing.assert_array_equal(loaded["target"], arrays["target"])
    assert metadata["artifact_id"] == manifest["artifact_id"]


def test_array_artifact_rejects_tampered_npz(tmp_path: Path):
    write_array_artifact(tmp_path / "bundle", "prediction", {"x": np.ones(2)}, {})
    (tmp_path / "bundle.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_array_artifact(tmp_path / "bundle", expected_kind="prediction")


def test_array_artifact_refuses_overwrite(tmp_path: Path):
    write_array_artifact(tmp_path / "bundle", "dataset", {"x": np.ones(2)}, {})
    with pytest.raises(FileExistsError):
        write_array_artifact(tmp_path / "bundle", "dataset", {"x": np.ones(2)}, {})
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/validation_v2/modern/test_artifacts.py -q`

Expected: import fails because `artifacts.py` does not exist.

- [ ] **Step 3: Implement atomic NPZ plus canonical JSON sidecars**

Implement these exact public signatures in `artifacts.py`: `sha256_file(path: Path) -> str`,
`canonical_json(value: object) -> str`, `write_array_artifact(base: Path, kind: str,
arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> dict[str, object]`,
and `read_array_artifact(base: Path, *, expected_kind: str) -> tuple[dict[str,
np.ndarray], dict[str, object]]`.

The manifest schema is exactly:

```python
{
    "schema_version": 1,
    "kind": kind,
    "artifact_id": sha256(canonical payload without artifact_id),
    "npz_sha256": sha256_file(base.with_suffix(".npz")),
    "arrays": {name: {"shape": list(array.shape), "dtype": str(array.dtype)}},
    "metadata": metadata,
}
```

Write temporary files in the destination directory, `fsync`, then publish with a no-clobber hard link as used by `validation_v2.experiments.evaluate`. On read, reject unknown top-level keys, wrong kind/version, duplicate JSON keys, path mismatch, file hash mismatch, array set mismatch, shape mismatch, or dtype mismatch.

- [ ] **Step 4: Run artifact tests**

Run: `python -m pytest tests/validation_v2/modern/test_artifacts.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add validation_v2/modern/artifacts.py tests/validation_v2/modern/test_artifacts.py
git commit -m "feat(modern): add immutable array artifacts"
```

### Task 3: V2-owned external model data export

**Files:**
- Create: `validation_v2/modern/export.py`
- Modify: `validation_v2/experiments/runner.py` to expose a thin preparation API without changing existing behavior
- Test: `tests/validation_v2/modern/test_export.py`

- [ ] **Step 1: Write failing equality and leakage-invariance tests**

```python
import numpy as np
import torch

from validation_v2.modern.export import build_observed_arrays, window_starts


def test_hidden_target_changes_do_not_change_external_input():
    target = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    mask = torch.tensor([[1, 0, 1, 1, 0, 1]] * 4, dtype=torch.float32)
    dt = torch.full((4,), 0.01)
    changed = torch.where(mask.bool(), target, target + 10000)
    first = build_observed_arrays(target, mask, dt)
    second = build_observed_arrays(changed, mask, dt)
    np.testing.assert_array_equal(first["X"], second["X"])
    np.testing.assert_array_equal(first["mask"], second["mask"])
    np.testing.assert_array_equal(first["dt"], second["dt"])


def test_window_starts_match_v2_half_overlap_and_tail():
    assert window_starts(73, seq_len=30) == (0, 15, 30, 43)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/validation_v2/modern/test_export.py -q`

Expected: import fails because `export.py` does not exist.

- [ ] **Step 3: Implement observed arrays and canonical export**

```python
def build_observed_arrays(target: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor) -> dict[str, np.ndarray]:
    observed = torch.where(mask.bool(), target, torch.full_like(target, float("nan")))
    return {
        "X": observed.cpu().numpy().astype(np.float32, copy=False),
        "mask": mask.cpu().numpy().astype(np.uint8, copy=False),
        "dt": dt.cpu().numpy().astype(np.float32, copy=False),
    }


def window_starts(length: int, *, seq_len: int) -> tuple[int, ...]:
    if length < seq_len:
        return (0,)
    stride = max(1, seq_len // 2)
    starts = list(range(0, length - seq_len + 1, stride))
    tail = length - seq_len
    if starts[-1] != tail:
        starts.append(tail)
    return tuple(starts)
```

Add `export_modern_dataset(config, seed, repository_root, output_dir)` that calls the same V2 functions used by `run_smoke` for record discovery, split assignment, train-only scaler fitting, `_prepared_sequence`, and mask generation. Export:

```text
train.npz/json: X, X_ori, mask, dt, recording_index
validation.npz/json: X, X_ori, mask, dt, recording_index
test/<condition>/<recording>.npz/json: X, X_ori, mask, dt, time, starts
split_manifest.csv, scaler.json, dataset_manifest.json
```

Training and validation use point-30% masks. Test exports all 12 topology/rate conditions and the single irregular case. The dataset manifest stores split/scaler hashes and the ordered artifact IDs. Add a public wrapper in `runner.py` only where necessary to prevent the exporter from reimplementing split or masking algorithms.

- [ ] **Step 4: Run export tests plus existing masking tests**

Run: `python -m pytest tests/validation_v2/modern/test_export.py tests/validation_v2/test_masking_and_features.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add validation_v2/modern/export.py validation_v2/experiments/runner.py tests/validation_v2/modern/test_export.py
git commit -m "feat(modern): export shared leakage-safe datasets"
```

### Task 4: Sample-wise stitching and probability metrics

**Files:**
- Create: `validation_v2/modern/probability.py`
- Test: `tests/validation_v2/modern/test_probability.py`

- [ ] **Step 1: Write failing analytic tests**

```python
import numpy as np

from validation_v2.modern.probability import empirical_crps, interval_metrics, stitch_samples


def test_stitch_samples_averages_each_sample_before_quantiles():
    windows = np.array([
        [[[[0.0], [2.0], [4.0]]], [[[10.0], [12.0], [14.0]]]],
        [[[[6.0], [8.0], [10.0]]], [[[16.0], [18.0], [20.0]]]],
    ]).reshape(2, 2, 3, 1)
    result = stitch_samples(windows, starts=(0, 2), length=5)
    np.testing.assert_allclose(result[:, :, 0], [[0, 2, 5, 8, 10], [10, 12, 15, 18, 20]])


def test_empirical_crps_matches_two_sample_closed_form():
    samples = np.array([[[0.0]], [[2.0]]])
    target = np.array([[1.0]])
    mask = np.array([[0]], dtype=np.uint8)
    assert empirical_crps(samples, target, mask) == 0.5


def test_interval_metrics_report_coverage_and_width():
    samples = np.array([[[-1.0]], [[0.0]], [[1.0]]])
    coverage, width = interval_metrics(samples, np.array([[0.5]]), np.array([[0]]), level=0.95)
    assert coverage == 1.0
    assert width > 1.0
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/validation_v2/modern/test_probability.py -q`

Expected: import fails because `probability.py` does not exist.

- [ ] **Step 3: Implement vectorized metrics with explicit sample axis**

```python
def empirical_crps(samples: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    missing = np.asarray(mask) == 0
    values = np.asarray(samples, dtype=np.float64)[:, missing]
    truth = np.asarray(target, dtype=np.float64)[missing]
    first = np.mean(np.abs(values - truth[None, :]), axis=0)
    pairwise = np.mean(np.abs(values[:, None, :] - values[None, :, :]), axis=(0, 1))
    return float(np.mean(first - 0.5 * pairwise))


def interval_metrics(samples, target, mask, *, level):
    alpha = 1.0 - level
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2], axis=0)
    missing = np.asarray(mask) == 0
    truth = np.asarray(target)[missing]
    coverage = np.mean((truth >= low[missing]) & (truth <= high[missing]))
    width = np.mean(high[missing] - low[missing])
    return float(coverage), float(width)
```

Implement `stitch_samples(window_samples, starts, length)` for shape `(windows, samples, steps, features)`. Reject nonfinite values, duplicate/unsorted starts, uncovered positions, wrong window length, or fewer than two samples. Return `(samples, length, features)`.

- [ ] **Step 4: Run probability tests**

Run: `python -m pytest tests/validation_v2/modern/test_probability.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add validation_v2/modern/probability.py tests/validation_v2/modern/test_probability.py
git commit -m "feat(modern): add probabilistic stitching and metrics"
```

## Milestone 2: Tuning and modern model workers

### Task 5: Pre-registered tuning matrix and immutable winner lock

**Files:**
- Create: `validation_v2/modern/tuning.py`
- Test: `tests/validation_v2/modern/test_tuning.py`

- [ ] **Step 1: Write failing candidate and tie-break tests**

```python
from validation_v2.modern.tuning import candidates, select_candidate


def test_each_modern_model_has_four_stable_candidates():
    for model in ("brits", "saits", "csdi", "sssd"):
        values = candidates(model)
        assert len(values) == 4
        assert len({item["configuration_id"] for item in values}) == 4


def test_selection_uses_rmse_then_parameters_then_latency_then_id():
    rows = [
        {"configuration_id": "b", "missing_rmse": 0.2, "parameters": 20, "latency_s": 1.0},
        {"configuration_id": "a", "missing_rmse": 0.2, "parameters": 10, "latency_s": 2.0},
    ]
    assert select_candidate(rows)["configuration_id"] == "a"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/validation_v2/modern/test_tuning.py -q`

Expected: import fails because `tuning.py` does not exist.

- [ ] **Step 3: Implement exact grids, selection, and no-clobber lock**

Encode the approved grids directly, adding fixed diffusion parameters to CSDI and SSSD
candidates. Compute `configuration_id` as SHA-256 of canonical JSON. Implement the exact
public functions `candidates(model: str) -> tuple[dict[str, object], ...]`,
`select_candidate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]`,
`write_selection_lock(path: Path, results: Mapping[str, Sequence[Mapping[str, object]]])
-> dict[str, object]`, and `read_selection_lock(path: Path, *, expected_plan_hash: str)
-> dict[str, object]`.

`select_candidate` rejects missing/nonfinite RMSE, failed rows, and duplicate configuration IDs. Sort by `(missing_rmse, parameters, latency_s, configuration_id)`. The selection lock contains all candidate results, chosen configs, tuning dataset artifact ID, seed 2026, sampling count 5, and its own hash. Refuse overwrite.

- [ ] **Step 4: Run tuning tests**

Run: `python -m pytest tests/validation_v2/modern/test_tuning.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add validation_v2/modern/tuning.py tests/validation_v2/modern/test_tuning.py
git commit -m "feat(modern): preregister validation tuning"
```

### Task 6: PyPOTS BRITS, SAITS, and CSDI worker

**Files:**
- Create: `validation_v2/modern/pypots_worker.py`
- Create: `requirements-modern-pypots.txt`
- Test: `tests/validation_v2/modern/test_pypots_worker.py`

- [ ] **Step 1: Write failing factory, input, and output-axis tests using injected classes**

```python
import numpy as np

from validation_v2.modern.pypots_worker import build_model, normalize_imputation


def test_saits_factory_uses_registered_shape_and_capacity():
    calls = {}
    class Fake:
        def __init__(self, **kwargs): calls.update(kwargs)
    build_model("saits", {"n_layers": 1, "d_model": 64, "learning_rate": 0.001},
                n_steps=30, n_features=6, batch_size=8, epochs=2, patience=1,
                device="cpu", classes={"saits": Fake})
    assert calls["n_steps"] == 30 and calls["n_features"] == 6
    assert calls["d_model"] == 64 and calls["n_heads"] == 4


def test_csdi_sample_axis_is_normalized():
    raw = np.zeros((3, 50, 30, 6), dtype=np.float32)
    assert normalize_imputation(raw, windows=3, samples=50, steps=30, features=6).shape == (3, 50, 30, 6)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/validation_v2/modern/test_pypots_worker.py -q`

Expected: import fails because `pypots_worker.py` does not exist.

- [ ] **Step 3: Implement lazy imports, exact factories, training, and prediction**

Use lazy imports so control-plane tests do not require PyPOTS. Factory mappings:

```python
from pypots.imputation import BRITS, CSDI, SAITS
from pypots.optim import Adam

optimizer = Adam(lr=config["learning_rate"])
BRITS(n_steps=30, n_features=6, rnn_hidden_size=config["hidden_size"],
      batch_size=batch_size, epochs=epochs, patience=patience, optimizer=optimizer,
      num_workers=0, device=device, saving_path=saving_path,
      model_saving_strategy="best", verbose=True)
SAITS(n_steps=30, n_features=6, n_layers=config["n_layers"], d_model=config["d_model"],
      n_heads=4, d_k=config["d_model"] // 4, d_v=config["d_model"] // 4,
      d_ffn=2 * config["d_model"], dropout=0.1, attn_dropout=0.1,
      batch_size=batch_size, epochs=epochs, patience=patience, optimizer=optimizer,
      num_workers=0, device=device, saving_path=saving_path,
      model_saving_strategy="best", verbose=True)
CSDI(n_steps=30, n_features=6, n_layers=4, n_heads=4,
     n_channels=config["n_channels"], d_time_embedding=128, d_feature_embedding=16,
     d_diffusion_embedding=128, n_diffusion_steps=50, target_strategy="random",
     schedule="quad", beta_start=0.0001, beta_end=0.5,
     batch_size=batch_size, epochs=epochs, patience=patience, optimizer=optimizer,
     num_workers=0, device=device, saving_path=saving_path,
     model_saving_strategy="best", verbose=True)
```

Train with `train_set={"X": train_X}` and `val_set={"X": val_X, "X_ori": val_X_ori}`. Save through the model API and write a checkpoint manifest. Predict deterministic models with one imputation and CSDI with `n_sampling_times` equal to 5 or 50. Normalize output to `(windows, samples, steps, features)`; deterministic models use sample dimension 1. Force observed positions from `X` after prediction and reject nonfinite missing predictions.

Expose a CLI:

```text
python -m validation_v2.modern.pypots_worker train --task task.json
python -m validation_v2.modern.pypots_worker predict --task task.json
python -m validation_v2.modern.pypots_worker preflight --output environment.json
```

Pin `pypots==1.5` and its resolved transitive requirements after installing into a clean Python 3.10 environment. Keep `torch==2.3.1` in the existing CUDA 12.1 installation path rather than allowing PyPOTS to replace it.

- [ ] **Step 4: Install the isolated local PyPOTS environment and run tests**

Run: `python -m venv .venv-modern-pypots`

Run on Windows: `.venv-modern-pypots\Scripts\python -m pip install -r requirements-modern-pypots.txt`

Run: `.venv-modern-pypots\Scripts\python -m pytest tests/validation_v2/modern/test_pypots_worker.py -q`

Expected: all tests pass, followed by a one-window BRITS/SAITS/CSDI CPU smoke with finite output.

- [ ] **Step 5: Commit**

```bash
git add validation_v2/modern/pypots_worker.py requirements-modern-pypots.txt tests/validation_v2/modern/test_pypots_worker.py
git commit -m "feat(modern): integrate PyPOTS imputers"
```

### Task 7: Exact-source SSSD-S4 worker and minimal compatibility patch

**Files:**
- Create: `validation_v2/modern/sssd_worker.py`
- Create: `requirements-modern-sssd.txt`
- Create: `third_party/sssd/manifest.json`
- Create: `third_party/sssd/LICENSE`
- Create: `third_party/sssd/compatibility.patch`
- Test: `tests/validation_v2/modern/test_sssd_worker.py`

- [ ] **Step 1: Write failing architecture, loss-mask, and sampling tests with an injected denoiser**

```python
import torch

from validation_v2.modern.sssd_worker import diffusion_loss, sssd_parameters


def test_sssd_parameters_match_preregistered_architecture():
    params = sssd_parameters(residual_width=32)
    assert params["T"] == 200
    assert params["num_res_layers"] == 36
    assert params["s4_lmax"] == 30
    assert params["s4_d_state"] == 64
    assert params["res_channels"] == params["skip_channels"] == 32


def test_diffusion_loss_scores_only_artificially_missing_values():
    target = torch.tensor([[[1.0, 2.0, 3.0]]])
    mask = torch.tensor([[[1.0, 0.0, 1.0]]])
    predicted_noise = torch.tensor([[[100.0, 0.0, 100.0]]])
    true_noise = torch.zeros_like(predicted_noise)
    assert diffusion_loss(predicted_noise, true_noise, mask).item() == 0.0
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/validation_v2/modern/test_sssd_worker.py -q`

Expected: import fails because `sssd_worker.py` does not exist.

- [ ] **Step 3: Pin upstream and implement the worker around the official denoiser**

The third-party manifest must contain:

```json
{
  "name": "AI4HealthUOL/SSSD",
  "url": "https://github.com/AI4HealthUOL/SSSD.git",
  "commit": "4d3b7a51c54b658945c0ba0bbb26e5ee1f763bed",
  "license": "MIT",
  "source_subdirectory": "src"
}
```

`sssd_worker.py` adds the packaged snapshot's `src` directory to `sys.path`, imports `SSSDS4Imputer`, and constructs it with 6 input/output channels, 36 residual layers, residual/skip width 32 or 64, diffusion embeddings 128/512/512, `s4_lmax=30`, `s4_d_state=64`, dropout 0, bidirectional S4, and layer norm.

Implement the official DDPM schedule `T=200`, `beta_0=1e-4`, `beta_T=0.02`; train noise prediction only at artificial missing positions. Conditioning contains observed normalized values and mask. The reverse sampler clamps observed values after every step and returns `(windows, samples, 30, 6)`. Use Adam at the registered learning rate, validation missing-RMSE, max 100 epochs, and patience 20. Save only the best validation state with SHA-256 metadata.

The compatibility patch may only change import paths, deprecated PyTorch APIs, and CUDA extension build flags. It must not change tensor equations, architecture, diffusion schedule, loss support, or sampling equations. Record pre/post hashes in the package manifest.

Expose the same `train`, `predict`, and `preflight` subcommands as the PyPOTS worker.

- [ ] **Step 4: Verify the contract locally and the real runtime on Linux**

Run locally: `python -m pytest tests/validation_v2/modern/test_sssd_worker.py -q`

Expected: injected-model contract tests pass without compiling CUDA.

Run on MatPool during `prepare`:

```bash
.venv-modern-sssd/bin/python -m validation_v2.modern.sssd_worker preflight \
  --source third_party/sssd/source --output .modern-campaign/sssd-environment.json
```

Expected: one real forward/backward step, checkpoint reload, and two non-identical finite samples that preserve observations.

- [ ] **Step 5: Commit**

```bash
git add validation_v2/modern/sssd_worker.py requirements-modern-sssd.txt third_party/sssd tests/validation_v2/modern/test_sssd_worker.py
git commit -m "feat(modern): integrate official SSSD-S4"
```

## Milestone 3: Campaign execution, evaluation, and transport

### Task 8: Atomic task state machine and safe resume

**Files:**
- Create: `validation_v2/modern/campaign.py`
- Test: `tests/validation_v2/modern/test_campaign.py`

- [ ] **Step 1: Write failing state and resume tests**

```python
from pathlib import Path

import pytest

from validation_v2.modern.campaign import claim_task, complete_task, pending_tasks


def test_completed_hash_matching_task_is_not_pending(tmp_path: Path):
    task = {"task_id": "a" * 64, "model": "brits", "seed": 2026}
    claim_task(tmp_path, task)
    complete_task(tmp_path, task, {"checkpoint_sha256": "b" * 64})
    assert pending_tasks(tmp_path, [task]) == ()


def test_inconsistent_completed_task_is_rejected(tmp_path: Path):
    task = {"task_id": "a" * 64, "model": "brits", "seed": 2026}
    claim_task(tmp_path, task)
    complete_task(tmp_path, task, {"checkpoint_sha256": "b" * 64})
    (tmp_path / task["task_id"] / "completed.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent completed task"):
        pending_tasks(tmp_path, [task])
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/validation_v2/modern/test_campaign.py -q`

Expected: import fails because `campaign.py` does not exist.

- [ ] **Step 3: Implement immutable tasks and subprocess dispatch**

Task identity is SHA-256 of canonical JSON containing phase, model, seed, configuration ID, dataset artifact ID, checkpoint input hash, sampling count, and condition list. States are `claimed`, `running`, `failed`, or `completed`. Publish JSON atomically with no clobber. A completed task is reusable only if every declared output exists and matches its hash.

Implement the exact public functions `build_tasks(config: ModernConfig, dataset_manifest:
Mapping[str, object], selection_lock: Mapping[str, object]) -> tuple[dict[str, object],
...]`, `claim_task(root: Path, task: Mapping[str, object]) -> Path`, `run_task(task_dir:
Path, command: Sequence[str], *, environment: Mapping[str, str]) -> None`,
`complete_task(root: Path, task: Mapping[str, object], outputs: Mapping[str, object]) ->
None`, and `pending_tasks(root: Path, tasks: Sequence[Mapping[str, object]]) ->
tuple[dict[str, object], ...]`.

On nonzero subprocess exit, write `failed.json` with return code and stderr/log paths; preserve all artifacts. Resume runs only tasks without a verified `completed.json`. Refuse two active claims for the same task ID.

- [ ] **Step 4: Run campaign tests**

Run: `python -m pytest tests/validation_v2/modern/test_campaign.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add validation_v2/modern/campaign.py tests/validation_v2/modern/test_campaign.py
git commit -m "feat(modern): add resumable campaign tasks"
```

### Task 9: Unified prediction evaluation and nine-model summary

**Files:**
- Create: `validation_v2/modern/evaluate.py`
- Create: `validation_v2/modern/summarize.py`
- Test: `tests/validation_v2/modern/test_evaluate_summarize.py`

- [ ] **Step 1: Write failing observed-preservation and completeness tests**

```python
import numpy as np
import pandas as pd
import pytest

from validation_v2.modern.evaluate import complete_samples
from validation_v2.modern.summarize import validate_stage_a_coverage


def test_complete_samples_preserves_observed_values():
    observed = np.array([[1.0], [np.nan], [3.0]], dtype=np.float32)
    mask = np.array([[1], [0], [1]], dtype=np.uint8)
    samples = np.full((2, 3, 1), 9.0, dtype=np.float32)
    result = complete_samples(observed, mask, samples)
    np.testing.assert_array_equal(result[:, [0, 2], 0], [[1, 3], [1, 3]])


def test_summary_rejects_missing_seed():
    frame = pd.DataFrame({"model": ["hybrid"], "seed": [2026], "condition_id": ["point-0.1"]})
    with pytest.raises(ValueError, match="incomplete stage A coverage"):
        validate_stage_a_coverage(frame, expected_recordings=("rec-a",))
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/validation_v2/modern/test_evaluate_summarize.py -q`

Expected: imports fail because evaluation modules do not exist.

- [ ] **Step 3: Implement evaluation and strict coverage validation**

For deterministic prediction artifacts, use the only sample. For CSDI/SSSD, require exactly 50 samples in formal mode. Stitch samples, preserve observations, compute the sample mean, then call existing `reconstruction_metrics` and trajectory diagnostic functions. Add empirical CRPS, coverage, and width rows only for CSDI/SSSD. Every row stores model, seed, recording, condition, metric, value, checkpoint SHA, dataset artifact ID, and prediction artifact ID.

`validate_stage_a_coverage(frame: pd.DataFrame, *, expected_recordings: Sequence[str])`
requires the Cartesian product of nine models, five seeds, 13 condition IDs, and every
explicitly supplied test recording ID. Summaries use recordings as paired units, existing
bootstrap/effect-size helpers, and Holm correction. Emit:

```text
per_record_metrics.csv
summary.csv
summary.json
pairwise_effects.csv
probability_calibration.csv
runtime.csv
figures/*.png
validation_report.json
```

Set `validation_report.json.status` to `complete` only after all artifact hashes and summary cardinalities validate.

- [ ] **Step 4: Run evaluation tests and existing statistics tests**

Run: `python -m pytest tests/validation_v2/modern/test_evaluate_summarize.py tests/validation_v2/test_provenance_and_statistics.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add validation_v2/modern/evaluate.py validation_v2/modern/summarize.py tests/validation_v2/modern/test_evaluate_summarize.py
git commit -m "feat(modern): evaluate and summarize nine models"
```

### Task 10: Modern CLI and reference-model campaign integration

**Files:**
- Create: `validation_v2/modern/cli.py`
- Modify: `validation_v2/experiments/runner.py` supported orchestration entry points only
- Test: `tests/validation_v2/modern/test_smoke.py`

- [ ] **Step 1: Write a failing CLI smoke test with subprocess workers injected**

```python
from pathlib import Path

from validation_v2.modern.cli import main


def test_cli_plan_emits_reference_and_modern_tasks(tmp_path: Path):
    result = main(["plan", "--config", "configs/validation_v2/modern_smoke.yaml",
                   "--output", str(tmp_path)])
    assert result == 0
    text = (tmp_path / "campaign-plan.json").read_text(encoding="utf-8")
    assert '"hybrid"' in text and '"brits"' in text and '"sssd"' in text
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/validation_v2/modern/test_smoke.py -q`

Expected: import fails because `cli.py` does not exist.

- [ ] **Step 3: Implement CLI phases and reference dispatch**

Subcommands:

```text
plan --config CONFIG --output ROOT
export --config CONFIG --output ROOT
tune --config CONFIG --output ROOT --pypots-python PATH --sssd-python PATH
run --config CONFIG --output ROOT --pypots-python PATH --sssd-python PATH
resume --config CONFIG --output ROOT --pypots-python PATH --sssd-python PATH
validate --config CONFIG --output ROOT
summarize --config CONFIG --output ROOT
package-results --config CONFIG --output ROOT --mode summary|full
```

Reference models are run by constructing an in-memory V2 config restricted to `strict_file`, the five approved reference models, and the same ordered condition list. Call existing V2 grouped execution rather than reimplementing their training. Modern workers consume exported artifacts. The plan contains 45 formal train tasks; smoke contains all nine model names but permits injected lightweight worker commands in tests.

- [ ] **Step 4: Run smoke-plan and all modern unit tests**

Run: `python -m pytest tests/validation_v2/modern -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add validation_v2/modern/cli.py validation_v2/experiments/runner.py tests/validation_v2/modern/test_smoke.py
git commit -m "feat(modern): orchestrate reference and modern models"
```

### Task 11: MatPool dual-environment launcher

**Files:**
- Create: `scripts/run_modern_imputation_matpool.sh`
- Test: `tests/validation_v2/modern/test_matpool_launcher.py`

- [ ] **Step 1: Write failing help, state, and safety tests**

```python
import subprocess


def test_launcher_help_lists_complete_lifecycle():
    result = subprocess.run(["bash", "scripts/run_modern_imputation_matpool.sh", "--help"],
                            text=True, capture_output=True, check=True)
    for command in ("prepare", "start", "status", "logs", "resume", "package-results"):
        assert command in result.stdout
```

Add tests that run the launcher against temporary fake Python executables and assert: dirty worktrees are rejected before environment writes; an active session blocks a second start; completed state is not overwritten; resume uses the recorded exact commit and campaign root.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/validation_v2/modern/test_matpool_launcher.py -q`

Expected: failure because the launcher does not exist.

- [ ] **Step 3: Implement the launcher by adapting existing hardened MatPool patterns**

Use `set -Eeuo pipefail`, exact 40-character commit checks, clean worktree checks, non-symlink private state directory, atomic `current.json`, tmux status validation, exit evidence, and no destructive stop command.

`prepare` must:

1. verify Linux, RTX 4090 name, at least 23 GiB GPU memory, driver, disk, and dataset hash;
2. create `.venv-modern-pypots` with Python 3.10, torch 2.3.1+cu121, V2 requirements, and PyPOTS lock;
3. create `.venv-modern-sssd` with Python 3.10, torch 1.13.0 and the minimal SSSD lock;
4. verify the packaged SSSD commit/source hashes and apply the recorded compatibility patch to a campaign-private copy;
5. run the full V2 test suite, modern tests, PyPOTS preflight, and real SSSD preflight;
6. write environment JSON and a sealed preparation marker.

`start` runs plan/export/tune/formal/evaluate/summarize/validate in one tmux session with one GPU worker. `resume` uses the same sealed campaign and invokes only pending tasks. `package-results` is allowed only after complete validation, except an explicitly named diagnostic mode internal to failure handling.

- [ ] **Step 4: Run launcher tests under Bash**

Run: `python -m pytest tests/validation_v2/modern/test_matpool_launcher.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_modern_imputation_matpool.sh tests/validation_v2/modern/test_matpool_launcher.py
git commit -m "feat(modern): add MatPool campaign launcher"
```

### Task 12: Windows upload bundle and result bundles

**Files:**
- Create: `scripts/package_modern_experiment.ps1`
- Modify: `validation_v2/modern/cli.py` result packaging implementation
- Test: `tests/validation_v2/modern/test_packaging.py`

- [ ] **Step 1: Write failing archive inclusion/exclusion tests**

```python
from pathlib import Path

from validation_v2.modern.cli import package_result_tree


def test_summary_excludes_checkpoints_and_samples(tmp_path: Path):
    source = tmp_path / "campaign"
    (source / "summary").mkdir(parents=True)
    (source / "summary" / "summary.csv").write_text("x\n", encoding="utf-8")
    (source / "checkpoints").mkdir()
    (source / "checkpoints" / "model.pt").write_bytes(b"weights")
    (source / "samples").mkdir()
    (source / "samples" / "samples.npz").write_bytes(b"samples")
    manifest = package_result_tree(source, tmp_path / "result", mode="summary")
    assert "summary/summary.csv" in manifest["files"]
    assert all("checkpoints/" not in name and "samples/" not in name for name in manifest["files"])
```

Add a PowerShell integration test that invokes the script in a temporary clean Git repository and verifies the upload tar contains a Git bundle/bootstrap, configs, scripts, third-party manifest/license/source snapshot, and optionally `Oxford Dataset`, while excluding `results`, `.worktrees`, virtual environments, and caches.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/validation_v2/modern/test_packaging.py -q`

Expected: failure because packaging functions and the PowerShell script do not exist.

- [ ] **Step 3: Implement deterministic package manifests**

The PowerShell script must resolve the repository with `Resolve-Path`, verify the target lies under an explicitly created staging directory, require a clean exact commit, create a Git bundle, archive the exact SSSD snapshot from its pinned commit, copy the dataset only with `-IncludeData`, generate `bootstrap.sh`, create `modern-imputation-upload-<commit>.tar.gz`, and write `<archive>.sha256`. It must never recursively delete a computed path without first checking the resolved path remains inside its newly created staging parent.

`package_result_tree` creates a canonical file manifest with per-file SHA-256. Summary includes configs, metrics, figures, logs, environments, manifests, tuning lock, and checkpoint identities. Full additionally includes actual checkpoints and 50-sample prediction artifacts. Use `tarfile` with stable relative paths and reject symlinks.

- [ ] **Step 4: Run packaging tests**

Run: `python -m pytest tests/validation_v2/modern/test_packaging.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/package_modern_experiment.ps1 validation_v2/modern/cli.py tests/validation_v2/modern/test_packaging.py
git commit -m "feat(modern): package uploads and results"
```

### Task 13: Chinese runbook, full regression, and local smoke artifact

**Files:**
- Create: `docs/modern_imputation_matpool_runbook_zh.md`
- Modify: `README.md` add one link to the modern runbook
- Test: `tests/validation_v2/modern/test_smoke.py`

- [ ] **Step 1: Extend the smoke test to require a completed nine-model manifest**

The test uses real bounded OxIOD records, one epoch, four windows, one condition, fake SSSD denoiser injection on Windows, and the real local PyPOTS environment when available. Assert the smoke report lists all nine models, uses one seed, keeps observed values, records two diffusion samples, and contains no formal-complete claim.

- [ ] **Step 2: Run it and verify RED for the missing end-to-end wiring**

Run: `python -m pytest tests/validation_v2/modern/test_smoke.py -q`

Expected: the new completed-manifest assertion fails until all phase wiring is connected.

- [ ] **Step 3: Connect the smoke path and write the exact operator runbook**

The runbook must include:

```powershell
.\scripts\package_modern_experiment.ps1 -IncludeData
```

and, after upload/extract/bootstrap:

```bash
bash scripts/run_modern_imputation_matpool.sh prepare
bash scripts/run_modern_imputation_matpool.sh start
bash scripts/run_modern_imputation_matpool.sh status
bash scripts/run_modern_imputation_matpool.sh logs
bash scripts/run_modern_imputation_matpool.sh resume
bash scripts/run_modern_imputation_matpool.sh package-results summary
bash scripts/run_modern_imputation_matpool.sh package-results full
sha256sum -c <download-package>.sha256
```

Explain paid-resource behavior, tmux persistence, completion criteria, failure evidence, exact output paths, summary/full contents, and local offline re-summarization. Add the runbook link to README.

- [ ] **Step 4: Run complete verification**

Run:

```powershell
python -m pytest tests/validation_v2 tests/validation_v2/modern -q
python -m validation_v2.modern.cli plan --config configs/validation_v2/modern_stage_a.yaml --output .modern-plan-check
python -m validation_v2.modern.cli validate --config configs/validation_v2/modern_smoke.yaml --output results/validation_v2/modern-smoke
```

Expected: all tests pass; the formal plan reports 45 train tasks and 13 conditions; smoke validation reports `status=completed` but `scope=smoke`.

Run static checks:

```powershell
git diff --check
rg -n "TBD|TODO|FIXME|placeholder" validation_v2/modern scripts/run_modern_imputation_matpool.sh scripts/package_modern_experiment.ps1 docs/modern_imputation_matpool_runbook_zh.md
```

Expected: no whitespace errors and no placeholder matches.

- [ ] **Step 5: Commit**

```bash
git add docs/modern_imputation_matpool_runbook_zh.md README.md tests/validation_v2/modern/test_smoke.py
git commit -m "docs(modern): add MatPool experiment runbook"
```

## Final implementation audit

- [ ] Confirm every production function added by Tasks 1-13 was introduced after a test failed for the expected missing behavior.
- [ ] Confirm `git status --short` contains only known user-owned pre-existing changes outside this feature.
- [ ] Confirm existing `validation_v2` results and legacy scripts were not used as evidence or overwritten.
- [ ] Confirm the current Hybrid is present in the formal plan and is retrained under the same split, scaler, masks, seeds, and 13 evaluation conditions.
- [ ] Confirm the upload bundle includes exact SSSD source commit `4d3b7a51c54b658945c0ba0bbb26e5ee1f763bed` and no server-time floating source download.
- [ ] Confirm the formal probability contract requires exactly 50 complete-record samples for CSDI and SSSD before summary completion.
- [ ] Confirm MatPool `prepare` is the only remaining platform-specific acceptance gate and that formal training cannot start unless it passes.
