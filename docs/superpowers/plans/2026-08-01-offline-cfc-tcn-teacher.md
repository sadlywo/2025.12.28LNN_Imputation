# Offline CfC–TCN Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and formally evaluate the leakage-safe offline bidirectional CfC–TCN teacher, stopping before student distillation unless it significantly beats the strongest eligible baseline.

**Architecture:** A 31-D observed-only feature contract feeds a bidirectional CfC encoder and a symmetric depthwise TCN. Their representations are fused with a timestamp-aware interpolation baseline, and separate gyro/accelerometer heads predict one residual that fills only missing entries. Validation v2 remains unchanged; v3 reuses only its already-tested split, scaler, mask, checkpoint, provenance, and statistical contracts.

**Tech Stack:** Python 3.10, PyTorch 2.3.1, ncps 1.0.1, NumPy 1.26.4, SciPy 1.13.1, pandas 2.3.3, PyYAML 6.0.3, pytest 8.4.2, PyPOTS 1.5.0 for BRITS/SAITS/CSDI.

---

## Scope and file map

This plan implements only the first stage of the approved design: offline teacher accuracy and its formal comparison. Fixed-lag students and Jetson deployment get separate plans only after the teacher success gate passes.

New files are grouped by responsibility:

```text
imputation_v3/
  __init__.py                 package identity
  config.py                   typed teacher experiment configuration
  types.py                    immutable feature/window/model component records
  data/features.py            31-D observed-only feature builder
  data/windows.py             deterministic shared window/mask materialization
  models/baselines.py         timestamp interpolation, completion, LOCF, PCHIP, RTS
  models/tcn.py               symmetric depthwise temporal encoder
  models/cfc.py               bidirectional CfC representation encoder and dt ablations
  models/teacher.py           feature fusion and residual heads
  models/native_controls.py   BiLSTM, BiCfC, TCN-only equal-input controls
  objectives/reconstruction.py channel-balanced missing-only losses
  experiments/training.py     teacher callbacks and checkpoint orchestration
  experiments/pypots.py       BRITS/SAITS/CSDI adapters
  experiments/evaluate.py     full-window aggregation and physical metrics
  experiments/runner.py       smoke/formal matrix orchestration
  cli.py                      command entry point
configs/imputation_v3/
  teacher_smoke.yaml
  teacher_full.yaml
tests/imputation_v3/
  test_config.py
  test_features.py
  test_windows.py
  test_baselines.py
  test_tcn.py
  test_cfc.py
  test_teacher.py
  test_objective.py
  test_training.py
  test_pypots.py
  test_runner.py
requirements-imputation-v3.txt
requirements-imputation-v3-baselines.txt
docs/imputation_v3_teacher_runbook.md
```

### Task 1: Package, dependencies, and typed configuration

**Files:**
- Create: `imputation_v3/__init__.py`
- Create: `imputation_v3/config.py`
- Create: `requirements-imputation-v3.txt`
- Create: `requirements-imputation-v3-baselines.txt`
- Create: `configs/imputation_v3/teacher_smoke.yaml`
- Create: `configs/imputation_v3/teacher_full.yaml`
- Test: `tests/imputation_v3/test_config.py`

- [ ] **Step 1: Write the failing configuration tests**

```python
from pathlib import Path

import pytest

from imputation_v3.config import load_teacher_config


def test_teacher_config_freezes_validation_only_selection(tmp_path: Path):
    path = tmp_path / "teacher.yaml"
    path.write_text(
        """
data_root: Oxford Dataset
output_root: results/imputation_v3/smoke
selection_split: validation
seeds: [2026]
window_seconds: [1.28, 2.56, 5.12]
nominal_dt_s: 0.01
batch_size: 2
epochs: 1
hidden_size: 64
tcn_width: 48
tcn_dilations: [1, 2, 4]
learning_rate: 0.001
training_rates: [0.1, 0.2, 0.3, 0.4]
training_topologies: [point, block, channel]
models: [linear, teacher]
""".strip(),
        encoding="utf-8",
    )

    config = load_teacher_config(path)

    assert config.selection_split == "validation"
    assert config.window_samples == (128, 256, 512)
    assert config.models == ("linear", "teacher")


def test_teacher_config_rejects_test_selection(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("selection_split: test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="selection_split"):
        load_teacher_config(path)
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run: `python -m pytest tests/imputation_v3/test_config.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'imputation_v3'`.

- [ ] **Step 3: Implement the package and strict config loader**

```python
# imputation_v3/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


ALLOWED_MODELS = frozenset(
    {"locf", "linear", "pchip", "rts", "bilstm", "bilnn", "tcn", "feature_mlp", "teacher", "brits", "saits", "csdi"}
)


@dataclass(frozen=True)
class TeacherConfig:
    data_root: Path
    output_root: Path
    selection_split: str
    seeds: tuple[int, ...]
    window_seconds: tuple[float, ...]
    nominal_dt_s: float
    batch_size: int
    epochs: int
    hidden_size: int
    tcn_width: int
    tcn_dilations: tuple[int, ...]
    learning_rate: float
    training_rates: tuple[float, ...]
    training_topologies: tuple[str, ...]
    models: tuple[str, ...]

    @property
    def window_samples(self) -> tuple[int, ...]:
        return tuple(round(seconds / self.nominal_dt_s) for seconds in self.window_seconds)


def _positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def load_teacher_config(path: Path) -> TeacherConfig:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    data: Mapping[str, Any] = loaded if isinstance(loaded, Mapping) else {}
    if data.get("selection_split") != "validation":
        raise ValueError("selection_split must be validation")
    models = tuple(str(value) for value in data["models"])
    unknown = sorted(set(models) - ALLOWED_MODELS)
    if unknown:
        raise ValueError(f"unsupported models: {unknown}")
    seeds = tuple(int(value) for value in data["seeds"])
    if not seeds:
        raise ValueError("seeds must not be empty")
    return TeacherConfig(
        data_root=Path(data["data_root"]),
        output_root=Path(data["output_root"]),
        selection_split="validation",
        seeds=seeds,
        window_seconds=tuple(_positive("window_seconds", float(value)) for value in data["window_seconds"]),
        nominal_dt_s=_positive("nominal_dt_s", float(data["nominal_dt_s"])),
        batch_size=int(_positive("batch_size", int(data["batch_size"]))),
        epochs=int(_positive("epochs", int(data["epochs"]))),
        hidden_size=int(_positive("hidden_size", int(data["hidden_size"]))),
        tcn_width=int(_positive("tcn_width", int(data["tcn_width"]))),
        tcn_dilations=tuple(int(_positive("tcn_dilations", int(value))) for value in data["tcn_dilations"]),
        learning_rate=_positive("learning_rate", float(data["learning_rate"])),
        training_rates=tuple(float(value) for value in data["training_rates"]),
        training_topologies=tuple(str(value) for value in data["training_topologies"]),
        models=models,
    )
```

Use these exact dependency and configuration files:

```text
# requirements-imputation-v3.txt
-r requirements-validation-v2.txt
scipy==1.13.1
```

```text
# requirements-imputation-v3-baselines.txt
-r requirements-imputation-v3.txt
pypots==1.5.0
```

```yaml
# configs/imputation_v3/teacher_smoke.yaml
data_root: Oxford Dataset
output_root: results/imputation_v3/smoke
selection_split: validation
seeds: [2026]
window_seconds: [1.28]
nominal_dt_s: 0.01
batch_size: 2
epochs: 1
hidden_size: 16
tcn_width: 16
tcn_dilations: [1, 2]
learning_rate: 0.001
training_rates: [0.2]
training_topologies: [point, block]
models: [linear, teacher]
```

```yaml
# configs/imputation_v3/teacher_full.yaml
data_root: Oxford Dataset
output_root: results/imputation_v3/formal
selection_split: validation
seeds: [2026, 2027, 2028, 2029, 2030]
window_seconds: [1.28, 2.56, 5.12]
nominal_dt_s: 0.01
batch_size: 64
epochs: 100
hidden_size: 64
tcn_width: 48
tcn_dilations: [1, 2, 4, 8, 16]
learning_rate: 0.001
training_rates: [0.1, 0.2, 0.3, 0.4]
training_topologies: [point, block, channel]
models: [locf, linear, pchip, rts, bilstm, bilnn, tcn, feature_mlp, teacher, brits, saits, csdi]
```

- [ ] **Step 4: Run configuration tests**

Run: `python -m pytest tests/imputation_v3/test_config.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit the configuration contract**

```bash
git add imputation_v3/__init__.py imputation_v3/config.py requirements-imputation-v3.txt requirements-imputation-v3-baselines.txt configs/imputation_v3 tests/imputation_v3/test_config.py
git commit -m "feat(imputation-v3): add teacher configuration contract"
```

### Task 2: Immutable 31-D observed-only features

**Files:**
- Create: `imputation_v3/types.py`
- Create: `imputation_v3/data/__init__.py`
- Create: `imputation_v3/data/features.py`
- Test: `tests/imputation_v3/test_features.py`

- [ ] **Step 1: Write leakage and column-order tests**

```python
import torch

from imputation_v3.data.features import build_features


def test_hidden_targets_cannot_change_v3_features():
    target_a = torch.arange(60, dtype=torch.float32).reshape(10, 6)
    target_b = target_a.clone()
    mask = torch.ones_like(target_a)
    mask[3:7, 2:5] = 0
    target_b[mask == 0] = 50_000
    dt = torch.full((10,), 0.01)
    torch.testing.assert_close(
        build_features(target_a, mask, dt).values,
        build_features(target_b, mask, dt).values,
        rtol=0,
        atol=0,
    )


def test_v3_feature_order_age_and_carried_slope():
    target = torch.tensor([[1.0] * 6, [2.0] * 6, [99.0] * 6, [4.0] * 6])
    mask = torch.ones_like(target)
    mask[2] = 0
    batch = build_features(target, mask, torch.full((4,), 0.1))
    assert batch.values.shape == (4, 31)
    torch.testing.assert_close(batch.values[2, 13:19], torch.full((6,), 0.1))
    torch.testing.assert_close(batch.values[2, 19:25], torch.full((6,), 10.0))
    torch.testing.assert_close(batch.values[2, 25:31], torch.ones(6))


def test_feature_batch_is_clone_on_read():
    target = torch.ones(4, 6)
    mask = torch.ones_like(target)
    batch = build_features(target, mask, torch.full((4,), 0.1))
    leaked = batch.values
    leaked.zero_()
    assert torch.count_nonzero(batch.values) > 0
```

- [ ] **Step 2: Run the feature tests and verify failure**

Run: `python -m pytest tests/imputation_v3/test_features.py -v`  
Expected: FAIL because `build_features` does not exist.

- [ ] **Step 3: Implement immutable feature state and causal age/slope computation**

```python
# imputation_v3/data/features.py
import torch

from imputation_v3.types import FeatureBatch


def build_features(target: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor) -> FeatureBatch:
    if target.ndim != 2 or target.shape[1] != 6 or mask.shape != target.shape:
        raise ValueError("target and mask must have shape (samples, 6)")
    if dt.shape != (target.shape[0],) or not dt.is_floating_point():
        raise ValueError("dt must be floating point with shape (samples,)")
    mask_values = mask.to(dtype=target.dtype)
    if not torch.all((mask_values == 0) | (mask_values == 1)):
        raise ValueError("mask values must be 0 or 1")
    if not torch.isfinite(dt).all() or not torch.all(dt > 0):
        raise ValueError("dt must be finite and positive")
    observed = torch.where(mask_values.bool(), target, torch.zeros_like(target))
    if not torch.isfinite(observed).all():
        raise ValueError("observed values must be finite")
    time = torch.cumsum(dt, dim=0) - dt[0]
    age = torch.zeros_like(target)
    slope = torch.zeros_like(target)
    slope_valid = torch.zeros_like(target)
    for channel in range(6):
        last_time = None
        last_value = None
        latest_slope = target.new_tensor(0.0)
        latest_valid = target.new_tensor(0.0)
        for index in range(target.shape[0]):
            if mask_values[index, channel] == 1:
                if last_time is not None:
                    latest_slope = (observed[index, channel] - last_value) / (time[index] - last_time)
                    latest_valid = target.new_tensor(1.0)
                last_time = time[index]
                last_value = observed[index, channel]
            elif last_time is not None:
                age[index, channel] = time[index] - last_time
            slope[index, channel] = latest_slope
            slope_valid[index, channel] = latest_valid
    values = torch.cat((observed, mask_values, dt[:, None], age, slope, slope_valid), dim=1)
    return FeatureBatch(values=values, dt=dt, mask=mask_values)
```

Add this exact clone-on-read type to `imputation_v3/types.py`:

```python
from dataclasses import dataclass, field

import torch


def _copy(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


@dataclass(frozen=True, init=False, repr=False, eq=False)
class FeatureBatch:
    _values: torch.Tensor = field(repr=False)
    _dt: torch.Tensor = field(repr=False)
    _mask: torch.Tensor = field(repr=False)

    def __init__(self, values: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor) -> None:
        object.__setattr__(self, "_values", _copy(values))
        object.__setattr__(self, "_dt", _copy(dt))
        object.__setattr__(self, "_mask", _copy(mask))

    @property
    def values(self) -> torch.Tensor:
        return _copy(self._values)

    @property
    def dt(self) -> torch.Tensor:
        return _copy(self._dt)

    @property
    def mask(self) -> torch.Tensor:
        return _copy(self._mask)
```

- [ ] **Step 4: Run v3 feature tests and v2 regression tests**

Run: `python -m pytest tests/imputation_v3/test_features.py tests/validation_v2/test_masking_and_features.py -v`  
Expected: PASS; v2 remains unchanged.

- [ ] **Step 5: Commit the feature contract**

```bash
git add imputation_v3/types.py imputation_v3/data tests/imputation_v3/test_features.py
git commit -m "feat(imputation-v3): add observed-only temporal features"
```

### Task 3: Full-window baselines and observed-value completion

**Files:**
- Create: `imputation_v3/models/__init__.py`
- Create: `imputation_v3/models/baselines.py`
- Test: `tests/imputation_v3/test_baselines.py`

- [ ] **Step 1: Write timestamp interpolation and preservation tests**

```python
import torch

from imputation_v3.models.baselines import (
    constant_velocity_rts,
    complete_signal,
    timestamp_linear,
    timestamp_locf,
    timestamp_pchip,
)


def test_timestamp_linear_uses_real_time_and_ignores_hidden_values():
    target = torch.tensor([[0.0], [999.0], [4.0]])
    mask = torch.tensor([[1.0], [0.0], [1.0]])
    time = torch.tensor([0.0, 0.25, 1.0])
    first = timestamp_linear(target, mask, time, empty_fill=0.0)
    changed = target.clone()
    changed[1] = -999.0
    second = timestamp_linear(changed, mask, time, empty_fill=0.0)
    assert first[1, 0].item() == 1.0
    torch.testing.assert_close(first, second)


def test_completion_preserves_observed_values_exactly():
    observed = torch.tensor([[1.0, float("nan")]])
    mask = torch.tensor([[1.0, 0.0]])
    completed = complete_signal(observed, mask, torch.tensor([[9.0, 3.0]]))
    torch.testing.assert_close(completed, torch.tensor([[1.0, 3.0]]))


def test_all_full_window_baselines_ignore_hidden_values_and_preserve_observations():
    source = torch.tensor([[0.0], [999.0], [2.0], [-999.0], [4.0]])
    changed = source.clone()
    mask = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0]])
    changed[mask == 0] *= -10
    time = torch.tensor([0.0, 0.1, 0.3, 0.7, 1.0])
    functions = (
        timestamp_locf,
        timestamp_linear,
        timestamp_pchip,
        lambda value, valid, stamps, empty_fill: constant_velocity_rts(
            value, valid, stamps, empty_fill=empty_fill,
            process_var=0.1, observation_var=0.01,
        ),
    )
    for function in functions:
        first = function(source, mask, time, empty_fill=0.0)
        second = function(changed, mask, time, empty_fill=0.0)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first[mask.bool()], source[mask.bool()])
```

- [ ] **Step 2: Run baseline tests and verify failure**

Run: `python -m pytest tests/imputation_v3/test_baselines.py -v`  
Expected: FAIL because the baseline module does not exist.

- [ ] **Step 3: Implement linear interpolation and completion**

```python
# imputation_v3/models/baselines.py
import torch


def complete_signal(observed: torch.Tensor, mask: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    if observed.shape != mask.shape or observed.shape != prediction.shape:
        raise ValueError("observed, mask, and prediction must have identical shapes")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must contain 0 and 1 only")
    return torch.where(mask.bool(), observed, prediction)


def timestamp_linear(
    source: torch.Tensor,
    mask: torch.Tensor,
    time: torch.Tensor,
    *,
    empty_fill: float,
) -> torch.Tensor:
    if source.ndim != 2 or mask.shape != source.shape or time.shape != (source.shape[0],):
        raise ValueError("source/mask/time shapes are inconsistent")
    if not torch.isfinite(time).all() or not torch.all(time[1:] > time[:-1]):
        raise ValueError("time must be finite and strictly increasing")
    output = torch.empty_like(source)
    for channel in range(source.shape[1]):
        indices = torch.where(mask[:, channel].bool())[0]
        if indices.numel() == 0:
            output[:, channel] = empty_fill
            continue
        observed_time = time[indices]
        observed_value = source[indices, channel]
        positions = torch.searchsorted(observed_time, time).clamp(max=indices.numel() - 1)
        right = positions
        left = (positions - 1).clamp(min=0)
        denominator = observed_time[right] - observed_time[left]
        fraction = torch.where(denominator > 0, (time - observed_time[left]) / denominator, 0.0)
        output[:, channel] = observed_value[left] + fraction * (observed_value[right] - observed_value[left])
    return complete_signal(source, mask, output)
```

Append these implementations to `imputation_v3/models/baselines.py`:

```python
def timestamp_locf(source, mask, time, *, empty_fill):
    if source.ndim != 2 or mask.shape != source.shape or time.shape != (source.shape[0],):
        raise ValueError("source/mask/time shapes are inconsistent")
    output = torch.full_like(source, empty_fill)
    for channel in range(source.shape[1]):
        carried = source.new_tensor(empty_fill)
        for index in range(source.shape[0]):
            if bool(mask[index, channel]):
                carried = source[index, channel]
            output[index, channel] = carried
    return complete_signal(source, mask, output)


def timestamp_pchip(source, mask, time, *, empty_fill):
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    output = torch.full_like(source, empty_fill)
    stamps = time.detach().cpu().numpy()
    for channel in range(source.shape[1]):
        indices = torch.where(mask[:, channel].bool())[0]
        if indices.numel() == 0:
            continue
        if indices.numel() == 1:
            output[:, channel] = source[indices[0], channel]
            continue
        x = time[indices].detach().cpu().numpy()
        y = source[indices, channel].detach().cpu().numpy()
        clipped = np.clip(stamps, x[0], x[-1])
        values = PchipInterpolator(x, y, extrapolate=False)(clipped)
        output[:, channel] = torch.as_tensor(values, dtype=source.dtype, device=source.device)
    return complete_signal(source, mask, output)


def constant_velocity_rts(
    source, mask, time, *, empty_fill, process_var, observation_var,
):
    if process_var <= 0 or observation_var <= 0:
        raise ValueError("RTS variances must be positive")
    output = torch.empty_like(source)
    identity = torch.eye(2, dtype=source.dtype, device=source.device)
    h = source.new_tensor([[1.0, 0.0]])
    for channel in range(source.shape[1]):
        observed_indices = torch.where(mask[:, channel].bool())[0]
        initial = source[observed_indices[0], channel] if observed_indices.numel() else source.new_tensor(empty_fill)
        state = torch.stack((initial, source.new_tensor(0.0)))
        covariance = identity * 10.0
        filtered_states, filtered_covariances = [], []
        predicted_states, predicted_covariances, transitions = [], [], []
        for index in range(source.shape[0]):
            delta = time[index] - time[index - 1] if index else time.new_tensor(1e-6)
            transition = torch.stack((
                torch.stack((time.new_tensor(1.0), delta)),
                time.new_tensor([0.0, 1.0]),
            ))
            q = process_var * torch.stack((
                torch.stack((delta ** 3 / 3, delta ** 2 / 2)),
                torch.stack((delta ** 2 / 2, delta)),
            ))
            predicted = transition @ state
            predicted_covariance = transition @ covariance @ transition.T + q
            if bool(mask[index, channel]):
                innovation = source[index, channel] - (h @ predicted)[0]
                innovation_covariance = (h @ predicted_covariance @ h.T)[0, 0] + observation_var
                gain = (predicted_covariance @ h.T)[:, 0] / innovation_covariance
                state = predicted + gain * innovation
                covariance = (identity - gain[:, None] @ h) @ predicted_covariance
            else:
                state, covariance = predicted, predicted_covariance
            predicted_states.append(predicted)
            predicted_covariances.append(predicted_covariance)
            transitions.append(transition)
            filtered_states.append(state)
            filtered_covariances.append(covariance)
        smoothed = list(filtered_states)
        smoothed_covariance = list(filtered_covariances)
        for index in range(source.shape[0] - 2, -1, -1):
            gain = torch.linalg.solve(
                predicted_covariances[index + 1].T,
                (filtered_covariances[index] @ transitions[index + 1].T).T,
            ).T
            smoothed[index] = filtered_states[index] + gain @ (smoothed[index + 1] - predicted_states[index + 1])
            smoothed_covariance[index] = filtered_covariances[index] + gain @ (
                smoothed_covariance[index + 1] - predicted_covariances[index + 1]
            ) @ gain.T
        output[:, channel] = torch.stack(smoothed)[:, 0]
    return complete_signal(source, mask, output)
```

The formal runner fixes `observation_var` from training residuals and selects `process_var` from `[1e-4, 1e-3, 1e-2, 1e-1, 1.0]` using validation RMSE only; test recordings never participate in this selection.

- [ ] **Step 4: Run baseline tests**

Run: `python -m pytest tests/imputation_v3/test_baselines.py tests/validation_v2/test_models.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit baseline primitives**

```bash
git add imputation_v3/models tests/imputation_v3/test_baselines.py
git commit -m "feat(imputation-v3): add full-window residual baselines"
```

### Task 4: Symmetric depthwise TCN encoder

**Files:**
- Create: `imputation_v3/models/tcn.py`
- Test: `tests/imputation_v3/test_tcn.py`

- [ ] **Step 1: Write shape, symmetry, and validation tests**

```python
import pytest
import torch

from imputation_v3.models.tcn import SymmetricTCNEncoder


def test_symmetric_tcn_returns_one_representation_per_time_step():
    model = SymmetricTCNEncoder(31, width=12, dilations=(1, 2, 4), dropout=0.0)
    output = model(torch.randn(2, 33, 31))
    assert output.shape == (2, 33, 12)
    assert torch.isfinite(output).all()


def test_symmetric_tcn_rejects_even_kernel():
    with pytest.raises(ValueError, match="odd"):
        SymmetricTCNEncoder(31, width=12, dilations=(1,), kernel_size=4)
```

- [ ] **Step 2: Run the TCN tests and verify failure**

Run: `python -m pytest tests/imputation_v3/test_tcn.py -v`  
Expected: FAIL because `SymmetricTCNEncoder` is missing.

- [ ] **Step 3: Implement depthwise residual blocks**

```python
# imputation_v3/models/tcn.py
import torch
from torch import nn


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(width, width, kernel_size, padding=padding, dilation=dilation, groups=width),
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.net(value.transpose(1, 2)).transpose(1, 2)
        return self.norm(value + residual)


class SymmetricTCNEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        *,
        width: int,
        dilations: tuple[int, ...],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd")
        self.input_size = input_size
        self.projection = nn.Linear(input_size, width)
        self.blocks = nn.ModuleList(
            DepthwiseResidualBlock(width, kernel_size, dilation, dropout) for dilation in dilations
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_size:
            raise ValueError("features must have shape (batch, time, input_size)")
        value = self.projection(features)
        for block in self.blocks:
            value = block(value)
        return value
```

- [ ] **Step 4: Run TCN tests**

Run: `python -m pytest tests/imputation_v3/test_tcn.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit the TCN encoder**

```bash
git add imputation_v3/models/tcn.py tests/imputation_v3/test_tcn.py
git commit -m "feat(imputation-v3): add symmetric temporal encoder"
```

### Task 5: Bidirectional CfC encoder and time-semantics controls

**Files:**
- Create: `imputation_v3/models/cfc.py`
- Test: `tests/imputation_v3/test_cfc.py`

- [ ] **Step 1: Write spy-based timespan tests**

```python
import torch
from torch import nn

from imputation_v3.models.cfc import BidirectionalCfCEncoder


class SpyCfC(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.timespans = None

    def forward(self, features, timespans):
        self.timespans = timespans.detach().clone()
        return features.new_zeros((*features.shape[:2], self.hidden_size)), None


def test_bidirectional_encoder_uses_direction_aligned_actual_timespans():
    spies = []

    def factory(input_size, hidden_size, **options):
        del input_size, options
        spy = SpyCfC(hidden_size)
        spies.append(spy)
        return spy

    model = BidirectionalCfCEncoder(31, 5, cfc_factory=factory)
    features = torch.randn(2, 4, 31)
    dt = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]])
    output = model(features, dt, mode="actual")
    assert output.shape == (2, 4, 10)
    torch.testing.assert_close(spies[0].timespans, dt)
    torch.testing.assert_close(spies[1].timespans, torch.cat((dt[:, -1:], dt[:, 1:].flip(1)), dim=1))


def test_time_modes_separate_timespan_and_dt_feature_evidence():
    model = BidirectionalCfCEncoder(31, 5, cfc_factory=lambda _i, h, **_k: SpyCfC(h))
    features = torch.randn(1, 4, 31)
    features[..., 12] = torch.tensor([[0.01, 0.02, 0.03, 0.04]])
    dt = features[..., 12].clone()
    captured = {}
    model.forward_cfc.register_forward_pre_hook(
        lambda _module, args, kwargs: captured.update(features=kwargs.get("features", args[0]).detach().clone()),
        with_kwargs=True,
    )
    model(features, dt, mode="constant", nominal_dt_s=0.01)
    torch.testing.assert_close(captured["features"][..., 12], torch.full_like(dt, 0.01))
    model(features, dt, mode="dt_feature_only", nominal_dt_s=0.01)
    torch.testing.assert_close(captured["features"][..., 12], dt)
    model(features, dt, mode="no_dt", nominal_dt_s=0.01)
    torch.testing.assert_close(captured["features"][..., 12], torch.zeros_like(dt))
```

- [ ] **Step 2: Run CfC tests and verify failure**

Run: `python -m pytest tests/imputation_v3/test_cfc.py -v`  
Expected: FAIL because the v3 encoder is missing.

- [ ] **Step 3: Implement representation-level bidirectional CfC**

```python
# imputation_v3/models/cfc.py
from collections.abc import Callable

import torch
from torch import nn


def _factory(*args, **kwargs):
    from ncps.torch import CfC
    return CfC(*args, **kwargs)


def reverse_aligned_dt(dt: torch.Tensor) -> torch.Tensor:
    return torch.cat((dt[:, -1:], dt[:, 1:].flip(1)), dim=1)


class BidirectionalCfCEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, *, cfc_factory: Callable | None = None) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._default = cfc_factory is None
        factory = cfc_factory or _factory
        options = {"batch_first": True, "return_sequences": True}
        self.forward_cfc = factory(input_size, hidden_size, **options)
        self.reverse_cfc = factory(input_size, hidden_size, **options)

    def _run(self, module: nn.Module, features: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        timespans = dt.unsqueeze(-1).expand(-1, -1, self.hidden_size) if self._default else dt
        result = module(features, timespans=timespans)
        return result[0] if isinstance(result, tuple) else result

    def forward(self, features: torch.Tensor, dt: torch.Tensor, *, mode: str, nominal_dt_s: float = 0.01) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_size:
            raise ValueError("invalid feature shape")
        if dt.shape != features.shape[:2] or not torch.isfinite(dt).all() or not torch.all(dt > 0):
            raise ValueError("dt must be finite, positive, and aligned")
        if mode not in {"actual", "constant", "dt_feature_only", "no_dt"}:
            raise ValueError("unsupported time mode")
        supplied_dt = dt if mode == "actual" else torch.full_like(dt, nominal_dt_s)
        supplied_features = features.clone()
        if mode == "constant":
            supplied_features[..., 12] = nominal_dt_s
        elif mode == "no_dt":
            supplied_features[..., 12] = 0
        forward = self._run(self.forward_cfc, supplied_features, supplied_dt)
        reverse_dt = reverse_aligned_dt(supplied_dt)
        reverse = self._run(self.reverse_cfc, supplied_features.flip(1), reverse_dt).flip(1)
        return torch.cat((forward, reverse), dim=-1)
```

- [ ] **Step 4: Run v3 and v2 CfC tests**

Run: `python -m pytest tests/imputation_v3/test_cfc.py tests/validation_v2/test_models.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit the CfC encoder**

```bash
git add imputation_v3/models/cfc.py tests/imputation_v3/test_cfc.py
git commit -m "feat(imputation-v3): add bidirectional continuous-time encoder"
```

### Task 6: Teacher fusion, residual heads, and reconstruction objective

**Files:**
- Create: `imputation_v3/models/teacher.py`
- Create: `imputation_v3/objectives/__init__.py`
- Create: `imputation_v3/objectives/reconstruction.py`
- Test: `tests/imputation_v3/test_teacher.py`
- Test: `tests/imputation_v3/test_objective.py`

- [ ] **Step 1: Write teacher and loss tests**

```python
import pytest
import torch

from imputation_v3.models.teacher import OfflineTeacher
from imputation_v3.objectives.reconstruction import channel_balanced_missing_mse


def test_teacher_returns_raw_completed_and_latent_components():
    model = OfflineTeacher(input_size=31, cfc_hidden=8, tcn_width=6, tcn_dilations=(1, 2))
    features = torch.randn(2, 12, 31)
    dt = torch.full((2, 12), 0.01)
    observed = torch.randn(2, 12, 6)
    mask = torch.ones_like(observed)
    mask[:, 4:8] = 0
    baseline = torch.randn_like(observed)
    result = model(features, dt, observed, mask, baseline)
    assert result.raw.shape == observed.shape
    assert result.latent.shape[:2] == observed.shape[:2]
    torch.testing.assert_close(result.completed[mask.bool()], observed[mask.bool()])


def test_raw_ablation_does_not_add_interpolation_baseline():
    model = OfflineTeacher(31, cfc_hidden=4, tcn_width=4, tcn_dilations=(1,), residual_mode="raw")
    features = torch.randn(1, 6, 31)
    dt = torch.full((1, 6), 0.01)
    observed = torch.zeros(1, 6, 6)
    mask = torch.zeros_like(observed)
    baseline = torch.full_like(observed, 1000.0)
    result = model(features, dt, observed, mask, baseline)
    torch.testing.assert_close(result.raw, result.residual)


def test_channel_balanced_loss_rejects_no_missing_values():
    with pytest.raises(ValueError, match="missing"):
        channel_balanced_missing_mse(torch.zeros(2, 3, 6), torch.zeros(2, 3, 6), torch.ones(2, 3, 6))
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/imputation_v3/test_teacher.py tests/imputation_v3/test_objective.py -v`  
Expected: FAIL because teacher and objective modules are missing.

- [ ] **Step 3: Implement teacher fusion and missing-only objective**

```python
# imputation_v3/models/teacher.py
from dataclasses import dataclass

import torch
from torch import nn

from .baselines import complete_signal
from .cfc import BidirectionalCfCEncoder
from .tcn import SymmetricTCNEncoder


@dataclass(frozen=True)
class TeacherOutput:
    raw: torch.Tensor
    completed: torch.Tensor
    residual: torch.Tensor
    latent: torch.Tensor


class OfflineTeacher(nn.Module):
    def __init__(
        self, input_size: int, cfc_hidden: int, tcn_width: int,
        tcn_dilations: tuple[int, ...], *, residual_mode: str = "residual",
        time_mode: str = "actual",
    ) -> None:
        super().__init__()
        if residual_mode not in {"residual", "raw"}:
            raise ValueError("residual_mode must be residual or raw")
        if time_mode not in {"actual", "constant", "dt_feature_only", "no_dt"}:
            raise ValueError("unsupported time_mode")
        self.residual_mode = residual_mode
        self.time_mode = time_mode
        self.cfc = BidirectionalCfCEncoder(input_size, cfc_hidden)
        self.tcn = SymmetricTCNEncoder(input_size, width=tcn_width, dilations=tcn_dilations)
        fusion_size = cfc_hidden * 2 + tcn_width + 6
        self.trunk = nn.Sequential(nn.Linear(fusion_size, 96), nn.GELU(), nn.Linear(96, 48), nn.GELU())
        self.gyro_head = nn.Linear(48, 3)
        self.acc_head = nn.Linear(48, 3)

    def forward(self, features, dt, observed, mask, baseline, *, time_mode=None, nominal_dt_s=0.01):
        selected_time_mode = self.time_mode if time_mode is None else time_mode
        cfc = self.cfc(features, dt, mode=selected_time_mode, nominal_dt_s=nominal_dt_s)
        tcn = self.tcn(features)
        latent = self.trunk(torch.cat((cfc, tcn, baseline), dim=-1))
        residual = torch.cat((self.gyro_head(latent), self.acc_head(latent)), dim=-1)
        raw = baseline + residual if self.residual_mode == "residual" else residual
        completed = complete_signal(observed, mask, raw)
        return TeacherOutput(raw=raw, completed=completed, residual=residual, latent=latent)
```

```python
# imputation_v3/objectives/reconstruction.py
import torch


def channel_balanced_missing_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or target.shape != mask.shape or target.shape[-1] != 6:
        raise ValueError("prediction, target, and mask must share (..., 6)")
    missing = 1.0 - mask
    counts = missing.sum(dim=tuple(range(missing.ndim - 1)))
    valid = counts > 0
    if not torch.any(valid):
        raise ValueError("at least one missing value is required")
    squared = ((prediction - target) ** 2 * missing).sum(dim=tuple(range(missing.ndim - 1)))
    return (squared[valid] / counts[valid]).mean()
```

- [ ] **Step 4: Run teacher and objective tests**

Run: `python -m pytest tests/imputation_v3/test_teacher.py tests/imputation_v3/test_objective.py -v`  
Expected: PASS with finite gradients through CfC, TCN, trunk, and both heads.

- [ ] **Step 5: Commit the teacher model**

```bash
git add imputation_v3/models/teacher.py imputation_v3/objectives tests/imputation_v3/test_teacher.py tests/imputation_v3/test_objective.py
git commit -m "feat(imputation-v3): add offline residual teacher"
```

### Task 7: Shared deterministic windows and mask curriculum

**Files:**
- Create: `imputation_v3/data/windows.py`
- Test: `tests/imputation_v3/test_windows.py`

- [ ] **Step 1: Write fairness and context-boundary tests**

```python
import numpy as np

from imputation_v3.data.windows import materialize_teacher_windows
from validation_v2.data.normalization import RobustTrainScaler
from validation_v2.types import Recording


def test_materialized_windows_share_targets_masks_and_ids_across_models():
    recording = Recording(
        id="r1",
        imu_time_s=np.arange(40) * 0.01,
        imu_six=np.arange(240, dtype=float).reshape(40, 6),
        vicon_time_s=np.arange(40) * 0.01,
        vicon_position_m=np.zeros((40, 3)),
        vicon_quaternion_xyzw=np.zeros((40, 4)),
        overlap_s=(0.0, 0.39),
        metadata={"scenario": "unit"},
    )
    scaler = RobustTrainScaler.fit([recording], allowed_ids={"r1"})
    windows = materialize_teacher_windows(
        [recording], scaler, window_samples=16, stride=8, seed=7,
        topologies=("point", "block", "channel"), rates=(0.1, 0.2, 0.3, 0.4),
    )
    assert windows
    assert len({window.window_id for window in windows}) == len(windows)
    assert all(window.target.shape == (16, 6) for window in windows)
    assert all(window.features.shape == (16, 31) for window in windows)

    exhaustive = materialize_teacher_windows(
        [recording], scaler, window_samples=16, stride=16, seed=7,
        topologies=("point", "block", "channel"), rates=(0.1, 0.2), exhaustive=True,
    )
    assert len(exhaustive) == 2 * 3 * 2
```

- [ ] **Step 2: Run window tests and verify failure**

Run: `python -m pytest tests/imputation_v3/test_windows.py -v`  
Expected: FAIL because window materialization is missing.

- [ ] **Step 3: Implement one immutable prepared-window contract**

Append this immutable record to `imputation_v3/types.py`:

```python
@dataclass(frozen=True, init=False, repr=False, eq=False)
class PreparedWindow:
    _features: torch.Tensor = field(repr=False)
    _target: torch.Tensor = field(repr=False)
    _observed: torch.Tensor = field(repr=False)
    _mask: torch.Tensor = field(repr=False)
    _dt: torch.Tensor = field(repr=False)
    _time: torch.Tensor = field(repr=False)
    _baseline: torch.Tensor = field(repr=False)
    window_id: str
    recording_id: str
    topology: str
    requested_fraction: float
    realized_fraction: float

    def __init__(self, *, features, target, observed, mask, dt, time, baseline,
                 window_id, recording_id, topology, requested_fraction, realized_fraction):
        for name, value in {
            "features": features, "target": target, "observed": observed, "mask": mask,
            "dt": dt, "time": time, "baseline": baseline,
        }.items():
            object.__setattr__(self, f"_{name}", _copy(value))
        object.__setattr__(self, "window_id", str(window_id))
        object.__setattr__(self, "recording_id", str(recording_id))
        object.__setattr__(self, "topology", str(topology))
        object.__setattr__(self, "requested_fraction", float(requested_fraction))
        object.__setattr__(self, "realized_fraction", float(realized_fraction))

    def _get(self, name: str) -> torch.Tensor:
        return _copy(object.__getattribute__(self, f"_{name}"))

    features = property(lambda self: self._get("features"))
    target = property(lambda self: self._get("target"))
    observed = property(lambda self: self._get("observed"))
    mask = property(lambda self: self._get("mask"))
    dt = property(lambda self: self._get("dt"))
    time = property(lambda self: self._get("time"))
    baseline = property(lambda self: self._get("baseline"))
```

```python
# imputation_v3/data/windows.py
import hashlib
from collections.abc import Sequence

import torch

from imputation_v3.data.features import build_features
from imputation_v3.models.baselines import timestamp_linear
from imputation_v3.types import PreparedWindow
from validation_v2.data.masking import channel_outage, contiguous_block, point_missing


GENERATORS = {"point": point_missing, "block": contiguous_block, "channel": channel_outage}


def materialize_teacher_windows(
    recordings, scaler, *, window_samples, stride, seed, topologies, rates, exhaustive=False,
):
    prepared = []
    for recording in sorted(recordings, key=lambda item: item.id):
        normalized = torch.tensor(scaler.transform(recording.imu_six), dtype=torch.float32)
        time = torch.tensor(recording.imu_time_s, dtype=torch.float32)
        dt = torch.diff(time, prepend=time[:1])
        dt[0] = torch.median(dt[1:])
        for start in range(0, len(time) - window_samples + 1, stride):
            stop = start + window_samples
            identity = f"{recording.id}:{start}:{stop}:{seed}"
            digest = hashlib.sha256(identity.encode("utf-8")).digest()
            target = normalized[start:stop]
            conditions = (
                [(topology, rate) for topology in topologies for rate in rates]
                if exhaustive else
                [(topologies[digest[0] % len(topologies)], rates[digest[1] % len(rates)])]
            )
            for condition_index, (topology, rate) in enumerate(conditions):
                condition_seed = int.from_bytes(digest[2:6], "big") + condition_index
                mask_result = GENERATORS[topology](
                    target, requested_fraction=rate, seed=condition_seed,
                )
                mask = mask_result.mask
                observed = torch.where(mask.bool(), target, torch.zeros_like(target))
                local_time = time[start:stop]
                local_dt = dt[start:stop]
                features = build_features(target, mask, local_dt).values
                baseline = timestamp_linear(observed, mask, local_time, empty_fill=0.0)
                prepared.append(PreparedWindow(
                    window_id=f"{recording.id}:{start}:{stop}:{topology}:{rate}",
                    recording_id=recording.id, features=features, target=target,
                    observed=observed, mask=mask, dt=local_dt, time=local_time,
                    baseline=baseline, topology=topology, requested_fraction=rate,
                    realized_fraction=mask_result.realized_fraction,
                ))
    return prepared
```

- [ ] **Step 4: Run window, scaler, and mask regression tests**

Run: `python -m pytest tests/imputation_v3/test_windows.py tests/validation_v2/test_splits_and_scaler.py tests/validation_v2/test_masking_and_features.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit shared experiment windows**

```bash
git add imputation_v3/types.py imputation_v3/data/windows.py tests/imputation_v3/test_windows.py
git commit -m "feat(imputation-v3): materialize fair teacher windows"
```

### Task 8: Teacher training callbacks and smoke execution

**Files:**
- Create: `imputation_v3/experiments/__init__.py`
- Create: `imputation_v3/experiments/training.py`
- Create: `imputation_v3/cli.py`
- Test: `tests/imputation_v3/test_training.py`

- [ ] **Step 1: Write one-step optimization and selection tests**

```python
from types import SimpleNamespace

import torch

from imputation_v3.experiments.training import make_teacher_callbacks
from imputation_v3.models.teacher import OfflineTeacher


def test_teacher_callback_updates_parameters_and_returns_missing_rmse():
    target = torch.randn(2, 8, 6)
    mask = torch.ones_like(target)
    mask[:, 2:6] = 0
    observed = torch.where(mask.bool(), target, 0.0)
    prepared_batch = SimpleNamespace(
        features=torch.randn(2, 8, 31),
        target=target,
        observed=observed,
        mask=mask,
        dt=torch.full((2, 8), 0.01),
        baseline=torch.zeros_like(target),
    )
    model = OfflineTeacher(31, cfc_hidden=4, tcn_width=4, tcn_dilations=(1,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_epoch, evaluate_epoch = make_teacher_callbacks(torch.device("cpu"))
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = train_epoch(model, optimizer, [prepared_batch], 1)
    assert set(metrics) == {"missing_rmse"}
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert evaluate_epoch(model, [prepared_batch], 1)["missing_rmse"] >= 0
```

- [ ] **Step 2: Run training tests and verify failure**

Run: `python -m pytest tests/imputation_v3/test_training.py -v`  
Expected: FAIL because training callbacks are missing.

- [ ] **Step 3: Implement callbacks around the validated checkpoint engine**

```python
# imputation_v3/experiments/training.py
import numpy as np
import torch

from imputation_v3.objectives.reconstruction import channel_balanced_missing_mse


def make_teacher_callbacks(device: torch.device):
    def move(batch):
        return {name: getattr(batch, name).to(device) for name in ("features", "target", "observed", "mask", "dt", "baseline")}

    def train_epoch(model, optimizer, loader, epoch):
        del epoch
        model.train()
        values = []
        for source in loader:
            batch = move(source)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["features"], batch["dt"], batch["observed"], batch["mask"], batch["baseline"])
            loss = channel_balanced_missing_mse(output.raw, batch["target"], batch["mask"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            values.append(float(torch.sqrt(loss.detach()).cpu()))
        return {"missing_rmse": float(np.mean(values))}

    def evaluate_epoch(model, loader, epoch):
        del epoch
        model.eval()
        values = []
        with torch.no_grad():
            for source in loader:
                batch = move(source)
                output = model(batch["features"], batch["dt"], batch["observed"], batch["mask"], batch["baseline"])
                loss = channel_balanced_missing_mse(output.raw, batch["target"], batch["mask"])
                values.append(float(torch.sqrt(loss).cpu()))
        return {"missing_rmse": float(np.mean(values))}

    return train_epoch, evaluate_epoch
```

Use `validation_v2.experiments.train.train_one_run` for atomic best-checkpoint selection and resume validation. The CLI exposes `python -m imputation_v3.cli teacher --config <yaml> --smoke` and refuses `test` selection.

- [ ] **Step 4: Run training tests and a one-epoch CPU smoke command**

Run: `python -m pytest tests/imputation_v3/test_training.py tests/validation_v2/test_train_evaluate.py -v`  
Expected: PASS.  
Run: `python -m imputation_v3.cli teacher --config configs/imputation_v3/teacher_smoke.yaml --smoke --device cpu`  
Expected: exit 0 and a run directory containing `run.json`, `history.json`, `best.pt`, and `checkpoint.json`.

- [ ] **Step 5: Commit teacher training**

```bash
git add imputation_v3/experiments imputation_v3/cli.py tests/imputation_v3/test_training.py
git commit -m "feat(imputation-v3): train offline teacher checkpoints"
```

### Task 9: Native controls and PyPOTS strong-baseline adapters

**Files:**
- Create: `imputation_v3/models/native_controls.py`
- Create: `imputation_v3/experiments/pypots.py`
- Test: `tests/imputation_v3/test_pypots.py`
- Modify: `imputation_v3/config.py`

- [ ] **Step 1: Write adapter contract tests with fake PyPOTS models**

```python
from types import SimpleNamespace

import numpy as np
import torch

from imputation_v3.experiments.pypots import PyPOTSAdapter, to_pypots_sets
from imputation_v3.models.native_controls import (
    BiCfCControl, BiLSTMControl, FeatureMLPControl, TCNControl, count_parameters,
)


class FakeModel:
    def fit(self, train_set, val_set):
        assert set(train_set) == {"X"}
        assert set(val_set) == {"X", "X_ori"}

    def impute(self, test_set, **kwargs):
        del kwargs
        return np.nan_to_num(test_set["X"], nan=7.0)


def test_pypots_adapter_uses_nan_only_at_declared_missing_positions():
    target = np.arange(48, dtype=np.float32).reshape(1, 8, 6)
    mask = np.ones_like(target)
    mask[:, 2:5, 1:4] = 0
    train = SimpleNamespace(target=target, mask=mask)
    validation = SimpleNamespace(target=target + 1, mask=mask)
    train_set, val_set = to_pypots_sets(train, validation)
    assert np.array_equal(np.isnan(train_set["X"]), train.mask == 0)
    adapter = PyPOTSAdapter(FakeModel())
    adapter.fit(train_set, val_set)
    result = adapter.impute({"X": train_set["X"]})
    assert result.shape == train_set["X"].shape


def test_native_controls_share_input_and_completion_contract():
    features = torch.randn(2, 10, 31)
    dt = torch.full((2, 10), 0.01)
    observed = torch.randn(2, 10, 6)
    mask = torch.ones_like(observed)
    mask[:, 3:7] = 0
    baseline = torch.randn_like(observed)
    models = (
        BiLSTMControl(31, 8), BiCfCControl(31, 8), TCNControl(31, 8, (1, 2)),
        FeatureMLPControl(31, 16),
    )
    for model in models:
        output = model(features, dt, observed, mask, baseline)
        assert output.raw.shape == observed.shape
        assert count_parameters(model) > 0
        torch.testing.assert_close(output.completed[mask.bool()], observed[mask.bool()])
```

- [ ] **Step 2: Install the pinned baseline environment and run the failing test**

Run: `python -m pip install -r requirements-imputation-v3-baselines.txt`  
Expected: successful installation with `pypots==1.5.0`.  
Run: `python -m pytest tests/imputation_v3/test_pypots.py -v`  
Expected: FAIL because adapters are missing.

- [ ] **Step 3: Implement one shared baseline interface**

```python
# imputation_v3/experiments/pypots.py
import numpy as np


def to_pypots_sets(train, validation):
    train_x = np.where(train.mask.astype(bool), train.target, np.nan)
    validation_x = np.where(validation.mask.astype(bool), validation.target, np.nan)
    return {"X": train_x}, {"X": validation_x, "X_ori": validation.target}


class PyPOTSAdapter:
    def __init__(self, model, *, impute_kwargs=None):
        self.model = model
        self.impute_kwargs = dict(impute_kwargs or {})

    def fit(self, train_set, validation_set):
        self.model.fit(train_set, validation_set)

    def impute(self, test_set):
        result = self.model.impute(test_set, **self.impute_kwargs)
        return np.asarray(result)


def build_pypots_model(name, *, n_steps, epochs, batch_size, device, saving_path):
    if name == "brits":
        from pypots.imputation import BRITS
        return PyPOTSAdapter(BRITS(n_steps=n_steps, n_features=6, rnn_hidden_size=64, batch_size=batch_size, epochs=epochs, device=device, saving_path=saving_path, model_saving_strategy="best"))
    if name == "saits":
        from pypots.imputation import SAITS
        return PyPOTSAdapter(SAITS(n_steps=n_steps, n_features=6, n_layers=2, d_model=64, n_heads=4, d_k=16, d_v=16, d_ffn=128, dropout=0.1, batch_size=batch_size, epochs=epochs, device=device, saving_path=saving_path, model_saving_strategy="best"))
    if name == "csdi":
        from pypots.imputation import CSDI
        model = CSDI(n_steps=n_steps, n_features=6, n_layers=4, n_heads=4, n_channels=64, d_time_embedding=64, d_feature_embedding=16, d_diffusion_embedding=64, n_diffusion_steps=50, batch_size=batch_size, epochs=epochs, device=device, saving_path=saving_path, model_saving_strategy="best")
        return PyPOTSAdapter(model, impute_kwargs={"n_sampling_times": 20})
    raise ValueError(f"unsupported PyPOTS model: {name}")
```

Create `imputation_v3/models/native_controls.py` with the shared contract below:

```python
import torch
from torch import nn

from .baselines import complete_signal
from .cfc import BidirectionalCfCEncoder
from .tcn import SymmetricTCNEncoder
from .teacher import TeacherOutput


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class _ResidualControl(nn.Module):
    def __init__(self, representation_size: int) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(representation_size + 6, 48), nn.GELU(), nn.Linear(48, 6))

    def finish(self, representation, observed, mask, baseline):
        latent = torch.cat((representation, baseline), dim=-1)
        residual = self.head(latent)
        raw = baseline + residual
        return TeacherOutput(raw, complete_signal(observed, mask, raw), residual, latent)


class BiLSTMControl(_ResidualControl):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__(hidden_size * 2)
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, features, dt, observed, mask, baseline):
        del dt
        representation, _ = self.encoder(features)
        return self.finish(representation, observed, mask, baseline)


class BiCfCControl(_ResidualControl):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__(hidden_size * 2)
        self.encoder = BidirectionalCfCEncoder(input_size, hidden_size)

    def forward(self, features, dt, observed, mask, baseline):
        representation = self.encoder(features, dt, mode="actual")
        return self.finish(representation, observed, mask, baseline)


class TCNControl(_ResidualControl):
    def __init__(self, input_size: int, width: int, dilations: tuple[int, ...]) -> None:
        super().__init__(width)
        self.encoder = SymmetricTCNEncoder(input_size, width=width, dilations=dilations)

    def forward(self, features, dt, observed, mask, baseline):
        del dt
        return self.finish(self.encoder(features), observed, mask, baseline)


class FeatureMLPControl(_ResidualControl):
    def __init__(self, input_size: int, width: int) -> None:
        super().__init__(width)
        self.encoder = nn.Sequential(nn.Linear(input_size, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, features, dt, observed, mask, baseline):
        del dt
        return self.finish(self.encoder(features), observed, mask, baseline)
```

Add this factory to `imputation_v3/experiments/runner.py`; condition names are explicit so ablations cannot silently drift into separate scripts:

```python
def build_native_model(condition, config):
    from imputation_v3.models.native_controls import (
        BiCfCControl, BiLSTMControl, FeatureMLPControl, TCNControl, count_parameters,
    )
    from imputation_v3.models.teacher import OfflineTeacher

    if condition == "bilstm":
        return BiLSTMControl(31, config.hidden_size)
    if condition == "bilnn":
        return BiCfCControl(31, config.hidden_size)
    if condition == "tcn":
        return TCNControl(31, config.tcn_width, config.tcn_dilations)
    if condition == "feature_mlp":
        teacher = OfflineTeacher(31, config.hidden_size, config.tcn_width, config.tcn_dilations)
        candidates = [FeatureMLPControl(31, width) for width in (32, 48, 64, 96, 128, 192)]
        return min(candidates, key=lambda model: abs(count_parameters(model) - count_parameters(teacher)))
    teacher_conditions = {
        "teacher_actual_residual": ("actual", "residual"),
        "teacher_constant_residual": ("constant", "residual"),
        "teacher_dt_feature_only_residual": ("dt_feature_only", "residual"),
        "teacher_no_dt_residual": ("no_dt", "residual"),
        "teacher_actual_raw": ("actual", "raw"),
    }
    if condition not in teacher_conditions:
        raise ValueError(f"unsupported native condition: {condition}")
    time_mode, residual_mode = teacher_conditions[condition]
    return OfflineTeacher(
        31, config.hidden_size, config.tcn_width, config.tcn_dilations,
        time_mode=time_mode, residual_mode=residual_mode,
    )
```

- [ ] **Step 4: Run adapter tests and import smoke checks**

Run: `python -m pytest tests/imputation_v3/test_pypots.py tests/imputation_v3/test_teacher.py -v`  
Expected: PASS.  
Run: `python -c "from pypots.imputation import BRITS, SAITS, CSDI; print('pypots-baselines-ok')"`  
Expected: `pypots-baselines-ok`.

- [ ] **Step 5: Commit baseline adapters**

```bash
git add imputation_v3/models/native_controls.py imputation_v3/experiments/pypots.py imputation_v3/config.py tests/imputation_v3/test_pypots.py
git commit -m "feat(imputation-v3): add strong imputation baselines"
```

### Task 10: Physical evaluation, formal matrix, and teacher success gate

**Files:**
- Create: `imputation_v3/experiments/evaluate.py`
- Create: `imputation_v3/experiments/runner.py`
- Test: `tests/imputation_v3/test_runner.py`
- Modify: `imputation_v3/cli.py`

- [ ] **Step 1: Write full-window aggregation and success-gate tests**

```python
import pandas as pd
import numpy as np

from imputation_v3.experiments.evaluate import aggregate_raw_windows, diagnostic_masks, physical_record_metrics
from imputation_v3.experiments.runner import teacher_success


def test_teacher_success_requires_ci_below_zero_against_strongest_baseline():
    summary = pd.DataFrame([
        {"model": "teacher", "baseline": "saits", "metric": "rmse_physical", "ci95_low": -0.3, "ci95_high": -0.1},
        {"model": "teacher", "baseline": "linear", "metric": "rmse_physical", "ci95_low": -0.4, "ci95_high": -0.2},
    ])
    assert teacher_success(summary, strongest_baseline="saits") is True
    summary.loc[0, "ci95_high"] = 0.01
    assert teacher_success(summary, strongest_baseline="saits") is False


def test_overlap_aggregation_precedes_physical_metrics():
    raw = [np.array([[0.0], [2.0], [4.0]]), np.array([[6.0], [8.0], [10.0]])]
    stitched = aggregate_raw_windows(raw, starts=[0, 1], recording_length=4)
    np.testing.assert_allclose(stitched[:, 0], [0.0, 4.0, 6.0, 10.0])
    rows = physical_record_metrics(
        prediction=stitched, target=np.array([[0.0], [3.0], [7.0], [10.0]]),
        missing=np.array([[False], [True], [True], [False]]), recording_id="r1",
    )
    assert rows == {"recording_id": "r1", "rmse_physical": 1.0, "mae_physical": 1.0}
    masks = diagnostic_masks(
        np.array([[False] * 6, [True] * 6, [True] * 6, [False] * 6]),
        np.array([0.0, 0.03, 0.07, 0.10]),
    )
    assert {"overall", "sensor/gyro", "sensor/accelerometer", "gap/50-200ms"} <= set(masks)
```

- [ ] **Step 2: Run runner tests and verify failure**

Run: `python -m pytest tests/imputation_v3/test_runner.py -v`  
Expected: FAIL because the formal runner is missing.

- [ ] **Step 3: Implement physical metrics and immutable formal outputs**

```python
# imputation_v3/experiments/evaluate.py
import numpy as np


def aggregate_raw_windows(raw_windows, *, starts, recording_length):
    if len(raw_windows) != len(starts) or not raw_windows:
        raise ValueError("raw_windows and starts must be non-empty and aligned")
    channels = raw_windows[0].shape[1]
    total = np.zeros((recording_length, channels), dtype=np.float64)
    count = np.zeros((recording_length, 1), dtype=np.int64)
    for raw, start in zip(raw_windows, starts):
        stop = start + len(raw)
        if start < 0 or stop > recording_length or raw.shape[1] != channels:
            raise ValueError("window lies outside the recording")
        total[start:stop] += raw
        count[start:stop] += 1
    if np.any(count == 0):
        raise ValueError("every evaluated sample must have prediction coverage")
    return total / count


def physical_record_metrics(*, prediction, target, missing, recording_id):
    selected = np.asarray(missing, dtype=bool)
    if prediction.shape != target.shape or selected.shape != target.shape or not selected.any():
        raise ValueError("physical metric arrays must align and contain missing values")
    error = np.asarray(prediction)[selected] - np.asarray(target)[selected]
    return {
        "recording_id": str(recording_id),
        "rmse_physical": float(np.sqrt(np.mean(error ** 2))),
        "mae_physical": float(np.mean(np.abs(error))),
    }


def diagnostic_masks(missing, time):
    missing = np.asarray(missing, dtype=bool)
    groups = {"overall": missing.copy()}
    groups["sensor/gyro"] = missing & np.array([True, True, True, False, False, False])
    groups["sensor/accelerometer"] = missing & np.array([False, False, False, True, True, True])
    for channel, name in enumerate(("gx", "gy", "gz", "ax", "ay", "az")):
        selected = np.zeros_like(missing)
        selected[:, channel] = missing[:, channel]
        groups[f"axis/{name}"] = selected
    gap_seconds = np.zeros_like(missing, dtype=np.float64)
    for channel in range(missing.shape[1]):
        start = None
        for index in range(len(time) + 1):
            active = index < len(time) and missing[index, channel]
            if active and start is None:
                start = index
            elif not active and start is not None:
                stop = index
                duration = float(time[stop - 1] - time[start]) + (
                    float(time[1] - time[0]) if len(time) > 1 else 0.0
                )
                gap_seconds[start:stop, channel] = duration
                start = None
    groups["gap/0-50ms"] = missing & (gap_seconds <= 0.05)
    groups["gap/50-200ms"] = missing & (gap_seconds > 0.05) & (gap_seconds <= 0.2)
    groups["gap/over-200ms"] = missing & (gap_seconds > 0.2)
    return {name: selected for name, selected in groups.items() if selected.any()}


def evaluate_record(*, raw_windows, starts, target_normalized, observed_mask, scaler, recording_id):
    raw = aggregate_raw_windows(
        raw_windows, starts=starts, recording_length=len(target_normalized),
    )
    completed = np.where(observed_mask, target_normalized, raw)
    prediction_physical = scaler.inverse_transform(completed)
    target_physical = scaler.inverse_transform(target_normalized)
    return physical_record_metrics(
        prediction=prediction_physical,
        target=target_physical,
        missing=~np.asarray(observed_mask, dtype=bool),
        recording_id=recording_id,
    )
```

```python
# imputation_v3/experiments/runner.py
def teacher_success(summary, *, strongest_baseline: str) -> bool:
    selected = summary.loc[
        (summary["model"] == "teacher")
        & (summary["baseline"] == strongest_baseline)
        & (summary["metric"] == "rmse_physical")
    ]
    if len(selected) != 1:
        raise ValueError("teacher success requires one preregistered strongest-baseline row")
    return float(selected.iloc[0]["ci95_high"]) < 0.0
```

The runner calls `evaluate_record` once per recording/model/seed and repeats the RMSE/MAE calculation over every non-empty mask returned by `diagnostic_masks`. It converts those values to the exact long-form `validation_v2.evaluation.statistics.PER_RECORD_COLUMNS` schema. For the preregistered primary comparison it first averages condition-level squared errors within each recording, and writes one additional stratum with `scenario="all"`, `protocol="teacher_primary"`, `topology="all"`, `metric="rmse_physical"`, and the mean realized fraction. This produces exactly one primary teacher-versus-baseline summary row while retaining sensor, axis, gap-duration, topology, rate, and scenario rows for secondary analysis.

Implement the formal runner in this fixed order:

1. load the frozen recording manifest and scaler;
2. materialize identical windows/masks for every model;
3. train context candidates on training data and choose context/capacity by validation RMSE only;
4. freeze one checkpoint per seed/model;
5. evaluate final checkpoints on test once;
6. call `validation_v2.evaluation.statistics.paired_model_summary` with required seeds 2026–2030;
7. write `per_record_metrics.csv`, `summary.csv`, `success_gate.json`, mask ledger, and artifact hashes without clobbering inconsistent prior outputs.

Use this exact success payload:

```python
passed = teacher_success(summary, strongest_baseline=strongest_baseline)
gate = {
    "candidate": "teacher",
    "strongest_baseline": strongest_baseline,
    "metric": "rmse_physical",
    "criterion": "paired_ci95_high_below_zero",
    "passed": passed,
    "next_stage": "plan_fixed_lag_students" if passed else "stop_and_analyze_teacher_failure",
}
```

- [ ] **Step 4: Run all v3 tests and the formal-matrix dry run**

Run: `python -m pytest tests/imputation_v3 -v`  
Expected: PASS.  
Run: `python -m imputation_v3.cli teacher-matrix --config configs/imputation_v3/teacher_smoke.yaml --dry-run`  
Expected: prints the exact seed × context × model × condition count without training or touching test results.

- [ ] **Step 5: Commit formal evaluation**

```bash
git add imputation_v3/experiments/evaluate.py imputation_v3/experiments/runner.py imputation_v3/cli.py tests/imputation_v3/test_runner.py
git commit -m "feat(imputation-v3): evaluate teacher accuracy gate"
```

### Task 11: End-to-end verification and operator runbook

**Files:**
- Create: `docs/imputation_v3_teacher_runbook.md`
- Modify: `README.md`
- Modify: `configs/imputation_v3/teacher_smoke.yaml`
- Modify: `configs/imputation_v3/teacher_full.yaml`

- [ ] **Step 1: Run the complete legacy and v3 test suites**

Run: `python -m pytest tests/validation_v2 tests/imputation_v3 -q`  
Expected: all tests pass; no Validation v2 snapshots or artifacts change.

- [ ] **Step 2: Run the real-data CPU smoke experiment twice**

Run: `python -m imputation_v3.cli teacher --config configs/imputation_v3/teacher_smoke.yaml --smoke --device cpu`  
Expected: exit 0 with finite train/validation metrics and complete provenance.  
Run the identical command again.  
Expected: validated resume of the same run ID; no checkpoint or manifest overwrite.

- [ ] **Step 3: Validate artifacts and anti-leakage invariants**

Run: `python -m imputation_v3.cli validate-artifacts --output results/imputation_v3/smoke`  
Expected: reports valid config, split, scaler, mask, checkpoint, and metrics hashes.  
Run: `python -m pytest tests/imputation_v3/test_features.py::test_hidden_targets_cannot_change_v3_features tests/imputation_v3/test_runner.py -v`  
Expected: PASS.

- [ ] **Step 4: Write the operator runbook and README entry**

Write `docs/imputation_v3_teacher_runbook.md` with these literal sections and commands:

```markdown
# Imputation v3 Offline Teacher Runbook

## Scope
This stage evaluates offline full-context accuracy only. CSDI is an offline high-compute comparator. Do not begin fixed-lag students unless `success_gate.json` contains `"passed": true`.

## Environment
`python -m venv .venv-v3`
`.venv-v3\Scripts\python -m pip install -r requirements-imputation-v3-baselines.txt`

## CPU smoke and deterministic resume
`.venv-v3\Scripts\python -m imputation_v3.cli teacher --config configs/imputation_v3/teacher_smoke.yaml --smoke --device cpu`
Run the same command again; it must validate and resume the same run ID.

## Matrix inspection
`.venv-v3\Scripts\python -m imputation_v3.cli teacher-matrix --config configs/imputation_v3/teacher_full.yaml --dry-run`

## Formal GPU run
`.venv-v3\Scripts\python -m imputation_v3.cli teacher-matrix --config configs/imputation_v3/teacher_full.yaml --device cuda`

## Artifact validation
`.venv-v3\Scripts\python -m imputation_v3.cli validate-artifacts --output results/imputation_v3/formal`

## Expected output
Each run contains `run.json`, `history.json`, `best.pt`, and `checkpoint.json`. The formal root contains `per_record_metrics.csv`, `summary.csv`, `mask_ledger.csv`, `artifact_hashes.json`, and `success_gate.json`.

## Gate interpretation
Pass only when the paired per-recording teacher-minus-strongest-baseline RMSE 95% confidence interval has `ci95_high < 0`. On failure, stop and analyze teacher errors; no student or Jetson claim is authorized.

## External baseline API
PyPOTS 1.5 documentation: https://docs.pypots.com/
PyPOTS source: https://github.com/WenjieDu/PyPOTS
```

Add one README bullet linking to `docs/imputation_v3_teacher_runbook.md` under the experiment documentation section; do not rewrite unrelated README content.

- [ ] **Step 5: Commit the verified teacher implementation handoff**

```bash
git add docs/imputation_v3_teacher_runbook.md README.md configs/imputation_v3
git commit -m "docs(imputation-v3): add offline teacher runbook"
```

## Final verification checklist

Before claiming this plan complete, run:

```bash
python -m pytest tests/validation_v2 tests/imputation_v3 -q
python -m imputation_v3.cli teacher-matrix --config configs/imputation_v3/teacher_full.yaml --dry-run
git status --short
```

The first two commands must exit 0. `git status --short` may show the user-owned `output/review/rereview_validation_v2/` directory, but it must not show uncommitted v3 source, test, config, or documentation files.

Only after formal GPU runs produce `success_gate.json` with `passed: true` should a new plan be written for fixed-lag student distillation. Jetson deployment remains a third plan after the student accuracy gate.
