# Offline CfC–TCN Teacher and Fixed-Lag Student Design

Date: 2026-07-31
Status: Approved design
Scope: New `imputation_v3` experiment family; Validation v2 remains unchanged

## 1. Objective

The project will replace the current hybrid-model research direction with a staged teacher–student experiment:

1. Establish an offline imputation accuracy upper bound with a full-window bidirectional CfC–TCN teacher.
2. Distill that teacher into fixed-lag CfC–TCN students with bounded future context.
3. Demonstrate the primary student on a Jetson Nano under a 100 ms end-to-end deadline.
4. Defer STM32F4 quantization, kernel implementation, and second-stage distillation until the Jetson experiment succeeds.

The primary scientific outcome is imputation accuracy on held-out recordings. Real-time deployment is a second-stage outcome. GPU hardware is used for offline training, not as embedded-deployment evidence.

## 2. Terminology and Claim Boundary

The fixed-lag student is a **temporally causal/online imputation model with bounded look-ahead**. This means that an estimate for target time `t` may use observations only up to `t + L`, where `L` is the declared future budget.

This is not statistical causal inference: the experiment does not identify interventions, treatment effects, structural causal models, or counterfactual outcomes. The paper must not use `do(·)` or causal-effect language for this architecture.

The offline teacher is explicitly an offline bidirectional imputer. It is not a real-time model.

## 3. Current-Code Audit

The repository contains three materially different LNN implementations:

- `models.py` is a legacy single-direction CfC model. Its input is `[masked IMU(6), mask(6), dt(1)]`; however, `dt` is only an ordinary feature and is not supplied through the CfC `timespans` argument. Its comment that CfC automatically handles irregular time is therefore not sufficient evidence of continuous-time operation.
- `legacy/pre_validation_v2/experiment_bidirectional_lnn_residual.py` is a historical residual hybrid. It uses forward/backward CfC and BiLSTM branches around an interpolation baseline, but it belongs to an older experiment path that previously included target-derived window features and did not establish valid CfC timespan semantics or gate attribution.
- `validation_v2` is the current trustworthy evidence path. Its 25-D observed-only feature contract is `[observed(6), mask(6), dt(1), delta(6), valid_delta(6)]`. `BidirectionalCfC` supplies actual forward and direction-aligned reverse intervals through `timespans`. The hybrid fuses BiCfC and BiLSTM outputs with a per-time, per-channel gate and preserves observed values exactly.

Validation v2 is nevertheless an offline model: both branches are bidirectional, inference stitches overlapping windows, and the formal objective is reconstruction-only. Existing formal results do not establish a real-time, physics-informed, or statistical-causal contribution. The v3 work starts from the leakage-safe data and provenance principles of Validation v2, not from its hybrid architecture.

## 4. Shared Data Contract

### 4.1 Observed-only features

Each input time step uses the following 31-D causal feature vector:

- observed IMU values: 6;
- binary observation mask: 6;
- true elapsed interval `dt`: 1;
- elapsed time since the last observed value, per channel: 6;
- causal slope from the two latest observed values, per channel: 6;
- slope-valid indicator, per channel: 6.

Hidden target values must never be evaluated while constructing features. All missing-value placeholders are deterministic. Normalization is fitted only on training recordings and then frozen.

The teacher and students share this feature contract. They differ only in model architecture, interpolation baseline, and permitted future context.

### 4.2 Missingness and time

Point missingness, contiguous blocks, and channel outage are separate value-missingness topologies. Timestamp irregularity is a separate experimental axis and must not be described as value missingness.

Every result records requested and realized missing fractions. Fully missing channels use a declared train-median/decay fallback; no hidden value or test statistic is available to the fallback.

### 4.3 Splits and identity

Splits are recording-level and frozen before training. Every run stores:

- commit and environment identity;
- resolved configuration and seed;
- recording split, scaler, and mask-manifest hashes;
- checkpoint and output hashes;
- per-record metrics and a test ledger.

Validation performs all architecture, context-length, capacity, loss-weight, and checkpoint selection. Final test checkpoints are evaluated once.

## 5. Offline Teacher

### 5.1 Architecture

The teacher is a residual full-window hybrid with three stages:

1. **Bidirectional CfC path.** A forward CfC consumes actual `timespans`. A reverse CfC consumes correctly direction-aligned reverse intervals. Their representations encode continuous-time dynamics from both directions.
2. **Symmetric multi-scale TCN path.** Depthwise-separable residual temporal blocks model short- and medium-range motion patterns with full past and future access inside the declared teacher window.
3. **Feature-level fusion and residual decoder.** CfC and TCN representations are concatenated with a timestamp-aware full-window linear-interpolation baseline. A shared decoder trunk feeds separate three-channel gyroscope and accelerometer residual heads. The model predicts one residual path; it does not create two complete predictions and interpret a learned branch gate.

The final missing-value prediction is `linear_base + learned_residual`. Observed entries are restored exactly with `torch.where`-equivalent semantics.

The initial capacity is CfC hidden size 64, TCN width 48 with three depthwise residual blocks, and a `96 -> 48` decoder. A small validation-only capacity grid may be specified in the implementation plan. Equal-capacity controls must use the same information budget.

### 5.2 Context

Candidate teacher windows are 1.28, 2.56, and 5.12 seconds. The chosen duration is selected on validation data. “Full context” means unrestricted bidirectional access within the selected offline window, not access to hidden targets or Vicon data.

Full-record inference uses deterministic overlapping windows and overlap aggregation. Observed entries are restored only once after aggregation. Window coverage must be complete and auditable.

### 5.3 Objective

The sole primary objective is channel-balanced, missing-only MSE in robustly normalized units:

- compare prediction and same-time target;
- multiply by `1 - mask`;
- normalize by the actual number of missing elements;
- reject batches or conditions with no realized missing values.

No smoothness, trajectory, uncertainty, energy, or physics term is part of the primary teacher. A first-difference auxiliary loss may be run only as a separately named, preregistered ablation. It cannot restore a `physics-informed` claim.

## 6. Fixed-Lag Students

### 6.1 Architecture

Each student uses:

1. a forward streaming CfC with persistent hidden state and actual `timespans`;
2. a timestamp-keyed ring buffer;
3. a lightweight bounded-future depthwise TCN over `[t - 200 ms, t + L]` with time-offset and padding masks;
4. a compact feature-fusion residual decoder;
5. a bounded interpolation baseline that may use a right observation only when it arrives before the deadline.

Observed channels pass through immediately. Missing channels are finalized when their deadline is reached. A recording boundary resets CfC state and the ring buffer.

### 6.2 Latency budgets

Students are trained and evaluated for future budgets of 0, 30, 50, 70, and 90 ms. The 90 ms model is the primary Jetson candidate. Approximately 10 ms is reserved for preprocessing, inference, and completion so that the declared end-to-end p95 remains no greater than 100 ms.

The buffer is governed by timestamps rather than a fixed sample count. Irregular sampling may change the number of available future samples but never the permitted elapsed-time boundary.

### 6.3 Distillation

The teacher is frozen before student training. The student loss is:

`L = L_GT + lambda_KD * L_teacher + lambda_feat * L_latent`

where:

- `L_GT` is missing-only ground-truth MSE;
- `L_teacher` matches teacher predictions only at hidden positions;
- `L_latent` matches projected teacher and student representations.

Distillation weights are selected on a small validation-only grid defined before formal runs. The grid includes `lambda_KD = 0` and `lambda_feat = 0`, ensuring a supervised-only control. Prediction-only and prediction-plus-latent distillation are reported separately.

## 7. Baselines and Attribution Controls

### 7.1 Traditional baselines

- last observation carried forward;
- timestamp-aware linear interpolation;
- cubic/PCHIP interpolation where valid;
- Kalman filter/RTS smoother with a declared state and tuning protocol.

### 7.2 Strong neural baselines

- equal-budget BiLSTM;
- equal-budget bidirectional CfC;
- BRITS;
- SAITS;
- CSDI as a high-compute probabilistic upper comparator.

CSDI latency is not compared as though it were a real-time model. Its role is offline accuracy context.

### 7.3 Required ablations

1. **CfC time semantics:** actual timespans, constant timespans, `dt` feature only, and no `dt`.
2. **Hybrid contribution:** CfC–TCN, CfC-only, TCN-only, and matched-capacity feature concatenation.
3. **Residual contribution:** interpolation plus residual, raw neural prediction, and interpolation only.
4. **Distillation contribution:** ground truth only, prediction distillation, and prediction plus latent distillation.
5. **Look-ahead contribution:** offline teacher and 0/30/50/70/90 ms students.

These controls are necessary to attribute performance to continuous-time updates, hybridization, residual learning, and distillation rather than parameter count or a strong interpolation baseline.

## 8. Evaluation and Success Criteria

### 8.1 Accuracy

Primary accuracy metrics are physical-unit RMSE and MAE on missing entries. Results are also reported by gyroscope/accelerometer, axis, missingness topology, realized rate, gap duration, and scenario.

The independent statistical unit is the held-out recording. Results are first summarized per recording, then across five preregistered seeds. Reports include mean and standard deviation, median and IQR, paired bootstrap 95% confidence intervals, and an effect size.

The teacher succeeds only if its paired RMSE difference against the strongest eligible baseline has a 95% confidence interval entirely below zero for the preregistered macro-average. Conditions in which it loses remain visible.

The student deployment stage succeeds only if the primary 90 ms student:

- beats timestamp-aware linear interpolation and its no-distillation counterpart under the paired protocol;
- reports its accuracy retention relative to the teacher;
- satisfies the Jetson latency contract below.

### 8.2 Jetson Nano deployment

Training remains on the available GPU platform. Deployment evidence is produced on a physical Jetson Nano with batch size 1 and a real streaming buffer.

The benchmark includes feature construction, buffering, model execution, residual completion, and output transfer. It records:

- warm-up policy and at least 1,000 measured frames;
- p50, p95, and p99 compute and end-to-end latency;
- throughput and peak memory;
- power mode, clocks, temperature, software versions, and precision;
- FP32 reference and FP16/export variants where numerical parity passes.

The required contract is end-to-end p95 no greater than 100 ms, with the primary design reserving 90 ms for look-ahead and approximately 10 ms for compute. If export output exceeds the predefined numerical tolerance relative to the PyTorch reference, that artifact cannot enter the latency benchmark.

### 8.3 Downstream trajectory

Trajectory metrics remain secondary diagnostics. They cannot select checkpoints or overturn the primary signal-level result because the present inertial integration pipeline has known drift and physical-validity limitations.

## 9. Runtime and Error Handling

The implementation fails closed on:

- non-finite or non-positive intervals;
- invalid shapes, devices, or dtypes;
- missing recording-boundary resets;
- incomplete stitched-window coverage;
- conditions with no realized hidden values;
- export/reference parity failure;
- missing provenance fields.

Late or out-of-order packets follow a declared policy: log and discard them rather than expanding a target’s future budget. The policy and count are included in deployment results.

## 10. Verification

### 10.1 Unit tests

- hidden-target perturbation cannot change features;
- student output at `t` cannot change when data after `t + L` changes;
- forward and reverse timespans align correctly;
- observed values are preserved exactly;
- ring-buffer release obeys timestamp deadlines;
- missing-loss denominators use realized hidden elements;
- full-channel fallback uses only train statistics.

### 10.2 Integration tests

- tiny teacher training and checkpoint resume;
- frozen-teacher student distillation;
- deterministic full-record stitching;
- all look-ahead variants on a small real-data slice;
- provenance and artifact validation;
- reference/export inference parity.

### 10.3 Formal experiment verification

- five-seed per-record statistical aggregation;
- requested versus realized missingness audit;
- deterministic rerun audit;
- Jetson streaming benchmark under recorded thermal and power conditions.

## 11. Proposed Code Layout

The new work is isolated from Validation v2:

```text
imputation_v3/
  data/
    features.py
    buffering.py
    masking.py
    windows.py
  models/
    common.py
    teacher.py
    student.py
    completion.py
  objectives/
    reconstruction.py
    distillation.py
  experiments/
    train_teacher.py
    distill_student.py
    matrix.py
    evaluate.py
    statistics.py
    provenance.py
  deploy/
    export.py
    streaming.py
    benchmark_jetson.py
configs/imputation_v3/
tests/imputation_v3/
```

Reusable Validation v2 primitives may be imported only when their contracts already satisfy v3 requirements. V3 must not silently modify Validation v2 behavior or historical artifacts.

## 12. Stop Conditions and Scope Control

- If the teacher does not significantly beat the strongest baseline, stop before distillation and analyze the failure.
- If the 90 ms student exceeds the Jetson compute budget, reduce student capacity or use a parity-checked FP16 path and measure once more. If it still fails, report that the 100 ms contract was not met.
- Do not add STM32F4 deployment, statistical causal inference, Vicon-supervised inputs, or a new physics loss to this implementation cycle.
- Do not use test performance to add baselines, change context length, alter loss weights, or revise the success threshold.

## 13. Primary References

- Hasani et al., “Closed-form Continuous-time Neural Models”: <https://arxiv.org/abs/2106.13898>
- Bai et al., “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling”: <https://openreview.net/pdf?id=H1VkMrJwG>
- Cao et al., “BRITS: Bidirectional Recurrent Imputation for Time Series”: <https://arxiv.org/abs/1805.10572>
- Du et al., “SAITS: Self-Attention-based Imputation for Time Series”: <https://arxiv.org/abs/2202.08516>
- Tashiro et al., “CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation”: <https://arxiv.org/abs/2107.03502>
- NVIDIA Jetson Nano specifications: <https://developer.nvidia.com/embedded/jetson-nano>
