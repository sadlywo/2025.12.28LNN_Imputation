# Trustworthy IMU Imputation Validation V2 Design

## 1. Purpose

Build an isolated, reproducible validation pipeline that determines whether the proposed BiLNN–BiLSTM hybrid improves missing-IMU reconstruction after removing the validity failures identified in the manuscript audit.

The V2 pipeline must answer two questions in order:

1. Does the hybrid improve leakage-free missing-point signal reconstruction relative to fair baselines?
2. If signal reconstruction improves, does it reduce physically meaningful full-record trajectory degradation under corrected synchronization and coordinate handling?

The existing scripts, checkpoints, and result directories remain unchanged. V2 produces new artifacts with explicit provenance and must not reuse legacy checkpoints implicitly.

## 2. Scope

### Included

- Fixed train/validation/test manifests at recording-file level.
- Train-only normalization.
- Model inputs computed exclusively from observed values, masks, timestamps, and observed-only derived features.
- Random, block, channel-outage, and irregular-timestamp validation with realized missingness reporting.
- Explicit CfC `timespans` handling.
- Missing-position reconstruction metrics in normalized and physical units.
- Corrected IMU/Vicon timestamp conversion and association.
- Full-record trajectory evaluation using measured attitude as evaluation-only metadata.
- Modern and diagnostic baselines, mechanism ablations, repeated seeds, confidence intervals, and effect sizes.
- Immutable run manifests and automatic table generation.
- A local smoke configuration for the `pinn_imu` conda environment and a server configuration for complete experiments.

### Excluded from the first V2 implementation

- Rewriting `Manuscript.tex`.
- Training a new attitude estimator.
- Claiming online, embedded, or physics-informed performance before the relevant acceptance tests pass.
- Reusing result numbers from legacy CSV files as V2 evidence.
- Full deployment profiling on hardware that is not locally available.

## 3. Isolation Strategy

New code lives under `validation_v2/`; tests live under `tests/validation_v2/`; configurations live under `configs/validation_v2/`; generated outputs live under `results/validation_v2/`.

Legacy modules may be imported only when they are demonstrably safe and covered by V2 tests. The legacy `dataset.py` is not used by V2 because its responsibilities—loading, normalization, masking, feature construction, Vicon synchronization, and trajectory evaluation—are coupled and contain confirmed leakage and synchronization defects.

No V2 command searches for the “latest” checkpoint. Checkpoints are referenced by an explicit run identifier or path.

## 4. Architecture

### 4.1 Package structure

```text
validation_v2/
  __init__.py
  data/
    oxiod.py              # Raw recording loading and timestamp/unit validation
    splits.py             # Deterministic recording manifests
    normalization.py      # Train-fit robust scaler
    masking.py            # Missingness generators and realized-rate metadata
    features.py           # Observed-only model input construction
    windows.py            # Training-window generation without split leakage
  models/
    adapters.py           # Observed-value preservation and explicit time tensors
    bilnn.py              # Forward/reverse CfC with correct timespans
    bilstm.py             # Bidirectional LSTM baseline branch
    hybrid.py             # Gated fusion and component outputs
    baselines.py          # Linear/LOCF/GRU/Transformer and optional external baselines
  objectives/
    reconstruction.py     # Missing-only reconstruction objective
    kinematic.py          # Optional, dimensionally explicit secondary objective
  evaluation/
    reconstruction.py     # Per-record signal metrics
    synchronization.py    # Vicon conversion, overlap and interpolation checks
    trajectory.py         # Frame-aware full-record trajectory metrics
    statistics.py         # Seed/record aggregation, CI, effect sizes
  experiments/
    train.py              # One model/seed/config training run
    evaluate.py           # Frozen-checkpoint evaluation
    matrix.py             # Experiment matrix orchestration
    provenance.py         # Run ID, hashes, environment and artifact manifest
    summarize.py          # Machine-generated tables and comparison files
  cli.py                  # `python -m validation_v2.cli ...`
```

### 4.2 Configuration structure

Two committed YAML configurations are provided:

- `configs/validation_v2/smoke.yaml`: CPU or single-GPU local test, a small recording subset, one seed, one epoch, and a reduced model matrix.
- `configs/validation_v2/server_full.yaml`: all recordings, the frozen split, at least five seeds, complete model/baseline/ablation matrix, per-record outputs, and statistical aggregation.

Every resolved configuration is copied into its run directory before training.

## 5. Data Flow

### 5.1 Raw data and time units

The loader reads IMU timestamps as seconds. Vicon timestamps are explicitly converted from nanoseconds to seconds after validating their magnitude and monotonicity. Association is restricted to the overlapping interval; extrapolation beyond either stream is an error.

Positions use linear interpolation. Quaternions use normalized SLERP. Each loaded recording must satisfy:

- strictly increasing timestamps after duplicate handling;
- nonempty IMU/Vicon overlap;
- nonconstant interpolated Vicon displacement for a moving source recording;
- finite, normalized quaternions;
- reported median and range of `dt`.

### 5.2 Split and normalization

A versioned CSV manifest assigns complete recording pairs to train, validation, and test. The default strategy is scenario-aware and file-disjoint; no window may cross a recording or split boundary.

The scaler is fit only on the training recordings. Validation and test recordings use the frozen training statistics. The scaler artifact includes channel order, center, scale, training-file hashes, and fit timestamp.

### 5.3 Masking and features

Mask conventions are fixed: `1 = observed`, `0 = artificially missing`.

Model inputs contain:

- masked six-channel IMU values;
- the six-channel mask;
- actual forward elapsed time;
- observed-only first differences with a validity mask;
- optional time-since-last-observed and time-to-next-observed features for offline models.

No statistic may be calculated from a missing target value. Window-level target means, variances, energy, and difference energy are removed. A leakage-invariance test changes hidden target values while keeping observed values and the mask fixed; the resulting model input must be bitwise identical.

Missingness generators return both the requested parameters and realized missing fraction per channel and per recording. `channel` missingness reports discrete channel counts rather than pretending that six channels can realize arbitrary percentages. Irregular-timestamp experiments modify event times or remove timestamped packets and are separate from value masking.

## 6. Model Semantics

### 6.1 Observed-value preservation

All model predictions are converted to a completed signal by

```text
completed = mask * observed_input + (1 - mask) * model_prediction
```

Signal and trajectory evaluation always use `completed`, so a model cannot improve or damage observed samples.

### 6.2 BiLNN timing

Forward CfC receives forward `timespans`. The reverse branch receives a correctly reversed and realigned positive interval tensor rather than the forward `dt` vector simply flipped inside the feature tensor.

The first controlled timing comparison contains:

- actual `timespans`;
- constant `timespans`;
- `dt` as an ordinary input feature only;
- no time information.

The paper may retain an irregular-time advantage claim only if actual `timespans` is superior under irregular timestamp tests.

### 6.3 Hybrid gate

The gate definition is fixed as the LNN branch weight. Component outputs include LNN prediction, LSTM prediction, gate, and completed output. Initial mechanism validation includes linear-only, branch-only, equal averaging, fixed gates `0/0.5/1`, learned gate, and no-extra-feature ablations.

Gate plots are descriptive unless intervention results demonstrate specialization.

## 7. Objectives

### 7.1 Primary objective

The first trustworthy experiment uses missing-only MSE or RMSE in normalized space, with no uncertainty weighting and no physics term. This isolates whether the architecture improves imputation after leakage removal.

### 7.2 Secondary kinematic objective

The kinematic objective is disabled by default until its tests pass. If enabled, it must:

- operate on denormalized acceleration in m/s²;
- use true elapsed seconds;
- identify the coordinate frame of every tensor;
- derive world-frame acceleration with measured attitude;
- compare a defined velocity/displacement quantity with synchronized labels;
- apply only the explicitly documented mask/support;
- expose each loss component and unit.

The term is called `kinematic_auxiliary_loss` until evidence justifies a stronger physics-informed label.

## 8. Trajectory Evaluation

### 8.1 Measured-attitude evaluation contract

OxIOD measured attitude is evaluation-only metadata. It is not part of the six-channel imputation model input and is not imputed. Its purpose is to rotate reconstructed user acceleration from the device/body frame into the reference/world frame, isolating acceleration-imputation error from attitude-estimation error.

Results are labeled `measured-attitude trajectory diagnostic`; they are not presented as a complete autonomous inertial-navigation system.

### 8.2 Full-record evaluation

Trajectory metrics are computed over complete held-out recordings with one initialization, not independently reset overlapping windows. The evaluation reports:

- ATE-RMSE in meters under an explicitly stated alignment policy;
- RPE/RTE at configured time intervals;
- endpoint drift in meters and as path-length percentage;
- velocity RMSE;
- imputation-induced delta relative to the complete-IMU pipeline.

Synthetic constant-acceleration, constant-velocity, stationary, and known-rotation cases validate synchronization, rotation and numerical integration before OxIOD results are accepted.

## 9. Experiment Protocol

### 9.1 Model selection

Hyperparameters and early stopping use validation data only. The test set is evaluated once per frozen run. No script prints test metrics during hyperparameter search.

### 9.2 Statistical unit and repeats

The independent unit is a recording, not a window. The server configuration runs at least five predefined seeds. Summaries include per-record values, mean ± SD, median/IQR, paired bootstrap 95% confidence intervals, and paired effect sizes. Multiplicity correction is applied to the declared primary model comparisons.

### 9.3 Baselines

The minimum matrix contains:

- zero/mean and LOCF diagnostics;
- linear interpolation;
- GRU and Transformer under the same observed-information budget;
- BiLSTM and BiLNN single branches;
- equal-average and learned-gate hybrid;
- optional BRITS/SAITS/CSDI integrations when dependencies and server budget permit.

Modern external baselines may be a second server phase, but linear interpolation and the diagnostic hybrid ablations are mandatory before attributing any gain to the gate.

## 10. Provenance and Outputs

Each run directory contains:

```text
results/validation_v2/<run_id>/
  resolved_config.yaml
  manifest.json
  split_manifest.csv
  scaler.json
  environment.txt
  checkpoints/
  per_record_metrics.csv
  predictions/              # optional compressed arrays
  training_history.csv
  summary.json
```

`manifest.json` records git commit, dirty-state digest, Python/PyTorch/CUDA/ncps versions, hostname/device, seed, input-file hashes, configuration hash, checkpoint hash, start/end times and command line.

The summarizer consumes only completed V2 manifests. Tables are generated automatically; manual copying of legacy values is prohibited.

## 11. Error Handling

The pipeline fails fast when:

- timestamp magnitudes imply unconverted units;
- IMU/Vicon overlap is empty;
- interpolation silently extrapolates;
- split manifests overlap;
- a scaler is fit with validation/test files;
- realized missingness is incompatible with the requested topology;
- a checkpoint/config hash mismatch occurs;
- required per-record metrics or seeds are absent from a summary.

Warnings are reserved for nonfatal conditions such as unusually large but valid `dt` or insufficient records for a requested subgroup statistic.

## 12. Testing Strategy

Tests are written before implementation and must be observed failing for the expected reason.

### Data invariants

- IMU/Vicon seconds conversion and nonconstant interpolation.
- Split disjointness at recording level.
- Train-only scaler fit and frozen transform.
- Leakage invariance under changed hidden targets.
- Observed-only delta features.
- Requested versus realized missingness metadata.

### Model invariants

- Observed values are preserved exactly.
- CfC receives explicit `timespans` in both directions.
- Reverse intervals are positive and correctly aligned.
- Gate `0/0.5/1` produces the expected branch mixtures.

### Objective invariants

- Reconstruction loss uses missing positions at the same time index.
- Kinematic loss uses true seconds and physical units.
- Zero-error synthetic motion gives near-zero kinematic residual.

### Trajectory invariants

- Stationary motion stays stationary.
- Constant acceleration matches the analytic position curve.
- Known body rotation maps acceleration to the expected world axis.
- Full-record evaluation initializes once.

### End-to-end checks

- Smoke configuration completes in `pinn_imu` and writes a valid manifest.
- Summarization refuses incomplete seed matrices.
- Server configuration can be parsed and its experiment matrix enumerated without starting training.

## 13. Local and Server Execution

Local validation uses:

```powershell
conda run -n pinn_imu python -m pytest tests/validation_v2 -q
conda run -n pinn_imu python -m validation_v2.cli smoke --config configs/validation_v2/smoke.yaml
```

Server execution uses:

```bash
conda run -n pinn_imu python -m validation_v2.cli matrix \
  --config configs/validation_v2/server_full.yaml
```

The server command supports resuming only explicitly identified incomplete runs. It never substitutes a checkpoint from another configuration.

## 14. Acceptance Criteria

The V2 implementation is ready for full server training only when:

1. All V2 unit and integration tests pass in `pinn_imu`.
2. The leakage-invariance test passes.
3. Timestamp association and synthetic trajectory tests pass.
4. The smoke run produces a complete, hash-linked manifest and per-record metrics.
5. The server experiment matrix can be enumerated deterministically.
6. No V2 code reads legacy result CSVs or automatically reuses legacy checkpoints.

The manuscript is ready for evidence-based revision only after the full server matrix finishes, all required seeds/recordings are present, and the generated summaries pass provenance validation.

## 15. Risks and Mitigations

- **Measured attitude may not share the assumed frame convention.** Mitigation: known-rotation tests, quaternion convention assertions and explicit diagnostic labeling.
- **Corrected leakage may materially reduce performance.** This is an intended scientific outcome; report it rather than tuning on test data.
- **Modern probabilistic baselines may exceed initial server budget.** Run the mandatory diagnostic matrix first, then add external baselines as a second frozen phase.
- **Full-record trajectory integration may remain dominated by sensor bias.** Report imputation-induced delta and duration-stratified metrics; do not interpret absolute inertial drift as imputation performance alone.
- **Legacy model classes may encode unsafe input assumptions.** Wrap only tested components; otherwise implement small V2 equivalents rather than importing the legacy class wholesale.
