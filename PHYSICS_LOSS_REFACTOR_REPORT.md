# Physics loss refactor report

## Status

The refactor is implemented on branch `codex/physics-loss-refactor`, created
from `validation_v2` commit `b94e1b3`. The Hybrid BiLNN/BiLSTM branches,
adaptive gate, six-axis output, missing-pattern generators, and file-level
split strategy were not redesigned.

The real-data `lambda_phy=0` pipeline and all 15 regular/irregular evaluation
cells pass a bounded OxIOD smoke run. Non-zero real-data physics training is
deliberately fail-closed because the clean diagnostic found an unresolved gyro
sign / IMU-to-Vicon frame convention. A non-zero synthetic Hybrid mini-step
passes and supplies gradients to both branches and the gate.

## A. Dataset semantics

- OxIOD input channels are `rotation_rate_x/y/z` in rad/s followed by
  `user_acc_x/y/z` in G.
- The repository ReadMe and CoreMotion naming identify `user_acc` as
  gravity-compensated linear acceleration. The OxIOD configuration therefore
  uses `acceleration_mode: gravity_compensated`; mechanization does not add
  gravity in this mode.
- The train-only normalization is the existing median/MAD
  `RobustTrainScaler`, not Z-score normalization. Completed IMU is transformed
  back with the fitted train center/scale before acceleration is converted from
  G to m/s^2.
- IMU timestamps are seconds. Raw Vicon timestamps are nanoseconds and are
  converted to seconds by the existing loader. No clipping is used to conceal
  bad time units.
- Vicon CSV columns are actually `rotation.x/y/z/w`, so the code uses `xyzw`.
  Quaternions are normalized, made sign-continuous, and interpolated by SLERP.
- Aligned Vicon velocity uses central differences internally and
  forward/backward differences at endpoints; it is never initialized to zero
  by default.
- The canonical adapter boundary is in
  `validation_v2/data/adapters.py`. EuRoC MAV and IDOL are registered without
  dataset-specific branches in training. Their detailed contract is in
  `docs/physics_loss_refactor_dataset_contract.md`.

## B. Coordinate conventions

The implemented mathematical convention is:

- `R`: body to world
- gyro: rad/s in body frame
- acceleration: m/s^2 in body frame
- velocity: m/s in world frame
- position: m in world frame
- `dt[:, i]`: seconds from sample `i-1` to sample `i`
- quaternion: normalized `xyzw`
- OxIOD acceleration: gravity-compensated, therefore no `+g`
- raw specific-force mode: `R @ (f-b_a) + g_world`

These are implementation conventions, not a claim that the supplied OxIOD
Vicon rigid body is already the same physical body frame as the phone IMU.
That fixed extrinsic and gyro sign remain unresolved, so a positive physics
weight cannot be enabled merely by editing lambda.

## C. Mathematical implementation

The only optimized objective is

```text
L_total = L_sig + lambda_phy * L_phy

L_sig = sum((1-M) * (prediction-target)^2) / (sum(1-M) + eps)

completed = M * target + (1-M) * prediction

r_R = Log(R_gt,end^T R_hat,end)^vee
r_v = v_hat,end - v_gt,end
r_p = p_hat,end - p_gt,end

L_phy = mean(
    ||r_R||^2 / sigma_R^2
  + ||r_v||^2 / sigma_v^2
  + ||r_p||^2 / sigma_p^2
)
```

Code correspondence:

| Mathematical item | Code |
|---|---|
| missing-only `L_sig` | `validation_v2.objectives.reconstruction.missing_mse` |
| exact completion | `validation_v2.models.hybrid.complete_signal` |
| denormalization and SI conversion | `validation_v2.data.normalization` |
| `skew`, `Exp`, `Log`, quaternion conversion | `validation_v2.physics.so3` |
| midpoint/trapezoidal six-axis propagation | `validation_v2.physics.mechanization.propagate_imu` |
| endpoint residuals and scaled `L_phy` | `validation_v2.objectives.physics_informed.IMUPhysicsInformedLoss` |
| training/logging integration | `validation_v2.experiments.runner._epoch_callbacks` |
| test-time physics diagnostics | `validation_v2.evaluation.physics.physics_endpoint_diagnostics` |

The loss path is pure PyTorch and contains no NumPy, SciPy, detach, or
CPU-array conversion before `total` is formed. Gyro and accelerometer biases
are explicit optional fixed inputs and default to zero; the model does not
predict a per-window bias. Vicon is passed only to the criterion/evaluator and
is not an input to model forward or deployment inference.

No distribution, smoothness, energy-threshold, gyro-derivative, arbitrary
temporal consistency, uncertainty-weighting, or physics-head loss enters the
new objective.

## D. Changed files

- `.gitignore`: isolates generated refactor diagnostics and runs.
- `configs/validation_v2/physics_refactor_smoke.yaml`: bounded Hybrid config,
  unchanged regular missingness axes, and 10%/20%/40% interval-jitter cases.
- `validation_v2/physics/{so3,mechanization}.py`: differentiable rotation and
  complete six-axis inertial propagation.
- `validation_v2/objectives/physics_informed.py`: formal two-term objective and
  component logging; `objectives/__init__.py` exports it.
- `validation_v2/data/normalization.py`: differentiable tensor normalize /
  denormalize and SI conversion.
- `validation_v2/data/adapters.py`: canonical dataset boundary for OxIOD,
  EuRoC MAV, and IDOL; `data/__init__.py` exports all three adapters.
- `validation_v2/data/euroc.py`, `validation_v2/data/idol.py`: unit-explicit
  loaders that convert source wxyz quaternions to canonical xyzw.
- `validation_v2/evaluation/synchronization.py`: quaternion normalization,
  shortest-path SLERP output rotations, and Vicon velocity.
- `validation_v2/evaluation/physics.py`: missing-only gyro/accelerometer RMSE
  and rotation/velocity/position endpoint diagnostics.
- `validation_v2/experiments/runner.py`: raw versus completed prediction,
  synchronized physics labels, criterion integration, gradient/component
  logs, frame-validation gate, and uniquely identified irregular conditions.
- `validation_v2/modern/{config,cli,export}.py`: retains the legacy irregular
  count API while carrying multiple explicit irregular case specifications and
  dataset names.
- `scripts/diagnose_imu_vicon_mechanization.py`: complete-IMU convention audit.
- `scripts/run_physics_lambda_ablation.py`: exact requested lambda grid
  `[0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]` with the same fail-closed gate.
- `scripts/initialize_external_datasets.py`: validates the downloaded archives,
  initializes EuRoC sensor-only files, and writes the data manifest.
- `validation_v2/cli.py`: rejects unsafe shard-plan outputs before importing
  heavy training modules, while retaining lazy imports for executable paths.
- `scripts/run_validation_v2_server.sh` and MatPool helpers: lock the cloud
  runtime to PyTorch 2.11/CUDA 12.8, audit RTX 4090/5090 devices, default to
  two workers, and pin shard processes round-robin across visible GPUs.
- `tests/validation_v2/test_physics_loss_refactor.py`: the requested analytic,
  gradient, device, and Hybrid mini-step tests.
- `results/legacy_archive/README.md`: immutable index of old scopes; historical
  outputs were not moved or overwritten.
- `results/physics_loss_refactor/README.md`: new result namespace contract.
- `docs/physics_loss_refactor_dataset_contract.md`: EuRoC/IDOL adapter contract.

## E. Tests

The dedicated test file covers all twelve requested behaviours:

1. exact prediction gives zero signal loss;
2. observed completion equals measured input exactly;
3. missing completion equals prediction;
4. observed prediction receives zero physics gradient;
5. missing gyro receives non-zero gradient;
6. missing accelerometer receives non-zero gradient;
7. small/normal-angle `Log(Exp(phi))` round trips;
8. zero gyro preserves orientation;
9. constant acceleration matches analytic velocity/position;
10. consistent synthetic IMU/pose gives near-zero physics loss;
11. available CPU/CUDA paths are finite;
12. a non-zero-lambda Hybrid step backpropagates through BiLNN, BiLSTM, and gate.

Executed release-candidate results on 2026-08-11:

- Full repository suite: `735 passed, 9 skipped` with process exit code 0.
- Physics loss, adapters, synchronization, and trajectory focus:
  `29 passed`.
- Python bytecode compilation for `run.py`, `validation_v2`, `scripts`, and
  `tests`: completed with exit code 0.
- Formal `server_full` matrix dry-run: 4,096 JSONL lines containing the header
  plus exactly 4,095 experiment combinations.
- Shell syntax checks passed for the generic server runner, MatPool launcher,
  and shared server helpers under Git Bash.

The nine skips are platform/feature-conditioned tests; they are not failed
physics, dataset, model, or server-contract assertions.

## F. Diagnostic and smoke results

The clean diagnostic used two 100 Hz OxIOD recordings, four 30-sample windows
per recording. A 1 Hz recording was explicitly excluded from the clean frame
audit because a 30-sample window spans 29 seconds and is not comparable to the
short-window assumption.

The best tested hypothesis was:

```text
Vicon rotation: body_to_world (not transposed)
gyro sign:      -1
acceleration:   gravity_compensated
mean rotation endpoint error: 3.157186 deg
mean velocity endpoint error: 0.198069 m/s
mean position endpoint error: 0.035684 m
```

The negative gyro sign conflicts with the current direct-use convention and
may also be standing in for an unmodelled fixed phone-IMU/Vicon rigid-body
extrinsic. Therefore `frame_validation_status` remains
`diagnostic_only_not_validated`.

The real lambda-zero matrix smoke completed with one training group and 15
evaluation cells: 12 unchanged regular conditions (point/block/channel at
10/20/30/40% missingness) plus point-missing interval jitter at 10/20/40%.
The training log recorded finite signal/physics components and gradient norm.

In addition, bounded real-data CUDA smoke runs completed independently for all
three adapters (OxIOD, EuRoC MAV, and IDOL). Each run loaded a real validation
recording, trained for one epoch, evaluated 120 samples, wrote immutable output
artifacts, and reported a completed status with split and scaler hashes. The
local smoke runtime was PyTorch 2.5.1/CUDA 12.1; the cloud launcher separately
enforces the RTX 5090-compatible PyTorch 2.11.0/CUDA 12.8 lock.

Comparison status:

- `lambda_phy=0`, real OxIOD: completed successfully.
- `lambda_phy=0.1`, synthetic consistent data: completed successfully; both
  Hybrid branches and gate received gradients.
- `lambda_phy>0`, real OxIOD: intentionally blocked until frame/extrinsic
  validation. No unsupported real-data result is reported as valid.

## G. Remaining uncertainties

1. Whether the Vicon quaternion is the phone body frame or a separate rigid
   marker frame with a fixed extrinsic rotation.
2. Why the clean comparison favours a global negative gyro sign.
3. Whether acceleration axes require the same extrinsic/sign mapping.
4. Whether a single mapping is valid for every OxIOD device/recording,
   including the 1 Hz stream.
5. EuRoC MAV and IDOL file layouts and acceleration semantics are implemented,
   but their frame-direction/extrinsic validation remains gated; parsing
   success does not authorize a non-zero physics weight.

The positive-lambda gate must remain closed until items 1-3 are resolved on
clean, unmasked data and recorded as an explicit fixed transform/convention.

## H. Reproduction commands

From the repository root:

```powershell
python -m pip install -r requirements-validation-v2.txt

python -m pytest tests/validation_v2/test_physics_loss_refactor.py `
  tests/validation_v2/test_synchronization_and_trajectory.py `
  tests/validation_v2/test_splits_and_scaler.py `
  tests/validation_v2/modern/test_config.py -q

python scripts/diagnose_imu_vicon_mechanization.py `
  --max-recordings 3 --windows-per-recording 4 --seq-len 30

python -m validation_v2.cli matrix `
  --config configs/validation_v2/physics_refactor_smoke.yaml --device cpu

python scripts/run_physics_lambda_ablation.py --dry-run
```

After the frame mapping is genuinely validated, encode the fixed mapping,
change `frame_validation_status` to `validated`, rerun the clean diagnostic,
and only then run:

```powershell
python scripts/run_physics_lambda_ablation.py --device cuda
```
