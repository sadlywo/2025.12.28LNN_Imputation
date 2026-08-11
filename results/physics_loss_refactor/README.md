# Physics-loss refactor result scope

Only the formal refactor writes below this directory.

- `v1/diagnostics/`: clean complete-IMU/Vicon convention diagnostics
- `v1/runs/lambda_*/`: signal-plus-physics training and evaluation artifacts

Non-zero physics weights are fail-closed until the OxIOD body/Vicon frame and
fixed extrinsic are validated. The lambda-zero run remains available as a
pipeline-equivalence and metric smoke test.
