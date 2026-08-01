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

The same command accepts the smoke root at `results/imputation_v3/smoke` or one completed smoke run directory. Validation is read-only and fails on incomplete, unexpected, noncanonical, hash-mismatched, symlinked, path-escaping, or provenance-inconsistent artifacts.

## Expected output
Each run contains `run.json`, `history.json`, `best.pt`, and `checkpoint.json`. The formal root contains `per_record_metrics.csv`, `summary.csv`, `mask_ledger.csv`, `coverage_ledger.csv`, `artifact_hashes.json`, and `success_gate.json`.

## Gate interpretation
Pass only when the paired per-recording teacher-minus-strongest-baseline RMSE 95% confidence interval has `ci95_high < 0`. On failure, stop and analyze teacher errors; no student or Jetson claim is authorized.

## External baseline API
PyPOTS 1.5 documentation: https://docs.pypots.com/
PyPOTS source: https://github.com/WenjieDu/PyPOTS
