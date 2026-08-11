# Pre-`validation_v2` code archive

This directory is a frozen archive of the original root-level pipeline. It
contains the old entrypoint, shared modules, demos, ablations, visualization
tools, and inference benchmarks as one dependency-coherent unit.

## Scope

- `main.py`, `config.py`, `dataset.py`, `models*.py`, `train.py`, and
  `visualization.py` form the old core pipeline.
- `experiment_*.py`, `demo_*.py`, `benchmark_*.py`, `inspect_*.py`, and
  `visualize_*.py` are historical one-off runs.
- `requirements-legacy.txt` is the old environment specification.
- `ORIGINAL_README.md` is the historical top-level documentation preserved
  byte-for-byte.

These files are kept for reproducibility only. They are not imported by the
current `validation_v2` package and should not receive new features.

## Reproducing an old run

Run from the repository root so the historical relative dataset and output
paths continue to resolve:

```powershell
python -m pip install -r legacy/pre_validation_v2/requirements-legacy.txt
python legacy/pre_validation_v2/main.py --root_dir "Oxford Dataset"
```

Direct execution adds this archive directory to `sys.path`, so the old
absolute imports such as `from dataset import ...` still resolve within the
archive. Historical scripts may write to their original result paths; do not
mix those artifacts with `results/physics_loss_refactor/`.
