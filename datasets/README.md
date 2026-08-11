# Local datasets

This directory is the local landing area for additional datasets. Large raw,
extracted, cached, and processed files are intentionally excluded from Git.

```text
raw/
  euroc_mav/
    archives/
    extracted/
  idol/
    archives/
    extracted/
processed/
cache/
manifests/
```

Keep download checksums and small, hand-reviewed recording manifests under
`manifests/`. Dataset adapters must emit the canonical sample contract defined
in `docs/physics_loss_refactor_dataset_contract.md` and must declare units,
frames, timestamp semantics, and calibration provenance explicitly.

Initialize and validate the currently downloaded subsets from the repository
root:

```powershell
python scripts/initialize_external_datasets.py
```

The initializer verifies the published IDOL MD5 checksums, loads every Feather
trajectory, and extracts only EuRoC IMU/ground-truth CSV and sensor YAML files
to `processed/euroc_mav/`. Camera images and ROS bags are not duplicated.

## Minimal cloud bundle

Do not upload `raw/` wholesale. The reproducible cloud bundle contains only:

- the OxIOD CSV tree under `Oxford Dataset/`;
- the initialized EuRoC IMU/ground-truth subset under `processed/euroc_mav/`;
- the three official IDOL building ZIP files; and
- the external-dataset validation manifest.

Create the bundle from the repository root on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package_minimal_dataset.ps1
```

The generated archive, SHA-256 checksum, and JSON package manifest are written
under `transfer/`, which is intentionally ignored by Git.
