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

After uploading the archive and its `.sha256` sidecar to a Linux server,
restore it from the cloned repository root:

```bash
cd /root/workspace/lnn-imputation
DATA_BUNDLE_DIR=/path/to/uploaded-bundle
(cd "$DATA_BUNDLE_DIR" && \
  sha256sum -c lnn-imputation-data-minimal-v1.tar.zst.sha256)
tar --use-compress-program=unzstd \
  -xf "$DATA_BUNDLE_DIR/lnn-imputation-data-minimal-v1.tar.zst" -C .

python run.py smoke --config euroc_adapter_smoke.yaml --device cuda
python run.py smoke --config idol_adapter_smoke.yaml --device cuda
```

The IDOL adapter reads the three ZIP files directly. The bundle already
contains the initialized EuRoC sensor/ground-truth subset, so the full EuRoC
camera archives are not required on the server.
