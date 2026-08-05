# External validation datasets

The dataset archives are deliberately excluded from Git because they are
third-party research data and exceed GitHub's normal file-size limits.  Restore
them on a new machine with the checked-in downloader:

```powershell
python scripts/download_external_datasets.py euroc idol
```

The downloader resumes partial transfers, verifies the exact byte count and
MD5 digest, and writes the archives below `external_datasets/euroc` and
`external_datasets/idol`.

To download only the datasets currently used by the validation work:

```powershell
python scripts/download_external_datasets.py euroc-room1 idol-building1
```

To inspect the manifest without downloading:

```powershell
python scripts/download_external_datasets.py --list
```

## Sources

- EuRoC MAV: ETH Zurich, DOI `10.3929/ethz-b-000690084`.  The script uses a
  byte-identical Hugging Face mirror because the ETH repository may block
  automated traffic from some providers.
- IDOL: Zenodo record `4484093`, DOI `10.5281/zenodo.4484093`.

Review and comply with each dataset's terms before redistribution or use.  The
archives are not owned or relicensed by this repository.
