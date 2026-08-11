"""Validate local EuRoC/IDOL data and initialize adapter-ready assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import BadZipFile, ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation_v2.data import get_dataset_adapter


EUROC_MEMBERS = (
    "mav0/imu0/data.csv",
    "mav0/imu0/sensor.yaml",
    "mav0/state_groundtruth_estimate0/data.csv",
    "mav0/state_groundtruth_estimate0/sensor.yaml",
)
IDOL_EXPECTED_MD5 = {
    "building1.zip": "4e676daf04e1b3db2646af34f68cd487",
    "building2.zip": "1ced63665db2c5dffc014f97d6efe66e",
    "building3.zip": "ed0743f9fb3dbc05938713eba266231a",
}


def _hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_if_identical_or_missing(path: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if _hash(path) != digest:
            raise ValueError(f"initialized file differs from source archive: {path}")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return digest


def _record_summary(recording) -> dict[str, object]:
    if recording.imu_six.ndim != 2 or recording.imu_six.shape[1] != 6:
        raise ValueError(f"recording {recording.id} has invalid IMU shape")
    if recording.vicon_position_m.shape[1:] != (3,):
        raise ValueError(f"recording {recording.id} has invalid position shape")
    if recording.vicon_quaternion_xyzw.shape[1:] != (4,):
        raise ValueError(f"recording {recording.id} has invalid quaternion shape")
    arrays = (
        recording.imu_time_s,
        recording.imu_six,
        recording.vicon_time_s,
        recording.vicon_position_m,
        recording.vicon_quaternion_xyzw,
    )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError(f"recording {recording.id} contains non-finite values")
    if np.any(np.diff(recording.imu_time_s) <= 0) or np.any(
        np.diff(recording.vicon_time_s) <= 0
    ):
        raise ValueError(f"recording {recording.id} has non-monotonic timestamps")
    quaternion_error = np.max(
        np.abs(np.linalg.norm(recording.vicon_quaternion_xyzw, axis=1) - 1.0)
    )
    if quaternion_error > 1e-6:
        raise ValueError(f"recording {recording.id} has non-unit quaternions")
    return {
        "recording_id": recording.id,
        "imu_rows": int(recording.imu_six.shape[0]),
        "reference_rows": int(recording.vicon_position_m.shape[0]),
        "duration_s": float(recording.overlap_s[1] - recording.overlap_s[0]),
        "median_imu_dt_s": float(np.median(np.diff(recording.imu_time_s))),
        "max_quaternion_norm_error": float(quaternion_error),
    }


def initialize_euroc(archive_root: Path, processed_root: Path) -> dict[str, object]:
    archives = sorted(
        path
        for path in archive_root.rglob("*.zip")
        if path.stem.startswith(("V1_", "V2_"))
    )
    if len(archives) != 6:
        raise ValueError(f"expected 6 EuRoC Vicon sequence ZIPs, found {len(archives)}")
    archive_rows: list[dict[str, object]] = []
    for archive in archives:
        sequence = archive.stem
        try:
            with ZipFile(archive) as bundle:
                names = set(bundle.namelist())
                missing = sorted(set(EUROC_MEMBERS) - names)
                if missing:
                    raise ValueError(f"{archive} is missing target members: {missing}")
                member_rows = []
                for member in EUROC_MEMBERS:
                    payload = bundle.read(member)  # ZipFile verifies this member's CRC.
                    destination = processed_root / sequence / Path(member)
                    member_rows.append(
                        {
                            "member": member,
                            "bytes": len(payload),
                            "sha256": _write_if_identical_or_missing(destination, payload),
                            "initialized_path": _relative(destination),
                        }
                    )
        except BadZipFile as exc:
            raise ValueError(f"invalid EuRoC ZIP: {archive}") from exc
        archive_rows.append(
            {
                "sequence": sequence,
                "archive": _relative(archive),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": _hash(archive),
                "target_members": member_rows,
            }
        )

    adapter = get_dataset_adapter("euroc_mav")
    pairs = adapter.discover(processed_root)
    recordings = [
        _record_summary(adapter.load(Path(pair["imu_path"]), Path(pair["vicon_path"])))
        for pair in pairs
    ]
    return {
        "adapter": adapter.name,
        "semantics": adapter.semantics.__dict__,
        "source_archive_count": len(archives),
        "recording_count": len(recordings),
        "archives": archive_rows,
        "recordings": recordings,
        "status": "validated_target_members_and_initialized",
    }


def initialize_idol(raw_root: Path) -> dict[str, object]:
    archive_rows = []
    expected_extracted_files: set[str] = set()
    for name, expected_md5 in IDOL_EXPECTED_MD5.items():
        archive = raw_root / name
        if not archive.is_file():
            raise ValueError(f"IDOL source archive is missing: {archive}")
        try:
            with ZipFile(archive) as bundle:
                members = {
                    name for name in bundle.namelist() if not name.endswith("/")
                }
                member_count = len(members)
                expected_extracted_files.update(members)
        except BadZipFile as exc:
            raise ValueError(f"invalid IDOL ZIP: {archive}") from exc
        actual_md5 = _hash(archive, "md5")
        if actual_md5 != expected_md5:
            raise ValueError(
                f"IDOL checksum mismatch for {name}: {actual_md5} != {expected_md5}"
            )
        archive_rows.append(
            {
                "archive": _relative(archive),
                "archive_bytes": archive.stat().st_size,
                "member_count": member_count,
                "md5": actual_md5,
                "expected_md5": expected_md5,
            }
        )

    actual_extracted_files = {
        path.relative_to(raw_root).as_posix()
        for building in ("building1", "building2", "building3")
        for path in (raw_root / building).rglob("*")
        if path.is_file() and path.suffix in {".feather", ".json"}
    }
    if actual_extracted_files != expected_extracted_files:
        missing = sorted(expected_extracted_files - actual_extracted_files)
        extra = sorted(actual_extracted_files - expected_extracted_files)
        raise ValueError(
            f"IDOL extraction does not match ZIP members; missing={missing}, extra={extra}"
        )

    adapter = get_dataset_adapter("idol")
    pairs = adapter.discover(raw_root)
    recordings = [
        _record_summary(adapter.load(Path(pair["imu_path"]), Path(pair["vicon_path"])))
        for pair in pairs
    ]
    counts: dict[str, int] = {}
    for pair in pairs:
        key = f"{pair['scenario']}/{pair['source_subset']}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "adapter": adapter.name,
        "semantics": adapter.semantics.__dict__,
        "source_archive_count": len(archive_rows),
        "extracted_file_count": len(actual_extracted_files),
        "extraction_matches_archives": True,
        "recording_count": len(recordings),
        "recording_counts": dict(sorted(counts.items())),
        "archives": archive_rows,
        "recordings": recordings,
        "status": "official_checksums_and_all_feather_recordings_validated",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--euroc-archives",
        type=Path,
        default=ROOT / "datasets" / "raw" / "euroc_mav" / "archives",
    )
    parser.add_argument(
        "--euroc-processed",
        type=Path,
        default=ROOT / "datasets" / "processed" / "euroc_mav",
    )
    parser.add_argument(
        "--idol-root",
        type=Path,
        default=ROOT / "datasets" / "raw" / "idol" / "archives",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets" / "manifests" / "external_datasets.json",
    )
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "euroc_mav": initialize_euroc(
            args.euroc_archives.resolve(), args.euroc_processed.resolve()
        ),
        "idol": initialize_idol(args.idol_root.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": _relative(args.output),
                "euroc_recordings": report["euroc_mav"]["recording_count"],
                "idol_recordings": report["idol"]["recording_count"],
                "status": "completed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
