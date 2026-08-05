"""Download and verify the external IMU validation datasets.

Large third-party archives are intentionally kept out of Git.  This script
recreates the local ``external_datasets`` directory with resumable downloads
and verifies every archive against its published size and MD5 checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    group: str
    filename: str
    url: str
    size: int
    md5: str


DATASETS = {
    "euroc-room1": DatasetSpec(
        key="euroc-room1",
        group="euroc",
        filename="vicon_room1.zip",
        url=(
            "https://huggingface.co/datasets/GlowBond/EuRoC_MAV_Dataset/"
            "resolve/main/vicon_room1.zip?download=true"
        ),
        size=6_042_263_426,
        md5="5ce06b405827e453a82523d3ca9c2fd0",
    ),
    "idol-building1": DatasetSpec(
        key="idol-building1",
        group="idol",
        filename="building1.zip",
        url="https://zenodo.org/api/records/4484093/files/building1.zip/content",
        size=580_418_073,
        md5="4e676daf04e1b3db2646af34f68cd487",
    ),
    "idol-building2": DatasetSpec(
        key="idol-building2",
        group="idol",
        filename="building2.zip",
        url="https://zenodo.org/api/records/4484093/files/building2.zip/content",
        size=281_164_824,
        md5="1ced63665db2c5dffc014f97d6efe66e",
    ),
    "idol-building3": DatasetSpec(
        key="idol-building3",
        group="idol",
        filename="building3.zip",
        url="https://zenodo.org/api/records/4484093/files/building3.zip/content",
        size=259_367_888,
        md5="ed0743f9fb3dbc05938713eba266231a",
    ),
}


def _md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path: Path, spec: DatasetSpec) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == spec.size
        and _md5(path).lower() == spec.md5.lower()
    )


def resolve_selection(names: Iterable[str]) -> list[DatasetSpec]:
    requested = list(names) or ["euroc", "idol"]
    resolved: list[DatasetSpec] = []
    seen: set[str] = set()
    for name in requested:
        if name in {"euroc", "idol"}:
            matches = [spec for spec in DATASETS.values() if spec.group == name]
        else:
            matches = [DATASETS[name]]
        for spec in matches:
            if spec.key not in seen:
                resolved.append(spec)
                seen.add(spec.key)
    return resolved


def _progress(filename: str, downloaded: int, total: int) -> None:
    percent = min(100.0, downloaded * 100.0 / total)
    print(
        f"\r{filename}: {downloaded:,}/{total:,} bytes ({percent:6.2f}%)",
        end="",
        flush=True,
    )


def download_file(spec: DatasetSpec, root: Path) -> Path:
    directory = root / spec.group
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / spec.filename
    partial = destination.with_suffix(destination.suffix + ".part")

    if validate_file(destination, spec):
        print(f"{destination}: already complete and verified")
        return destination

    if destination.exists():
        if destination.stat().st_size < spec.size and not partial.exists():
            destination.replace(partial)
        else:
            raise RuntimeError(
                f"Refusing to overwrite invalid archive: {destination}. "
                "Move it aside and retry."
            )

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "LNN-Imputation dataset downloader/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(spec.url, headers=headers)

    with urllib.request.urlopen(request, timeout=120) as response:
        response_status = getattr(response, "status", response.getcode())
        if offset and response_status != 206:
            offset = 0
            mode = "wb"
        else:
            mode = "ab" if offset else "wb"
        downloaded = offset
        with partial.open(mode) as output:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                _progress(spec.filename, downloaded, spec.size)
    print()

    if not validate_file(partial, spec):
        raise RuntimeError(
            f"Downloaded archive failed size or MD5 validation: {partial}"
        )
    partial.replace(destination)
    print(f"{destination}: verified ({spec.md5})")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=["euroc", "idol", *DATASETS],
        help="Dataset groups or individual archives; defaults to euroc and idol.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "external_datasets",
        help="Destination root (default: repository/external_datasets).",
    )
    parser.add_argument("--list", action="store_true", help="List archives and exit.")
    args = parser.parse_args(argv)

    specs = resolve_selection(args.datasets)
    if args.list:
        for spec in specs:
            print(f"{spec.key:16} {spec.size:>13,} bytes  {spec.md5}")
        return 0

    try:
        for spec in specs:
            download_file(spec, args.root)
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
