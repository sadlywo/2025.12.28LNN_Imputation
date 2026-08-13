"""Recording-level dataset splits with source-file traceability.

Relative source paths are accepted only when ``base_dir`` is supplied.  This
keeps manifests reproducible instead of interpreting paths relative to an
arbitrary process working directory.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


MANIFEST_COLUMNS = (
    "recording_id",
    "scenario",
    "imu_path",
    "vicon_path",
    "split",
    "imu_sha256",
    "vicon_sha256",
)
_INDEX_COLUMNS = ("recording_id", "scenario", "imu_path", "vicon_path")
_SPLIT_NAMES = ("train", "validation", "test")


def _content_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_path(value: object, *, base_dir: Path | None, field: str) -> Path:
    try:
        path = Path(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{field} must contain filesystem paths") from exc
    path = path.expanduser()
    if not path.is_absolute():
        if base_dir is None:
            raise ValueError(f"relative {field} requires an explicit base_dir")
        path = base_dir / path
    path = path.resolve(strict=False)
    if not path.is_file():
        raise ValueError(f"{field} is not a readable file or does not exist: {path}")
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise ValueError(f"{field} is not a readable file: {path}") from exc
    return path


def _prepare_index(
    recording_index: pd.DataFrame | Sequence[Mapping[str, object]],
    *,
    base_dir: str | Path | None,
) -> pd.DataFrame:
    if isinstance(recording_index, pd.DataFrame):
        frame = recording_index.copy()
    elif isinstance(recording_index, Sequence) and not isinstance(
        recording_index, (str, bytes)
    ):
        frame = pd.DataFrame(list(recording_index))
    else:
        raise TypeError("recording_index must be a DataFrame or sequence of mappings")

    missing = [column for column in _INDEX_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"recording_index missing required columns: {missing}")
    if frame.empty:
        raise ValueError("recording_index must contain at least one recording")
    frame = frame.loc[:, _INDEX_COLUMNS].copy()

    if frame["recording_id"].isna().any() or (
        frame["recording_id"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("recording_id values must be non-empty")
    frame["recording_id"] = frame["recording_id"].astype(str)
    if frame["recording_id"].duplicated().any():
        duplicates = sorted(
            frame.loc[
                frame["recording_id"].duplicated(False), "recording_id"
            ].unique()
        )
        raise ValueError(f"duplicate recording_id values: {duplicates}")
    if frame["scenario"].isna().any() or (
        frame["scenario"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("scenario values must be non-empty")
    frame["scenario"] = frame["scenario"].astype(str)

    normalized_base = None
    if base_dir is not None:
        normalized_base = Path(base_dir).expanduser().resolve(strict=False)
    for column in ("imu_path", "vicon_path"):
        frame[column] = [
            _normalize_path(value, base_dir=normalized_base, field=column)
            for value in frame[column]
        ]

    source_owner: dict[Path, str] = {}
    duplicate_sources: set[str] = set()
    for row in frame.itertuples(index=False):
        recording_id = str(row.recording_id)
        # A synchronized container may legitimately provide both IMU and pose
        # for one recording (for example, one IDOL Feather trajectory).  The
        # leakage boundary is the recording, not the role a file plays inside
        # that recording.
        for path in {row.imu_path, row.vicon_path}:
            owner = source_owner.setdefault(path, recording_id)
            if owner != recording_id:
                duplicate_sources.add(str(path))
    if duplicate_sources:
        raise ValueError(
            "a source file is associated with multiple recordings: "
            f"{sorted(duplicate_sources)}"
        )
    return frame.sort_values("recording_id", kind="stable").reset_index(drop=True)


def _validate_ratios(ratios: Sequence[float]) -> np.ndarray:
    values = np.asarray(tuple(ratios), dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("ratios must contain three finite values")
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("ratios must be non-negative and sum to 1")
    if not np.any(values > 0):
        raise ValueError("at least one split ratio must be positive")
    return values


def _largest_remainder_counts(size: int, ratios: np.ndarray) -> np.ndarray:
    exact = ratios * size
    counts = np.floor(exact).astype(int)
    remainder = size - int(counts.sum())
    order = sorted(range(3), key=lambda index: (-float(exact[index] % 1), index))
    for index in order[:remainder]:
        counts[index] += 1

    positive = np.flatnonzero(ratios > 0)
    if size >= len(positive):
        for empty_index in positive[counts[positive] == 0]:
            donors = [index for index in positive if counts[index] > 1]
            donor = max(donors, key=lambda index: (counts[index], ratios[index], -index))
            counts[donor] -= 1
            counts[empty_index] += 1
    return counts


def _stratified_assignments(
    frame: pd.DataFrame,
    *,
    seed: int,
    ratios: np.ndarray,
) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    assignments: dict[str, str] = {}
    for _, group in frame.groupby("scenario", sort=True):
        ids = np.asarray(sorted(group["recording_id"]), dtype=object)
        ids = ids[rng.permutation(len(ids))]
        counts = _largest_remainder_counts(len(ids), ratios)
        start = 0
        for split, count in zip(_SPLIT_NAMES, counts):
            for recording_id in ids[start : start + count]:
                assignments[str(recording_id)] = split
            start += int(count)

    # Tiny strata may all choose the majority split.  If the dataset as a
    # whole is large enough, move records to empty requested splits.  Moving
    # preserves the one-recording/one-split invariant.
    active = [name for name, ratio in zip(_SPLIT_NAMES, ratios) if ratio > 0]
    if len(frame) >= len(active):
        for empty in active:
            if empty in assignments.values():
                continue
            donor = max(
                active,
                key=lambda name: (list(assignments.values()).count(name), -active.index(name)),
            )
            candidates = sorted(key for key, value in assignments.items() if value == donor)
            assignments[candidates[-1]] = empty
    return assignments


def _build_manifest(frame: pd.DataFrame, assignments: Mapping[str, str]) -> pd.DataFrame:
    manifest = frame.copy()
    manifest["imu_path"] = manifest["imu_path"].map(str)
    manifest["vicon_path"] = manifest["vicon_path"].map(str)
    manifest["split"] = manifest["recording_id"].map(assignments)
    manifest["imu_sha256"] = [
        _content_sha256(Path(path)) for path in manifest["imu_path"]
    ]
    manifest["vicon_sha256"] = [
        _content_sha256(Path(path)) for path in manifest["vicon_path"]
    ]
    return manifest.loc[:, MANIFEST_COLUMNS].reset_index(drop=True)


def stratified_file_split(
    recording_index: pd.DataFrame | Sequence[Mapping[str, object]],
    *,
    seed: int = 2026,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    base_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return a deterministic, scenario-stratified recording manifest.

    Each scenario is allocated with the largest-remainder method.  When a
    scenario has too few recordings to represent every requested split, the
    deterministic global fallback fills empty splits by moving (never copying)
    a recording from the largest split.
    """
    frame = _prepare_index(recording_index, base_dir=base_dir)
    ratio_values = _validate_ratios(ratios)
    assignments = _stratified_assignments(frame, seed=seed, ratios=ratio_values)
    return _build_manifest(frame, assignments)


def leave_one_scenario_out(
    recording_index: pd.DataFrame | Sequence[Mapping[str, object]],
    held_out_scenario: str,
    *,
    seed: int = 2026,
    validation_ratio: float = 0.15,
    base_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Hold one complete scenario out for test and split the remainder."""
    frame = _prepare_index(recording_index, base_dir=base_dir)
    held_out_scenario = str(held_out_scenario)
    if held_out_scenario not in set(frame["scenario"]):
        raise ValueError(f"held-out scenario not found: {held_out_scenario}")
    if not np.isfinite(validation_ratio) or not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be strictly between 0 and 1")

    test_mask = frame["scenario"] == held_out_scenario
    remaining = frame.loc[~test_mask]
    if remaining.empty:
        raise ValueError("leave-one-scenario-out requires recordings outside the held-out scenario")
    ratios = np.array((1.0 - validation_ratio, validation_ratio, 0.0))
    assignments = _stratified_assignments(remaining, seed=seed, ratios=ratios)
    assignments.update(
        {
            recording_id: "test"
            for recording_id in frame.loc[test_mask, "recording_id"]
        }
    )
    return _build_manifest(frame, assignments)
