"""Dataset adapters for six-axis IMU + pose-reference corpora.

OxIOD, EuRoC MAV, and IDOL emit the canonical ``Recording`` contract, so the
training pipeline does not need dataset-specific branches.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import re

from validation_v2.types import Recording

from .euroc import EUROC_IMU_CHANNEL_NAMES, load_euroc_recording
from .idol import IDOL_IMU_CHANNEL_NAMES, load_idol_recording
from .oxiod import IMU_CHANNEL_NAMES, load_recording


@dataclass(frozen=True)
class DatasetSemantics:
    gyro_unit: str
    acceleration_unit: str
    acceleration_mode: str
    quaternion_order: str
    rotation_mapping: str
    time_unit: str = "s"


class DatasetAdapter(Protocol):
    """Canonical loader contract for a dataset-specific directory layout."""

    name: str
    semantics: DatasetSemantics
    channel_names: tuple[str, ...]

    def discover(self, root: Path) -> Sequence[Mapping[str, str]]: ...

    def load(self, imu_path: Path, reference_path: Path) -> Recording: ...


def _oxiod_scenario(directory_name: str) -> str:
    if directory_name.startswith("handbag-"):
        return "handbag"
    if directory_name.startswith("handheld-"):
        return "handheld"
    if directory_name == "slow walking":
        return "slow_walking"
    return directory_name


class OxIODAdapter:
    name = "oxiod"
    channel_names = tuple(IMU_CHANNEL_NAMES)
    semantics = DatasetSemantics(
        gyro_unit="rad/s",
        acceleration_unit="G",
        acceleration_mode="gravity_compensated",
        quaternion_order="xyzw",
        rotation_mapping="body_to_world_unverified_extrinsic",
    )

    def discover(self, root: Path) -> Sequence[Mapping[str, str]]:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f"dataset root is not a directory: {root}")
        pairs: list[dict[str, str]] = []
        for directory in sorted(
            (item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name
        ):
            imu = {
                int(match.group(1)): path
                for path in directory.glob("imu*.csv")
                if (match := re.fullmatch(r"imu(\d+)\.csv", path.name))
            }
            reference = {
                int(match.group(1)): path
                for path in directory.glob("vi*.csv")
                if (match := re.fullmatch(r"vi(\d+)\.csv", path.name))
            }
            if set(imu) != set(reference):
                raise ValueError(f"unpaired IMU/reference files in {directory}")
            for index in sorted(imu):
                pairs.append(
                    {
                        "recording_id": f"{directory.name}/imu{index}",
                        "scenario": _oxiod_scenario(directory.name),
                        "imu_path": str(imu[index].resolve()),
                        "vicon_path": str(reference[index].resolve()),
                    }
                )
        if not pairs:
            raise ValueError(f"no paired recordings found under {root}")
        return pairs

    def load(self, imu_path: Path, reference_path: Path) -> Recording:
        return load_recording(imu_path, reference_path)


class EuRoCMAVAdapter:
    """Adapter for sensor-only extractions of EuRoC Vicon Room sequences."""

    name = "euroc_mav"
    channel_names = tuple(EUROC_IMU_CHANNEL_NAMES)
    semantics = DatasetSemantics(
        gyro_unit="rad/s",
        acceleration_unit="m/s^2",
        acceleration_mode="specific_force",
        quaternion_order="xyzw",
        rotation_mapping="sensor_to_reference_documented_not_validated",
    )

    def discover(self, root: Path) -> Sequence[Mapping[str, str]]:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f"dataset root is not a directory: {root}")
        pairs: list[dict[str, str]] = []
        for imu_path in sorted(root.rglob("data.csv")):
            if imu_path.parent.name != "imu0" or imu_path.parent.parent.name != "mav0":
                continue
            sequence_root = imu_path.parents[2]
            reference_path = (
                imu_path.parent.parent / "state_groundtruth_estimate0" / "data.csv"
            )
            if not reference_path.is_file():
                raise ValueError(f"EuRoC ground truth is missing for {imu_path}")
            sequence = sequence_root.name
            room_match = re.fullmatch(r"V([12])_\d{2}_.+", sequence)
            if not room_match:
                raise ValueError(f"unexpected EuRoC Vicon sequence name: {sequence}")
            pairs.append(
                {
                    "recording_id": f"euroc_mav/{sequence}",
                    "scenario": f"vicon_room{room_match.group(1)}",
                    "imu_path": str(imu_path.resolve()),
                    "vicon_path": str(reference_path.resolve()),
                }
            )
        if not pairs:
            raise ValueError(
                f"no extracted EuRoC imu0/ground-truth pairs found under {root}; "
                "run scripts/initialize_external_datasets.py first"
            )
        return pairs

    def load(self, imu_path: Path, reference_path: Path) -> Recording:
        return load_euroc_recording(imu_path, reference_path)


class IDOLAdapter:
    """Adapter for IDOL trajectories using Stencil IMU and SLAM ground truth."""

    name = "idol"
    channel_names = tuple(IDOL_IMU_CHANNEL_NAMES)
    semantics = DatasetSemantics(
        gyro_unit="rad/s",
        acceleration_unit="m/s^2",
        acceleration_mode="specific_force",
        quaternion_order="xyzw",
        rotation_mapping="stencil_to_global_documented_not_validated",
    )

    def discover(self, root: Path) -> Sequence[Mapping[str, str]]:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f"dataset root is not a directory: {root}")
        pairs: list[dict[str, str]] = []
        for path in sorted(root.rglob("*.feather")):
            subset = path.parent.name
            building = path.parent.parent.name
            if building not in {"building1", "building2", "building3"}:
                continue
            if subset not in {"train", "known", "unknown"}:
                raise ValueError(f"unexpected IDOL subset directory: {path.parent}")
            metadata_path = path.parent / "metadata.json"
            if not metadata_path.is_file():
                raise ValueError(f"IDOL metadata is missing: {metadata_path}")
            pairs.append(
                {
                    "recording_id": f"idol/{building}/{subset}/{path.stem}",
                    "scenario": building,
                    "source_subset": subset,
                    "imu_path": str(path.resolve()),
                    "vicon_path": str(path.resolve()),
                }
            )
        if not pairs:
            raise ValueError(f"no IDOL Feather trajectories found under {root}")
        return pairs

    def load(self, imu_path: Path, reference_path: Path) -> Recording:
        return load_idol_recording(imu_path, reference_path)


_ADAPTERS: dict[str, DatasetAdapter] = {
    "euroc_mav": EuRoCMAVAdapter(),
    "idol": IDOLAdapter(),
    "oxiod": OxIODAdapter(),
}


def register_dataset_adapter(adapter: DatasetAdapter) -> None:
    """Register a future adapter; duplicate names are rejected."""

    name = getattr(adapter, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("dataset adapter requires a non-empty name")
    if name in _ADAPTERS:
        raise ValueError(f"dataset adapter already registered: {name}")
    _ADAPTERS[name] = adapter


def get_dataset_adapter(name: str) -> DatasetAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError as error:
        available = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"unknown dataset adapter {name!r}; available: {available}."
        ) from error


__all__ = [
    "DatasetAdapter",
    "DatasetSemantics",
    "EuRoCMAVAdapter",
    "IDOLAdapter",
    "OxIODAdapter",
    "get_dataset_adapter",
    "register_dataset_adapter",
]
