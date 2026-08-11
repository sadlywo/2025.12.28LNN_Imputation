"""Dataset loading, recording-level splits, and train-only normalization."""

from .adapters import (
    DatasetAdapter,
    DatasetSemantics,
    EuRoCMAVAdapter,
    IDOLAdapter,
    get_dataset_adapter,
    register_dataset_adapter,
)

from .features import build_features
from .euroc import load_euroc_recording
from .idol import load_idol_recording
from .masking import (
    channel_outage,
    contiguous_block,
    generate_interval_jittered_time,
    point_missing,
)
from .normalization import (
    RobustTrainScaler,
    denormalize_imu_tensor,
    imu_dataset_units_to_si,
)
from .oxiod import load_recording, overlapping_interval
from .splits import leave_one_scenario_out, stratified_file_split
from .windows import make_windows

__all__ = [
    "RobustTrainScaler",
    "DatasetAdapter",
    "DatasetSemantics",
    "EuRoCMAVAdapter",
    "IDOLAdapter",
    "build_features",
    "channel_outage",
    "contiguous_block",
    "generate_interval_jittered_time",
    "get_dataset_adapter",
    "denormalize_imu_tensor",
    "imu_dataset_units_to_si",
    "leave_one_scenario_out",
    "load_recording",
    "load_euroc_recording",
    "load_idol_recording",
    "make_windows",
    "overlapping_interval",
    "point_missing",
    "register_dataset_adapter",
    "stratified_file_split",
]
