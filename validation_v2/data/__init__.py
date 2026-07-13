"""Dataset loading, recording-level splits, and train-only normalization."""

from .features import build_features
from .masking import (
    channel_outage,
    contiguous_block,
    generate_interval_jittered_time,
    point_missing,
)
from .normalization import RobustTrainScaler
from .oxiod import load_recording, overlapping_interval
from .splits import leave_one_scenario_out, stratified_file_split
from .windows import make_windows

__all__ = [
    "RobustTrainScaler",
    "build_features",
    "channel_outage",
    "contiguous_block",
    "generate_interval_jittered_time",
    "leave_one_scenario_out",
    "load_recording",
    "make_windows",
    "overlapping_interval",
    "point_missing",
    "stratified_file_split",
]
