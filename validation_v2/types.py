"""Shared types for the validation v2 package."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Recording:
    """One raw IMU/Vicon pair with explicit units and time overlap."""

    id: str
    imu_time_s: np.ndarray
    imu_six: np.ndarray
    vicon_time_s: np.ndarray
    vicon_position_m: np.ndarray
    vicon_quaternion_xyzw: np.ndarray
    overlap_s: tuple[float, float]
    metadata: Mapping[str, object]
