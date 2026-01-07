"""Configuration for CfC-IMU imputation project."""
from dataclasses import dataclass
from typing import Literal


@dataclass
class DataConfig:
    """Dataset configuration."""
    root_dir: str = "Oxford Dataset"
    seq_len: int = 50  # 更短序列，CfC擅长长期依赖
    mask_rate: float = 0.3
    missing_mode: Literal["random", "block", "channel"] = "random"
    batch_size: int = 16
    num_workers: int = 4
    
    # IMU specific
    imu_channels: int = 6  # gyro(3) + acc(3)
    sampling_rate: float = 100.0  # Hz (approximate)
    split_ratio: float = 0.8  # train/val split


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # CfC backbone
    input_dim: int = 13  # [masked_imu(6), mask(6), dt(1)]
    hidden_units: int = 64  # CfC隐藏单元数
    output_dim: int = 6  # [gyro(3), acc(3)]
    
    # CfC specific parameters
    use_physics_prior: bool = True  # 使用物理先验（分离陀螺仪/加速度计头）
    mixed_memory: bool = True  # 使用混合记忆单元


@dataclass
class TrainingConfig:
    """Training procedure configuration."""
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    
    # Loss weights
    w_recon: float = 1.0  # 重建损失权重
    w_consistency: float = 0.5  # 时间一致性损失
    w_smooth: float = 0.1  # 平滑度损失
    
    # Scheduler
    use_scheduler: bool = True
    warmup_epochs: int = 5
    
    # Device
    device: str = "cuda"  # "cuda" or "cpu"
