"""Optimized dataset for CfC-based IMU imputation."""
import os
import glob
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Column definitions
IMU_COLUMNS = [
    "rotation_rate_x", "rotation_rate_y", "rotation_rate_z",
    "user_acc_x", "user_acc_y", "user_acc_z"
]
# Optional extra feature columns
GRAVITY_COLUMNS = ["grav_x", "grav_y", "grav_z"]
ATTITUDE_COLUMNS = ["att_roll", "att_pitch", "att_yaw"]
VICON_POS_COLUMNS = ["translation.x", "translation.y", "translation.z"]
VICON_QUAT_COLUMNS = ["qw", "qx", "qy", "qz"]


class CfCIMUDataset(Dataset):
    """
    Dataset optimized for CfC neural networks.
    
    Key features:
    1. Preserves actual time intervals (irregular sampling)
    2. Minimal preprocessing (CfC handles raw data well)
    3. Returns time-aware masks
    4. Robust MAD normalization (per-file)
    """
    
    def __init__(
        self,
        root_dir: str,
        seq_len: int = 50,
        mask_rate: float = 0.3,
        missing_mode: str = "random",
        split: str = "train",
        split_ratio: float = 0.8,
        eval_mode: bool = False,
        drift_scale: float = 0.0,
        return_stats: bool = False,
        use_gravity: bool = False,
        use_attitude: bool = False,
        return_vicon: bool = False,
    ):
        """
        Args:
            root_dir: Root directory containing subfolders (handbag, iPhone 5, pocket)
            seq_len: Sequence length for windowing
            mask_rate: Fraction of data to mask (0-1)
            missing_mode: "random", "block", or "channel"
            split: "train" or "val"
            split_ratio: Fraction of files for training
            drift_scale: Scale of random walk drift to add (data augmentation)
            return_stats: If True, return normalization stats for denormalization
            use_gravity: If True, include gravity vector (3 dims) in input features
            use_attitude: If True, include attitude (roll/pitch/yaw) (3 dims) in input features
            return_vicon: If True, return Vicon ground truth (position + quaternion) for ATE computation
        """
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.mask_rate = mask_rate
        self.missing_mode = missing_mode
        self.eval_mode = eval_mode
        self.drift_scale = drift_scale
        self.return_stats = return_stats
        self.use_gravity = use_gravity
        self.use_attitude = use_attitude
        self.return_vicon = return_vicon
        
        # Calculate feature dimensions
        # Base: gyro(3) + acc(3) = 6
        self.base_dim = 6
        self.extra_dim = 0
        if use_gravity:
            self.extra_dim += 3
        if use_attitude:
            self.extra_dim += 3
        self.feature_dim = self.base_dim + self.extra_dim
        # Input dim: feature_dim + mask(feature_dim) + dt(1) = feature_dim * 2 + 1
        self.input_dim = self.feature_dim * 2 + 1
        
        self.sequences: List[dict] = []
        self._load_all_sequences(split, split_ratio)
        
        if len(self.sequences) == 0:
            raise ValueError(f"No sequences loaded for split={split}")
        
        print(f"[Dataset] Loaded {len(self.sequences)} sequences for {split}")
        print(f"[Dataset] Feature dim: {self.feature_dim} (base=6, gravity={use_gravity}, attitude={use_attitude})")
        print(f"[Dataset] Input dim: {self.input_dim}")
    
    def _load_all_sequences(self, split: str, split_ratio: float):
        """Load and split sequences by file (not by window)."""
        subfolders = ["handbag-1","handbag-2","handheld-1","handheld-2","pocket-1","pocket-2","running","slow walking","trolley","user-2"]
        all_file_pairs = []
        
        for subfolder in subfolders:
            folder = os.path.join(self.root_dir, subfolder)
            if not os.path.exists(folder):
                print(f"[Warning] Folder not found: {folder}")
                continue
            
            imu_files = sorted(glob.glob(os.path.join(folder, "imu*.csv")))
            for imu_path in imu_files:
                idx = os.path.splitext(os.path.basename(imu_path))[0].replace("imu", "")
                vi_path = os.path.join(folder, f"vi{idx}.csv")
                if os.path.exists(vi_path):
                    all_file_pairs.append((imu_path, vi_path))
        
        if len(all_file_pairs) == 0:
            raise ValueError(f"No valid file pairs found in {self.root_dir}")
        
        # Split by file to avoid data leakage
        n_files = len(all_file_pairs)
        n_train = int(n_files * split_ratio)
        
        if split == "train":
            file_pairs = all_file_pairs[:n_train]
        else:
            file_pairs = all_file_pairs[n_train:]
        
        print(f"[Dataset] Processing {len(file_pairs)} file pairs for {split}...")
        for imu_path, vi_path in file_pairs:
            self._process_file_pair(imu_path, vi_path)
    
    def _process_file_pair(self, imu_path: str, vi_path: str):
        """Process a single file pair into sequences."""
        try:
            # Load IMU data (headerless CSV)
            imu_df = pd.read_csv(imu_path, header=None)
            # Assign column names based on expected structure
            # Format: Time, attitude(roll/pitch/yaw), rotation_rate(x/y/z), gravity(x/y/z), 
            #         user_acc(x/y/z), magnetic_field(x/y/z)
            expected_cols = ["Time"] + ["att_roll", "att_pitch", "att_yaw"] + \
                           ["rotation_rate_x", "rotation_rate_y", "rotation_rate_z"] + \
                           ["grav_x", "grav_y", "grav_z"] + \
                           ["user_acc_x", "user_acc_y", "user_acc_z"] + \
                           ["mag_x", "mag_y", "mag_z"]
            
            if len(imu_df.columns) == len(expected_cols):
                imu_df.columns = expected_cols
            else:
                # Fallback: minimal columns
                imu_df.columns = ["Time"] + [f"col_{i}" for i in range(len(imu_df.columns) - 1)]
            
            # Load Vicon data (headerless CSV)
            vi_df = pd.read_csv(vi_path, header=None)
            vi_expected_cols = ["Time", "translation.x", "translation.y", "translation.z"] + \
                              ["qw", "qx", "qy", "qz"]
            if len(vi_df.columns) == len(vi_expected_cols):
                vi_df.columns = vi_expected_cols
            else:
                vi_df.columns = ["Time"] + [f"vi_col_{i}" for i in range(len(vi_df.columns) - 1)]
                
        except Exception as e:
            print(f"[Warning] Failed to load {imu_path}: {e}")
            return
        
        # Drop rows with NaN in critical columns
        required_imu_cols = ["Time"] + IMU_COLUMNS
        imu_df = imu_df.dropna(subset=[c for c in required_imu_cols if c in imu_df.columns])
        
        if len(imu_df) < self.seq_len:
            return
        
        # Extract data
        imu_time = imu_df["Time"].to_numpy(dtype=np.float64)
        
        # Extract base IMU values (rotation_rate + user_acc)
        try:
            base_values = imu_df[IMU_COLUMNS].to_numpy(dtype=np.float32)
        except KeyError:
            # Fallback: use columns by index
            # rotation_rate is columns 4-6, user_acc is columns 10-12
            try:
                base_values = imu_df.iloc[:, [4, 5, 6, 10, 11, 12]].to_numpy(dtype=np.float32)
            except:
                print(f"[Warning] Cannot extract IMU columns from {imu_path}")
                return
        
        # Extract optional features
        extra_features = []
        
        # Gravity (columns 7-9 in original, or by name)
        if self.use_gravity:
            try:
                gravity = imu_df[GRAVITY_COLUMNS].to_numpy(dtype=np.float32)
            except KeyError:
                try:
                    gravity = imu_df.iloc[:, [7, 8, 9]].to_numpy(dtype=np.float32)
                except:
                    print(f"[Warning] Cannot extract gravity columns from {imu_path}")
                    return
            extra_features.append(gravity)
        
        # Attitude (columns 1-3 in original, or by name)
        if self.use_attitude:
            try:
                attitude = imu_df[ATTITUDE_COLUMNS].to_numpy(dtype=np.float32)
            except KeyError:
                try:
                    attitude = imu_df.iloc[:, [1, 2, 3]].to_numpy(dtype=np.float32)
                except:
                    print(f"[Warning] Cannot extract attitude columns from {imu_path}")
                    return
            extra_features.append(attitude)
        
        # Combine all features: [gyro(3), acc(3), gravity(3)?, attitude(3)?]
        if extra_features:
            imu_values = np.concatenate([base_values] + extra_features, axis=1)
        else:
            imu_values = base_values
        
        # Align lengths
        min_len = min(len(imu_time), len(imu_values))
        imu_time = imu_time[:min_len]
        imu_values = imu_values[:min_len]
        
        # Physical unit conversion
        # Gyro: already in rad/s (typically)
        # Acc: G -> m/s²
        imu_values[:, 3:6] *= 9.81
        # Note: gravity is also in G units, convert if used
        if self.use_gravity:
            grav_start = 6  # After gyro(3) and acc(3)
            imu_values[:, grav_start:grav_start+3] *= 9.81
        
        # Extract Vicon ground truth if needed
        vicon_data = None
        if self.return_vicon:
            try:
                vicon_pos = vi_df[VICON_POS_COLUMNS].to_numpy(dtype=np.float32)
                vicon_quat = vi_df[VICON_QUAT_COLUMNS].to_numpy(dtype=np.float32)
                vicon_time = vi_df["Time"].to_numpy(dtype=np.float64)
                
                # Interpolate Vicon to IMU timestamps
                vicon_interp = np.zeros((len(imu_time), 7), dtype=np.float32)  # pos(3) + quat(4)
                for i in range(3):
                    vicon_interp[:, i] = np.interp(imu_time, vicon_time, vicon_pos[:, i])
                for i in range(4):
                    vicon_interp[:, 3+i] = np.interp(imu_time, vicon_time, vicon_quat[:, i])
                vicon_data = vicon_interp
            except Exception as e:
                print(f"[Warning] Cannot extract Vicon data from {vi_path}: {e}")
                vicon_data = None
        
        # Compute time intervals (preserve irregular sampling)
        dt = np.diff(imu_time, prepend=imu_time[0])
        dt = np.clip(dt, 1e-4, 1.0)  # Prevent extreme values
        
        # Robust normalization: Median Absolute Deviation (MAD)
        # More robust to outliers than standard Z-score
        imu_median = np.median(imu_values, axis=0)
        imu_mad = np.median(np.abs(imu_values - imu_median), axis=0) + 1e-6
        imu_norm = (imu_values - imu_median) / (1.4826 * imu_mad)  # MAD normalization
        
        # Create sliding windows with stride
        stride = max(1, self.seq_len // 2)
        for start in range(0, len(imu_norm) - self.seq_len + 1, stride):
            end = start + self.seq_len
            
            seq_dict = {
                "imu": torch.from_numpy(imu_norm[start:end]).float(),
                "dt": torch.from_numpy(dt[start:end]).float(),
                "stats": torch.tensor([*imu_median, *imu_mad], dtype=torch.float32),
            }
            
            # Add Vicon data if available
            if vicon_data is not None:
                seq_dict["vicon"] = torch.from_numpy(vicon_data[start:end]).float()
            
            self.sequences.append(seq_dict)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            inputs: (seq_len, input_dim) = [masked_imu(feature_dim), mask(feature_dim), dt(1)]
            targets: (seq_len, feature_dim) = ground truth IMU features
            mask: (seq_len, feature_dim) = 1 for observed, 0 for missing
            stats (optional): (feature_dim*2,) = [median(feature_dim), mad(feature_dim)]
            vicon (optional): (seq_len, 7) = [pos(3), quat(4)] Vicon ground truth
        """
        seq = self.sequences[idx]
        target_imu = seq["imu"]  # (seq_len, feature_dim) - Clean target
        dt = seq["dt"]    # (seq_len,)
        
        # Clone for input modification
        input_imu = target_imu.clone()
        
        # Apply physical drift augmentation (Random Walk) to INPUT ONLY
        # This creates a denoising task: Dirty Input -> Clean Target
        if self.drift_scale > 0 and not self.eval_mode:
            drift_noise = torch.randn_like(input_imu) * self.drift_scale
            drift = torch.cumsum(drift_noise, dim=0)
            input_imu = input_imu + drift
        
        # Apply missing pattern (fixed seed in eval mode for reproducibility)
        if self.eval_mode:
            rng_state = torch.get_rng_state()
            torch.manual_seed(idx)  # Deterministic mask based on idx
        
        mask = torch.ones_like(input_imu)
        feature_dim = input_imu.shape[-1]  # Dynamic feature dimension
        
        if self.missing_mode == "random":
            drop = torch.rand_like(input_imu) < self.mask_rate
            mask[drop] = 0.0
        elif self.missing_mode == "block":
            block_len = max(1, int(self.seq_len * self.mask_rate))
            max_start = max(1, self.seq_len - block_len + 1)
            for channel in range(feature_dim):
                start = torch.randint(0, max_start, (1,)).item()
                mask[start:start + block_len, channel] = 0.0
        elif self.missing_mode == "channel":
            n_mask = max(1, int(feature_dim * self.mask_rate))
            channels = torch.randperm(feature_dim)[:n_mask]
            mask[:, channels] = 0.0
        
        if self.eval_mode:
            torch.set_rng_state(rng_state)  # Restore RNG state
        
        imu_masked = input_imu * mask
        
        # Construct input: [masked_imu(feature_dim), mask(feature_dim), dt(1)]
        inputs = torch.cat([imu_masked, mask, dt.unsqueeze(-1)], dim=-1)
        
        # Build return tuple based on options
        result = [inputs, target_imu, mask]
        
        if self.return_stats:
            result.append(seq["stats"])
        
        if self.return_vicon and "vicon" in seq:
            result.append(seq["vicon"])
        
        return tuple(result)


def compute_ate(
    pred_acc: torch.Tensor,
    gt_pos: torch.Tensor,
    dt: torch.Tensor,
    initial_vel: Optional[torch.Tensor] = None,
    stats: Optional[torch.Tensor] = None,
    acc_indices: Tuple[int, int, int] = (3, 4, 5),
) -> dict:
    """
    Compute Absolute Trajectory Error (ATE) from predicted acceleration.
    
    This function performs double integration of acceleration to estimate
    position trajectory, then compares with Vicon ground truth.
    
    Args:
        pred_acc: (batch, seq_len, feature_dim) - Predicted IMU features (normalized)
        gt_pos: (batch, seq_len, 3) or (batch, seq_len, 7) - Vicon ground truth position
                If 7-dim, assumes [pos(3), quat(4)] format
        dt: (batch, seq_len) - Time intervals between samples
        initial_vel: (batch, 3) - Initial velocity, default zeros
        stats: (batch, feature_dim*2) - [median, mad] for denormalization
               If None, assumes pred_acc is already in physical units (m/s²)
        acc_indices: Tuple of indices for accelerometer channels in feature_dim
    
    Returns:
        dict containing:
            - ate: float - Absolute Trajectory Error (RMSE of position errors)
            - ate_per_axis: (3,) - ATE per axis (x, y, z)
            - max_drift: float - Maximum position drift
            - pred_trajectory: (batch, seq_len, 3) - Predicted positions
            - position_error: (batch, seq_len, 3) - Position errors
    """
    batch_size, seq_len, feature_dim = pred_acc.shape
    device = pred_acc.device
    
    # Extract acceleration (indices 3, 4, 5 for user_acc by default)
    acc = pred_acc[:, :, list(acc_indices)]  # (batch, seq_len, 3)
    
    # Denormalize if stats provided
    if stats is not None:
        # stats: (batch, feature_dim*2) = [median(feature_dim), mad(feature_dim)]
        half = stats.shape[-1] // 2
        median = stats[:, list(acc_indices)].unsqueeze(1)  # (batch, 1, 3)
        mad = stats[:, [i + half for i in acc_indices]].unsqueeze(1)  # (batch, 1, 3)
        acc = acc * (1.4826 * mad) + median  # Reverse MAD normalization
    
    # Extract ground truth position (first 3 dims if 7-dim)
    if gt_pos.shape[-1] == 7:
        gt_pos = gt_pos[:, :, :3]
    
    # Initialize velocity and position
    if initial_vel is None:
        vel = torch.zeros(batch_size, 3, device=device)
    else:
        vel = initial_vel.clone()
    
    # Double integration: acc -> vel -> pos
    # Using trapezoidal integration for better accuracy
    pred_pos = torch.zeros(batch_size, seq_len, 3, device=device)
    
    # First position from ground truth (we need a reference)
    pred_pos[:, 0, :] = gt_pos[:, 0, :]
    
    # Estimate initial velocity from ground truth (optional, for better alignment)
    # Using first few samples to estimate velocity
    if seq_len > 1:
        dt_init = dt[:, 1].unsqueeze(-1).clamp(min=1e-4)  # (batch, 1)
        vel = (gt_pos[:, 1, :] - gt_pos[:, 0, :]) / dt_init
    
    for t in range(1, seq_len):
        dt_t = dt[:, t].unsqueeze(-1).clamp(min=1e-4)  # (batch, 1)
        
        # Trapezoidal integration for velocity: v(t) = v(t-1) + 0.5*(a(t-1) + a(t))*dt
        acc_avg = 0.5 * (acc[:, t-1, :] + acc[:, t, :])
        vel = vel + acc_avg * dt_t
        
        # Trapezoidal integration for position
        vel_avg = vel  # Simplified: could use 0.5*(v(t-1) + v(t))
        pred_pos[:, t, :] = pred_pos[:, t-1, :] + vel_avg * dt_t
    
    # Compute position errors
    pos_error = pred_pos - gt_pos  # (batch, seq_len, 3)
    
    # ATE: RMSE of position errors
    ate_per_sample = torch.sqrt((pos_error ** 2).sum(dim=-1))  # (batch, seq_len)
    ate = ate_per_sample.mean().item()
    
    # ATE per axis
    ate_per_axis = torch.sqrt((pos_error ** 2).mean(dim=(0, 1))).cpu().numpy()  # (3,)
    
    # Maximum drift
    max_drift = ate_per_sample.max().item()
    
    return {
        "ate": ate,
        "ate_per_axis": ate_per_axis,
        "max_drift": max_drift,
        "pred_trajectory": pred_pos,
        "position_error": pos_error,
    }


def compute_relative_trajectory_error(
    pred_acc: torch.Tensor,
    gt_pos: torch.Tensor,
    dt: torch.Tensor,
    delta_t: float = 1.0,
    stats: Optional[torch.Tensor] = None,
    acc_indices: Tuple[int, int, int] = (3, 4, 5),
) -> dict:
    """
    Compute Relative Trajectory Error (RTE) over fixed time intervals.
    
    RTE measures drift over short segments, which is less affected by
    long-term integration drift and better reflects local prediction quality.
    
    Args:
        pred_acc: (batch, seq_len, feature_dim) - Predicted IMU features
        gt_pos: (batch, seq_len, 3 or 7) - Vicon ground truth
        dt: (batch, seq_len) - Time intervals
        delta_t: Target time interval for segments (seconds)
        stats: Normalization stats for denormalization
        acc_indices: Indices for accelerometer channels
    
    Returns:
        dict containing:
            - rte: float - Relative Trajectory Error (mean segment error)
            - rte_std: float - Standard deviation of segment errors
            - segment_errors: list - Error for each segment
    """
    batch_size, seq_len, _ = pred_acc.shape
    device = pred_acc.device
    
    # First compute full trajectory
    ate_result = compute_ate(pred_acc, gt_pos, dt, stats=stats, acc_indices=acc_indices)
    pred_pos = ate_result["pred_trajectory"]
    
    # Extract ground truth position
    if gt_pos.shape[-1] == 7:
        gt_pos = gt_pos[:, :, :3]
    
    # Find segment boundaries based on cumulative time
    cum_time = dt.cumsum(dim=1)  # (batch, seq_len)
    
    segment_errors = []
    for b in range(batch_size):
        t = 0
        while t < seq_len - 1:
            # Find end of segment
            target_time = cum_time[b, t] + delta_t
            t_end = (cum_time[b, t:] <= target_time).sum().item() + t
            t_end = min(t_end, seq_len - 1)
            
            if t_end <= t:
                break
            
            # Compute relative error for this segment
            # Align start points
            pred_delta = pred_pos[b, t_end, :] - pred_pos[b, t, :]
            gt_delta = gt_pos[b, t_end, :] - gt_pos[b, t, :]
            
            error = torch.sqrt(((pred_delta - gt_delta) ** 2).sum()).item()
            segment_errors.append(error)
            
            t = t_end
    
    if len(segment_errors) == 0:
        return {"rte": 0.0, "rte_std": 0.0, "segment_errors": []}
    
    rte = np.mean(segment_errors)
    rte_std = np.std(segment_errors)
    
    return {
        "rte": rte,
        "rte_std": rte_std,
        "segment_errors": segment_errors,
    }
