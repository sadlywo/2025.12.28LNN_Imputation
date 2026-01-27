"""Optimized dataset for CfC-based IMU imputation."""
import os
import glob
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Column definitions
IMU_COLUMNS = [
    "rotation_rate_x", "rotation_rate_y", "rotation_rate_z",
    "user_acc_x", "user_acc_y", "user_acc_z"
]
VICON_POS_COLUMNS = ["translation.x", "translation.y", "translation.z"]


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
        """
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.mask_rate = mask_rate
        self.missing_mode = missing_mode
        self.eval_mode = eval_mode
        self.drift_scale = drift_scale
        
        self.sequences: List[dict] = []
        self._load_all_sequences(split, split_ratio)
        
        if len(self.sequences) == 0:
            raise ValueError(f"No sequences loaded for split={split}")
        
        print(f"[Dataset] Loaded {len(self.sequences)} sequences for {split}")
    
    def _load_all_sequences(self, split: str, split_ratio: float):
        """Load and split sequences by file (not by window)."""
        subfolders = ["device-iPhone 5", "device-iPhone 6","handbag-1","handbag-2","handheld-1","handheld-2","handheld-3","handheld-4","handheld-5","pocket-1","pocket-2","running","slow walking","trolley","user-1","user-2","user-3","user-4"]
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
            expected_cols = ["Time"] + ["att_" + s for s in ["roll", "pitch", "yaw"]] + \
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
        
        # Extract IMU values (rotation_rate + user_acc)
        try:
            imu_values = imu_df[IMU_COLUMNS].to_numpy(dtype=np.float32)
        except KeyError:
            # Fallback: use columns by index
            # Assuming rotation_rate is columns 4-6, user_acc is columns 10-12
            try:
                imu_values = imu_df.iloc[:, [4, 5, 6, 10, 11, 12]].to_numpy(dtype=np.float32)
            except:
                print(f"[Warning] Cannot extract IMU columns from {imu_path}")
                return
        
        # Align lengths
        min_len = min(len(imu_time), len(imu_values))
        imu_time = imu_time[:min_len]
        imu_values = imu_values[:min_len]
        
        # Physical unit conversion
        # Gyro: already in rad/s (typically)
        # Acc: G -> m/s²
        imu_values[:, 3:6] *= 9.81
        
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
            self.sequences.append(seq_dict)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            inputs: (seq_len, 13) = [masked_imu(6), mask(6), dt(1)]
            targets: (seq_len, 6) = ground truth IMU
            mask: (seq_len, 6) = 1 for observed, 0 for missing
        """
        seq = self.sequences[idx]
        target_imu = seq["imu"]  # (seq_len, 6) - Clean target
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
        if self.missing_mode == "random":
            drop = torch.rand_like(input_imu) < self.mask_rate
            mask[drop] = 0.0
        elif self.missing_mode == "block":
            block_len = max(1, int(self.seq_len * self.mask_rate))
            max_start = max(1, self.seq_len - block_len + 1)
            for channel in range(6):
                start = torch.randint(0, max_start, (1,)).item()
                mask[start:start + block_len, channel] = 0.0
        elif self.missing_mode == "channel":
            n_mask = max(1, int(6 * self.mask_rate))
            channels = torch.randperm(6)[:n_mask]
            mask[:, channels] = 0.0
        
        if self.eval_mode:
            torch.set_rng_state(rng_state)  # Restore RNG state
        
        imu_masked = input_imu * mask
        
        # Construct input: [masked_imu(6), mask(6), dt(1)]
        inputs = torch.cat([imu_masked, mask, dt.unsqueeze(-1)], dim=-1)
        
        return inputs, target_imu, mask
