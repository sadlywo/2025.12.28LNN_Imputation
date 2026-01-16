"""Generate synthetic IMU test samples with missing values and save to .npz

Data format matches model input: [masked_imu(6), mask(6), dt(1)] -> 13 features

Usage:
    python make_test_data.py --out onnx_test/data/test_samples.npz --num-samples 256 --seq-len 50
"""
import argparse
import os
import numpy as np


def make_samples(num_samples: int = 256, seq_len: int = 50, missing_rate: float = 0.3, seed: int = 1234):
    rng = np.random.default_rng(seed)
    # masked_imu: gyro(3) + acc(3)
    imu = rng.normal(scale=0.5, size=(num_samples, seq_len, 6)).astype(np.float32)

    # mask: 1 means observed, 0 means missing
    mask = (rng.random((num_samples, seq_len, 6)) > missing_rate).astype(np.float32)

    # zero-out missing values in masked_imu
    masked_imu = imu * mask

    # dt channel: constant 1/sampling_rate (e.g., 0.01s for 100Hz)
    dt = np.full((num_samples, seq_len, 1), 0.01, dtype=np.float32)

    # concat: [masked_imu, mask, dt]
    inputs = np.concatenate([masked_imu, mask, dt], axis=-1)

    # Also create 'targets' (ground truth) for optional comparison
    targets = imu.astype(np.float32)

    return inputs, targets, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="onnx_test/data/test_samples.npz")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    inputs, targets, mask = make_samples(args.num_samples, args.seq_len, args.missing_rate, args.seed)
    np.savez_compressed(args.out, inputs=inputs, targets=targets, mask=mask)
    print(f"Saved test samples to {args.out}")


if __name__ == '__main__':
    main()
