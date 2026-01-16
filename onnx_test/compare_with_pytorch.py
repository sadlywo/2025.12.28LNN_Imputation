"""Optional: run same test with PyTorch model for baseline comparison.

This script loads the checkpoint with `build_model` and runs inference on CPU.
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models import build_model


def load_data(path):
    d = np.load(path)
    return d["inputs"].astype(np.float32), d.get("targets", None), d.get("mask", None)


def run_pytorch(model, inputs, batch_size=32, warmup=10, runs=50, device='cpu'):
    model.to(device)
    model.eval()
    n = inputs.shape[0]
    batch_idxs = [np.arange(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            for idxs in batch_idxs:
                batch = torch.from_numpy(inputs[idxs]).to(device)
                _ = model(batch)

        timings = []
        for r in range(runs):
            t0 = time.perf_counter()
            for idxs in batch_idxs:
                batch = torch.from_numpy(inputs[idxs]).to(device)
                _ = model(batch)
            t1 = time.perf_counter()
            timings.append(t1 - t0)

    return np.array(timings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/test_samples.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--model-name", type=str, default="cfc")
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    inputs, targets, mask = load_data(args.data)

    model = build_model(args.model_name, input_dim=13, hidden_dim=args.hidden_units, output_dim=6)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    # Try flexible load
    try:
        model.load_state_dict(ckpt)
    except Exception as e:
        print("Warning: strict load failed, trying non-strict load:", e)
        model.load_state_dict(ckpt, strict=False)

    timings = run_pytorch(model, inputs, batch_size=args.batch_size, warmup=args.warmup, runs=args.runs, device='cpu')

    stats = {
        'runs': len(timings),
        'total_time_s': float(timings.sum()),
        'mean_s': float(timings.mean()),
        'median_s': float(np.median(timings)),
        'std_s': float(timings.std())
    }
    pd.DataFrame([stats]).to_csv(os.path.join(args.out_dir, 'pytorch_inference_stats.csv'), index=False)
    print('Saved PyTorch stats to', os.path.join(args.out_dir, 'pytorch_inference_stats.csv'))


if __name__ == '__main__':
    main()
