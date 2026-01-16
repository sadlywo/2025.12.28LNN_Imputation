"""Benchmark PyTorch (CPU/GPU) and ONNXRuntime (CPU/GPU) inference time.

Outputs a CSV table with timing statistics and prints a summary table.
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models import build_model


def load_data(path: str):
    d = np.load(path)
    return d["inputs"].astype(np.float32)


def detect_model_name(ckpt_keys):
    if any("encoder.layers" in k for k in ckpt_keys):
        return "transformer"
    if any(k.startswith("rnn.") for k in ckpt_keys):
        return "gru"
    if any("physics_coupling" in k for k in ckpt_keys):
        return "physics"
    if any("cfc." in k for k in ckpt_keys):
        return "cfc"
    return "cfc"


def infer_hidden_units(ckpt):
    if "cfc.lstm.input_map.weight" in ckpt:
        # LSTM input_map has shape (4*hidden_units, input_dim)
        return ckpt["cfc.lstm.input_map.weight"].shape[0] // 4
    if "input_proj.weight" in ckpt:
        return ckpt["input_proj.weight"].shape[0]
    if "rnn.weight_ih_l0" in ckpt:
        return ckpt["rnn.weight_ih_l0"].shape[0] // 3
    for k in ckpt:
        if "cfc.rnn_cell.layer_0.ff1.weight" in k:
            return ckpt[k].shape[0]
    return 64


def time_pytorch(model, inputs, batch_size=32, warmup=10, runs=50, device="cpu"):
    model.to(device)
    model.eval()
    n = inputs.shape[0]
    batch_idxs = [np.arange(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            for idxs in batch_idxs:
                batch = torch.from_numpy(inputs[idxs]).to(device)
                _ = model(batch)
        sync()

        timings = []
        for _ in range(runs):
            t0 = time.perf_counter()
            for idxs in batch_idxs:
                batch = torch.from_numpy(inputs[idxs]).to(device)
                _ = model(batch)
            sync()
            t1 = time.perf_counter()
            timings.append(t1 - t0)

    return np.array(timings)


def time_onnx(onnx_path, inputs, batch_size=32, warmup=10, runs=50, provider="CPU"):
    providers = ["CPUExecutionProvider"] if provider == "CPU" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    actual_providers = sess.get_providers()
    if provider == "CUDA" and "CUDAExecutionProvider" not in actual_providers:
        raise RuntimeError("CUDAExecutionProvider not available; fell back to CPU")
    input_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    n = inputs.shape[0]
    batch_idxs = [np.arange(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]

    for _ in range(warmup):
        for idxs in batch_idxs:
            batch = inputs[idxs]
            sess.run(out_names, {input_name: batch})

    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for idxs in batch_idxs:
            batch = inputs[idxs]
            sess.run(out_names, {input_name: batch})
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    return np.array(timings)


def summarize(label, timings, num_samples, runs):
    total = float(timings.sum())
    stats = {
        "method": label,
        "runs": runs,
        "total_time_s": total,
        "mean_s": float(timings.mean()),
        "median_s": float(np.median(timings)),
        "std_s": float(timings.std()),
        "samples_per_sec": float((num_samples * runs) / total) if total > 0 else 0.0,
    }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/test_samples.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--hidden-units", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    inputs = load_data(args.data)
    n = inputs.shape[0]

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_name = args.model_name or detect_model_name(ckpt.keys())
    hidden_units = args.hidden_units or infer_hidden_units(ckpt)

    model = build_model(model_name, input_dim=13, hidden_dim=hidden_units, output_dim=6)
    try:
        model.load_state_dict(ckpt)
    except Exception as e:
        print("Warning: strict load failed, trying non-strict load:", e)
        model.load_state_dict(ckpt, strict=False)

    results = []

    # PyTorch CPU
    cpu_timings = time_pytorch(model, inputs, batch_size=args.batch_size, warmup=args.warmup, runs=args.runs, device="cpu")
    results.append(summarize("pytorch_cpu", cpu_timings, n, args.runs))

    # PyTorch GPU (if available)
    if torch.cuda.is_available():
        gpu_timings = time_pytorch(model, inputs, batch_size=args.batch_size, warmup=args.warmup, runs=args.runs, device="cuda")
        results.append(summarize("pytorch_gpu", gpu_timings, n, args.runs))
    else:
        print("CUDA not available; skipping PyTorch GPU benchmark")

    # ONNX CPU
    onnx_cpu = time_onnx(args.onnx, inputs, batch_size=args.batch_size, warmup=args.warmup, runs=args.runs, provider="CPU")
    results.append(summarize("onnx_cpu", onnx_cpu, n, args.runs))

    # ONNX GPU (if available)
    try:
        onnx_gpu = time_onnx(args.onnx, inputs, batch_size=args.batch_size, warmup=args.warmup, runs=args.runs, provider="CUDA")
        results.append(summarize("onnx_gpu", onnx_gpu, n, args.runs))
    except Exception as e:
        print("ONNX GPU benchmark failed or provider unavailable:", e)

    df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "benchmark_table.csv")
    try:
        df.to_csv(out_csv, index=False)
        print("Saved benchmark table to", out_csv)
    except PermissionError:
        alt_csv = os.path.join(args.out_dir, f"benchmark_table_{int(time.time())}.csv")
        df.to_csv(alt_csv, index=False)
        print("benchmark_table.csv is locked; saved to", alt_csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
