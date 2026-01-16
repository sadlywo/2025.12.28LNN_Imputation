"""Run ONNX inference on IMU test samples and measure throughput/latency.

Produces a CSV with timing statistics and saves a sample output array for inspection.

Usage example:
  python run_onnx_inference.py --onnx ../best_model.onnx --data data/test_samples.npz --batch-size 32 --runs 200 --warmup 20
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import onnxruntime as ort


def load_data(path):
    d = np.load(path)
    return d["inputs"].astype(np.float32), d.get("targets", None), d.get("mask", None)


def measure_session(session, inputs, batch_size=32, warmup=10, runs=100):
    input_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]

    n = inputs.shape[0]
    # iterate in batches
    batch_idxs = [np.arange(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]

    # Warmup
    for _ in range(max(1, warmup)):
        for idxs in batch_idxs:
            batch = inputs[idxs]
            session.run(out_names, {input_name: batch})

    # Timed runs: run 'runs' iterations over the entire dataset
    timings = []
    outputs_sample = None
    for r in range(runs):
        t0 = time.perf_counter()
        for i, idxs in enumerate(batch_idxs):
            batch = inputs[idxs]
            out = session.run(out_names, {input_name: batch})
            if r == 0 and i == 0:
                outputs_sample = [o.copy() for o in out]
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    timings = np.array(timings)
    return timings, out_names, outputs_sample


def summarize_and_save(timings, batch_size, out_csv):
    # timings are full-epoch times. Compute per-sample latency
    stats = {
        "runs": len(timings),
        "total_time_s": float(timings.sum()),
        "mean_s": float(timings.mean()),
        "median_s": float(np.median(timings)),
        "std_s": float(timings.std()),
        "min_s": float(timings.min()),
        "max_s": float(timings.max()),
    }
    df = pd.DataFrame([stats])
    df.to_csv(out_csv, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--data", type=str, default="data/test_samples.npz", help=".npz with inputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--provider", type=str, default="CPU", choices=["CPU","CUDA"], help="Execution provider")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading data from {args.data}...")
    inputs, targets, mask = load_data(args.data)
    print(f"Inputs shape: {inputs.shape}")

    # create session
    providers = ["CPUExecutionProvider"] if args.provider == "CPU" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print(f"Creating ONNXRuntime session with providers: {providers}")
    sess = ort.InferenceSession(args.onnx, providers=providers)

    print("Running benchmark...")
    timings, out_names, outputs_sample = measure_session(sess, inputs, batch_size=args.batch_size, warmup=args.warmup, runs=args.runs)

    # Save stats
    out_csv = os.path.join(args.out_dir, "onnx_inference_stats.csv")
    df = summarize_and_save(timings, args.batch_size, out_csv)
    print("Saved stats to", out_csv)

    # Save a small sample of outputs for inspection
    out_sample_file = os.path.join(args.out_dir, "onnx_outputs_sample.npz")
    np.savez_compressed(out_sample_file, outputs=outputs_sample, out_names=out_names)
    print("Saved sample outputs to", out_sample_file)

    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
