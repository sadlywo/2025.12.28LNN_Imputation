# ONNX inference benchmark

This folder contains scripts to generate synthetic IMU test data with missing values and to benchmark an ONNX model exported from this repo.

Files:

- `make_test_data.py` — generate `data/test_samples.npz` containing inputs/targets/mask
- `run_onnx_inference.py` — load the ONNX model and run repeated inference to measure timings
- `compare_with_pytorch.py` — optional baseline using the PyTorch model and checkpoint
- `benchmark_all.py` — one-click CPU/GPU/ONNX benchmark table
- `requirements.txt` — Python packages needed for this folder

Quick start (from repo root, Windows PowerShell):

```powershell
# create test samples (256 sequences)
python onnx_test/make_test_data.py --out onnx_test/data/test_samples.npz --num-samples 256 --seq-len 50

# run ONNX benchmark (assumes best_model.onnx is at repo root)
python onnx_test/run_onnx_inference.py --onnx best_model.onnx --data onnx_test/data/test_samples.npz --batch-size 32 --runs 200 --warmup 20 --out-dir onnx_test/results

# (optional) compare with PyTorch checkpoint
python onnx_test/compare_with_pytorch.py --checkpoint best_model.pt --data onnx_test/data/test_samples.npz --batch-size 32 --runs 100 --out-dir onnx_test/results
```

Output:

- `onnx_test/results/onnx_inference_stats.csv` — timing summary for ONNX
- `onnx_test/results/onnx_outputs_sample.npz` — sample outputs
- `onnx_test/results/pytorch_inference_stats.csv` — PyTorch baseline timings (if run)
- `onnx_test/results/benchmark_table.csv` — CPU/GPU/ONNX benchmark table (if run)

Notes:

- `onnxruntime` defaults to CPU provider here; if you have ONNX Runtime GPU installed you can pass `--provider CUDA` to `run_onnx_inference.py`.
- The synthetic data generator uses mask=1 for observed values and zero-imputes missing channels.

Benchmark table (CPU/GPU/ONNX in one command):

```powershell
python onnx_test/benchmark_all.py --onnx best_model.onnx --checkpoint best_model.pt --data onnx_test/data/test_samples.npz --batch-size 32 --runs 100 --warmup 10 --out-dir onnx_test/results
```
