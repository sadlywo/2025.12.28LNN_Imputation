from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from experiment_bidirectional_lnn_residual import (
    BidirectionalLNNResidual,
    ResidualBiLSTM,
    ResidualHybridBiLNNBiLSTM,
)


class GRUImputer(torch.nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 128, output_dim: int = 6):
        super().__init__()
        self.rnn = torch.nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        self.uncertainty_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, output_dim),
            torch.nn.Softplus(),
        )

    def forward(self, x: torch.Tensor):
        h, _ = self.rnn(x)
        pred = self.head(h)
        uncert = self.uncertainty_head(h)
        return pred, uncert


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x[:, :, -1]
        t = torch.cumsum(dt, dim=1)
        b, t_len = t.shape
        device = t.device
        pe = torch.zeros(b, t_len, self.d_model, device=device)
        position = t.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device) * (-np.log(10000.0) / self.d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class TransformerImputer(torch.nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 128, output_dim: int = 6, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.posenc = PositionalEncoding(hidden_dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        self.uncertainty_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, output_dim),
            torch.nn.Softplus(),
        )

    def forward(self, x: torch.Tensor):
        z = self.input_proj(x)
        z = z + self.posenc(x)
        h = self.encoder(z)
        pred = self.head(h)
        uncert = self.uncertainty_head(h)
        return pred, uncert


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _state_dict_size_mb(state_dict: Dict[str, torch.Tensor]) -> float:
    total = 0
    for v in state_dict.values():
        total += v.nelement() * v.element_size()
    return total / (1024 * 1024)


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _infer_transformer_layers(state: Dict[str, torch.Tensor], default_layers: int = 2) -> int:
    idxs = []
    for k in state.keys():
        if k.startswith("encoder.layers."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                idxs.append(int(parts[2]))
    return (max(idxs) + 1) if idxs else default_layers


def _infer_lstm_layers_from_state(state: Dict[str, torch.Tensor], prefix: str) -> int:
    layer_ids = set()
    key_prefix = f"{prefix}.lstm.weight_ih_l"
    for k in state.keys():
        if k.startswith(key_prefix):
            tail = k[len(key_prefix):]
            num = ""
            for ch in tail:
                if ch.isdigit():
                    num += ch
                else:
                    break
            if num:
                layer_ids.add(int(num))
    return max(layer_ids) + 1 if layer_ids else 2


def _build_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    name = ckpt_path.name
    state = torch.load(ckpt_path, map_location=device)

    if "Bidirectional_LNN" in name:
        hidden = int(state["head.0.weight"].shape[0])
        out_dim = int(state["head.2.weight"].shape[0])
        feature_dim = out_dim
        if "forward_lnn.lstm.input_map.weight" in state:
            residual_input_dim = int(state["forward_lnn.lstm.input_map.weight"].shape[1])
        else:
            residual_input_dim = int(state["forward_lnn.rnn_cell.ff1.weight"].shape[1])
        input_dim = residual_input_dim - feature_dim - 4
        model = BidirectionalLNNResidual(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_units=hidden,
            output_dim=out_dim,
        )
        model_input_dim = input_dim
    elif "Hybrid_BiLNN_BiLSTM" in name:
        lnn_hidden = int(state["bi_lnn.head.0.weight"].shape[0])
        out_dim = int(state["bi_lnn.head.2.weight"].shape[0])
        feature_dim = out_dim
        if "bi_lnn.forward_lnn.lstm.input_map.weight" in state:
            residual_input_dim = int(state["bi_lnn.forward_lnn.lstm.input_map.weight"].shape[1])
        else:
            residual_input_dim = int(state["bi_lnn.forward_lnn.rnn_cell.ff1.weight"].shape[1])
        input_dim = residual_input_dim - feature_dim - 4
        lstm_hidden = int(state["bi_lstm.backbone.lstm.weight_hh_l0"].shape[1])
        lstm_layers = _infer_lstm_layers_from_state(state, prefix="bi_lstm.backbone")
        model = ResidualHybridBiLNNBiLSTM(
            input_dim=input_dim,
            feature_dim=feature_dim,
            lnn_hidden=lnn_hidden,
            lstm_hidden=lstm_hidden,
            output_dim=out_dim,
            lstm_layers=lstm_layers,
        )
        model_input_dim = input_dim
    elif "BiLSTM" in name:
        hidden = int(state["backbone.lstm.weight_hh_l0"].shape[1])
        out_dim = int(state["backbone.head.2.weight"].shape[0])
        feature_dim = out_dim
        residual_input_dim = int(state["backbone.lstm.weight_ih_l0"].shape[1])
        input_dim = residual_input_dim - feature_dim - 4
        lstm_layers = _infer_lstm_layers_from_state(state, prefix="backbone")
        model = ResidualBiLSTM(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden,
            output_dim=out_dim,
            num_layers=lstm_layers,
        )
        model_input_dim = input_dim
    elif "gru" in name.lower():
        in_dim = int(state["rnn.weight_ih_l0"].shape[1])
        hidden = int(state["rnn.weight_hh_l0"].shape[1])
        out_dim = int(state["head.2.weight"].shape[0])
        model = GRUImputer(input_dim=in_dim, hidden_dim=hidden, output_dim=out_dim)
        model_input_dim = in_dim
    elif "transformer" in name.lower():
        in_dim = int(state["input_proj.weight"].shape[1])
        hidden = int(state["input_proj.weight"].shape[0])
        out_dim = int(state["head.2.weight"].shape[0])
        layers = _infer_transformer_layers(state, default_layers=2)
        model = TransformerImputer(
            input_dim=in_dim,
            hidden_dim=hidden,
            output_dim=out_dim,
            nhead=4,
            nlayers=layers,
        )
        model_input_dim = in_dim
    else:
        raise ValueError(f"Unsupported checkpoint: {name}")

    model.load_state_dict(state)
    model = model.to(device).eval()
    return model, model_input_dim, state


def _benchmark_forward(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    warmup_iters: int,
    bench_iters: int,
    device: torch.device,
) -> Tuple[float, float]:
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(bench_iters):
            _ = model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
    avg_ms = (t1 - t0) * 1000.0 / bench_iters
    samples_per_sec = sample_input.shape[0] / ((t1 - t0) / bench_iters)
    return avg_ms, samples_per_sec


def run_benchmark(
    checkpoint_dir: str = "results/Downstream_Tra_Plot",
    output_dir: str = "results/Downstream_Tra_Plot",
    batch_size: int = 32,
    seq_len: int = 30,
    warmup_iters: int = 20,
    bench_iters: int = 100,
    use_cuda: bool = True,
):
    ckpt_dir = Path(checkpoint_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    ckpts = sorted(ckpt_dir.glob("best_model_*.pt"))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    rows: List[dict] = []
    for ckpt in ckpts:
        model, input_dim, state = _build_model_from_checkpoint(ckpt, device=device)
        sample_input = torch.randn(batch_size, seq_len, input_dim, device=device)
        avg_ms, samples_per_sec = _benchmark_forward(
            model=model,
            sample_input=sample_input,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            device=device,
        )
        rows.append(
            {
                "model_file": ckpt.name,
                "model_type": ckpt.stem.replace("best_model_", ""),
                "device": str(device),
                "batch_size": batch_size,
                "seq_len": seq_len,
                "input_dim": input_dim,
                "parameters": int(_count_parameters(model)),
                "state_dict_size_mb": float(_state_dict_size_mb(state)),
                "checkpoint_file_size_mb": float(_file_size_mb(ckpt)),
                "avg_forward_ms_per_batch": float(avg_ms),
                "throughput_samples_per_sec": float(samples_per_sec),
            }
        )

    df = pd.DataFrame(rows).sort_values("avg_forward_ms_per_batch", ascending=True).reset_index(drop=True)
    csv_path = out_dir / f"inference_speed_benchmark_{timestamp}.csv"
    xlsx_path = out_dir / f"inference_speed_benchmark_{timestamp}.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    print(df.to_string(index=False))
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {xlsx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark forward inference speed for all checkpoints in Downstream_Tra_Plot")
    parser.add_argument("--checkpoint_dir", type=str, default="results/Downstream_Tra_Plot")
    parser.add_argument("--output_dir", type=str, default="results/Downstream_Tra_Plot")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--bench_iters", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run_benchmark(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        use_cuda=not args.cpu,
    )
