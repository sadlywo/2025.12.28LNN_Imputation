from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset import CfCIMUDataset


class GRUImputer(nn.Module):
    def __init__(self, input_dim: int = 43, hidden_dim: int = 128, output_dim: int = 6):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.rnn(x)
        pred = self.head(h)
        uncert = self.uncertainty_head(h)
        return pred, uncert


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x[:, :, 12]
        t = torch.cumsum(dt, dim=1)
        b, t_len = t.shape
        device = t.device
        pe = torch.zeros(b, t_len, self.d_model, device=device)
        position = t.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device) * (-np.log(10000.0) / self.d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class TransformerImputer(nn.Module):
    def __init__(
        self,
        input_dim: int = 43,
        hidden_dim: int = 128,
        output_dim: int = 6,
        nhead: int = 4,
        nlayers: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.posenc = PositionalEncoding(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.input_proj(x)
        z = z + self.posenc(x)
        h = self.encoder(z)
        pred = self.head(h)
        uncert = self.uncertainty_head(h)
        return pred, uncert


def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _state_dict_size_mb(state_dict: Dict[str, torch.Tensor]) -> float:
    total = 0
    for v in state_dict.values():
        total += v.nelement() * v.element_size()
    return total / (1024 * 1024)


def _checkpoint_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _missing_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    missing = 1.0 - mask
    mse_missing = ((pred - target) ** 2 * missing).sum() / (missing.sum() + 1e-8)
    return torch.sqrt(mse_missing + 1e-12)


def _train_one_epoch(model: nn.Module, loader, optimizer, device: torch.device) -> float:
    model.train()
    losses = []
    for inputs, targets, mask in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        pred, _ = model(inputs)
        loss = _missing_rmse(pred, targets, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def _evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, targets, mask in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            pred, _ = model(inputs)
            loss = _missing_rmse(pred, targets, mask)
            losses.append(float(loss.item()))
    return {"rmse_missing": float(np.mean(losses))}


def _benchmark_forward(model: nn.Module, sample_input: torch.Tensor, warmup: int, iters: int, device: torch.device):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
    avg_ms = (t1 - t0) * 1000.0 / iters
    throughput = sample_input.shape[0] / ((t1 - t0) / iters)
    return float(avg_ms), float(throughput)


def run_training(
    output_dir: str = "results/train_43d_gru_transformer",
    epochs: int = 20,
    batch_size: int = 32,
    seq_len: int = 30,
    mask_rate: float = 0.3,
    missing_mode: str = "random",
    hidden_dim: int = 128,
    transformer_nhead: int = 4,
    transformer_nlayers: int = 2,
    lr: float = 1e-3,
    warmup_iters: int = 20,
    bench_iters: int = 100,
):
    _seed_all(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_ds = CfCIMUDataset(
        root_dir="Oxford Dataset",
        seq_len=seq_len,
        mask_rate=mask_rate,
        missing_mode=missing_mode,
        split="train",
        split_ratio=0.8,
        val_ratio=0.1,
        eval_mode=False,
        include_window_features=True,
    )
    val_ds = CfCIMUDataset(
        root_dir="Oxford Dataset",
        seq_len=seq_len,
        mask_rate=mask_rate,
        missing_mode=missing_mode,
        split="val",
        split_ratio=0.8,
        val_ratio=0.1,
        eval_mode=True,
        include_window_features=True,
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    sample_batch = next(iter(val_loader))[0].to(device)

    models = {
        "GRU_43d": GRUImputer(input_dim=train_ds.input_dim, hidden_dim=hidden_dim, output_dim=6),
        "Transformer_43d": TransformerImputer(
            input_dim=train_ds.input_dim,
            hidden_dim=hidden_dim,
            output_dim=6,
            nhead=transformer_nhead,
            nlayers=transformer_nlayers,
        ),
    }

    summary_rows: List[dict] = []
    history_rows: List[dict] = []

    for name, model in models.items():
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        best_val = float("inf")
        best_epoch = -1
        best_path = save_dir / f"best_model_{name}_{timestamp}.pt"
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            train_rmse = _train_one_epoch(model, train_loader, optimizer, device)
            val_rmse = _evaluate(model, val_loader, device)["rmse_missing"]
            history_rows.append(
                {
                    "model": name,
                    "epoch": epoch,
                    "train_rmse_missing": train_rmse,
                    "val_rmse_missing": val_rmse,
                }
            )
            if val_rmse < best_val:
                best_val = val_rmse
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
        train_time = time.time() - t0

        best_state = torch.load(best_path, map_location=device)
        model.load_state_dict(best_state)
        model.eval()
        avg_ms, throughput = _benchmark_forward(
            model=model,
            sample_input=sample_batch,
            warmup=warmup_iters,
            iters=bench_iters,
            device=device,
        )
        summary_rows.append(
            {
                "model": name,
                "input_dim": train_ds.input_dim,
                "best_epoch": best_epoch,
                "best_val_rmse_missing": float(best_val),
                "parameters": int(_count_params(model)),
                "state_dict_size_mb": float(_state_dict_size_mb(best_state)),
                "checkpoint_file_size_mb": float(_checkpoint_size_mb(best_path)),
                "avg_forward_ms_per_batch": avg_ms,
                "throughput_samples_per_sec": throughput,
                "train_time_sec": float(train_time),
                "checkpoint_path": str(best_path),
                "device": str(device),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("avg_forward_ms_per_batch", ascending=True)
    history_df = pd.DataFrame(history_rows)

    summary_csv = save_dir / f"summary_43d_gru_transformer_{timestamp}.csv"
    history_csv = save_dir / f"history_43d_gru_transformer_{timestamp}.csv"
    config_json = save_dir / f"config_43d_gru_transformer_{timestamp}.json"
    summary_df.to_csv(summary_csv, index=False)
    history_df.to_csv(history_csv, index=False)

    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "mask_rate": mask_rate,
        "missing_mode": missing_mode,
        "hidden_dim": hidden_dim,
        "transformer_nhead": transformer_nhead,
        "transformer_nlayers": transformer_nlayers,
        "lr": lr,
        "warmup_iters": warmup_iters,
        "bench_iters": bench_iters,
        "input_dim": train_ds.input_dim,
        "device": str(device),
    }
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(summary_df.to_string(index=False))
    print(f"[Saved] {summary_csv}")
    print(f"[Saved] {history_csv}")
    print(f"[Saved] {config_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRU/Transformer with 43-dim input and save model/profile statistics")
    parser.add_argument("--output_dir", type=str, default="results/train_43d_gru_transformer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--missing_mode", type=str, default="random", choices=["random", "block", "channel"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--transformer_nhead", type=int, default=4)
    parser.add_argument("--transformer_nlayers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--bench_iters", type=int, default=100)
    args = parser.parse_args()
    run_training(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        mask_rate=args.mask_rate,
        missing_mode=args.missing_mode,
        hidden_dim=args.hidden_dim,
        transformer_nhead=args.transformer_nhead,
        transformer_nlayers=args.transformer_nlayers,
        lr=args.lr,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
