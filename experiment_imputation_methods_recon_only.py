"""Experiment: imputation accuracy comparison under reconstruction-only objective."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CfCIMUDataset
from models import (
    PhysicsAwareIMUImputer,
    GRUImputer,
    TransformerImputer,
    ReconstructionOnlyLoss,
    count_parameters,
)
from train import train_one_epoch, evaluate
from visualization import plot_training_curves, plot_imputation_samples


def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    mse_all = F.mse_loss(pred, target).item()
    missing_err = ((pred - target) ** 2 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
    mse_masked = missing_err.item()
    return mse_all, mse_masked


def _compute_target_mean(train_loader, device: torch.device) -> torch.Tensor:
    total = torch.zeros(6, device=device)
    count = 0
    for _inputs, targets, _mask in train_loader:
        targets = targets.to(device)
        total += targets.sum(dim=(0, 1))
        count += targets.shape[0] * targets.shape[1]
    return total / max(count, 1)

def _state_dict_size_mb(model: nn.Module) -> float:
    total_bytes = 0
    for v in model.state_dict().values():
        if torch.is_tensor(v):
            total_bytes += v.numel() * v.element_size()
    return float(total_bytes) / (1024.0 * 1024.0)


def _measure_deep_inference_time(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, int, int]:
    model.eval()
    total_time = 0.0
    total_batches = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, _targets, _mask in loader:
            if max_batches is not None and total_batches >= max_batches:
                break
            inputs = inputs.to(device)
            _sync(device)
            t0 = time.time()
            out = model(inputs)
            _ = out[0] if isinstance(out, (tuple, list)) else out
            _sync(device)
            total_time += time.time() - t0
            total_batches += 1
            total_samples += int(inputs.shape[0])
    return float(total_time), int(total_batches), int(total_samples)


def locf_impute_sequence(x_masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = x_masked.clone()
    t_len, channels = x.shape
    for c in range(channels):
        last = None
        for t in range(t_len):
            if mask[t, c] > 0.5:
                last = x[t, c].item()
            else:
                if last is None:
                    pass
                else:
                    x[t, c] = last
    return x


def knn_impute_sequence(x_masked: torch.Tensor, mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    x = x_masked.clone()
    t_len, channels = x.shape

    for t in range(t_len):
        for c in range(channels):
            if mask[t, c] > 0.5:
                continue

            candidates = []
            for s in range(t_len):
                if mask[s, c] < 0.5:
                    continue

                common = (mask[t] > 0.5) & (mask[s] > 0.5)
                if common.sum().item() == 0:
                    continue

                dist = torch.norm(x[t, common] - x[s, common], p=2).item()
                candidates.append((dist, x[s, c].item()))

            if not candidates:
                observed = x_masked[mask[:, c] > 0.5, c]
                if observed.numel() > 0:
                    x[t, c] = observed.mean()
                continue

            candidates.sort(key=lambda z: z[0])
            top = candidates[:k]
            x[t, c] = float(np.mean([v for _d, v in top]))

    return x


def trmf_impute_sequence(
    x_masked: torch.Tensor,
    mask: torch.Tensor,
    rank: int = 3,
    steps: int = 300,
    lr: float = 0.05,
    lambda_time: float = 0.1,
    lambda_l2: float = 1e-3,
) -> torch.Tensor:
    device = x_masked.device
    t_len, channels = x_masked.shape

    u = torch.randn(t_len, rank, device=device, requires_grad=True) * 0.01
    v = torch.randn(channels, rank, device=device, requires_grad=True) * 0.01

    optimizer = torch.optim.Adam([u, v], lr=lr)
    obs = mask

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        y_hat = u @ v.t()

        recon = ((y_hat - x_masked) ** 2 * obs).sum() / (obs.sum() + 1e-8)

        if t_len >= 3:
            d2 = u[2:] - 2 * u[1:-1] + u[:-2]
            time_reg = (d2**2).mean()
        else:
            time_reg = torch.zeros((), device=device)

        l2 = (u**2).mean() + (v**2).mean()
        loss = recon + lambda_time * time_reg + lambda_l2 * l2
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return (u @ v.t()).detach()


class GRUDImputer(nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 128, output_dim: int = 6, x_mean: Optional[torch.Tensor] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.gru_cell = nn.GRUCell(output_dim * 2, hidden_dim)
        self.decay_x_weight = nn.Parameter(torch.ones(output_dim))
        self.decay_x_bias = nn.Parameter(torch.zeros(output_dim))
        self.decay_h_weight = nn.Parameter(torch.ones(hidden_dim))
        self.decay_h_bias = nn.Parameter(torch.zeros(hidden_dim))

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

        if x_mean is None:
            self.register_buffer("x_mean", torch.zeros(output_dim))
        else:
            self.register_buffer("x_mean", x_mean.detach().clone().view(-1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_val = x[:, :, :6]
        m = x[:, :, 6:12]
        dt = x[:, :, 12:13]
        bsz, t_len, channels = x_val.shape

        x_mean = self.x_mean.view(1, 1, channels).expand(bsz, t_len, channels)

        delta = torch.zeros(bsz, t_len, channels, device=x.device, dtype=x.dtype)
        running = torch.zeros(bsz, channels, device=x.device, dtype=x.dtype)
        for t in range(t_len):
            if t == 0:
                running = dt[:, t].expand(-1, channels)
            else:
                running = running + dt[:, t].expand(-1, channels)
            observed = m[:, t] > 0.5
            running = torch.where(observed, torch.zeros_like(running), running)
            delta[:, t] = running

        x_last = torch.zeros(bsz, channels, device=x.device, dtype=x.dtype)
        h = torch.zeros(bsz, self.hidden_dim, device=x.device, dtype=x.dtype)
        preds = []
        uncs = []

        for t in range(t_len):
            gamma_x = torch.exp(-F.relu(delta[:, t] * self.decay_x_weight + self.decay_x_bias))
            gamma_h = torch.exp(-F.relu(dt[:, t] * self.decay_h_weight + self.decay_h_bias))
            h = gamma_h * h

            x_last = torch.where(m[:, t] > 0.5, x_val[:, t], x_last)
            x_hat = m[:, t] * x_val[:, t] + (1 - m[:, t]) * (gamma_x * x_last + (1 - gamma_x) * x_mean[:, t])

            gru_in = torch.cat([x_hat, m[:, t]], dim=-1)
            h = self.gru_cell(gru_in, h)

            pred_t = self.head(h)
            unc_t = self.uncertainty_head(h)
            preds.append(pred_t.unsqueeze(1))
            uncs.append(unc_t.unsqueeze(1))

        pred = torch.cat(preds, dim=1)
        unc = torch.cat(uncs, dim=1)
        return pred, unc


class GAINImputer(nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 128, output_dim: int = 6, noise_scale: float = 0.1):
        super().__init__()
        self.noise_scale = noise_scale
        self.net = nn.Sequential(
            nn.Linear(input_dim + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, t_len, _ = x.shape
        noise = torch.randn(bsz, t_len, 6, device=x.device, dtype=x.dtype) * self.noise_scale
        z = torch.cat([x, noise], dim=-1)
        pred = self.net(z)
        unc = self.uncertainty_head(pred)
        return pred, unc


@dataclass
class MethodResult:
    method: str
    kind: str
    mse_all: float
    mse_masked: float
    train_time_sec: float
    inference_time_sec: float
    num_params: Optional[int]
    param_size_mb: Optional[float]


def _evaluate_classical(
    val_loader,
    imputer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, int]:
    mse_all_list: List[float] = []
    mse_masked_list: List[float] = []
    batches = 0

    for inputs, targets, mask in val_loader:
        batches += 1
        if max_batches is not None and batches > max_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        x_masked = inputs[:, :, :6]

        preds = []
        for i in range(x_masked.shape[0]):
            pred_i = imputer(x_masked[i], mask[i])
            preds.append(pred_i.unsqueeze(0))
        pred = torch.cat(preds, dim=0)

        a, m = _masked_mse(pred, targets, mask)
        mse_all_list.append(a)
        mse_masked_list.append(m)

    return float(np.mean(mse_all_list)), float(np.mean(mse_masked_list)), batches


def run_comparison():
    config = {
        "root_dir": "Oxford Dataset",
        "seq_len": 50,
        "mask_rate": 0.3,
        "missing_mode": "random",
        "batch_size": 16,
        "epochs": 30,
        "lr": 1e-3,
        "hidden_units": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": "results/recon_only_method_comparison",
        "num_workers": 4,
        "seed": 42,
        "classical_max_val_batches": None,
        "trmf_rank": 3,
        "trmf_steps": 300,
        "trmf_lr": 0.05,
        "trmf_lambda_time": 0.1,
        "trmf_lambda_l2": 1e-3,
        "knn_k": 5,
        "transformer_nhead": 4,
        "transformer_nlayers": 2,
        "gain_noise_scale": 0.1,
        "deep_inference_max_val_batches": None,
    }

    _seed_all(config["seed"])

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("RECONSTRUCTION-ONLY IMPUTATION METHOD COMPARISON")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Output: {output_path}")
    print(f"Missing: {config['missing_mode']} @ {config['mask_rate'] * 100:.0f}%")
    print("=" * 80)

    train_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=config["mask_rate"],
        missing_mode=config["missing_mode"],
        split="train",
        eval_mode=False,
        drift_scale=0.01,
    )
    val_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=config["mask_rate"],
        missing_mode=config["missing_mode"],
        split="val",
        eval_mode=True,
        drift_scale=0.0,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
    )

    x_mean = _compute_target_mean(train_loader, device=device)

    deep_methods: List[Tuple[str, Callable[[], nn.Module]]] = [
        ("LNN", lambda: PhysicsAwareIMUImputer(input_dim=13, hidden_units=config["hidden_units"], output_dim=6, use_physics_prior=True, mixed_memory=True)),
        ("GRU-D", lambda: GRUDImputer(input_dim=13, hidden_dim=config["hidden_units"], output_dim=6, x_mean=x_mean)),
        ("Transformer", lambda: TransformerImputer(input_dim=13, hidden_dim=config["hidden_units"], output_dim=6, nhead=config["transformer_nhead"], nlayers=config["transformer_nlayers"])),
        ("GAIN", lambda: GAINImputer(input_dim=13, hidden_dim=config["hidden_units"], output_dim=6, noise_scale=config["gain_noise_scale"])),
    ]

    classical_methods: List[Tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = [
        ("LOCF", lambda x, m: locf_impute_sequence(x, m)),
        ("KNN", lambda x, m: knn_impute_sequence(x, m, k=config["knn_k"])),
        (
            "TRMF",
            lambda x, m: trmf_impute_sequence(
                x,
                m,
                rank=config["trmf_rank"],
                steps=config["trmf_steps"],
                lr=config["trmf_lr"],
                lambda_time=config["trmf_lambda_time"],
                lambda_l2=config["trmf_lambda_l2"],
            ),
        ),
    ]

    results: List[MethodResult] = []
    per_method_histories: Dict[str, dict] = {}
    method_param_rows: List[dict] = []

    criterion = ReconstructionOnlyLoss(w_recon=1.0)

    for name, model_factory in deep_methods:
        model = model_factory().to(device)
        num_params = count_parameters(model)
        param_size_mb = _state_dict_size_mb(model)
        method_param_rows.append(
            {
                "method": name,
                "category": "deep",
                "epochs": config["epochs"],
                "batch_size": config["batch_size"],
                "seq_len": config["seq_len"],
                "mask_rate": config["mask_rate"],
                "missing_mode": config["missing_mode"],
                "lr": config["lr"],
                "hidden_units": config["hidden_units"],
                "num_params": int(num_params),
                "transformer_nhead": config["transformer_nhead"] if name == "Transformer" else np.nan,
                "transformer_nlayers": config["transformer_nlayers"] if name == "Transformer" else np.nan,
                "gain_noise_scale": config["gain_noise_scale"] if name == "GAIN" else np.nan,
                "grud_x_mean_source": "train_target_mean" if name == "GRU-D" else "",
                "lnn_use_physics_prior": True if name == "LNN" else np.nan,
                "lnn_mixed_memory": True if name == "LNN" else np.nan,
            }
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["lr"],
            epochs=config["epochs"],
            steps_per_epoch=len(train_loader),
        )

        history = {"train_loss": [], "val_loss": [], "val_mse_all": [], "val_mse_masked": []}
        best_val_loss = float("inf")

        start = time.time()
        for epoch in range(1, config["epochs"] + 1):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, use_physics=False)
            val_metrics = evaluate(model, val_loader, criterion, device, use_physics=False)

            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_metrics["total"])
            history["val_mse_all"].append(val_metrics["mse_all"])
            history["val_mse_masked"].append(val_metrics["mse_masked"])

            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                torch.save(model.state_dict(), output_path / f"best_model_{name}_{timestamp}.pt")

            if epoch % 10 == 0:
                sample_inputs, sample_targets, sample_preds, sample_masks = val_metrics["samples"]
                plot_imputation_samples(
                    sample_inputs,
                    sample_targets,
                    sample_preds,
                    sample_masks,
                    num_samples=3,
                    save_path=output_path / f"samples_epoch{epoch:03d}_{name}_{timestamp}.png",
                )

        train_time = time.time() - start
        per_method_histories[name] = history
        plot_training_curves(history, save_path=output_path / f"training_curves_{name}_{timestamp}.png")

        best_model_path = output_path / f"best_model_{name}_{timestamp}.pt"
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_eval_metrics = evaluate(model, val_loader, criterion, device, use_physics=False)
        infer_time, infer_batches, infer_samples = _measure_deep_inference_time(
            model, val_loader, device=device, max_batches=config["deep_inference_max_val_batches"]
        )

        results.append(
            MethodResult(
                method=name,
                kind="deep",
                mse_all=float(best_eval_metrics["mse_all"]),
                mse_masked=float(best_eval_metrics["mse_masked"]),
                train_time_sec=float(train_time),
                inference_time_sec=float(infer_time),
                num_params=int(num_params),
                param_size_mb=float(param_size_mb),
            )
        )

    for name, imputer in classical_methods:
        start = time.time()
        a, m, batches = _evaluate_classical(
            val_loader,
            imputer=imputer,
            device=device,
            max_batches=config["classical_max_val_batches"],
        )
        elapsed = time.time() - start

        results.append(
            MethodResult(
                method=name,
                kind=f"classical(batches={batches})",
                mse_all=a,
                mse_masked=m,
                train_time_sec=0.0,
                inference_time_sec=float(elapsed),
                num_params=None,
                param_size_mb=None,
            )
        )
        method_param_rows.append(
            {
                "method": name,
                "category": "classical",
                "seq_len": config["seq_len"],
                "mask_rate": config["mask_rate"],
                "missing_mode": config["missing_mode"],
                "classical_max_val_batches": config["classical_max_val_batches"],
                "knn_k": config["knn_k"] if name == "KNN" else np.nan,
                "trmf_rank": config["trmf_rank"] if name == "TRMF" else np.nan,
                "trmf_steps": config["trmf_steps"] if name == "TRMF" else np.nan,
                "trmf_lr": config["trmf_lr"] if name == "TRMF" else np.nan,
                "trmf_lambda_time": config["trmf_lambda_time"] if name == "TRMF" else np.nan,
                "trmf_lambda_l2": config["trmf_lambda_l2"] if name == "TRMF" else np.nan,
            }
        )

    df = pd.DataFrame([r.__dict__ for r in results]).sort_values(by=["mse_masked", "mse_all"], ascending=[True, True])
    df["total_time_sec"] = df["train_time_sec"].fillna(0.0) + df["inference_time_sec"].fillna(0.0)
    df["inference_ms_per_batch"] = np.nan
    for idx, row in df.iterrows():
        if row["inference_time_sec"] and isinstance(row["kind"], str) and row["kind"].startswith("deep"):
            df.loc[idx, "inference_ms_per_batch"] = 1000.0 * row["inference_time_sec"] / max(len(val_loader), 1)

    summary_path = output_path / f"summary_recon_only_methods_{timestamp}.csv"
    df.to_csv(summary_path, index=False)
    print(f"\n[Saved] Summary CSV: {summary_path}")
    print(df.to_string(index=False))

    df_params = df[df["num_params"].notna()][["method", "num_params"]].copy()
    df_params["num_params"] = df_params["num_params"].astype(int)
    params_path = output_path / f"model_parameter_counts_{timestamp}.csv"
    df_params.to_csv(params_path, index=False)
    print(f"[Saved] Parameter counts CSV: {params_path}")

    df_method_params = pd.DataFrame(method_param_rows)
    method_params_path = output_path / f"method_key_parameters_{timestamp}.csv"
    df_method_params.to_csv(method_params_path, index=False)
    print(f"[Saved] Method key parameters CSV: {method_params_path}")

    df_config = pd.DataFrame([{"key": k, "value": v} for k, v in config.items()])
    excel_path = output_path / f"comparison_metrics_{timestamp}.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="metrics")
            df_params.to_excel(writer, index=False, sheet_name="param_counts")
            df_method_params.to_excel(writer, index=False, sheet_name="method_params")
            df_config.to_excel(writer, index=False, sheet_name="config")
        print(f"[Saved] Excel: {excel_path}")
    except Exception as e:
        print(f"[Warning] Excel export failed: {e}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.bar(df["method"].tolist(), df["mse_masked"].tolist())
        plt.title("Imputation Accuracy (Reconstruction-Only, Masked MSE)")
        plt.xlabel("Method")
        plt.ylabel("Validation MSE (Masked)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plot_path = output_path / f"comparison_recon_only_methods_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Saved] Comparison plot: {plot_path}")

    except Exception as e:
        print(f"[Warning] Plotting failed: {e}")

    torch.save({"config": config, "results": results, "histories": per_method_histories}, output_path / f"raw_results_{timestamp}.pt")
    return results


if __name__ == "__main__":
    run_comparison()
