"""Experiment: compare GRU vs Transformer trained on block missingness."""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict

from dataset import CfCIMUDataset
from models import build_model, ReconstructionOnlyLoss
from train import train_one_epoch, evaluate
from visualization import plot_training_curves, plot_imputation_samples


def experiment_block_missing_gru_transformer():
    config = {
        "root_dir": "Oxford Dataset",
        "seq_len": 50,
        "train_mask_rate": 0.3,
        "train_missing_mode": "block",
        "eval_missing_mode": "block",
        "models": ["gru", "transformer"],
        "batch_size": 16,
        "epochs": 50,
        "lr": 1e-3,
        "hidden_units": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": "results/block_missing_gru_transformer",
        "num_workers": 4,
        "block_sizes": [5, 10, 15, 20],
    }

    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*80}")
    print("START EXPERIMENT: GRU vs TRANSFORMER (TRAIN ON BLOCK, EVAL ON BLOCK)")
    print(f"{'='*80}")
    print(f"Device: {config['device']}")
    print(f"Output: {config['output_dir']}")
    print(f"Train missing_mode: {config['train_missing_mode']} @ {config['train_mask_rate'] * 100:.0f}%")
    print(f"Eval missing_mode: {config['eval_missing_mode']}")
    print(f"Models: {', '.join(m.upper() for m in config['models'])}")
    print(f"{'='*80}\n")

    eval_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=config["train_mask_rate"],
        missing_mode=config["eval_missing_mode"],
        split="val",
        eval_mode=True,
        drift_scale=0.0,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True if config["device"] == "cuda" else False,
    )

    model_results: Dict[str, dict] = {}
    for model_name in config["models"]:
        exp_key = f"{model_name}_train_block"
        print(f"\n{'='*80}")
        print(f"Training {model_name.upper()} with missing_mode='block'")
        print(f"{'='*80}")

        train_ds = CfCIMUDataset(
            root_dir=config["root_dir"],
            seq_len=config["seq_len"],
            mask_rate=config["train_mask_rate"],
            missing_mode=config["train_missing_mode"],
            split="train",
            eval_mode=False,
            drift_scale=0.01,
        )
        val_ds = CfCIMUDataset(
            root_dir=config["root_dir"],
            seq_len=config["seq_len"],
            mask_rate=config["train_mask_rate"],
            missing_mode=config["eval_missing_mode"],
            split="val",
            eval_mode=True,
            drift_scale=0.0,
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True if config["device"] == "cuda" else False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True if config["device"] == "cuda" else False,
        )

        model = build_model(
            model_name=model_name,
            input_dim=13,
            hidden_dim=config["hidden_units"],
            output_dim=6,
        ).to(config["device"])

        criterion = ReconstructionOnlyLoss(w_recon=1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["lr"],
            epochs=config["epochs"],
            steps_per_epoch=len(train_loader),
        )

        history = {"train_loss": [], "val_loss": [], "val_mse_all": [], "val_mse_masked": []}
        best_val_loss = float("inf")
        model_path = output_path / f"best_model_{exp_key}.pt"

        for epoch in range(1, config["epochs"] + 1):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, config["device"], epoch, use_physics=False
            )
            val_metrics = evaluate(model, val_loader, criterion, config["device"], use_physics=False)

            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_metrics["total"])
            history["val_mse_all"].append(val_metrics["mse_all"])
            history["val_mse_masked"].append(val_metrics["mse_masked"])

            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                torch.save(model.state_dict(), model_path)

            if epoch % 10 == 0:
                sample_inputs, sample_targets, sample_preds, sample_masks = val_metrics["samples"]
                plot_imputation_samples(
                    sample_inputs,
                    sample_targets,
                    sample_preds,
                    sample_masks,
                    num_samples=3,
                    save_path=output_path / f"samples_epoch{epoch:03d}_{exp_key}.png",
                )

        plot_training_curves(history, save_path=output_path / f"training_curves_{exp_key}.png")

        model.load_state_dict(torch.load(model_path))
        final_metrics = evaluate(model, eval_loader, criterion, config["device"], use_physics=False)

        model_results[exp_key] = {
            "model_name": model_name,
            "history": history,
            "best_val_loss": float(best_val_loss),
            "final_mse_masked": float(final_metrics["mse_masked"]),
            "final_mse_all": float(final_metrics["mse_all"]),
        }

        print(f"[{exp_key}] Best val loss: {best_val_loss:.6f}")
        print(f"[{exp_key}] Final eval MSE (masked): {final_metrics['mse_masked']:.6f}")
        print(f"[{exp_key}] Final eval MSE (all): {final_metrics['mse_all']:.6f}")

    print(f"\n{'='*80}")
    print("Evaluating different block lengths...")
    print(f"{'='*80}")

    block_results: Dict[str, dict] = {}
    for exp_key, info in model_results.items():
        model_name = info["model_name"]
        print(f"\nEvaluating {model_name.upper()} across block sizes...")

        model = build_model(
            model_name=model_name,
            input_dim=13,
            hidden_dim=config["hidden_units"],
            output_dim=6,
        ).to(config["device"])
        model_path = output_path / f"best_model_{exp_key}.pt"
        model.load_state_dict(torch.load(model_path))
        model.eval()

        model_block_results = {}
        for block_size in config["block_sizes"]:
            mask_rate = float(block_size) / float(config["seq_len"])
            custom_test_ds = CfCIMUDataset(
                root_dir=config["root_dir"],
                seq_len=config["seq_len"],
                mask_rate=mask_rate,
                missing_mode="block",
                split="val",
                eval_mode=True,
                drift_scale=0.0,
            )
            custom_test_loader = torch.utils.data.DataLoader(
                custom_test_ds,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["num_workers"],
            )

            mse_masked_list = []
            with torch.no_grad():
                for inputs, targets, mask in custom_test_loader:
                    inputs = inputs.to(config["device"])
                    targets = targets.to(config["device"])
                    mask = mask.to(config["device"])
                    pred, _ = model(inputs)
                    missing_err = ((pred - targets) ** 2 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
                    mse_masked_list.append(missing_err.item())

            mse_masked = float(np.mean(mse_masked_list))
            model_block_results[block_size] = mse_masked
            print(f"Block size {block_size}: MSE (masked) = {mse_masked:.6f}")

        block_results[model_name] = model_block_results

    _generate_report(model_results, block_results, output_path, timestamp)
    print(f"\n{'='*80}")
    print("DONE: GRU vs TRANSFORMER (BLOCK)")
    print(f"Saved to: {config['output_dir']}")
    print(f"{'='*80}")
    return model_results, block_results


def _generate_report(model_results: Dict[str, dict], block_results: Dict[str, dict], output_path: Path, timestamp: str):
    summary_data = []
    for exp_key, result in model_results.items():
        summary_data.append(
            {
                "model": result["model_name"].upper(),
                "best_val_loss": f"{result['best_val_loss']:.6f}",
                "final_mse_masked": f"{result['final_mse_masked']:.6f}",
                "final_mse_all": f"{result['final_mse_all']:.6f}",
            }
        )

    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    summary_path = output_path / f"summary_gru_transformer_block_{timestamp}.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[Saved] Summary CSV: {summary_path}")

    if block_results:
        block_sizes = sorted(next(iter(block_results.values())).keys())
        rows = []
        for b in block_sizes:
            row = {"block_size": b}
            for model_name, d in block_results.items():
                row[f"{model_name.upper()}_mse_masked"] = f"{d[b]:.6f}"
            rows.append(row)
        df_block = pd.DataFrame(rows)
        print(f"\n{'='*80}")
        print("MSE comparison across block sizes")
        print(f"{'='*80}")
        print(df_block.to_string(index=False))
        block_path = output_path / f"block_size_comparison_gru_transformer_{timestamp}.csv"
        df_block.to_csv(block_path, index=False)
        print(f"\n[Saved] Block size CSV: {block_path}")

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        for model_name, d in block_results.items():
            xs = list(d.keys())
            ys = list(d.values())
            plt.plot(xs, ys, marker="o", linewidth=2, label=model_name.upper())
        plt.title("Masked MSE vs. Block Size (Eval: Block Missingness)")
        plt.xlabel("Block Size")
        plt.ylabel("MSE (masked)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = output_path / f"comparison_block_sizes_gru_transformer_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Saved] Plot: {plot_path}")


if __name__ == "__main__":
    experiment_block_missing_gru_transformer()

