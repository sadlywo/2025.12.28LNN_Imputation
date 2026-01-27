"""Experiment: Physics loss weight ablation (no smooth/consistency terms)."""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from dataset import CfCIMUDataset
from models import PhysicsInformedIMUImputer, PhysicsInformedLoss
from train import train_one_epoch, evaluate
from visualization import plot_training_curves, plot_imputation_samples


def experiment_physics_weights_ablation():
    config = {
        "root_dir": "Oxford Dataset",
        "seq_len": 50,
        "mask_rate": 0.3,
        "missing_mode": "random",
        "batch_size": 16,
        "epochs": 50,
        "lr": 1e-3,
        "hidden_units": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": "results/physics_loss_weights_ablation",
        "num_workers": 4,
        "drift_scale": 0.01,
        "physics_strength": 0.1,
        "seed": 42,
    }

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("PHYSICS LOSS WEIGHT ABLATION (NO SMOOTH/CONSISTENCY)")
    print("=" * 80)
    print(f"Device: {config['device']}")
    print(f"Output: {output_path}")
    print(f"Missing: {config['missing_mode']} @ {config['mask_rate'] * 100:.0f}%")
    print(f"Epochs: {config['epochs']}")
    print("=" * 80)

    train_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=config["mask_rate"],
        missing_mode=config["missing_mode"],
        split="train",
        eval_mode=False,
        drift_scale=config["drift_scale"],
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
        pin_memory=True if config["device"] == "cuda" else False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True if config["device"] == "cuda" else False,
    )

    w_integration_values = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    w_energy_values = [0.0, 0.05, 0.1, 0.2, 0.5]

    results = {}
    for w_integration in w_integration_values:
        for w_energy in w_energy_values:
            exp_name = f"pi_no_cs_wi{w_integration:g}_we{w_energy:g}"
            exp_label = f"w_int={w_integration:g}, w_energy={w_energy:g}"

            print(f"\n{'=' * 80}")
            print(f"Training: {exp_label}")
            print(f"{'=' * 80}")

            model = PhysicsInformedIMUImputer(
                input_dim=13,
                hidden_units=config["hidden_units"],
                output_dim=6,
                use_physics_prior=True,
                mixed_memory=True,
                physics_strength=config["physics_strength"],
            ).to(config["device"])

            criterion = PhysicsInformedLoss(
                w_recon=1.0,
                w_consistency=0.0,
                w_smooth=0.0,
                w_physics_integration=w_integration,
                w_physics_energy=w_energy,
            )

            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config["lr"],
                epochs=config["epochs"],
                steps_per_epoch=len(train_loader),
            )

            history = {
                "train_loss": [],
                "val_loss": [],
                "val_mse_all": [],
                "val_mse_masked": [],
            }

            best_val_loss = float("inf")
            best_epoch = 0

            for epoch in range(1, config["epochs"] + 1):
                train_metrics = train_one_epoch(
                    model, train_loader, criterion, optimizer, scheduler, config["device"], epoch, use_physics=True
                )
                val_metrics = evaluate(model, val_loader, criterion, config["device"], use_physics=True)

                history["train_loss"].append(train_metrics["total"])
                history["val_loss"].append(val_metrics["total"])
                history["val_mse_all"].append(val_metrics["mse_all"])
                history["val_mse_masked"].append(val_metrics["mse_masked"])

                if val_metrics["total"] < best_val_loss:
                    best_val_loss = val_metrics["total"]
                    best_epoch = epoch
                    model_path = output_path / f"best_model_{exp_name}.pt"
                    torch.save(model.state_dict(), model_path)

                if epoch % 10 == 0:
                    sample_inputs, sample_targets, sample_preds, sample_masks = val_metrics["samples"]
                    plot_imputation_samples(
                        sample_inputs,
                        sample_targets,
                        sample_preds,
                        sample_masks,
                        num_samples=3,
                        save_path=output_path / f"samples_epoch{epoch:03d}_{exp_name}.png",
                    )

            plot_training_curves(history, save_path=output_path / f"training_curves_{exp_name}.png")

            results[exp_name] = {
                "label": exp_label,
                "w_physics_integration": float(w_integration),
                "w_physics_energy": float(w_energy),
                "best_val_loss": float(best_val_loss),
                "best_epoch": int(best_epoch),
                "final_mse_masked": float(history["val_mse_masked"][-1]),
                "final_mse_all": float(history["val_mse_all"][-1]),
                "final_train_loss": float(history["train_loss"][-1]),
                "final_val_loss": float(history["val_loss"][-1]),
                "history": history,
            }

            print(f"Best Val Loss: {best_val_loss:.6f} @ epoch {best_epoch}")
            print(f"Final Val MSE (masked): {history['val_mse_masked'][-1]:.6f}")
            print(f"Final Val MSE (all): {history['val_mse_all'][-1]:.6f}")

    generate_report(results, output_path, timestamp, config)

    print("\n" + "=" * 80)
    print("DONE: PHYSICS LOSS WEIGHT ABLATION")
    print(f"Results saved to: {output_path}")
    print("=" * 80)

    return results


def generate_report(results: dict, output_path: Path, timestamp: str, config: dict):
    summary_rows = []
    for exp_name, r in results.items():
        summary_rows.append(
            {
                "experiment": r.get("label", exp_name),
                "w_physics_integration": r["w_physics_integration"],
                "w_physics_energy": r["w_physics_energy"],
                "best_val_loss": r["best_val_loss"],
                "best_epoch": r["best_epoch"],
                "final_val_loss": r["final_val_loss"],
                "final_train_loss": r["final_train_loss"],
                "final_mse_masked": r["final_mse_masked"],
                "final_mse_all": r["final_mse_all"],
            }
        )

    df_summary = pd.DataFrame(summary_rows).sort_values(
        by=["final_mse_masked", "best_val_loss"], ascending=[True, True]
    )
    summary_path = output_path / f"summary_physics_loss_weights_{timestamp}.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[Saved] Summary CSV: {summary_path}")

    mse_path = output_path / f"val_mse_masked_by_epoch_{timestamp}.csv"
    epochs = list(range(1, config["epochs"] + 1))
    mse_rows = []
    for epoch_idx, epoch in enumerate(epochs):
        row = {"epoch": epoch}
        for exp_name, r in results.items():
            if epoch_idx < len(r["history"]["val_mse_masked"]):
                row[r.get("label", exp_name)] = r["history"]["val_mse_masked"][epoch_idx]
        mse_rows.append(row)
    pd.DataFrame(mse_rows).to_csv(mse_path, index=False)
    print(f"[Saved] Val MSE(masked) by epoch CSV: {mse_path}")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        heat_val = df_summary.pivot(index="w_physics_integration", columns="w_physics_energy", values="final_mse_masked")
        plt.figure(figsize=(10, 6))
        sns.heatmap(heat_val, annot=True, fmt=".4f", cmap="viridis")
        plt.title("Final Validation MSE (Masked)")
        plt.xlabel("w_physics_energy")
        plt.ylabel("w_physics_integration")
        plt.tight_layout()
        heat_path = output_path / f"heatmap_final_mse_masked_{timestamp}.png"
        plt.savefig(heat_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Saved] Heatmap: {heat_path}")

        heat_loss = df_summary.pivot(index="w_physics_integration", columns="w_physics_energy", values="best_val_loss")
        plt.figure(figsize=(10, 6))
        sns.heatmap(heat_loss, annot=True, fmt=".4f", cmap="magma")
        plt.title("Best Validation Loss")
        plt.xlabel("w_physics_energy")
        plt.ylabel("w_physics_integration")
        plt.tight_layout()
        heat2_path = output_path / f"heatmap_best_val_loss_{timestamp}.png"
        plt.savefig(heat2_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Saved] Heatmap: {heat2_path}")

    except Exception as e:
        print(f"[Warning] Plotting failed: {e}")


if __name__ == "__main__":
    experiment_physics_weights_ablation()

