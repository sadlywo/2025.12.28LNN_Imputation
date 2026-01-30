"""Experiment: baseline training under different missing modes, evaluated on block missingness."""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple

# 导入项目模块
from dataset import CfCIMUDataset
from models import build_model
from models import ReconstructionOnlyLoss
from train import train_one_epoch, evaluate
from visualization import plot_training_curves, plot_imputation_samples


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


def experiment_block_missing():
    """
    Compare GRU/Transformer/GAIN trained with different missing modes and evaluate on block missingness.
    Block missingness here means a continuous missing segment within a single channel.
    """
    # 实验配置
    config = {
        "root_dir": "Oxford Dataset",
        "seq_len": 50,
        "mask_rate": 0.3,
        "train_missing_modes": ["random", "block"],
        "eval_missing_mode": "block",
        "models": ["gru", "transformer", "gain"],
        "batch_size": 16,
        "epochs": 50,
        "lr": 1e-3,
        "hidden_units": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": "results/block_missing_training_comparison_baselines",
        "num_workers": 4,
        "gain_noise_scale": 0.1,
    }
    
    # 创建输出目录
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*80}")
    print("START EXPERIMENT: TRAINING MODE COMPARISON (EVAL ON BLOCK MISSINGNESS)")
    print(f"{'='*80}")
    print(f"Device: {config['device']}")
    print(f"Output: {config['output_dir']}")
    print(f"Train modes: {', '.join(config['train_missing_modes'])}")
    print(f"Eval mode: {config['eval_missing_mode']}")
    print(f"{'='*80}\n")
    
    eval_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=config["mask_rate"],
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

    model_results = {}
    for model_name in config["models"]:
        for train_missing_mode in config["train_missing_modes"]:
            exp_key = f"{model_name}_train_{train_missing_mode}"
            print(f"\n{'='*80}")
            print(f"Training {model_name.upper()} with missing_mode='{train_missing_mode}'")
            print(f"{'='*80}")

            train_ds = CfCIMUDataset(
                root_dir=config["root_dir"],
                seq_len=config["seq_len"],
                mask_rate=config["mask_rate"],
                missing_mode=train_missing_mode,
                split="train",
                eval_mode=False,
                drift_scale=0.01,
            )
            val_ds = CfCIMUDataset(
                root_dir=config["root_dir"],
                seq_len=config["seq_len"],
                mask_rate=config["mask_rate"],
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

            if model_name == "gain":
                model = GAINImputer(
                    input_dim=13,
                    hidden_dim=config["hidden_units"],
                    output_dim=6,
                    noise_scale=config["gain_noise_scale"],
                ).to(config["device"])
            else:
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
                    model_path = output_path / f"best_model_{exp_key}.pt"
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
                "train_missing_mode": train_missing_mode,
                "history": history,
                "best_val_loss": best_val_loss,
                "final_mse_masked": float(final_metrics["mse_masked"]),
                "final_mse_all": float(final_metrics["mse_all"]),
            }

            print(f"[{exp_key}] Best val loss: {best_val_loss:.6f}")
            print(f"[{exp_key}] Final eval MSE (masked): {final_metrics['mse_masked']:.6f}")
            print(f"[{exp_key}] Final eval MSE (all): {final_metrics['mse_all']:.6f}")
    
    # 在不同块大小下评估模型性能
    print(f"\n{'='*80}")
    print("Evaluating different block lengths...")
    print(f"{'='*80}")
    
    # 不同块大小配置
    block_sizes = [5, 10, 15, 20]  # 连续丢失的时间步数
    block_results = {}
    
    for exp_key, info in model_results.items():
        model_name = info["model_name"]
        train_missing_mode = info["train_missing_mode"]
        print(f"\nEvaluating {exp_key} across block sizes...")

        if model_name == "gain":
            model = GAINImputer(
                input_dim=13,
                hidden_dim=config["hidden_units"],
                output_dim=6,
                noise_scale=config["gain_noise_scale"],
            ).to(config["device"])
        else:
            model = build_model(
                model_name=model_name,
                input_dim=13,
                hidden_dim=config["hidden_units"],
                output_dim=6,
            ).to(config["device"])
        model_path = output_path / f"best_model_{exp_key}.pt"
        model.load_state_dict(torch.load(model_path))
        
        model_block_results = {}
        
        for block_size in block_sizes:
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

            # 创建数据加载器
            custom_test_loader = torch.utils.data.DataLoader(
                custom_test_ds,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["num_workers"],
            )
            
            # 评估模型
            model.eval()
            mse_masked_list = []
            
            with torch.no_grad():
                for inputs, targets, mask in custom_test_loader:
                    inputs = inputs.to(config["device"])
                    targets = targets.to(config["device"])
                    mask = mask.to(config["device"])
                    
                    pred, _ = model(inputs)
                    missing_err = ((pred - targets) ** 2 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
                    mse_masked_list.append(missing_err.item())
            
            mse_masked = np.mean(mse_masked_list)
            model_block_results[block_size] = mse_masked
            print(f"Block size {block_size}: MSE (masked) = {mse_masked:.6f}")
        
        block_results[exp_key] = {
            "model_name": model_name,
            "train_missing_mode": train_missing_mode,
            "by_block_size": model_block_results,
        }
    
    # 生成对比报告
    generate_comparison_report(model_results, block_results, output_path, timestamp, config)
    
    print(f"\n{'='*80}")
    print("DONE: TRAINING MODE COMPARISON")
    print(f"Saved to: {config['output_dir']}")
    print(f"{'='*80}")
    
    return model_results, block_results


def generate_comparison_report(model_results, block_results, output_path, timestamp, config):
    """Generate comparison report."""
    print(f"\n{'='*80}")
    print("Generating report")
    print(f"{'='*80}")
    
    # 收集所有方法的结果
    all_results = {}
    
    # 添加深度学习方法结果
    for method, result in model_results.items():
        all_results[method] = {
            "mse_masked": result["final_mse_masked"],
            "mse_all": result["final_mse_all"],
            "type": "deep",
        }
    
    # 生成总结表格
    summary_data = []
    for method, result in all_results.items():
        label = f"{result['model_name'].upper()} (Train: {result['train_missing_mode'].capitalize()})"
        summary_data.append({
            "method": label,
            "final_mse_masked": f"{result['mse_masked']:.6f}",
            "final_mse_all": f"{result['mse_all']:.6f}",
        })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    # 保存总结表格
    summary_path = output_path / f"summary_training_modes_{timestamp}.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[Saved] Summary CSV: {summary_path}")
    
    # 生成不同块大小的对比表格
    if block_results:
        block_data = []
        block_sizes = list(next(iter(block_results.values()))["by_block_size"].keys())
        
        for block_size in block_sizes:
            row = {"block_size": block_size}
            for method, results in block_results.items():
                label = f"{results['model_name'].upper()} (Train: {results['train_missing_mode'].capitalize()})"
                row[f"{label}_mse_masked"] = f"{results['by_block_size'][block_size]:.6f}"
            block_data.append(row)
        
        df_block = pd.DataFrame(block_data)
        print(f"\n{'='*80}")
        print("MSE comparison across block sizes")
        print(f"{'='*80}")
        print(df_block.to_string(index=False))
        
        # 保存块大小对比表格
        block_path = output_path / f"block_size_comparison_{timestamp}.csv"
        df_block.to_csv(block_path, index=False)
        print(f"\n[Saved] Block size CSV: {block_path}")
    
    # 绘制对比图
    import matplotlib.pyplot as plt
    
    # 1. 不同方法的MSE对比
    plt.figure(figsize=(12, 6))
    
    methods = list(all_results.keys())
    mse_values = [all_results[m]["mse_masked"] for m in methods]
    method_labels = [method_name_map.get(m, m) for m in methods]
    
    bars = plt.bar(method_labels, mse_values, capsize=5)
    plt.title("Final Masked MSE (Eval: Block Missingness)")
    plt.xlabel("Method")
    plt.ylabel("MSE (masked)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    
    # 在柱状图上显示数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    comparison_plot_path = output_path / f"comparison_training_modes_{timestamp}.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Plot: {comparison_plot_path}")
    
    # 2. 不同块大小的MSE对比
    if block_results:
        plt.figure(figsize=(12, 6))
        
        for method, results in block_results.items():
            block_sizes = list(results["by_block_size"].keys())
            mse_values = list(results["by_block_size"].values())
            plt.plot(block_sizes, mse_values, marker='o', linewidth=2, 
                     label=f"{results['model_name'].upper()} (Train: {results['train_missing_mode'].capitalize()})")
        
        plt.title("Masked MSE vs. Block Size (Eval: Block Missingness)")
        plt.xlabel("Block Size")
        plt.ylabel("MSE (masked)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        block_size_plot_path = output_path / f"comparison_block_sizes_{timestamp}.png"
        plt.savefig(block_size_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Saved] Plot: {block_size_plot_path}")


if __name__ == "__main__":
    # 运行实验
    experiment_block_missing()
