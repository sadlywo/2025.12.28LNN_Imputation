"""实验1：有无物理输出头的对比实验"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# 导入项目模块
from dataset import CfCIMUDataset
from models import PhysicsAwareIMUImputer, AdaptiveLoss
from train import train_one_epoch, evaluate
from visualization import plot_training_curves, plot_imputation_samples


def experiment_physical_head():
    """
    对比有无物理输出头的模型性能
    - 模型1：使用物理输出头（分离的陀螺仪和加速度计头）
    - 模型2：不使用物理输出头（单一输出头）
    """
    # 实验配置
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
        "output_dir": "results/physical_head_experiment",
        "num_workers": 4,
    }
    
    # 创建输出目录
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*80}")
    print(f"开始实验：有无物理输出头对比")
    print(f"{'='*80}")
    print(f"设备: {config['device']}")
    print(f"输出目录: {config['output_dir']}")
    print(f"{'='*80}\n")
    
    # 加载数据集
    print("加载数据集...")
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
    
    # 定义模型
    models = {
        "with_physical_head": PhysicsAwareIMUImputer(
            input_dim=13,
            hidden_units=config["hidden_units"],
            output_dim=6,
            use_physics_prior=True,  # 使用物理输出头
            mixed_memory=True,
        ),
        "without_physical_head": PhysicsAwareIMUImputer(
            input_dim=13,
            hidden_units=config["hidden_units"],
            output_dim=6,
            use_physics_prior=False,  # 不使用物理输出头
            mixed_memory=True,
        )
    }
    
    # 训练两个模型
    results = {}
    for model_name, model in models.items():
        print(f"\n{'='*80}")
        print(f"训练模型: {model_name}")
        print(f"{'='*80}")
        
        # 设置设备
        model = model.to(config["device"])
        
        # 定义损失函数和优化器
        criterion = AdaptiveLoss(
            w_recon=1.0,
            w_consistency=0.0,
            w_smooth=0.00,
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["lr"],
            epochs=config["epochs"],
            steps_per_epoch=len(train_loader),
        )
        
        # 训练历史记录
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_mse_all": [],
            "val_mse_masked": [],
        }
        
        best_val_loss = float("inf")
        
        # 训练循环
        for epoch in range(1, config["epochs"] + 1):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, 
                config["device"], epoch, use_physics=False
            )
            
            val_metrics = evaluate(
                model, val_loader, criterion, config["device"], use_physics=False
            )
            
            # 保存历史记录
            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_metrics["total"])
            history["val_mse_all"].append(val_metrics["mse_all"])
            history["val_mse_masked"].append(val_metrics["mse_masked"])
            
            # 保存最佳模型
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                model_path = output_path / f"best_model_{model_name}.pt"
                torch.save(model.state_dict(), model_path)
            
            # 每10个epoch可视化一次
            if epoch % 10 == 0:
                sample_inputs, sample_targets, sample_preds, sample_masks = val_metrics["samples"]
                plot_imputation_samples(
                    sample_inputs, sample_targets, sample_preds, sample_masks,
                    num_samples=3,
                    save_path=output_path / f"samples_epoch{epoch:03d}_{model_name}.png"
                )
        
        # 保存训练曲线
        plot_training_curves(
            history,
            save_path=output_path / f"training_curves_{model_name}.png"
        )
        
        # 保存模型结果
        results[model_name] = {
            "history": history,
            "best_val_loss": best_val_loss,
            "final_mse_masked": history["val_mse_masked"][-1],
            "final_mse_all": history["val_mse_all"][-1],
        }
        
        print(f"\n[{model_name}] 训练完成")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        print(f"最终MSE (masked): {history['val_mse_masked'][-1]:.4f}")
        print(f"最终MSE (all): {history['val_mse_all'][-1]:.4f}")
    
    # 生成对比报告
    generate_comparison_report(results, output_path, timestamp, config)
    
    print(f"\n{'='*80}")
    print("实验1：有无物理输出头对比实验完成")
    print(f"结果保存至: {config['output_dir']}")
    print(f"{'='*80}")
    
    return results


def generate_comparison_report(results, output_path, timestamp, config):
    """生成对比报告"""
    print(f"\n{'='*80}")
    print("生成对比报告")
    print(f"{'='*80}")
    
    # 生成总结表格
    summary_data = []
    for model_name, result in results.items():
        summary_data.append({
            "模型": "有物理输出头" if "with" in model_name else "无物理输出头",
            "最佳验证损失": f"{result['best_val_loss']:.4f}",
            "最终MSE (masked)": f"{result['final_mse_masked']:.4f}",
            "最终MSE (all)": f"{result['final_mse_all']:.4f}",
            "训练损失": f"{result['history']['train_loss'][-1]:.4f}",
        })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    # 保存总结表格
    summary_path = output_path / f"summary_physical_head_{timestamp}.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n总结表格保存至: {summary_path}")
    
    # 生成详细的MSE对比表格
    mse_data = []
    for epoch in range(len(results["with_physical_head"]["history"]["val_mse_masked"])):
        mse_data.append({
            "epoch": epoch + 1,
            "有物理输出头_MSE_masked": results["with_physical_head"]["history"]["val_mse_masked"][epoch],
            "无物理输出头_MSE_masked": results["without_physical_head"]["history"]["val_mse_masked"][epoch],
            "有物理输出头_MSE_all": results["with_physical_head"]["history"]["val_mse_all"][epoch],
            "无物理输出头_MSE_all": results["without_physical_head"]["history"]["val_mse_all"][epoch],
        })
    
    df_mse = pd.DataFrame(mse_data)
    mse_path = output_path / f"mse_comparison_physical_head_{timestamp}.csv"
    df_mse.to_csv(mse_path, index=False)
    print(f"MSE对比表格保存至: {mse_path}")
    
    # 绘制最终对比图
    import matplotlib.pyplot as plt
    
    epochs = list(range(1, config["epochs"] + 1))
    
    plt.figure(figsize=(12, 6))
    
    # MSE (masked) 对比
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["with_physical_head"]["history"]["val_mse_masked"], 
             label="With Physical Output Head", linewidth=2)
    plt.plot(epochs, results["without_physical_head"]["history"]["val_mse_masked"], 
             label="Without Physical Output Head", linewidth=2)
    plt.title("Validation MSE (masked)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MSE (all) 对比
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["with_physical_head"]["history"]["val_mse_all"], 
             label="With Physical Output Head", linewidth=2)
    plt.plot(epochs, results["without_physical_head"]["history"]["val_mse_all"], 
             label="Without Physical Output Head", linewidth=2)
    plt.title("Validation MSE (all)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_plot_path = output_path / f"comparison_physical_head_{timestamp}.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"对比图保存至: {comparison_plot_path}")


if __name__ == "__main__":
    # 运行实验
    experiment_physical_head()
