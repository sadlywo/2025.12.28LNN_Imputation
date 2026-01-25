"""实验：预训练-微调策略验证实验"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# 导入项目模块
from dataset import CfCIMUDataset
from models import PhysicsAwareIMUImputer, AdaptiveLoss
from train import train_one_epoch, evaluate, evaluate_multi_missing_rates
from visualization import plot_training_curves, plot_imputation_samples

def run_pretraining_experiment():
    """
    执行预训练-微调实验流程：
    1. 预训练阶段：去噪自编码器任务（输入加噪，目标纯净，无缺失或低缺失）
    2. 微调阶段：下游补缺任务（标准缺失率）
    3. 基准对比：无预训练直接进行补缺任务训练
    """
    # 通用配置
    config = {
        "root_dir": "Oxford Dataset",
        "seq_len": 50,
        "batch_size": 16,
        "hidden_units": 128,  # 增加模型容量
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": "results/pretraining_experiment",
        "num_workers": 4,
    }
    
    # 预训练配置 (Pre-training)
    # 目标：学习物理动力学和去噪
    pretrain_config = {
        "epochs": 20,
        "lr": 1e-3,
        "mask_rate": 0.0,       # 几乎无缺失，专注重建
        "drift_scale": 0.05,    # 强噪声/漂移，迫使模型学习去噪
        "missing_mode": "random",
    }
    
    # 微调/下游任务配置 (Fine-tuning / Downstream)
    # 目标：完成补缺任务
    finetune_config = {
        "epochs": 30,
        "lr": 5e-4,             # 较低的学习率用于微调
        "mask_rate": 0.3,       # 标准缺失率
        "drift_scale": 0.01,    # 弱噪声用于正则化
        "missing_mode": "random", # 这里先用random，也可以测block
    }
    
    # 创建输出目录
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*80}")
    print(f"开始实验：预训练 (Denoising Autoencoder) -> 微调策略")
    print(f"{'='*80}")
    print(f"设备: {config['device']}")
    
    # ==========================================
    # Phase 1: 预训练 (Pre-training)
    # ==========================================
    print(f"\n\n[Phase 1] 开始预训练: 去噪自编码器模式")
    print(f"配置: mask_rate={pretrain_config['mask_rate']}, drift_scale={pretrain_config['drift_scale']}")
    
    # 加载预训练数据
    pt_train_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=pretrain_config["mask_rate"],
        missing_mode=pretrain_config["missing_mode"],
        split="train",
        eval_mode=False,
        drift_scale=pretrain_config["drift_scale"],
    )
    # 验证集也用同样的去噪任务设置，但无drift(dataset逻辑)或小drift
    # 注意：dataset.py中eval_mode=True时强制drift_scale无效。
    # 为了验证去噪能力，我们在eval时可以临时允许drift，或者我们主要看recon loss
    pt_val_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=pretrain_config["mask_rate"],
        missing_mode=pretrain_config["missing_mode"],
        split="val",
        eval_mode=True, 
        drift_scale=0.0, # 验证集通常用纯净数据看重建能力
    )
    
    pt_train_loader = torch.utils.data.DataLoader(pt_train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    pt_val_loader = torch.utils.data.DataLoader(pt_val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
    # 初始化模型
    model = PhysicsAwareIMUImputer(
        input_dim=13,
        hidden_units=config["hidden_units"],
        output_dim=6,
        use_physics_prior=True,
        mixed_memory=True,
    ).to(config["device"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=pretrain_config["lr"])
    criterion = AdaptiveLoss(w_recon=1.0, w_consistency=0.1, w_smooth=0.01)
    
    # 预训练循环
    pt_history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, pretrain_config["epochs"] + 1):
        train_metrics = train_one_epoch(model, pt_train_loader, criterion, optimizer, None, config["device"], epoch, use_physics=False)
        val_metrics = evaluate(model, pt_val_loader, criterion, config["device"], use_physics=False)
        
        pt_history["train_loss"].append(train_metrics["total"])
        pt_history["val_loss"].append(val_metrics["total"])
        print(f"  Pretrain Epoch {epoch}: Train Loss={train_metrics['total']:.4f} | Val Loss={val_metrics['total']:.4f}")
    
    # 保存预训练模型
    pretrained_path = output_path / "pretrained_model.pt"
    torch.save(model.state_dict(), pretrained_path)
    print(f"预训练模型已保存: {pretrained_path}")
    
    # ==========================================
    # Phase 2: 微调 (Fine-tuning)
    # ==========================================
    print(f"\n\n[Phase 2] 开始微调: 下游补缺任务")
    print(f"配置: mask_rate={finetune_config['mask_rate']}, drift_scale={finetune_config['drift_scale']}")
    
    # 加载微调数据
    ft_train_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=finetune_config["mask_rate"],
        missing_mode=finetune_config["missing_mode"],
        split="train",
        eval_mode=False,
        drift_scale=finetune_config["drift_scale"],
    )
    ft_val_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=finetune_config["mask_rate"],
        missing_mode=finetune_config["missing_mode"],
        split="val",
        eval_mode=True,
        drift_scale=0.0,
    )
    
    ft_train_loader = torch.utils.data.DataLoader(ft_train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    ft_val_loader = torch.utils.data.DataLoader(ft_val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
    # 重置优化器（通常微调使用较小LR）
    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_config["lr"])
    
    ft_history = {"val_mse_masked": [], "val_mse_all": []}
    best_ft_loss = float('inf')
    
    for epoch in range(1, finetune_config["epochs"] + 1):
        train_metrics = train_one_epoch(model, ft_train_loader, criterion, optimizer, None, config["device"], epoch, use_physics=False)
        val_metrics = evaluate(model, ft_val_loader, criterion, config["device"], use_physics=False)
        
        ft_history["val_mse_masked"].append(val_metrics["mse_masked"])
        ft_history["val_mse_all"].append(val_metrics["mse_all"])
        
        if val_metrics["total"] < best_ft_loss:
            best_ft_loss = val_metrics["total"]
            torch.save(model.state_dict(), output_path / "best_finetuned_model.pt")
            
        print(f"  Finetune Epoch {epoch}: Val MSE(masked)={val_metrics['mse_masked']:.4f}")

    # ==========================================
    # Phase 3: 基准对比 (Baseline: No Pretraining)
    # ==========================================
    print(f"\n\n[Phase 3] 基准测试: 无预训练直接训练")
    
    # 初始化新模型
    baseline_model = PhysicsAwareIMUImputer(
        input_dim=13,
        hidden_units=config["hidden_units"],
        output_dim=6,
        use_physics_prior=True,
        mixed_memory=True,
    ).to(config["device"])
    
    # 使用与微调相同的配置（但LR可能可以用标准的1e-3，为了公平这里用config中较大的lr或者finetune的lr？
    # 通常从头训练需要较大LR。我们使用1e-3作为从头训练的标准）
    optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-3)
    
    bl_history = {"val_mse_masked": [], "val_mse_all": []}
    best_bl_loss = float('inf')
    
    for epoch in range(1, finetune_config["epochs"] + 1):
        train_metrics = train_one_epoch(baseline_model, ft_train_loader, criterion, optimizer, None, config["device"], epoch, use_physics=False)
        val_metrics = evaluate(baseline_model, ft_val_loader, criterion, config["device"], use_physics=False)
        
        bl_history["val_mse_masked"].append(val_metrics["mse_masked"])
        bl_history["val_mse_all"].append(val_metrics["mse_all"])
        
        if val_metrics["total"] < best_bl_loss:
            best_bl_loss = val_metrics["total"]
            torch.save(baseline_model.state_dict(), output_path / "best_baseline_model.pt")
            
        print(f"  Baseline Epoch {epoch}: Val MSE(masked)={val_metrics['mse_masked']:.4f}")

    # ==========================================
    # 结果展示
    # ==========================================
    print(f"\n\n{'='*80}")
    print("实验结果对比")
    print(f"{'='*80}")
    print(f"微调模型 (Pretrained + Finetuned) 最终 MSE (masked): {ft_history['val_mse_masked'][-1]:.4f}")
    print(f"基准模型 (Scratch)               最终 MSE (masked): {bl_history['val_mse_masked'][-1]:.4f}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    epochs = range(1, finetune_config["epochs"] + 1)
    plt.plot(epochs, ft_history['val_mse_masked'], 'r-', label='Pretrained + Finetuned', linewidth=2)
    plt.plot(epochs, bl_history['val_mse_masked'], 'b--', label='Train from Scratch', linewidth=2)
    plt.xlabel('Finetuning Epochs')
    plt.ylabel('MSE (Masked)')
    plt.title('Effect of Pretraining on Imputation Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "pretraining_comparison.png")
    print(f"对比图已保存至: {output_path / 'pretraining_comparison.png'}")

if __name__ == "__main__":
    run_pretraining_experiment()
