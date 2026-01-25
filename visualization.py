"""Visualization utilities for training results and comparisons."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_curves(history: dict, save_path: str = "training_curves.png"):
    """
    绘制训练过程中的Loss曲线
    
    Args:
        history: 训练历史字典,包含 train_loss, val_loss, val_mse_all, val_mse_masked
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    # 1. Total Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r--', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Total Loss', fontsize=12)
    axes[0, 0].set_title('训练/验证损失曲线', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. MSE (All channels)
    axes[0, 1].plot(epochs, history['val_mse_all'], 'g-', label='MSE (All)', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('MSE', fontsize=12)
    axes[0, 1].set_title('全通道MSE变化', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. MSE (Masked only)
    axes[1, 0].plot(epochs, history['val_mse_masked'], 'm-', label='MSE (Masked)', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('MSE', fontsize=12)
    axes[1, 0].set_title('缺失位置MSE变化', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Loss comparison
    axes[1, 1].plot(epochs, history['train_loss'], 'b-', label='Train', alpha=0.7, linewidth=2)
    axes[1, 1].plot(epochs, history['val_loss'], 'r-', label='Val', alpha=0.7, linewidth=2)
    axes[1, 1].plot(epochs, history['val_mse_masked'], 'g-', label='MSE(Masked)', alpha=0.7, linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Value', fontsize=12)
    axes[1, 1].set_title('综合对比', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Visualization] Training curves saved to {save_path}")
    plt.close()


def plot_multi_model_comparison(results_dict: dict, save_path: str = "model_comparison.png"):
    """
    绘制多个模型在不同缺失模式下的对比
    
    Args:
        results_dict: {model_name: multi_results} 字典
        save_path: 保存路径
    """
    patterns = ["random", "block", "channel"]
    rates = [10, 20, 30, 40, 50]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, pattern in enumerate(patterns):
        ax = axes[idx]
        
        for model_name, multi_results in results_dict.items():
            mse_values = []
            for rate in rates:
                key = f"{pattern}_{rate}"
                if key in multi_results:
                    mse_values.append(multi_results[key]['mse_masked'])
                else:
                    mse_values.append(np.nan)
            
            ax.plot(rates, mse_values, marker='o', linewidth=2, markersize=8, label=model_name)
        
        ax.set_xlabel('Missing Rate (%)', fontsize=12)
        ax.set_ylabel('MSE (Masked)', fontsize=12)
        ax.set_title(f'{pattern.capitalize()} Missing Pattern', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(rates)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Visualization] Model comparison saved to {save_path}")
    plt.close()


def plot_imputation_samples(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    mask: torch.Tensor,
    num_samples: int = 3,
    save_path: str = "imputation_samples.png"
):
    """
    可视化插补结果样本
    
    Args:
        inputs, targets, predictions, mask: 数据张量 (B, T, 6)
        num_samples: 显示样本数量
        save_path: 保存路径
    """
    num_samples = min(num_samples, inputs.shape[0])
    channel_names = ['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']
    
    fig, axes = plt.subplots(num_samples, 6, figsize=(20, num_samples * 3))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(num_samples):
        for ch_idx in range(6):
            ax = axes[sample_idx, ch_idx]
            
            t = np.arange(targets.shape[1])
            target_vals = targets[sample_idx, :, ch_idx].cpu().numpy()
            pred_vals = predictions[sample_idx, :, ch_idx].cpu().numpy()
            mask_vals = mask[sample_idx, :, ch_idx].cpu().numpy()
            
            # Ground truth
            ax.plot(t, target_vals, 'k-', label='Ground Truth', linewidth=2, alpha=0.7)
            
            # Predictions (only on masked positions)
            masked_t = t[mask_vals == 0]
            masked_pred = pred_vals[mask_vals == 0]
            ax.scatter(masked_t, masked_pred, c='r', s=30, label='Predicted', zorder=5)
            
            # Observed points
            obs_t = t[mask_vals == 1]
            obs_vals = target_vals[mask_vals == 1]
            ax.scatter(obs_t, obs_vals, c='b', s=20, label='Observed', alpha=0.5, zorder=4)
            
            ax.set_xlabel('Time Step', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.set_title(f'Sample {sample_idx+1} - {channel_names[ch_idx]}', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Visualization] Imputation samples saved to {save_path}")
    plt.close()


def create_comparison_table(
    results_dict: dict,
    param_counts: dict,
    save_path: str = "comparison_table.csv"
):
    """
    创建模型对比表格(CSV + Markdown)
    
    Args:
        results_dict: {model_name: multi_results}
        param_counts: {model_name: num_params}
        save_path: 保存路径
    """
    rows = []
    
    for model_name in results_dict.keys():
        multi_results = results_dict[model_name]
        params = param_counts.get(model_name, 0)
        
        # 计算各模式平均MSE
        random_mse = np.mean([v['mse_masked'] for k, v in multi_results.items() if k.startswith('random')])
        block_mse = np.mean([v['mse_masked'] for k, v in multi_results.items() if k.startswith('block')])
        channel_mse = np.mean([v['mse_masked'] for k, v in multi_results.items() if k.startswith('channel')])
        overall_mse = np.mean([v['mse_masked'] for v in multi_results.values()])
        
        rows.append({
            'Model': model_name,
            'Parameters': params,
            'Random MSE': f"{random_mse:.4f}",
            'Block MSE': f"{block_mse:.4f}",
            'Channel MSE': f"{channel_mse:.4f}",
            'Overall MSE': f"{overall_mse:.4f}"
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"[Comparison] Table saved to {save_path}")
    
    # 同时保存Markdown格式
    md_path = save_path.replace('.csv', '.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(df.to_markdown(index=False))
    print(f"[Comparison] Markdown table saved to {md_path}")
    
    return df