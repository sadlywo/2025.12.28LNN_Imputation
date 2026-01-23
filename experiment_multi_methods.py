"""实验3：多种插补方法的对比实验"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 导入项目模块
from dataset import CfCIMUDataset
from models import LNNImputer, GRUImputer, build_model
from models import ReconstructionOnlyLoss, AdaptiveLoss
from train import train_one_epoch, evaluate


class SimpleImputer:
    """简单插补方法实现"""
    
    @staticmethod
    def locf(data, mask):
        """
        Last Observation Carried Forward (LOCF)插补方法
        Args:
            data: (B, T, C) - 输入数据，包含缺失值（缺失值用0表示，由mask指示）
            mask: (B, T, C) - 掩码，1表示观察值，0表示缺失值
        Returns:
            imputed_data: (B, T, C) - 插补后的数据
        """
        imputed = data.copy()
        B, T, C = data.shape
        
        for b in range(B):
            for c in range(C):
                for t in range(1, T):
                    if mask[b, t, c] == 0:  # 如果当前值缺失
                        imputed[b, t, c] = imputed[b, t-1, c]  # 使用前一个观察值
        return imputed
    
    @staticmethod
    def mean_imputation(data, mask):
        """
        均值插补方法
        Args:
            data: (B, T, C) - 输入数据
            mask: (B, T, C) - 掩码
        Returns:
            imputed_data: (B, T, C) - 插补后的数据
        """
        imputed = data.copy()
        B, T, C = data.shape
        
        for c in range(C):
            # 计算每个通道的均值
            channel_data = data[:, :, c]
            channel_mask = mask[:, :, c]
            valid_values = channel_data[channel_mask == 1]
            if len(valid_values) > 0:
                mean_val = valid_values.mean()
                # 填充缺失值
                imputed[:, :, c][channel_mask == 0] = mean_val
        return imputed


def kfold_cross_validation(X, y, mask, k=10):
    """
    10折交叉验证
    Args:
        X: (N, T, C) - 输入数据
        y: (N, T, C) - 真实值
        mask: (N, T, C) - 掩码
        k: 折数
    Returns:
        results: dict - 各方法的交叉验证结果
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = {
        "locf": [],
        "mean": [],
        "lnn": [],
        "gru": [],
    }
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n折 {fold+1}/{k}")
        
        # 分割数据
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        mask_train, mask_test = mask[train_idx], mask[test_idx]
        
        # LOCF插补
        locf_imputed = SimpleImputer.locf(X_test.copy(), mask_test)
        locf_mse = mean_squared_error(y_test[mask_test == 0], locf_imputed[mask_test == 0])
        results["locf"].append(locf_mse)
        print(f"LOCF MSE: {locf_mse:.4f}")
        
        # 均值插补
        mean_imputed = SimpleImputer.mean_imputation(X_test.copy(), mask_test)
        mean_mse = mean_squared_error(y_test[mask_test == 0], mean_imputed[mask_test == 0])
        results["mean"].append(mean_mse)
        print(f"均值插补 MSE: {mean_mse:.4f}")
        
        # TODO: 实现其他方法（TRMF、GRU-D、RITS、GAIN）
        # 这里只实现了LOCF和均值插补作为示例
        # 实际使用中需要添加其他方法的实现
    
    return results


def experiment_multi_methods():
    """
    对比多种插补方法的性能
    - LNN（提出的方法）
    - LOCF（Last Observation Carried Forward）
    - TRMF（Temporal Regularized Matrix Factorization）
    - GRU-D（GRU with missingness indicators）
    - RITS（Recurrent Imputation for Time Series）
    - GAIN（Generative Adversarial Imputation Networks）
    """
    # 实验配置
    config = {
        "root_dir": "Oxford Dataset",
        "seq_len": 50,
        "mask_rate": 0.3,
        "missing_mode": "random",
        "batch_size": 16,
        "epochs": 50,
        "lr": 1e-3,
        "hidden_units": 64,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": "results/multi_methods_experiment",
        "num_workers": 4,
        "n_splits": 10,  # 10折交叉验证
    }
    
    # 创建输出目录
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*80}")
    print(f"开始实验：多种插补方法对比")
    print(f"{'='*80}")
    print(f"设备: {config['device']}")
    print(f"输出目录: {config['output_dir']}")
    print(f"交叉验证折数: {config['n_splits']}")
    print(f"{'='*80}\n")
    
    # 加载完整数据集
    print("加载数据集...")
    full_dataset = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=config["mask_rate"],
        missing_mode=config["missing_mode"],
        split="train",
        split_ratio=1.0,  # 加载所有数据
        eval_mode=True,  # 固定掩码
    )
    
    # 准备数据用于交叉验证
    print("准备数据用于交叉验证...")
    all_inputs = []
    all_targets = []
    all_masks = []
    
    for i in range(len(full_dataset)):
        inputs, targets, mask = full_dataset[i]
        all_inputs.append(inputs.numpy())
        all_targets.append(targets.numpy())
        all_masks.append(mask.numpy())
    
    all_inputs = np.array(all_inputs)  # (N, T, 13) - 13 = [masked_imu(6), mask(6), dt(1)]
    all_targets = np.array(all_targets)  # (N, T, 6) - 6 = [gyro(3), acc(3)]
    all_masks = np.array(all_masks)  # (N, T, 6)
    
    # 提取IMU数据（前6个通道）
    imu_data = all_inputs[:, :, :6]  # (N, T, 6) - 带缺失值的IMU数据
    
    # 运行10折交叉验证
    print(f"\n开始{config['n_splits']}折交叉验证...")
    cv_results = kfold_cross_validation(imu_data, all_targets, all_masks, k=config['n_splits'])
    
    # 训练和评估深度学习模型
    print(f"\n{'='*80}")
    print("训练深度学习模型...")
    print(f"{'='*80}")
    
    # 模型列表
    models_to_test = ["lnn", "gru"]
    model_results = {}
    
    for model_name in models_to_test:
        print(f"\n训练 {model_name} 模型...")
        
        # 构建模型
        model = build_model(
            model_name=model_name,
            input_dim=13,
            hidden_dim=config["hidden_units"],
            output_dim=6,
        ).to(config["device"])
        
        # 定义损失函数和优化器
        if model_name == "lnn":
            criterion = AdaptiveLoss(
                w_recon=1.0,
                w_consistency=0.1,
                w_smooth=0.01,
            )
        else:
            criterion = ReconstructionOnlyLoss(w_recon=1.0)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=1e-5
        )
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True if config["device"] == "cuda" else False,
        )
        
        val_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True if config["device"] == "cuda" else False,
        )
        
        # 训练循环
        best_val_loss = float("inf")
        for epoch in range(1, config["epochs"] + 1):
            # 训练
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, None, 
                config["device"], epoch, use_physics=False
            )
            
            # 验证
            val_metrics = evaluate(
                model, val_loader, criterion, config["device"], use_physics=False
            )
            
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                model_path = output_path / f"best_model_{model_name}.pt"
                torch.save(model.state_dict(), model_path)
        
        # 加载最佳模型并评估
        model.load_state_dict(torch.load(model_path))
        final_metrics = evaluate(
            model, val_loader, criterion, config["device"], use_physics=False
        )
        
        model_results[model_name] = {
            "mse_masked": final_metrics["mse_masked"],
            "mse_all": final_metrics["mse_all"],
        }
        
        print(f"{model_name} 模型评估结果:")
        print(f"  MSE (masked): {final_metrics['mse_masked']:.4f}")
        print(f"  MSE (all): {final_metrics['mse_all']:.4f}")
    
    # 生成综合对比报告
    generate_comparison_report(cv_results, model_results, output_path, timestamp, config)
    
    print(f"\n{'='*80}")
    print("实验3：多种插补方法对比实验完成")
    print(f"结果保存至: {config['output_dir']}")
    print(f"{'='*80}")
    
    return cv_results, model_results


def generate_comparison_report(cv_results, model_results, output_path, timestamp, config):
    """生成对比报告"""
    print(f"\n{'='*80}")
    print("生成对比报告")
    print(f"{'='*80}")
    
    # 方法名称映射
    method_name_map = {
        "locf": "LOCF",
        "mean": "均值插补",
        "lnn": "LNN（提出的方法）",
        "gru": "GRU",
        # TODO: 添加其他方法的名称映射
    }
    
    # 收集所有方法的结果
    all_results = {}
    
    # 添加简单方法的交叉验证结果
    for method, results in cv_results.items():
        all_results[method] = {
            "mse": np.mean(results),
            "std": np.std(results),
            "type": "simple",
        }
    
    # 添加深度学习方法的结果
    for method, results in model_results.items():
        all_results[method] = {
            "mse": results["mse_masked"],
            "std": 0.0,  # 深度学习方法没有交叉验证的标准差
            "type": "deep",
        }
    
    # 生成总结表格
    summary_data = []
    for method, result in all_results.items():
        summary_data.append({
            "方法": method_name_map.get(method, method),
            "平均MSE": f"{result['mse']:.4f}",
            "标准差": f"{result['std']:.4f}",
            "类型": "简单方法" if result['type'] == "simple" else "深度学习方法",
        })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    # 保存总结表格
    summary_path = output_path / f"summary_multi_methods_{timestamp}.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n总结表格保存至: {summary_path}")
    
    # 生成详细的交叉验证结果表格
    if cv_results:
        cv_data = []
        for split in range(config["n_splits"]):
            row = {"折数": split + 1}
            for method, results in cv_results.items():
                if split < len(results):
                    row[f"{method_name_map.get(method, method)}_MSE"] = f"{results[split]:.4f}"
            cv_data.append(row)
        
        df_cv = pd.DataFrame(cv_data)
        cv_path = output_path / f"cv_results_{timestamp}.csv"
        df_cv.to_csv(cv_path, index=False)
        print(f"交叉验证详细结果保存至: {cv_path}")
    
    # 绘制对比图
    import matplotlib.pyplot as plt
    
    methods = list(all_results.keys())
    mse_values = [all_results[m]["mse"] for m in methods]
    std_values = [all_results[m]["std"] for m in methods]
    method_labels = [method_name_map.get(m, m) for m in methods]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(method_labels, mse_values, yerr=std_values, capsize=5)
    plt.title("不同插补方法的MSE对比")
    plt.xlabel("插补方法")
    plt.ylabel("平均MSE")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    
    # 在柱状图上显示数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    comparison_plot_path = output_path / f"comparison_multi_methods_{timestamp}.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"对比图保存至: {comparison_plot_path}")


if __name__ == "__main__":
    # 运行实验
    experiment_multi_methods()
