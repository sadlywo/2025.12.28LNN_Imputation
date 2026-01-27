"""实验4：块状丢失模式下的性能对比实验"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# 导入项目模块
from dataset import CfCIMUDataset
from models import build_model
from models import AdaptiveLoss, ReconstructionOnlyLoss
from train import train_one_epoch, evaluate, evaluate_multi_missing_rates
from visualization import plot_training_curves, plot_imputation_samples
from experiment_multi_methods import SimpleImputer


def experiment_block_missing():
    """
    对比不同方法在块状丢失模式下的性能
    - 块状丢失：一个通道内连续丢失数据
    - 比较方法：LNN（提出的方法）、GRU、LOCF、均值插补
    """
    # 实验配置
    config = {
        "root_dir": "Oxford Dataset",
        "seq_len": 50,
        "mask_rate": 0.3,
        "missing_mode": "block",  # 块状丢失模式
        "batch_size": 16,
        "epochs": 50,
        "lr": 1e-3,
        "hidden_units": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": "results/block_missing_experiment",
        "num_workers": 4,
    }
    
    # 创建输出目录
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*80}")
    print(f"开始实验：块状丢失模式性能对比")
    print(f"{'='*80}")
    print(f"设备: {config['device']}")
    print(f"输出目录: {config['output_dir']}")
    print(f"缺失模式: {config['missing_mode']}")
    print(f"{'='*80}\n")
    
    # 加载数据集
    print("加载数据集...")
    train_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=config["mask_rate"],
        missing_mode=config["missing_mode"],  # 块状丢失模式
        split="train",
        eval_mode=False,
        drift_scale=0.01,
    )
    
    val_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=config["mask_rate"],
        missing_mode=config["missing_mode"],  # 块状丢失模式
        split="val",
        eval_mode=True,
    )
    
    test_ds = CfCIMUDataset(
        root_dir=config["root_dir"],
        seq_len=config["seq_len"],
        mask_rate=config["mask_rate"],
        missing_mode=config["missing_mode"],  # 块状丢失模式
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
    
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True if config["device"] == "cuda" else False,
    )
    
    # 深度学习模型列表
    models_to_test = ["lnn", "gru"]
    model_results = {}
    
    # 训练和评估深度学习模型
    print(f"\n{'='*80}")
    print("训练深度学习模型...")
    print(f"{'='*80}")
    
    for model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"训练模型: {model_name}")
        print(f"{'='*80}")
        
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
        
        # 加载最佳模型进行最终评估
        model.load_state_dict(torch.load(model_path))
        final_metrics = evaluate(
            model, test_loader, criterion, config["device"], use_physics=False
        )
        
        # 保存模型结果
        model_results[model_name] = {
            "history": history,
            "best_val_loss": best_val_loss,
            "final_mse_masked": final_metrics["mse_masked"],
            "final_mse_all": final_metrics["mse_all"],
        }
        
        print(f"\n[{model_name}] 训练完成")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        print(f"最终MSE (masked): {final_metrics['mse_masked']:.4f}")
        print(f"最终MSE (all): {final_metrics['mse_all']:.4f}")
    
    # 评估简单插补方法
    print(f"\n{'='*80}")
    print("评估简单插补方法...")
    print(f"{'='*80}")
    
    # 准备测试数据用于简单插补方法评估
    test_data = []
    test_targets = []
    test_masks = []
    
    for i in range(len(test_ds)):
        inputs, targets, mask = test_ds[i]
        test_data.append(inputs.numpy())
        test_targets.append(targets.numpy())
        test_masks.append(mask.numpy())
    
    test_data = np.array(test_data)  # (N, T, 13)
    test_targets = np.array(test_targets)  # (N, T, 6)
    test_masks = np.array(test_masks)  # (N, T, 6)
    
    # 提取IMU数据（前6个通道）
    imu_data = test_data[:, :, :6]  # (N, T, 6) - 带缺失值的IMU数据
    
    # 评估LOCF方法
    locf_imputed = SimpleImputer.locf(imu_data.copy(), test_masks)
    locf_mse = np.mean(((locf_imputed - test_targets) ** 2 * (1 - test_masks)).sum() / ((1 - test_masks).sum() + 1e-8))
    print(f"LOCF MSE (masked): {locf_mse:.4f}")
    
    # 评估均值插补方法
    mean_imputed = SimpleImputer.mean_imputation(imu_data.copy(), test_masks)
    mean_mse = np.mean(((mean_imputed - test_targets) ** 2 * (1 - test_masks)).sum() / ((1 - test_masks).sum() + 1e-8))
    print(f"均值插补 MSE (masked): {mean_mse:.4f}")
    
    # 保存简单方法结果
    simple_results = {
        "locf": {
            "final_mse_masked": locf_mse,
            "final_mse_all": np.mean(((locf_imputed - test_targets) ** 2).sum() / (test_targets.size + 1e-8)),
        },
        "mean": {
            "final_mse_masked": mean_mse,
            "final_mse_all": np.mean(((mean_imputed - test_targets) ** 2).sum() / (test_targets.size + 1e-8)),
        },
    }
    
    # 在不同块大小下评估模型性能
    print(f"\n{'='*80}")
    print("在不同块大小下评估模型性能...")
    print(f"{'='*80}")
    
    # 不同块大小配置
    block_sizes = [5, 10, 15, 20]  # 连续丢失的时间步数
    block_results = {}
    
    for model_name in models_to_test:
        print(f"\n评估 {model_name} 在不同块大小下的性能...")
        
        # 加载最佳模型
        model = build_model(
            model_name=model_name,
            input_dim=13,
            hidden_dim=config["hidden_units"],
            output_dim=6,
        ).to(config["device"])
        model_path = output_path / f"best_model_{model_name}.pt"
        model.load_state_dict(torch.load(model_path))
        
        model_block_results = {}
        
        for block_size in block_sizes:
            # 创建自定义块状丢失数据集
            custom_test_ds = CfCIMUDataset(
                root_dir=config["root_dir"],
                seq_len=config["seq_len"],
                mask_rate=config["mask_rate"],
                missing_mode="block",
                split="val",
                eval_mode=True,
            )
            
            # 重写缺失模式生成函数，使用固定块大小
            original___getitem__ = custom_test_ds.__getitem__
            
            def custom_getitem(idx):
                inputs, targets, mask = original___getitem__(idx)
                # 重新生成块状丢失掩码
                new_mask = torch.ones_like(mask)
                B, T, C = new_mask.shape
                
                for c in range(C):
                    # 每个通道随机生成一个连续块
                    start = torch.randint(0, max(1, T - block_size + 1), (1,)).item()
                    new_mask[:, start:start + block_size, c] = 0.0
                
                # 重新生成输入数据
                masked_imu = targets * new_mask
                new_inputs = torch.cat([masked_imu, new_mask, inputs[:, :, -1:]], dim=-1)
                
                return new_inputs, targets, new_mask
            
            custom_test_ds.__getitem__ = custom_getitem
            
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
            print(f"块大小 {block_size}: MSE (masked) = {mse_masked:.4f}")
        
        block_results[model_name] = model_block_results
    
    # 生成对比报告
    generate_comparison_report(model_results, simple_results, block_results, output_path, timestamp, config)
    
    print(f"\n{'='*80}")
    print("实验4：块状丢失模式性能对比完成")
    print(f"结果保存至: {config['output_dir']}")
    print(f"{'='*80}")
    
    return model_results, simple_results, block_results


def generate_comparison_report(model_results, simple_results, block_results, output_path, timestamp, config):
    """生成对比报告"""
    print(f"\n{'='*80}")
    print("生成对比报告")
    print(f"{'='*80}")
    
    # 方法名称映射
    method_name_map = {
        "lnn": "LNN (Proposed)",
        "gru": "GRU",
        "locf": "LOCF",
        "mean": "Mean Imputation",
    }
    
    # 收集所有方法的结果
    all_results = {}
    
    # 添加深度学习方法结果
    for method, result in model_results.items():
        all_results[method] = {
            "mse_masked": result["final_mse_masked"],
            "mse_all": result["final_mse_all"],
            "type": "deep",
        }
    
    # 添加简单方法结果
    for method, result in simple_results.items():
        all_results[method] = {
            "mse_masked": result["final_mse_masked"],
            "mse_all": result["final_mse_all"],
            "type": "simple",
        }
    
    # 生成总结表格
    summary_data = []
    for method, result in all_results.items():
        summary_data.append({
            "方法": method_name_map.get(method, method),
            "MSE (masked)": f"{result['mse_masked']:.4f}",
            "MSE (all)": f"{result['mse_all']:.4f}",
            "类型": "深度学习方法" if result['type'] == "deep" else "简单方法",
        })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    # 保存总结表格
    summary_path = output_path / f"summary_block_missing_{timestamp}.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n总结表格保存至: {summary_path}")
    
    # 生成不同块大小的对比表格
    if block_results:
        block_data = []
        block_sizes = list(next(iter(block_results.values())).keys())
        
        for block_size in block_sizes:
            row = {"块大小": block_size}
            for method, results in block_results.items():
                if block_size in results:
                    row[f"{method_name_map.get(method, method)}_MSE_masked"] = f"{results[block_size]:.4f}"
            block_data.append(row)
        
        df_block = pd.DataFrame(block_data)
        print(f"\n{'='*80}")
        print("不同块大小下的MSE对比")
        print(f"{'='*80}")
        print(df_block.to_string(index=False))
        
        # 保存块大小对比表格
        block_path = output_path / f"block_size_comparison_{timestamp}.csv"
        df_block.to_csv(block_path, index=False)
        print(f"\n块大小对比表格保存至: {block_path}")
    
    # 绘制对比图
    import matplotlib.pyplot as plt
    
    # 1. 不同方法的MSE对比
    plt.figure(figsize=(12, 6))
    
    methods = list(all_results.keys())
    mse_values = [all_results[m]["mse_masked"] for m in methods]
    method_labels = [method_name_map.get(m, m) for m in methods]
    
    bars = plt.bar(method_labels, mse_values, capsize=5)
    plt.title("MSE Comparison Under Block Missingness")
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
    comparison_plot_path = output_path / f"comparison_block_methods_{timestamp}.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"对比图保存至: {comparison_plot_path}")
    
    # 2. 不同块大小的MSE对比
    if block_results:
        plt.figure(figsize=(12, 6))
        
        for method, results in block_results.items():
            block_sizes = list(results.keys())
            mse_values = list(results.values())
            plt.plot(block_sizes, mse_values, marker='o', linewidth=2, 
                     label=method_name_map.get(method, method))
        
        plt.title("MSE vs. Block Size")
        plt.xlabel("Block Size")
        plt.ylabel("MSE (masked)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        block_size_plot_path = output_path / f"comparison_block_sizes_{timestamp}.png"
        plt.savefig(block_size_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"块大小对比图保存至: {block_size_plot_path}")


if __name__ == "__main__":
    # 运行实验
    experiment_block_missing()
