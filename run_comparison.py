"""
批量运行多个模型对比实验
"""
import torch
from train import train
from visualization import plot_multi_model_comparison, create_comparison_table
import os

def run_all_models(
    root_dir: str = "Oxford Dataset",
    epochs: int = 50,
    output_dir: str = "comparison_results",
):
    """
    运行所有baseline模型并对比
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义模型列表
    models_config = {
        "CfC": {"model_name": "cfc", "hidden_units": 64, "lr": 1e-3},
        "Physics-CfC": {"model_name": "physics", "hidden_units": 64, "lr": 1e-3},
        "GRU": {"model_name": "gru", "hidden_units": 128, "lr": 1e-3},
        "Transformer": {"model_name": "transformer", "hidden_units": 128, "lr": 5e-4},
    }
    
    all_results = {}
    all_histories = {}
    param_counts = {}
    
    print("="*80)
    print("批量训练对比实验")
    print("="*80)
    
    for display_name, config in models_config.items():
        print(f"\n{'='*80}")
        print(f"开始训练: {display_name}")
        print(f"{'='*80}\n")
        
        try:
            model, history, multi_results, num_params = train(
                root_dir=root_dir,
                epochs=epochs,
                model_name=config["model_name"],
                hidden_units=config["hidden_units"],
                lr=config["lr"],
                output_dir=output_dir,
                w_recon=1.0,
                w_consistency=0.5 if config["model_name"] in ["cfc", "physics"] else 0.0,
                w_smooth=0.1 if config["model_name"] in ["cfc", "physics"] else 0.0,
                w_physics_integration=0.2 if config["model_name"] == "physics" else 0.0,
                w_physics_energy=0.1 if config["model_name"] == "physics" else 0.0,
            )
            
            all_results[display_name] = multi_results
            all_histories[display_name] = history
            param_counts[display_name] = num_params
            
            print(f"\n✓ {display_name} 训练完成! 参数量: {num_params:,}")
            
        except Exception as e:
            print(f"\n✗ {display_name} 训练失败: {e}")
            continue
    
    # 生成对比图表
    print(f"\n{'='*80}")
    print("生成对比结果...")
    print(f"{'='*80}\n")
    
    # 1. 多模型对比图
    plot_multi_model_comparison(
        all_results,
        save_path=os.path.join(output_dir, "all_models_comparison.png")
    )
    
    # 2. 对比表格
    df = create_comparison_table(
        all_results,
        param_counts,
        save_path=os.path.join(output_dir, "comparison_table.csv")
    )
    
    print("\n对比结果:")
    print(df.to_string(index=False))
    
    # 3. 保存完整结果
    torch.save({
        "results": all_results,
        "histories": all_histories,
        "param_counts": param_counts,
    }, os.path.join(output_dir, "all_results.pt"))
    
    print(f"\n✓ 所有结果已保存到: {output_dir}/")
    
    return all_results, all_histories, param_counts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="Oxford Dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="comparison_results")
    args = parser.parse_args()
    
    run_all_models(
        root_dir=args.root_dir,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )