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
    )"""Complete comparison experiment script."""
import torch
import json
import time
from datetime import datetime
from pathlib import Path
from train import train, evaluate_multi_missing_rates
from models import build_model
import pandas as pd


def run_comparison_experiment(
    root_dir: str = "Oxford Dataset",
    seq_len: int = 50,
    mask_rate: float = 0.3,
    missing_mode: str = "random",
    batch_size: int = 16,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden_units: int = 64,
    device: str = "cuda",
    output_dir: str = "comparison_results",
):
    """
    Run comparison experiment across all models.
    
    Models to compare:
    - CfC (physics-aware)
    - LNN (alias for CfC with physics prior)
    - Physics (alias for CfC with physics prior)
    - GRU (baseline)
    - Transformer (baseline)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Models to compare
    models_to_test = ["cfc", "gru", "transformer"]
    
    # Results storage
    all_results = {}
    training_times = {}
    
    print("="*80)
    print("STARTING COMPARISON EXPERIMENT")
    print("="*80)
    print(f"Dataset: {root_dir}")
    print(f"Sequence Length: {seq_len}")
    print(f"Training Missing Mode: {missing_mode} @ {mask_rate*100:.0f}%")
    print(f"Epochs: {epochs}")
    print(f"Hidden Units: {hidden_units}")
    print(f"Models: {', '.join(models_to_test)}")
    print("="*80 + "\n")
    
    # Train each model
    for model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL: {model_name.upper()}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            # Set loss weights based on model type
            if model_name in ["cfc", "lnn", "physics"]:
                w_recon = 1.0
                w_consistency = 0.1
                w_smooth = 0.01
            else:
                w_recon = 1.0
                w_consistency = 0.0
                w_smooth = 0.0
            
            # Train model
            model, history, multi_results = train(
                root_dir=root_dir,
                seq_len=seq_len,
                mask_rate=mask_rate,
                missing_mode=missing_mode,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                device=device,
                model_name=model_name,
                hidden_units=hidden_units,
                w_recon=w_recon,
                w_consistency=w_consistency,
                w_smooth=w_smooth,
                num_workers=4,
                use_scheduler=True,
            )
            
            training_time = time.time() - start_time
            training_times[model_name] = training_time
            
            # Save model
            model_path = output_path / f"{model_name}_best_model.pt"
            torch.save(model.state_dict(), model_path)
            print(f"\n[Saved] Model weights to {model_path}")
            
            # Store results
            all_results[model_name] = {
                "history": history,
                "multi_pattern_results": multi_results,
                "training_time": training_time,
                "final_val_mse_masked": history["val_mse_masked"][-1],
                "final_val_mse_all": history["val_mse_all"][-1],
            }
            
            print(f"\n[{model_name.upper()}] Training completed in {training_time/60:.2f} minutes")
            print(f"[{model_name.upper()}] Final MSE (masked): {history['val_mse_masked'][-1]:.4f}")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {"error": str(e)}
    
    # Generate comparison report
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORT")
    print("="*80 + "\n")
    
    generate_comparison_report(all_results, training_times, output_path, timestamp)
    
    # Save raw results
    results_file = output_path / f"raw_results_{timestamp}.pt"
    torch.save(all_results, results_file)
    print(f"\n[Saved] Raw results to {results_file}")
    
    print("\n" + "="*80)
    print("COMPARISON EXPERIMENT COMPLETED")
    print("="*80)
    print(f"Results saved in: {output_path}")
    
    return all_results


def generate_comparison_report(results, training_times, output_path, timestamp):
    """Generate comparison report in multiple formats."""
    
    # 1. Summary Table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    summary_data = []
    for model_name, result in results.items():
        if "error" in result:
            continue
        
        summary_data.append({
            "Model": model_name.upper(),
            "Training Time (min)": f"{training_times[model_name]/60:.2f}",
            "Final MSE (all)": f"{result['final_val_mse_all']:.4f}",
            "Final MSE (masked)": f"{result['final_val_mse_masked']:.4f}",
            "Best Val Loss": f"{min(result['history']['val_loss']):.4f}",
        })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    # Save summary CSV
    summary_csv = output_path / f"summary_{timestamp}.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\n[Saved] Summary table to {summary_csv}")
    
    # 2. Multi-Pattern Performance
    print("\n" + "="*80)
    print("MULTI-PATTERN EVALUATION")
    print("="*80)
    
    multi_pattern_data = []
    for model_name, result in results.items():
        if "error" in result or "multi_pattern_results" not in result:
            continue
        
        for key, metrics in result["multi_pattern_results"].items():
            pattern, rate = key.rsplit("_", 1)
            multi_pattern_data.append({
                "Model": model_name.upper(),
                "Pattern": pattern,
                "Missing Rate": f"{rate}%",
                "MSE (all)": f"{metrics['mse_all']:.4f}",
                "MSE (masked)": f"{metrics['mse_masked']:.4f}",
            })
    
    if multi_pattern_data:
        df_multi = pd.DataFrame(multi_pattern_data)
        print(df_multi.to_string(index=False))
        
        # Save multi-pattern CSV
        multi_csv = output_path / f"multi_pattern_{timestamp}.csv"
        df_multi.to_csv(multi_csv, index=False)
        print(f"\n[Saved] Multi-pattern results to {multi_csv}")
    
    # 3. Best Model Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    valid_models = {k: v for k, v in results.items() if "error" not in v}
    if valid_models:
        best_model = min(valid_models.items(), 
                        key=lambda x: x[1]["final_val_mse_masked"])
        print(f"\n✅ BEST MODEL: {best_model[0].upper()}")
        print(f"   MSE (masked): {best_model[1]['final_val_mse_masked']:.4f}")
        print(f"   Training time: {training_times[best_model[0]]/60:.2f} min")
        
        # Save recommendation
        with open(output_path / f"recommendation_{timestamp}.txt", "w") as f:
            f.write(f"Best Model: {best_model[0].upper()}\n")
            f.write(f"MSE (masked): {best_model[1]['final_val_mse_masked']:.4f}\n")
            f.write(f"MSE (all): {best_model[1]['final_val_mse_all']:.4f}\n")
            f.write(f"Training time: {training_times[best_model[0]]/60:.2f} min\n")


def quick_comparison(epochs=20):
    """Quick comparison with reduced epochs for testing."""
    return run_comparison_experiment(
        epochs=epochs,
        batch_size=32,
        seq_len=30,
        output_dir="comparison_results_quick",
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comparison experiment")
    parser.add_argument("--root_dir", type=str, default="Oxford Dataset")
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--missing_mode", type=str, default="random")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="comparison_results")
    parser.add_argument("--quick", action="store_true", help="Quick test with 20 epochs")
    
    args = parser.parse_args()
    
    if args.quick:
        print("\n[Quick Mode] Running with 20 epochs for fast testing\n")
        quick_comparison(epochs=20)
    else:
        run_comparison_experiment(
            root_dir=args.root_dir,
            seq_len=args.seq_len,
            mask_rate=args.mask_rate,
            missing_mode=args.missing_mode,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            hidden_units=args.hidden_units,
            device=args.device,
            output_dir=args.output_dir,
        )
