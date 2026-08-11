"""Entry point with CLI support for CfC-IMU Imputation."""
import argparse
import torch
from train import train


def parse_args():
    parser = argparse.ArgumentParser(
        description="CfC-based IMU Imputation with Physics Prior"
    )
    
    # Data arguments
    parser.add_argument("--root_dir", type=str, default="Oxford Dataset",
                       help="Root directory of dataset")
    parser.add_argument("--seq_len", type=int, default=50,
                       help="Sequence length for windowing")
    parser.add_argument("--mask_rate", type=float, default=0.3,
                       help="Missing rate (0-1)")
    parser.add_argument("--missing_mode", type=str, default="random",
                       choices=["random", "block", "channel"],
                       help="Missing pattern")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of DataLoader workers")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="cfc",
                       choices=["cfc", "lnn", "physics", "pinn", "gru", "transformer"],
                       help="Model to train")
    parser.add_argument("--hidden_units", type=int, default=64,
                       help="Hidden units")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--use_scheduler", action="store_true", default=True,
                       help="Use OneCycleLR scheduler")
    
    # Loss weights
    parser.add_argument("--w_recon", type=float, default=1.0,
                       help="Reconstruction loss weight")
    parser.add_argument("--w_consistency", type=float, default=0.1,
                       help="Temporal consistency loss weight")
    parser.add_argument("--w_smooth", type=float, default=0.01,
                       help="Smoothness loss weight")
    
    # Physics loss weights (新增)
    parser.add_argument("--w_physics_integration", type=float, default=0.2,
                       help="Physics integration loss weight")
    parser.add_argument("--w_physics_energy", type=float, default=0.1,
                       help="Physics energy loss weight")
    
    # Output directory (新增)
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("CfC-IMU Imputation Training")
    print("="*60)
    print(f"Dataset: {args.root_dir}")
    print(f"Model: {args.model_name} (hidden_units={args.hidden_units})")
    print(f"Output: {args.output_dir}")
    print("="*60 + "\n")
    
    # Train
    model, history, multi_results, num_params = train(
        root_dir=args.root_dir,
        seq_len=args.seq_len,
        mask_rate=args.mask_rate,
        missing_mode=args.missing_mode,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        model_name=args.model_name,
        hidden_units=args.hidden_units,
        w_recon=args.w_recon,
        w_consistency=args.w_consistency,
        w_smooth=args.w_smooth,
        w_physics_integration=args.w_physics_integration,
        w_physics_energy=args.w_physics_energy,
        num_workers=args.num_workers,
        use_scheduler=args.use_scheduler,
        output_dir=args.output_dir,
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Parameters: {num_params:,}")
    print(f"Best val MSE (masked): {min(history['val_mse_masked']):.4f}")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
