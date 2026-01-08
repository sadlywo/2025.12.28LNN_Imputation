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
                       choices=["cfc", "lnn", "physics", "gru", "transformer"],
                       help="Model to train: cfc/lnn/physics/gru/transformer")
    parser.add_argument("--hidden_units", type=int, default=64,
                       help="Hidden units / d_model for the chosen model")
    
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
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("CfC-IMU Imputation Training")
    print("="*60)
    print(f"Dataset: {args.root_dir}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Missing Pattern: {args.missing_mode} @ {args.mask_rate*100:.0f}%")
    print(f"Model: {args.model_name} (hidden_units={args.hidden_units})")
    print(f"Training: {args.epochs} epochs, lr={args.lr}")
    print(f"Loss Weights: recon={args.w_recon}, consistency={args.w_consistency}, smooth={args.w_smooth}")
    print("="*60 + "\n")
    
    # Train
    model, history, multi_results = train(
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
        num_workers=args.num_workers,
        use_scheduler=args.use_scheduler,
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model saved to: best_model.pt")
    print(f"Final validation MSE (masked): {history['val_mse_masked'][-1]:.4f}")
    
    # Save training history
    torch.save({
        "history": history,
        "multi_results": multi_results,
        "args": vars(args),
    }, "training_results.pt")
    print(f"Training history saved to: training_results.pt")


if __name__ == "__main__":
    main()
