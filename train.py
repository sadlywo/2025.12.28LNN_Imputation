"""Training script with CfC-optimized procedures."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np

from dataset import CfCIMUDataset
from models import AdaptiveLoss, ReconstructionOnlyLoss, build_model


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    losses = {"total": [], "recon": [], "consistency": [], "smooth": []}
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}")
    for inputs, targets, mask in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        dt = inputs[:, :, -1:]  # Extract dt from input
        
        optimizer.zero_grad()
        pred, uncertainty = model(inputs)
        loss, comps = criterion(pred, targets, mask, uncertainty, dt)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Accumulate losses
        losses["total"].append(loss.item())
        for k, v in comps.items():
            losses[k].append(v)
        
        # Update progress bar
        avg_losses = {k: np.mean(v) for k, v in losses.items()}
        pbar.set_postfix(avg_losses)
    
    return {k: np.mean(v) for k, v in losses.items()}


def evaluate(model, loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    losses = {"total": [], "recon": [], "consistency": [], "smooth": []}
    mse_all, mse_masked = [], []
    
    with torch.no_grad():
        for inputs, targets, mask in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            dt = inputs[:, :, -1:]
            
            pred, uncertainty = model(inputs)
            loss, comps = criterion(pred, targets, mask, uncertainty, dt)
            
            losses["total"].append(loss.item())
            for k, v in comps.items():
                losses[k].append(v)
            
            # Imputation metrics
            mse_all.append(F.mse_loss(pred, targets).item())
            missing_err = ((pred - targets) ** 2 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
            mse_masked.append(missing_err.item())
    
    metrics = {k: np.mean(v) for k, v in losses.items()}
    metrics["mse_all"] = np.mean(mse_all)
    metrics["mse_masked"] = np.mean(mse_masked)
    
    return metrics


def evaluate_multi_missing_rates(model, root_dir, device, seq_len=50):
    """Evaluate model on multiple missing patterns and rates."""
    patterns = ["random", "block", "channel"]
    rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    results = {}
    print("\n" + "="*60)
    print("Multi-Pattern Evaluation")
    print("="*60)
    
    for pattern in patterns:
        for rate in rates:
            try:
                test_ds = CfCIMUDataset(
                    root_dir=root_dir,
                    seq_len=seq_len,
                    mask_rate=rate,
                    missing_mode=pattern,
                    split="val",
                    eval_mode=True,  # Fixed mask for reproducible evaluation
                )
                test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
                
                model.eval()
                mse_all_list, mse_masked_list = [], []
                
                with torch.no_grad():
                    for inputs, targets, mask in test_loader:
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        mask = mask.to(device)
                        
                        pred, _ = model(inputs)
                        
                        mse_all_list.append(F.mse_loss(pred, targets).item())
                        missing_err = ((pred - targets) ** 2 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
                        mse_masked_list.append(missing_err.item())
                
                mse_all = np.mean(mse_all_list)
                mse_masked = np.mean(mse_masked_list)
                
                key = f"{pattern}_{int(rate*100)}"
                results[key] = {"mse_all": mse_all, "mse_masked": mse_masked}
                
                print(f"{pattern:8s} @ {int(rate*100):2d}% | MSE(all): {mse_all:.4f} | MSE(masked): {mse_masked:.4f}")
            
            except Exception as e:
                print(f"[Warning] Failed to evaluate {pattern} @ {rate}: {e}")
    
    print("="*60)
    return results


def train(
    root_dir: str = "Oxford Dataset",
    seq_len: int = 50,
    mask_rate: float = 0.3,
    missing_mode: str = "random",
    batch_size: int = 16,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
    model_name: str = "cfc",
    hidden_units: int = 64,
    w_recon: float = 1.0,
    w_consistency: float = 0.1,
    w_smooth: float = 0.01,
    num_workers: int = 4,
    use_scheduler: bool = True,
):
    """
    Main training function.
    
    Args:
        root_dir: Dataset root directory
        seq_len: Sequence length
        mask_rate: Missing rate
        missing_mode: Missing pattern
        batch_size: Batch size
        epochs: Number of epochs
        lr: Learning rate
        device: "cuda" or "cpu"
        hidden_units: CfC hidden units
        w_recon: Reconstruction loss weight
        w_consistency: Consistency loss weight
        w_smooth: Smoothness loss weight
        num_workers: DataLoader workers
        use_scheduler: Use OneCycleLR scheduler
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    
    # Datasets
    print("\n[Data] Loading datasets...")
    train_ds = CfCIMUDataset(
        root_dir=root_dir,
        seq_len=seq_len,
        mask_rate=mask_rate,
        missing_mode=missing_mode,
        split="train",
    )
    val_ds = CfCIMUDataset(
        root_dir=root_dir,
        seq_len=seq_len,
        mask_rate=mask_rate,
        missing_mode=missing_mode,
        split="val",
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # Model & Optimizer
    print(f"\n[Model] Initializing '{model_name}'...")
    model = build_model(
        model_name=model_name,
        input_dim=13,
        hidden_dim=hidden_units,
        output_dim=6,
    ).to(device)
    
    if model_name.lower() in ["cfc", "lnn", "physics"]:
        criterion = AdaptiveLoss(
            w_recon=w_recon,
            w_consistency=w_consistency,
            w_smooth=w_smooth,
        )
    else:
        criterion = ReconstructionOnlyLoss(w_recon=w_recon)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    scheduler = None
    if use_scheduler:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
        )
        print(f"[Scheduler] OneCycleLR with max_lr={lr}")
    
    print(f"\n[Training] Starting for {epochs} epochs...")
    print(f"[Loss] w_recon={w_recon}, w_consistency={w_consistency}, w_smooth={w_smooth}")
    
    # Training loop
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mse_all": [],
        "val_mse_masked": [],
    }
    
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Logging
        print(f"\n[Epoch {epoch:03d}/{epochs:03d}]")
        print(f"  Train: loss={train_metrics['total']:.4f} | "
              f"recon={train_metrics['recon']:.4f} | "
              f"cons={train_metrics['consistency']:.4f} | "
              f"smooth={train_metrics['smooth']:.4f}")
        print(f"  Val:   loss={val_metrics['total']:.4f} | "
              f"recon={val_metrics['recon']:.4f} | "
              f"cons={val_metrics['consistency']:.4f} | "
              f"smooth={val_metrics['smooth']:.4f}")
        print(f"  Imputation: MSE(all)={val_metrics['mse_all']:.4f} | "
              f"MSE(masked)={val_metrics['mse_masked']:.4f}")
        
        # Save history
        history["train_loss"].append(train_metrics["total"])
        history["val_loss"].append(val_metrics["total"])
        history["val_mse_all"].append(val_metrics["mse_all"])
        history["val_mse_masked"].append(val_metrics["mse_masked"])
        
        # Save best model
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save(model.state_dict(), "best_model.pt")
            print("  ✓ Model saved to best_model.pt")
    
    # Final evaluation
    print("\n[Final] Evaluating on multiple missing patterns...")
    model.load_state_dict(torch.load("best_model.pt"))
    multi_results = evaluate_multi_missing_rates(model, root_dir, device, seq_len)
    
    return model, history, multi_results
