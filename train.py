"""Training script with CfC-optimized procedures."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np

from dataset import CfCIMUDataset
from models import (
    AdaptiveLoss, ReconstructionOnlyLoss, PhysicsInformedLoss,
    build_model, count_parameters
)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, use_physics=False):
    """Train for one epoch."""
    model.train()
    losses = {"total": [], "recon": [], "consistency": [], "smooth": []}
    if use_physics:
        losses["integration"] = []
        losses["energy"] = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}")
    for inputs, targets, mask in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        dt = inputs[:, :, -1:]
        
        optimizer.zero_grad()
        
        # 根据模型类型调整前向传播
        if use_physics:
            pred, uncertainty, physics_info = model(inputs)
            loss, comps = criterion(pred, targets, mask, uncertainty, dt, physics_info)
        else:
            pred, uncertainty = model(inputs)
            loss, comps = criterion(pred, targets, mask, uncertainty, dt)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        losses["total"].append(loss.item())
        for k, v in comps.items():
            if k in losses:
                losses[k].append(v)
        
        avg_losses = {k: np.mean(v) for k, v in losses.items()}
        pbar.set_postfix(avg_losses)
    
    return {k: np.mean(v) for k, v in losses.items()}


def evaluate(model, loader, criterion, device, use_physics=False):
    """Evaluate on validation set."""
    model.eval()
    losses = {"total": [], "recon": [], "consistency": [], "smooth": []}
    if use_physics:
        losses["integration"] = []
        losses["energy"] = []
    mse_all, mse_masked = [], []
    
    # 保存样本用于可视化
    sample_inputs, sample_targets, sample_preds, sample_masks = None, None, None, None
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, mask) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            dt = inputs[:, :, -1:]
            
            if use_physics:
                pred, uncertainty, physics_info = model(inputs)
                loss, comps = criterion(pred, targets, mask, uncertainty, dt, physics_info)
            else:
                pred, uncertainty = model(inputs)
                loss, comps = criterion(pred, targets, mask, uncertainty, dt)
            
            losses["total"].append(loss.item())
            for k, v in comps.items():
                if k in losses:
                    losses[k].append(v)
            
            mse_all.append(F.mse_loss(pred, targets).item())
            missing_err = ((pred - targets) ** 2 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
            mse_masked.append(missing_err.item())
            
            # 保存第一个batch用于可视化
            if batch_idx == 0:
                sample_inputs = inputs[:3].cpu()
                sample_targets = targets[:3].cpu()
                sample_preds = pred[:3].cpu()
                sample_masks = mask[:3].cpu()
    
    metrics = {k: np.mean(v) for k, v in losses.items()}
    metrics["mse_all"] = np.mean(mse_all)
    metrics["mse_masked"] = np.mean(mse_masked)
    metrics["samples"] = (sample_inputs, sample_targets, sample_preds, sample_masks)
    
    return metrics


def evaluate_multi_missing_rates(model, root_dir, device, seq_len=50, use_physics: bool = False):
    """Evaluate model on multiple missing patterns and rates.

    Args:
        model: trained model (may return pred,uncertainty or pred,uncertainty,physics_info)
        root_dir: dataset root
        device: torch device
        seq_len: sequence length
        use_physics: whether model returns physics_info as third output
    """
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

                        # Handle models with/without physics_info output
                        if use_physics:
                            pred, _unc, _phys = model(inputs)
                        else:
                            pred, _unc = model(inputs)

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
    w_physics_integration: float = 0.2,
    w_physics_energy: float = 0.1,
    num_workers: int = 4,
    use_scheduler: bool = True,
    output_dir: str = "results",
):
    """
    主训练函数,增加物理约束和可视化
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
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
        eval_mode=False,  # Random masking for training
    )
    val_ds = CfCIMUDataset(
        root_dir=root_dir,
        seq_len=seq_len,
        mask_rate=mask_rate,
        missing_mode=missing_mode,
        split="val",
        eval_mode=True,  # Deterministic masking for validation
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
    
    # 统计参数量
    num_params = count_parameters(model)
    print(f"[Model] Total parameters: {num_params:,}")
    
    # 根据模型选择损失函数
    use_physics = model_name.lower() in ["physics", "pinn", "pi"]
    if use_physics:
        criterion = PhysicsInformedLoss(
            w_recon=w_recon,
            w_consistency=w_consistency,
            w_smooth=w_smooth,
            w_physics_integration=w_physics_integration,
            w_physics_energy=w_physics_energy,
        )
        print(f"[Loss] Using PhysicsInformedLoss")
    elif model_name.lower() in ["cfc", "lnn"]:
        criterion = AdaptiveLoss(
            w_recon=w_recon,
            w_consistency=w_consistency,
            w_smooth=w_smooth,
        )
        print(f"[Loss] Using AdaptiveLoss")
    else:
        criterion = ReconstructionOnlyLoss(w_recon=w_recon)
        print(f"[Loss] Using ReconstructionOnlyLoss")
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    scheduler = None
    if use_scheduler:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
        )
    
    # Training loop
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mse_all": [],
        "val_mse_masked": [],
    }
    
    from visualization import plot_training_curves, plot_imputation_samples
    
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, use_physics
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_physics)
        
        # Logging
        print(f"\n[Epoch {epoch:03d}/{epochs:03d}]")
        log_str = f"  Train: loss={train_metrics['total']:.4f}"
        for k in ['recon', 'consistency', 'smooth', 'integration', 'energy']:
            if k in train_metrics:
                log_str += f" | {k}={train_metrics[k]:.4f}"
        print(log_str)
        
        log_str = f"  Val:   loss={val_metrics['total']:.4f}"
        for k in ['recon', 'consistency', 'smooth', 'integration', 'energy']:
            if k in val_metrics:
                log_str += f" | {k}={val_metrics[k]:.4f}"
        print(log_str)
        
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
            save_path = os.path.join(output_dir, f"best_model_{model_name}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Model saved to {save_path}")
        
        # 每10个epoch可视化一次
        if epoch % 10 == 0:
            sample_inputs, sample_targets, sample_preds, sample_masks = val_metrics["samples"]
            plot_imputation_samples(
                sample_inputs, sample_targets, sample_preds, sample_masks,
                num_samples=3,
                save_path=os.path.join(output_dir, f"samples_epoch{epoch:03d}_{model_name}.png")
            )
    
    # 绘制训练曲线
    plot_training_curves(
        history,
        save_path=os.path.join(output_dir, f"training_curves_{model_name}.png")
    )
    
    # Final evaluation
    print("\n[Final] Evaluating on multiple missing patterns...")
    model.load_state_dict(torch.load(os.path.join(output_dir, f"best_model_{model_name}.pt")))
    multi_results = evaluate_multi_missing_rates(model, root_dir, device, seq_len, use_physics)
    
    return model, history, multi_results, num_params
