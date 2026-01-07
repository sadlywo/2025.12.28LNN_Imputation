"""CfC-based IMU imputation model with physics prior."""
import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from typing import Tuple


class PhysicsAwareIMUImputer(nn.Module):
    """
    CfC-based imputation model with built-in physics understanding.
    
    Architecture:
    1. CfC backbone (handles irregular time + long dependencies)
    2. Physics-constrained decoder (separate gyro/acc heads)
    3. Uncertainty estimation head
    
    Key design principles:
    - Let CfC learn temporal dynamics (no manual integration)
    - Separate processing for gyro and acc (different physical meanings)
    - Output uncertainty for adaptive loss weighting
    """
    
    def __init__(
        self,
        input_dim: int = 13,
        hidden_units: int = 64,
        output_dim: int = 6,
        use_physics_prior: bool = True,
        mixed_memory: bool = True,
    ):
        """
        Args:
            input_dim: Input dimension (masked_imu(6) + mask(6) + dt(1))
            hidden_units: Number of CfC hidden units
            output_dim: Output dimension (gyro(3) + acc(3))
            use_physics_prior: Use separate heads for gyro and acc
            mixed_memory: Use mixed memory in CfC
        """
        super().__init__()
        self.use_physics_prior = use_physics_prior
        
        # CfC backbone with NCP wiring (sparse + interpretable)
        wiring = AutoNCP(hidden_units, output_dim)
        self.cfc = CfC(
            input_dim,
            wiring,
            batch_first=True,
            mixed_memory=mixed_memory,
        )
        
        # Physics-aware projection heads
        if use_physics_prior:
            # Separate heads for gyro and acc (different physical properties)
            self.gyro_head = nn.Sequential(
                nn.Linear(output_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 3),
            )
            self.acc_head = nn.Sequential(
                nn.Linear(output_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 3),
            )
        else:
            self.output_proj = nn.Linear(output_dim, output_dim)
        
        # Learnable uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(output_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
            nn.Softplus(),  # Ensure positive uncertainty
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, 13) = [masked_imu(6), mask(6), dt(1)]
        
        Returns:
            pred: (B, T, 6) predicted IMU [gyro(3), acc(3)]
            uncertainty: (B, T, 6) prediction uncertainty
        """
        # CfC processes irregular time automatically via input
        cfc_out, _ = self.cfc(x)  # (B, T, output_dim)
        
        if self.use_physics_prior:
            gyro = self.gyro_head(cfc_out)  # (B, T, 3)
            acc = self.acc_head(cfc_out)    # (B, T, 3)
            pred = torch.cat([gyro, acc], dim=-1)
        else:
            pred = self.output_proj(cfc_out)
        
        uncertainty = self.uncertainty_head(cfc_out)
        
        return pred, uncertainty


class AdaptiveLoss(nn.Module):
    """
    Loss function that adapts to prediction uncertainty.
    
    Components:
    1. Weighted reconstruction loss (inverse uncertainty weighting)
    2. Temporal consistency loss (predictions should be smooth)
    3. Smoothness regularization (penalize high-frequency noise)
    
    Inspired by "What Uncertainties Do We Need in Bayesian Deep Learning?"
    """
    
    def __init__(
        self,
        w_recon: float = 1.0,
        w_consistency: float = 0.5,
        w_smooth: float = 0.1,
    ):
        super().__init__()
        self.w_recon = w_recon
        self.w_consistency = w_consistency
        self.w_smooth = w_smooth
    
    def forward(
        self,
        pred: torch.Tensor,       # (B, T, 6)
        target: torch.Tensor,     # (B, T, 6)
        mask: torch.Tensor,       # (B, T, 6)
        uncertainty: torch.Tensor, # (B, T, 6)
        dt: torch.Tensor,         # (B, T, 1)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute adaptive loss with uncertainty weighting.
        
        Returns:
            total_loss: Weighted sum of all loss components
            components: Dict of individual loss values for logging
        """
        # 1. Weighted reconstruction loss (higher uncertainty = lower weight)
        # Only compute on missing positions (mask=0)
        recon_err = (pred - target) ** 2
        # Uncertainty weighting: minimize error/uncertainty + log(uncertainty)
        weighted_recon = recon_err / (uncertainty + 1e-6) + torch.log(uncertainty + 1e-6)
        loss_recon = (weighted_recon * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
        
        # 2. Temporal consistency loss (predictions should be smooth)
        pred_diff = pred[:, 1:] - pred[:, :-1]  # (B, T-1, 6)
        target_diff = target[:, 1:] - target[:, :-1]
        loss_consistency = ((pred_diff - target_diff) ** 2).mean()
        
        # 3. Smoothness regularization (penalize high acceleration)
        dt_expanded = dt[:, 1:]  # (B, T-1, 1)
        pred_accel = pred_diff / (dt_expanded + 1e-6)  # Discrete derivative
        loss_smooth = (pred_accel ** 2).mean()
        
        total = (
            self.w_recon * loss_recon +
            self.w_consistency * loss_consistency +
            self.w_smooth * loss_smooth
        )
        
        components = {
            "recon": loss_recon.item(),
            "consistency": loss_consistency.item(),
            "smooth": loss_smooth.item(),
        }
        
        return total, components


# Legacy compatibility: keep old model names but use new implementation
class LNNImputer(PhysicsAwareIMUImputer):
    """Alias for backward compatibility."""
    pass


def build_model(
    model_name: str,
    input_dim: int = 13,
    hidden_dim: int = 64,
    output_dim: int = 6,
) -> nn.Module:
    """
    Build model by name.
    
    Args:
        model_name: "lnn", "cfc", or "physics"
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
    
    Returns:
        Initialized model
    """
    if model_name.lower() in ["lnn", "cfc", "physics"]:
        return PhysicsAwareIMUImputer(
            input_dim=input_dim,
            hidden_units=hidden_dim,
            output_dim=output_dim,
            use_physics_prior=True,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
