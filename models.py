"""CfC-based IMU imputation model with physics prior."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from typing import Tuple
import math


class PhysicsAwareIMUImputer(nn.Module):
    """
    CfC-based imputation model with built-in physics understanding.
    """
    
    def __init__(
        self,
        input_dim: int = 13,
        hidden_units: int = 64,
        output_dim: int = 6,
        use_physics_prior: bool = True,
        mixed_memory: bool = True,
    ):
        super().__init__()
        self.use_physics_prior = use_physics_prior
        self.hidden_units = hidden_units
        
        # AutoNCP constraint: output_size must be <= units - 2
        # We set output dim (command neurons) to roughly half of hidden_units for stability
        # if hidden_units=64 -> cfc_out_dim=32; if 16 -> 8
        self.cfc_out_dim = max(hidden_units // 2, 4)
        if self.cfc_out_dim > hidden_units - 2:
            self.cfc_out_dim = max(hidden_units - 3, 1)

        # Wiring: total units = hidden_units, output units = cfc_out_dim
        wiring = AutoNCP(hidden_units, self.cfc_out_dim)
        
        self.cfc = CfC(
            input_dim,
            wiring,
            batch_first=True,
            mixed_memory=mixed_memory,
        )
        
        # Simple output projection
        if use_physics_prior:
            # Separate heads for gyro and acc
            # Note: Input is now self.cfc_out_dim
            self.gyro_head = nn.Linear(self.cfc_out_dim, 3)
            self.acc_head = nn.Linear(self.cfc_out_dim, 3)
        else:
            self.output_proj = nn.Linear(self.cfc_out_dim, output_dim)
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.cfc_out_dim, output_dim),
            nn.Softplus(),
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
        cfc_out, _ = self.cfc(x)  # (B, T, cfc_out_dim)
        
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
        if self.w_smooth != 0.0:
            dt_expanded = dt[:, 1:]  # (B, T-1, 1)
            pred_accel = pred_diff / (dt_expanded + 1e-6)  # Discrete derivative
            loss_smooth = (pred_accel ** 2).mean()
        else:
            loss_smooth = torch.tensor(0.0, device=pred.device)
        
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


class ReconstructionOnlyLoss(nn.Module):
    """
    Reconstruction-only loss used for non-CfC baselines (GRU/Transformer).
    Computes MSE only on the missing positions indicated by mask.
    """
    def __init__(self, w_recon: float = 1.0):
        super().__init__()
        self.w_recon = w_recon

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        uncertainty: torch.Tensor = None,
        dt: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        recon = ((pred - target) ** 2 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
        total = self.w_recon * recon
        components = {"recon": recon.item(), "consistency": 0.0, "smooth": 0.0}
        return total, components


# Legacy compatibility: keep old model names but use new implementation
class LNNImputer(PhysicsAwareIMUImputer):
    """Alias for backward compatibility."""
    pass


class GRUImputer(nn.Module):
    """GRU baseline for IMU imputation with an uncertainty head."""

    def __init__(self, input_dim: int = 13, hidden_dim: int = 128, output_dim: int = 6):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.rnn(x)
        pred = self.head(h)
        uncert = self.uncertainty_head(h)
        return pred, uncert


class PositionalEncoding(nn.Module):
    """Sine-cosine positional encoding using cumulative time (from dt channel)."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D); assume last channel of x is dt
        dt = x[:, :, -1]  # (B, T)
        t = torch.cumsum(dt, dim=1)  # (B, T)
        B, T = t.shape
        device = t.device
        pe = torch.zeros(B, T, self.d_model, device=device)
        position = t.unsqueeze(-1)  # (B, T, 1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device) * (-math.log(10000.0) / self.d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class TransformerImputer(nn.Module):
    """Transformer baseline using dt-aware positional encoding and uncertainty head."""

    def __init__(self, input_dim: int = 13, hidden_dim: int = 128, output_dim: int = 6, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.posenc = PositionalEncoding(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.input_proj(x)
        z = z + self.posenc(x)
        h = self.encoder(z)
        pred = self.head(h)
        uncert = self.uncertainty_head(h)
        return pred, uncert


class PhysicsInformedIMUImputer(nn.Module):
    """
    物理信息增强的IMU插补模型
    """
    
    def __init__(
        self,
        input_dim: int = 13,
        hidden_units: int = 64,
        output_dim: int = 6,
        use_physics_prior: bool = True,
        mixed_memory: bool = True,
        physics_strength: float = 0.1,
    ):
        super().__init__()
        self.use_physics_prior = use_physics_prior
        self.hidden_units = hidden_units
        self.physics_strength = physics_strength
        
        # CfC backbone
        # Fix AutoNCP constraint: output_size must be <= units - 2
        self.cfc_out_dim = max(hidden_units // 2, 4)
        if self.cfc_out_dim > hidden_units - 2:
            self.cfc_out_dim = max(hidden_units - 2, 1)
        
        wiring = AutoNCP(hidden_units, self.cfc_out_dim)
        self.cfc = CfC(
            input_dim,
            wiring,
            batch_first=True,
            mixed_memory=mixed_memory,
        )
        
        # 分离的物理头
        if use_physics_prior:
            # 陀螺仪分支:考虑角速度的物理特性
            self.gyro_head = nn.Sequential(
                nn.Linear(self.cfc_out_dim, hidden_units // 2),
                nn.Tanh(),  # 有界激活,符合角速度范围
                nn.Linear(hidden_units // 2, 3)
            )
            
            # 加速度计分支:考虑线性加速度+重力
            self.acc_head = nn.Sequential(
                nn.Linear(self.cfc_out_dim, hidden_units // 2),
                nn.ReLU(),  # 隐层激活
                nn.Linear(hidden_units // 2, 3)
            )
            
            # 物理耦合层
            self.physics_coupling = nn.Sequential(
                nn.Linear(6, hidden_units // 4),
                nn.Tanh(),
                nn.Linear(hidden_units // 4, 6)
            )
        else:
            self.output_proj = nn.Linear(self.cfc_out_dim, output_dim)
        
        # 不确定性估计
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.cfc_out_dim, output_dim),
            nn.Softplus(),
        )
        
        # 物理参数学习
        self.gyro_noise_scale = nn.Parameter(torch.ones(3) * 0.01)
        self.acc_noise_scale = nn.Parameter(torch.ones(3) * 0.1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        前向传播,增加物理信息输出
        
        Returns:
            pred: (B, T, 6) 预测的IMU
            uncertainty: (B, T, 6) 不确定性
            physics_info: dict,包含物理量用于损失计算
        """
        cfc_out, _ = self.cfc(x)  # (B, T, cfc_out_dim)
        
        if self.use_physics_prior:
            gyro_raw = self.gyro_head(cfc_out)  # (B, T, 3)
            acc_raw = self.acc_head(cfc_out)    # (B, T, 3)
            
            # 物理耦合
            combined = torch.cat([gyro_raw, acc_raw], dim=-1)  # (B, T, 6)
            coupling_correction = self.physics_coupling(combined)
            pred = combined + self.physics_strength * coupling_correction
        else:
            pred = self.output_proj(cfc_out)
        
        uncertainty = self.uncertainty_head(cfc_out)
        
        # 物理信息字典
        physics_info = {
            'gyro': pred[:, :, :3],
            'acc': pred[:, :, 3:],
            'gyro_noise_scale': self.gyro_noise_scale,
            'acc_noise_scale': self.acc_noise_scale,
        }
        
        return pred, uncertainty, physics_info


class PhysicsInformedLoss(nn.Module):
    """
    物理信息损失函数
    
    新增损失项:
    1. 角速度积分一致性
    2. 加速度能量约束
    3. 传感器噪声正则化
    """
    
    def __init__(
        self,
        w_recon: float = 1.0,
        w_consistency: float = 0.5,
        w_smooth: float = 0.1,
        w_physics_integration: float = 0.2,
        w_physics_energy: float = 0.1,
    ):
        super().__init__()
        self.w_recon = w_recon
        self.w_consistency = w_consistency
        self.w_smooth = w_smooth
        self.w_physics_integration = w_physics_integration
        self.w_physics_energy = w_physics_energy
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        uncertainty: torch.Tensor,
        dt: torch.Tensor,
        physics_info: dict = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算物理信息增强损失
        """
        # 1. 基础重建损失
        recon_err = (pred - target) ** 2
        weighted_recon = recon_err / (uncertainty + 1e-6) + torch.log(uncertainty + 1e-6)
        loss_recon = (weighted_recon * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
        
        # 2. 时间一致性
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        loss_consistency = ((pred_diff - target_diff) ** 2).mean()
        
        # 3. 平滑度
        dt_expanded = dt[:, 1:]
        if self.w_smooth != 0.0:
            pred_accel = pred_diff / (dt_expanded + 1e-6)
            loss_smooth = (pred_accel ** 2).mean()
        else:
            loss_smooth = torch.tensor(0.0, device=pred.device)
        
        # 4. 物理约束
        if physics_info is not None:
            # 修正: 比较预测的动力学特征(导数)与目标的动力学特征，而不是错误的积分对比
            # 原始代码试图比较积分(角度)与差分(角加速度)，这是量纲不匹配的
            # 新逻辑: 强化动力学一致性，要求预测的变化率与真实变化率一致
            
            # 计算预测的角速度变化率 (近似角加速度)
            pred_gyro = physics_info['gyro']
            pred_gyro_diff = pred_gyro[:, 1:] - pred_gyro[:, :-1]
            pred_gyro_accel = pred_gyro_diff / (dt_expanded + 1e-6)
            
            # 计算目标的角速度变化率
            target_gyro = target[:, :, :3]
            target_gyro_diff = target_gyro[:, 1:] - target_gyro[:, :-1]
            target_gyro_accel = target_gyro_diff / (dt_expanded + 1e-6)
            
            # 只在观测点计算此损失(或者在全序列计算以利用物理先验)
            # 这里我们选择在全序列计算，作为物理约束
            loss_integration = ((pred_gyro_accel - target_gyro_accel) ** 2).mean()
            
            # 能量约束: 限制加速度模长
            acc_pred = physics_info['acc']
            acc_magnitude = torch.norm(acc_pred, dim=-1)
            loss_energy = F.relu(acc_magnitude - 20.0).mean()
        else:
            loss_integration = torch.tensor(0.0, device=pred.device)
            loss_energy = torch.tensor(0.0, device=pred.device)
        
        total = (
            self.w_recon * loss_recon +
            self.w_consistency * loss_consistency +
            self.w_smooth * loss_smooth +
            self.w_physics_integration * loss_integration +
            self.w_physics_energy * loss_energy
        )
        
        components = {
            "recon": loss_recon.item(),
            "consistency": loss_consistency.item(),
            "smooth": loss_smooth.item(),
            "integration": loss_integration.item(),
            "energy": loss_energy.item(),
        }
        
        return total, components


def build_model(
    model_name: str,
    input_dim: int = 13,
    hidden_dim: int = 64,
    output_dim: int = 6,
) -> nn.Module:
    """
    构建模型,新增physics-informed版本
    """
    name = model_name.lower()
    if name in ["lnn", "cfc"]:
        return PhysicsAwareIMUImputer(
            input_dim=input_dim,
            hidden_units=hidden_dim,
            output_dim=output_dim,
            use_physics_prior=True,
        )
    elif name in ["physics", "pinn", "pi"]:
        # 物理信息增强版本
        return PhysicsInformedIMUImputer(
            input_dim=input_dim,
            hidden_units=hidden_dim,
            output_dim=output_dim,
            use_physics_prior=True,
            physics_strength=0.1,
        )
    elif name in ["gru"]:
        return GRUImputer(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    elif name in ["transformer", "tfm"]:
        return TransformerImputer(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
