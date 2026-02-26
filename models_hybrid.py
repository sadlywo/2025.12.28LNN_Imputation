"""
Hybrid LNN-LSTM Imputation Model.

Strategy:
- LNN (CfC): focuses on short-term kinematics (local temporal patterns)
- LSTM: focuses on long-term motion patterns (global sequential dependencies)
- Fusion: RMSE-based adaptive weighting of both predictions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from typing import Tuple, Dict


class ShortTermLNN(nn.Module):
    """
    CfC-based short-term imputer.
    Focuses on local kinematics: angular velocity continuity, acceleration patterns.
    Uses physics-aware dual heads (gyro + acc).
    """

    def __init__(
        self,
        input_dim: int = 13,
        hidden_units: int = 64,
        output_dim: int = 6,
        mixed_memory: bool = True,
    ):
        super().__init__()
        self.hidden_units = hidden_units

        # CfC backbone — good at capturing continuous-time ODE dynamics
        self.cfc_out_dim = max(hidden_units // 2, 4)
        if self.cfc_out_dim > hidden_units - 2:
            self.cfc_out_dim = max(hidden_units - 2, 1)

        wiring = AutoNCP(hidden_units, self.cfc_out_dim)
        self.cfc = CfC(input_dim, wiring, batch_first=True, mixed_memory=mixed_memory)

        # Physics-aware dual heads
        self.gyro_head = nn.Sequential(
            nn.Linear(self.cfc_out_dim, hidden_units // 2),
            nn.Tanh(),
            nn.Linear(hidden_units // 2, 3),
        )
        self.acc_head = nn.Sequential(
            nn.Linear(self.cfc_out_dim, hidden_units // 2),
            nn.ReLU(),
            nn.Linear(hidden_units // 2, 3),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.cfc_out_dim, output_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, 13)
        Returns:
            pred: (B, T, 6)
            uncertainty: (B, T, 6)
        """
        cfc_out, _ = self.cfc(x)
        gyro = self.gyro_head(cfc_out)
        acc = self.acc_head(cfc_out)
        pred = torch.cat([gyro, acc], dim=-1)
        uncertainty = self.uncertainty_head(cfc_out)
        return pred, uncertainty


class LongTermLSTM(nn.Module):
    """
    LSTM-based long-term imputer.
    Focuses on capturing global motion patterns: turns, periodic gaits, etc.
    Uses bidirectional LSTM to leverage both past and future context.
    """

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 128,
        output_dim: int = 6,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Bidirectional → 2 * hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, 13)
        Returns:
            pred: (B, T, 6)
            uncertainty: (B, T, 6)
        """
        h, _ = self.lstm(x)
        pred = self.head(h)
        uncertainty = self.uncertainty_head(h)
        return pred, uncertainty


class HybridLNNLSTM(nn.Module):
    """
    Hybrid model that fuses LNN (short-term) and LSTM (long-term) predictions.

    Fusion strategy:
    - Each sub-model produces predictions and uncertainty estimates
    - A learnable gating network computes per-timestep, per-channel weights
    - The gate is informed by both sub-model uncertainties and a learned context
    - Final prediction = gate * lnn_pred + (1 - gate) * lstm_pred

    Additionally supports RMSE-based weighting during inference.
    """

    def __init__(
        self,
        input_dim: int = 13,
        lnn_hidden: int = 64,
        lstm_hidden: int = 128,
        output_dim: int = 6,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        fusion_mode: str = "learned",  # "learned" | "rmse" | "uncertainty"
    ):
        super().__init__()
        self.output_dim = output_dim
        self.fusion_mode = fusion_mode

        # Sub-models
        self.lnn = ShortTermLNN(
            input_dim=input_dim,
            hidden_units=lnn_hidden,
            output_dim=output_dim,
        )
        self.lstm = LongTermLSTM(
            input_dim=input_dim,
            hidden_dim=lstm_hidden,
            output_dim=output_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )

        # Learnable gating network
        # Input: lnn_pred(6) + lstm_pred(6) + lnn_unc(6) + lstm_unc(6) + original_input(13) = 37
        gate_input_dim = output_dim * 4 + input_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid(),  # Output in [0, 1]: weight for LNN
        )

        # Combined uncertainty
        self.combined_uncertainty = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Softplus(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, 13)
        Returns:
            pred: (B, T, 6) fused prediction
            uncertainty: (B, T, 6) combined uncertainty
        """
        lnn_pred, lnn_unc = self.lnn(x)        # (B, T, 6), (B, T, 6)
        lstm_pred, lstm_unc = self.lstm(x)      # (B, T, 6), (B, T, 6)

        if self.fusion_mode == "learned":
            gate_input = torch.cat([lnn_pred, lstm_pred, lnn_unc, lstm_unc, x], dim=-1)
            gate = self.gate_net(gate_input)  # (B, T, 6) in [0, 1]
        elif self.fusion_mode == "uncertainty":
            # Inverse uncertainty weighting
            lnn_w = 1.0 / (lnn_unc + 1e-6)
            lstm_w = 1.0 / (lstm_unc + 1e-6)
            gate = lnn_w / (lnn_w + lstm_w)
        else:
            # Equal weighting fallback (rmse mode is applied post-hoc)
            gate = torch.full_like(lnn_pred, 0.5)

        pred = gate * lnn_pred + (1.0 - gate) * lstm_pred

        # Combined uncertainty
        unc = self.combined_uncertainty(torch.cat([lnn_unc, lstm_unc], dim=-1))

        return pred, unc

    def forward_with_components(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning all components for analysis.
        """
        lnn_pred, lnn_unc = self.lnn(x)
        lstm_pred, lstm_unc = self.lstm(x)

        gate_input = torch.cat([lnn_pred, lstm_pred, lnn_unc, lstm_unc, x], dim=-1)
        gate = self.gate_net(gate_input)

        pred = gate * lnn_pred + (1.0 - gate) * lstm_pred
        unc = self.combined_uncertainty(torch.cat([lnn_unc, lstm_unc], dim=-1))

        return {
            "pred": pred,
            "uncertainty": unc,
            "lnn_pred": lnn_pred,
            "lstm_pred": lstm_pred,
            "lnn_uncertainty": lnn_unc,
            "lstm_uncertainty": lstm_unc,
            "gate": gate,  # LNN weight per (B, T, channel)
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rmse_based_reweight(
    lnn_pred: torch.Tensor,
    lstm_pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, float, float]:
    """
    Post-hoc RMSE-based reweighting on observed positions.

    Args:
        lnn_pred:  (B, T, 6)
        lstm_pred: (B, T, 6)
        target:    (B, T, 6)
        mask:      (B, T, 6)  1=observed, 0=missing

    Returns:
        fused: (B, T, 6) reweighted prediction
        w_lnn: scalar weight for LNN
        w_lstm: scalar weight for LSTM
    """
    observed = mask > 0.5  # only evaluate on observed positions

    lnn_err = ((lnn_pred - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    lstm_err = ((lstm_pred - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)

    lnn_rmse = torch.sqrt(lnn_err + 1e-8)
    lstm_rmse = torch.sqrt(lstm_err + 1e-8)

    # Inverse RMSE weighting
    w_lnn = (1.0 / (lnn_rmse + 1e-6))
    w_lstm = (1.0 / (lstm_rmse + 1e-6))
    w_sum = w_lnn + w_lstm
    w_lnn = (w_lnn / w_sum).item()
    w_lstm = (w_lstm / w_sum).item()

    fused = w_lnn * lnn_pred + w_lstm * lstm_pred
    return fused, w_lnn, w_lstm