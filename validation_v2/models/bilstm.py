"""Bidirectional LSTM imputation branch."""

from __future__ import annotations

import torch
from torch import nn


class BiLSTMImputer(nn.Module):
    """Predict six IMU channels from exactly the supplied feature tensor."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 6,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0 or num_layers <= 0:
            raise ValueError("model dimensions and num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_size * 2, output_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch tensor")
        if features.ndim != 3 or features.shape[-1] != self.input_size:
            raise ValueError("features must have shape (batch, time, input_size)")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        sequence, _ = self.lstm(features)
        return self.projection(sequence)
