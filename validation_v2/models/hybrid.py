"""Hybrid branch fusion and observed-value preservation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .bilnn import BidirectionalCfC
from .bilstm import BiLSTMImputer


@dataclass(frozen=True, init=False, repr=False)
class HybridComponents:
    """Inspectable branch, gate, prediction, and completed outputs."""

    _lnn: torch.Tensor = field(repr=False)
    _lstm: torch.Tensor = field(repr=False)
    _gate: torch.Tensor = field(repr=False)
    _raw: torch.Tensor = field(repr=False)
    _completed: torch.Tensor = field(repr=False)

    def __init__(
        self,
        lnn: torch.Tensor,
        lstm: torch.Tensor,
        gate: torch.Tensor,
        raw: torch.Tensor,
        completed: torch.Tensor,
    ) -> None:
        for name, value in (
            ("lnn", lnn),
            ("lstm", lstm),
            ("gate", gate),
            ("raw", raw),
            ("completed", completed),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch tensor")
            # Clone, but deliberately do not detach: public isolation must not
            # sever the training graph used by HybridImputer.forward().
            object.__setattr__(self, f"_{name}", value.clone())

    @property
    def lnn(self) -> torch.Tensor:
        return self._lnn.clone()

    @property
    def lstm(self) -> torch.Tensor:
        return self._lstm.clone()

    @property
    def gate(self) -> torch.Tensor:
        return self._gate.clone()

    @property
    def raw(self) -> torch.Tensor:
        return self._raw.clone()

    @property
    def completed(self) -> torch.Tensor:
        return self._completed.clone()


def _require_compatible_tensors(
    observed: torch.Tensor, mask: torch.Tensor, prediction: torch.Tensor
) -> None:
    if not all(isinstance(value, torch.Tensor) for value in (observed, mask, prediction)):
        raise TypeError("observed, mask, and prediction must be torch tensors")
    if observed.shape != mask.shape or observed.shape != prediction.shape:
        raise ValueError("observed, mask, and prediction must have identical shapes")
    if observed.device != mask.device or observed.device != prediction.device:
        raise ValueError("observed, mask, and prediction must be on the same device")
    if observed.dtype != mask.dtype or observed.dtype != prediction.dtype:
        raise TypeError("observed, mask, and prediction must have the same dtype")
    if not observed.is_floating_point() or not prediction.is_floating_point():
        raise TypeError("observed and prediction must be floating point")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must contain only 0 and 1")


def complete_signal(
    observed: torch.Tensor, mask: torch.Tensor, prediction: torch.Tensor
) -> torch.Tensor:
    """Keep observed entries exactly and fill only missing entries.

    ``torch.where`` is intentional: unlike arithmetic masking, NaNs stored at
    hidden positions in ``observed`` cannot contaminate the completed signal.
    """

    _require_compatible_tensors(observed, mask, prediction)
    return torch.where(mask.bool(), observed, prediction)


def fuse(
    lnn_prediction: torch.Tensor,
    lstm_prediction: torch.Tensor,
    lnn_gate: torch.Tensor,
) -> torch.Tensor:
    """Fuse branches with ``lnn_gate`` explicitly defined as the LNN weight."""

    if not all(
        isinstance(value, torch.Tensor)
        for value in (lnn_prediction, lstm_prediction, lnn_gate)
    ):
        raise TypeError("predictions and gate must be torch tensors")
    if lnn_prediction.shape != lstm_prediction.shape:
        raise ValueError("branch predictions must have identical shapes")
    if lnn_prediction.device != lstm_prediction.device:
        raise ValueError("branch predictions must be on the same device")
    if lnn_prediction.dtype != lstm_prediction.dtype:
        raise TypeError("branch predictions must have the same dtype")
    if not lnn_prediction.is_floating_point():
        raise TypeError("branch predictions must be floating point")
    if lnn_gate.device != lnn_prediction.device:
        raise ValueError("lnn_gate must be on the predictions device")
    if lnn_gate.dtype != lnn_prediction.dtype:
        raise TypeError("lnn_gate must have the predictions dtype")
    if not lnn_gate.is_floating_point():
        raise TypeError("lnn_gate must be floating point")
    try:
        broadcast_shape = torch.broadcast_shapes(
            lnn_prediction.shape, lnn_gate.shape
        )
    except RuntimeError as error:
        raise ValueError("lnn_gate must broadcast to the prediction shape") from error
    if broadcast_shape != lnn_prediction.shape:
        raise ValueError("lnn_gate must broadcast to exactly the prediction shape")
    if not torch.isfinite(lnn_gate).all() or not torch.all(
        (lnn_gate >= 0) & (lnn_gate <= 1)
    ):
        raise ValueError("lnn_gate must be finite and in [0, 1]")
    return lnn_gate * lnn_prediction + (1.0 - lnn_gate) * lstm_prediction


class HybridImputer(nn.Module):
    """Leakage-safe hybrid whose gate is always the LNN branch weight."""

    def __init__(
        self,
        input_size: int,
        *,
        lnn_hidden_size: int = 32,
        lstm_hidden_size: int = 32,
        lstm_num_layers: int = 1,
        lnn_branch: nn.Module | None = None,
        lstm_branch: nn.Module | None = None,
        gate_network: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        self.lnn_branch = lnn_branch or BidirectionalCfC(
            input_size, lnn_hidden_size, output_size=6
        )
        self.lstm_branch = lstm_branch or BiLSTMImputer(
            input_size,
            lstm_hidden_size,
            output_size=6,
            num_layers=lstm_num_layers,
        )
        self.gate_network = gate_network or nn.Linear(input_size, 6)

    def forward_components(
        self,
        features: torch.Tensor,
        forward_dt: torch.Tensor,
        reverse_dt: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
    ) -> HybridComponents:
        """Return both branches, the LNN-weight gate, raw fusion, and completion."""

        lnn_prediction = self.lnn_branch(features, forward_dt, reverse_dt)
        lstm_prediction = self.lstm_branch(features)
        gate = torch.sigmoid(self.gate_network(features))
        raw = fuse(lnn_prediction, lstm_prediction, gate)
        completed = complete_signal(observed, mask, raw)
        return HybridComponents(
            lnn=lnn_prediction,
            lstm=lstm_prediction,
            gate=gate,
            raw=raw,
            completed=completed,
        )

    def forward(
        self,
        features: torch.Tensor,
        forward_dt: torch.Tensor,
        reverse_dt: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the completed six-channel signal, preserving observed entries.

        The model accepts leakage-safe ``features`` plus observed values, their
        binary mask, and explicit forward/reverse intervals. It never accepts a
        complete target or Vicon data.
        """

        return self.forward_components(
            features, forward_dt, reverse_dt, observed, mask
        ).completed
