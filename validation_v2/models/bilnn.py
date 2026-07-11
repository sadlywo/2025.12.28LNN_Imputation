"""Bidirectional closed-form continuous-time model with explicit intervals."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def _default_cfc_factory(*args, **kwargs):
    from ncps.torch import CfC

    return CfC(*args, **kwargs)


def _sequence_output(result):
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError("CfC tuple output must contain (output, hidden_state)")
        result = result[0]
    if not isinstance(result, torch.Tensor):
        raise TypeError("CfC must return a tensor or (tensor, hidden_state)")
    return result


class BidirectionalCfC(nn.Module):
    """Two CfC directions driven by caller-provided, direction-aligned timespans."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 6,
        *,
        cfc_factory: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise ValueError("input_size, hidden_size, and output_size must be positive")
        self._uses_default_cfc = cfc_factory is None
        factory = cfc_factory or _default_cfc_factory
        options = {"batch_first": True, "return_sequences": True}
        self.forward_cfc = factory(input_size, hidden_size, **options)
        self.reverse_cfc = factory(input_size, hidden_size, **options)
        self.projection = nn.Linear(hidden_size * 2, output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def _run_cfc(
        self, cfc: nn.Module, features: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        if not self._uses_default_cfc:
            return _sequence_output(cfc(features, timespans=dt))
        # ncps 1.0.1 applies an unconditional squeeze to each (B,) timespan
        # slice, which cannot broadcast against (B, hidden). Repeating the same
        # scalar interval over hidden units produces (B, hidden) after that
        # squeeze without changing the time value supplied to any unit.
        expanded_dt = dt.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        return _sequence_output(cfc(features, timespans=expanded_dt))

    @staticmethod
    def _validate_dt(name: str, dt: torch.Tensor, features: torch.Tensor) -> None:
        if not isinstance(dt, torch.Tensor):
            raise TypeError(f"{name} must be a torch tensor")
        if dt.shape != features.shape[:2]:
            raise ValueError(f"{name} must have shape (batch, time)")
        if dt.device != features.device:
            raise ValueError(f"{name} must be on the features device")
        if dt.dtype != features.dtype:
            raise TypeError(f"{name} must have the features dtype")
        if not torch.isfinite(dt).all() or not torch.all(dt > 0):
            raise ValueError(f"{name} must be finite and strictly positive")

    def forward(
        self,
        features: torch.Tensor,
        forward_dt: torch.Tensor,
        reverse_dt: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch tensor")
        if features.ndim != 3 or features.shape[-1] != self.input_size:
            raise ValueError("features must have shape (batch, time, input_size)")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        self._validate_dt("forward_dt", forward_dt, features)
        self._validate_dt("reverse_dt", reverse_dt, features)

        forward = self._run_cfc(self.forward_cfc, features, forward_dt)
        reversed_features = features.flip(1)
        reverse = self._run_cfc(
            self.reverse_cfc, reversed_features, reverse_dt
        ).flip(1)
        return self.projection(torch.cat((forward, reverse), dim=-1))
