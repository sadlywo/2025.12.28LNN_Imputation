"""Symmetric full-window temporal convolutional features for the teacher."""

from collections.abc import Sequence
from math import isfinite
from numbers import Integral, Real

import torch
from torch import nn


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _odd_kernel_size(value: object) -> int:
    converted = _positive_integer(value, "kernel_size")
    if converted % 2 == 0:
        raise ValueError("kernel_size must be odd for symmetric length preservation")
    return converted


def _dropout_probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("dropout must be a finite numeric value in [0, 1)")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError("dropout must be finite")
    if not 0.0 <= converted < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    return converted


def _positive_dilations(values: object) -> tuple[int, ...]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
    ):
        raise TypeError("dilations must be a non-empty sequence of positive integers")
    converted = []
    for value in values:
        try:
            converted.append(_positive_integer(value, "dilations"))
        except (TypeError, ValueError) as exc:
            raise type(exc)(
                "dilations must contain only positive integers"
            ) from exc
    return tuple(converted)


class DepthwiseResidualBlock(nn.Module):
    """A symmetric depthwise-separable temporal residual block."""

    def __init__(
        self,
        width: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.width = _positive_integer(width, "width")
        self.kernel_size = _odd_kernel_size(kernel_size)
        self.dilation = _positive_integer(dilation, "dilation")
        probability = _dropout_probability(dropout)
        padding = self.dilation * (self.kernel_size - 1) // 2

        self.depthwise = nn.Conv1d(
            self.width,
            self.width,
            self.kernel_size,
            dilation=self.dilation,
            groups=self.width,
            padding=padding,
        )
        self.pointwise = nn.Conv1d(self.width, self.width, 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(probability)
        self.normalization = nn.LayerNorm(self.width)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Transform features in explicit batch-time-channel layout."""
        residual = features
        channels_first = features.transpose(1, 2)
        transformed = self.depthwise(channels_first)
        transformed = self.pointwise(transformed)
        transformed = transformed.transpose(1, 2)
        transformed = self.activation(transformed)
        transformed = self.dropout(transformed)
        return self.normalization(residual + transformed)


class SymmetricTCNEncoder(nn.Module):
    """Encode every input time step using symmetric full-window context."""

    def __init__(
        self,
        input_size: int,
        *,
        width: int,
        dilations: Sequence[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = _positive_integer(input_size, "input_size")
        self.width = _positive_integer(width, "width")
        self.dilations = _positive_dilations(dilations)
        self.kernel_size = _odd_kernel_size(kernel_size)
        probability = _dropout_probability(dropout)

        self.projection = nn.Linear(self.input_size, self.width)
        self.blocks = nn.ModuleList(
            DepthwiseResidualBlock(
                self.width,
                self.kernel_size,
                dilation,
                probability,
            )
            for dilation in self.dilations
        )

    @property
    def receptive_field(self) -> int:
        """Return the exact full-window receptive-field width in samples."""
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one width-dimensional representation per input time step."""
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch tensor")
        if not features.is_floating_point():
            raise TypeError("features must be a floating tensor")
        if features.ndim != 3:
            raise ValueError("features must be a 3-D (batch, time, input_size) tensor")
        if features.shape[0] == 0:
            raise ValueError("features batch axis must be non-empty")
        if features.shape[1] == 0:
            raise ValueError("features time axis must be non-empty")
        if features.shape[2] != self.input_size:
            raise ValueError(
                f"features final dimension must equal input_size={self.input_size}"
            )
        if not torch.isfinite(features).all().item():
            raise ValueError("features must contain only finite values")

        encoded = self.projection(features)
        for block in self.blocks:
            encoded = block(encoded)
        return encoded
