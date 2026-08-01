"""Immutable tensor records shared by imputation v3."""

from dataclasses import dataclass, field

import torch


def _copy(value: torch.Tensor) -> torch.Tensor:
    """Return tensor storage isolated from callers and autograd history."""

    return value.detach().clone()


@dataclass(frozen=True, init=False, repr=False, eq=False)
class FeatureBatch:
    """Leakage-safe features with their aligned time steps and mask."""

    _values: torch.Tensor = field(repr=False)
    _dt: torch.Tensor = field(repr=False)
    _mask: torch.Tensor = field(repr=False)

    def __init__(self, values: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor) -> None:
        object.__setattr__(self, "_values", _copy(values))
        object.__setattr__(self, "_dt", _copy(dt))
        object.__setattr__(self, "_mask", _copy(mask))

    @property
    def values(self) -> torch.Tensor:
        return _copy(self._values)

    @property
    def dt(self) -> torch.Tensor:
        return _copy(self._dt)

    @property
    def mask(self) -> torch.Tensor:
        return _copy(self._mask)


__all__ = ["FeatureBatch"]
