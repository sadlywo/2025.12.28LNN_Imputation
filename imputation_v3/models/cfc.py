"""Bidirectional continuous-time encoder with explicit elapsed-time controls."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real

import torch
from torch import nn


_TIME_MODES = frozenset({"actual", "constant", "dt_feature_only", "no_dt"})
_DT_FEATURE_INDEX = 12


def _default_cfc_factory(*args, **kwargs) -> nn.Module:
    """Import ncps only when the default encoder is actually constructed."""

    from ncps.torch import CfC

    return CfC(*args, **kwargs)


def _validate_positive_integer(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        qualifier = f"at least {minimum}" if minimum > 1 else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _validate_nominal_dt(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("nominal_dt_s must be a real number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("nominal_dt_s must be finite and strictly positive") from error
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError("nominal_dt_s must be finite and strictly positive")
    return converted


def reverse_aligned_dt(dt: torch.Tensor) -> torch.Tensor:
    """Align positive elapsed intervals with a reversed ``(B, T)`` sequence.

    The first reversed sample reuses the final forward interval because there is
    no interval beyond the original sequence endpoint. For ``T >= 2`` this is
    ``cat(dt[:, -1:], dt[:, 1:].flip(1))``; a singleton sequence is unchanged.
    """

    if not isinstance(dt, torch.Tensor):
        raise TypeError("dt must be a torch tensor")
    if dt.ndim != 2:
        raise ValueError("dt must be a 2-D tensor with shape (batch, time)")
    if dt.shape[0] == 0:
        raise ValueError("dt batch axis must be nonempty")
    if dt.shape[1] == 0:
        raise ValueError("dt time axis must be nonempty")
    if not dt.is_floating_point():
        raise TypeError("dt must be floating point")
    if not torch.isfinite(dt).all():
        raise ValueError("dt must be finite")
    if not torch.all(dt > 0):
        raise ValueError("dt must be strictly positive")
    if dt.shape[1] == 1:
        return dt.clone()
    return torch.cat((dt[:, -1:], dt[:, 1:].flip(1)), dim=1)


def _sequence_output(
    result: object,
    *,
    batch_size: int,
    time_steps: int,
    hidden_size: int,
) -> torch.Tensor:
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError("CfC tuple output must contain exactly 2 items")
        result = result[0]
    if not isinstance(result, torch.Tensor):
        raise TypeError("CfC must return a tensor or (tensor, hidden_state)")
    expected_shape = (batch_size, time_steps, hidden_size)
    if result.shape != expected_shape:
        raise ValueError(f"CfC output must have shape {expected_shape}")
    if not torch.isfinite(result).all():
        raise ValueError("CfC output must be finite")
    return result


class BidirectionalCfCEncoder(nn.Module):
    """Encode a sequence in both directions using explicit CfC timespans.

    Time modes control the direct elapsed-time feature at column 12 and the
    separate ``timespans`` argument passed to CfC. ``no_dt`` only zeros that
    direct column; it deliberately retains age, slope, and any other
    time-derived columns in the shared 31-D feature contract.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        cfc_factory: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.input_size = _validate_positive_integer(
            "input_size", input_size, minimum=_DT_FEATURE_INDEX + 1
        )
        self.hidden_size = _validate_positive_integer("hidden_size", hidden_size)
        if cfc_factory is not None and not callable(cfc_factory):
            raise TypeError("cfc_factory must be callable or None")

        self._uses_default_cfc = cfc_factory is None
        factory = _default_cfc_factory if cfc_factory is None else cfc_factory
        options = {"batch_first": True, "return_sequences": True}
        forward_cfc = factory(self.input_size, self.hidden_size, **options)
        if not isinstance(forward_cfc, nn.Module):
            raise TypeError("cfc_factory must construct an nn.Module")
        reverse_cfc = factory(self.input_size, self.hidden_size, **options)
        if not isinstance(reverse_cfc, nn.Module):
            raise TypeError("cfc_factory must construct an nn.Module")
        self.forward_cfc = forward_cfc
        self.reverse_cfc = reverse_cfc

    def _run_cfc(
        self, module: nn.Module, features: torch.Tensor, timespans: torch.Tensor
    ) -> torch.Tensor:
        supplied_timespans = timespans
        if self._uses_default_cfc:
            # ncps 1.0.1 squeezes each (B,) time slice before combining it with
            # (B, hidden). Repeating the scalar across hidden units preserves
            # its value while producing the broadcast-compatible shape.
            supplied_timespans = timespans.unsqueeze(-1).expand(
                -1, -1, self.hidden_size
            )
        result = module(features, timespans=supplied_timespans)
        return _sequence_output(
            result,
            batch_size=features.shape[0],
            time_steps=features.shape[1],
            hidden_size=self.hidden_size,
        )

    def forward(
        self,
        features: torch.Tensor,
        dt: torch.Tensor,
        *,
        mode: str,
        nominal_dt_s: Real = 0.01,
    ) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch tensor")
        if not features.is_floating_point():
            raise TypeError("features must be floating point")
        if features.ndim != 3:
            raise ValueError("features must be a 3-D tensor")
        if features.shape[0] == 0:
            raise ValueError("features batch axis must be nonempty")
        if features.shape[1] == 0:
            raise ValueError("features time axis must be nonempty")
        if features.shape[2] != self.input_size:
            raise ValueError(
                f"features final dimension must equal input_size ({self.input_size})"
            )
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")

        if not isinstance(dt, torch.Tensor):
            raise TypeError("dt must be a torch tensor")
        if dt.shape != features.shape[:2]:
            raise ValueError("dt must have shape (batch, time)")
        if dt.dtype != features.dtype:
            raise TypeError("dt must have the same dtype as features")
        if dt.device != features.device:
            raise ValueError("dt must be on the same device as features")
        if not torch.isfinite(dt).all():
            raise ValueError("dt must be finite")
        if not torch.all(dt > 0):
            raise ValueError("dt must be strictly positive")

        if not isinstance(mode, str) or mode not in _TIME_MODES:
            declared = ", ".join(sorted(_TIME_MODES))
            raise ValueError(f"mode must be one of: {declared}")
        nominal = _validate_nominal_dt(nominal_dt_s)

        encoded_features = features.clone()
        actual_dt = dt.clone()
        constant_dt = torch.full_like(actual_dt, nominal)
        if not torch.isfinite(constant_dt).all() or not torch.all(constant_dt > 0):
            raise ValueError("nominal_dt_s is not representable in the input dtype")

        if mode == "constant":
            encoded_features[..., _DT_FEATURE_INDEX] = nominal
        elif mode == "no_dt":
            encoded_features[..., _DT_FEATURE_INDEX] = 0

        if mode == "actual":
            forward_dt = actual_dt
            reverse_dt = reverse_aligned_dt(actual_dt)
        else:
            forward_dt = constant_dt
            reverse_dt = constant_dt.clone()

        forward = self._run_cfc(self.forward_cfc, encoded_features, forward_dt)
        reverse = self._run_cfc(
            self.reverse_cfc, encoded_features.flip(1), reverse_dt
        ).flip(1)
        return torch.cat((forward, reverse), dim=-1)


__all__ = ["BidirectionalCfCEncoder", "reverse_aligned_dt"]
