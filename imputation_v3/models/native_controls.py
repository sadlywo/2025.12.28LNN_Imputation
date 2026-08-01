"""Native equal-input controls for the offline teacher experiment."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from types import MappingProxyType

import torch
from torch import nn

from .baselines import complete_signal
from .cfc import BidirectionalCfCEncoder
from .tcn import SymmetricTCNEncoder
from .teacher import TeacherOutput


CONTROL_CONDITIONS = ("bilstm", "bilnn", "tcn", "feature_mlp")
TEACHER_CONDITION_MODES = MappingProxyType(
    {
        "teacher_actual_residual": ("actual", "residual"),
        "teacher_constant_residual": ("constant", "residual"),
        "teacher_dt_feature_only_residual": ("dt_feature_only", "residual"),
        "teacher_no_dt_residual": ("no_dt", "residual"),
        "teacher_actual_raw": ("actual", "raw"),
    }
)
NATIVE_CONDITIONS = (*CONTROL_CONDITIONS, *TEACHER_CONDITION_MODES)
_OUTPUT_CHANNELS = 6


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def count_parameters(model: nn.Module) -> int:
    """Count trainable scalar parameters in ``model``."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch nn.Module")
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def teacher_condition_modes(condition: str) -> tuple[str, str]:
    """Return the frozen ``(time_mode, residual_mode)`` teacher ablation pair."""
    if not isinstance(condition, str):
        raise TypeError("condition must be a string")
    try:
        return TEACHER_CONDITION_MODES[condition]
    except KeyError as exc:
        raise ValueError(f"unsupported teacher condition: {condition}") from exc


class _ResidualControl(nn.Module):
    def __init__(self, input_size: int, representation_size: int) -> None:
        super().__init__()
        self.input_size = _positive_integer(input_size, "input_size")
        self.representation_size = _positive_integer(
            representation_size, "representation_size"
        )
        self.head = nn.Sequential(
            nn.Linear(self.representation_size + _OUTPUT_CHANNELS, 48),
            nn.GELU(),
            nn.Linear(48, _OUTPUT_CHANNELS),
        )

    def _validate_inputs(
        self,
        features: torch.Tensor,
        dt: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        tensors = {
            "features": features,
            "dt": dt,
            "observed": observed,
            "mask": mask,
            "baseline": baseline,
        }
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch tensor")
        for name in ("features", "dt", "observed", "baseline"):
            if not tensors[name].is_floating_point():
                raise TypeError(f"{name} must be floating point")

        if features.ndim != 3:
            raise ValueError("features must have shape (batch, time, input_size)")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("features batch and time axes must be nonempty")
        if features.shape[2] != self.input_size:
            raise ValueError(
                f"features final dimension must equal input_size ({self.input_size})"
            )
        sequence_shape = features.shape[:2]
        if dt.shape != sequence_shape:
            raise ValueError("dt shape must match features batch and time axes")
        signal_shape = (*sequence_shape, _OUTPUT_CHANNELS)
        for name in ("observed", "mask", "baseline"):
            if tensors[name].shape != signal_shape:
                raise ValueError(f"{name} must have shape (batch, time, 6)")

        for name in ("dt", "observed", "baseline"):
            value = tensors[name]
            if value.dtype != features.dtype:
                raise TypeError(f"{name} must have the same dtype as features")
            if value.device != features.device:
                raise ValueError(f"{name} must be on the same device as features")
        if mask.device != features.device:
            raise ValueError("mask must be on the same device as features")
        if mask.dtype != torch.bool and (
            not mask.is_floating_point() or mask.dtype != features.dtype
        ):
            raise TypeError(
                "mask must be bool or have the same floating dtype as features"
            )
        if not torch.all((mask == 0) | (mask == 1)).item():
            raise ValueError("mask must be exact binary 0 or 1")

        if not torch.isfinite(features).all().item():
            raise ValueError("features must be finite")
        if not torch.all(torch.isfinite(dt) & (dt > 0)).item():
            raise ValueError("dt must be finite and strictly positive")
        if not torch.isfinite(baseline).all().item():
            raise ValueError("baseline must be finite")
        mask_bool = mask.to(dtype=torch.bool)
        if not torch.isfinite(observed[mask_bool]).all().item():
            raise ValueError("observed values selected by mask must be finite")
        return mask_bool

    def _finish(
        self,
        representation: torch.Tensor,
        observed: torch.Tensor,
        mask_bool: torch.Tensor,
        baseline: torch.Tensor,
    ) -> TeacherOutput:
        expected = (*observed.shape[:2], self.representation_size)
        if representation.shape != expected:
            raise ValueError(
                "encoder representation must have shape "
                f"(batch, time, {self.representation_size})"
            )
        if representation.dtype != baseline.dtype:
            raise TypeError("encoder representation must preserve input dtype")
        if representation.device != baseline.device:
            raise ValueError("encoder representation must preserve input device")
        if not torch.isfinite(representation).all().item():
            raise ValueError("encoder representation must be finite")

        latent = torch.cat((representation, baseline), dim=-1)
        residual = self.head(latent)
        raw = baseline + residual
        if raw.dtype != baseline.dtype or raw.device != baseline.device:
            raise TypeError("control output must preserve input dtype and device")
        if not torch.isfinite(raw).all().item():
            raise ValueError("control output must be finite")
        flat_shape = (-1, _OUTPUT_CHANNELS)
        completed = complete_signal(
            observed.reshape(flat_shape),
            mask_bool.reshape(flat_shape),
            raw.reshape(flat_shape),
        ).reshape_as(observed)
        return TeacherOutput(
            raw=raw,
            completed=completed,
            residual=residual,
            latent=latent,
        )


class BiLSTMControl(_ResidualControl):
    """Bidirectional LSTM with the teacher's baseline-residual head."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        hidden = _positive_integer(hidden_size, "hidden_size")
        super().__init__(input_size, hidden * 2)
        self.hidden_size = hidden
        self.encoder = nn.LSTM(
            self.input_size, hidden, batch_first=True, bidirectional=True
        )

    def forward(self, features, dt, observed, mask, baseline) -> TeacherOutput:
        mask_bool = self._validate_inputs(features, dt, observed, mask, baseline)
        representation, _ = self.encoder(features)
        return self._finish(representation, observed, mask_bool, baseline)


class BiCfCControl(_ResidualControl):
    """Bidirectional CfC control using actual elapsed times."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        hidden = _positive_integer(hidden_size, "hidden_size")
        super().__init__(input_size, hidden * 2)
        self.hidden_size = hidden
        self.encoder = BidirectionalCfCEncoder(self.input_size, hidden)

    def forward(self, features, dt, observed, mask, baseline) -> TeacherOutput:
        mask_bool = self._validate_inputs(features, dt, observed, mask, baseline)
        representation = self.encoder(features, dt, mode="actual")
        return self._finish(representation, observed, mask_bool, baseline)


class TCNControl(_ResidualControl):
    """Symmetric TCN-only equal-input control."""

    def __init__(
        self, input_size: int, width: int, dilations: Sequence[int]
    ) -> None:
        validated_input = _positive_integer(input_size, "input_size")
        self.encoder: SymmetricTCNEncoder
        encoder = SymmetricTCNEncoder(
            validated_input, width=width, dilations=dilations
        )
        super().__init__(validated_input, encoder.width)
        self.encoder = encoder

    def forward(self, features, dt, observed, mask, baseline) -> TeacherOutput:
        mask_bool = self._validate_inputs(features, dt, observed, mask, baseline)
        representation = self.encoder(features)
        return self._finish(representation, observed, mask_bool, baseline)


class FeatureMLPControl(_ResidualControl):
    """Per-time-step MLP control with no temporal encoder."""

    def __init__(self, input_size: int, width: int) -> None:
        validated_input = _positive_integer(input_size, "input_size")
        validated_width = _positive_integer(width, "width")
        super().__init__(validated_input, validated_width)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, validated_width),
            nn.GELU(),
            nn.Linear(validated_width, validated_width),
        )

    def forward(self, features, dt, observed, mask, baseline) -> TeacherOutput:
        mask_bool = self._validate_inputs(features, dt, observed, mask, baseline)
        representation = self.encoder(features)
        return self._finish(representation, observed, mask_bool, baseline)


__all__ = [
    "BiCfCControl",
    "BiLSTMControl",
    "CONTROL_CONDITIONS",
    "FeatureMLPControl",
    "NATIVE_CONDITIONS",
    "TCNControl",
    "TEACHER_CONDITION_MODES",
    "count_parameters",
    "teacher_condition_modes",
]
