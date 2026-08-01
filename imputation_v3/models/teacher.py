"""Offline full-context teacher for six-channel inertial reconstruction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import torch
from torch import nn

from imputation_v3.models.baselines import complete_signal
from imputation_v3.models.cfc import BidirectionalCfCEncoder
from imputation_v3.models.tcn import SymmetricTCNEncoder


_RESIDUAL_MODES = frozenset({"residual", "raw"})
_TIME_MODES = frozenset({"actual", "constant", "dt_feature_only", "no_dt"})
_OUTPUT_CHANNELS = 6


@dataclass(frozen=True)
class TeacherOutput:
    """Live teacher tensors, with immutable field assignment."""

    raw: torch.Tensor
    completed: torch.Tensor
    residual: torch.Tensor
    latent: torch.Tensor


def _validate_mode(name: str, value: object, choices: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string mode")
    if value not in choices:
        declared = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {declared}")
    return value


class OfflineTeacher(nn.Module):
    """Fuse bidirectional CfC, symmetric TCN, and linear baseline features."""

    def __init__(
        self,
        input_size: int,
        cfc_hidden: int,
        tcn_width: int,
        tcn_dilations: Sequence[int],
        *,
        residual_mode: str = "residual",
        time_mode: str = "actual",
    ) -> None:
        super().__init__()
        self.residual_mode = _validate_mode(
            "residual_mode", residual_mode, _RESIDUAL_MODES
        )
        self.time_mode = _validate_mode("time_mode", time_mode, _TIME_MODES)

        self.cfc_encoder = BidirectionalCfCEncoder(input_size, cfc_hidden)
        self.tcn_encoder = SymmetricTCNEncoder(
            input_size, width=tcn_width, dilations=tcn_dilations
        )
        fusion_size = 2 * self.cfc_encoder.hidden_size + self.tcn_encoder.width + 6
        self.trunk = nn.Sequential(
            nn.Linear(fusion_size, 96),
            nn.GELU(),
            nn.Linear(96, 48),
            nn.GELU(),
        )
        self.gyro_head = nn.Linear(48, 3)
        self.acc_head = nn.Linear(48, 3)

    def forward(
        self,
        features: torch.Tensor,
        dt: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
        baseline: torch.Tensor,
        *,
        time_mode: str | None = None,
        nominal_dt_s: Real = 0.01,
    ) -> TeacherOutput:
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

        for name, value in (
            ("features", features),
            ("dt", dt),
            ("observed", observed),
            ("baseline", baseline),
        ):
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point")

        if features.ndim != 3:
            raise ValueError("features must have shape (batch, time, input_size)")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("features batch and time axes must be nonempty")
        expected_features = (
            features.shape[0],
            features.shape[1],
            self.cfc_encoder.input_size,
        )
        if features.shape != expected_features:
            raise ValueError(
                "features final dimension must equal "
                f"input_size ({self.cfc_encoder.input_size})"
            )

        expected_sequence = features.shape[:2]
        if dt.shape != expected_sequence:
            raise ValueError("dt shape must match features batch and time axes")
        expected_signal = (*expected_sequence, _OUTPUT_CHANNELS)
        for name, value in (
            ("observed", observed),
            ("mask", mask),
            ("baseline", baseline),
        ):
            if value.shape != expected_signal:
                raise ValueError(f"{name} must have shape (batch, time, 6)")

        for name, value in (
            ("dt", dt),
            ("observed", observed),
            ("baseline", baseline),
        ):
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
        if not torch.isfinite(dt).all().item():
            raise ValueError("dt must be finite")
        if not torch.isfinite(baseline).all().item():
            raise ValueError("baseline must be finite")
        mask_bool = mask.to(dtype=torch.bool)
        if not torch.isfinite(observed[mask_bool]).all().item():
            raise ValueError("observed values selected by mask must be finite")

        selected_time_mode = self.time_mode if time_mode is None else time_mode
        cfc = self.cfc_encoder(
            features,
            dt,
            mode=selected_time_mode,
            nominal_dt_s=nominal_dt_s,
        )
        tcn = self.tcn_encoder(features)
        latent = self.trunk(torch.cat((cfc, tcn, baseline), dim=-1))
        residual = torch.cat((self.gyro_head(latent), self.acc_head(latent)), dim=-1)
        raw = baseline + residual if self.residual_mode == "residual" else residual

        flat_shape = (-1, _OUTPUT_CHANNELS)
        completed = complete_signal(
            observed.reshape(flat_shape),
            mask.reshape(flat_shape),
            raw.reshape(flat_shape),
        ).reshape_as(observed)
        return TeacherOutput(
            raw=raw, completed=completed, residual=residual, latent=latent
        )


__all__ = ["OfflineTeacher", "TeacherOutput"]
