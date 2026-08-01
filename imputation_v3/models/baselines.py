"""Deterministic full-window baselines and observed-value completion."""

from math import isfinite
from numbers import Real

import torch


_RTS_DIFFUSE_PRIOR_VARIANCE = 1e6


def _validate_finite_scalar(value: Real, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real finite scalar")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if positive and converted <= 0:
        raise ValueError(f"{name} must be strictly positive")
    return converted


def _validate_source_and_mask(
    source: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if not isinstance(source, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("source and mask must be torch tensors")
    if source.ndim != 2:
        raise ValueError("source must be a 2-D (time, channels) tensor")
    if source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError("source must be non-empty")
    if not source.is_floating_point():
        raise TypeError("source must be a floating tensor")
    if mask.shape != source.shape:
        raise ValueError("mask shape must match source shape")
    if mask.device != source.device:
        raise ValueError("source and mask must share a device")
    if mask.is_complex():
        raise TypeError("mask must be real-valued and contain only 0 or 1")
    if not torch.all((mask == 0) | (mask == 1)).item():
        raise ValueError("mask must contain exactly 0 or 1")

    mask_bool = mask.to(dtype=torch.bool)
    if not torch.isfinite(source[mask_bool]).all().item():
        raise ValueError("observed source values must be finite")
    return mask_bool


def _validate_time(
    time: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    if not isinstance(time, torch.Tensor):
        raise TypeError("time must be a torch tensor")
    if time.ndim != 1 or time.shape[0] != source.shape[0]:
        raise ValueError("time must have shape (time,)")
    if time.device != source.device:
        raise ValueError("source, mask, and time must share a device")
    if not time.is_floating_point():
        raise TypeError("time must be a floating tensor")
    if not torch.isfinite(time).all().item():
        raise ValueError("time must contain only finite values")
    if time.numel() > 1 and not torch.all(time[1:] > time[:-1]).item():
        raise ValueError("time must be strictly increasing")
    return time


def _validate_baseline_inputs(
    source: torch.Tensor,
    mask: torch.Tensor,
    time: torch.Tensor,
    empty_fill: Real,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    fill = _validate_finite_scalar(empty_fill, "empty_fill")
    mask_bool = _validate_source_and_mask(source, mask)
    valid_time = _validate_time(time, source)
    return mask_bool, valid_time, fill


def complete_signal(
    observed: torch.Tensor,
    mask: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    """Fill missing entries while returning every observed entry unchanged."""
    mask_bool = _validate_source_and_mask(observed, mask)
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("prediction must be a torch tensor")
    if prediction.shape != observed.shape:
        raise ValueError("prediction shape must match observed shape")
    if prediction.device != observed.device:
        raise ValueError("observed, mask, and prediction must share a device")
    if not prediction.is_floating_point():
        raise TypeError("prediction must be a floating tensor")
    if prediction.dtype != observed.dtype:
        raise ValueError("prediction dtype must match observed dtype")

    missing = ~mask_bool
    if not torch.isfinite(prediction[missing]).all().item():
        raise ValueError("prediction values used at missing entries must be finite")
    result = torch.where(mask_bool, observed, prediction)
    if not torch.isfinite(result).all().item():
        raise ValueError("completed signal must be finite")
    return result


def timestamp_locf(
    source: torch.Tensor,
    mask: torch.Tensor,
    time: torch.Tensor,
    *,
    empty_fill: Real,
) -> torch.Tensor:
    """Carry the latest observation forward, using ``empty_fill`` before it."""
    mask_bool, _, fill = _validate_baseline_inputs(source, mask, time, empty_fill)
    rows = torch.arange(source.shape[0], device=source.device)[:, None]
    observed_rows = torch.where(mask_bool, rows, torch.full_like(rows, -1))
    last_rows = torch.cummax(observed_rows, dim=0).values
    gathered = source.gather(0, last_rows.clamp_min(0))
    prediction = torch.where(
        last_rows >= 0, gathered, torch.full_like(source, fill)
    )
    return complete_signal(source, mask_bool, prediction)


def timestamp_linear(
    source: torch.Tensor,
    mask: torch.Tensor,
    time: torch.Tensor,
    *,
    empty_fill: Real,
) -> torch.Tensor:
    """Linearly interpolate in real time with nearest-value boundary holds."""
    mask_bool, valid_time, fill = _validate_baseline_inputs(
        source, mask, time, empty_fill
    )
    prediction = torch.full_like(source, fill)
    query = valid_time.to(dtype=torch.float64)

    for channel in range(source.shape[1]):
        observed_rows = torch.nonzero(mask_bool[:, channel], as_tuple=False).flatten()
        count = observed_rows.numel()
        if count == 0:
            continue
        values = source[observed_rows, channel].to(dtype=torch.float64)
        if count == 1:
            channel_prediction = values.expand(source.shape[0])
        else:
            observed_time = query[observed_rows]
            positions = torch.searchsorted(observed_time, query)
            right = positions.clamp(max=count - 1)
            left = (positions - 1).clamp(min=0, max=count - 1)
            same = left == right
            denominator = observed_time[right] - observed_time[left]
            weight = torch.where(
                same,
                torch.zeros_like(query),
                (query - observed_time[left]) / denominator.clamp_min(
                    torch.finfo(torch.float64).tiny
                ),
            )
            channel_prediction = values[left] + weight * (values[right] - values[left])
        prediction[:, channel] = channel_prediction.to(dtype=source.dtype)

    return complete_signal(source, mask_bool, prediction)


def timestamp_pchip(
    source: torch.Tensor,
    mask: torch.Tensor,
    time: torch.Tensor,
    *,
    empty_fill: Real,
) -> torch.Tensor:
    """Apply CPU/SciPy PCHIP interpolation with nearest-value boundary holds."""
    mask_bool, valid_time, fill = _validate_baseline_inputs(
        source, mask, time, empty_fill
    )
    from scipy.interpolate import PchipInterpolator

    query = valid_time.detach().to(device="cpu", dtype=torch.float64).numpy()
    prediction = torch.full(
        source.shape, fill, dtype=torch.float64, device="cpu"
    )
    cpu_mask = mask_bool.detach().to(device="cpu")
    cpu_source = source.detach().to(device="cpu", dtype=torch.float64)

    for channel in range(source.shape[1]):
        observed_rows = torch.nonzero(cpu_mask[:, channel], as_tuple=False).flatten()
        count = observed_rows.numel()
        if count == 0:
            continue
        values = cpu_source[observed_rows, channel].numpy()
        if count == 1:
            prediction[:, channel].fill_(float(values[0]))
            continue
        observed_time = query[observed_rows.numpy()]
        clipped_query = query.clip(observed_time[0], observed_time[-1])
        interpolator = PchipInterpolator(observed_time, values, extrapolate=False)
        prediction[:, channel] = torch.from_numpy(interpolator(clipped_query))

    restored = prediction.to(device=source.device, dtype=source.dtype)
    return complete_signal(source, mask_bool, restored)


def _transition_and_noise(
    dt: torch.Tensor, process_var: float
) -> tuple[torch.Tensor, torch.Tensor]:
    one = torch.ones((), dtype=torch.float64, device=dt.device)
    zero = torch.zeros((), dtype=torch.float64, device=dt.device)
    transition = torch.stack(
        (torch.stack((one, dt)), torch.stack((zero, one)))
    )
    dt2 = dt * dt
    noise = process_var * torch.stack(
        (
            torch.stack((dt2 * dt / 3.0, dt2 / 2.0)),
            torch.stack((dt2 / 2.0, dt)),
        )
    )
    return transition, noise


def _require_finite_rts(*values: torch.Tensor) -> None:
    if any(not torch.isfinite(value).all().item() for value in values):
        raise ValueError("RTS linear algebra produced non-finite values")


def constant_velocity_rts(
    source: torch.Tensor,
    mask: torch.Tensor,
    time: torch.Tensor,
    *,
    empty_fill: Real,
    process_var: Real,
    observation_var: Real,
) -> torch.Tensor:
    """Run an offline constant-velocity RTS smoother independently per channel.

    The state is ``[value, velocity]``. Real timestamp gaps define
    ``F=[[1, dt], [0, 1]]`` and
    ``Q=q*[[dt^3/3, dt^2/2], [dt^2/2, dt]]``; observations use
    ``H=[1, 0]`` and scalar ``R=observation_var``. At row zero the filter uses
    an identity transition and an observation-independent diffuse prior with
    mean ``[empty_fill, 0]``. Thus every measurement is assimilated exactly
    once at its own timestamp, while the backward RTS pass legitimately gives
    this offline baseline access to future observations.

    A channel with no observations is filled with ``empty_fill``. Channels
    with one or all rows observed use the same filter/smoother; final observed
    entries are restored exactly by :func:`complete_signal`.
    """
    mask_bool, valid_time, fill = _validate_baseline_inputs(
        source, mask, time, empty_fill
    )
    q = _validate_finite_scalar(process_var, "process_var", positive=True)
    r = _validate_finite_scalar(observation_var, "observation_var", positive=True)

    with torch.no_grad():
        internal_time = valid_time.to(dtype=torch.float64)
        prediction = torch.full_like(source, fill)
        identity = torch.eye(2, dtype=torch.float64, device=source.device)
        observation = torch.tensor(
            [1.0, 0.0], dtype=torch.float64, device=source.device
        )

        for channel in range(source.shape[1]):
            observed_rows = torch.nonzero(
                mask_bool[:, channel], as_tuple=False
            ).flatten()
            if observed_rows.numel() == 0:
                continue
            state = torch.stack(
                (
                    torch.as_tensor(fill, dtype=torch.float64, device=source.device),
                    torch.zeros((), dtype=torch.float64, device=source.device),
                )
            )
            covariance = identity * _RTS_DIFFUSE_PRIOR_VARIANCE

            predicted_states: list[torch.Tensor] = []
            predicted_covariances: list[torch.Tensor] = []
            filtered_states: list[torch.Tensor] = []
            filtered_covariances: list[torch.Tensor] = []
            transitions: list[torch.Tensor] = []

            for row in range(source.shape[0]):
                if row == 0:
                    transition = identity
                    predicted_state = state
                    predicted_covariance = covariance
                else:
                    dt = internal_time[row] - internal_time[row - 1]
                    transition, noise = _transition_and_noise(dt, q)
                    predicted_state = transition @ state
                    predicted_covariance = (
                        transition @ covariance @ transition.T + noise
                    )
                predicted_covariance = (
                    predicted_covariance + predicted_covariance.T
                ) / 2.0
                _require_finite_rts(predicted_state, predicted_covariance)

                if mask_bool[row, channel].item():
                    innovation = (
                        source[row, channel].to(dtype=torch.float64)
                        - observation @ predicted_state
                    )
                    innovation_covariance = (
                        observation @ predicted_covariance @ observation + r
                    )
                    gain = (
                        predicted_covariance @ observation
                    ) / innovation_covariance
                    state = predicted_state + gain * innovation
                    residual_transform = identity - torch.outer(gain, observation)
                    covariance = (
                        residual_transform
                        @ predicted_covariance
                        @ residual_transform.T
                        + r * torch.outer(gain, gain)
                    )
                else:
                    state = predicted_state
                    covariance = predicted_covariance
                covariance = (covariance + covariance.T) / 2.0
                _require_finite_rts(state, covariance)
                transitions.append(transition)
                predicted_states.append(predicted_state)
                predicted_covariances.append(predicted_covariance)
                filtered_states.append(state)
                filtered_covariances.append(covariance)

            smoothed_states = list(filtered_states)
            smoothed_covariance = filtered_covariances[-1]
            for row in range(source.shape[0] - 2, -1, -1):
                transition = transitions[row + 1]
                try:
                    smoother_gain = torch.linalg.solve(
                        predicted_covariances[row + 1],
                        transition @ filtered_covariances[row],
                    ).T
                except RuntimeError as exc:
                    raise ValueError(
                        "RTS smoothing solve failed for "
                        f"channel {channel} at time index {row}"
                    ) from exc
                smoothed_states[row] = filtered_states[row] + smoother_gain @ (
                    smoothed_states[row + 1] - predicted_states[row + 1]
                )
                smoothed_covariance = filtered_covariances[row] + smoother_gain @ (
                    smoothed_covariance - predicted_covariances[row + 1]
                ) @ smoother_gain.T
                smoothed_covariance = (
                    smoothed_covariance + smoothed_covariance.T
                ) / 2.0
                _require_finite_rts(smoothed_states[row], smoothed_covariance)

            channel_prediction = torch.stack(smoothed_states)[:, 0]
            _require_finite_rts(channel_prediction)
            prediction[:, channel] = channel_prediction.to(dtype=source.dtype)

        return complete_signal(source, mask_bool, prediction)
