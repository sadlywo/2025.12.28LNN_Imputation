import math

import pytest
import torch
from torch import nn

from imputation_v3.models.cfc import BidirectionalCfCEncoder, reverse_aligned_dt
from validation_v2.experiments.runner import reverse_aligned_dt as reverse_aligned_dt_v2


class SpyCfC(nn.Module):
    def __init__(self, hidden_size, direction, output_factory=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.direction = direction
        self.output_factory = output_factory
        self.calls = []

    def forward(self, features, *, timespans):
        self.calls.append((features.detach().clone(), timespans.detach().clone()))
        if self.output_factory is not None:
            return self.output_factory(self, features, timespans)
        return features.new_zeros((*features.shape[:2], self.hidden_size))


class SpyFactory:
    def __init__(self, output_factory=None):
        self.modules = []
        self.constructor_calls = []
        self.output_factory = output_factory

    def __call__(self, input_size, hidden_size, **kwargs):
        self.constructor_calls.append((input_size, hidden_size, kwargs))
        module = SpyCfC(hidden_size, len(self.modules), self.output_factory)
        self.modules.append(module)
        return module


def make_inputs(batch=2, time=4, input_size=31):
    features = torch.arange(
        batch * time * input_size, dtype=torch.float32
    ).reshape(batch, time, input_size)
    dt = torch.tensor(
        [[0.01, 0.02, 0.04, 0.08], [0.03, 0.05, 0.07, 0.11]],
        dtype=features.dtype,
    )[:batch, :time]
    return features, dt


def test_actual_mode_passes_direction_aligned_elapsed_time_and_bidirectional_shape():
    factory = SpyFactory()
    model = BidirectionalCfCEncoder(31, 5, cfc_factory=factory)
    features, dt = make_inputs()

    output = model(features, dt, mode="actual")

    assert output.shape == (2, 4, 10)
    assert len(factory.modules) == 2
    assert factory.constructor_calls == [
        (31, 5, {"batch_first": True, "return_sequences": True}),
        (31, 5, {"batch_first": True, "return_sequences": True}),
    ]
    forward_features, forward_timespans = factory.modules[0].calls[0]
    reverse_features, reverse_timespans = factory.modules[1].calls[0]
    torch.testing.assert_close(forward_features, features)
    torch.testing.assert_close(forward_timespans, dt)
    torch.testing.assert_close(reverse_features, features.flip(1))
    torch.testing.assert_close(reverse_timespans, reverse_aligned_dt(dt))


@pytest.mark.parametrize(
    ("mode", "use_actual_timespans", "dt_feature"),
    (
        ("actual", True, "actual"),
        ("constant", False, "nominal"),
        ("dt_feature_only", False, "actual"),
        ("no_dt", False, "zero"),
    ),
)
def test_time_modes_control_only_direct_dt_feature_and_cfc_timespans(
    mode, use_actual_timespans, dt_feature
):
    factory = SpyFactory()
    model = BidirectionalCfCEncoder(31, 3, cfc_factory=factory)
    features, dt = make_inputs(batch=1)
    original_features = features.clone()
    original_dt = dt.clone()
    nominal = 0.125

    model(features, dt, mode=mode, nominal_dt_s=nominal)

    expected_features = features.clone()
    if dt_feature == "nominal":
        expected_features[..., 12] = nominal
    elif dt_feature == "zero":
        expected_features[..., 12] = 0
    expected_forward_dt = dt if use_actual_timespans else torch.full_like(dt, nominal)
    expected_reverse_dt = (
        reverse_aligned_dt(dt)
        if use_actual_timespans
        else torch.full_like(dt, nominal)
    )
    forward_features, forward_timespans = factory.modules[0].calls[0]
    reverse_features, reverse_timespans = factory.modules[1].calls[0]
    torch.testing.assert_close(forward_features, expected_features)
    torch.testing.assert_close(reverse_features, expected_features.flip(1))
    torch.testing.assert_close(forward_timespans, expected_forward_dt)
    torch.testing.assert_close(reverse_timespans, expected_reverse_dt)
    # Age, slope, and every other shared 31-D feature are deliberately retained.
    torch.testing.assert_close(forward_features[..., :12], features[..., :12])
    torch.testing.assert_close(forward_features[..., 13:], features[..., 13:])
    torch.testing.assert_close(features, original_features)
    torch.testing.assert_close(dt, original_dt)


def test_reverse_alignment_handles_a_single_timestep_and_encoder_shape():
    factory = SpyFactory()
    model = BidirectionalCfCEncoder(31, 2, cfc_factory=factory)
    features = torch.randn(2, 1, 31)
    dt = torch.tensor([[0.01], [0.03]])

    torch.testing.assert_close(reverse_aligned_dt(dt), dt)
    output = model(features, dt, mode="actual")

    assert output.shape == (2, 1, 4)
    torch.testing.assert_close(factory.modules[1].calls[0][1], dt)


def test_reverse_representation_is_flipped_back_before_concatenation():
    def time_codes(module, features, timespans):
        del timespans
        base = 100 * module.direction
        codes = torch.arange(features.shape[1], device=features.device) + base
        return codes.to(features.dtype)[None, :, None].expand(
            features.shape[0], -1, module.hidden_size
        )

    factory = SpyFactory(time_codes)
    model = BidirectionalCfCEncoder(31, 2, cfc_factory=factory)
    features, dt = make_inputs(batch=1, time=4)

    output = model(features, dt, mode="actual")

    expected_forward = torch.tensor([0, 1, 2, 3], dtype=features.dtype)
    expected_reverse = torch.tensor([103, 102, 101, 100], dtype=features.dtype)
    torch.testing.assert_close(output[0, :, 0], expected_forward)
    torch.testing.assert_close(output[0, :, 2], expected_reverse)


def test_reverse_alignment_matches_declared_exact_values_and_validation_v2():
    dt = torch.tensor([[0.01, 0.03, 0.02, 0.09], [0.4, 0.5, 0.8, 1.3]])
    expected = torch.tensor([[0.09, 0.09, 0.02, 0.03], [1.3, 1.3, 0.8, 0.5]])

    actual = reverse_aligned_dt(dt)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, reverse_aligned_dt_v2(dt))


@pytest.mark.parametrize(
    ("dt", "error", "message"),
    (
        (None, TypeError, "tensor"),
        ([[0.1]], TypeError, "tensor"),
        (torch.ones(2), ValueError, "2-D"),
        (torch.ones(1, 2, 1), ValueError, "2-D"),
        (torch.empty(0, 2), ValueError, "batch"),
        (torch.empty(1, 0), ValueError, "time"),
        (torch.ones(1, 2, dtype=torch.int64), TypeError, "floating"),
        (torch.tensor([[0.1, 0.0]]), ValueError, "positive"),
        (torch.tensor([[0.1, -0.2]]), ValueError, "positive"),
        (torch.tensor([[0.1, math.nan]]), ValueError, "finite"),
        (torch.tensor([[0.1, math.inf]]), ValueError, "finite"),
    ),
)
def test_reverse_alignment_rejects_malformed_dt(dt, error, message):
    with pytest.raises(error, match=message):
        reverse_aligned_dt(dt)


@pytest.mark.parametrize("input_size", (True, 12, 0, -1, 13.5, "31"))
def test_constructor_rejects_invalid_input_size(input_size):
    with pytest.raises((TypeError, ValueError), match="input_size"):
        BidirectionalCfCEncoder(input_size, 4, cfc_factory=SpyFactory())


@pytest.mark.parametrize("hidden_size", (True, 0, -1, 4.5, "4"))
def test_constructor_rejects_invalid_hidden_size(hidden_size):
    with pytest.raises((TypeError, ValueError), match="hidden_size"):
        BidirectionalCfCEncoder(31, hidden_size, cfc_factory=SpyFactory())


@pytest.mark.parametrize("factory", (False, 1, "factory", object()))
def test_constructor_rejects_noncallable_factory(factory):
    with pytest.raises(TypeError, match="cfc_factory"):
        BidirectionalCfCEncoder(31, 4, cfc_factory=factory)


@pytest.mark.parametrize("bad_direction", (0, 1))
def test_constructor_rejects_non_module_factory_results(bad_direction):
    modules = [SpyCfC(4, 0), SpyCfC(4, 1)]
    modules[bad_direction] = object()
    calls = iter(modules)

    with pytest.raises(TypeError, match="nn.Module"):
        BidirectionalCfCEncoder(31, 4, cfc_factory=lambda *args, **kwargs: next(calls))


@pytest.mark.parametrize(
    ("features", "error", "message"),
    (
        (None, TypeError, "tensor"),
        ([[[1.0] * 31]], TypeError, "tensor"),
        (torch.ones(1, 2, 31, dtype=torch.int64), TypeError, "floating"),
        (torch.ones(2, 31), ValueError, "3-D"),
        (torch.ones(1, 2, 31, 1), ValueError, "3-D"),
        (torch.empty(0, 2, 31), ValueError, "batch"),
        (torch.empty(1, 0, 31), ValueError, "time"),
        (torch.ones(1, 2, 30), ValueError, "final dimension"),
        (torch.full((1, 2, 31), math.nan), ValueError, "finite"),
        (torch.full((1, 2, 31), math.inf), ValueError, "finite"),
    ),
)
def test_forward_rejects_malformed_features(features, error, message):
    model = BidirectionalCfCEncoder(31, 4, cfc_factory=SpyFactory())
    dt = torch.ones(1, 2) * 0.01

    with pytest.raises(error, match=message):
        model(features, dt, mode="actual")


@pytest.mark.parametrize(
    ("dt", "error", "message"),
    (
        (None, TypeError, "tensor"),
        ([[0.1, 0.2]], TypeError, "tensor"),
        (torch.ones(2), ValueError, "shape"),
        (torch.ones(1, 3), ValueError, "shape"),
        (torch.ones(1, 2, dtype=torch.int64), TypeError, "dtype"),
        (torch.tensor([[0.1, 0.0]]), ValueError, "positive"),
        (torch.tensor([[0.1, -0.1]]), ValueError, "positive"),
        (torch.tensor([[0.1, math.nan]]), ValueError, "finite"),
        (torch.tensor([[0.1, math.inf]]), ValueError, "finite"),
    ),
)
def test_forward_rejects_malformed_dt(dt, error, message):
    model = BidirectionalCfCEncoder(31, 4, cfc_factory=SpyFactory())
    features = torch.ones(1, 2, 31)

    with pytest.raises(error, match=message):
        model(features, dt, mode="actual")


def test_forward_rejects_dt_on_a_different_device_when_cuda_is_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = BidirectionalCfCEncoder(31, 4, cfc_factory=SpyFactory()).cuda()

    with pytest.raises(ValueError, match="device"):
        model(
            torch.ones(1, 2, 31, device="cuda"),
            torch.ones(1, 2, device="cpu"),
            mode="actual",
        )


@pytest.mark.parametrize(
    "mode", (None, "Actual", "constant ", "feature_only", 1, [])
)
def test_forward_rejects_undeclared_modes(mode):
    model = BidirectionalCfCEncoder(31, 4, cfc_factory=SpyFactory())
    features, dt = make_inputs(batch=1)

    with pytest.raises(ValueError, match="mode"):
        model(features, dt, mode=mode)


@pytest.mark.parametrize(
    "nominal_dt_s", (True, "0.01", 0, -0.1, math.nan, math.inf, -math.inf)
)
@pytest.mark.parametrize("mode", ("actual", "constant", "dt_feature_only", "no_dt"))
def test_all_modes_validate_nominal_dt(nominal_dt_s, mode):
    model = BidirectionalCfCEncoder(31, 4, cfc_factory=SpyFactory())
    features, dt = make_inputs(batch=1)

    with pytest.raises((TypeError, ValueError), match="nominal_dt_s"):
        model(features, dt, mode=mode, nominal_dt_s=nominal_dt_s)


@pytest.mark.parametrize(
    ("bad_result", "error", "message"),
    (
        ((torch.ones(1), None, None), ValueError, "tuple.*2"),
        (("not a tensor", None), TypeError, "tensor"),
        ("not a tensor", TypeError, "tensor"),
        (torch.ones(1, 3, 3), ValueError, "shape"),
        (torch.full((1, 4, 3), math.nan), ValueError, "finite"),
        (torch.full((1, 4, 3), math.inf), ValueError, "finite"),
    ),
)
def test_malformed_cfc_outputs_raise_deterministic_errors(bad_result, error, message):
    def result_factory(module, features, timespans):
        del module, features, timespans
        return bad_result

    model = BidirectionalCfCEncoder(31, 3, cfc_factory=SpyFactory(result_factory))
    features, dt = make_inputs(batch=1)

    with pytest.raises(error, match=message):
        model(features, dt, mode="actual")


def test_two_item_cfc_tuple_uses_sequence_output():
    def tuple_factory(module, features, timespans):
        del timespans
        output = features.new_ones((*features.shape[:2], module.hidden_size))
        return output, features.new_zeros(features.shape[0], module.hidden_size)

    model = BidirectionalCfCEncoder(31, 3, cfc_factory=SpyFactory(tuple_factory))
    features, dt = make_inputs(batch=1)

    output = model(features, dt, mode="actual")

    torch.testing.assert_close(output, torch.ones_like(output))


def test_default_factory_expands_scalar_timespans_across_hidden_units(monkeypatch):
    import ncps.torch

    factory = SpyFactory()
    monkeypatch.setattr(ncps.torch, "CfC", factory)
    model = BidirectionalCfCEncoder(31, 3)
    features, dt = make_inputs(batch=1)

    model(features, dt, mode="actual")

    forward_timespans = factory.modules[0].calls[0][1]
    reverse_timespans = factory.modules[1].calls[0][1]
    assert forward_timespans.shape == (1, 4, 3)
    assert reverse_timespans.shape == (1, 4, 3)
    torch.testing.assert_close(forward_timespans, dt.unsqueeze(-1).expand(-1, -1, 3))
    expected_reverse = reverse_aligned_dt(dt).unsqueeze(-1).expand(-1, -1, 3)
    torch.testing.assert_close(reverse_timespans, expected_reverse)


def test_default_ncps_cfc_smoke_has_finite_nonzero_input_and_parameter_gradients():
    torch.manual_seed(7)
    model = BidirectionalCfCEncoder(31, 4)
    features = torch.randn(2, 3, 31, requires_grad=True)
    dt = torch.tensor([[0.01, 0.02, 0.03], [0.015, 0.025, 0.04]])

    output = model(features, dt, mode="actual")
    weights = torch.linspace(0.2, 1.1, output.numel()).reshape_as(output)
    (output * weights).sum().backward()

    assert output.shape == (2, 3, 8)
    assert torch.isfinite(output).all()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0
    parameter_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert parameter_gradients
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)
    assert sum(gradient.abs().sum() for gradient in parameter_gradients) > 0
