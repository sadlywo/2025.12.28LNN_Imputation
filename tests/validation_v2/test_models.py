import pytest
import torch
from torch import nn

from validation_v2.models import (
    BiLSTMImputer,
    BidirectionalCfC,
    HybridImputer,
    complete_signal,
    fuse,
)
from validation_v2.models.baselines import (
    equal_average,
    fixed_gate,
    linear_interpolation,
    locf,
    single_branch,
)


def test_complete_signal_preserves_observed_and_ignores_hidden_nan():
    observed = torch.tensor([[[1.0], [torch.nan], [3.0]]], dtype=torch.float64)
    mask = torch.tensor([[[1.0], [0.0], [1.0]]], dtype=torch.float64)
    prediction = torch.tensor([[[9.0], [2.0], [9.0]]], dtype=torch.float64)

    completed = complete_signal(observed, mask, prediction)

    torch.testing.assert_close(
        completed, torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float64)
    )
    assert completed.dtype == observed.dtype
    assert completed.device == observed.device


@pytest.mark.parametrize(
    ("observed", "mask", "prediction"),
    [
        (torch.ones(1, 2, 1), torch.ones(1, 3, 1), torch.ones(1, 2, 1)),
        (torch.ones(1, 2, 1), torch.full((1, 2, 1), 0.5), torch.ones(1, 2, 1)),
        (torch.ones(1, 2, 1), torch.ones(1, 2, 1), torch.ones(1, 2, 1, dtype=torch.float64)),
    ],
)
def test_complete_signal_rejects_incompatible_shape_dtype_and_nonbinary_mask(
    observed, mask, prediction
):
    with pytest.raises((TypeError, ValueError)):
        complete_signal(observed, mask, prediction)


@pytest.mark.parametrize("gate,expected", [(0.0, 4.0), (0.5, 3.0), (1.0, 2.0)])
def test_fuse_declares_gate_as_lnn_weight(gate, expected):
    result = fuse(torch.tensor(2.0), torch.tensor(4.0), torch.tensor(gate))
    assert result.item() == pytest.approx(expected)


def test_bidirectional_cfc_uses_keyword_timespans_aligned_to_each_direction():
    spies = []

    class SpyCfC(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.inputs = None
            self.timespans = None

        def forward(self, inputs, *, timespans):
            self.inputs = inputs.detach().clone()
            self.timespans = timespans.detach().clone()
            return inputs[..., : self.hidden_size], torch.zeros(1)

    def factory(input_size, hidden_size, **kwargs):
        spy = SpyCfC(hidden_size)
        spies.append(spy)
        return spy

    model = BidirectionalCfC(
        input_size=4, hidden_size=3, output_size=6, cfc_factory=factory
    )
    features = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    forward_dt = torch.tensor([[0.1, 0.2, 0.4], [0.3, 0.5, 0.7]])
    reverse_dt = torch.tensor([[0.9, 0.8, 0.6], [0.4, 0.2, 0.1]])

    output = model(features, forward_dt, reverse_dt)

    assert output.shape == (2, 3, 6)
    torch.testing.assert_close(spies[0].inputs, features)
    torch.testing.assert_close(spies[0].timespans, forward_dt)
    torch.testing.assert_close(spies[1].inputs, features.flip(1))
    torch.testing.assert_close(spies[1].timespans, reverse_dt)


def test_actual_ncps_bidirectional_cfc_cpu_forward_has_six_channels():
    model = BidirectionalCfC(input_size=4, hidden_size=5, output_size=6)
    features = torch.randn(2, 4, 4)
    forward_dt = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]])
    reverse_dt = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.5, 0.4, 0.3, 0.2]])

    output = model(features, forward_dt, reverse_dt)

    assert output.shape == (2, 4, 6)
    assert output.device.type == "cpu"
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "bad_dt",
    [
        torch.ones(2, 3),
        torch.tensor([[0.1, 0.2, 0.3, 0.0], [0.1, 0.2, 0.3, 0.4]]),
        torch.tensor([[0.1, 0.2, 0.3, torch.nan], [0.1, 0.2, 0.3, 0.4]]),
    ],
)
def test_bidirectional_cfc_rejects_bad_dt_shape_and_values(bad_dt):
    model = BidirectionalCfC(input_size=4, hidden_size=3)
    features = torch.randn(2, 4, 4)
    with pytest.raises(ValueError):
        model(features, bad_dt, torch.full((2, 4), 0.1))


def test_bilstm_uses_only_declared_features_and_outputs_six_channels():
    model = BiLSTMImputer(input_size=7, hidden_size=4)
    features = torch.randn(2, 5, 7)

    output = model(features)

    assert output.shape == (2, 5, 6)
    with pytest.raises(TypeError):
        model(features, torch.ones(2, 5))


def test_hybrid_components_define_sigmoid_gate_as_lnn_weight_and_complete():
    class LnnBranch(nn.Module):
        def forward(self, features, forward_dt, reverse_dt):
            return features.new_full((*features.shape[:2], 6), 2.0)

    class LstmBranch(nn.Module):
        def forward(self, features):
            return features.new_full((*features.shape[:2], 6), 4.0)

    class GateLogits(nn.Module):
        def forward(self, features):
            return features.new_zeros((*features.shape[:2], 6))

    model = HybridImputer(
        input_size=4,
        lnn_branch=LnnBranch(),
        lstm_branch=LstmBranch(),
        gate_network=GateLogits(),
    )
    features = torch.randn(1, 3, 4)
    forward_dt = torch.full((1, 3), 0.1)
    reverse_dt = torch.full((1, 3), 0.2)
    observed = torch.tensor(
        [[[1.0] * 6, [torch.nan] * 6, [3.0] * 6]]
    )
    mask = torch.tensor([[[1.0] * 6, [0.0] * 6, [1.0] * 6]])

    components = model.forward_components(
        features, forward_dt, reverse_dt, observed, mask
    )

    torch.testing.assert_close(components.lnn, torch.full((1, 3, 6), 2.0))
    torch.testing.assert_close(components.lstm, torch.full((1, 3, 6), 4.0))
    torch.testing.assert_close(components.gate, torch.full((1, 3, 6), 0.5))
    torch.testing.assert_close(components.raw, torch.full((1, 3, 6), 3.0))
    torch.testing.assert_close(
        components.completed,
        torch.tensor([[[1.0] * 6, [3.0] * 6, [3.0] * 6]]),
    )
    torch.testing.assert_close(
        model(features, forward_dt, reverse_dt, observed, mask), components.completed
    )


def test_locf_and_linear_interpolation_are_per_batch_per_channel_and_complete():
    observed = torch.tensor(
        [
            [[torch.nan, 10.0], [1.0, torch.nan], [torch.nan, torch.nan], [3.0, 16.0]],
            [[8.0, torch.nan], [torch.nan, 2.0], [torch.nan, 4.0], [14.0, torch.nan]],
        ]
    )
    mask = torch.isfinite(observed).to(observed.dtype)

    linear = linear_interpolation(observed, mask)
    carried = locf(observed, mask)

    torch.testing.assert_close(
        linear,
        torch.tensor(
            [
                [[1.0, 10.0], [1.0, 12.0], [2.0, 14.0], [3.0, 16.0]],
                [[8.0, 2.0], [10.0, 2.0], [12.0, 4.0], [14.0, 4.0]],
            ]
        ),
    )
    torch.testing.assert_close(
        carried,
        torch.tensor(
            [
                [[1.0, 10.0], [1.0, 10.0], [1.0, 10.0], [3.0, 16.0]],
                [[8.0, 2.0], [8.0, 2.0], [8.0, 4.0], [14.0, 4.0]],
            ]
        ),
    )
    hidden_changed = torch.where(mask.bool(), observed, torch.full_like(observed, 9999.0))
    torch.testing.assert_close(linear_interpolation(hidden_changed, mask), linear)
    torch.testing.assert_close(locf(hidden_changed, mask), carried)
    with pytest.raises(ValueError, match="no observed"):
        linear_interpolation(torch.full((1, 3, 1), torch.nan), torch.zeros(1, 3, 1))


@pytest.mark.parametrize("gate,missing_value", [(0.0, 4.0), (0.5, 3.0), (1.0, 2.0)])
def test_branch_diagnostics_and_fixed_gates_complete_without_hidden_targets(
    gate, missing_value
):
    target_a = torch.tensor([[[1.0], [20.0], [3.0]]])
    target_b = torch.tensor([[[1.0], [9999.0], [3.0]]])
    mask = torch.tensor([[[1.0], [0.0], [1.0]]])
    observed_a = target_a
    observed_b = target_b
    lnn = torch.full_like(target_a, 2.0)
    lstm = torch.full_like(target_a, 4.0)

    result_a = fixed_gate(observed_a, mask, lnn, lstm, gate)
    result_b = fixed_gate(observed_b, mask, lnn, lstm, gate)

    torch.testing.assert_close(result_a, result_b)
    assert result_a[0, 1, 0].item() == pytest.approx(missing_value)
    torch.testing.assert_close(single_branch(observed_a, mask, lnn), fixed_gate(observed_a, mask, lnn, lstm, 1.0))
    torch.testing.assert_close(equal_average(observed_a, mask, lnn, lstm), fixed_gate(observed_a, mask, lnn, lstm, 0.5))
