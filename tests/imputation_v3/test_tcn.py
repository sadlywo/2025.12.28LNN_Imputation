import math
from fractions import Fraction

import pytest
import torch

from imputation_v3.models.tcn import DepthwiseResidualBlock, SymmetricTCNEncoder


def test_encoder_preserves_time_axis_and_produces_finite_features():
    model = SymmetricTCNEncoder(
        31, width=12, dilations=(1, 2, 4), dropout=0.0
    )

    output = model(torch.randn(2, 33, 31))

    assert output.shape == (2, 33, 12)
    assert torch.isfinite(output).all()


def test_encoder_rejects_even_kernel_size():
    with pytest.raises(ValueError, match="odd"):
        SymmetricTCNEncoder(3, width=4, dilations=(1,), kernel_size=2)


def test_blocks_use_depthwise_then_pointwise_temporal_convolutions():
    width = 7
    model = SymmetricTCNEncoder(
        3, width=width, dilations=(4, 1, 2), kernel_size=5
    )

    assert [block.dilation for block in model.blocks] == [4, 1, 2]
    for block in model.blocks:
        assert block.depthwise.in_channels == width
        assert block.depthwise.out_channels == width
        assert block.depthwise.groups == width
        assert block.depthwise.kernel_size == (5,)
        assert block.pointwise.in_channels == width
        assert block.pointwise.out_channels == width
        assert block.pointwise.kernel_size == (1,)
        assert block.pointwise.groups == 1


def test_receptive_field_matches_stacked_symmetric_dilations_and_is_read_only():
    model = SymmetricTCNEncoder(
        2, width=4, dilations=(1, 2, 4), kernel_size=3
    )

    assert model.receptive_field == 15
    with pytest.raises(AttributeError):
        model.receptive_field = 99


def test_controlled_block_accesses_both_past_and_future_neighbors():
    model = SymmetricTCNEncoder(
        2, width=2, dilations=(1,), kernel_size=3, dropout=0.0
    )
    with torch.no_grad():
        model.projection.weight.copy_(torch.eye(2))
        model.projection.bias.zero_()
        block = model.blocks[0]
        block.depthwise.weight.zero_()
        block.depthwise.bias.zero_()
        block.depthwise.weight[0, 0, 0] = 1.0
        block.depthwise.weight[0, 0, 2] = 1.0
        block.pointwise.weight.copy_(torch.eye(2).unsqueeze(-1))
        block.pointwise.bias.zero_()

    baseline = model(torch.zeros(1, 5, 2))
    impulse = torch.zeros(1, 5, 2)
    impulse[0, 2, 0] = 1.0
    response = model(impulse) - baseline

    assert response[0, 1].abs().sum() > 0
    assert response[0, 3].abs().sum() > 0
    torch.testing.assert_close(response[0, 1], response[0, 3])


def test_backward_reaches_projection_depthwise_and_pointwise_weights():
    torch.manual_seed(4)
    model = SymmetricTCNEncoder(
        5, width=6, dilations=(1, 2), kernel_size=3, dropout=0.0
    )
    output = model(torch.randn(2, 9, 5))
    weights = torch.linspace(0.25, 1.25, output.numel()).reshape_as(output)

    (output * weights).sum().backward()

    parameters = [model.projection.weight]
    parameters.extend(block.depthwise.weight for block in model.blocks)
    parameters.extend(block.pointwise.weight for block in model.blocks)
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "constructor",
    (
        lambda: SymmetricTCNEncoder(3, width=1, dilations=(1,)),
        lambda: DepthwiseResidualBlock(
            width=1, kernel_size=3, dilation=1, dropout=0.0
        ),
    ),
    ids=("encoder", "block"),
)
def test_public_tcn_constructors_reject_width_one(constructor):
    with pytest.raises(ValueError, match="width.*at least 2"):
        constructor()


def test_width_two_remains_input_dependent_with_finite_input_gradients():
    model = SymmetricTCNEncoder(
        2, width=2, dilations=(1,), kernel_size=3, dropout=0.0
    )
    direct_block = DepthwiseResidualBlock(
        width=2, kernel_size=3, dilation=1, dropout=0.0
    )
    assert direct_block.width == 2

    with torch.no_grad():
        model.projection.weight.copy_(torch.eye(2))
        model.projection.bias.zero_()
        block = model.blocks[0]
        block.depthwise.weight.zero_()
        block.depthwise.bias.zero_()
        block.pointwise.weight.zero_()
        block.pointwise.bias.zero_()

    features = torch.tensor([[[0.25, -0.25]]], requires_grad=True)
    output = model(features)
    output[0, 0, 0].backward()

    assert output[0, 0, 0] != output[0, 0, 1]
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0


@pytest.mark.parametrize("field", ("input_size", "width"))
@pytest.mark.parametrize("value", (True, 0, -1, 1.5, "3"))
def test_encoder_rejects_non_positive_or_non_integer_dimensions(field, value):
    kwargs = dict(input_size=3, width=4, dilations=(1,))
    kwargs[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        SymmetricTCNEncoder(**kwargs)


@pytest.mark.parametrize(
    "dilations",
    (None, (), [], "12", (True,), (0,), (-1,), (1.5,), ("1",)),
)
def test_encoder_rejects_invalid_dilation_sequences(dilations):
    with pytest.raises((TypeError, ValueError), match="dilations"):
        SymmetricTCNEncoder(3, width=4, dilations=dilations)


@pytest.mark.parametrize("kernel_size", (True, 0, -1, 2, 2.5, "3"))
def test_encoder_rejects_invalid_kernel_sizes(kernel_size):
    with pytest.raises((TypeError, ValueError), match="kernel_size|odd"):
        SymmetricTCNEncoder(
            3, width=4, dilations=(1,), kernel_size=kernel_size
        )


@pytest.mark.parametrize(
    "dropout", (True, "0.1", -0.1, 1.0, math.nan, math.inf, -math.inf)
)
def test_encoder_rejects_invalid_dropout(dropout):
    with pytest.raises((TypeError, ValueError), match="dropout"):
        SymmetricTCNEncoder(3, width=4, dilations=(1,), dropout=dropout)


def test_dropout_overflow_is_reported_as_a_deterministic_value_error():
    too_large = Fraction(10**10000, 1)

    with pytest.raises(ValueError, match="dropout"):
        SymmetricTCNEncoder(3, width=4, dilations=(1,), dropout=too_large)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"width": True}, "width"),
        ({"width": 0}, "width"),
        ({"dilation": True}, "dilation"),
        ({"dilation": 0}, "dilation"),
        ({"kernel_size": 2}, "odd"),
        ({"dropout": math.nan}, "dropout"),
    ),
)
def test_block_validates_its_constructor_arguments(kwargs, message):
    valid = dict(width=4, kernel_size=3, dilation=1, dropout=0.0)
    valid.update(kwargs)

    with pytest.raises((TypeError, ValueError), match=message):
        DepthwiseResidualBlock(**valid)


@pytest.mark.parametrize(
    ("features", "error", "message"),
    (
        (None, TypeError, "tensor"),
        ([[[1.0, 2.0, 3.0]]], TypeError, "tensor"),
        (torch.ones(1, 2, 3, dtype=torch.int64), TypeError, "floating"),
        (torch.ones(2, 3), ValueError, "3-D"),
        (torch.ones(1, 2, 3, 1), ValueError, "3-D"),
        (torch.empty(0, 2, 3), ValueError, "batch"),
        (torch.empty(1, 0, 3), ValueError, "time"),
        (torch.ones(1, 2, 4), ValueError, "final dimension"),
        (
            torch.tensor([[[1.0, math.nan, 3.0]]]),
            ValueError,
            "finite",
        ),
        (
            torch.tensor([[[1.0, math.inf, 3.0]]]),
            ValueError,
            "finite",
        ),
    ),
)
def test_forward_rejects_malformed_features(features, error, message):
    model = SymmetricTCNEncoder(3, width=4, dilations=(1,))

    with pytest.raises(error, match=message):
        model(features)


def test_eval_mode_is_deterministic_with_dropout_enabled():
    model = SymmetricTCNEncoder(
        3, width=5, dilations=(1, 2), dropout=0.5
    ).eval()
    features = torch.randn(2, 11, 3)

    first = model(features)
    second = model(features)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("length", (1, 2, 10))
def test_odd_kernels_preserve_short_and_non_multiple_sequence_lengths(length):
    model = SymmetricTCNEncoder(
        3, width=4, dilations=(1, 3), kernel_size=5, dropout=0.0
    )

    output = model(torch.randn(2, length, 3))

    assert output.shape == (2, length, 4)


def test_module_dtype_follows_normal_pytorch_semantics_without_silent_casting():
    model = SymmetricTCNEncoder(
        3, width=4, dilations=(1,), dropout=0.0
    ).double()
    output = model(torch.randn(2, 5, 3, dtype=torch.float64))
    assert output.dtype == torch.float64

    with pytest.raises(RuntimeError):
        model(torch.randn(2, 5, 3, dtype=torch.float32))
