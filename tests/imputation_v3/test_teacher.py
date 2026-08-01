import math

import pytest
import torch

from imputation_v3.models.teacher import OfflineTeacher, TeacherOutput
from imputation_v3.objectives.reconstruction import channel_balanced_missing_mse


def _inputs(*, batch=2, time=5, input_size=13, dtype=torch.float32):
    torch.manual_seed(123)
    features = torch.randn(batch, time, input_size, dtype=dtype)
    dt = torch.full((batch, time), 0.02, dtype=dtype)
    observed = torch.randn(batch, time, 6, dtype=dtype)
    mask = torch.tensor(
        [[[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]]], dtype=torch.bool
    ).repeat(batch, (time + 1) // 2, 1)[:, :time]
    baseline = torch.randn(batch, time, 6, dtype=dtype)
    return features, dt, observed, mask, baseline


def _teacher(**kwargs):
    options = dict(
        input_size=13,
        cfc_hidden=4,
        tcn_width=5,
        tcn_dilations=(1, 2),
    )
    options.update(kwargs)
    return OfflineTeacher(**options)


def test_teacher_shapes_components_fusion_order_and_exact_completion():
    model = _teacher().eval()
    inputs = _inputs()
    captured = {}
    handles = [
        model.cfc_encoder.register_forward_hook(
            lambda _module, _args, output: captured.__setitem__("cfc", output)
        ),
        model.tcn_encoder.register_forward_hook(
            lambda _module, _args, output: captured.__setitem__("tcn", output)
        ),
        model.trunk[0].register_forward_pre_hook(
            lambda _module, args: captured.__setitem__("fusion", args[0])
        ),
    ]
    try:
        output = model(*inputs)
    finally:
        for handle in handles:
            handle.remove()

    assert isinstance(output, TeacherOutput)
    assert (
        output.raw.shape
        == output.completed.shape
        == output.residual.shape
        == (2, 5, 6)
    )
    assert output.latent.shape == (2, 5, 48)
    torch.testing.assert_close(
        captured["fusion"],
        torch.cat((captured["cfc"], captured["tcn"], inputs[4]), dim=-1),
    )
    assert model.trunk[0].in_features == 2 * 4 + 5 + 6
    torch.testing.assert_close(
        output.completed[inputs[3]], inputs[2][inputs[3]], rtol=0, atol=0
    )
    torch.testing.assert_close(output.completed[~inputs[3]], output.raw[~inputs[3]])


@pytest.mark.parametrize("mode", ("residual", "raw"))
def test_output_parameterization_keeps_baseline_in_fusion(mode):
    model = _teacher(residual_mode=mode).eval()
    inputs = _inputs()
    output = model(*inputs)

    expected = inputs[4] + output.residual if mode == "residual" else output.residual
    torch.testing.assert_close(output.raw, expected)
    assert model.trunk[0].in_features == 2 * 4 + 5 + 6


def test_raw_mode_is_direct_output_with_baseline_retained_in_fusion():
    model = _teacher(residual_mode="raw").eval()
    inputs = _inputs()
    captured = []
    handle = model.trunk[0].register_forward_pre_hook(
        lambda _module, args: captured.append(args[0])
    )
    try:
        output = model(*inputs)
    finally:
        handle.remove()

    torch.testing.assert_close(output.raw, output.residual)
    torch.testing.assert_close(captured[0][..., -6:], inputs[4])


def test_gyro_then_accelerometer_head_order_and_separate_parameters():
    model = _teacher().eval()
    with torch.no_grad():
        model.gyro_head.weight.zero_()
        model.gyro_head.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
        model.acc_head.weight.zero_()
        model.acc_head.bias.copy_(torch.tensor([4.0, 5.0, 6.0]))
    output = model(*_inputs())

    expected = torch.tensor([1, 2, 3, 4, 5, 6], dtype=output.residual.dtype)
    torch.testing.assert_close(output.residual, expected.expand_as(output.residual))
    assert model.gyro_head.weight is not model.acc_head.weight
    assert model.gyro_head.bias is not model.acc_head.bias


def test_default_time_mode_and_explicit_override_are_delegated():
    inputs = _inputs()
    model = _teacher(time_mode="no_dt").eval()
    seen = []
    handle = model.cfc_encoder.register_forward_pre_hook(
        lambda _module, _args, kwargs: seen.append(kwargs["mode"]), with_kwargs=True
    )
    try:
        model(*inputs)
        model(*inputs, time_mode="constant", nominal_dt_s=0.03)
    finally:
        handle.remove()
    assert seen == ["no_dt", "constant"]

    with pytest.raises(ValueError, match="mode"):
        model(*inputs, time_mode="invalid")


@pytest.mark.parametrize("field", ("residual_mode", "time_mode"))
@pytest.mark.parametrize("value", (None, 1, "invalid"))
def test_constructor_rejects_invalid_modes(field, value):
    with pytest.raises((TypeError, ValueError), match="mode"):
        _teacher(**{field: value})


@pytest.mark.parametrize(
    ("index", "replacement", "error", "message"),
    (
        (0, None, TypeError, "features"),
        (1, None, TypeError, "dt"),
        (2, None, TypeError, "observed"),
        (3, None, TypeError, "mask"),
        (4, None, TypeError, "baseline"),
        (0, torch.ones(2, 5, 13, dtype=torch.int64), TypeError, "floating"),
        (1, torch.ones(2, 5, dtype=torch.int64), TypeError, "floating"),
        (2, torch.ones(2, 5, 6, dtype=torch.int64), TypeError, "floating"),
        (4, torch.ones(2, 5, 6, dtype=torch.int64), TypeError, "floating"),
        (0, torch.ones(2, 5, 12), ValueError, "input_size"),
        (1, torch.ones(2, 4), ValueError, "shape"),
        (2, torch.ones(2, 5, 5), ValueError, "shape"),
        (3, torch.ones(2, 5, 5), ValueError, "shape"),
        (4, torch.ones(2, 5, 5), ValueError, "shape"),
        (0, torch.empty(0, 5, 13), ValueError, "nonempty"),
        (0, torch.empty(2, 0, 13), ValueError, "nonempty"),
        (3, torch.full((2, 5, 6), 0.5), ValueError, "binary|0 or 1"),
        (3, torch.ones(2, 5, 6, dtype=torch.int64), TypeError, "bool|dtype"),
    ),
)
def test_forward_rejects_malformed_inputs(index, replacement, error, message):
    inputs = list(_inputs())
    inputs[index] = replacement
    with pytest.raises(error, match=message):
        _teacher().eval()(*inputs)


@pytest.mark.parametrize("index", (1, 2, 4))
def test_forward_rejects_mixed_floating_dtypes(index):
    inputs = list(_inputs())
    inputs[index] = inputs[index].double()
    with pytest.raises(TypeError, match="dtype"):
        _teacher().eval()(*inputs)


@pytest.mark.parametrize("index", (0, 1, 4))
def test_forward_rejects_nonfinite_required_inputs(index):
    inputs = list(_inputs())
    inputs[index] = inputs[index].clone()
    inputs[index].reshape(-1)[0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        _teacher().eval()(*inputs)


def test_hidden_observed_placeholders_may_be_nonfinite_but_observed_values_may_not():
    inputs = list(_inputs())
    inputs[2] = inputs[2].clone()
    inputs[2][~inputs[3]] = math.nan
    result = _teacher().eval()(*inputs)
    assert torch.isfinite(result.completed).all()

    inputs[2][inputs[3]] = math.inf
    with pytest.raises(ValueError, match="observed.*finite|finite.*observed"):
        _teacher().eval()(*inputs)


def test_forward_does_not_mutate_caller_tensors():
    inputs = _inputs()
    originals = tuple(value.clone() for value in inputs)
    _teacher().eval()(*inputs)
    for actual, expected in zip(inputs, originals):
        torch.testing.assert_close(actual, expected, equal_nan=True)


def test_completion_preserves_observed_negative_zero_bit_exactly():
    inputs = list(_inputs())
    inputs[2] = inputs[2].clone()
    observed_index = inputs[3].nonzero()[0]
    index = tuple(observed_index.tolist())
    inputs[2][index] = -0.0

    output = _teacher().eval()(*inputs)

    assert output.completed[index].item() == 0.0
    assert torch.signbit(output.completed[index]).item()


def test_teacher_output_fields_keep_live_backward_connectivity():
    model = _teacher().eval()
    output = model(*_inputs())
    fields = (output.raw, output.completed, output.residual, output.latent)
    for field in fields:
        assert field.grad_fn is not None
        field.retain_grad()

    sum(field.sum() for field in fields).backward()

    assert all(field.grad is not None for field in fields)
    assert model.trunk[0].weight.grad is not None
    assert torch.count_nonzero(model.trunk[0].weight.grad).item() > 0


def test_real_teacher_backward_reaches_every_named_branch():
    torch.manual_seed(8)
    model = _teacher().eval()
    features, dt, observed, _mask, baseline = _inputs(batch=2, time=7)
    mask = torch.zeros_like(observed, dtype=torch.bool)
    target = torch.randn_like(observed)

    output = model(features, dt, observed, mask, baseline)
    loss = channel_balanced_missing_mse(output.raw, target, mask)
    loss.backward()

    parameter_groups = {
        "forward CfC": tuple(model.cfc_encoder.forward_cfc.parameters()),
        "reverse CfC": tuple(model.cfc_encoder.reverse_cfc.parameters()),
        "TCN projection": tuple(model.tcn_encoder.projection.parameters()),
        "trunk": tuple(model.trunk.parameters()),
        "gyro head": tuple(model.gyro_head.parameters()),
        "accelerometer head": tuple(model.acc_head.parameters()),
    }
    for name, parameters in parameter_groups.items():
        gradients = [parameter.grad for parameter in parameters]
        assert all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        ), name
        assert any(
            torch.count_nonzero(gradient).item() > 0 for gradient in gradients
        ), name

    assert all(
        torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.tcn_encoder.projection.parameters()
    )

    for block_index, block in enumerate(model.tcn_encoder.blocks):
        for name, parameters in (
            ("depthwise", tuple(block.depthwise.parameters())),
            ("pointwise", tuple(block.pointwise.parameters())),
            ("normalization", tuple(block.normalization.parameters())),
        ):
            gradients = [parameter.grad for parameter in parameters]
            label = f"TCN block {block_index} {name}"
            assert all(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in gradients
            ), label
            assert all(
                torch.count_nonzero(gradient).item() > 0 for gradient in gradients
            ), label
