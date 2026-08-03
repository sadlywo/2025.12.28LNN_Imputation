import math
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

import imputation_v3.experiments.pypots as pypots_module
from imputation_v3.experiments.pypots import (
    SUPPORTED_PYPOTS_MODELS,
    PyPOTSAdapter,
    build_pypots_model,
    to_pypots_sets,
)

from imputation_v3.models.native_controls import (
    BiCfCControl,
    BiLSTMControl,
    CONTROL_CONDITIONS,
    FeatureMLPControl,
    NATIVE_CONDITIONS,
    TCNControl,
    TEACHER_CONDITION_MODES,
    count_parameters,
    teacher_condition_modes,
)


class FakeModel:
    def __init__(self):
        self.fit_arguments = None
        self.impute_arguments = None

    def fit(self, train_set, validation_set):
        self.fit_arguments = (train_set, validation_set)

    def impute(self, test_set, **kwargs):
        self.impute_arguments = (test_set, kwargs)
        result = np.full_like(test_set["X"], 7.0)
        result[np.isfinite(test_set["X"])] = -999.0
        return result


def test_native_controls_share_input_and_exact_completion_contract():
    torch.manual_seed(17)
    features = torch.randn(2, 7, 31)
    dt = torch.full((2, 7), 0.01)
    observed = torch.randn(2, 7, 6)
    mask = torch.ones_like(observed, dtype=torch.bool)
    mask[:, 2:6, 1:5] = False
    baseline = torch.randn_like(observed)

    models = (
        BiLSTMControl(31, 4),
        BiCfCControl(31, 4),
        TCNControl(31, 4, (1, 2)),
        FeatureMLPControl(31, 8),
    )
    for model in models:
        output = model(features, dt, observed, mask, baseline)
        assert output.raw.shape == observed.shape
        assert output.residual.shape == observed.shape
        assert output.latent.shape[:2] == observed.shape[:2]
        assert count_parameters(model) > 0
        torch.testing.assert_close(
            output.completed[mask], observed[mask], rtol=0, atol=0
        )
        torch.testing.assert_close(output.completed[~mask], output.raw[~mask])


def test_pypots_sets_hide_only_declared_missing_values_without_mutation_or_leak():
    target = np.arange(48, dtype=np.float32).reshape(1, 8, 6)
    mask = np.ones_like(target)
    mask[:, 2:5, 1:4] = 0
    alternate = target.copy()
    alternate[mask == 0] = 50_000
    train = SimpleNamespace(target=target, mask=mask)
    validation = SimpleNamespace(target=target + 1, mask=mask)
    originals = (target.copy(), mask.copy(), validation.target.copy())

    train_set, validation_set = to_pypots_sets(train, validation)
    alternate_set, _ = to_pypots_sets(
        SimpleNamespace(target=alternate, mask=mask), validation
    )

    assert train_set["X"].dtype == np.float32
    assert validation_set["X_ori"].dtype == np.float32
    assert np.array_equal(np.isnan(train_set["X"]), mask == 0)
    np.testing.assert_array_equal(train_set["X"], alternate_set["X"])
    np.testing.assert_array_equal(target, originals[0])
    np.testing.assert_array_equal(mask, originals[1])
    np.testing.assert_array_equal(validation.target, originals[2])
    assert not np.shares_memory(train_set["X"], target)
    assert not np.shares_memory(validation_set["X_ori"], validation.target)


def test_pypots_adapter_restores_observations_and_forwards_private_kwargs():
    x = np.arange(48, dtype=np.float32).reshape(1, 8, 6)
    x[:, 2:5, 1:4] = np.nan
    original = x.copy()
    model = FakeModel()
    adapter = PyPOTSAdapter(model, impute_kwargs={"n_sampling_times": 20})

    adapter.fit({"X": x}, {"X": x, "X_ori": np.nan_to_num(x)})
    result = adapter.impute({"X": x})

    assert model.fit_arguments is not None
    assert model.impute_arguments[1] == {"n_sampling_times": 20}
    np.testing.assert_array_equal(result[np.isfinite(x)], x[np.isfinite(x)])
    np.testing.assert_array_equal(result[np.isnan(x)], np.full(9, 7.0))
    np.testing.assert_array_equal(x, original)
    assert result.dtype == x.dtype
    assert np.isfinite(result).all()


def _native_inputs():
    torch.manual_seed(23)
    features = torch.randn(2, 5, 31)
    dt = torch.full((2, 5), 0.01)
    observed = torch.randn(2, 5, 6)
    mask = torch.ones_like(observed, dtype=torch.bool)
    mask[:, 1::2, 1:5] = False
    baseline = torch.randn_like(observed)
    return features, dt, observed, mask, baseline


def test_native_controls_ignore_hidden_observed_placeholders_and_do_not_mutate_inputs():
    models = (
        BiLSTMControl(31, 4),
        BiCfCControl(31, 4),
        TCNControl(31, 4, (1, 2)),
        FeatureMLPControl(31, 8),
    )
    for model in models:
        model.eval()
        inputs = _native_inputs()
        alternate = list(inputs)
        alternate[2] = alternate[2].clone()
        alternate[2][~alternate[3]] = math.nan
        originals = tuple(value.clone() for value in alternate)

        output = model(*inputs)
        alternate_output = model(*alternate)

        torch.testing.assert_close(output.raw, alternate_output.raw)
        torch.testing.assert_close(output.completed, alternate_output.completed)
        for actual, expected in zip(alternate, originals):
            torch.testing.assert_close(actual, expected, equal_nan=True)


@pytest.mark.parametrize(
    ("index", "replacement", "error", "message"),
    (
        (0, None, TypeError, "features"),
        (1, None, TypeError, "dt"),
        (2, None, TypeError, "observed"),
        (3, None, TypeError, "mask"),
        (4, None, TypeError, "baseline"),
        (0, torch.ones(2, 5, 31, dtype=torch.int64), TypeError, "floating"),
        (1, torch.ones(2, 5, dtype=torch.int64), TypeError, "floating"),
        (0, torch.ones(2, 5, 30), ValueError, "input_size"),
        (1, torch.ones(2, 4), ValueError, "shape"),
        (2, torch.ones(2, 5, 5), ValueError, "shape"),
        (3, torch.full((2, 5, 6), 0.5), ValueError, "binary"),
        (3, torch.ones(2, 5, 6, dtype=torch.int64), TypeError, "bool|dtype"),
    ),
)
def test_native_control_rejects_malformed_shared_inputs(
    index, replacement, error, message
):
    inputs = list(_native_inputs())
    inputs[index] = replacement
    with pytest.raises(error, match=message):
        FeatureMLPControl(31, 8)(*inputs)


@pytest.mark.parametrize("index", (0, 1, 4))
def test_native_control_rejects_nonfinite_required_inputs(index):
    inputs = list(_native_inputs())
    inputs[index] = inputs[index].clone()
    inputs[index].reshape(-1)[0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        FeatureMLPControl(31, 8)(*inputs)


def test_native_condition_names_and_teacher_modes_are_frozen_for_task10_factory():
    assert CONTROL_CONDITIONS == ("bilstm", "bilnn", "tcn", "feature_mlp")
    assert tuple(TEACHER_CONDITION_MODES) == (
        "teacher_actual_residual",
        "teacher_constant_residual",
        "teacher_dt_feature_only_residual",
        "teacher_no_dt_residual",
        "teacher_actual_raw",
    )
    assert NATIVE_CONDITIONS == (*CONTROL_CONDITIONS, *TEACHER_CONDITION_MODES)
    assert teacher_condition_modes("teacher_no_dt_residual") == (
        "no_dt",
        "residual",
    )
    with pytest.raises(ValueError, match="unsupported teacher condition"):
        teacher_condition_modes("teacher")


@pytest.mark.parametrize(
    "mutator",
    (
        lambda target, mask: (target[:, :, :5], mask[:, :, :5]),
        lambda target, mask: (target.astype(np.int64), mask),
        lambda target, mask: (target, mask[:, :-1]),
        lambda target, mask: (target, np.full_like(mask, 0.5)),
    ),
)
def test_pypots_sets_reject_malformed_targets_and_masks(mutator):
    target = np.ones((1, 4, 6), dtype=np.float32)
    mask = np.ones_like(target)
    bad_target, bad_mask = mutator(target, mask)
    with pytest.raises((TypeError, ValueError)):
        to_pypots_sets(
            SimpleNamespace(target=bad_target, mask=bad_mask),
            SimpleNamespace(target=target, mask=mask),
        )


def test_pypots_sets_reject_nonfinite_observed_and_incomplete_validation_truth():
    target = np.ones((1, 4, 6), dtype=np.float32)
    mask = np.ones_like(target)
    bad_observed = target.copy()
    bad_observed[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="observed.*finite"):
        to_pypots_sets(
            SimpleNamespace(target=bad_observed, mask=mask),
            SimpleNamespace(target=target, mask=mask),
        )

    incomplete_truth = target.copy()
    incomplete_mask = mask.copy()
    incomplete_truth[0, 0, 0] = np.nan
    incomplete_mask[0, 0, 0] = 0
    with pytest.raises(ValueError, match="ground truth.*finite"):
        to_pypots_sets(
            SimpleNamespace(target=target, mask=mask),
            SimpleNamespace(target=incomplete_truth, mask=incomplete_mask),
        )


@pytest.mark.parametrize(
    "validation_target",
    (
        np.ones((1, 5, 6), dtype=np.float32),
        np.ones((1, 4, 6), dtype=np.float64),
    ),
)
def test_pypots_sets_require_matching_train_validation_sequence_contract(
    validation_target,
):
    train_target = np.ones((2, 4, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="sequence shape and dtype"):
        to_pypots_sets(
            SimpleNamespace(target=train_target, mask=np.ones_like(train_target)),
            SimpleNamespace(
                target=validation_target, mask=np.ones_like(validation_target)
            ),
        )


def test_pypots_adapter_requires_validation_truth_to_match_observed_input():
    x = np.ones((1, 3, 6), dtype=np.float32)
    x[0, 1, 2] = np.nan
    truth = np.nan_to_num(x)
    truth[0, 0, 0] = 99
    with pytest.raises(ValueError, match="observed.*X_ori"):
        PyPOTSAdapter(FakeModel()).fit({"X": x}, {"X": x, "X_ori": truth})


@pytest.mark.parametrize(
    ("result", "error", "message"),
    (
        (np.ones((1, 3, 5), dtype=np.float32), ValueError, "shape"),
        (np.ones((1, 3, 6), dtype=np.float64), TypeError, "dtype"),
        (np.full((1, 3, 6), np.nan, dtype=np.float32), ValueError, "finite"),
        ([[[1.0] * 6] * 3], TypeError, "numpy"),
    ),
)
def test_pypots_adapter_rejects_malformed_imputation_results(result, error, message):
    class ResultModel(FakeModel):
        def impute(self, test_set, **kwargs):
            del test_set, kwargs
            return result

    x = np.ones((1, 3, 6), dtype=np.float32)
    x[0, 1, 2] = np.nan
    with pytest.raises(error, match=message):
        PyPOTSAdapter(ResultModel()).impute({"X": x})


def _install_fake_pypots(monkeypatch):
    monkeypatch.setattr(pypots_module.metadata, "version", lambda name: "1.5.0")
    package = ModuleType("pypots")
    imputation = ModuleType("pypots.imputation")
    classes = {}
    for name in ("BRITS", "SAITS", "CSDI"):
        model_class = type(
            name,
            (),
            {
                "__init__": lambda self, **kwargs: setattr(self, "kwargs", kwargs),
                "fit": lambda self, train, validation: None,
                "impute": lambda self, test, **kwargs: np.nan_to_num(
                    test["X"], nan=0.0
                ),
            },
        )
        setattr(imputation, name, model_class)
        classes[name] = model_class
    package.imputation = imputation
    monkeypatch.setitem(sys.modules, "pypots", package)
    monkeypatch.setitem(sys.modules, "pypots.imputation", imputation)
    return classes


def test_build_pypots_model_rejects_installed_version_mismatch(monkeypatch):
    _install_fake_pypots(monkeypatch)
    monkeypatch.setattr(pypots_module.metadata, "version", lambda name: "1.5.1")

    with pytest.raises(RuntimeError, match=r"requires exactly pypots==1\.5\.0.*1\.5\.1"):
        build_pypots_model(
            "brits",
            n_steps=8,
            epochs=1,
            batch_size=1,
            device="cpu",
            saving_path="results",
        )


def test_installed_pypots_version_accepts_equivalent_release_metadata(monkeypatch):
    monkeypatch.setattr(pypots_module.metadata, "version", lambda name: "1.5")

    assert pypots_module.installed_pypots_version() == "1.5.0"


@pytest.mark.parametrize("name", SUPPORTED_PYPOTS_MODELS)
def test_build_pypots_model_uses_pinned_1_5_constructor_contract(monkeypatch, name):
    classes = _install_fake_pypots(monkeypatch)
    adapter = build_pypots_model(
        name,
        n_steps=128,
        epochs=3,
        batch_size=4,
        device="cpu",
        saving_path="results/baseline",
    )

    expected_class = {"brits": "BRITS", "saits": "SAITS", "csdi": "CSDI"}[name]
    assert isinstance(adapter.model, classes[expected_class])
    common = {
        "n_steps": 128,
        "n_features": 6,
        "batch_size": 4,
        "epochs": 3,
        "device": "cpu",
        "saving_path": "results/baseline",
        "model_saving_strategy": "best",
    }
    assert common.items() <= adapter.model.kwargs.items()
    if name == "brits":
        assert adapter.model.kwargs["rnn_hidden_size"] == 64
    elif name == "saits":
        assert {
            "n_layers": 2,
            "d_model": 64,
            "n_heads": 4,
            "d_k": 16,
            "d_v": 16,
            "d_ffn": 128,
            "dropout": 0.1,
        }.items() <= adapter.model.kwargs.items()
    else:
        assert {
            "n_layers": 4,
            "n_heads": 4,
            "n_channels": 64,
            "d_time_embedding": 64,
            "d_feature_embedding": 16,
            "d_diffusion_embedding": 64,
            "n_diffusion_steps": 50,
        }.items() <= adapter.model.kwargs.items()
        assert adapter.impute_kwargs == {"n_sampling_times": 20}


def test_build_pypots_model_rejects_unknown_name_before_optional_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "pypots", raising=False)
    monkeypatch.delitem(sys.modules, "pypots.imputation", raising=False)
    with pytest.raises(ValueError, match="unsupported PyPOTS model"):
        build_pypots_model(
            "BRITS",
            n_steps=8,
            epochs=1,
            batch_size=1,
            device="cpu",
            saving_path="results",
        )
