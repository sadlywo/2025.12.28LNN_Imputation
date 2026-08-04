import numpy as np

from validation_v2.modern.pypots_worker import build_model, normalize_imputation


def test_saits_factory_uses_registered_shape_and_capacity():
    calls = {}

    class Fake:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    build_model(
        "saits",
        {"n_layers": 1, "d_model": 64, "learning_rate": 0.001},
        n_steps=30,
        n_features=6,
        batch_size=8,
        epochs=2,
        patience=1,
        device="cpu",
        classes={"saits": Fake},
    )
    assert calls["n_steps"] == 30 and calls["n_features"] == 6
    assert calls["d_model"] == 64 and calls["n_heads"] == 4


def test_csdi_sample_axis_is_normalized():
    raw = np.zeros((3, 50, 30, 6), dtype=np.float32)
    assert normalize_imputation(
        raw, windows=3, samples=50, steps=30, features=6
    ).shape == (3, 50, 30, 6)
