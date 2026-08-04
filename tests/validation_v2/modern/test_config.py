from pathlib import Path

import pytest

from validation_v2.modern.config import (
    MODERN_MODELS,
    REFERENCE_MODELS,
    load_modern_config,
)


def test_registry_declares_exact_main_table_models():
    assert REFERENCE_MODELS == ("linear", "locf", "bilstm", "bilnn", "hybrid")
    assert MODERN_MODELS == ("brits", "saits", "csdi", "sssd")


def test_stage_a_is_strict_file_five_seed_thirteen_condition_campaign():
    config = load_modern_config(Path("configs/validation_v2/modern_stage_a.yaml"))
    assert config.protocol == "strict_file"
    assert config.seeds == (2026, 2027, 2028, 2029, 2030)
    assert config.rates == (0.1, 0.2, 0.3, 0.4)
    assert config.topologies == ("point", "block", "channel")
    assert config.irregular_cases == 1
    assert config.n_sampling_times == 50
    assert config.models == REFERENCE_MODELS + MODERN_MODELS


def test_smoke_is_bounded_to_one_condition_and_two_samples():
    config = load_modern_config(Path("configs/validation_v2/modern_smoke.yaml"))
    assert config.seeds == (2026,)
    assert config.rates == (0.3,)
    assert config.topologies == ("point",)
    assert config.max_train_windows == 4
    assert config.epochs == 1
    assert config.n_sampling_times == 2


def test_config_rejects_unknown_key(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("data_root: Oxford Dataset\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_modern_config(path)
