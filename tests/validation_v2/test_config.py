from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import get_type_hints

import pytest

from validation_v2.config import ExperimentConfig, load_config


def test_config_declares_selection_split_as_str():
    assert get_type_hints(ExperimentConfig)["selection_split"] is str


def test_config_rejects_test_tuning(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "selection_split: test\nseeds: [2026]\n",
        encoding="utf-8",
    )

    try:
        load_config(path)
    except ValueError as exc:
        assert "selection_split must be validation" in str(exc)
    else:
        raise AssertionError("test-based selection must be rejected")


def test_config_converts_declared_field_types(tmp_path: Path):
    path = tmp_path / "valid.yaml"
    path.write_text(
        "\n".join(
            [
                "data_root: ./data",
                "output_root: ./results",
                "seeds: [2026, '2027']",
                "seq_len: '64'",
                "batch_size: '32'",
                "epochs: '10'",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config == ExperimentConfig(
        data_root=Path("data"),
        output_root=Path("results"),
        selection_split="validation",
        seeds=(2026, 2027),
        seq_len=64,
        batch_size=32,
        epochs=10,
    )
    with pytest.raises(FrozenInstanceError):
        config.epochs = 11  # type: ignore[misc]


@pytest.mark.parametrize("seeds_line", ["", "seeds: []\n"])
def test_config_rejects_missing_or_empty_seeds(
    tmp_path: Path,
    seeds_line: str,
):
    path = tmp_path / "no-seeds.yaml"
    path.write_text(
        "\n".join(
            [
                "data_root: ./data",
                "output_root: ./results",
                "seq_len: 64",
                "batch_size: 32",
                "epochs: 10",
                seeds_line,
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="seeds must be a non-empty list"):
        load_config(path)
