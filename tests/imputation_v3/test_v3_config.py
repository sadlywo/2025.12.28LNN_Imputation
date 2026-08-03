from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from imputation_v3.config import ALLOWED_MODELS, TeacherConfig, load_teacher_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


VALID_CONFIG = """\
data_root: Oxford Dataset
output_root: results/imputation_v3/smoke
selection_split: validation
seeds: [2026]
window_seconds: [1.28, 2.56, 5.12]
nominal_dt_s: 0.01
batch_size: 2
epochs: 1
hidden_size: 64
tcn_width: 48
tcn_dilations: [1, 2, 4]
learning_rate: 0.001
training_rates: [0.1, 0.2, 0.3, 0.4]
training_topologies: [point, block, channel]
models: [linear, teacher]
"""


def _write_config(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "teacher.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def test_load_teacher_config_converts_to_immutable_typed_values(tmp_path: Path):
    config = load_teacher_config(_write_config(tmp_path, VALID_CONFIG))

    assert config.selection_split == "validation"
    assert config.window_samples == (128, 256, 512)
    assert config.models == ("linear", "teacher")
    assert config.data_root == Path("Oxford Dataset")
    assert config.seeds == (2026,)
    assert isinstance(config, TeacherConfig)
    with pytest.raises(FrozenInstanceError):
        config.epochs = 2  # type: ignore[misc]


def test_allowed_models_matches_the_supported_baselines_and_teacher():
    assert ALLOWED_MODELS == frozenset(
        {
            "locf",
            "linear",
            "pchip",
            "rts",
            "bilstm",
            "bilnn",
            "tcn",
            "feature_mlp",
            "teacher",
            "brits",
            "saits",
            "csdi",
        }
    )


def test_config_rejects_test_selection_even_when_other_fields_are_absent(
    tmp_path: Path,
):
    path = _write_config(tmp_path, "selection_split: test\n")

    with pytest.raises(ValueError, match="selection_split"):
        load_teacher_config(path)


def test_config_rejects_an_unsupported_model(tmp_path: Path):
    path = _write_config(
        tmp_path,
        VALID_CONFIG.replace("models: [linear, teacher]", "models: [linear, oracle]"),
    )

    with pytest.raises(ValueError, match="unsupported models: oracle"):
        load_teacher_config(path)


def test_config_rejects_an_empty_seed_list(tmp_path: Path):
    path = _write_config(
        tmp_path,
        VALID_CONFIG.replace("seeds: [2026]", "seeds: []"),
    )

    with pytest.raises(ValueError, match="seeds must be a non-empty list"):
        load_teacher_config(path)


@pytest.mark.parametrize(
    ("original", "replacement", "field"),
    [
        ("window_seconds: [1.28, 2.56, 5.12]", "window_seconds: [0]", "window_seconds"),
        ("nominal_dt_s: 0.01", "nominal_dt_s: 0", "nominal_dt_s"),
        ("batch_size: 2", "batch_size: 0", "batch_size"),
        ("epochs: 1", "epochs: -1", "epochs"),
        ("hidden_size: 64", "hidden_size: 0", "hidden_size"),
        ("tcn_width: 48", "tcn_width: -1", "tcn_width"),
        ("tcn_dilations: [1, 2, 4]", "tcn_dilations: [1, 0]", "tcn_dilations"),
        ("learning_rate: 0.001", "learning_rate: 0", "learning_rate"),
        ("training_rates: [0.1, 0.2, 0.3, 0.4]", "training_rates: [0]", "training_rates"),
    ],
)
def test_config_rejects_non_positive_numeric_values(
    tmp_path: Path,
    original: str,
    replacement: str,
    field: str,
):
    path = _write_config(tmp_path, VALID_CONFIG.replace(original, replacement))

    with pytest.raises(ValueError, match=field):
        load_teacher_config(path)


@pytest.mark.parametrize(
    ("original", "replacement", "field"),
    [
        ("seeds: [2026]", "seeds: [true]", "seeds"),
        ("seeds: [2026]", "seeds: ['2026']", "seeds"),
        ("seeds: [2026]", "seeds: [2026.5]", "seeds"),
        ("batch_size: 2", "batch_size: true", "batch_size"),
        ("batch_size: 2", "batch_size: '2'", "batch_size"),
        ("batch_size: 2", "batch_size: 1.9", "batch_size"),
        ("nominal_dt_s: 0.01", "nominal_dt_s: true", "nominal_dt_s"),
        ("nominal_dt_s: 0.01", "nominal_dt_s: '0.01'", "nominal_dt_s"),
        ("nominal_dt_s: 0.01", "nominal_dt_s: .nan", "nominal_dt_s"),
        ("nominal_dt_s: 0.01", "nominal_dt_s: .inf", "nominal_dt_s"),
        ("nominal_dt_s: 0.01", "nominal_dt_s: -.inf", "nominal_dt_s"),
        ("tcn_dilations: [1, 2, 4]", "tcn_dilations: [true]", "tcn_dilations"),
        ("tcn_dilations: [1, 2, 4]", "tcn_dilations: ['2']", "tcn_dilations"),
        ("tcn_dilations: [1, 2, 4]", "tcn_dilations: [1.9]", "tcn_dilations"),
        (
            "training_rates: [0.1, 0.2, 0.3, 0.4]",
            "training_rates: [true]",
            "training_rates",
        ),
        (
            "training_rates: [0.1, 0.2, 0.3, 0.4]",
            "training_rates: ['0.2']",
            "training_rates",
        ),
        (
            "training_rates: [0.1, 0.2, 0.3, 0.4]",
            "training_rates: [.nan]",
            "training_rates",
        ),
        (
            "training_rates: [0.1, 0.2, 0.3, 0.4]",
            "training_rates: [.inf]",
            "training_rates",
        ),
        (
            "training_rates: [0.1, 0.2, 0.3, 0.4]",
            "training_rates: [-.inf]",
            "training_rates",
        ),
    ],
)
def test_config_rejects_silent_numeric_coercion_and_non_finite_values(
    tmp_path: Path,
    original: str,
    replacement: str,
    field: str,
):
    path = _write_config(tmp_path, VALID_CONFIG.replace(original, replacement))

    with pytest.raises(ValueError, match=field):
        load_teacher_config(path)


def test_committed_smoke_config_loads_with_expected_values():
    config = load_teacher_config(
        REPOSITORY_ROOT / "configs" / "imputation_v3" / "teacher_smoke.yaml"
    )

    assert config.output_root == Path("results/imputation_v3/smoke")
    assert config.window_samples == (128,)
    assert config.hidden_size == 16
    assert config.models == ("linear", "teacher")


def test_committed_full_config_loads_with_expected_values():
    config = load_teacher_config(
        REPOSITORY_ROOT / "configs" / "imputation_v3" / "teacher_full.yaml"
    )

    assert config.output_root == Path("results/imputation_v3/formal")
    assert config.seeds == (2026, 2027, 2028, 2029, 2030)
    assert config.window_samples == (128, 256, 512)
    assert config.models == (
        "locf",
        "linear",
        "pchip",
        "rts",
        "bilstm",
        "bilnn",
        "tcn",
        "feature_mlp",
        "teacher",
        "brits",
        "saits",
        "csdi",
    )


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            "requirements-imputation-v3.txt",
            "-r requirements-validation-v2.txt\nscipy==1.13.1\n",
        ),
        (
            "requirements-imputation-v3-baselines.txt",
            "-r requirements-imputation-v3.txt\npypots==1.5.0\n",
        ),
    ],
)
def test_dependency_manifest_has_exact_contents(filename: str, expected: str):
    assert (REPOSITORY_ROOT / filename).read_text(encoding="utf-8") == expected
