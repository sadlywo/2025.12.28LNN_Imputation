from __future__ import annotations

import math
import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from imputation_v3.data.windows import (
    collate_prepared_windows,
    materialize_teacher_windows,
)
from imputation_v3.models.teacher import OfflineTeacher, TeacherOutput
from imputation_v3.config import TeacherConfig
from validation_v2.data.normalization import RobustTrainScaler
from validation_v2.types import Recording
from validation_v2.experiments.provenance import canonical_json


def _recording(recording_id: str, rows: int = 40) -> Recording:
    values = np.arange(rows * 6, dtype=np.float64).reshape(rows, 6) / 100.0
    values[:, 0] += len(recording_id)
    return Recording(
        id=recording_id,
        imu_time_s=np.arange(rows, dtype=np.float64) * 0.01,
        imu_six=values,
        vicon_time_s=np.arange(rows, dtype=np.float64) * 0.01,
        vicon_position_m=np.zeros((rows, 3), dtype=np.float64),
        vicon_quaternion_xyzw=np.zeros((rows, 4), dtype=np.float64),
        overlap_s=(0.0, (rows - 1) * 0.01),
        metadata={"scenario": "walk"},
    )


def _windows(count: int = 3):
    recording = _recording("training")
    scaler = RobustTrainScaler.fit([recording], allowed_ids={recording.id})
    windows = materialize_teacher_windows(
        [recording],
        scaler,
        window_samples=8,
        stride=8,
        seed=2026,
        topologies=("point", "block", "channel"),
        rates=(0.2, 0.5),
        exhaustive=True,
    )
    return windows[:count]


def _loader(*, count: int = 3, batch_size: int = 2):
    return DataLoader(
        _windows(count),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_prepared_windows,
    )


def _config(output_root: Path) -> TeacherConfig:
    return TeacherConfig(
        data_root=Path("data"),
        output_root=output_root,
        selection_split="validation",
        seeds=(2026,),
        window_seconds=(0.08,),
        nominal_dt_s=0.01,
        batch_size=2,
        epochs=1,
        hidden_size=2,
        tcn_width=2,
        tcn_dilations=(1,),
        learning_rate=1e-3,
        training_rates=(0.2,),
        training_topologies=("point",),
        models=("linear", "teacher"),
    )


def _patch_smoke_data(monkeypatch, module, *, split_token: str = "base"):
    import pandas as pd

    pairs = [
        {
            "recording_id": f"{split}/imu{index}",
            "scenario": "walk",
            "imu_path": f"/{split_token}/{split}/imu{index}.csv",
            "vicon_path": f"/{split_token}/{split}/vi{index}.csv",
        }
        for split, index in (("train", 1), ("train", 2), ("validation", 3), ("test", 4))
    ]
    assignments = {item["recording_id"]: item["recording_id"].split("/")[0] for item in pairs}

    def fake_split(index, *, seed):
        assert seed == 2026
        return pd.DataFrame(
            [
                {
                    **item,
                    "split": assignments[item["recording_id"]],
                    "imu_sha256": ("a" if split_token == "base" else "c") * 64,
                    "vicon_sha256": "b" * 64,
                }
                for item in index
            ]
        )

    loaded = []

    def fake_load(imu_path, vicon_path):
        del vicon_path
        path = Path(imu_path)
        recording_id = f"{path.parent.name}/{path.stem}"
        loaded.append(recording_id)
        return _recording(recording_id, rows=24)

    monkeypatch.setattr(module, "discover_oxiod_pairs", lambda root: pairs)
    monkeypatch.setattr(module, "stratified_file_split", fake_split)
    monkeypatch.setattr(module, "load_recording", fake_load)
    monkeypatch.setattr(
        module,
        "git_worktree_identity",
        lambda root: {"git_commit": "abc", "dirty_state_digest": ""},
    )
    return loaded


def _write_fake_core_artifacts(run_dir: Path, manifest: dict) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_bytes(
        (canonical_json(manifest) + "\n").encode("utf-8")
    )
    history = [
        {"epoch": 1, "train": {"missing_rmse": 1.0}, "validation": {"missing_rmse": 0.5}}
    ]
    (run_dir / "history.json").write_bytes(
        (canonical_json(history) + "\n").encode("utf-8")
    )
    hyperparameters = manifest["config"]["hyperparameters"]
    model = OfflineTeacher(
        31,
        hyperparameters["hidden_size"],
        hyperparameters["tcn_width"],
        tuple(hyperparameters["tcn_dilations"]),
        residual_mode="residual",
        time_mode="actual",
    )
    torch.save(model.state_dict(), run_dir / "best.pt")
    digest = hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest()
    metadata = {
        "run_id": manifest["run_id"],
        "best_epoch": 1,
        "selection_split": "validation",
        "selection_metric": "missing_rmse",
        "checkpoint_sha256": digest,
    }
    (run_dir / "checkpoint.json").write_bytes(
        (canonical_json(metadata) + "\n").encode("utf-8")
    )
    return metadata


def test_teacher_callback_factory_is_public():
    from imputation_v3.experiments.training import make_teacher_callbacks

    train_epoch, evaluate_epoch = make_teacher_callbacks(torch.device("cpu"))

    assert callable(train_epoch)
    assert callable(evaluate_epoch)


def test_real_offline_teacher_train_updates_and_eval_does_not():
    from imputation_v3.experiments.training import make_teacher_callbacks

    torch.manual_seed(3)
    model = OfflineTeacher(31, 4, 4, (1,), residual_mode="residual", time_mode="actual")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_epoch, evaluate_epoch = make_teacher_callbacks(torch.device("cpu"))
    before = [parameter.detach().clone() for parameter in model.parameters()]

    train_metrics = train_epoch(model, optimizer, _loader(count=1, batch_size=1), 1)

    assert model.training
    assert set(train_metrics) == {"missing_rmse"}
    assert math.isfinite(train_metrics["missing_rmse"])
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))

    trained = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer_state = {
        key: {
            name: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for name, value in state.items()
        }
        for key, state in optimizer.state.items()
    }
    evaluation_metrics = evaluate_epoch(model, _loader(count=1, batch_size=1), 1)

    assert not model.training
    assert set(evaluation_metrics) == {"missing_rmse"}
    assert math.isfinite(evaluation_metrics["missing_rmse"])
    assert all(torch.equal(old, new) for old, new in zip(trained, model.parameters()))
    for key, state in optimizer_state.items():
        for name, value in state.items():
            current = optimizer.state[key][name]
            assert torch.equal(value, current) if isinstance(value, torch.Tensor) else value == current


class _OffsetTeacher(nn.Module):
    def __init__(self, offset: float = 0.25):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(offset))

    def forward(self, features, dt, observed, mask, baseline):
        raw = baseline + self.offset
        return TeacherOutput(raw=raw, completed=raw, residual=raw - baseline, latent=features)


def test_metric_is_exact_whole_loader_channel_balanced_rmse():
    from imputation_v3.experiments.training import make_teacher_callbacks

    model = _OffsetTeacher()
    _, evaluate_epoch = make_teacher_callbacks(torch.device("cpu"))
    loader = _loader(count=3, batch_size=2)

    actual = evaluate_epoch(model, loader, 1)["missing_rmse"]

    sse = torch.zeros(6, dtype=torch.float64)
    counts = torch.zeros(6, dtype=torch.float64)
    batch_rmses = []
    with torch.inference_mode():
        for batch in loader:
            prediction = model(
                batch.features, batch.dt, batch.observed, batch.mask, batch.baseline
            ).raw
            missing = batch.mask == 0
            errors = torch.where(missing, prediction - batch.target, 0.0).double()
            batch_sse = errors.square().sum(dim=(0, 1))
            batch_counts = missing.sum(dim=(0, 1)).double()
            represented = batch_counts > 0
            batch_rmses.append(
                torch.sqrt((batch_sse[represented] / batch_counts[represented]).mean()).item()
            )
            sse += batch_sse
            counts += batch_counts
    represented = counts > 0
    oracle = torch.sqrt((sse[represented] / counts[represented]).mean()).item()

    assert actual == pytest.approx(oracle, rel=1e-12, abs=1e-12)
    assert actual != pytest.approx(sum(batch_rmses) / len(batch_rmses), rel=1e-7)


def test_callbacks_reject_empty_loader_and_malformed_batches():
    from imputation_v3.experiments.training import make_teacher_callbacks

    train_epoch, evaluate_epoch = make_teacher_callbacks(torch.device("cpu"))
    model = _OffsetTeacher()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="empty"):
        train_epoch(model, optimizer, [], 1)
    with pytest.raises(ValueError, match="empty"):
        evaluate_epoch(model, [], 1)
    with pytest.raises(TypeError, match="PreparedBatch|batch"):
        evaluate_epoch(model, [object()], 1)


def test_callbacks_reject_no_missing_values_and_invalid_devices():
    from imputation_v3.experiments.training import make_teacher_callbacks

    with pytest.raises(TypeError, match="torch.device"):
        make_teacher_callbacks("cpu")
    with pytest.raises(ValueError, match="cpu or cuda"):
        make_teacher_callbacks(torch.device("meta"))

    batch = next(iter(_loader(count=1, batch_size=1)))
    batch.mask.fill_(1)
    model = _OffsetTeacher()
    train_epoch, evaluate_epoch = make_teacher_callbacks(torch.device("cpu"))
    with pytest.raises(ValueError, match="missing"):
        train_epoch(model, torch.optim.SGD(model.parameters(), lr=0.1), [batch], 1)
    with pytest.raises(ValueError, match="missing"):
        evaluate_epoch(model, [batch], 1)


def test_callbacks_move_without_mutating_prepared_batch_source():
    from imputation_v3.experiments.training import make_teacher_callbacks

    batch = next(iter(_loader(count=1, batch_size=1)))
    expected = {
        name: getattr(batch, name).clone()
        for name in ("features", "target", "observed", "mask", "dt", "baseline")
    }
    _, evaluate_epoch = make_teacher_callbacks(torch.device("cpu"))

    evaluate_epoch(_OffsetTeacher(), [batch], 1)

    assert all(torch.equal(getattr(batch, name), value) for name, value in expected.items())


def test_callbacks_reject_nonfinite_loss_and_gradient_norm():
    from imputation_v3.experiments.training import make_teacher_callbacks

    train_epoch, _ = make_teacher_callbacks(torch.device("cpu"))
    loader = _loader(count=1, batch_size=1)

    nonfinite_model = _OffsetTeacher(float("inf"))
    with pytest.raises(ValueError, match="finite|nonfinite"):
        train_epoch(
            nonfinite_model,
            torch.optim.SGD(nonfinite_model.parameters(), lr=0.1),
            loader,
            1,
        )

    gradient_model = _OffsetTeacher()
    gradient_model.offset.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    with pytest.raises(ValueError, match="gradient norm|finite"):
        train_epoch(
            gradient_model,
            torch.optim.SGD(gradient_model.parameters(), lr=0.1),
            loader,
            1,
        )


def test_smoke_orchestration_is_train_validation_only_and_seeded(tmp_path, monkeypatch):
    import imputation_v3.experiments.training as training

    loaded = _patch_smoke_data(monkeypatch, training)
    captured = {}

    def fake_train_one_run(run_dir, manifest, **kwargs):
        captured.update(run_dir=run_dir, manifest=manifest, **kwargs)
        return _write_fake_core_artifacts(run_dir, manifest)

    monkeypatch.setattr(training, "train_one_run", fake_train_one_run)
    report = training.run_teacher_smoke(
        _config(Path("configured-output")),
        repository_root=tmp_path,
        requested_device="cpu",
        output_root=Path("override-output"),
    )

    resolved = captured["manifest"]["config"]
    assert loaded == ["train/imu1", "train/imu2", "validation/imu3"]
    assert resolved["mode"] == "imputation_v3_teacher_smoke"
    assert resolved["selection_split"] == "validation"
    assert resolved["model"] == "teacher"
    assert resolved["test_evaluation"] is False
    assert resolved["selected_recording_ids"] == {
        "train": ["train/imu1", "train/imu2"],
        "validation": ["validation/imu3"],
    }
    assert resolved["scaler_training_ids"] == ["train/imu1", "train/imu2"]
    assert len(resolved["selected_window_ids"]["train"]) <= 4
    assert len(resolved["selected_window_ids"]["validation"]) <= 4
    assert captured["train_loader"].collate_fn is collate_prepared_windows
    assert type(captured["train_loader"].sampler).__name__ == "RandomSampler"
    assert type(captured["validation_loader"].sampler).__name__ == "SequentialSampler"
    assert captured["epochs"] == 1
    assert report["status"] == "completed"
    assert Path(report["run_dir"]).parent == tmp_path / "override-output"


def test_smoke_enables_fail_closed_deterministic_algorithms(tmp_path, monkeypatch):
    import imputation_v3.experiments.training as training

    _patch_smoke_data(monkeypatch, training)
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    original = torch.use_deterministic_algorithms
    calls = []
    observed = {}

    def tracked(enabled, *, warn_only=False):
        calls.append((enabled, warn_only))
        return original(enabled, warn_only=warn_only)

    def fake_train_one_run(run_dir, manifest, **kwargs):
        observed["enabled"] = torch.are_deterministic_algorithms_enabled()
        observed["warn_only"] = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.zeros(1).put_(torch.tensor([0]), torch.tensor([1.0]))
        except RuntimeError as error:
            observed["unsupported_error"] = str(error)
        else:
            observed["unsupported_error"] = None
        return _write_fake_core_artifacts(run_dir, manifest)

    monkeypatch.setattr(torch, "use_deterministic_algorithms", tracked)
    monkeypatch.setattr(training, "train_one_run", fake_train_one_run)
    try:
        training.run_teacher_smoke(
            _config(Path("runs")), repository_root=tmp_path, requested_device="cpu"
        )

        assert calls == [(True, False)]
        assert observed["enabled"] is True
        assert observed["warn_only"] is False
        assert "does not have a deterministic implementation" in observed[
            "unsupported_error"
        ]
    finally:
        original(previous_enabled, warn_only=previous_warn_only)


def test_smoke_consumes_at_most_four_windows_per_split(tmp_path, monkeypatch):
    import imputation_v3.experiments.training as training

    _patch_smoke_data(monkeypatch, training)
    original_iter = training.iter_teacher_windows
    yielded = {"train": 0, "validation": 0}

    def tracked_iter(recordings, scaler, **kwargs):
        split = "validation" if recordings[0].id.startswith("validation/") else "train"
        for window in original_iter(recordings, scaler, **kwargs):
            yielded[split] += 1
            if yielded[split] > 4:
                raise AssertionError("smoke requested more than four prepared windows")
            yield window

    monkeypatch.setattr(training, "iter_teacher_windows", tracked_iter)
    monkeypatch.setattr(
        training,
        "train_one_run",
        lambda run_dir, manifest, **kwargs: _write_fake_core_artifacts(
            run_dir, manifest
        ),
    )

    training.run_teacher_smoke(
        _config(Path("runs")), repository_root=tmp_path, requested_device="cpu"
    )

    assert yielded == {"train": 4, "validation": 4}


def test_smoke_seals_evidence_and_validates_resume(tmp_path, monkeypatch):
    import imputation_v3.experiments.training as training

    _patch_smoke_data(monkeypatch, training)
    config = _config(Path("runs"))

    first = training.run_teacher_smoke(
        config, repository_root=tmp_path, requested_device="cpu"
    )
    run_dir = Path(first["run_dir"])

    assert first["status"] == "completed"
    assert {path.name for path in run_dir.iterdir()} == {
        "run.json",
        "history.json",
        "best.pt",
        "checkpoint.json",
        "evidence.json",
    }
    assert json.loads((run_dir / "checkpoint.json").read_text())["checkpoint_sha256"]

    resumed = training.run_teacher_smoke(
        config, repository_root=tmp_path, requested_device="cpu"
    )
    assert resumed["status"] == "resumed"
    assert resumed["checkpoint"] == first["checkpoint"]

    (run_dir / "history.json").unlink()
    with pytest.raises(ValueError, match="partial|inconsistent"):
        training.run_teacher_smoke(
            config, repository_root=tmp_path, requested_device="cpu"
        )


def test_smoke_recovers_evidence_after_crash_between_core_and_seal(
    tmp_path, monkeypatch
):
    import imputation_v3.experiments.training as training

    _patch_smoke_data(monkeypatch, training)
    monkeypatch.setattr(
        training,
        "train_one_run",
        lambda run_dir, manifest, **kwargs: _write_fake_core_artifacts(
            run_dir, manifest
        ),
    )
    original_write = training._write_stable
    failed = False

    def fail_first_evidence(path, content):
        nonlocal failed
        if path.name == "evidence.json" and not failed:
            failed = True
            raise RuntimeError("injected evidence publish crash")
        return original_write(path, content)

    monkeypatch.setattr(training, "_write_stable", fail_first_evidence)
    config = _config(Path("runs"))
    with pytest.raises(RuntimeError, match="injected evidence"):
        training.run_teacher_smoke(
            config, repository_root=tmp_path, requested_device="cpu"
        )

    run_dirs = [path for path in (tmp_path / "runs").iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    assert {path.name for path in run_dirs[0].iterdir()} == {
        "run.json", "history.json", "best.pt", "checkpoint.json"
    }

    recovered = training.run_teacher_smoke(
        config, repository_root=tmp_path, requested_device="cpu"
    )
    assert recovered["status"] == "resumed"
    assert recovered["evidence_recovered"] is True
    assert {path.name for path in Path(recovered["run_dir"]).iterdir()} == {
        "run.json", "history.json", "best.pt", "checkpoint.json", "evidence.json"
    }


@pytest.mark.parametrize("corruption", ("noncanonical_history", "extra_run_field"))
def test_smoke_refuses_to_seal_corrupted_exact_core_four(
    tmp_path, monkeypatch, corruption
):
    import imputation_v3.experiments.training as training

    _patch_smoke_data(monkeypatch, training)
    monkeypatch.setattr(
        training,
        "train_one_run",
        lambda run_dir, manifest, **kwargs: _write_fake_core_artifacts(
            run_dir, manifest
        ),
    )
    original_write = training._write_stable

    def crash_before_evidence(path, content):
        if path.name == "evidence.json":
            raise RuntimeError("injected evidence publish crash")
        return original_write(path, content)

    monkeypatch.setattr(training, "_write_stable", crash_before_evidence)
    config = _config(Path("runs"))
    with pytest.raises(RuntimeError, match="injected evidence"):
        training.run_teacher_smoke(
            config, repository_root=tmp_path, requested_device="cpu"
        )
    run_dir = next(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    if corruption == "noncanonical_history":
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
    else:
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        manifest["unexpected"] = True
        (run_dir / "run.json").write_text(
            canonical_json(manifest) + "\n", encoding="utf-8"
        )
    monkeypatch.setattr(training, "_write_stable", original_write)

    with pytest.raises(ValueError, match="canonical|schema|match|provenance"):
        training.run_teacher_smoke(
            config, repository_root=tmp_path, requested_device="cpu"
        )
    assert not (run_dir / "evidence.json").exists()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"seeds": (1, 2)}, "exactly one seed"),
        ({"models": ("linear",)}, "teacher"),
        ({"selection_split": "test"}, "validation"),
    ],
)
def test_smoke_rejects_non_smoke_configuration(tmp_path, changes, message):
    from dataclasses import replace
    from imputation_v3.experiments.training import run_teacher_smoke

    with pytest.raises(ValueError, match=message):
        run_teacher_smoke(
            replace(_config(Path("runs")), **changes),
            repository_root=tmp_path,
            requested_device="cpu",
        )


def _write_config(path: Path, *, selection_split: str = "validation") -> Path:
    path.write_text(
        "\n".join(
            (
                "data_root: data",
                "output_root: runs",
                f"selection_split: {selection_split}",
                "seeds: [2026]",
                "window_seconds: [0.08]",
                "nominal_dt_s: 0.01",
                "batch_size: 2",
                "epochs: 1",
                "hidden_size: 2",
                "tcn_width: 2",
                "tcn_dilations: [1]",
                "learning_rate: 0.001",
                "training_rates: [0.2]",
                "training_topologies: [point]",
                "models: [teacher]",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_cli_success_prints_only_canonical_json(tmp_path, monkeypatch, capsys):
    import imputation_v3.cli as cli

    config_path = _write_config(tmp_path / "teacher.yaml")
    expected = {
        "status": "completed",
        "run_id": "abc",
        "run_dir": "runs/abc",
        "checkpoint": {"checkpoint_sha256": "f" * 64},
        "counts": {"test_recordings_loaded": 0},
    }
    monkeypatch.setattr(cli, "run_teacher_smoke", lambda *args, **kwargs: expected)

    result = cli.main(
        [
            "teacher",
            "--config",
            str(config_path),
            "--smoke",
            "--device",
            "cpu",
            "--output-root",
            "override",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert captured.out == json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"


def test_cli_requires_smoke_and_rejects_test_selection(tmp_path, capsys):
    import imputation_v3.cli as cli

    config_path = _write_config(tmp_path / "teacher.yaml")
    assert cli.main(
        ["teacher", "--config", str(config_path), "--device", "cpu"]
    ) != 0
    assert "Task 10" in capsys.readouterr().err

    test_config = _write_config(tmp_path / "test.yaml", selection_split="test")
    assert cli.main(
        [
            "teacher",
            "--config",
            str(test_config),
            "--smoke",
            "--device",
            "cpu",
        ]
    ) != 0
    assert "selection_split" in capsys.readouterr().err
