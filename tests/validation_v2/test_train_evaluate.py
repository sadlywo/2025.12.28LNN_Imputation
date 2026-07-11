import hashlib
import inspect
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import torch

from validation_v2.experiments.provenance import collect_provenance, write_run_manifest


SCHEMA = [
    "run_id",
    "seed",
    "recording_id",
    "scenario",
    "protocol",
    "topology",
    "requested_fraction",
    "realized_fraction",
    "model",
    "metric",
    "value",
    "checkpoint_sha256",
]


def test_orchestration_api_is_exported_for_cli_callers():
    from validation_v2 import experiments

    assert callable(experiments.train_one_run)
    assert callable(experiments.evaluate_test_once)
    assert callable(experiments.enumerate_matrix)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed_run(tmp_path: Path) -> tuple[Path, dict, dict]:
    manifest = collect_provenance({"model": "pinn_imu"}, seed=7)
    run_dir = tmp_path / manifest["run_id"]
    write_run_manifest(run_dir, manifest)
    checkpoint = run_dir / "best.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    metadata = {
        "run_id": manifest["run_id"],
        "best_epoch": 2,
        "selection_split": "validation",
        "selection_metric": "missing_rmse",
        "checkpoint_sha256": _sha256(checkpoint),
    }
    (run_dir / "checkpoint.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    return run_dir, manifest, metadata


def _rows(manifest: dict, metadata: dict, recording_id: str = "rec-a") -> list[dict]:
    base = {
        "run_id": manifest["run_id"],
        "seed": manifest["seed"],
        "recording_id": recording_id,
        "scenario": "walk",
        "protocol": "holdout",
        "topology": "point",
        "requested_fraction": 0.2,
        "realized_fraction": 0.19,
        "model": "pinn_imu",
        "checkpoint_sha256": metadata["checkpoint_sha256"],
    }
    return [
        {**base, "metric": "reconstruction_normalized", "value": 0.3},
        {**base, "metric": "reconstruction_physical", "value": 0.4},
    ]


def test_selection_uses_only_validation_missing_rmse_and_earliest_tie():
    from validation_v2.experiments.train import select_best_checkpoint

    history = [
        {
            "epoch": 1,
            "train": {"missing_rmse": -999.0},
            "validation": {"missing_rmse": 0.5},
        },
        {
            "epoch": 2,
            "train": {"missing_rmse": 999.0},
            "validation": {"missing_rmse": 0.2},
        },
        {
            "epoch": 3,
            "train": {"missing_rmse": -9999.0},
            "validation": {"missing_rmse": 0.2},
        },
    ]

    assert select_best_checkpoint(history) == 2
    with pytest.raises(ValueError, match="test"):
        select_best_checkpoint([{**history[0], "test": {"missing_rmse": -1e9}}])


@pytest.mark.parametrize(
    "history, message",
    [
        ([], "empty"),
        ([{"epoch": 1, "train": {}, "validation": {"missing_rmse": float("nan")}}], "finite"),
        ([{"train": {}, "validation": {"missing_rmse": 1.0}}], "epoch"),
        (
            [
                {"epoch": 1, "train": {}, "validation": {"missing_rmse": 1.0}},
                {"epoch": 1, "train": {}, "validation": {"missing_rmse": 0.5}},
            ],
            "duplicate epoch",
        ),
    ],
)
def test_selection_rejects_invalid_history(history, message):
    from validation_v2.experiments.train import select_best_checkpoint

    with pytest.raises(ValueError, match=message):
        select_best_checkpoint(history)


@pytest.mark.parametrize("epochs", [[0, 1], [-1, 1], [1, 3], [2, 1]])
def test_selection_requires_epochs_in_strict_input_order_one_through_n(epochs):
    from validation_v2.experiments.train import select_best_checkpoint

    history = [
        {"epoch": epoch, "train": {}, "validation": {"missing_rmse": float(index)}}
        for index, epoch in enumerate(epochs)
    ]
    with pytest.raises(ValueError, match="epochs.*1.*N"):
        select_best_checkpoint(history)


def test_train_one_run_saves_only_best_validation_state_and_can_strictly_resume(tmp_path: Path):
    from validation_v2.experiments.train import train_one_run

    assert "test" not in inspect.signature(train_one_run).parameters
    assert "test_loader" not in inspect.signature(train_one_run).parameters
    manifest = collect_provenance({"model": "linear", "epochs": 2}, seed=11)
    run_dir = tmp_path / manifest["run_id"]
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [(torch.tensor([[1.0]]), torch.tensor([[1.0]])), (torch.tensor([[2.0]]), torch.tensor([[2.0]]))]
    states: dict[int, torch.Tensor] = {}

    def train_epoch(current, current_optimizer, loader, epoch):
        for x, y in loader:
            current_optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(current(x), y)
            loss.backward()
            current_optimizer.step()
        return {"missing_rmse": float(loss.detach().sqrt())}

    def validate_epoch(current, loader, epoch):
        assert loader is batches
        states[epoch] = current.weight.detach().clone()
        return {"missing_rmse": {1: 0.1, 2: 0.8}[epoch]}

    metadata = train_one_run(
        run_dir,
        manifest,
        model=model,
        optimizer=optimizer,
        train_loader=batches,
        validation_loader=batches,
        epochs=2,
        train_epoch=train_epoch,
        evaluate_epoch=validate_epoch,
    )

    assert metadata["best_epoch"] == 1
    assert metadata["selection_split"] == "validation"
    assert metadata["selection_metric"] == "missing_rmse"
    assert metadata["checkpoint_sha256"] == _sha256(run_dir / "best.pt")
    saved = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    assert torch.equal(saved["weight"], states[1])
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in history] == [1, 2]
    assert all(set(row) == {"epoch", "train", "validation"} for row in history)

    def forbidden(*args):
        raise AssertionError("resume must not invoke epoch callbacks")

    resumed = train_one_run(
        run_dir,
        manifest,
        model=torch.nn.Linear(1, 1),
        optimizer=torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1),
        train_loader=[],
        validation_loader=[],
        epochs=2,
        train_epoch=forbidden,
        evaluate_epoch=forbidden,
        expected_checkpoint_sha256=metadata["checkpoint_sha256"],
    )
    assert resumed == metadata
    with pytest.raises(ValueError, match="checkpoint hash"):
        train_one_run(
            run_dir,
            manifest,
            model=model,
            optimizer=optimizer,
            train_loader=[],
            validation_loader=[],
            epochs=2,
            train_epoch=forbidden,
            evaluate_epoch=forbidden,
            expected_checkpoint_sha256="0" * 64,
        )

    history[1]["epoch"] = 3
    (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    with pytest.raises(ValueError, match="epochs.*1.*N"):
        train_one_run(
            run_dir,
            manifest,
            model=model,
            optimizer=optimizer,
            train_loader=[],
            validation_loader=[],
            epochs=2,
            train_epoch=forbidden,
            evaluate_epoch=forbidden,
            expected_checkpoint_sha256=metadata["checkpoint_sha256"],
        )


def test_train_one_run_supports_factories_and_propagates_callback_errors(tmp_path: Path):
    from validation_v2.experiments.train import train_one_run

    manifest = collect_provenance({"factory": True}, seed=3)
    made = []

    def model_factory():
        model = torch.nn.Linear(1, 1)
        made.append(model)
        return model

    def optimizer_factory(model):
        assert model is made[0]
        return torch.optim.SGD(model.parameters(), lr=0.01)

    def explode(*args):
        raise RuntimeError("epoch exploded")

    with pytest.raises(RuntimeError, match="epoch exploded"):
        train_one_run(
            tmp_path / manifest["run_id"],
            manifest,
            model_factory=model_factory,
            optimizer_factory=optimizer_factory,
            train_loader=[],
            validation_loader=[],
            epochs=1,
            train_epoch=explode,
            evaluate_epoch=lambda *args: {"missing_rmse": 1.0},
        )
    assert len(made) == 1


def test_tampered_checkpoint_is_rejected_before_test_loader_factory(tmp_path: Path):
    from validation_v2.experiments.evaluate import evaluate_test_once

    run_dir, manifest, metadata = _sealed_run(tmp_path)
    (run_dir / "best.pt").write_bytes(b"tampered")
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return ["rec-a"]

    with pytest.raises(ValueError, match="hash"):
        evaluate_test_once(
            run_dir,
            test_loader_factory=factory,
            evaluate_record=lambda record, checkpoint: _rows(manifest, metadata, record),
        )
    assert calls == 0


def test_test_evaluation_is_allowed_exactly_once(tmp_path: Path):
    from validation_v2.experiments.evaluate import evaluate_test_once

    run_dir, manifest, metadata = _sealed_run(tmp_path)
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return ["rec-a"]

    callback = lambda record, checkpoint: _rows(manifest, metadata, record)
    evaluate_test_once(run_dir, factory, callback)
    with pytest.raises(ValueError, match="already evaluated"):
        evaluate_test_once(run_dir, factory, callback)
    assert calls == 1


def test_concurrent_test_evaluations_construct_at_most_one_loader(tmp_path: Path):
    from validation_v2.experiments.evaluate import evaluate_test_once

    run_dir, manifest, metadata = _sealed_run(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    lock = threading.Lock()

    def factory():
        nonlocal calls
        with lock:
            calls += 1
        entered.set()
        assert release.wait(timeout=5)
        return ["rec-a"]

    callback = lambda record, checkpoint: _rows(manifest, metadata, record)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(evaluate_test_once, run_dir, factory, callback)
        assert entered.wait(timeout=5)
        second = executor.submit(evaluate_test_once, run_dir, factory, callback)
        with pytest.raises(ValueError, match="already evaluated"):
            second.result(timeout=5)
        release.set()
        first.result(timeout=5)
    assert calls == 1


def test_failed_test_evaluation_is_ledgered_and_not_automatically_retried(tmp_path: Path):
    from validation_v2.experiments.evaluate import evaluate_test_once

    run_dir, _, _ = _sealed_run(tmp_path)
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        raise RuntimeError("data failed")

    with pytest.raises(RuntimeError, match="data failed"):
        evaluate_test_once(run_dir, factory, lambda *_: [])
    ledger = json.loads((run_dir / "test_evaluation.json").read_text(encoding="utf-8"))
    assert ledger["status"] == "failed"
    assert "started_at" in ledger and "failed_at" in ledger
    with pytest.raises(ValueError, match="already evaluated"):
        evaluate_test_once(run_dir, factory, lambda *_: [])
    assert calls == 1


def test_evaluation_writes_fixed_schema_and_requires_reconstruction_metrics(tmp_path: Path):
    from validation_v2.experiments.evaluate import evaluate_test_once

    run_dir, manifest, metadata = _sealed_run(tmp_path)
    metrics = evaluate_test_once(
        run_dir,
        lambda: ["rec-a"],
        lambda record, checkpoint: _rows(manifest, metadata, record),
    )
    assert metrics.read_text(encoding="utf-8").splitlines()[0].split(",") == SCHEMA
    ledger = json.loads((run_dir / "test_evaluation.json").read_text(encoding="utf-8"))
    assert ledger["status"] == "completed"
    assert "started_at" in ledger and "completed_at" in ledger

    other_dir, other_manifest, other_metadata = _sealed_run(tmp_path / "other")
    incomplete = _rows(other_manifest, other_metadata)[:1]
    with pytest.raises(ValueError, match="reconstruction_physical"):
        evaluate_test_once(other_dir, lambda: ["rec-a"], lambda *_: incomplete)


@pytest.mark.parametrize(
    "field, other_value",
    [
        ("scenario", "run"),
        ("model", "other_model"),
        ("topology", "block"),
        ("requested_fraction", 0.3),
        ("realized_fraction", 0.29),
    ],
)
def test_reconstruction_metrics_must_coexist_in_each_evaluation_cell(
    tmp_path: Path, field: str, other_value
):
    from validation_v2.experiments.evaluate import evaluate_test_once

    run_dir, manifest, metadata = _sealed_run(tmp_path)
    split_rows = _rows(manifest, metadata)
    split_rows[1][field] = other_value

    with pytest.raises(ValueError, match="evaluation cell.*missing required metrics"):
        evaluate_test_once(run_dir, lambda: ["rec-a"], lambda *_: split_rows)


def test_trajectory_metrics_must_coexist_in_each_evaluation_cell(tmp_path: Path):
    from validation_v2.experiments.evaluate import evaluate_test_once

    run_dir, manifest, metadata = _sealed_run(tmp_path)
    base = _rows(manifest, metadata)
    template = dict(base[0])
    trajectory = [
        {**template, "metric": metric, "value": 0.1}
        for metric in ("ate_rmse_m", "rpe_rmse_m", "endpoint_drift_m", "velocity_rmse_mps")
    ]
    trajectory.append(
        {**template, "scenario": "run", "metric": "delta_ate_rmse_m", "value": 0.1}
    )

    with pytest.raises(ValueError, match="evaluation cell.*missing required metrics"):
        evaluate_test_once(
            run_dir,
            lambda: ["rec-a"],
            lambda *_: base + trajectory,
            trajectory_enabled=True,
        )


def test_trajectory_evaluation_requires_all_trajectory_metrics(tmp_path: Path):
    from validation_v2.experiments.evaluate import evaluate_test_once

    run_dir, manifest, metadata = _sealed_run(tmp_path)
    with pytest.raises(ValueError, match="ate"):
        evaluate_test_once(
            run_dir,
            lambda: ["rec-a"],
            lambda record, checkpoint: _rows(manifest, metadata, record),
            trajectory_enabled=True,
        )


def test_trajectory_accepts_task7_physical_metric_names(tmp_path: Path):
    from validation_v2.experiments.evaluate import evaluate_test_once

    run_dir, manifest, metadata = _sealed_run(tmp_path)
    base = _rows(manifest, metadata)
    template = dict(base[0])
    trajectory_rows = [
        {**template, "metric": metric, "value": 0.1}
        for metric in (
            "ate_rmse_m",
            "rpe_rmse_m",
            "endpoint_drift_m",
            "velocity_rmse_mps",
            "delta_ate_rmse_m",
        )
    ]
    evaluate_test_once(
        run_dir,
        lambda: ["rec-a"],
        lambda *_: base + trajectory_rows,
        trajectory_enabled=True,
    )


def test_evaluation_rejects_nonfinite_scenario_fractions_and_duplicate_rows(tmp_path: Path):
    from validation_v2.experiments.evaluate import evaluate_test_once

    run_dir, manifest, metadata = _sealed_run(tmp_path)
    rows = _rows(manifest, metadata)
    rows[0]["realized_fraction"] = float("nan")
    with pytest.raises(ValueError, match="realized_fraction must be finite"):
        evaluate_test_once(run_dir, lambda: ["rec-a"], lambda *_: rows)

    other_dir, other_manifest, other_metadata = _sealed_run(tmp_path / "duplicate")
    duplicate = _rows(other_manifest, other_metadata)
    duplicate.append(dict(duplicate[0]))
    with pytest.raises(ValueError, match="duplicate metric row"):
        evaluate_test_once(other_dir, lambda: ["rec-a"], lambda *_: duplicate)


def test_inconsistent_preexisting_metrics_are_rejected_before_loader_creation(tmp_path: Path):
    from validation_v2.experiments.evaluate import evaluate_test_once

    run_dir, manifest, metadata = _sealed_run(tmp_path)
    (run_dir / "per_record_metrics.csv").write_text("legacy,output\n", encoding="utf-8")
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return ["rec-a"]

    with pytest.raises(ValueError, match="partial or inconsistent"):
        evaluate_test_once(run_dir, factory, lambda record, _: _rows(manifest, metadata, record))
    assert calls == 0


def test_matrix_is_deterministic_content_addressed_and_separates_irregular_cases():
    from validation_v2.experiments.matrix import enumerate_matrix

    config = {
        "models": ["pinn_imu", "linear"],
        "seeds": [2, 1],
        "topologies": ["block", "point"],
        "rates": [0.3, 0.1],
        "protocols": ["cross_session", "holdout"],
        "irregular_cases": [
            {"method": "drop_timestamps", "requested_irregularity": 0.2}
        ],
    }
    reordered = {
        "protocols": list(reversed(config["protocols"])),
        "rates": list(reversed(config["rates"])),
        "topologies": list(reversed(config["topologies"])),
        "seeds": list(reversed(config["seeds"])),
        "models": list(reversed(config["models"])),
        "irregular_cases": list(config["irregular_cases"]),
    }

    first = enumerate_matrix(config)
    assert first == enumerate_matrix(reordered)
    assert len(first) == 2 * 2 * 2 * 2 * 2 + 2 * 2 * 2
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        enumerate_matrix(config), sort_keys=True, separators=(",", ":")
    )
    assert len({item["combination_id"] for item in first}) == len(first)
    assert all(len(item["combination_id"]) == 64 for item in first)
    normal = [item for item in first if item["case_type"] == "missingness"]
    irregular = [item for item in first if item["case_type"] == "irregular"]
    assert all(item["realized_fraction"] is None for item in normal)
    assert all(item["topology"] is None and item["requested_fraction"] is None for item in irregular)


def test_matrix_rejects_duplicate_combinations():
    from validation_v2.experiments.matrix import enumerate_matrix

    with pytest.raises(ValueError, match="duplicate combination"):
        enumerate_matrix(
            {
                "models": ["pinn_imu", "pinn_imu"],
                "seeds": [1],
                "topologies": ["point"],
                "rates": [0.1],
                "protocols": ["holdout"],
            }
        )
