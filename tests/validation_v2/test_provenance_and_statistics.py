import csv
import hashlib
import importlib
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

import validation_v2.experiments.provenance as provenance
from validation_v2.experiments.provenance import run_id
from validation_v2.experiments.summarize import summarize_runs


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


def test_run_id_is_canonical_and_sensitive_to_reproducibility_inputs():
    first = run_id(
        {"seq_len": 30, "data_root": Path("data")},
        2026,
        split_hash="split-a",
        scaler_hash="scaler-a",
        git_commit="abc",
        dirty_digest="clean",
    )
    reordered = run_id(
        {"data_root": Path("data"), "seq_len": 30},
        2026,
        split_hash="split-a",
        scaler_hash="scaler-a",
        git_commit="abc",
        dirty_digest="clean",
    )

    assert first == reordered
    assert len(first) == 16
    assert first != run_id({"seq_len": 50, "data_root": Path("data")}, 2026)
    assert first != run_id({"seq_len": 30, "data_root": Path("data")}, 2027)
    assert first != run_id(
        {"seq_len": 30, "data_root": Path("data")},
        2026,
        split_hash="split-b",
        scaler_hash="scaler-a",
        git_commit="abc",
        dirty_digest="clean",
    )

    base = ({"seq_len": 30}, 2026, "split", "scaler", "git", "dirty")
    for position, replacement in enumerate(({"seq_len": 50}, 2027, "x", "x", "x", "x")):
        changed = list(base)
        changed[position] = replacement
        assert run_id(*base) != run_id(*changed)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {1, 2}, object()])
def test_run_id_rejects_noncanonical_values(bad):
    with pytest.raises((TypeError, ValueError)):
        run_id({"bad": bad}, 1)


def test_collect_provenance_resolves_config_and_marks_missing_packages(monkeypatch):
    def fake_version(name: str) -> str:
        if name == "ncps":
            raise provenance.metadata.PackageNotFoundError(name)
        return f"version-of-{name}"

    monkeypatch.setattr(provenance.metadata, "version", fake_version)
    manifest = provenance.collect_provenance(
        {"path": Path("data"), "seq_len": 30},
        seed=2026,
        split_hash="split",
        scaler_hash="scaler",
        git_commit="git",
        dirty_digest="dirty",
    )

    assert manifest["run_id"] == run_id(
        {"path": Path("data"), "seq_len": 30},
        2026,
        "split",
        "scaler",
        "git",
        "dirty",
    )
    assert manifest["config"] == {"path": "data", "seq_len": 30}
    assert len(manifest["config_sha256"]) == 64
    assert manifest["dirty_state_digest"] == "dirty"
    assert manifest["package_versions"]["PyYAML"] == "version-of-PyYAML"
    assert manifest["package_versions"]["ncps"] == "not-installed"
    assert set(manifest) == {
        "run_id",
        "seed",
        "config",
        "config_sha256",
        "split_hash",
        "scaler_hash",
        "git_commit",
        "dirty_state_digest",
        "package_versions",
        "python",
        "platform",
    }


def test_runtime_fingerprint_is_public_and_collect_provenance_reuses_it(monkeypatch):
    monkeypatch.setattr(provenance.metadata, "version", lambda name: f"v-{name}")
    monkeypatch.setattr(provenance.sys, "version", "3.9.19 custom")
    monkeypatch.setattr(provenance.platform_module, "platform", lambda: "test-platform")

    runtime = provenance.runtime_fingerprint()
    manifest = provenance.collect_provenance({"seq_len": 30}, seed=7)

    assert runtime == {
        "package_versions": {
            name: f"v-{name}" for name in provenance.PACKAGE_DISTRIBUTIONS
        },
        "python": "3.9.19",
        "platform": "test-platform",
    }
    assert {field: manifest[field] for field in runtime} == runtime
    assert manifest["run_id"] == provenance.run_id({"seq_len": 30}, 7)


def test_git_worktree_identity_uses_repository_root_from_external_cwd(
    tmp_path: Path, monkeypatch
):
    repository_root = Path(__file__).resolve().parents[2]
    expected_commit = subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty_text = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    monkeypatch.chdir(tmp_path)

    identity = provenance.git_worktree_identity(repository_root)

    assert identity == {
        "git_commit": expected_commit,
        "dirty_state_digest": (
            hashlib.sha256(dirty_text.encode("utf-8")).hexdigest()
            if dirty_text
            else ""
        ),
    }


def test_write_run_manifest_is_idempotent_and_rejects_content_change(tmp_path: Path):
    manifest = provenance.collect_provenance({"seq_len": 30}, seed=1)
    path = provenance.write_run_manifest(tmp_path / manifest["run_id"], manifest)
    original = path.read_bytes()

    assert provenance.write_run_manifest(path.parent, dict(manifest)) == path
    assert path.read_bytes() == original
    changed = dict(manifest)
    changed["git_commit"] = "different"
    with pytest.raises(ValueError, match="different content"):
        provenance.write_run_manifest(path.parent, changed)


def test_write_run_manifest_rejects_a_tampered_content_id(tmp_path: Path):
    manifest = provenance.collect_provenance({"seq_len": 30}, seed=1)
    manifest["run_id"] = "0" * 16
    with pytest.raises(ValueError, match="run_id does not match provenance content"):
        provenance.write_run_manifest(tmp_path / "tampered", manifest)


def test_write_run_manifest_validates_preexisting_identical_invalid_bytes(tmp_path: Path):
    manifest = provenance.collect_provenance({"seq_len": 30}, seed=1)
    manifest["run_id"] = "0" * 16
    run_dir = tmp_path / "preexisting"
    run_dir.mkdir()
    (run_dir / "run.json").write_bytes(
        provenance.canonical_json(manifest).encode("utf-8") + b"\n"
    )

    with pytest.raises(ValueError, match="run_id does not match provenance content"):
        provenance.write_run_manifest(run_dir, manifest)


def test_write_run_manifest_rejects_incorrect_config_sha256(tmp_path: Path):
    manifest = provenance.collect_provenance({"seq_len": 30}, seed=1)
    manifest["config_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="config_sha256 does not match resolved config"):
        provenance.write_run_manifest(tmp_path / "bad-config-hash", manifest)


def test_write_run_manifest_rejects_missing_required_field(tmp_path: Path):
    manifest = provenance.collect_provenance({"seq_len": 30}, seed=1)
    del manifest["config_sha256"]

    with pytest.raises(ValueError, match="missing manifest fields: config_sha256"):
        provenance.write_run_manifest(tmp_path / "missing-field", manifest)


@pytest.mark.parametrize("field", ["split_hash", "scaler_hash", "dirty_state_digest"])
def test_write_run_manifest_rejects_malformed_content_digest(tmp_path: Path, field: str):
    argument = "dirty_digest" if field == "dirty_state_digest" else field
    kwargs = {argument: "not-a-sha256"}
    manifest = provenance.collect_provenance({"seq_len": 30}, seed=1, **kwargs)

    with pytest.raises(ValueError, match=rf"{field} must be empty or 64 lowercase hex"):
        provenance.write_run_manifest(tmp_path / field, manifest)


def test_write_run_manifest_does_not_clobber_target_created_during_publish(
    tmp_path: Path, monkeypatch
):
    manifest = provenance.collect_provenance({"seq_len": 30}, seed=1)
    run_dir = tmp_path / "racing"
    target = run_dir / "run.json"
    competing_content = b'{"winner":"other-process"}\n'

    def competing_link(_source, destination):
        Path(destination).write_bytes(competing_content)
        raise FileExistsError

    monkeypatch.setattr(provenance.os, "link", competing_link)

    with pytest.raises(ValueError, match="already has different content"):
        provenance.write_run_manifest(run_dir, manifest)
    assert target.read_bytes() == competing_content


def _write_run(root: Path, run_name: str, seed: int, rows: list[dict]) -> None:
    run_dir = root / run_name
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": run_name, "seed": seed}), encoding="utf-8"
    )
    with (run_dir / "per_record_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA)
        writer.writeheader()
        writer.writerows(rows)


def _rows(seed: int, run_name: str) -> list[dict]:
    values = {
        1: {"rec-a": {"baseline": 1.0, "pinn_imu": 0.5}, "rec-b": {"baseline": 2.0, "pinn_imu": 1.0}},
        2: {"rec-a": {"baseline": 1.2, "pinn_imu": 0.7}, "rec-b": {"baseline": 2.2, "pinn_imu": 1.2}},
    }
    return [
        {
            "run_id": run_name,
            "seed": seed,
            "recording_id": recording,
            "scenario": "walk",
            "protocol": "holdout",
            "topology": "point",
            "requested_fraction": 0.2,
            "realized_fraction": 0.18 + 0.01 * seed,
            "model": model,
            "metric": "rmse",
            "value": value,
            "checkpoint_sha256": "a" * 64,
        }
        for recording, models in values[seed].items()
        for model, value in models.items()
    ]


def test_paired_summary_uses_recordings_not_seed_rows_and_is_deterministic():
    statistics = importlib.import_module("validation_v2.evaluation.statistics")
    frame = pd.DataFrame(_rows(1, "run-1") + _rows(2, "run-2"))

    first = statistics.paired_model_summary(
        frame, baseline="baseline", bootstrap_seed=17, bootstrap_samples=1000
    )
    second = statistics.paired_model_summary(
        frame, baseline="baseline", bootstrap_seed=17, bootstrap_samples=1000
    )
    pd.testing.assert_frame_equal(first, second)
    candidate = first.loc[first["model"] == "pinn_imu"].iloc[0]

    assert candidate["mean"] == pytest.approx(0.85)
    assert candidate["sd"] == pytest.approx(0.3535533905932738)
    assert candidate["median"] == pytest.approx(0.85)
    assert candidate["iqr"] == pytest.approx(0.25)
    assert candidate["mean_difference"] == pytest.approx(-0.75)
    assert candidate["ci95_low"] <= -0.75 <= candidate["ci95_high"]
    assert candidate["rank_biserial"] == pytest.approx(-1.0)
    assert candidate["n_recordings"] == 2
    assert candidate["n_seeds"] == 2


def test_summarize_runs_writes_sorted_deterministic_csv_and_json(tmp_path: Path):
    _write_run(tmp_path, "run-2", 2, _rows(2, "run-2"))
    _write_run(tmp_path, "run-1", 1, _rows(1, "run-1"))

    summary = summarize_runs(
        tmp_path,
        required_seeds=(1, 2),
        baseline="baseline",
        bootstrap_seed=11,
        bootstrap_samples=500,
    )

    assert list(summary["model"]) == ["baseline", "pinn_imu"]
    assert (tmp_path / "summary.csv").read_text(encoding="utf-8").startswith("scenario,")
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))[1]["model"] == "pinn_imu"
    assert summary.loc[0, "realized_fraction_min"] == pytest.approx(0.19)
    assert summary.loc[0, "realized_fraction_max"] == pytest.approx(0.20)


def test_summarize_runs_pools_strict_file_recordings_across_scenarios(tmp_path: Path):
    for seed, run_name in ((1, "run-1"), (2, "run-2")):
        rows = _rows(seed, run_name)
        for row in rows:
            row["protocol"] = "strict_file"
            row["scenario"] = "walk" if row["recording_id"] == "rec-a" else "run"
        _write_run(tmp_path, run_name, seed, rows)

    summary = summarize_runs(
        tmp_path,
        required_seeds=(1, 2),
        baseline="baseline",
        bootstrap_seed=11,
        bootstrap_samples=500,
    )

    assert set(summary["scenario"]) == {"overall"}
    assert set(summary["n_recordings"]) == {2}


def test_summarize_runs_rejects_extra_seed(tmp_path: Path):
    _write_run(tmp_path, "run-1", 1, _rows(1, "run-1"))
    with pytest.raises(ValueError, match="unexpected seeds: 1"):
        summarize_runs(tmp_path, required_seeds=(2,))


def test_summarize_runs_rejects_nonfinite_value(tmp_path: Path):
    rows = _rows(1, "run-1")
    rows[0]["value"] = "nan"
    _write_run(tmp_path, "run-1", 1, rows)
    with pytest.raises(ValueError, match="value must be finite"):
        summarize_runs(tmp_path, required_seeds=(1,))


def test_summarize_runs_rejects_group_missing_an_entire_required_seed(tmp_path: Path):
    seed_one = _rows(1, "run-1")
    seed_two = _rows(2, "run-2")
    for row in seed_two:
        row["scenario"] = "run"
    _write_run(tmp_path, "run-1", 1, seed_one)
    _write_run(tmp_path, "run-2", 2, seed_two)

    with pytest.raises(ValueError, match="seed-record-model matrix has missing cells"):
        summarize_runs(tmp_path, required_seeds=(1, 2))


def test_paired_summary_rejects_missing_seed_record_model_cell():
    statistics = importlib.import_module("validation_v2.evaluation.statistics")
    frame = pd.DataFrame(_rows(1, "run-1") + _rows(2, "run-2")).iloc[:-1]
    with pytest.raises(ValueError, match="seed-record-model matrix has missing cells"):
        statistics.paired_model_summary(frame, baseline="baseline")


def test_paired_summary_rejects_duplicate_comparison_key():
    statistics = importlib.import_module("validation_v2.evaluation.statistics")
    frame = pd.DataFrame(_rows(1, "run-1") + _rows(2, "run-2"))
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate per-record metric"):
        statistics.paired_model_summary(frame, baseline="baseline")


def test_paired_summary_requires_two_recordings_for_bootstrap_ci():
    statistics = importlib.import_module("validation_v2.evaluation.statistics")
    frame = pd.DataFrame([row for row in _rows(1, "run-1") if row["recording_id"] == "rec-a"])
    with pytest.raises(ValueError, match="at least 2 recordings"):
        statistics.paired_model_summary(frame, baseline="baseline")


def test_summarize_runs_reports_missing_required_seed(tmp_path: Path):
    with pytest.raises(ValueError, match="missing required seeds: 2027"):
        summarize_runs(tmp_path, required_seeds=(2027,))
