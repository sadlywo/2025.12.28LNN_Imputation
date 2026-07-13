"""Executable contracts for the Python 3.12 server validation runner."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run_validation_v2_server.sh"
HELPERS = REPO_ROOT / "scripts" / "lib" / "validation_v2_server_helpers.sh"
METRICS = REPO_ROOT / "scripts" / "collect_validation_v2_stage_metrics.py"


def _bash() -> str:
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if git_bash.is_file():
        return str(git_bash)
    bash = shutil.which("bash")
    assert bash is not None, "Bash is required to test the server runner"
    return bash


def _run_runner(*arguments: str, environment=None) -> subprocess.CompletedProcess:
    variables = os.environ.copy()
    if environment:
        variables.update(environment)
    variables["MSYS2_ARG_CONV_EXCL"] = "*"
    return subprocess.run(
        [_bash(), RUNNER.as_posix(), *arguments],
        cwd=REPO_ROOT,
        env=variables,
        capture_output=True,
        check=False,
        text=True,
    )


def _make_fake_python(tmp_path: Path, version: str) -> tuple[Path, Path]:
    log = tmp_path / "fake-python.log"
    executable = tmp_path / "python3"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "for argument in \"$@\"; do\n"
        "  printf '<%s>\\n' \"$argument\" >> \"$FAKE_PYTHON_LOG\"\n"
        "done\n"
        "if [ \"${1:-}\" = \"--version\" ]; then\n"
        f"  printf '%s\\n' 'Python {version}'\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' 'unexpected fake Python invocation' >&2\n"
        "exit 97\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, log


def _make_clean_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    shutil.copytree(REPO_ROOT / "configs", repository / "configs")
    shutil.copy2(
        REPO_ROOT / "requirements-validation-v2.txt",
        repository / "requirements-validation-v2.txt",
    )
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Runner test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "--quiet", "-m", "test"],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    return repository, commit


def test_runner_help_declares_full_and_preflight_modes() -> None:
    completed = _run_runner("--help")

    assert completed.returncode == 0, completed.stderr
    assert "--mode preflight|full" in completed.stdout
    assert "--commit COMMIT" in completed.stdout
    assert "--skip-dependency-install" in completed.stdout


def test_runner_rejects_non_python312_before_installation(tmp_path: Path) -> None:
    fake_python, log = _make_fake_python(tmp_path, "3.11.9")
    repository, commit = _make_clean_repository(tmp_path)
    completed = _run_runner(
        "--commit",
        commit,
        "--mode",
        "preflight",
        "--repo",
        repository.as_posix(),
        environment={
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        },
    )

    assert completed.returncode == 2
    assert "Python 3.12" in completed.stderr
    invocations = log.read_text(encoding="utf-8") if log.exists() else ""
    assert invocations.splitlines() == ["<--version>"]
    assert not (repository / ".venv-server").exists()


def test_runner_uses_local_venv_and_explicit_cuda121_torch_index() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert ".venv-server" in source
    assert "PYTHON3_BIN" in source
    assert "-m venv" in source
    assert "https://download.pytorch.org/whl/cu121" in source
    assert "torch==2.3.1" in source
    assert 'torch.__version__ == "2.3.1+cu121"' in source
    assert "/root/miniconda3" not in source
    assert "conda activate" not in source
    assert "pinn_imu" not in source


def test_runner_full_mode_calls_formal_workflow() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert 'run_formal_campaign "$MODE"' in source
    for token in (
        "test_linux_rename_noreplace_survives_real_directory_race",
        "-m pytest -q",
        "validation_v2.cli shard-plan",
        "validation_v2.cli merge-shards",
        "validation_v2.experiments.validate_artifacts",
        "validation_v2.cli summarize",
    ):
        assert token in source


def _write_completed_stage(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    shards_root = tmp_path / "shards"
    run_root = shards_root / "000" / "group-000"
    run_root.mkdir(parents=True)
    (shards_root / "000" / "shard_execution.json").write_text(
        '{"status":"started","started_at":"2026-01-01T00:00:00+00:00",'
        '"group_runs":[{"run_ids":["group-000"]}]}\n',
        encoding="utf-8",
    )
    (run_root / "test_evaluation.json").write_text(
        '{"completed_at":"2026-01-01T00:00:10+00:00"}\n', encoding="utf-8"
    )
    gpu_csv = tmp_path / "gpu.csv"
    gpu_csv.write_text(
        "timestamp_utc,memory_used_mib,memory_total_mib,utilization_percent\n"
        "2026-01-01T00:00:10+00:00,900,1000,90\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        '{"groups_per_hour":1000000,"median_group_seconds":1}\n', encoding="utf-8"
    )
    return shards_root, gpu_csv, baseline, tmp_path / "metrics.json"


def _run_metrics(*arguments: str, optimized: bool = False) -> subprocess.CompletedProcess:
    command = [sys.executable]
    if optimized:
        command.append("-O")
    command.extend([str(METRICS), *arguments])
    return subprocess.run(command, capture_output=True, check=False, text=True)


def test_stage_metrics_fails_closed_under_optimized_python_after_writing_gate(
    tmp_path: Path,
) -> None:
    shards_root, gpu_csv, baseline, output = _write_completed_stage(tmp_path)

    completed = _run_metrics(
        "--shards-root", str(shards_root), "--indices", "000",
        "--stage-start", "2026-01-01T00:00:00+00:00",
        "--gpu-csv", str(gpu_csv), "--minimum-groups", "1",
        "--baseline", str(baseline), "--output", str(output), optimized=True,
    )

    assert completed.returncode == 10, completed.stderr
    assert '"gate_passed": false' in output.read_text(encoding="utf-8")


def test_stage_metrics_rejects_an_existing_output_path(tmp_path: Path) -> None:
    shards_root, gpu_csv, _, output = _write_completed_stage(tmp_path)
    output.write_text("do not replace\n", encoding="utf-8")

    completed = _run_metrics(
        "--shards-root", str(shards_root), "--indices", "000",
        "--stage-start", "2026-01-01T00:00:00+00:00",
        "--gpu-csv", str(gpu_csv), "--minimum-groups", "1", "--output", str(output),
    )

    assert completed.returncode == 3
    assert output.read_text(encoding="utf-8") == "do not replace\n"


def test_stage_metrics_rejects_a_linked_output_path(tmp_path: Path) -> None:
    shards_root, gpu_csv, _, output = _write_completed_stage(tmp_path)
    linked_output = tmp_path / "linked-metrics.json"
    try:
        linked_output.symlink_to(output)
    except OSError as error:
        pytest.skip("symbolic links are unavailable: {}".format(error))

    completed = _run_metrics(
        "--shards-root", str(shards_root), "--indices", "000",
        "--stage-start", "2026-01-01T00:00:00+00:00",
        "--gpu-csv", str(gpu_csv), "--minimum-groups", "1",
        "--output", str(linked_output),
    )

    assert completed.returncode == 3
    assert not output.exists()


def test_stage_metrics_ignores_pre_stage_gpu_samples(tmp_path: Path) -> None:
    shards_root, gpu_csv, _, output = _write_completed_stage(tmp_path)
    gpu_csv.write_text(
        "timestamp_utc,memory_used_mib,memory_total_mib,utilization_percent\n"
        "2025-12-31T23:59:59+00:00,10,1000,1\n",
        encoding="utf-8",
    )

    completed = _run_metrics(
        "--shards-root", str(shards_root), "--indices", "000",
        "--stage-start", "2026-01-01T00:00:00+00:00",
        "--gpu-csv", str(gpu_csv), "--minimum-groups", "1", "--output", str(output),
    )

    assert completed.returncode == 4
    assert not output.exists()


def test_helpers_have_pipefail_and_atomic_shard_reservation() -> None:
    source = HELPERS.read_text(encoding="utf-8")

    assert "set -o pipefail" in source
    assert "mkdir \"$reservation\"" in source
    assert '"$PYTHON_BIN" -m validation_v2.cli shard' in source
    assert "SAMPLER_READY_MAX_SECONDS" in source
    assert "validation_v2_sampler_has_data" in source
    assert 'validation_v2_verify_sampler "$label" || return $?' in source
