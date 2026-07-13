"""Executable contracts for the Python 3.12 server validation runner."""

import os
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

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
    variables.setdefault("MSYS2_ARG_CONV_EXCL", "*")
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
    shutil.copytree(REPO_ROOT / "scripts" / "lib", repository / "scripts" / "lib")
    shutil.copy2(
        REPO_ROOT / "requirements-validation-v2.txt",
        repository / "requirements-validation-v2.txt",
    )
    shutil.copy2(REPO_ROOT / ".gitignore", repository / ".gitignore")
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


def _runner_python_heredoc(name: str) -> str:
    source = RUNNER.read_text(encoding="utf-8")
    match = re.search(
        rf"{re.escape(name)}\(\) \{{.*?<<'PY'\n(.*?)\nPY",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def test_runner_runtime_and_plan_checks_survive_optimized_python(tmp_path: Path) -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "assert " not in source

    environment = tmp_path / "environment.json"
    runtime = subprocess.run(
        [sys.executable, "-O", "-", str(environment)],
        input=_runner_python_heredoc("verify_runtime"),
        capture_output=True,
        check=False,
        text=True,
    )
    assert runtime.returncode != 0
    assert not environment.exists()

    plan = tmp_path / "invalid-plan.json"
    plan.write_text("{}\n", encoding="utf-8")
    plan_check = source.split('"$PYTHON_BIN" - "$PLAN" <<\'PY\'\n', 1)[1].split(
        "\nPY", 1
    )[0]
    completed = subprocess.run(
        [sys.executable, "-O", "-", str(plan)],
        input=plan_check,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode != 0


def test_runner_ignores_its_venv_but_rejects_other_untracked_files(tmp_path: Path) -> None:
    repository, commit = _make_clean_repository(tmp_path)
    (repository / ".venv-server").mkdir()
    (repository / ".venv-server" / "sentinel").write_text("ignored\n", encoding="utf-8")
    fake_python, log = _make_fake_python(tmp_path, "3.11.9")
    ignored = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment={
            "MSYS2_ARG_CONV_EXCL": "",
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        },
    )
    assert ignored.returncode == 2
    assert "Python 3.12" in ignored.stderr
    assert "dirty" not in ignored.stderr.lower()
    assert log.read_text(encoding="utf-8").splitlines() == ["<--version>"]
    assert subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ) == ""

    (repository / "foreign-sentinel").write_text("must fail\n", encoding="utf-8")
    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    valid_python, valid_log = _make_fake_python(valid_dir, "3.12.3")
    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment={
            "MSYS2_ARG_CONV_EXCL": "",
            "PYTHON3_BIN": valid_python.as_posix(),
            "FAKE_PYTHON_LOG": valid_log.as_posix(),
        },
    )
    assert completed.returncode == 2
    assert "Git worktree must be clean" in completed.stderr
    assert valid_log.read_text(encoding="utf-8").splitlines() == ["<--version>"]


def test_runner_rejects_a_preexisting_audit_seal_before_venv_or_other_writes(
    tmp_path: Path,
) -> None:
    repository, commit = _make_clean_repository(tmp_path)
    audit_dir = tmp_path / f"validation-v2-audit-{commit}-sharded-v2-py312"
    audit_dir.mkdir()
    fake_python, log = _make_fake_python(tmp_path, "3.12.3")
    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment={
            "MSYS2_ARG_CONV_EXCL": "",
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        },
    )
    assert completed.returncode == 2
    assert "AUDIT_DIR" in completed.stderr
    assert log.read_text(encoding="utf-8").splitlines() == ["<--version>"]
    assert audit_dir.is_dir()
    assert not (repository / ".venv-server").exists()
    assert not (tmp_path / f"validation-v2-preflight-{commit}-sharded-v2-py312").exists()
    assert not (
        repository / "results" / "validation_v2" / f"server-full-shards-{commit}-sharded-v2-py312"
    ).exists()


def test_runner_rejects_a_linked_audit_seal_before_venv(tmp_path: Path) -> None:
    repository, commit = _make_clean_repository(tmp_path)
    audit_dir = tmp_path / f"validation-v2-audit-{commit}-sharded-v2-py312"
    target = tmp_path / "linked-target"
    target.mkdir()
    try:
        audit_dir.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip("symbolic links are unavailable: {}".format(error))
    fake_python, log = _make_fake_python(tmp_path, "3.12.3")

    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment={
            "MSYS2_ARG_CONV_EXCL": "",
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        },
    )
    assert completed.returncode == 2
    assert "AUDIT_DIR" in completed.stderr
    assert audit_dir.is_symlink()
    assert not (repository / ".venv-server").exists()


def test_runner_preserves_a_new_audit_seal_and_exit_note_after_a_later_failure(
    tmp_path: Path,
) -> None:
    repository, commit = _make_clean_repository(tmp_path)
    fake_python, log = _make_fake_python(tmp_path, "3.12.3")
    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment={
            "MSYS2_ARG_CONV_EXCL": "",
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        },
    )

    assert completed.returncode == 2
    assert "requires Linux" in completed.stderr
    audit_dir = tmp_path / f"validation-v2-audit-{commit}-sharded-v2-py312"
    assert audit_dir.is_dir()
    assert "status=2" in (audit_dir / "runner-exit-note.txt").read_text(encoding="utf-8")
    assert not (repository / ".venv-server").exists()


def _prepare_linux_runner_environment(tmp_path: Path, repository: Path) -> dict[str, str]:
    fake_python, log = _make_fake_python(tmp_path, "3.12.3")
    venv_python = repository / ".venv-server" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    _write_bash_executable(venv_python, "exit 97\n")
    bash_env = tmp_path / "bash-env"
    bash_env.write_text("uname() { printf '%s\\n' Linux; }\n", encoding="utf-8")
    return {
        "MSYS2_ARG_CONV_EXCL": "",
        "BASH_ENV": bash_env.as_posix(),
        "PYTHON3_BIN": fake_python.as_posix(),
        "FAKE_PYTHON_LOG": log.as_posix(),
    }


def test_runner_creates_real_shard_output_parents_before_campaign_leaf(tmp_path: Path) -> None:
    repository, commit = _make_clean_repository(tmp_path)
    environment = _prepare_linux_runner_environment(tmp_path, repository)
    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        "--skip-dependency-install", environment=environment,
    )

    assert completed.returncode == 97
    results = repository / "results"
    validation_root = results / "validation_v2"
    shard_root = validation_root / f"server-full-shards-{commit}-sharded-v2-py312"
    assert results.is_dir() and not results.is_symlink()
    assert validation_root.is_dir() and not validation_root.is_symlink()
    assert shard_root.is_dir() and not shard_root.is_symlink()


def test_runner_rejects_a_linked_shard_output_parent(tmp_path: Path) -> None:
    repository, commit = _make_clean_repository(tmp_path)
    target = tmp_path / "linked-results-target"
    target.mkdir()
    try:
        (repository / "results").symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip("symbolic links are unavailable: {}".format(error))
    (repository / ".git" / "info" / "exclude").write_text("results/\n", encoding="utf-8")
    environment = _prepare_linux_runner_environment(tmp_path, repository)

    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        "--skip-dependency-install", environment=environment,
    )

    assert completed.returncode == 2
    assert "shard output parent" in completed.stderr
    assert not (target / "validation_v2").exists()


def test_runner_cleanup_stops_recorded_samplers_and_preserves_campaign_seal(
    tmp_path: Path,
) -> None:
    source = RUNNER.read_text(encoding="utf-8")
    match = re.search(
        r"cleanup_runner\(\) \{\n(.*?)\n\}\ntrap cleanup_runner EXIT",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    stops = tmp_path / "stops.txt"
    script = (
        "set -Eeuo pipefail\n"
        "stop_gpu_sampler() { printf '%s\\n' \"$1\" >> \"$STOP_LOG\"; }\n"
        + match.group(0).replace("\ntrap cleanup_runner EXIT", "")
        + "\nAUDIT_DIR=\"$1\"\nSTOP_LOG=\"$2\"\n"
        + "GPU_SAMPLER_LABELS=(baseline stage-2worker)\nSHARDS_LAUNCHED=1\n"
        + "cleanup_runner 17\n"
    )
    completed = subprocess.run(
        [_bash(), "-c", script, "_", _git_bash_path(audit_dir), _git_bash_path(stops)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 17
    assert stops.read_text(encoding="utf-8").splitlines() == ["baseline", "stage-2worker"]
    note = (audit_dir / "runner-exit-note.txt").read_text(encoding="utf-8")
    assert "status=17" in note
    assert "shards_launched=1" in note
    assert "preserved" in note


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


def _git_bash_path(path: Path) -> str:
    resolved = path.resolve().as_posix()
    if len(resolved) >= 3 and resolved[1:3] == ":/":
        return "/{}/{}".format(resolved[0].lower(), resolved[3:])
    return resolved


def _write_bash_executable(path: Path, source: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + source, encoding="utf-8")
    path.chmod(0o755)


def _run_helper_bash(script: str, *arguments: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_bash(), "-c", script, "_", *(_git_bash_path(item) for item in arguments)],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )


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


def test_stage_metrics_removes_publish_reservation_after_success(tmp_path: Path) -> None:
    shards_root, gpu_csv, _, output = _write_completed_stage(tmp_path)
    gpu_csv.write_text(
        "timestamp_utc,memory_used_mib,memory_total_mib,utilization_percent\n"
        "2026-01-01T00:00:10+00:00,100,1000,10\n",
        encoding="utf-8",
    )

    completed = _run_metrics(
        "--shards-root", str(shards_root), "--indices", "000",
        "--stage-start", "2026-01-01T00:00:00+00:00",
        "--gpu-csv", str(gpu_csv), "--minimum-groups", "1", "--output", str(output),
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["gate_passed"] is True
    assert not (tmp_path / ".metrics.json.publish.lock").exists()


def test_audit_active_rejects_an_unexpected_pgrep_error(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_bash_executable(bin_dir / "pgrep", "exit 2\n")
    fake_python = tmp_path / "fake-python"
    _write_bash_executable(fake_python, "exit 97\n")
    audit_dir = tmp_path / "audit"
    shards_root = tmp_path / "shards"
    audit_dir.mkdir()
    shards_root.mkdir()

    completed = _run_helper_bash(
        'export PATH="$1:$PATH" PYTHON_BIN="$2" AUDIT_DIR="$3" SHARDS_ROOT="$4"; '
        'source "$5"; audit_active 000',
        bin_dir, fake_python, audit_dir, shards_root, HELPERS,
    )

    assert completed.returncode == 3
    assert "cannot inspect shard processes: rc=2" in completed.stderr


def test_launch_shard_reserves_ownership_before_a_second_launch(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_bash_executable(bin_dir / "pgrep", "exit 1\n")
    fake_python = tmp_path / "fake-python"
    invocation_log = tmp_path / "python.log"
    _write_bash_executable(
        fake_python,
        'printf "%s\\n" "$*" >> "$FAKE_PYTHON_LOG"\nsleep 30\n',
    )
    audit_dir = tmp_path / "audit"
    shards_root = tmp_path / "shards"
    audit_dir.mkdir()
    shards_root.mkdir()
    config = tmp_path / "config.yaml"
    plan = tmp_path / "plan.json"
    config.write_text("{}\n", encoding="utf-8")
    plan.write_text("{}\n", encoding="utf-8")

    completed = _run_helper_bash(
        'export PATH="$1:$PATH" PYTHON_BIN="$2" AUDIT_DIR="$3" SHARDS_ROOT="$4" '
        'CONFIG="$5" PLAN="$6" FAKE_PYTHON_LOG="$7"; source "$8"; '
        'launch_shard 000; first=$?; launch_shard 000; second=$?; '
        'test "$first" -eq 0; test "$second" -eq 2; '
        'test -d "$AUDIT_DIR/shard-000.ownership.lock"; '
        'test "$(wc -l < "$FAKE_PYTHON_LOG")" -eq 1; '
        'kill "$(cat "$AUDIT_DIR/shard-000.pid")" 2>/dev/null || true; exit 0',
        bin_dir, fake_python, audit_dir, shards_root, config, plan, invocation_log, HELPERS,
    )

    assert completed.returncode == 0, completed.stderr


def test_gpu_sampler_waits_for_a_valid_row_before_returning(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_bash_executable(
        bin_dir / "nvidia-smi",
        'sleep 0.4\nprintf "%s\\n" "10, 1000, 1"\n',
    )
    fake_python = tmp_path / "fake-python"
    _write_bash_executable(
        fake_python,
        'if [ "${1:-}" = "-" ]; then grep -q "10,1000,1" "$2"; exit $?; fi\n'
        "exit 97\n",
    )
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()

    started = time.monotonic()
    completed = _run_helper_bash(
        'export PATH="$1:$PATH" PYTHON_BIN="$2" AUDIT_DIR="$3"; '
        'source "$4"; start_gpu_sampler sample; rc=$?; '
        'test "$rc" -eq 0; test -s "$AUDIT_DIR/gpu-sample.csv"; '
        'stop_gpu_sampler sample; exit 0',
        bin_dir, fake_python, audit_dir, HELPERS,
    )
    elapsed = time.monotonic() - started

    assert completed.returncode == 0, completed.stderr
    assert elapsed >= 0.3


def test_helpers_have_pipefail_and_atomic_shard_reservation() -> None:
    source = HELPERS.read_text(encoding="utf-8")

    assert "set -o pipefail" in source
    assert "mkdir \"$reservation\"" in source
    assert '"$PYTHON_BIN" -m validation_v2.cli shard' in source
    assert "SAMPLER_READY_MAX_SECONDS" in source
    assert "validation_v2_sampler_has_data" in source
    assert 'validation_v2_verify_sampler "$label" || return $?' in source
