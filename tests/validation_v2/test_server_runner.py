"""Executable contracts for the CPython 3.10--3.12 server validation runner."""

from __future__ import annotations

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
    candidates = [Path(r"C:\Program Files\Git\bin\bash.exe")]
    git = shutil.which("git")
    if git:
        candidates.insert(0, Path(git).resolve().parents[1] / "bin" / "bash.exe")
    for git_bash in candidates:
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


def _make_fake_python(
    tmp_path: Path,
    version: str,
    *,
    implementation: str = "CPython",
    venv_available: bool = True,
    ensurepip_available: bool = True,
) -> tuple[Path, Path]:
    log = tmp_path / "fake-python.log"
    executable = tmp_path / "python3"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "for argument in \"$@\"; do\n"
        "  printf '<%s>\\n' \"$argument\" >> \"$FAKE_PYTHON_LOG\"\n"
        "done\n"
        "if [ \"${1:-}\" = \"-c\" ]; then\n"
        f"  printf '%s\\n' '{implementation} {'.'.join(version.split('.')[:2])}'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"${1:-}\" = \"-m\" ] && [ \"${2:-}\" = \"venv\" ] "
        "&& [ \"${3:-}\" = \"--help\" ]; then\n"
        f"  exit {0 if venv_available else 1}\n"
        "fi\n"
        "if [ \"${1:-}\" = \"-m\" ] && [ \"${2:-}\" = \"ensurepip\" ] "
        "&& [ \"${3:-}\" = \"--version\" ]; then\n"
        f"  exit {0 if ensurepip_available else 1}\n"
        "fi\n"
        "printf '%s\\n' 'unexpected fake Python invocation' >&2\n"
        "exit 97\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, log


def _fake_python_probe_log() -> list[str]:
    return [
        "<-c>",
        "<import platform, sys; print(platform.python_implementation(), '{}.{}'.format(*sys.version_info[:2]))>",
        "<-m>",
        "<venv>",
        "<--help>",
        "<-m>",
        "<ensurepip>",
        "<--version>",
    ]


def _make_fake_venv_python(
    repository: Path,
    version: str,
    *,
    implementation: str = "CPython",
) -> tuple[Path, Path]:
    log = repository.parent / "fake-venv-python.log"
    executable = repository / ".venv-server" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "for argument in \"$@\"; do\n"
        "  printf '<%s>\\n' \"$argument\" >> \"$FAKE_VENV_PYTHON_LOG\"\n"
        "done\n"
        "if [ \"${1:-}\" = \"-c\" ]; then\n"
        f"  printf '%s\\n' '{implementation} {'.'.join(version.split('.')[:2])}'\n"
        "  exit 0\n"
        "fi\n"
        "exit 97\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, log


def _make_host_bash_env(
    tmp_path: Path, *, kernel: str = "Linux"
) -> tuple[Path, Path]:
    bash_env = tmp_path / "host-bash-env"
    probe_log = tmp_path / "host-probes.log"
    bash_env.write_text(
        "uname() {\n"
        "  printf '<uname:%s>\\n' \"${1:-}\" >> \"$FAKE_HOST_PROBE_LOG\"\n"
        f"  printf '%s\\n' '{kernel}'\n"
        "}\n"
        "command() {\n"
        "  if [ \"${1:-}\" = \"-v\" ] && [ \"$#\" -eq 2 ]; then\n"
        "    printf '<command-v:%s>\\n' \"$2\" >> \"$FAKE_HOST_PROBE_LOG\"\n"
        "    if [ \"$2\" = \"${FAKE_MISSING_HOST_COMMAND:-}\" ]; then\n"
        "      return 1\n"
        "    fi\n"
        "    printf '/controlled/%s\\n' \"$2\"\n"
        "    return 0\n"
        "  fi\n"
        "  builtin command \"$@\"\n"
        "}\n",
        encoding="utf-8",
    )
    return bash_env, probe_log


def _host_environment(
    tmp_path: Path, *, kernel: str = "Linux", missing_command: str = ""
) -> tuple[dict[str, str], Path]:
    bash_env, probe_log = _make_host_bash_env(tmp_path, kernel=kernel)
    return {
        "MSYS2_ARG_CONV_EXCL": "",
        "BASH_ENV": bash_env.as_posix(),
        "FAKE_HOST_PROBE_LOG": probe_log.as_posix(),
        "FAKE_MISSING_HOST_COMMAND": missing_command,
    }, probe_log


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
    assert "--max-workers 1|2|4|8" in completed.stdout
    assert "default: 2" in completed.stdout
    assert "sharded-v2-py310/py311/py312" in completed.stdout


@pytest.mark.parametrize(
    "value,error",
    [
        (None, "--max-workers requires 1, 2, 4, or 8"),
        ("0", "--max-workers must be one of 1, 2, 4, or 8"),
        ("3", "--max-workers must be one of 1, 2, 4, or 8"),
        ("16", "--max-workers must be one of 1, 2, 4, or 8"),
        ("four", "--max-workers must be one of 1, 2, 4, or 8"),
    ],
)
def test_runner_rejects_invalid_max_workers_before_campaign_writes(
    tmp_path: Path, value: str | None, error: str
) -> None:
    repository, commit = _make_clean_repository(tmp_path)
    arguments = [
        "--commit",
        commit,
        "--mode",
        "preflight",
        "--repo",
        repository.as_posix(),
        "--max-workers",
    ]
    if value is not None:
        arguments.append(value)

    completed = _run_runner(*arguments)

    assert completed.returncode == 2
    assert error in completed.stderr
    assert not list(tmp_path.glob("validation-v2-audit-*"))
    assert not (repository / ".venv-server").exists()


@pytest.mark.parametrize(
    "version,suffix",
    [("3.10.14", "py310"), ("3.11.9", "py311"), ("3.12.3", "py312")],
)
def test_runner_accepts_supported_cpython_and_uses_dynamic_default_suffix(
    tmp_path: Path, version: str, suffix: str
) -> None:
    case = tmp_path / suffix
    case.mkdir()
    fake_python, log = _make_fake_python(case, version)
    repository, commit = _make_clean_repository(case)
    environment, _ = _host_environment(case)
    environment.update(
        {
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        }
    )

    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment=environment,
    )

    assert completed.returncode == 97
    assert (case / f"validation-v2-audit-{commit}-sharded-v2-{suffix}").is_dir()
    assert log.read_text(encoding="utf-8").splitlines()[:8] == _fake_python_probe_log()


@pytest.mark.parametrize("version", ["3.9.19", "3.13.0"])
def test_runner_rejects_unsupported_cpython_before_any_write(
    tmp_path: Path, version: str
) -> None:
    fake_python, log = _make_fake_python(tmp_path, version)
    repository, commit = _make_clean_repository(tmp_path)

    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment={
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        },
    )

    assert completed.returncode == 2
    assert "CPython 3.10, 3.11, or 3.12 is required" in completed.stderr
    assert log.read_text(encoding="utf-8").splitlines() == _fake_python_probe_log()[:2]
    assert not (repository / ".venv-server").exists()
    assert not list(tmp_path.glob("validation-v2-audit-*"))


def test_runner_rejects_non_cpython_before_any_write(tmp_path: Path) -> None:
    fake_python, log = _make_fake_python(tmp_path, "3.11.9", implementation="PyPy")
    repository, commit = _make_clean_repository(tmp_path)

    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment={
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        },
    )

    assert completed.returncode == 2
    assert "CPython 3.10, 3.11, or 3.12 is required" in completed.stderr
    assert not (repository / ".venv-server").exists()
    assert not list(tmp_path.glob("validation-v2-audit-*"))


def test_runner_probes_venv_before_any_write_and_names_matching_package(
    tmp_path: Path,
) -> None:
    fake_python, log = _make_fake_python(tmp_path, "3.11.9", venv_available=False)
    repository, commit = _make_clean_repository(tmp_path)

    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment={
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        },
    )

    assert completed.returncode == 2
    assert "python3.11-venv" in completed.stderr
    assert log.read_text(encoding="utf-8").splitlines()[-3:] == [
        "<-m>", "<venv>", "<--help>",
    ]
    assert not (repository / ".venv-server").exists()
    assert not list(tmp_path.glob("validation-v2-audit-*"))


def test_runner_preserves_explicit_campaign_suffix(tmp_path: Path) -> None:
    fake_python, _ = _make_fake_python(tmp_path, "3.10.14")
    repository, commit = _make_clean_repository(tmp_path)
    explicit = "operator-selected"
    environment, _ = _host_environment(tmp_path)
    environment.update(
        {
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": (tmp_path / "log").as_posix(),
        }
    )

    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        "--campaign-suffix", explicit,
        environment=environment,
    )

    assert completed.returncode == 97
    assert (tmp_path / f"validation-v2-audit-{commit}-{explicit}").is_dir()


def test_runner_rejects_an_explicit_empty_campaign_suffix_before_campaign_writes(
    tmp_path: Path,
) -> None:
    fake_python, log = _make_fake_python(tmp_path, "3.11.9")
    repository, commit = _make_clean_repository(tmp_path)
    environment, _ = _host_environment(tmp_path)
    environment.update(
        {
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        }
    )

    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        "--campaign-suffix", "",
        environment=environment,
    )

    assert completed.returncode == 2
    assert "invalid --campaign-suffix" in completed.stderr
    assert log.read_text(encoding="utf-8").splitlines() == _fake_python_probe_log()
    assert not (repository / ".venv-server").exists()
    assert not list(tmp_path.glob("validation-v2-audit-*"))


def test_runner_rejects_missing_ensurepip_before_audit_or_venv_writes(
    tmp_path: Path,
) -> None:
    fake_python, log = _make_fake_python(
        tmp_path, "3.11.9", ensurepip_available=False
    )
    repository, commit = _make_clean_repository(tmp_path)

    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment={
            "MSYS2_ARG_CONV_EXCL": "",
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        },
    )

    assert completed.returncode == 2
    assert "ensurepip" in completed.stderr
    assert "python3.11-venv" in completed.stderr
    assert log.read_text(encoding="utf-8").splitlines() == _fake_python_probe_log()
    assert not (repository / ".venv-server").exists()
    assert not list(tmp_path.glob("validation-v2-audit-*"))


def test_runner_uses_existing_venv_minor_for_the_dynamic_default_suffix(
    tmp_path: Path,
) -> None:
    fake_python, system_log = _make_fake_python(tmp_path, "3.10.14")
    repository, commit = _make_clean_repository(tmp_path)
    _, venv_log = _make_fake_venv_python(repository, "3.12.3")
    environment, _ = _host_environment(tmp_path)
    environment.update(
        {
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": system_log.as_posix(),
            "FAKE_VENV_PYTHON_LOG": venv_log.as_posix(),
        }
    )

    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        "--skip-dependency-install", environment=environment,
    )

    assert completed.returncode == 97
    assert (tmp_path / f"validation-v2-audit-{commit}-sharded-v2-py312").is_dir()
    assert not (tmp_path / f"validation-v2-audit-{commit}-sharded-v2-py310").exists()
    assert venv_log.read_text(encoding="utf-8").splitlines()[:2] == [
        "<-c>",
        "<import platform, sys; print(platform.python_implementation(), '{}.{}'.format(*sys.version_info[:2]))>",
    ]


@pytest.mark.parametrize(
    "implementation,version",
    [("CPython", "3.9.19"), ("CPython", "3.13.0"), ("PyPy", "3.11.9")],
)
def test_runner_rejects_an_unsupported_existing_venv_before_campaign_writes(
    tmp_path: Path, implementation: str, version: str
) -> None:
    fake_python, system_log = _make_fake_python(tmp_path, "3.10.14")
    repository, commit = _make_clean_repository(tmp_path)
    _, venv_log = _make_fake_venv_python(
        repository, version, implementation=implementation
    )
    environment, _ = _host_environment(tmp_path)
    environment.update(
        {
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": system_log.as_posix(),
            "FAKE_VENV_PYTHON_LOG": venv_log.as_posix(),
        }
    )

    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment=environment,
    )

    assert completed.returncode == 2
    assert "existing .venv-server must use CPython 3.10, 3.11, or 3.12" in completed.stderr
    assert not list(tmp_path.glob("validation-v2-audit-*"))


def test_runner_explicit_suffix_is_authoritative_with_an_existing_venv(
    tmp_path: Path,
) -> None:
    fake_python, system_log = _make_fake_python(tmp_path, "3.10.14")
    repository, commit = _make_clean_repository(tmp_path)
    _, venv_log = _make_fake_venv_python(repository, "3.12.3")
    environment, _ = _host_environment(tmp_path)
    environment.update(
        {
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": system_log.as_posix(),
            "FAKE_VENV_PYTHON_LOG": venv_log.as_posix(),
        }
    )

    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        "--campaign-suffix", "chosen",
        "--skip-dependency-install", environment=environment,
    )

    assert completed.returncode == 97
    assert (tmp_path / f"validation-v2-audit-{commit}-chosen").is_dir()
    assert not list(tmp_path.glob(f"validation-v2-audit-{commit}-sharded-v2-*"))


def test_runner_skip_dependency_install_requires_an_existing_venv(tmp_path: Path) -> None:
    repository, commit = _make_clean_repository(tmp_path)
    fake_python, log = _make_fake_python(tmp_path, "3.12.3")
    environment, _ = _host_environment(tmp_path)
    environment.update(
        {
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        }
    )
    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        "--skip-dependency-install", environment=environment,
    )

    assert completed.returncode == 2
    assert "existing .venv-server" in completed.stderr
    assert not (repository / ".venv-server").exists()
    assert log.read_text(encoding="utf-8").splitlines() == _fake_python_probe_log()


def test_runner_uses_local_venv_and_explicit_cuda128_torch_index() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert ".venv-server" in source
    assert "PYTHON3_BIN" in source
    assert "-m venv" in source
    assert "https://download.pytorch.org/whl/cu128" in source
    assert "torch==2.11.0" in source
    assert 'torch.__version__ == "2.11.0+cu128"' in source
    assert "/root/miniconda3" not in source
    assert "conda activate" not in source
    assert "pinn_imu" not in source


def test_runner_full_mode_calls_formal_workflow() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "MAX_WORKERS=2" in source
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


def _formal_campaign_function() -> str:
    source = RUNNER.read_text(encoding="utf-8")
    match = re.search(
        r"run_formal_campaign\(\) \{\n(.*?)\n\}\n\ncd \"\$REPO\"",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return "run_formal_campaign() {\n" + match.group(1) + "\n}\n"


def _run_formal_campaign_contract(
    tmp_path: Path,
    max_workers: int,
    *,
    direct_parallel: bool = False,
    stage2_rc: int = 0,
    stage4_rc: int = 0,
    stage8_rc: int = 0,
    expected_rc: int = 0,
) -> list[str]:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    log = tmp_path / "schedule.log"
    script = (
        "set -Eeuo pipefail\n"
        "log() { printf '%s\\n' \"$*\" >> \"$SCHEDULE_LOG\"; }\n"
        "start_managed_sampler() { log start \"$1\"; }\n"
        "stop_managed_sampler() { log stop \"$1\"; }\n"
        "launch_formal_shard() { log launch \"$1\"; }\n"
        "wait_until_groups() { :; }\n"
        "wait_stage_metrics() {\n"
        "  log gate \"$1\"\n"
        "  case \"$1\" in\n"
        "    stage-2worker) return \"$STAGE2_RC\" ;;\n"
        "    stage-4worker) return \"$STAGE4_RC\" ;;\n"
        "    stage-8worker) return \"$STAGE8_RC\" ;;\n"
        "    *) return 0 ;;\n"
        "  esac\n"
        "}\n"
        "wait_shard() { log wait \"$1\"; }\n"
        "wait_all_shards() { log wait-all \"$@\"; }\n"
        "run_queue() { log queue \"$@\"; }\n"
        "fake_python() {\n"
        "  if [ \"${1:-}\" = '-c' ]; then\n"
        "    printf '%s\\n' '2026-01-01T00:00:00+00:00'\n"
        "  elif [ \"${1:-}\" = '-' ]; then\n"
        "    cat >/dev/null\n"
        "  fi\n"
        "}\n"
        + _formal_campaign_function()
        + "run_formal_campaign full\n"
    )
    completed = subprocess.run(
        [_bash(), "-c", script],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "AUDIT_DIR": _git_bash_path(audit_dir),
            "SHARDS_ROOT": _git_bash_path(tmp_path / "shards"),
            "FINAL_ROOT": _git_bash_path(tmp_path / "final"),
            "SCHEDULE_LOG": _git_bash_path(log),
            "PYTHON_BIN": "fake_python",
            "CONFIG": "config.yaml",
            "PLAN": "plan.json",
            "MAX_WORKERS": str(max_workers),
            "VALIDATION_V2_DIRECT_PARALLEL": "1" if direct_parallel else "0",
            "SHARDS_LAUNCHED": "0",
            "STAGE2_RC": str(stage2_rc),
            "STAGE4_RC": str(stage4_rc),
            "STAGE8_RC": str(stage8_rc),
        },
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == expected_rc, completed.stderr
    return log.read_text(encoding="utf-8").splitlines()


def test_formal_campaign_can_start_all_matpool_workers_immediately(
    tmp_path: Path,
) -> None:
    events = _run_formal_campaign_contract(
        tmp_path,
        2,
        direct_parallel=True,
    )

    assert events == [
        "start direct-parallel",
        "queue 2 000 001 002 003 004 005 006 007",
        "stop direct-parallel",
        "wait-all 000 001 002 003 004 005 006 007",
    ]


@pytest.mark.parametrize(
    "max_workers,stage2_rc,stage4_rc,stage8_rc,expected_stages,expected_queue",
    [
        (1, 0, 0, 0, [], "queue 1 001 002 003 004 005 006 007"),
        (2, 0, 0, 0, ["stage-2worker"], "queue 2 002 003 004 005 006 007"),
        (2, 10, 0, 0, ["stage-2worker"], "queue 1 002 003 004 005 006 007"),
        (
            4,
            0,
            0,
            0,
            ["stage-2worker", "stage-4worker"],
            "queue 4 004 005 006 007",
        ),
        (4, 10, 0, 0, ["stage-2worker"], "queue 1 002 003 004 005 006 007"),
        (
            4,
            0,
            10,
            0,
            ["stage-2worker", "stage-4worker"],
            "queue 2 004 005 006 007",
        ),
        (8, 10, 0, 0, ["stage-2worker"], "queue 1 002 003 004 005 006 007"),
        (
            8,
            0,
            10,
            0,
            ["stage-2worker", "stage-4worker"],
            "queue 2 004 005 006 007",
        ),
        (8, 0, 0, 0, ["stage-2worker", "stage-4worker", "stage-8worker"], None),
        (8, 0, 0, 10, ["stage-2worker", "stage-4worker", "stage-8worker"], None),
    ],
)
def test_formal_campaign_respects_worker_ceiling_and_schedules_all_shards(
    tmp_path: Path,
    max_workers: int,
    stage2_rc: int,
    stage4_rc: int,
    stage8_rc: int,
    expected_stages: list[str],
    expected_queue: str | None,
) -> None:
    events = _run_formal_campaign_contract(
        tmp_path,
        max_workers,
        stage2_rc=stage2_rc,
        stage4_rc=stage4_rc,
        stage8_rc=stage8_rc,
    )

    actual_stages = [
        line.removeprefix("start ")
        for line in events
        if line.startswith("start stage-")
    ]
    assert actual_stages == expected_stages
    queues = [line for line in events if line.startswith("queue ")]
    assert queues == ([] if expected_queue is None else [expected_queue])
    scheduled = [line.split()[1] for line in events if line.startswith("launch ")]
    for queue in queues:
        scheduled.extend(queue.split()[2:])
    assert scheduled == [f"{index:03d}" for index in range(8)]
    assert events.count("wait-all 000 001 002 003 004 005 006 007") == 1
    if expected_queue is not None:
        queue_index = events.index(expected_queue)
        if expected_queue == "queue 1 001 002 003 004 005 006 007":
            assert events.index("wait 000") < queue_index
        elif expected_queue.endswith("002 003 004 005 006 007"):
            assert events.index("wait 001") < queue_index
        elif expected_queue == "queue 4 004 005 006 007":
            assert events.index("wait-all 000 001 002 003") < queue_index
        else:
            assert events.index("wait 003") < queue_index


@pytest.mark.parametrize(
    "max_workers,stage2_rc,stage4_rc", [(2, 2, 0), (4, 3, 0), (8, 0, 4)]
)
def test_formal_campaign_preserves_gate_error_statuses(
    tmp_path: Path, max_workers: int, stage2_rc: int, stage4_rc: int
) -> None:
    expected_rc = stage2_rc or stage4_rc

    events = _run_formal_campaign_contract(
        tmp_path,
        max_workers,
        stage2_rc=stage2_rc,
        stage4_rc=stage4_rc,
        expected_rc=expected_rc,
    )

    assert not any(line.startswith("queue ") for line in events)
    assert not any(line.startswith("wait-all ") for line in events)


def _runner_python_heredoc(name: str) -> str:
    source = RUNNER.read_text(encoding="utf-8")
    match = re.search(
        rf"{re.escape(name)}\(\) \{{.*?<<'PY'\n(.*?)\nPY",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _run_fake_runtime(
    tmp_path: Path,
    *,
    gpu_name: str,
    gpu_memory_bytes: int = 24 * 1024**3,
    gpu_count: int = 1,
    compute_capability: tuple[int, int] = (8, 9),
    python_implementation: str = "CPython",
    python_version: tuple[int, int] = (3, 12),
    torch_version: str = "2.11.0+cu128",
    torch_cuda: str = "12.8",
    cuda_available: bool = True,
    nvidia_smi_available: bool = True,
    nvidia_smi_returncode: int = 0,
    nvidia_smi_output: tuple[str, ...] = ("555.42",),
    output: Path | None = None,
) -> tuple[subprocess.CompletedProcess, Path, Path]:
    fake_modules = tmp_path / "fake-modules"
    fake_modules.mkdir()
    (fake_modules / "torch.py").write_text(
        "import json\n"
        "import os\n"
        "from types import SimpleNamespace\n"
        "__version__ = os.environ['FAKE_TORCH_VERSION']\n"
        "version = SimpleNamespace(cuda=os.environ['FAKE_TORCH_CUDA'])\n"
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def is_available(): return os.environ['FAKE_CUDA_AVAILABLE'] == '1'\n"
        "    @staticmethod\n"
        "    def device_count(): return int(os.environ['FAKE_GPU_COUNT'])\n"
        "    @staticmethod\n"
        "    def get_device_name(index):\n"
        "        return json.loads(os.environ['FAKE_GPU_NAMES'])[index]\n"
        "    @staticmethod\n"
        "    def get_device_properties(index):\n"
        "        return SimpleNamespace(total_memory=int(os.environ['FAKE_GPU_MEMORY']))\n"
        "    @staticmethod\n"
        "    def get_device_capability(index):\n"
        "        return tuple(json.loads(os.environ['FAKE_COMPUTE_CAPABILITY']))\n"
        "cuda = _Cuda()\n",
        encoding="utf-8",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvidia_log = tmp_path / "nvidia-smi.log"
    if nvidia_smi_available:
        posix_output = "".join(
            f"printf '%s\\n' '{line}'\n" for line in nvidia_smi_output
        )
        posix_nvidia_smi = bin_dir / "nvidia-smi"
        posix_nvidia_smi.write_text(
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            "printf '%s\\n' \"$*\" >> \"$FAKE_NVIDIA_SMI_LOG\"\n"
            + posix_output
            + f"exit {nvidia_smi_returncode}\n",
            encoding="utf-8",
        )
        posix_nvidia_smi.chmod(0o755)
        windows_output = "".join(f"echo {line}\n" for line in nvidia_smi_output)
        windows_nvidia_smi = bin_dir / "nvidia-smi.cmd"
        windows_nvidia_smi.write_text(
            "@echo off\n"
            "echo %* >>\"%FAKE_NVIDIA_SMI_LOG%\"\n"
            + windows_output
            + f"exit /b {nvidia_smi_returncode}\n",
            encoding="utf-8",
        )
    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_GPU_NAMES": json.dumps([gpu_name] * gpu_count),
            "FAKE_GPU_COUNT": str(gpu_count),
            "FAKE_GPU_MEMORY": str(gpu_memory_bytes),
            "FAKE_COMPUTE_CAPABILITY": json.dumps(compute_capability),
            "FAKE_TORCH_VERSION": torch_version,
            "FAKE_TORCH_CUDA": torch_cuda,
            "FAKE_CUDA_AVAILABLE": "1" if cuda_available else "0",
            "FAKE_NVIDIA_SMI_LOG": str(nvidia_log),
            "PATH": (
                str(bin_dir) + os.pathsep + environment.get("PATH", "")
                if nvidia_smi_available
                else str(bin_dir)
            ),
            "PYTHONPATH": str(fake_modules),
        }
    )
    output = output or tmp_path / "environment.json"
    prelude = (
        "import os, platform, sys\n"
        f"platform.python_implementation = lambda: {python_implementation!r}\n"
        f"sys.version_info = ({python_version[0]}, {python_version[1]}, 0, 'final', 0)\n"
        "os.sched_getaffinity = lambda pid: set(range(4))\n"
        "os.sysconf = lambda key: "
        "{'SC_PAGE_SIZE': 4096, 'SC_PHYS_PAGES': 8388608}[key]\n"
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-", str(output)],
        input=prelude + _runner_python_heredoc("verify_runtime"),
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    return completed, output, nvidia_log


@pytest.mark.parametrize("python_version", [(3, 10), (3, 11), (3, 12)])
def test_runner_venv_runtime_accepts_supported_cpython_versions_under_optimization(
    tmp_path: Path, python_version: tuple[int, int]
) -> None:
    completed, output, _ = _run_fake_runtime(
        tmp_path,
        gpu_name="NVIDIA GeForce RTX 4090",
        python_version=python_version,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.is_file()


@pytest.mark.parametrize(
    "python_implementation,python_version",
    [("CPython", (3, 9)), ("CPython", (3, 13)), ("PyPy", (3, 11))],
)
def test_runner_venv_runtime_rejects_unsupported_python_under_optimization(
    tmp_path: Path,
    python_implementation: str,
    python_version: tuple[int, int],
) -> None:
    completed, output, _ = _run_fake_runtime(
        tmp_path,
        gpu_name="NVIDIA GeForce RTX 4090",
        python_implementation=python_implementation,
        python_version=python_version,
    )

    assert completed.returncode != 0
    assert not output.exists()


@pytest.mark.parametrize(
    "runtime_overrides",
    [
        {"torch_version": "2.11.0"},
        {"torch_cuda": "12.6"},
        {"cuda_available": False},
    ],
)
def test_runner_venv_runtime_rejects_incompatible_torch_or_cuda_under_optimization(
    tmp_path: Path, runtime_overrides: dict[str, object]
) -> None:
    completed, output, _ = _run_fake_runtime(
        tmp_path,
        gpu_name="NVIDIA GeForce RTX 4090",
        **runtime_overrides,
    )

    assert completed.returncode != 0
    assert not output.exists()


def test_fake_nvidia_smi_includes_a_posix_executable_and_never_uses_the_host(
    tmp_path: Path,
) -> None:
    completed, _, nvidia_log = _run_fake_runtime(
        tmp_path, gpu_name="NVIDIA GeForce RTX 4090"
    )

    fake_nvidia_smi = tmp_path / "bin" / "nvidia-smi"
    assert completed.returncode == 0, completed.stderr
    assert fake_nvidia_smi.is_file()
    assert os.access(fake_nvidia_smi, os.X_OK)
    assert "--query-gpu=driver_version" in nvidia_log.read_text(encoding="utf-8")
    bash_probe = subprocess.run(
        [
            _bash(),
            "-c",
            'export PATH="$1:$PATH" FAKE_NVIDIA_SMI_LOG="$2"; nvidia-smi --probe',
            "_",
            _git_bash_path(fake_nvidia_smi.parent),
            _git_bash_path(nvidia_log),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert bash_probe.returncode == 0, bash_probe.stderr
    assert bash_probe.stdout.strip() == "555.42"
    assert "--probe" in nvidia_log.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "nvidia_smi_overrides",
    [
        {"nvidia_smi_available": False},
        {"nvidia_smi_returncode": 9},
        {"nvidia_smi_output": ()},
        {"nvidia_smi_output": ("555.42", "556.01")},
    ],
)
def test_runner_runtime_rejects_invalid_nvidia_smi_results_without_manifest(
    tmp_path: Path, nvidia_smi_overrides: dict[str, object]
) -> None:
    completed, output, _ = _run_fake_runtime(
        tmp_path,
        gpu_name="NVIDIA GeForce RTX 4090",
        **nvidia_smi_overrides,
    )

    assert completed.returncode != 0
    assert not output.exists()


@pytest.mark.parametrize(
    "gpu_name,compute_capability",
    [
        ("NVIDIA GeForce RTX 4090", (8, 9)),
        ("NVIDIA GeForce RTX 4090 D", (8, 9)),
        ("NVIDIA GeForce RTX 5090", (12, 0)),
    ],
)
def test_runner_runtime_accepts_supported_gpu_variants_and_publishes_provenance(
    tmp_path: Path, gpu_name: str, compute_capability: tuple[int, int]
) -> None:
    completed, output, nvidia_log = _run_fake_runtime(
        tmp_path,
        gpu_name=gpu_name,
        compute_capability=compute_capability,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["gpu_0"] == gpu_name
    assert manifest["gpu_memory_bytes"] == 24 * 1024**3
    assert manifest["gpu_count"] == 1
    assert manifest["gpu_names"] == [gpu_name]
    assert manifest["gpu_memory_bytes_all"] == [24 * 1024**3]
    assert manifest["compute_capability"] == list(compute_capability)
    assert manifest["compute_capabilities"] == [list(compute_capability)]
    assert manifest["driver_version"] == "555.42"
    assert manifest["cpu_affinity_count"] == 4
    assert manifest["host_memory_bytes"] == 32 * 1024**3
    assert "--query-gpu=driver_version" in nvidia_log.read_text(encoding="utf-8")


def test_runner_runtime_audits_two_visible_5090_devices(tmp_path: Path) -> None:
    gpu_name = "NVIDIA GeForce RTX 5090"
    completed, output, _ = _run_fake_runtime(
        tmp_path,
        gpu_name=gpu_name,
        gpu_count=2,
        gpu_memory_bytes=32 * 1024**3,
        compute_capability=(12, 0),
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["gpu_count"] == 2
    assert manifest["gpu_names"] == [gpu_name, gpu_name]
    assert manifest["gpu_memory_bytes_all"] == [32 * 1024**3, 32 * 1024**3]
    assert manifest["compute_capabilities"] == [[12, 0], [12, 0]]


@pytest.mark.parametrize(
    "gpu_name,gpu_memory_bytes",
    [
        ("NVIDIA RTX A6000", 48 * 1024**3),
        ("NVIDIA GeForce RTX 4090", 22 * 1024**3),
    ],
)
def test_runner_runtime_rejects_wrong_gpu_or_insufficient_memory_without_manifest(
    tmp_path: Path, gpu_name: str, gpu_memory_bytes: int
) -> None:
    completed, output, _ = _run_fake_runtime(
        tmp_path, gpu_name=gpu_name, gpu_memory_bytes=gpu_memory_bytes
    )

    assert completed.returncode != 0
    assert not output.exists()


def test_runner_runtime_preserves_exclusive_environment_manifest(tmp_path: Path) -> None:
    output = tmp_path / "environment.json"
    output.write_text("reserved\n", encoding="utf-8")

    completed, _, _ = _run_fake_runtime(
        tmp_path, gpu_name="NVIDIA GeForce RTX 4090", output=output
    )

    assert completed.returncode != 0
    assert output.read_text(encoding="utf-8") == "reserved\n"


def test_runner_runtime_and_plan_checks_survive_optimized_python(tmp_path: Path) -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "assert " not in source

    runtime, environment, _ = _run_fake_runtime(
        tmp_path,
        gpu_name="NVIDIA A100",
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
    fake_python, log = _make_fake_python(tmp_path, "3.13.0")
    ignored = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment={
            "MSYS2_ARG_CONV_EXCL": "",
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        },
    )
    assert ignored.returncode == 2
    assert "CPython 3.10, 3.11, or 3.12 is required" in ignored.stderr
    assert "dirty" not in ignored.stderr.lower()
    assert log.read_text(encoding="utf-8").splitlines() == _fake_python_probe_log()[:2]
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
    assert valid_log.read_text(encoding="utf-8").splitlines() == _fake_python_probe_log()


def test_runner_fails_closed_when_git_status_cannot_be_inspected(tmp_path: Path) -> None:
    repository, commit = _make_clean_repository(tmp_path)
    fake_python, python_log = _make_fake_python(tmp_path, "3.12.3")
    bash_env = tmp_path / "git-status-error-bash-env"
    bash_env.write_text(
        "git() {\n"
        "  if [ \"${1:-}\" = \"-C\" ] && [ \"${3:-}\" = \"status\" ]; then\n"
        "    return 86\n"
        "  fi\n"
        "  builtin command git \"$@\"\n"
        "}\n",
        encoding="utf-8",
    )

    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment={
            "MSYS2_ARG_CONV_EXCL": "",
            "BASH_ENV": bash_env.as_posix(),
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": python_log.as_posix(),
        },
    )

    assert completed.returncode == 2
    assert "cannot inspect Git worktree status" in completed.stderr
    assert python_log.read_text(encoding="utf-8").splitlines() == _fake_python_probe_log()
    assert not list(tmp_path.glob("validation-v2-audit-*"))
    assert not list(tmp_path.glob("validation-v2-preflight-*"))
    assert not (repository / ".venv-server").exists()
    assert not (repository / "results").exists()


def test_runner_rejects_a_preexisting_audit_seal_before_venv_or_other_writes(
    tmp_path: Path,
) -> None:
    repository, commit = _make_clean_repository(tmp_path)
    audit_dir = tmp_path / f"validation-v2-audit-{commit}-sharded-v2-py312"
    audit_dir.mkdir()
    fake_python, log = _make_fake_python(tmp_path, "3.12.3")
    environment, _ = _host_environment(tmp_path)
    environment.update(
        {
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        }
    )
    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment=environment,
    )
    assert completed.returncode == 2
    assert "AUDIT_DIR" in completed.stderr
    assert log.read_text(encoding="utf-8").splitlines() == _fake_python_probe_log()
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
    environment, _ = _host_environment(tmp_path)
    environment.update(
        {
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        }
    )

    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment=environment,
    )
    assert completed.returncode == 2
    assert "AUDIT_DIR" in completed.stderr
    assert audit_dir.is_symlink()
    assert not (repository / ".venv-server").exists()


def test_runner_rejects_non_linux_before_any_campaign_or_repository_writes(
    tmp_path: Path,
) -> None:
    repository, commit = _make_clean_repository(tmp_path)
    fake_python, log = _make_fake_python(tmp_path, "3.12.3")
    environment, host_log = _host_environment(tmp_path, kernel="TestOS")
    environment.update(
        {
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        }
    )
    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment=environment,
    )

    assert completed.returncode == 2
    assert "requires Linux" in completed.stderr
    assert log.read_text(encoding="utf-8").splitlines() == _fake_python_probe_log()
    assert host_log.read_text(encoding="utf-8").splitlines() == ["<uname:-s>"]
    assert not list(tmp_path.glob("validation-v2-audit-*"))
    assert not list(tmp_path.glob("validation-v2-preflight-*"))
    assert not (repository / ".venv-server").exists()
    assert not (repository / "results").exists()


@pytest.mark.parametrize("missing_command", ["nvidia-smi", "pgrep", "nohup", "tee"])
def test_runner_rejects_a_missing_host_command_before_any_write(
    tmp_path: Path, missing_command: str
) -> None:
    repository, commit = _make_clean_repository(tmp_path)
    fake_python, python_log = _make_fake_python(tmp_path, "3.12.3")
    environment, host_log = _host_environment(
        tmp_path, missing_command=missing_command
    )
    environment.update(
        {
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": python_log.as_posix(),
        }
    )

    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment=environment,
    )

    required_commands = ["nvidia-smi", "pgrep", "nohup", "tee"]
    missing_index = required_commands.index(missing_command)
    assert completed.returncode == 2
    assert missing_command in completed.stderr
    assert python_log.read_text(encoding="utf-8").splitlines() == _fake_python_probe_log()
    assert host_log.read_text(encoding="utf-8").splitlines() == [
        "<uname:-s>",
        *[
            f"<command-v:{command}>"
            for command in required_commands[: missing_index + 1]
        ],
    ]
    assert not list(tmp_path.glob("validation-v2-audit-*"))
    assert not list(tmp_path.glob("validation-v2-preflight-*"))
    assert not (repository / ".venv-server").exists()
    assert not (repository / "results").exists()


def _prepare_linux_runner_environment(tmp_path: Path, repository: Path) -> dict[str, str]:
    fake_python, log = _make_fake_python(tmp_path, "3.12.3")
    venv_python = repository / ".venv-server" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    _write_bash_executable(
        venv_python,
        'if [ "${1:-}" = "-c" ]; then printf "%s\\n" "CPython 3.12"; exit 0; fi\n'
        "exit 97\n",
    )
    environment, _ = _host_environment(tmp_path)
    environment.update(
        {
            "PYTHON3_BIN": fake_python.as_posix(),
            "FAKE_PYTHON_LOG": log.as_posix(),
        }
    )
    return environment


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
    (repository / ".git" / "info" / "exclude").write_text("/results\n", encoding="utf-8")
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


def test_launch_shard_pins_workers_round_robin_to_visible_gpus(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_bash_executable(bin_dir / "pgrep", "exit 1\n")
    fake_python = tmp_path / "fake-python"
    invocation_log = tmp_path / "python.log"
    _write_bash_executable(
        fake_python,
        'printf "%s|%s\\n" "${CUDA_VISIBLE_DEVICES:-missing}" "$*" '
        '>> "$FAKE_PYTHON_LOG"\nsleep 30\n',
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
        'CONFIG="$5" PLAN="$6" FAKE_PYTHON_LOG="$7" VALIDATION_V2_GPU_COUNT=2; '
        'source "$8"; launch_shard 001; '
        'for attempt in 1 2 3 4 5; do test -s "$FAKE_PYTHON_LOG" && break; sleep 0.1; done; '
        'grep -F "1|-m validation_v2.cli shard" "$FAKE_PYTHON_LOG"; '
        'kill "$(cat "$AUDIT_DIR/shard-001.pid")" 2>/dev/null || true; exit 0',
        bin_dir, fake_python, audit_dir, shards_root, config, plan, invocation_log, HELPERS,
    )

    assert completed.returncode == 0, completed.stderr


def test_launch_shard_rejects_invalid_visible_gpu_count(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python"
    _write_bash_executable(fake_python, "exit 97\n")
    completed = _run_helper_bash(
        'export PYTHON_BIN="$1" VALIDATION_V2_GPU_COUNT=0; '
        'source "$2"; launch_shard 000',
        fake_python, HELPERS,
    )

    assert completed.returncode == 2
    assert "invalid VALIDATION_V2_GPU_COUNT" in completed.stderr


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
