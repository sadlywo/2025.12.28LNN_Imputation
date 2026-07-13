"""Executable contracts for the Python 3.12 server validation runner."""

import os
from pathlib import Path
import shutil
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run_validation_v2_server.sh"


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
        "printf '%s\\n' \"$*\" >> \"$FAKE_PYTHON_LOG\"\n"
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
    assert "-m venv" not in invocations
    assert "-m pip" not in invocations


def test_runner_uses_local_venv_and_explicit_cuda121_torch_index() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert ".venv-server" in source
    assert "PYTHON3_BIN" in source
    assert "-m venv" in source
    assert "https://download.pytorch.org/whl/cu121" in source
    assert "torch==2.3.1" in source
    assert "torch.__version__" in source
    assert "2.3.1+cu121" in source
    assert "conda activate" not in source
    assert "pinn_imu" not in source


def test_runner_full_mode_calls_formal_workflow() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    for token in (
        "test_linux_rename_noreplace_survives_real_directory_race",
        "-m pytest -q",
        "validation_v2.cli shard-plan",
        "run_formal_campaign",
        "validation_v2.cli merge-shards",
        "validation_v2.experiments.validate_artifacts",
        "validation_v2.cli summarize",
    ):
        assert token in source
