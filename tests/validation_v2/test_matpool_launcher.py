"""Executable contracts for the fail-closed MatPool validation launcher."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "run_validation_v2_matpool.sh"


def _bash() -> str:
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if git_bash.is_file():
        return str(git_bash)
    executable = shutil.which("bash")
    assert executable is not None, "Bash is required to test the MatPool launcher"
    return executable


def _write_executable(path: Path, source: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + source, encoding="utf-8")
    path.chmod(0o755)


def _git(*arguments: str, cwd: Path) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=cwd, text=True, encoding="utf-8"
    ).strip()


def _from_bash_path(value: str) -> Path:
    match = re.fullmatch(r"/([a-zA-Z])(?:/(.*))?", value)
    if os.name == "nt" and match:
        remainder = match.group(2) or ""
        return Path(f"{match.group(1).upper()}:/{remainder}")
    return Path(value)


def _to_bash_path(path: Path) -> str:
    resolved = path.resolve().as_posix()
    if os.name == "nt" and re.match(r"^[A-Za-z]:/", resolved):
        return f"/{resolved[0].lower()}/{resolved[3:]}"
    return resolved


def _is_absolute_serialized_path(value: str) -> bool:
    return value.startswith("/") or re.match(r"^[A-Za-z]:/", value) is not None


def _make_repository(tmp_path: Path) -> tuple[Path, str, dict[str, str]]:
    repository = tmp_path / "clean repo with spaces"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(LAUNCHER, scripts / LAUNCHER.name)
    _write_executable(
        scripts / "run_validation_v2_server.sh",
        "printf 'PWD=%s\\n' \"$PWD\" >> \"$FAKE_GENERIC_ARGS\"\n"
        "for argument in \"$@\"; do printf '<%s>\\n' \"$argument\" >> \"$FAKE_GENERIC_ARGS\"; done\n"
        "printf '%s\\n' 'fake generic runner output'\n"
        "exit \"${FAKE_GENERIC_RC:-0}\"\n",
    )
    (repository / ".gitignore").write_text(
        ".validation-v2-matpool/\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Launcher Tests"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"], cwd=repository, check=True
    )
    commit = _git("rev-parse", "HEAD", cwd=repository)

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "uname", "printf '%s\\n' Linux\n")
    python = _to_bash_path(Path(sys.executable))
    bash_environment = tmp_path / "bash-env"
    bash_environment.write_text(
        "uname() { printf '%s\\n' Linux; }\n"
        "tail() { command \"$FAKE_BIN/tail\" \"$@\"; }\n"
        "date() {\n"
        "  if [ -n \"${FAKE_DATE_VALUE:-}\" ]; then printf '%s\\n' \"$FAKE_DATE_VALUE\";\n"
        "  else command /usr/bin/date \"$@\"; fi\n"
        "}\n"
        "python3() {\n"
        "  local -a converted=()\n"
        "  local argument\n"
        "  for argument in \"$@\"; do\n"
        "    case \"$argument\" in\n"
        "      /[A-Za-z]/*) argument=\"${argument:1:1}:${argument:2}\" ;;\n"
        "    esac\n"
        "    converted+=(\"$argument\")\n"
        "  done\n"
        "  local state_file=\"${MATPOOL_STATE_FILE:-}\"\n"
        "  local state_temp=\"${MATPOOL_STATE_TEMP:-}\"\n"
        "  case \"$state_file\" in /[A-Za-z]/*) state_file=\"${state_file:1:1}:${state_file:2}\" ;; esac\n"
        "  case \"$state_temp\" in /[A-Za-z]/*) state_temp=\"${state_temp:1:1}:${state_temp:2}\" ;; esac\n"
        f"  MSYS2_ARG_CONV_EXCL='*' MSYS2_ENV_CONV_EXCL='*' MATPOOL_STATE_FILE=\"$state_file\" MATPOOL_STATE_TEMP=\"$state_temp\" {shlex.quote(python)} \"${{converted[@]}}\"\n"
        "}\n",
        encoding="utf-8",
    )
    _write_executable(
        fake_bin / "python3",
        "converted=()\n"
        "for argument in \"$@\"; do\n"
        "  case \"$argument\" in\n"
        "    /[A-Za-z]/*) argument=\"${argument:1:1}:${argument:2}\" ;;\n"
        "  esac\n"
        "  converted+=(\"$argument\")\n"
        "done\n"
        "export MSYS2_ARG_CONV_EXCL='*'\n"
        "export MSYS2_ENV_CONV_EXCL='*'\n"
        f"exec {shlex.quote(python)} \"${{converted[@]}}\"\n",
    )
    _write_executable(
        fake_bin / "tmux",
        "printf '%s\\n' \"$*\" >> \"$FAKE_TMUX_LOG\"\n"
        "case \"${1:-}\" in\n"
        "  has-session) exit \"${FAKE_TMUX_HAS_RC:-1}\" ;;\n"
        "  new-session)\n"
        "    if [ -n \"${FAKE_LOCK_OBSERVATION:-}\" ]; then\n"
        "      if [ -d \"$FAKE_EXPECT_LOCK_PATH\" ]; then\n"
        "        printf '%s\\n' locked >> \"$FAKE_LOCK_OBSERVATION\"\n"
        "      else\n"
        "        printf '%s\\n' unlocked >> \"$FAKE_LOCK_OBSERVATION\"\n"
        "      fi\n"
        "    fi\n"
        "    if [ -n \"${FAKE_TMUX_BARRIER_ENTERED:-}\" ]; then\n"
        "      : > \"$FAKE_TMUX_BARRIER_ENTERED\"\n"
        "      attempts=0\n"
        "      while [ ! -e \"$FAKE_TMUX_BARRIER_RELEASE\" ]; do\n"
        "        attempts=$((attempts + 1))\n"
        "        [ \"$attempts\" -lt 500 ] || exit 96\n"
        "        sleep 0.01\n"
        "      done\n"
        "    fi\n"
        "    new_rc=\"${FAKE_TMUX_NEW_RC:-0}\"\n"
        "    if [ \"$new_rc\" -ne 0 ]; then exit \"$new_rc\"; fi\n"
        "    shift\n"
        "    [ \"${1:-}\" = -d ] && shift\n"
        "    [ \"${1:-}\" = -s ] && shift 2\n"
        "    set +e\n"
        "    \"$@\"\n"
        "    exit 0 ;;\n"
        "  *) exit 97 ;;\n"
        "esac\n",
    )
    _write_executable(
        fake_bin / "tail",
        "printf '%s\\n' \"$*\" >> \"$FAKE_TAIL_LOG\"\n"
        "if [ \"${1:-}\" = -n ]; then exec /usr/bin/tail \"$@\"; fi\n"
        "exit 0\n",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": _to_bash_path(fake_bin) + ":" + environment["PATH"],
            "FAKE_BIN": _to_bash_path(fake_bin),
            "FAKE_GENERIC_ARGS": _to_bash_path(tmp_path / "generic-args.log"),
            "FAKE_TMUX_LOG": _to_bash_path(tmp_path / "tmux.log"),
            "FAKE_TAIL_LOG": _to_bash_path(tmp_path / "tail.log"),
            "BASH_ENV": _to_bash_path(bash_environment),
            "MSYS2_ARG_CONV_EXCL": "",
        }
    )
    return repository, commit, environment


def _run(
    repository: Path, environment: dict[str, str], *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_bash(), _to_bash_path(repository / "scripts" / LAUNCHER.name), *arguments],
        cwd=repository,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
    )


def _popen(
    repository: Path, environment: dict[str, str], *arguments: str
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [_bash(), _to_bash_path(repository / "scripts" / LAUNCHER.name), *arguments],
        cwd=repository,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )


def _wait_until(predicate, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for concurrent launcher condition")


def _state(repository: Path) -> dict[str, object]:
    return json.loads(
        (repository / ".validation-v2-matpool" / "current.json").read_text(
            encoding="utf-8"
        )
    )


def test_help_lists_only_start_status_and_logs(tmp_path: Path) -> None:
    repository, _, environment = _make_repository(tmp_path)

    completed = _run(repository, environment, "--help")
    stopped = _run(repository, environment, "stop")

    assert completed.returncode == 0
    assert "start" in completed.stdout
    assert "status" in completed.stdout
    assert "logs" in completed.stdout
    assert "stop" not in completed.stdout.lower()
    assert stopped.returncode == 2
    assert "unknown command" in stopped.stderr.lower()


def test_start_forwards_exact_commit_full_unique_suffix_and_default_workers(
    tmp_path: Path,
) -> None:
    repository, commit, environment = _make_repository(tmp_path)

    completed = _run(repository, environment, "start")

    assert completed.returncode == 0, completed.stderr
    state = _state(repository)
    suffix = str(state["campaign_suffix"])
    assert re.fullmatch(r"matpool-\d{8}T\d{6}Z", suffix)
    assert state["schema_version"] == 1
    assert state["commit"] == commit
    assert state["session"] == f"validation-v2-{commit[:12]}-{suffix}"
    assert state["max_workers"] == 4
    assert state["skip_dependency_install"] is False
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", str(state["created_at"]))
    for key in (
        "command_file",
        "log_path",
        "exit_status_path",
        "audit_dir",
        "shards_root",
        "final_root",
    ):
        assert _is_absolute_serialized_path(str(state[key]))
    assert str(state["audit_dir"]).endswith(
        f"validation-v2-audit-{commit}-{suffix}"
    )
    assert str(state["shards_root"]).endswith(
        f"results/validation_v2/server-full-shards-{commit}-{suffix}"
    )
    assert str(state["final_root"]).endswith(
        f"results/validation_v2/server-full-final-{commit}-{suffix}"
    )
    command_file = _from_bash_path(str(state["command_file"]))
    log_path = _from_bash_path(str(state["log_path"]))
    exit_path = _from_bash_path(str(state["exit_status_path"]))
    assert command_file.is_file()
    assert log_path.read_text(encoding="utf-8").count("fake generic runner output") == 1
    assert exit_path.read_text(encoding="utf-8").strip() == "0"
    invocation = _from_bash_path(environment["FAKE_GENERIC_ARGS"]).read_text(encoding="utf-8")
    assert f"<{commit}>" in invocation
    assert "<full>" in invocation
    assert f"<{suffix}>" in invocation
    assert "<4>" in invocation
    assert "--skip-dependency-install" not in invocation


def test_start_overrides_workers_and_forwards_skip_dependency_install(
    tmp_path: Path,
) -> None:
    repository, _, environment = _make_repository(tmp_path)

    completed = _run(
        repository,
        environment,
        "start",
        "--max-workers",
        "2",
        "--skip-dependency-install",
    )

    assert completed.returncode == 0, completed.stderr
    state = _state(repository)
    assert state["max_workers"] == 2
    assert state["skip_dependency_install"] is True
    invocation = _from_bash_path(environment["FAKE_GENERIC_ARGS"]).read_text(encoding="utf-8")
    assert "<2>" in invocation
    assert "<--skip-dependency-install>" in invocation


def test_dirty_repository_fails_before_creating_launcher_state(tmp_path: Path) -> None:
    repository, _, environment = _make_repository(tmp_path)
    (repository / "untracked-sentinel").write_text("dirty\n", encoding="utf-8")

    completed = _run(repository, environment, "start")

    assert completed.returncode == 2
    assert "clean" in completed.stderr.lower()
    assert not (repository / ".validation-v2-matpool").exists()
    assert not _from_bash_path(environment["FAKE_TMUX_LOG"]).exists()


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (("start", "--max-workers"), "requires"),
        (("start", "--max-workers", "3"), "one of"),
        (("start", "--unknown"), "unknown option"),
        (("status", "--max-workers", "2"), "does not accept"),
        (("logs", "extra"), "does not accept"),
        ((), "command is required"),
    ],
)
def test_invalid_commands_and_options_fail_with_status_two(
    tmp_path: Path, arguments: tuple[str, ...], expected: str
) -> None:
    repository, _, environment = _make_repository(tmp_path)

    completed = _run(repository, environment, *arguments)

    assert completed.returncode == 2
    assert expected in completed.stderr.lower()


def test_state_is_atomic_complete_and_command_quotes_repository_with_spaces(
    tmp_path: Path,
) -> None:
    repository, _, environment = _make_repository(tmp_path)

    completed = _run(repository, environment, "start")

    assert completed.returncode == 0, completed.stderr
    state_dir = repository / ".validation-v2-matpool"
    state = _state(repository)
    expected_keys = {
        "schema_version",
        "commit",
        "campaign_suffix",
        "session",
        "command_file",
        "log_path",
        "exit_status_path",
        "audit_dir",
        "shards_root",
        "final_root",
        "max_workers",
        "skip_dependency_install",
        "created_at",
    }
    assert set(state) == expected_keys
    assert not list(state_dir.glob(".current.json.*"))
    command = _from_bash_path(str(state["command_file"])).read_text(encoding="utf-8")
    assert "set -o pipefail" in command
    assert "PIPESTATUS[0]" in command
    assert "run_validation_v2_server.sh" in command
    assert "validation_v2.cli" not in command
    assert "full-8-shards" not in command
    invocation = _from_bash_path(environment["FAKE_GENERIC_ARGS"]).read_text(encoding="utf-8")
    assert "clean repo with spaces" in invocation.splitlines()[0]


def test_live_current_session_refuses_duplicate_start(tmp_path: Path) -> None:
    repository, _, environment = _make_repository(tmp_path)
    first = _run(repository, environment, "start")
    assert first.returncode == 0, first.stderr
    old_state = _state(repository)
    old_command = _from_bash_path(str(old_state["command_file"]))
    old_bytes = old_command.read_bytes()
    environment["FAKE_TMUX_HAS_RC"] = "0"

    duplicate = _run(repository, environment, "start")

    assert duplicate.returncode == 2
    assert "already active" in duplicate.stderr.lower()
    assert _state(repository) == old_state
    assert old_command.read_bytes() == old_bytes


def test_concurrent_starts_are_serialized_before_state_publish_and_tmux(
    tmp_path: Path,
) -> None:
    repository, _, environment = _make_repository(tmp_path)
    entered_first = tmp_path / "tmux-entered-first"
    entered_second = tmp_path / "tmux-entered-second"
    release = tmp_path / "tmux-release"
    first_environment = environment.copy()
    first_environment.update(
        {
            "FAKE_DATE_VALUE": "20260713T110001Z",
            "FAKE_TMUX_BARRIER_ENTERED": _to_bash_path(entered_first),
            "FAKE_TMUX_BARRIER_RELEASE": _to_bash_path(release),
        }
    )
    second_environment = environment.copy()
    second_environment.update(
        {
            "FAKE_DATE_VALUE": "20260713T110002Z",
            "FAKE_TMUX_BARRIER_ENTERED": _to_bash_path(entered_second),
            "FAKE_TMUX_BARRIER_RELEASE": _to_bash_path(release),
        }
    )

    first = _popen(repository, first_environment, "start")
    try:
        _wait_until(entered_first.exists)
        second = _popen(repository, second_environment, "start")
        try:
            _wait_until(lambda: second.poll() is not None or entered_second.exists())
            release.write_text("release\n", encoding="utf-8")
            first_stdout, first_stderr = first.communicate(timeout=10)
            second_stdout, second_stderr = second.communicate(timeout=10)
        finally:
            if second.poll() is None:
                second.kill()
                second.communicate()
    finally:
        if first.poll() is None:
            release.write_text("release\n", encoding="utf-8")
            first.kill()
            first.communicate()

    assert sorted((first.returncode, second.returncode)) == [0, 2], (
        first_stdout,
        first_stderr,
        second_stdout,
        second_stderr,
    )
    loser_stderr = first_stderr if first.returncode == 2 else second_stderr
    assert "already starting" in loser_stderr.lower() or "locked" in loser_stderr.lower()
    tmux_lines = _from_bash_path(environment["FAKE_TMUX_LOG"]).read_text(
        encoding="utf-8"
    ).splitlines()
    new_sessions = [line for line in tmux_lines if line.startswith("new-session ")]
    assert len(new_sessions) == 1
    state = _state(repository)
    assert str(state["session"]) in new_sessions[0]
    state_dir = repository / ".validation-v2-matpool"
    assert len(list(state_dir.glob("run-*.sh"))) == 1
    assert len(list(state_dir.glob("run-*.log"))) == 1
    assert len(list(state_dir.glob("run-*.exit"))) == 1


def test_success_holds_start_lock_through_tmux_then_cleans_it(tmp_path: Path) -> None:
    repository, _, environment = _make_repository(tmp_path)
    state_dir = repository / ".validation-v2-matpool"
    observation = tmp_path / "lock-observation"
    environment.update(
        {
            "FAKE_LOCK_OBSERVATION": _to_bash_path(observation),
            "FAKE_EXPECT_LOCK_PATH": _to_bash_path(state_dir / "start.lock"),
        }
    )

    completed = _run(repository, environment, "start")

    assert completed.returncode == 0, completed.stderr
    assert observation.read_text(encoding="utf-8").splitlines() == ["locked"]
    assert not (state_dir / "start.lock").exists()


def test_stale_start_lock_fails_closed_and_requires_manual_confirmation(
    tmp_path: Path,
) -> None:
    repository, _, environment = _make_repository(tmp_path)
    state_dir = repository / ".validation-v2-matpool"
    start_lock = state_dir / "start.lock"
    start_lock.mkdir(parents=True)

    completed = _run(repository, environment, "start")

    assert completed.returncode == 2
    assert "lock" in completed.stderr.lower()
    assert "manual" in completed.stderr.lower()
    assert start_lock.is_dir()
    assert not _from_bash_path(environment["FAKE_TMUX_LOG"]).exists()


def test_inactive_current_session_allows_new_campaign_without_overwriting_history(
    tmp_path: Path,
) -> None:
    repository, _, environment = _make_repository(tmp_path)
    first = _run(repository, environment, "start")
    assert first.returncode == 0, first.stderr
    old_state = _state(repository)
    old_paths = [
        _from_bash_path(str(old_state[key]))
        for key in ("command_file", "log_path", "exit_status_path")
    ]
    old_contents = [path.read_bytes() for path in old_paths]
    environment["FAKE_TMUX_HAS_RC"] = "1"
    time.sleep(1.1)

    second = _run(repository, environment, "start", "--max-workers", "1")

    assert second.returncode == 0, second.stderr
    new_state = _state(repository)
    assert new_state["campaign_suffix"] != old_state["campaign_suffix"]
    assert new_state["max_workers"] == 1
    assert all(path.is_file() for path in old_paths)
    assert [path.read_bytes() for path in old_paths] == old_contents


@pytest.mark.parametrize("command", ["start", "status", "logs"])
def test_malformed_state_fails_closed_without_tmux_or_tail(
    tmp_path: Path, command: str
) -> None:
    repository, _, environment = _make_repository(tmp_path)
    state_dir = repository / ".validation-v2-matpool"
    state_dir.mkdir()
    (state_dir / "current.json").write_text(
        '{"schema_version":1,"commit":"not-a-commit"}\n', encoding="utf-8"
    )

    completed = _run(repository, environment, command)

    assert completed.returncode == 2
    assert "malformed state" in completed.stderr.lower()
    assert not _from_bash_path(environment["FAKE_TMUX_LOG"]).exists()
    assert not _from_bash_path(environment["FAKE_TAIL_LOG"]).exists()


def test_status_is_read_only_and_reports_active_state_and_paths(tmp_path: Path) -> None:
    repository, commit, environment = _make_repository(tmp_path)
    started = _run(repository, environment, "start")
    assert started.returncode == 0, started.stderr
    state = _state(repository)
    state_dir = repository / ".validation-v2-matpool"
    before = {path.name: path.read_bytes() for path in state_dir.iterdir()}
    environment["FAKE_TMUX_HAS_RC"] = "0"

    completed = _run(repository, environment, "status")

    after = {path.name: path.read_bytes() for path in state_dir.iterdir()}
    assert completed.returncode == 0, completed.stderr
    assert "state: active" in completed.stdout.lower()
    assert commit in completed.stdout
    assert str(state["campaign_suffix"]) in completed.stdout
    assert str(state["session"]) in completed.stdout
    assert "max_workers: 4" in completed.stdout
    for key in ("audit_dir", "shards_root", "final_root", "log_path", "exit_status_path"):
        assert str(state[key]) in completed.stdout
    assert completed.stdout.count("fake generic runner output") <= 1
    assert before == after


def test_status_distinguishes_inactive_from_tmux_inspection_error(tmp_path: Path) -> None:
    repository, _, environment = _make_repository(tmp_path)
    started = _run(repository, environment, "start")
    assert started.returncode == 0, started.stderr
    environment["FAKE_TMUX_HAS_RC"] = "1"
    inactive = _run(repository, environment, "status")
    environment["FAKE_TMUX_HAS_RC"] = "7"
    errored = _run(repository, environment, "status")

    assert inactive.returncode == 0
    assert "state: inactive" in inactive.stdout.lower()
    assert errored.returncode == 2
    assert "tmux inspection failed" in errored.stderr.lower()
    assert "rc=7" in errored.stderr


@pytest.mark.parametrize("command", ["status", "logs"])
def test_read_commands_require_state(tmp_path: Path, command: str) -> None:
    repository, _, environment = _make_repository(tmp_path)

    completed = _run(repository, environment, command)

    assert completed.returncode == 2
    assert "no current state" in completed.stderr.lower()


def test_logs_validates_campaign_log_and_invokes_tail_follow(tmp_path: Path) -> None:
    repository, _, environment = _make_repository(tmp_path)
    started = _run(repository, environment, "start")
    assert started.returncode == 0, started.stderr
    state = _state(repository)
    tail_log = _from_bash_path(environment["FAKE_TAIL_LOG"])
    if tail_log.exists():
        tail_log.unlink()

    completed = _run(repository, environment, "logs")

    assert completed.returncode == 0, completed.stderr
    invocation = tail_log.read_text(encoding="utf-8")
    assert invocation.strip() == f"-F -- {state['log_path']}"


def test_logs_rejects_missing_campaign_log(tmp_path: Path) -> None:
    repository, _, environment = _make_repository(tmp_path)
    started = _run(repository, environment, "start")
    assert started.returncode == 0, started.stderr
    state = _state(repository)
    _from_bash_path(str(state["log_path"])).unlink()

    completed = _run(repository, environment, "logs")

    assert completed.returncode == 2
    assert "log does not exist" in completed.stderr.lower()
    assert not _from_bash_path(environment["FAKE_TAIL_LOG"]).exists()


def test_tmux_start_failure_preserves_published_state_log_and_diagnostic(
    tmp_path: Path,
) -> None:
    repository, _, environment = _make_repository(tmp_path)
    environment["FAKE_TMUX_NEW_RC"] = "9"
    state_dir = repository / ".validation-v2-matpool"
    observation = tmp_path / "lock-observation"
    environment["FAKE_LOCK_OBSERVATION"] = _to_bash_path(observation)
    environment["FAKE_EXPECT_LOCK_PATH"] = _to_bash_path(state_dir / "start.lock")

    completed = _run(repository, environment, "start")

    assert completed.returncode != 0
    assert "tmux start failed" in completed.stderr.lower()
    state = _state(repository)
    assert _from_bash_path(str(state["command_file"])).is_file()
    log = _from_bash_path(str(state["log_path"])).read_text(encoding="utf-8")
    assert "tmux start failed" in log.lower()
    assert not _from_bash_path(str(state["exit_status_path"])).exists()
    assert "active" not in completed.stdout.lower()
    assert observation.read_text(encoding="utf-8").splitlines() == ["locked"]
    assert not (state_dir / "start.lock").exists()


def test_launcher_state_directory_is_ignored() -> None:
    entries = REPO_ROOT.joinpath(".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".validation-v2-matpool/" in entries


def test_launcher_declares_strict_bash_and_never_offers_stop() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "set -Eeuo pipefail" in source
    assert re.search(r"\bstop\b", source, flags=re.IGNORECASE) is None
