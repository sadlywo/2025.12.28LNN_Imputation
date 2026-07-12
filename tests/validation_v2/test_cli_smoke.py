import json
import hashlib
import os
from collections.abc import Iterator
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import pandas as pd
import pytest
import torch
import yaml
import numpy as np

from validation_v2.experiments.runner import (
    build_execution_model,
    discover_oxiod_pairs,
    resolve_protocol_records,
)
from validation_v2.experiments.provenance import collect_provenance
from validation_v2.types import Recording


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def external_repo_tmp_path() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(
        prefix="validation-v2-cli-", dir=REPO_ROOT.parents[2]
    ) as directory:
        yield Path(directory)


def _cli(
    *arguments: str,
    cwd: Path = REPO_ROOT,
    timeout=None,
) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "validation_v2.cli", *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _server_runbook() -> str:
    return (REPO_ROOT / "docs" / "validation_v2_server_runbook.md").read_text(
        encoding="utf-8"
    )


def _runbook_python_blocks(runbook: str) -> list[str]:
    return re.findall(r"<<'PY'\n(.*?)\nPY", runbook, flags=re.DOTALL)


def _runbook_bash_function(runbook: str, name: str) -> str:
    match = re.search(
        rf"^{name}\(\) \{{\n(.*?)^\}}$",
        runbook,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert match is not None, name
    return match.group(1)


def _run_bash(tmp_path: Path, source: str, *, timeout: float = 10) -> subprocess.CompletedProcess:
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    assert bash is not None
    script = tmp_path / "runbook-contract.sh"
    script.write_bytes(source.encode("utf-8"))
    environment = os.environ.copy()
    environment["MSYS2_ARG_CONV_EXCL"] = "*"
    return subprocess.run(
        [
            bash, "-c", 'timeout "$1" bash "$2"', "runbook-contract",
            str(timeout), script.as_posix(),
        ],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
        timeout=timeout + 5, check=False,
    )


def test_server_runbook_launch_preconditions_never_reach_nohup(tmp_path: Path):
    launch = _runbook_bash_function(_server_runbook(), "launch_shard")
    root = tmp_path.as_posix()
    result = _run_bash(tmp_path, f'''set -Eeuo pipefail
launch_shard() {{
{launch}
}}
nohup() {{ printf 'called\\n' >> "$NOHUP_LOG"; return 0; }}
pgrep() {{ return 1; }}
export CONFIG="{root}/config.yaml"
export PLAN="{root}/plan.json"
export SHARDS_ROOT="{root}/shards"
export AUDIT_DIR="{root}/audit"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NOHUP_LOG="{root}/nohup.log"
mkdir -p "$SHARDS_ROOT" "$AUDIT_DIR"
printf 'config\\n' > "$CONFIG"
printf '{{}}\\n' > "$PLAN"
if launch_shard 008; then exit 90; fi
mkdir -p "$SHARDS_ROOT/000"
if launch_shard 000; then exit 91; fi
test ! -e "$NOHUP_LOG"
''')

    assert result.returncode == 0, result.stderr


def test_server_runbook_queue_returns_timeout_instead_of_looping(tmp_path: Path):
    queue = _runbook_bash_function(_server_runbook(), "run_queue")
    root = tmp_path.as_posix()
    result = _run_bash(tmp_path, f'''set -Eeuo pipefail
run_queue() {{
{queue}
}}
launch_shard() {{
  local shard="$1"
  mkdir -p "$SHARDS_ROOT/$shard"
  printf '{{"status":"started","group_runs":[]}}\\n' > "$SHARDS_ROOT/$shard/shard_execution.json"
  printf '%s\\n' "$$" > "$AUDIT_DIR/shard-$shard.pid"
}}
audit_active() {{ return 0; }}
python() {{ printf 'started 0\\n'; }}
export SHARDS_ROOT="{root}/shards"
export AUDIT_DIR="{root}/audit"
export QUEUE_MAX_SECONDS=1
export QUEUE_MAX_IDLE_SECONDS=1
export QUEUE_POLL_SECONDS=0.1
mkdir -p "$SHARDS_ROOT" "$AUDIT_DIR"
if run_queue 1 000; then rc=0; else rc=$?; fi
test "$rc" -eq 4
''', timeout=3)

    assert result.returncode == 0, result.stderr


def test_server_runbook_sampler_never_kills_a_mismatched_process(tmp_path: Path):
    stop = _runbook_bash_function(_server_runbook(), "stop_gpu_sampler")
    root = tmp_path.as_posix()
    stat_wrong = " ".join(["123", "(bash)", "S", *("0" for _ in range(18)), "778"])
    stat_right = " ".join(["123", "(bash)", "S", *("0" for _ in range(18)), "777"])
    result = _run_bash(tmp_path, f'''set -Eeuo pipefail
stop_gpu_sampler() {{
{stop}
}}
kill() {{ printf '%s\\n' "$1" >> "$KILL_LOG"; return 0; }}
wait() {{ return 0; }}
export AUDIT_DIR="{root}/audit"
export PROC_ROOT="{root}/proc"
export KILL_LOG="{root}/kill.log"
mkdir -p "$AUDIT_DIR" "$PROC_ROOT/123"
printf '123 777\\n' > "$AUDIT_DIR/gpu-test.pid"
printf '%s\\n' '{stat_wrong}' > "$PROC_ROOT/123/stat"
printf 'validation-v2-gpu-sampler-test\\0nvidia-smi\\0' > "$PROC_ROOT/123/cmdline"
declare -Ag GPU_SAMPLER_JOBS=([test]=123)
if stop_gpu_sampler test; then exit 90; fi
test ! -e "$KILL_LOG"
printf '%s\\n' '{stat_right}' > "$PROC_ROOT/123/stat"
stop_gpu_sampler test
grep -Fx '123' "$KILL_LOG"
''')

    assert result.returncode == 0, result.stderr


def test_server_runbook_strict_shell_explicitly_propagates_wait_failures():
    runbook = _server_runbook()

    assert "set -Eeuo pipefail" in runbook
    wait_calls = [
        line.strip()
        for line in runbook.splitlines()
        if re.match(r"wait_shard (?!\(\))", line.strip())
    ]
    queue_calls = [
        line.strip()
        for line in runbook.splitlines()
        if re.match(r"run_queue [12] ", line.strip())
    ]
    assert wait_calls and all("|| exit $?" in line for line in wait_calls)
    assert queue_calls and all("|| exit $?" in line for line in queue_calls)
    assert "wait_all_shards 000 001 002 003 004 005 006 007 || exit $?" in runbook
    assert not re.search(r"^\s+wait_shard\s+\S+\s*$", runbook, flags=re.MULTILINE)


def test_server_runbook_waits_check_markers_pids_completion_and_timeouts():
    runbook = _server_runbook()
    wait_groups = _runbook_bash_function(runbook, "wait_until_groups")
    wait_stage = _runbook_bash_function(runbook, "wait_stage_metrics")

    for block in (wait_groups, wait_stage):
        assert "kill -0" in block
        assert "failed" in block
        assert "group_runs" in block
        assert "tail -n" in block
        assert "deadline" in block
        assert "return 2" in block
        assert "return 3" in block
        assert "return 4" in block
    assert "completed before required group count" in wait_groups
    assert "14400" in wait_stage
    assert 'SystemExit(10)' in runbook
    for stage in ("STAGE2_RC", "STAGE4_RC"):
        case_block = re.search(
            rf'case "\${stage}" in(.*?)esac', runbook, flags=re.DOTALL
        )
        assert case_block is not None
        assert "10)" in case_block.group(1)
        assert "2|3|4)" in case_block.group(1)


def test_server_runbook_audits_active_marker_pid_process_and_gpu_each_minute():
    runbook = _server_runbook()
    audit = _runbook_bash_function(runbook, "audit_active")

    assert "60" in audit
    assert "shard_execution.json" in audit
    assert "group_runs" in audit
    assert "kill -0" in audit
    assert "pgrep -af" in audit
    assert "nvidia-smi" in audit
    assert "tee -a" in audit
    for function in ("wait_shard", "wait_until_groups", "wait_stage_metrics"):
        assert "audit_active" in _runbook_bash_function(runbook, function)


def test_server_runbook_waits_for_all_eight_shards_in_one_audited_scan():
    runbook = _server_runbook()
    wait_all = _runbook_bash_function(runbook, "wait_all_shards")

    assert 'local -a shards=("$@")' in wait_all
    assert 'for shard in "${shards[@]}"' in wait_all
    assert "shard_execution.json" in wait_all
    assert "group_runs" in wait_all
    assert "kill -0" in wait_all
    assert "failed" in wait_all and "return 2" in wait_all
    assert "missing" in wait_all and "started" in wait_all and "return 3" in wait_all
    assert "deadline" in wait_all and "return 4" in wait_all
    assert "sleep 60" in wait_all
    assert "pgrep -af" in wait_all and "nvidia-smi" in wait_all
    assert "tee -a" in wait_all
    assert "wait_all_shards 000 001 002 003 004 005 006 007 || exit $?" in runbook
    assert not re.search(
        r"for SHARD in 000 001 002 003 004 005 006 007; do\n\s+wait_shard",
        runbook,
    )


def test_server_runbook_legacy_stop_handles_zero_one_or_many_matches_safely():
    runbook = _server_runbook()
    legacy = next(
        block for block in re.findall(r"```bash\n(.*?)\n```", runbook, re.DOTALL)
        if "OLD_ROOT=" in block
    )

    assert legacy.startswith("set -Eeuo pipefail\n")
    assert 'case "${#OLD_MATCHES[@]}" in' in legacy
    assert "0)" in legacy and "no legacy process; not sending a signal" in legacy
    assert "1)" in legacy and "kill -INT" in legacy
    assert "*)" in legacy and "exit 2" in legacy
    assert '[[ "$OLD_PID" =~ ^[0-9]+$ ]]' in legacy
    assert 'grep -F -- "$OLD_ROOT"' in legacy
    assert '"$PROC_ROOT/$OLD_PID/cmdline"' in legacy
    assert 'OLD_STARTTIME="$(awk' in legacy
    assert '"python -m validation_v2.cli matrix"' in legacy
    assert '"$OLD_CMDLINE" == *"$OLD_ROOT"*' in legacy
    assert "ps -ww" in legacy and 'while test -r "$PROC_ROOT/$OLD_PID/stat"' in legacy
    assert 'CURRENT_STARTTIME' in legacy and '!= "$OLD_STARTTIME"' in legacy
    assert legacy.index("1)") < legacy.index("kill -INT") < legacy.index("*)")
    assert "rm " not in legacy and "rm -" not in legacy


def test_server_runbook_resume_proves_the_recorded_pid_is_absent_first():
    runbook = _server_runbook()
    resume = next(
        block for block in re.findall(r"```bash\n(.*?)\n```", runbook, re.DOTALL)
        if "OLD_SHARD_PID" in block
    )

    assert '[[ -f "$PID_FILE" && -r "$PID_FILE" ]]' in resume
    assert 'read -r OLD_SHARD_PID < "$PID_FILE"' in resume
    assert '[[ "$OLD_SHARD_PID" =~ ^[0-9]+$ ]]' in resume
    assert 'if kill -0 "$OLD_SHARD_PID"' in resume
    assert 'ps -p "$OLD_SHARD_PID"' in resume
    assert "exit 2" in resume
    assert '! kill -0 "$(cat' not in resume
    proof_end = resume.index("explicitly absent")
    rerun = resume.index("python -m validation_v2.cli shard")
    assert proof_end < rerun


def test_server_runbook_rebuilds_commit_scoped_paths_in_the_offline_shell():
    runbook = _server_runbook()
    offline = runbook.index("Open a new offline shell")
    expected_order = (
        'cd "$REPO"',
        'export COMMIT="$(git rev-parse HEAD)"',
        'export PREFLIGHT_DIR=',
        'export AUDIT_DIR=',
        'export PLAN=',
        'export SHARDS_ROOT=',
        'export FINAL_ROOT=',
    )
    positions = []
    for token in expected_order:
        match = re.search(re.escape(token), runbook[offline:])
        assert match is not None, token
        positions.append(offline + match.start())

    assert positions == sorted(positions)
    assert all("${COMMIT}" in runbook[position : position + 180] for position in positions[2:])


def test_server_runbook_asserts_every_pinned_runtime_version_exactly():
    runbook = _server_runbook()
    env_block = next(
        block for block in _runbook_python_blocks(runbook)
        if "torch.cuda.is_available" in block
    )

    assert "import importlib.metadata as md" in env_block
    assert 'platform.python_version().startswith("3.9.")' in env_block
    for package, version in {
        "numpy": "1.26.4",
        "pandas": "2.3.3",
        "scipy": "1.13.1",
        "PyYAML": "6.0.3",
        "pytest": "8.4.2",
        "ncps": "1.0.1",
    }.items():
        assert f'"{package}": "{version}"' in env_block
    assert '"torch": "2.3.1+cu121"' in env_block
    assert "actual == expected" in env_block
    assert 'torch.__version__ == "2.3.1+cu121"' in env_block
    assert 'torch.version.cuda == "12.1"' in env_block
    assert '"4090 D" in torch.cuda.get_device_name(0)' in env_block


def test_server_runbook_metrics_code_compiles_and_uses_real_group_durations():
    runbook = _server_runbook()
    python_blocks = _runbook_python_blocks(runbook)
    assert python_blocks
    for index, block in enumerate(python_blocks):
        compile(block, f"validation_v2_server_runbook.py:{index}", "exec")
    metrics_block = next(
        (block for block in python_blocks if "median_group_seconds" in block), None
    )

    assert metrics_block is not None
    assert 'marker["started_at"]' in metrics_block
    assert 'marker["group_runs"]' in metrics_block
    assert 'binding["run_ids"][0]' in metrics_block
    assert '"test_evaluation.json"' in metrics_block
    assert 'ledger["completed_at"]' in metrics_block
    assert "completed - previous" in metrics_block
    assert "previous = completed" in metrics_block
    assert "statistics.median(durations)" in metrics_block
    assert "max(" in metrics_block and "peak_gpu_memory_ratio" in metrics_block
    assert re.search(
        r"assert\s+.*groups_per_hour.*>=.*groups_per_hour.*\*\s*1\.5",
        metrics_block,
    )
    assert re.search(
        r"assert\s+.*median_group_seconds.*<.*median_group_seconds.*\*\s*1\.8",
        metrics_block,
    )
    assert re.search(r"assert\s+.*peak_gpu_memory_ratio.*<\s*0\.8", metrics_block)


def test_server_runbook_has_executable_staged_gates_and_bounded_fallback_queues():
    runbook = _server_runbook()

    for function in (
        "launch_shard", "wait_shard", "run_queue", "start_gpu_sampler",
        "stop_gpu_sampler", "wait_stage_metrics", "wait_all_shards",
    ):
        assert re.search(rf"^{function}\(\) \{{", runbook, flags=re.MULTILINE)
    assert runbook.index("launch_shard 000") < runbook.index("launch_shard 001")
    assert "baseline-1worker.json" in runbook
    assert "stage-2worker-start.txt" in runbook
    assert "stage-2worker-metrics.json" in runbook
    assert "stage-4worker-start.txt" in runbook
    assert "stage-4worker-metrics.json" in runbook
    assert "stage-8worker-metrics.json" in runbook
    assert "launch_shard 002" in runbook and "launch_shard 003" in runbook
    assert all(f"launch_shard {index:03d}" in runbook for index in range(4, 8))
    assert "run_queue 1 002 003 004 005 006 007" in runbook
    assert "run_queue 2 004 005 006 007" in runbook
    assert "nvidia-smi" in runbook and "tee -a" in runbook and "sleep 10" in runbook
    assert "wait_all_shards 000 001 002 003 004 005 006 007" in runbook


def test_server_runbook_pins_the_offline_linux_shard_plan_contract():
    runbook = _server_runbook()
    bash_blocks = re.findall(r"```bash\n(.*?)\n```", runbook, flags=re.DOTALL)
    turbo_blocks = [block for block in bash_blocks if "source /etc/network_turbo" in block]

    for variable in (
        "REPO", "CONFIG", "AUDIT_DIR", "PLAN", "SHARDS_ROOT", "FINAL_ROOT",
    ):
        assert f"export {variable}=" in runbook
    assert 'export CONFIG="/root/' in runbook
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in runbook
    assert "conda activate /root/miniconda3/envs/pinn_imu" in runbook
    assert "shard-plan" in runbook
    assert "--shard-count 8" in runbook
    assert "total_groups" in runbook and "175" in runbook
    assert "total_cells" in runbook and "4095" in runbook
    assert "shard_count" in runbook and "8" in runbook
    assert "test_linux_rename_noreplace_survives_real_directory_race" in runbook
    assert "1 passed" in runbook and "skipped" in runbook
    assert "python -m pytest" in runbook
    assert turbo_blocks
    assert all(
        any(token in block for token in ("git clone", "git fetch", "pip install"))
        and "validation_v2.cli shard" not in block
        for block in turbo_blocks
    )
    assert "training requires no network" in runbook.lower()


def test_server_runbook_stages_independent_shards_two_four_eight_and_resumes_safely():
    runbook = _server_runbook()

    assert "2 -> 4 -> 8" in runbook
    assert "shard 000" in runbook and "shard 001" in runbook
    assert "002 003" in runbook and "004 005 006 007" in runbook
    assert '"$SHARDS_ROOT/$SHARD"' in runbook
    assert '"$AUDIT_DIR/shard-$SHARD.log"' in runbook
    assert "nohup" in runbook and "--device cuda" in runbook
    assert "python -m validation_v2.cli shard" in runbook
    assert "completed groups/hour" in runbook
    assert ">= 50%" in runbook and "< 80%" in runbook
    assert "GPU memory" in runbook and "failed" in runbook
    assert "shard_execution.json" in runbook
    assert '"completed"' in runbook and '"started"' in runbook and '"failed"' in runbook
    assert "60" in runbook and "nvidia-smi" in runbook and "pgrep" in runbook
    assert ".shard_execution.lock" in runbook
    assert "full assigned group list" in runbook
    assert "stdout" in runbook and "final JSON" in runbook
    assert "same commit, config, plan, device, shard index, and shard root" in runbook
    assert "new formal campaign" in runbook


def test_server_runbook_never_mixes_legacy_or_replacement_roots_and_strictly_publishes():
    runbook = _server_runbook()

    assert "fcf81f8" in runbook
    assert "kill -INT" in runbook and "pgrep" in runbook
    assert "Never delete the old root" in runbook
    assert "must not be mixed" in runbook
    assert "test ! -e \"$SHARDS_ROOT\"" in runbook
    assert "test ! -e \"$FINAL_ROOT\"" in runbook
    assert "python -m validation_v2.cli merge-shards" in runbook
    assert "python -m validation_v2.experiments.validate_artifacts" in runbook
    assert 'report["status"] == "complete"' in runbook
    assert "python -m validation_v2.cli summarize" in runbook
    assert "--required-seeds 2026 2027 2028 2029 2030" in runbook
    for artifact in (
        "matrix_execution.json", "validation_report.json", "summary.csv",
        "summary.json", "run.json", "history.json", "best.pt",
        "checkpoint.json", "test_evaluation.json", "per_record_metrics.csv",
    ):
        assert artifact in runbook


def test_server_runbook_contains_no_ssh_secret_and_checks_secret_leakage():
    runbook = _server_runbook()

    assert "<enter interactively" not in runbook
    assert "connect.westb.seetacloud.com" not in runbook
    assert not re.search(r"(?i)(password|passwd)\s*[:=]\s*\S+", runbook)
    assert "git grep" in runbook
    assert "AUDIT_DIR" in runbook and "grep -R" in runbook


def test_shard_plan_writes_formal_server_plan_as_one_canonical_json_line(
    tmp_path: Path,
):
    plan_path = tmp_path / "server-plan.json"
    (tmp_path / "server_full.yaml").write_text(
        "models: [malicious-cwd-shadow]\n", encoding="utf-8"
    )

    result = _cli(
        "shard-plan",
        "--config",
        "server_full.yaml",
        "--shard-count",
        "8",
        "--output",
        str(plan_path),
        "--device",
        "cuda",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr.decode()
    assert result.stderr == b""
    assert len(result.stdout.decode("utf-8").splitlines()) == 1
    plan = json.loads(result.stdout)
    assert plan == json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["git_commit"] == _git_head()
    assert plan["device"] == "cuda"
    assert plan["shard_count"] == 8
    assert plan["total_groups"] == 175
    assert plan["total_cells"] == 4095
    assert len(plan["plan_sha256"]) == 64


def test_shard_plan_rerun_is_idempotent_but_different_plan_does_not_clobber(
    tmp_path: Path,
):
    plan_path = tmp_path / "server-plan.json"
    arguments = (
        "shard-plan",
        "--config",
        "server_full.yaml",
        "--shard-count",
        "8",
        "--output",
        str(plan_path),
        "--device",
        "cpu",
    )

    first = _cli(*arguments)
    assert first.returncode == 0, first.stderr.decode()
    before = plan_path.read_bytes()
    second = _cli(*arguments)
    different = _cli(*arguments[:-5], "7", *arguments[-4:])

    assert second.returncode == 0
    assert first.stdout == second.stdout
    assert second.stderr == b""
    assert different.returncode == 2
    assert different.stdout == b""
    assert different.stderr.startswith(b"validation-v2: ")
    assert b"different" in different.stderr or b"shard_count" in different.stderr
    assert plan_path.read_bytes() == before


def test_shard_plan_rejects_broken_link_output_without_hanging_or_recreating_target(
    tmp_path: Path,
):
    output = tmp_path / "plan.json"
    target = tmp_path / "missing-target"
    target.mkdir()
    try:
        output.symlink_to(target, target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"symlink creation is unavailable: {error}")
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(output), str(target)],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            pytest.skip(f"junction creation is unavailable: {completed.stderr}")
    target.rmdir()
    assert os.path.lexists(output)
    before = os.lstat(output)

    result = _cli(
        "shard-plan", "--config", "server_full.yaml", "--shard-count", "8",
        "--output", str(output), "--device", "cpu", timeout=5,
    )

    after = os.lstat(output)
    assert result.returncode == 2
    assert result.stdout == b""
    assert len(result.stderr.decode("utf-8").splitlines()) == 1
    assert b"linked" in result.stderr or b"regular" in result.stderr
    assert (after.st_mode, getattr(after, "st_file_attributes", 0)) == (
        before.st_mode, getattr(before, "st_file_attributes", 0)
    )
    assert not target.exists()


@pytest.mark.parametrize("shard_index", ["-1", "8"])
def test_shard_rejects_negative_and_out_of_range_index(
    tmp_path: Path, shard_index: str,
):
    plan_path = tmp_path / "server-plan.json"
    planned = _cli(
        "shard-plan", "--config", "server_full.yaml", "--shard-count", "8",
        "--output", str(plan_path), "--device", "cpu",
    )
    assert planned.returncode == 0, planned.stderr.decode()

    result = _cli(
        "shard", "--config", "server_full.yaml", "--plan", str(plan_path),
        "--shard-index", shard_index, "--output-root", str(tmp_path / "shard"),
        "--device", "cpu",
    )

    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr.startswith(b"validation-v2: ")
    assert b"shard_index" in result.stderr


@pytest.mark.parametrize(
    "arguments",
    [
        ("shard-plan", "--config", "server_full.yaml", "--shard-count", "0",
         "--output", "plan.json", "--device", "cpu"),
        ("shard-plan", "--config", "server_full.yaml", "--shard-count", "8",
         "--output", "plan.json", "--device", "auto"),
        ("shard", "--config", "server_full.yaml", "--plan", "missing.json",
         "--shard-index", "-1", "--output-root", "shards/000", "--device", "cpu"),
    ],
)
def test_shard_commands_reject_invalid_arguments(arguments: tuple[str, ...]):
    result = _cli(*arguments)

    assert result.returncode == 2
    assert result.stdout == b""


def test_two_cli_shards_execute_and_merge_through_strict_validator(
    external_repo_tmp_path: Path,
):
    tmp_path = external_repo_tmp_path
    if not (REPO_ROOT / "Oxford Dataset" / "handbag-1" / "imu1.csv").is_file():
        pytest.skip("real OxIOD files are not available")
    config_value = "configs/validation_v2/smoke.yaml"
    shadow = tmp_path / config_value
    shadow.parent.mkdir(parents=True)
    shadow.write_text("models: [malicious-cwd-shadow]\n", encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    shards_root = tmp_path / "shards"
    merged_root = tmp_path / "merged"
    dirty_text = subprocess.check_output(
        [
            "git", "-C", str(REPO_ROOT), "status", "--porcelain=v1",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    expected_dirty_digest = (
        hashlib.sha256(dirty_text.encode("utf-8")).hexdigest()
        if dirty_text else ""
    )

    planned = _cli(
        "shard-plan", "--config", config_value, "--shard-count", "2",
        "--output", str(plan_path), "--device", "cpu",
        cwd=tmp_path,
    )
    assert planned.returncode == 0, planned.stderr.decode()
    assert json.loads(planned.stdout)["total_groups"] == 3

    for shard_index in range(2):
        executed = _cli(
            "shard", "--config", config_value, "--plan", str(plan_path),
            "--shard-index", str(shard_index), "--output-root",
            str(shards_root / f"{shard_index:03d}"), "--device", "cpu",
            cwd=tmp_path,
        )
        assert executed.returncode == 0, executed.stderr.decode()
        assert executed.stderr == b""
        assert len(executed.stdout.decode("utf-8").splitlines()) == 1
        report = json.loads(executed.stdout)
        assert report["status"] == "completed"
        assert report["shard_index"] == shard_index

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for manifest_path in shards_root.glob("*/*/run.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["git_commit"] == plan["git_commit"]
        assert manifest["dirty_state_digest"] == expected_dirty_digest

    merged = _cli(
        "merge-shards", "--config", config_value, "--plan", str(plan_path),
        "--shards-root", str(shards_root), "--output-root", str(merged_root),
        cwd=tmp_path,
    )

    assert merged.returncode == 0, merged.stderr.decode()
    assert merged.stderr == b""
    assert len(merged.stdout.decode("utf-8").splitlines()) == 1
    assert json.loads(merged.stdout)["status"] == "complete"
    validation_report = json.loads(
        (merged_root / "validation_report.json").read_text(encoding="utf-8")
    )
    assert validation_report["status"] == "complete"
    assert not (merged_root / "shard_execution.json").exists()


def test_merge_shards_error_is_one_prefixed_stderr_line(tmp_path: Path):
    result = _cli(
        "merge-shards", "--config", "smoke.yaml", "--plan",
        str(tmp_path / "missing-plan.json"), "--shards-root",
        str(tmp_path / "missing-shards"), "--output-root",
        str(tmp_path / "merged"),
    )

    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr.startswith(b"validation-v2: ")
    assert len(result.stderr.decode("utf-8").splitlines()) == 1


def test_missing_config_from_temporary_cwd_is_a_clear_exit_two(tmp_path: Path):
    result = _cli(
        "shard-plan", "--config", "missing.yaml", "--shard-count", "1",
        "--output", str(tmp_path / "plan.json"), "--device", "cpu",
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr.decode("utf-8").splitlines() == [
        "validation-v2: config file does not exist: missing.yaml"
    ]


def test_server_matrix_dry_run_is_byte_stable_and_complete():
    arguments = ("matrix", "--config", "server_full.yaml", "--dry-run")

    first = _cli(*arguments)
    second = _cli(*arguments)

    assert first.returncode == 0, first.stderr.decode()
    assert first.stdout == second.stdout
    lines = first.stdout.decode("utf-8").splitlines()
    header = json.loads(lines[0])
    combinations = [json.loads(line) for line in lines[1:]]
    assert header == {
        "command": "matrix",
        "combination_count": len(combinations),
        "dry_run": True,
    }
    assert {item["seed"] for item in combinations} == {
        2026,
        2027,
        2028,
        2029,
        2030,
    }
    irregular = [item for item in combinations if item["case_type"] == "irregular"]
    assert irregular
    assert all(item["irregular_method"] == "interval_jitter" for item in irregular)
    assert all(item["value_topology"] == "point" for item in irregular)
    assert all(item["value_requested_fraction"] == 0.3 for item in irregular)


def test_server_config_declares_real_scenarios_and_bounded_execution_inputs():
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "validation_v2" / "server_full.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["protocols"] == [
        "strict_file",
        "scenario_holdout:handbag",
        "scenario_holdout:handheld",
        "scenario_holdout:running",
        "scenario_holdout:slow_walking",
        "scenario_holdout:trolley",
        "scenario_holdout:user-2",
    ]
    assert config["irregular_cases"] == [
        {
            "method": "interval_jitter",
            "requested_irregularity": 0.2,
            "value_topology": "point",
            "value_requested_fraction": 0.3,
        }
    ]
    assert config["max_train_windows"] > 0
    assert config["max_eval_samples"] is None
    assert config["split_seed"] == 2026


def test_resolved_provenance_changes_with_hidden_size_or_learning_rate(tmp_path: Path):
    from validation_v2.experiments.runner import resolved_execution_config
    from validation_v2.experiments.train import resume_run, train_one_run

    source = yaml.safe_load(
        (REPO_ROOT / "configs" / "validation_v2" / "smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    source["output_root"] = "ignored/location"
    source["_execution_conditions"] = ["internal"]
    source["_skip_descriptive_summary"] = True
    conditions = [{"topology": "point", "requested_fraction": 0.3}]

    base = resolved_execution_config(
        source, model="linear", seed=2026, protocol="strict_file",
        conditions=conditions, resolved_device="cpu",
    )
    hidden_source = {**source, "hidden_size": source["hidden_size"] + 1}
    learning_source = {**source, "learning_rate": source["learning_rate"] * 2}
    hidden = resolved_execution_config(
        hidden_source, model="linear", seed=2026, protocol="strict_file",
        conditions=conditions, resolved_device="cpu",
    )
    learning = resolved_execution_config(
        learning_source, model="linear", seed=2026, protocol="strict_file",
        conditions=conditions, resolved_device="cpu",
    )

    manifests = [collect_provenance(item, seed=2026) for item in (base, hidden, learning)]
    run_ids = {manifest["run_id"] for manifest in manifests}
    assert len(run_ids) == 3
    assert "output_root" not in base["source_config"]
    assert not any(key.startswith("_") for key in base["source_config"])
    assert base["source_config"]["epochs"] == source["epochs"]
    assert base["source_config"]["batch_size"] == source["batch_size"]
    assert base["evaluation_scope"] == "bounded_overlap_slice"
    server_source = {**source, "max_eval_samples": None}
    server = resolved_execution_config(
        server_source, model="linear", seed=2026, protocol="strict_file",
        conditions=conditions, resolved_device="cpu",
    )
    assert server["evaluation_scope"] == "full_overlap_record"

    model = torch.nn.Linear(1, 1)
    metadata = train_one_run(
        tmp_path / manifests[0]["run_id"],
        manifests[0],
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        train_loader=[],
        validation_loader=[],
        epochs=1,
        train_epoch=lambda *_: {"missing_rmse": 1.0},
        evaluate_epoch=lambda *_: {"missing_rmse": 1.0},
    )
    with pytest.raises(ValueError, match="config hash|run_id"):
        resume_run(
            tmp_path / manifests[0]["run_id"],
            manifests[1],
            metadata["checkpoint_sha256"],
        )


def test_split_seed_freezes_protocol_across_all_training_seeds():
    from validation_v2.experiments.runner import resolve_configured_records

    config = {"split_seed": 2026}
    pairs = discover_oxiod_pairs(REPO_ROOT / "Oxford Dataset")
    manifests = [
        resolve_configured_records(
            config,
            data_root=REPO_ROOT / "Oxford Dataset",
            protocol="strict_file",
            training_seed=training_seed,
        )
        for training_seed in range(2026, 2031)
    ]

    split_views = [
        [(item["recording_id"], item["split"]) for item in manifest]
        for manifest in manifests
    ]
    test_ids = [
        [item["recording_id"] for item in manifest if item["split"] == "test"]
        for manifest in manifests
    ]
    split_hashes = [
        hashlib.sha256(
            json.dumps(view, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for view in split_views
    ]
    assert all(view == split_views[0] for view in split_views[1:])
    assert all(ids == test_ids[0] for ids in test_ids[1:])
    assert len(set(split_hashes)) == 1


def test_reverse_dt_aligns_intervals_to_reversed_input():
    from validation_v2.experiments.runner import reverse_aligned_dt

    dt = torch.tensor([[0.25, 0.1, 0.7, 0.2]], dtype=torch.float64)

    reversed_dt = reverse_aligned_dt(dt)

    torch.testing.assert_close(
        reversed_dt,
        torch.tensor([[0.2, 0.2, 0.7, 0.1]], dtype=torch.float64),
    )


def test_unlimited_overlap_slice_keeps_every_available_sample():
    from validation_v2.experiments.runner import _slice_recording

    imu_time = np.arange(150, dtype=np.float64) * 0.01
    recording = Recording(
        id="synthetic/imu1",
        imu_time_s=imu_time,
        imu_six=np.zeros((150, 6), dtype=np.float64),
        vicon_time_s=np.array([0.2, 1.2], dtype=np.float64),
        vicon_position_m=np.zeros((2, 3), dtype=np.float64),
        vicon_quaternion_xyzw=np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        overlap_s=(0.2, 1.2),
        metadata={},
    )

    unlimited_time, unlimited_values = _slice_recording(recording, None)
    zero_time, zero_values = _slice_recording(recording, 0)

    assert len(unlimited_time) == len(zero_time) == 101
    assert unlimited_values.shape == zero_values.shape == (101, 6)
    assert unlimited_time[0] == pytest.approx(0.2)
    assert unlimited_time[-1] == pytest.approx(1.2)


def _synthetic_training_recording(
    recording_id: str = "synthetic/train", *, length: int = 240
) -> Recording:
    intervals = 0.01 + (np.arange(length, dtype=np.float64) % 5) * 0.001
    time = np.cumsum(intervals)
    values = np.column_stack(
        [np.arange(length, dtype=np.float64) + channel for channel in range(6)]
    )
    return Recording(
        id=recording_id,
        imu_time_s=time,
        imu_six=values,
        vicon_time_s=np.array([time[0], time[-1]], dtype=np.float64),
        vicon_position_m=np.zeros((2, 3), dtype=np.float64),
        vicon_quaternion_xyzw=np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64
        ),
        overlap_s=(float(time[0]), float(time[-1])),
        metadata={},
    )


@pytest.mark.parametrize("topology", ["point", "block", "channel"])
@pytest.mark.parametrize("rate", [0.1, 0.2, 0.3, 0.4])
def test_every_training_window_has_requested_missingness(topology: str, rate: float):
    from validation_v2.data.normalization import RobustTrainScaler
    from validation_v2.experiments.runner import _windows

    recording = _synthetic_training_recording()
    scaler = RobustTrainScaler(
        center_=np.zeros(6), scale_=np.ones(6), training_ids=(recording.id,)
    )

    windows = _windows(
        [recording], scaler, seq_len=30, maximum_windows=6,
        rate=rate, seed=2026, topology=topology,
    )

    assert len(windows) == 6
    realized = [float((window.mask == 0).float().mean()) for window in windows]
    assert all(value > 0 for value in realized)
    if topology in {"point", "block"}:
        expected = round(30 * (6 if topology == "point" else 1) * rate) / (
            30 * (6 if topology == "point" else 1)
        )
    else:
        expected = max(1, int(6 * rate)) / 6
    assert realized == pytest.approx([expected] * 6)


def test_training_windows_preserve_timing_boundaries_and_hidden_target_invariance():
    from validation_v2.data.normalization import RobustTrainScaler
    from validation_v2.experiments.runner import _windows

    first = _synthetic_training_recording("synthetic/first", length=60)
    second = _synthetic_training_recording("synthetic/second", length=60)
    second = Recording(
        **{**second.__dict__, "imu_six": second.imu_six + 1000.0}
    )
    scaler = RobustTrainScaler(
        center_=np.zeros(6), scale_=np.ones(6),
        training_ids=tuple(sorted((first.id, second.id))),
    )
    windows = _windows(
        [first, second], scaler, seq_len=30, maximum_windows=4,
        rate=0.1, seed=2026, topology="block",
    )

    expected_dt = []
    for recording in (first, second):
        dt = np.empty(60, dtype=np.float32)
        dt[1:] = np.diff(recording.imu_time_s).astype(np.float32)
        dt[0] = dt[1]
        expected_dt.extend((dt[:30], dt[30:]))
    for window, dt in zip(windows, expected_dt):
        np.testing.assert_allclose(window.dt.numpy(), dt)
    assert all(float(window.target.mean()) < 500 for window in windows[:2])
    assert all(float(window.target.mean()) > 500 for window in windows[2:])

    changed_values = first.imu_six.copy()
    first_masks = torch.cat([window.mask for window in windows[:2]]).numpy()
    changed_values[first_masks == 0] += 10_000.0
    changed = Recording(**{**first.__dict__, "imu_six": changed_values})
    changed_windows = _windows(
        [changed], scaler, seq_len=30, maximum_windows=2,
        rate=0.1, seed=2026, topology="block",
    )
    for original, hidden_changed in zip(windows[:2], changed_windows):
        assert torch.equal(original.mask, hidden_changed.mask)
        assert torch.equal(original.features, hidden_changed.features)


def test_formal_like_block_training_callback_has_missing_targets_in_every_batch():
    from validation_v2.data.normalization import RobustTrainScaler
    from validation_v2.experiments.runner import _batches, _epoch_callbacks, _windows

    recording = _synthetic_training_recording()
    scaler = RobustTrainScaler(
        center_=np.zeros(6), scale_=np.ones(6), training_ids=(recording.id,)
    )
    batches = _batches(
        _windows(
            [recording], scaler, seq_len=30, maximum_windows=6,
            rate=0.1, seed=2026, topology="block",
        ),
        batch_size=2,
    )
    model = build_execution_model("bilnn", hidden_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_epoch, _ = _epoch_callbacks("bilnn", torch.device("cpu"))

    assert len(batches) == 3
    assert all(torch.any(batch.mask == 0) for batch in batches)
    metrics = train_epoch(model, optimizer, batches, epoch=1)
    assert np.isfinite(metrics["missing_rmse"])


def test_stitched_neural_prediction_windows_cover_and_average_every_sample():
    from validation_v2.experiments.runner import predict_stitched_sequence

    class WindowPositionSpy:
        name = "bilstm"

        def __init__(self):
            self.lengths = []

        def predict(self, features, mask, dt, reported_model=None):
            del mask, dt, reported_model
            self.lengths.append(features.shape[1])
            local = torch.arange(
                features.shape[1], dtype=features.dtype, device=features.device
            )
            return local[None, :, None].expand(features.shape[0], -1, 6)

    spy = WindowPositionSpy()
    length, seq_len = 13, 4
    features = torch.zeros(length, 25)
    mask = torch.zeros(length, 6)
    dt = torch.full((length,), 0.1)

    prediction, coverage = predict_stitched_sequence(
        spy, features, mask, dt, seq_len=seq_len, batch_size=2,
        return_coverage=True,
    )

    starts = [0, 2, 4, 6, 8, 9]
    expected_sum = torch.zeros(length)
    expected_count = torch.zeros(length)
    for start in starts:
        expected_sum[start : start + seq_len] += torch.arange(seq_len)
        expected_count[start : start + seq_len] += 1
    assert prediction.shape == (length, 6)
    torch.testing.assert_close(prediction[:, 0], expected_sum / expected_count)
    torch.testing.assert_close(coverage, expected_count)
    assert torch.all(coverage > 0)
    assert spy.lengths and max(spy.lengths) == seq_len


def test_irregular_linear_signal_is_resampled_at_jittered_timestamps():
    from validation_v2.experiments.runner import resample_physical_time

    source_time = np.array([0.0, 0.2, 0.5, 0.7, 1.0])
    physical = np.column_stack(
        [(axis + 1) * source_time for axis in range(6)]
    )
    query_time = np.array([0.0, 0.1, 0.4, 0.85, 1.0])

    resampled = resample_physical_time(source_time, physical, query_time)

    np.testing.assert_allclose(
        resampled,
        np.column_stack([(axis + 1) * query_time for axis in range(6)]),
    )
    assert query_time[0] == source_time[0]
    assert query_time[-1] == source_time[-1]


def test_matrix_dry_run_ignores_mapping_and_axis_list_order(tmp_path: Path):
    first_config = tmp_path / "first.yaml"
    second_config = tmp_path / "second.yaml"
    first_config.write_text(
        "models: [hybrid, linear]\n"
        "seeds: [2027, 2026]\n"
        "topologies: [block, point]\n"
        "rates: [0.3, 0.1]\n"
        "protocols: [strict_file]\n",
        encoding="utf-8",
    )
    second_config.write_text(
        "protocols: [strict_file]\n"
        "rates: [0.1, 0.3]\n"
        "topologies: [point, block]\n"
        "seeds: [2026, 2027]\n"
        "models: [linear, hybrid]\n",
        encoding="utf-8",
    )

    first = _cli("matrix", "--config", str(first_config), "--dry-run")
    second = _cli("matrix", "--config", str(second_config), "--dry-run")

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout


def test_matrix_bad_config_has_clear_stderr_and_nonzero_exit(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: [linear]\n", encoding="utf-8")

    result = _cli("matrix", "--config", str(bad), "--dry-run")

    assert result.returncode != 0
    assert b"missing matrix axes" in result.stderr
    assert result.stdout == b""


def test_smoke_config_declares_the_bounded_real_oxiod_protocol():
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "validation_v2" / "smoke.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["data_root"] == "Oxford Dataset"
    assert config["output_root"] == "results/validation_v2/smoke"
    assert config["seeds"] == [2026]
    assert (config["epochs"], config["batch_size"], config["seq_len"]) == (1, 4, 30)
    assert config["models"] == ["linear", "bilstm", "hybrid"]
    assert config["topologies"] == ["point"]
    assert config["rates"] == [0.3]
    assert config["max_train_windows"] > 0
    assert config["max_eval_samples"] > 0
    splits = config["recordings"]
    assert [record["split"] for record in splits].count("train") == 2
    assert [record["split"] for record in splits].count("validation") == 1
    assert [record["split"] for record in splits].count("test") == 1
    assert [(record["imu"], record["vicon"]) for record in splits] == [
        (f"handbag-1/imu{index}.csv", f"handbag-1/vi{index}.csv")
        for index in range(1, 5)
    ]


def test_real_smoke_writes_frozen_runs_and_descriptive_summary(tmp_path: Path):
    data_root = REPO_ROOT / "Oxford Dataset"
    if not (data_root / "handbag-1" / "imu1.csv").is_file():
        pytest.skip("real OxIOD files are not available")
    output_root = tmp_path / "real-smoke"

    result = _cli(
        "smoke",
        "--config",
        "smoke.yaml",
        "--output-root",
        str(output_root),
        "--device",
        "cpu",
    )

    assert result.returncode == 0, result.stderr.decode()
    report = json.loads(result.stdout)
    assert report["status"] == "completed"
    assert report["real_data"] is True
    assert report["descriptive_only"] is True
    assert report["n_recordings"] == 1
    manifest = pd.read_csv(output_root / "split_manifest.csv")
    assert manifest["split"].value_counts().to_dict() == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }
    scaler = json.loads((output_root / "scaler.json").read_text(encoding="utf-8"))
    assert scaler["training_ids"] == ["handbag-1/imu1", "handbag-1/imu2"]
    assert scaler["channel_order"] == [
        "rotation_rate_x",
        "rotation_rate_y",
        "rotation_rate_z",
        "user_acc_x",
        "user_acc_y",
        "user_acc_z",
    ]
    assert scaler["split_hash"] == report["split_hash"]
    run_dirs = sorted(path.parent for path in output_root.glob("*/run.json"))
    assert len(run_dirs) == 3
    all_models = set()
    for run_dir in run_dirs:
        for name in (
            "run.json",
            "history.json",
            "best.pt",
            "checkpoint.json",
            "test_evaluation.json",
            "per_record_metrics.csv",
        ):
            assert (run_dir / name).is_file(), f"missing {name} in {run_dir.name}"
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        all_models.add(run["config"]["model"])
        metrics = pd.read_csv(run_dir / "per_record_metrics.csv")
        assert set(metrics["recording_id"]) == {"handbag-1/imu4"}
        assert {
            "reconstruction_normalized",
            "reconstruction_physical",
            "ate_rmse_m",
            "rpe_rmse_m",
            "endpoint_drift_m",
            "velocity_rmse_mps",
        }.issubset(set(metrics["metric"]))
        assert any(metrics["metric"].str.startswith("delta_"))
    assert all_models == {"linear", "bilstm", "hybrid"}
    smoke_summary = json.loads(
        (output_root / "smoke_summary.json").read_text(encoding="utf-8")
    )
    assert smoke_summary["descriptive_only"] is True
    assert smoke_summary["n_recordings"] == 1
    assert (output_root / "summary.csv").is_file()
    assert (output_root / "summary.json").is_file()


def test_matrix_explicit_limit_runs_real_cell_and_marks_partial(tmp_path: Path):
    if not (REPO_ROOT / "Oxford Dataset" / "handbag-1" / "imu1.csv").is_file():
        pytest.skip("real OxIOD files are not available")
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "validation_v2" / "smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["models"] = ["linear"]
    config["rates"] = [0.3, 0.4]
    config_path = tmp_path / "mini-matrix.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_root = tmp_path / "matrix-output"

    result = _cli(
        "matrix",
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
        "--device",
        "cpu",
        "--max-combinations",
        "1",
    )

    assert result.returncode == 0, result.stderr.decode()
    report = json.loads(result.stdout)
    assert report["partial"] is True
    assert report["selected_cells"] == 1
    assert report["total_cells"] == 2
    marker = json.loads(
        (output_root / "matrix_execution.json").read_text(encoding="utf-8")
    )
    assert marker["partial"] is True
    assert len(list(output_root.glob("*/per_record_metrics.csv"))) == 1

    summary = _cli("summarize", "--root", str(output_root), "--baseline", "linear")
    assert summary.returncode != 0
    assert b"partial" in summary.stderr


def test_server_scan_finds_all_real_pairs_and_normalizes_scenarios():
    pairs = discover_oxiod_pairs(REPO_ROOT / "Oxford Dataset")

    assert len(pairs) == 45
    assert len({pair["recording_id"] for pair in pairs}) == 45
    assert {pair["scenario"] for pair in pairs} == {
        "handbag",
        "handheld",
        "running",
        "slow_walking",
        "trolley",
        "user-2",
    }
    assert all(Path(pair["imu_path"]).is_file() for pair in pairs)
    assert all(Path(pair["vicon_path"]).is_file() for pair in pairs)


def test_server_protocol_splits_are_disjoint_and_hold_out_complete_scenario():
    pairs = discover_oxiod_pairs(REPO_ROOT / "Oxford Dataset")

    strict = resolve_protocol_records(pairs, "strict_file", seed=2026)
    holdout = resolve_protocol_records(
        pairs, "scenario_holdout:handbag", seed=2026
    )

    assert len(strict) == len(holdout) == 45
    assert set(item["split"] for item in strict) == {"train", "validation", "test"}
    assert len({item["recording_id"] for item in strict}) == 45
    assert all(
        (item["split"] == "test") == (item["scenario"] == "handbag")
        for item in holdout
    )
    assert {item["split"] for item in holdout if item["scenario"] != "handbag"} == {
        "train",
        "validation",
    }


@pytest.mark.parametrize(
    "model_name",
    [
        "linear",
        "locf",
        "bilstm",
        "bilnn",
        "hybrid",
        "equal_average",
        "fixed_gate_0",
        "fixed_gate_0.5",
        "fixed_gate_1",
    ],
)
def test_every_server_model_constructs_and_forwards(model_name: str):
    model = build_execution_model(model_name, hidden_size=2)
    features = torch.zeros(1, 6, 25)
    target = torch.zeros(1, 6, 6)
    mask = torch.ones_like(target)
    mask[:, 2, :] = 0
    dt = torch.full((1, 6), 0.01)

    prediction = model.predict(features, mask, dt)

    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()


def test_gate_labels_share_identical_hybrid_branch_predictions():
    from validation_v2.models.hybrid import HybridComponents

    class BranchSpy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward_components(self, features, forward_dt, reverse_dt, observed, mask):
            self.inputs.append(
                (features.clone(), forward_dt.clone(), reverse_dt.clone(), mask.clone())
            )
            lnn = torch.full_like(observed, 2.0)
            lstm = torch.full_like(observed, 4.0)
            gate = torch.full_like(observed, 0.25)
            raw = gate * lnn + (1.0 - gate) * lstm
            return HybridComponents(lnn, lstm, gate, raw, raw)

    model = build_execution_model("hybrid", hidden_size=2)
    spy = BranchSpy()
    model.core = spy
    features = torch.zeros(1, 6, 25)
    mask = torch.zeros(1, 6, 6)
    dt = torch.full((1, 6), 0.1)
    labels = [
        "hybrid", "equal_average", "fixed_gate_0", "fixed_gate_0.5", "fixed_gate_1"
    ]

    predictions = {
        label: model.predict(features, mask, dt, reported_model=label)
        for label in labels
    }

    assert len(spy.inputs) == 5
    for inputs in spy.inputs[1:]:
        for actual, expected in zip(inputs, spy.inputs[0]):
            torch.testing.assert_close(actual, expected)
    assert predictions["hybrid"][0, 0, 0].item() == pytest.approx(3.5)
    assert predictions["equal_average"][0, 0, 0].item() == pytest.approx(3.0)
    assert predictions["fixed_gate_0"][0, 0, 0].item() == pytest.approx(4.0)
    assert predictions["fixed_gate_0.5"][0, 0, 0].item() == pytest.approx(3.0)
    assert predictions["fixed_gate_1"][0, 0, 0].item() == pytest.approx(2.0)


def test_complete_mini_matrix_groups_gate_family_under_one_checkpoint(tmp_path: Path):
    if not (REPO_ROOT / "Oxford Dataset" / "handbag-1" / "imu1.csv").is_file():
        pytest.skip("real OxIOD files are not available")
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "validation_v2" / "smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    gate_models = [
        "hybrid", "equal_average", "fixed_gate_0", "fixed_gate_0.5", "fixed_gate_1"
    ]
    config["models"] = gate_models
    config["rates"] = [0.3]
    config["irregular_cases"] = [
        {
            "method": "interval_jitter",
            "requested_irregularity": 0.2,
            "value_topology": "point",
            "value_requested_fraction": 0.3,
        }
    ]
    config_path = tmp_path / "complete-mini.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_root = tmp_path / "complete-output"

    result = _cli(
        "matrix", "--config", str(config_path), "--output-root", str(output_root),
        "--device", "cpu",
    )

    assert result.returncode == 0, result.stderr.decode()
    marker = json.loads(result.stdout)
    assert marker["partial"] is False
    assert marker["selected_cells"] == 10
    assert marker["training_groups"] == 1
    assert marker["grouping_key"] == [
        "training_family", "seed", "protocol", "objective"
    ]
    metrics_paths = list(output_root.glob("*/per_record_metrics.csv"))
    assert len(metrics_paths) == 1
    metrics = pd.read_csv(metrics_paths[0])
    assert set(metrics["model"]) == set(gate_models)
    assert metrics["checkpoint_sha256"].nunique() == 1
    assert set(metrics["requested_fraction"]) == {0.3}
    assert "irregular:interval_jitter+point" in set(metrics["topology"])
    irregular_metrics = metrics.loc[
        metrics["topology"] == "irregular:interval_jitter+point"
    ]
    assert set(irregular_metrics["metric"]).issuperset(
        {"irregularity_requested", "irregularity_realized"}
    )
    requested = irregular_metrics.loc[
        irregular_metrics["metric"] == "irregularity_requested", "value"
    ]
    realized = irregular_metrics.loc[
        irregular_metrics["metric"] == "irregularity_realized", "value"
    ]
    assert set(requested) == {0.2}
    assert set(realized) != {0.2}
