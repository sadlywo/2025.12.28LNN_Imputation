# MatPool Validation Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the immutable Validation v2 campaign runnable on the Python 3.10/RTX 4090 MatPool host through one short, observable command without changing any scientific input or result rule.

**Architecture:** Keep `scripts/run_validation_v2_server.sh` as the only formal execution engine, generalize its host policy, and add a thin MatPool `tmux` launcher for start/status/logs. Add an explicit scheduling ceiling so MatPool defaults to four simultaneous workers while all eight shards still complete and merge under the existing provenance and validation contracts.

**Tech Stack:** Bash 5, CPython 3.10--3.12 `venv`, PyTorch 2.3.1+cu121, `tmux`, `nvidia-smi`, pytest, Git.

---

## File responsibility map

- `scripts/run_validation_v2_server.sh`: authoritative host validation, environment creation, preflight, bounded shard scheduling, merge, validation, and summary.
- `scripts/run_validation_v2_matpool.sh`: MatPool-only operator UX; delegates all experiment work to the generic runner.
- `tests/validation_v2/test_server_runner.py`: executable contracts for supported runtimes, provenance, dependency safety, and scheduling ceilings.
- `tests/validation_v2/test_matpool_launcher.py`: isolated fake-`tmux` launcher behavior tests; never starts training.
- `.gitignore`: ignores repository-local launcher state and logs.
- `docs/validation_v2_server_runbook.md`: generic current path and audited historical reference.
- `docs/validation_v2_server_runbook_zh.md`: concise Chinese operator path, including MatPool commands.

### Task 1: Generalize runtime validation and provenance

**Files:**
- Modify: `tests/validation_v2/test_server_runner.py`
- Modify: `scripts/run_validation_v2_server.sh`

- [ ] **Step 1: Write failing interpreter-policy tests**

Parameterize accepted versions `3.10.12`, `3.11.9`, and `3.12.3`; parameterize
rejected versions `3.9.19` and `3.13.0`. Extend `_make_fake_python` with a
`venv_available` argument and support for the implementation and
`-m venv --help` probes. The essential assertions are:

```python
@pytest.mark.parametrize("version", ["3.10.12", "3.11.9", "3.12.3"])
def test_runner_accepts_supported_cpython_and_derives_suffix(tmp_path, version):
    fake_python, log = _make_fake_python(tmp_path, version)
    repository, commit = _make_clean_repository(tmp_path)
    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment={"PYTHON3_BIN": fake_python.as_posix(),
                     "FAKE_PYTHON_LOG": log.as_posix()},
    )
    suffix = "py3" + version.split(".")[1]
    assert "Python 3.10, 3.11, or 3.12 is required" not in completed.stderr
    assert (tmp_path / f"validation-v2-audit-{commit}-sharded-v2-{suffix}").is_dir()


@pytest.mark.parametrize("version", ["3.9.19", "3.13.0"])
def test_runner_rejects_unsupported_python_before_writes(tmp_path, version):
    fake_python, log = _make_fake_python(tmp_path, version)
    repository, commit = _make_clean_repository(tmp_path)
    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment={"PYTHON3_BIN": fake_python.as_posix(),
                     "FAKE_PYTHON_LOG": log.as_posix()},
    )
    assert completed.returncode == 2
    assert "Python 3.10, 3.11, or 3.12 is required" in completed.stderr
    assert not (repository / ".venv-server").exists()
    assert not list(tmp_path.glob("validation-v2-audit-*"))


def test_runner_checks_venv_support_before_campaign_seal(tmp_path):
    fake_python, log = _make_fake_python(tmp_path, "3.10.12", venv_available=False)
    repository, commit = _make_clean_repository(tmp_path)
    completed = _run_runner(
        "--commit", commit, "--mode", "preflight", "--repo", repository.as_posix(),
        environment={"PYTHON3_BIN": fake_python.as_posix(),
                     "FAKE_PYTHON_LOG": log.as_posix()},
    )
    assert completed.returncode == 2
    assert "python3-venv" in completed.stderr
    assert not list(tmp_path.glob("validation-v2-audit-*"))
```

- [ ] **Step 2: Run tests and verify RED**

```bash
conda run -n pinn_imu python -m pytest -q tests/validation_v2/test_server_runner.py \
  -k 'supported_cpython or unsupported_python or venv_support'
```

Expected: Python 3.10/3.11 and dynamic-suffix assertions fail.

- [ ] **Step 3: Implement the interpreter policy and dynamic suffix**

Initialize an empty suffix plus an explicit flag, set the flag while parsing
`--campaign-suffix`, then execute these checks before Git or filesystem writes:

```bash
PYTHON3_BIN="${PYTHON3_BIN:-python3}"
PYTHON3_VERSION="$("$PYTHON3_BIN" --version 2>&1)" \
  || die "cannot run PYTHON3_BIN: $PYTHON3_BIN"
[[ "$PYTHON3_VERSION" =~ ^Python[[:space:]]3\.(10|11|12)(\.|[[:space:]]) ]] \
  || die "Python 3.10, 3.11, or 3.12 is required; found: $PYTHON3_VERSION"
PYTHON_MINOR="${BASH_REMATCH[1]}"
PYTHON3_IMPLEMENTATION="$("$PYTHON3_BIN" -c \
  'import platform; print(platform.python_implementation())')" \
  || die "cannot inspect PYTHON3_BIN: $PYTHON3_BIN"
[[ "$PYTHON3_IMPLEMENTATION" == CPython ]] \
  || die "CPython is required; found: $PYTHON3_IMPLEMENTATION"
"$PYTHON3_BIN" -m venv --help >/dev/null 2>&1 \
  || die 'Python venv support is required; install the matching python3-venv package'
if (( ! CAMPAIGN_SUFFIX_EXPLICIT )); then
  CAMPAIGN_SUFFIX="sharded-v2-py3${PYTHON_MINOR}"
fi
```

Update help text and suffix-sensitive tests while preserving explicit suffixes.

- [ ] **Step 4: Add failing GPU and provenance tests**

Run the extracted `verify_runtime` heredoc with a fake `torch` package and fake
`nvidia-smi`. Test both supported names and both rejection dimensions:

```python
@pytest.mark.parametrize("gpu_name", [
    "NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090 D",
])
def test_runtime_accepts_supported_4090_names(tmp_path, gpu_name):
    completed, environment = _run_runtime_probe(tmp_path, gpu_name=gpu_name)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(environment.read_text(encoding="utf-8"))
    assert payload["gpu_0"] == gpu_name
    assert payload["gpu_memory_bytes"] >= 23 * 1024**3
    assert payload["driver_version"] == "565.77"
    assert payload["cpu_affinity_count"] > 0
    assert payload["host_memory_bytes"] > 0


@pytest.mark.parametrize("gpu_name,memory_bytes", [
    ("NVIDIA A100-SXM4-40GB", 40 * 1024**3),
    ("NVIDIA GeForce RTX 4090", 22 * 1024**3),
])
def test_runtime_rejects_wrong_gpu_or_memory(tmp_path, gpu_name, memory_bytes):
    completed, environment = _run_runtime_probe(
        tmp_path, gpu_name=gpu_name, memory_bytes=memory_bytes)
    assert completed.returncode != 0
    assert not environment.exists()
```

The fake exposes Torch version, CUDA version/availability, device name,
properties, and compute capability; fake `nvidia-smi` prints `565.77`.

- [ ] **Step 5: Run GPU tests and verify RED**

```bash
conda run -n pinn_imu python -m pytest -q tests/validation_v2/test_server_runner.py \
  -k 'runtime_accepts or wrong_gpu or provenance'
```

Expected: ordinary 4090 and expanded provenance fail.

- [ ] **Step 6: Implement generalized runtime verification**

Keep explicit `require` calls under `python -O`, accept supported CPython
minors, exact Torch/CUDA, `4090` in the device name, and at least 23 GiB:

```python
properties = torch.cuda.get_device_properties(0)
memory_bytes = int(properties.total_memory)
require("4090" in name.upper(), name)
require(memory_bytes >= 23 * 1024**3, str(memory_bytes))
driver_version = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
    text=True,
).splitlines()[0].strip()
cpu_affinity_count = (
    len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
    else (os.cpu_count() or 1)
)
host_memory_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
```

Publish these plus compute capability in the exclusive `environment.json`.

- [ ] **Step 7: Verify and commit Task 1**

```bash
conda run -n pinn_imu python -m pytest -q tests/validation_v2/test_server_runner.py
git add scripts/run_validation_v2_server.sh tests/validation_v2/test_server_runner.py
git commit -m "feat: support Python 3.10 and RTX 4090 validation hosts"
```

### Task 2: Add a bounded worker ceiling without changing the shard plan

**Files:**
- Modify: `tests/validation_v2/test_server_runner.py`
- Modify: `scripts/run_validation_v2_server.sh`

- [ ] **Step 1: Write failing CLI and cutoff tests**

```python
def test_runner_help_declares_worker_ceiling():
    completed = _run_runner("--help")
    assert completed.returncode == 0
    assert "--max-workers 1|2|4|8" in completed.stdout


@pytest.mark.parametrize("value", ["0", "3", "16", "four"])
def test_runner_rejects_invalid_worker_ceiling(value):
    completed = _run_runner("--max-workers", value)
    assert completed.returncode == 2
    assert "--max-workers must be 1, 2, 4, or 8" in completed.stderr


def test_runner_worker_cutoffs_queue_every_remaining_shard():
    source = RUNNER.read_text(encoding="utf-8")
    assert 'if (( MAX_WORKERS == 1 ))' in source
    assert 'run_queue 1 001 002 003 004 005 006 007' in source
    assert 'if (( MAX_WORKERS == 2 ))' in source
    assert 'run_queue 2 002 003 004 005 006 007' in source
    assert 'if (( MAX_WORKERS == 4 ))' in source
    assert 'run_queue 4 004 005 006 007' in source
    assert 'wait_all_shards 000 001 002 003 004 005 006 007' in source
```

- [ ] **Step 2: Run tests and verify RED**

```bash
conda run -n pinn_imu python -m pytest -q tests/validation_v2/test_server_runner.py \
  -k 'worker_ceiling or worker_cutoffs'
```

Expected: option and cutoff assertions fail.

- [ ] **Step 3: Implement parsing and scheduling cutoffs**

Add `MAX_WORKERS=8`, parse `--max-workers`, and validate before repository
writes:

```bash
[[ "$MAX_WORKERS" == 1 || "$MAX_WORKERS" == 2 \
   || "$MAX_WORKERS" == 4 || "$MAX_WORKERS" == 8 ]] \
  || die '--max-workers must be 1, 2, 4, or 8'
```

After the baseline gate, use:

```bash
if (( MAX_WORKERS == 1 )); then
  wait_shard 000
  run_queue 1 001 002 003 004 005 006 007
else
  date -u +%Y-%m-%dT%H:%M:%S+00:00 | tee "$AUDIT_DIR/stage-2worker-start.txt"
  stage2_start="$(cat "$AUDIT_DIR/stage-2worker-start.txt")"
  start_managed_sampler stage-2worker
  launch_formal_shard 001
  # The existing wait_stage_metrics call and exact stage2_rc case follow here.
fi
```

Wrap the complete existing two-/four-/eight-worker block in this `else`
without extracting new orchestration functions. Immediately after the existing
successful two-worker status case, cap at two with:

```bash
if (( MAX_WORKERS == 2 )); then
  wait_shard 000
  wait_shard 001
  run_queue 2 002 003 004 005 006 007
else
  date -u +%Y-%m-%dT%H:%M:%S+00:00 | tee "$AUDIT_DIR/stage-4worker-start.txt"
  stage4_start="$(cat "$AUDIT_DIR/stage-4worker-start.txt")"
  start_managed_sampler stage-4worker
  launch_formal_shard 002
  launch_formal_shard 003
  # The existing wait_stage_metrics call and exact stage4_rc case follow here.
fi
```

Keep the existing four-worker block directly in this `else`. Immediately after
the existing successful four-worker status case, cap at four with:

```bash
if (( MAX_WORKERS == 4 )); then
  wait_all_shards 000 001 002 003
  run_queue 4 004 005 006 007
else
  date -u +%Y-%m-%dT%H:%M:%S+00:00 | tee "$AUDIT_DIR/stage-8worker-start.txt"
  stage8_start="$(cat "$AUDIT_DIR/stage-8worker-start.txt")"
  start_managed_sampler stage-8worker
  launch_formal_shard 004
  launch_formal_shard 005
  launch_formal_shard 006
  launch_formal_shard 007
  # The existing eight-worker wait_stage_metrics call and status case follow here.
fi
```

Do not introduce a second merge path or new stage helper. Do not change failed
two-/four-worker fallbacks or their return statuses. Every successful path
returns to the single final eight-shard wait, merge, validator, and summary
block.

- [ ] **Step 4: Verify and commit Task 2**

```bash
bash -n scripts/run_validation_v2_server.sh
conda run -n pinn_imu python -m pytest -q tests/validation_v2/test_server_runner.py
git add scripts/run_validation_v2_server.sh tests/validation_v2/test_server_runner.py
git commit -m "feat: bound formal validation worker concurrency"
```

### Task 3: Add the MatPool start/status/logs launcher

**Files:**
- Create: `scripts/run_validation_v2_matpool.sh`
- Create: `tests/validation_v2/test_matpool_launcher.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing launcher tests**

Create a fixture that initializes a temporary clean Git repository, copies the
launcher under `scripts/`, creates a fake generic runner, and prepends a fake
`tmux` executable that records arguments. Add these contracts:

```python
def test_launcher_help_exposes_only_safe_commands(matpool_repo):
    completed = _run_launcher(matpool_repo, "--help")
    assert completed.returncode == 0
    assert all(word in completed.stdout for word in ("start", "status", "logs"))
    assert "stop" not in completed.stdout


def test_start_forwards_exact_commit_suffix_and_four_workers(matpool_repo):
    completed = _run_launcher(matpool_repo, "start")
    assert completed.returncode == 0, completed.stderr
    state = _state(matpool_repo)
    assert state["commit"] == _head(matpool_repo)
    assert state["campaign_suffix"].startswith("matpool-")
    command = Path(state["command_file"]).read_text(encoding="utf-8")
    assert f'--commit {state["commit"]}' in command
    assert "--mode full" in command
    assert "--max-workers 4" in command


def test_start_refuses_dirty_worktree_without_state_writes(matpool_repo):
    (matpool_repo / "foreign-file").write_text("dirty\n", encoding="utf-8")
    completed = _run_launcher(matpool_repo, "start")
    assert completed.returncode == 2
    assert "clean" in completed.stderr.lower()
    assert not (matpool_repo / ".validation-v2-matpool").exists()
```

- [ ] **Step 2: Run tests and verify RED**

```bash
conda run -n pinn_imu python -m pytest -q tests/validation_v2/test_matpool_launcher.py
```

Expected: tests fail because the launcher does not exist.

- [ ] **Step 3: Implement safe `start`**

The launcher uses `set -Eeuo pipefail`, discovers `REPO` from its own path,
defaults to four, and accepts only `start`, `status`, `logs`, `--max-workers`,
and `--skip-dependency-install`. Before creating state:

```bash
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || die 'cannot resolve exact Git commit'
[[ -z "$(git -C "$REPO" status --porcelain)" ]] \
  || die 'Git worktree must be clean before MatPool start'
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CAMPAIGN_SUFFIX="matpool-$TIMESTAMP"
SESSION="validation-v2-${COMMIT:0:12}-$TIMESTAMP"
```

Reject linked state directories. If validated `current.json` names a live
session, reject duplicate start. Write a campaign command file using
`printf '%q'` for every path and argument. It must invoke the generic runner,
append both streams to the log, and atomically publish its exit status. Publish
`current.json` with a same-directory temporary file opened in exclusive mode
and `os.replace`. Start only with:

```bash
tmux new-session -d -s "$SESSION" bash "$COMMAND_FILE"
```

Store `commit`, `campaign_suffix`, `session`, `command_file`, `log_path`,
`audit_dir`, `shards_root`, `final_root`, and `max_workers` in JSON. Never
`source` state content.

- [ ] **Step 4: Add failing option and observation tests**

```python
def test_start_forwards_override_and_dependency_reuse(matpool_repo):
    completed = _run_launcher(
        matpool_repo, "start", "--max-workers", "2", "--skip-dependency-install")
    assert completed.returncode == 0
    command = Path(_state(matpool_repo)["command_file"]).read_text("utf-8")
    assert "--max-workers 2" in command
    assert "--skip-dependency-install" in command


def test_status_is_read_only_and_reports_paths(matpool_repo):
    assert _run_launcher(matpool_repo, "start").returncode == 0
    state_path = matpool_repo / ".validation-v2-matpool/current.json"
    before = state_path.read_bytes()
    completed = _run_launcher(matpool_repo, "status")
    assert completed.returncode == 0
    assert "campaign_suffix=" in completed.stdout
    assert "log_path=" in completed.stdout
    assert state_path.read_bytes() == before


def test_logs_fails_clearly_without_state(matpool_repo):
    completed = _run_launcher(matpool_repo, "logs")
    assert completed.returncode == 2
    assert "No MatPool campaign state" in completed.stderr


def test_duplicate_live_session_is_refused(matpool_repo):
    assert _run_launcher(matpool_repo, "start").returncode == 0
    completed = _run_launcher(matpool_repo, "start")
    assert completed.returncode == 2
    assert "already active" in completed.stderr
```

- [ ] **Step 5: Implement read-only `status` and `logs`**

Parse JSON with system Python. Require string values for `session`, `commit`,
`campaign_suffix`, `log_path`, `audit_dir`, `shards_root`, and `final_root`,
and reject embedded newlines. `status` calls only `tmux has-session`, prints
validated values and at most 20 final log lines. `logs` calls:

```bash
tail -F -- "$LOG_PATH"
```

Neither command writes state or sends signals.

- [ ] **Step 6: Ignore state, verify, and commit Task 3**

Add `.validation-v2-matpool/` to `.gitignore`, then run:

```bash
bash -n scripts/run_validation_v2_matpool.sh
conda run -n pinn_imu python -m pytest -q tests/validation_v2/test_matpool_launcher.py
git add .gitignore scripts/run_validation_v2_matpool.sh \
  tests/validation_v2/test_matpool_launcher.py
git commit -m "feat: add MatPool validation launcher"
```

### Task 4: Update operator contracts and secret checks

**Files:**
- Modify: `docs/validation_v2_server_runbook.md`
- Modify: `docs/validation_v2_server_runbook_zh.md`
- Modify: `tests/validation_v2/test_cli_smoke.py`
- Modify: `tests/validation_v2/test_matpool_launcher.py`

- [ ] **Step 1: Write failing documentation and secret tests**

```python
def test_current_runbooks_support_matpool_python310_and_4090():
    english = ENGLISH_RUNBOOK.read_text(encoding="utf-8")
    chinese = CHINESE_RUNBOOK.read_text(encoding="utf-8")
    for text in (english, chinese):
        assert "Python 3.10" in text
        assert "RTX 4090" in text
        assert "run_validation_v2_matpool.sh start" in text
        assert "run_validation_v2_matpool.sh status" in text
        assert "run_validation_v2_matpool.sh logs" in text
    assert "/2025.12.28LNN_Imputation" in chinese


def test_tracked_runner_material_contains_no_ssh_credentials():
    public_material = (
        RUNNER,
        REPO_ROOT / "scripts" / "run_validation_v2_matpool.sh",
        ENGLISH_RUNBOOK,
        CHINESE_RUNBOOK,
    )
    for path in public_material:
        content = path.read_bytes()
        assert credential_leak_categories(content) == ()
```

- [ ] **Step 2: Run tests and verify RED**

```bash
conda run -n pinn_imu python -m pytest -q \
  tests/validation_v2/test_cli_smoke.py tests/validation_v2/test_matpool_launcher.py \
  -k 'runbooks or credentials'
```

Expected: current-path tests fail because the runbooks are still 3.12/4090 D
specific and omit the MatPool launcher.

- [ ] **Step 3: Update English and Chinese current paths**

Add this MatPool block without SSH access data:

```bash
cd /2025.12.28LNN_Imputation
git status --short --branch
bash scripts/run_validation_v2_matpool.sh start
bash scripts/run_validation_v2_matpool.sh status
bash scripts/run_validation_v2_matpool.sh logs
```

State that system CPython 3.10--3.12 creates `.venv-server`, Torch remains
2.3.1+cu121, MatPool defaults to four simultaneous workers, all eight shards
still execute, and the launcher includes complete preflight. Document
`--max-workers 8` as an explicit opt-in after four-worker evidence. Keep
historical Conda/manual material clearly separated.

- [ ] **Step 4: Update obsolete source assertions**

Replace Python-3.12-only and `"4090 D"`-only current-path assertions in
`test_cli_smoke.py` with the Python range, `4090`, 23 GiB, provenance, and
worker-ceiling contracts. Preserve existing safety tests for commit-qualified
paths, PID ownership, atomic publication, full-shard wait, and merge validation.

- [ ] **Step 5: Verify and commit Task 4**

```bash
conda run -n pinn_imu python -m pytest -q \
  tests/validation_v2/test_cli_smoke.py tests/validation_v2/test_matpool_launcher.py
git diff --check
git add docs/validation_v2_server_runbook.md \
  docs/validation_v2_server_runbook_zh.md \
  tests/validation_v2/test_cli_smoke.py \
  tests/validation_v2/test_matpool_launcher.py
git commit -m "docs: add MatPool validation operations"
```

### Task 5: Verify, review, push, and run real-host preflight

**Files:**
- Verify every changed file
- Verify that no manuscript file changed

- [ ] **Step 1: Run fresh complete local verification**

```bash
bash -n scripts/run_validation_v2_server.sh scripts/run_validation_v2_matpool.sh \
  scripts/lib/validation_v2_server_helpers.sh
conda run -n pinn_imu python -m compileall -q validation_v2 tests/validation_v2 scripts
conda run -n pinn_imu python -m pytest -q
git diff --check origin/main..HEAD
git diff --quiet origin/main..HEAD -- els-cas-templates/Manuscript.tex Manuscript.tex
git status --short
```

Expected: shell, compile, diff, and manuscript commands exit 0; pytest has zero
failures; worktree is clean.

- [ ] **Step 2: Run independent final reviews**

Give specification and code-quality reviewers the confirmed design, this plan,
and `origin/main..HEAD`. Resolve every Critical or Important issue, then rerun
Step 1 after any fix.

- [ ] **Step 3: Push the reviewed branch**

```bash
git push -u origin codex/matpool-runner
```

Expected: remote branch equals the verified local HEAD.

- [ ] **Step 4: Update the authorized clean MatPool checkout**

```bash
cd /2025.12.28LNN_Imputation
git fetch origin codex/matpool-runner
git checkout --detach "$(git rev-parse origin/codex/matpool-runner)"
test -z "$(git status --porcelain)"
```

Expected: detached HEAD equals the pushed reviewed commit.

- [ ] **Step 5: Run real MatPool preflight only**

```bash
cd /2025.12.28LNN_Imputation
bash scripts/run_validation_v2_server.sh \
  --commit "$(git rev-parse HEAD)" \
  --mode preflight \
  --max-workers 4 \
  --campaign-suffix "matpool-preflight-$(date -u +%Y%m%dT%H%M%SZ)"
```

Expected: Python 3.10/RTX 4090 verification, Linux atomic test, full pytest,
175-group/4,095-cell/eight-shard plan, and no training shard launch.

- [ ] **Step 6: Report the formal-start handoff**

After verifying preflight artifacts, provide:

```bash
cd /2025.12.28LNN_Imputation
bash scripts/run_validation_v2_matpool.sh start --skip-dependency-install
bash scripts/run_validation_v2_matpool.sh status
bash scripts/run_validation_v2_matpool.sh logs
```

Do not execute `start` during rollout until the user explicitly authorizes the
paid multi-day campaign after reviewing real preflight evidence.
