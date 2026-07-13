# Python 3.12 Server Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add one safe command that creates a Python 3.12 repository-local CUDA environment and runs the full eight-shard Validation v2 formal campaign.

**Architecture:** Move the runbook's server orchestration into a sourceable Bash library parameterized by PYTHON_BIN. A thin runner owns argument parsing, .venv-server installation, immutable campaign paths, preflight, and the 1→2→4→8 campaign sequence. A permanent Python stage-metrics helper replaces the dynamically written runbook program.

**Tech Stack:** Bash 5, Python 3.12 venv, pip, PyTorch 2.3.1+cu121, existing validation-v2 CLI, pytest, Linux procfs and nvidia-smi.

---

## File structure

- Create: scripts/run_validation_v2_server.sh — command-line runner and environment setup.
- Create: scripts/lib/validation_v2_server_helpers.sh — launch, wait, queue, PID, GPU, and staged-gate functions.
- Create: scripts/collect_validation_v2_stage_metrics.py — durable stage metrics program.
- Create: tests/validation_v2/test_server_runner.py — Bash and runner contracts.
- Modify: docs/validation_v2_server_runbook.md — link to the executable runner.
- Modify: docs/validation_v2_server_runbook_zh.md — document Python 3.12 .venv-server operation.

### Task 1: Define runner contracts before implementation

**Files:**
- Create: tests/validation_v2/test_server_runner.py

- [ ] **Step 1: Write failing help and interpreter tests**

~~~python
def test_runner_help_declares_full_and_preflight_modes() -> None:
    completed = subprocess.run(
        ["bash", str(RUNNER), "--help"], text=True, capture_output=True, check=True
    )
    assert "--mode preflight|full" in completed.stdout
    assert "--commit COMMIT" in completed.stdout
    assert "--skip-dependency-install" in completed.stdout


def test_runner_rejects_non_python312_before_installation(tmp_path: Path) -> None:
    completed = run_runner(
        tmp_path, "--commit", "a" * 40, "--mode", "preflight",
        environment={"PYTHON3_BIN": str(fake_python(tmp_path, "3.11.9"))},
    )
    assert completed.returncode == 2
    assert "Python 3.12" in completed.stderr
~~~

- [ ] **Step 2: Run test to verify RED**

Run: python -m pytest tests/validation_v2/test_server_runner.py -q

Expected: FAIL because scripts/run_validation_v2_server.sh is absent.

- [ ] **Step 3: Add CUDA and full-workflow contracts**

~~~python
def test_runner_uses_local_venv_and_explicit_cuda121_torch_index() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert ".venv-server" in source
    assert "python3 -m venv" in source
    assert "https://download.pytorch.org/whl/cu121" in source
    assert "torch==2.3.1" in source
    assert 'torch.__version__ == "2.3.1+cu121"' in source
    assert "conda activate" not in source


def test_runner_full_mode_calls_formal_workflow() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for token in (
        "test_linux_rename_noreplace_survives_real_directory_race",
        "validation_v2.cli shard-plan",
        "run_formal_campaign",
        "validation_v2.cli merge-shards",
        "validation_v2.experiments.validate_artifacts",
        "validation_v2.cli summarize",
    ):
        assert token in source
~~~

- [ ] **Step 4: Commit RED tests**

~~~bash
git add tests/validation_v2/test_server_runner.py
git commit -m "test: define Python 3.12 server runner contracts"
~~~

### Task 2: Extract durable server helpers

**Files:**
- Create: scripts/lib/validation_v2_server_helpers.sh
- Create: scripts/collect_validation_v2_stage_metrics.py
- Test: tests/validation_v2/test_server_runner.py

- [ ] **Step 1: Write failing helper contracts**

~~~python
def test_helper_library_requires_venv_python_and_uses_it_for_shards() -> None:
    source = HELPERS.read_text(encoding="utf-8")
    assert "PYTHON_BIN must name an executable Python interpreter" in source
    assert '"$PYTHON_BIN" -m validation_v2.cli shard' in source
    assert "/root/miniconda3/envs/pinn_imu/bin/python" not in source


def test_stage_metrics_is_per_shard_and_fails_closed() -> None:
    source = METRICS.read_text(encoding="utf-8")
    assert "per_shard_new_groups" in source
    assert "completed_without_progress" in source
    assert "raise SystemExit(2)" in source
    assert "raise SystemExit(3)" in source
    assert "raise SystemExit(10)" in source
~~~

- [ ] **Step 2: Run test to verify RED**

Run: python -m pytest tests/validation_v2/test_server_runner.py -q

Expected: FAIL because the helper files are absent.

- [ ] **Step 3: Implement the stage metrics program**

Create scripts/collect_validation_v2_stage_metrics.py with arguments
--shards-root, --indices, --stage-start, --gpu-csv, --minimum-groups, --output,
and optional --baseline. Preserve the existing runbook semantics: group
durations are ledger completions in marker order; every active shard must have
one completion after stage start; failed markers exit 2; completed
zero-contribution shards exit 3; incomplete samples exit 4; resource or
performance gate failures write metrics then exit 10.

- [ ] **Step 4: Implement the Bash helper library**

Port audit_active, launch_shard, wait_shard, wait_all_shards, run_queue,
start_gpu_sampler, stop_gpu_sampler, wait_until_groups, and wait_stage_metrics
from the current runbook. At library load, require executable PYTHON_BIN:

~~~bash
test -n "$PYTHON_BIN" && test -x "$PYTHON_BIN" || {
  echo "PYTHON_BIN must name an executable Python interpreter: $PYTHON_BIN" >&2
  return 2
}
~~~

All CLI and JSON-inspection Python calls use "$PYTHON_BIN"; no conda path is
permitted. The metrics function invokes the permanent program at
"$REPO/scripts/collect_validation_v2_stage_metrics.py".

- [ ] **Step 5: Run helper checks and commit**

~~~bash
python -m pytest tests/validation_v2/test_server_runner.py -q
bash -n scripts/lib/validation_v2_server_helpers.sh
python -m py_compile scripts/collect_validation_v2_stage_metrics.py
git add scripts/lib/validation_v2_server_helpers.sh scripts/collect_validation_v2_stage_metrics.py tests/validation_v2/test_server_runner.py
git commit -m "feat: extract server rollout helpers"
~~~

### Task 3: Implement Python 3.12 runner

**Files:**
- Create: scripts/run_validation_v2_server.sh
- Test: tests/validation_v2/test_server_runner.py

- [ ] **Step 1: Implement strict option parsing**

Accept only --commit, --mode preflight|full, --repo, --campaign-suffix, and
--skip-dependency-install. Reject unknown options, non-40-hex commits, non-Linux
hosts, dirty worktrees, a HEAD that differs from --commit, pre-existing
campaign paths, and non-Python-3.12 interpreters.

- [ ] **Step 2: Implement venv installation and runtime verification**

Use PYTHON3_BIN with default python3 and create REPO/.venv-server. Install:

~~~bash
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1
"$PYTHON_BIN" -m pip install -r "$REPO/requirements-validation-v2.txt"
~~~

Fail unless the venv reports Python 3.12, Torch 2.3.1+cu121, CUDA 12.1,
available CUDA, and a GPU 0 name containing 4090 D.

- [ ] **Step 3: Implement preflight and plan**

Create commit-qualified AUDIT_DIR, PLAN, SHARDS_ROOT, and FINAL_ROOT. Run the
Linux rename race test and complete pytest suite. Generate the dry-run matrix
and plan, then assert schema 2, 175 groups, 4,095 cells, eight shards, clean
dirty digest, and group counts [22,22,22,22,22,22,22,21]. Preflight mode exits
successfully here.

- [ ] **Step 4: Implement full campaign orchestration**

Source the helper library and run the existing formal sequence: one-worker
baseline, 2→4→8 staged gates, safe 1- or 2-worker fallback queues, final
wait_all_shards, merge-shards, validate-artifacts, and summarize with seeds
2026–2030 and baseline linear. Propagate every non-zero exit code.

- [ ] **Step 5: Verify GREEN and commit**

~~~bash
python -m pytest tests/validation_v2/test_server_runner.py -q
bash -n scripts/run_validation_v2_server.sh
scripts/run_validation_v2_server.sh --help
git add scripts/run_validation_v2_server.sh tests/validation_v2/test_server_runner.py
git commit -m "feat: add Python 3.12 server validation runner"
~~~

### Task 4: Align documentation and run final validation

**Files:**
- Modify: docs/validation_v2_server_runbook.md
- Modify: docs/validation_v2_server_runbook_zh.md
- Modify: tests/validation_v2/test_cli_smoke.py
- Test: tests/validation_v2/test_server_runner.py

- [ ] **Step 1: Add failing documentation contract**

~~~python
def test_chinese_runbook_points_to_python312_runner_without_conda() -> None:
    chinese = (REPO_ROOT / "docs/validation_v2_server_runbook_zh.md").read_text(encoding="utf-8")
    assert "scripts/run_validation_v2_server.sh" in chinese
    assert ".venv-server" in chinese
    assert "pinn_imu" not in chinese
~~~

- [ ] **Step 2: Verify RED, update both runbooks, and verify GREEN**

Run:

~~~bash
python -m pytest tests/validation_v2/test_server_runner.py -q
python -m pytest tests/validation_v2/test_server_runner.py tests/validation_v2/test_cli_smoke.py -q
python -m compileall -q validation_v2 tests/validation_v2 scripts
git diff --check
bash -n scripts/run_validation_v2_server.sh scripts/lib/validation_v2_server_helpers.sh
~~~

Document the canonical command:

~~~bash
bash scripts/run_validation_v2_server.sh --commit "$(git rev-parse HEAD)" --mode full
~~~

Explain preflight mode, .venv-server, explicit CUDA 12.1 Torch installation,
runtime provenance, and multi-day full-mode duration. Remove mandatory
pinn_imu/conda activation from the Python-3.12 server path.

- [ ] **Step 3: Run final suite, review scope, and push**

~~~bash
conda run -n pinn_imu python -m pytest -q
git diff --check fcf81f8..HEAD
git diff --quiet fcf81f8..HEAD -- els-cas-templates/Manuscript.tex Manuscript.tex
git status --short --branch
git push origin codex/validation-v2
~~~

Expected: all local tests pass; Windows may skip only the existing Linux
renameat2 race test; no manuscript changes; branch is pushed.
