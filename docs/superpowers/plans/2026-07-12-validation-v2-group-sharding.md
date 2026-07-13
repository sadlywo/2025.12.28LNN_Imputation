# Validation v2 Training-Group Sharding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic training-group shards, isolated shard execution, and a strict atomic merge so the 175-group formal validation can use multiple RTX 4090 D workers without changing scientific inputs.

**Architecture:** Extract the existing grouping rules into a dependency-free `groups.py` module shared by serial and sharded execution. Implement plan/worker/merge orchestration in `sharding.py`; every worker writes one exclusive root, and merge validates complete disjoint coverage before building a fresh formal root and invoking the existing artifact validator. Keep the serial `matrix` command behavior compatible.

**Tech Stack:** Python 3.9, argparse, dataclasses/typed mappings, pathlib, hashlib, canonical JSON, PyYAML, pytest, existing validation-v2 provenance and artifact validation modules.

---

## File structure

- Create `validation_v2/experiments/groups.py`: stable training-group enumeration and conversion to the exact existing `run_smoke` group config.
- Create `validation_v2/experiments/sharding.py`: shard plan schema, plan validation, shard execution markers, merge preflight, artifact promotion, and final validation.
- Modify `validation_v2/experiments/runner.py`: use the shared group enumerator for serial execution.
- Modify `validation_v2/cli.py`: add `shard-plan`, `shard`, and `merge-shards` commands.
- Modify `validation_v2/experiments/__init__.py`: export the new public orchestration functions.
- Create `tests/validation_v2/test_groups.py`: grouping invariants and serial compatibility.
- Create `tests/validation_v2/test_sharding.py`: plan, worker, merge, tamper, and idempotency tests.
- Modify `tests/validation_v2/test_cli_smoke.py`: subprocess-level CLI integration tests.
- Modify `docs/validation_v2_server_runbook.md`: exact local-to-server and 2/4/8-worker commands.

### Task 1: Shared training-group enumeration

**Files:**
- Create: `validation_v2/experiments/groups.py`
- Modify: `validation_v2/experiments/runner.py`
- Test: `tests/validation_v2/test_groups.py`

- [ ] **Step 1: Write failing grouping tests**

Add tests that load `server_full.yaml` and assert 175 unique groups, 4,095 unique combination IDs, five training families, 35 groups per family, and stable group IDs across repeated calls. Add a mini config test proving gate variants share one `hybrid_shared` group. Add a compatibility test proving `group_execution_config()` produces the same private execution fields currently constructed by `run_matrix()`.

```python
def test_formal_config_has_expected_training_groups():
    config = yaml.safe_load(SERVER_CONFIG.read_text(encoding="utf-8"))
    groups = enumerate_training_groups(config)
    assert len(groups) == 175
    assert sum(len(group.combination_ids) for group in groups) == 4095
    assert Counter(group.training_family for group in groups) == {
        "bilnn": 35,
        "bilstm": 35,
        "hybrid_shared": 35,
        "linear": 35,
        "locf": 35,
    }
    assert len({group.group_id for group in groups}) == 175
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```powershell
pytest tests/validation_v2/test_groups.py -q
```

Expected: collection/import failure because `validation_v2.experiments.groups` does not exist.

- [ ] **Step 3: Implement immutable group plans**

Implement a frozen dataclass and two functions:

```python
@dataclass(frozen=True)
class TrainingGroup:
    group_index: int
    group_id: str
    training_family: str
    training_model: str
    reported_models: tuple[str, ...]
    seed: int
    protocol: str
    objective: str
    conditions: tuple[Mapping[str, Any], ...]
    combination_ids: tuple[str, ...]

TrainingGroupEnumerator = Callable[
    [Mapping[str, Any]], tuple[TrainingGroup, ...]
]
GroupExecutionConfigBuilder = Callable[
    [Mapping[str, Any], TrainingGroup], dict[str, Any]
]
```

Use `enumerate_matrix()`, the existing gate-family set, tuple sorting by `(training_family, seed, protocol, objective)`, and SHA-256 of canonical JSON containing the group key and ordered combination IDs. Validate that group IDs and combination IDs are unique.

- [ ] **Step 4: Replace runner-local grouping**

Change `run_matrix()` to call `enumerate_training_groups(config)` and `group_execution_config(config, group)`. Preserve marker fields, execution order, selected-cell behavior, and `--max-combinations` semantics. For a bounded selection, enumerate groups from the selected cells through an optional `combinations` argument rather than accidentally regrouping the full matrix.

- [ ] **Step 5: Run focused and compatibility tests**

Run:

```powershell
pytest tests/validation_v2/test_groups.py tests/validation_v2/test_cli_smoke.py -q
```

Expected: all tests pass, including the existing complete mini-matrix gate-family test.

- [ ] **Step 6: Commit**

```powershell
git add validation_v2/experiments/groups.py validation_v2/experiments/runner.py tests/validation_v2/test_groups.py
git commit -m "refactor: share validation training groups"
```

### Task 2: Deterministic shard plans

**Files:**
- Create: `validation_v2/experiments/sharding.py`
- Test: `tests/validation_v2/test_sharding.py`

- [ ] **Step 1: Write failing plan tests**

Test deterministic round-robin assignment, complete/disjoint coverage, stable group IDs across shard counts, invalid counts/indices, plan digest tampering, source-config mismatch, and JSON round trips.

```python
def test_shard_plan_is_disjoint_and_exhaustive(formal_config):
    plan = build_shard_plan(formal_config, shard_count=8, git_commit="a" * 40)
    assigned = [group_id for shard in plan["shards"] for group_id in shard["group_ids"]]
    assert len(assigned) == len(set(assigned)) == 175
    assert plan["total_cells"] == 4095
    for shard in plan["shards"]:
        assert all(index % 8 == shard["shard_index"] for index in shard["group_indices"])
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```powershell
pytest tests/validation_v2/test_sharding.py -q
```

Expected: import failure for the new sharding API.

- [ ] **Step 3: Implement plan construction and strict loading**

Implement:

```python
SHARD_SCHEMA_VERSION = 1

ShardPlanBuilder = Callable[
    [Mapping[str, Any], int, str, str], dict[str, Any]
]
ShardPlanWriter = Callable[[Path, Mapping[str, Any]], None]
ShardPlanLoader = Callable[
    [Path, Mapping[str, Any], str, str], dict[str, Any]
]
```

Calculate `source_config_sha256` from canonical JSON after excluding only `output_root`. Calculate `plan_sha256` after excluding `created_at` and `plan_sha256`. Write through a same-directory temporary file and `Path.replace()`, while rejecting an existing non-identical plan.

- [ ] **Step 4: Run plan tests**

Run:

```powershell
pytest tests/validation_v2/test_sharding.py -k plan -q
```

Expected: all plan tests pass.

- [ ] **Step 5: Commit**

```powershell
git add validation_v2/experiments/sharding.py tests/validation_v2/test_sharding.py
git commit -m "feat: add deterministic validation shard plans"
```

### Task 3: Isolated shard execution

**Files:**
- Modify: `validation_v2/experiments/sharding.py`
- Modify: `validation_v2/experiments/__init__.py`
- Test: `tests/validation_v2/test_sharding.py`

- [ ] **Step 1: Write failing worker tests**

Monkeypatch `run_smoke()` with a deterministic fake that creates one complete run directory per group. Assert workers execute only assigned groups, write no full-matrix marker, record exact run IDs, atomically complete the shard marker, and resume an already completed shard without invoking the fake again. Assert a partial run directory and a failed marker are rejected.

```python
def test_execute_shard_runs_only_assigned_groups(tmp_path, mini_config, monkeypatch):
    plan = build_shard_plan(mini_config, shard_count=2, git_commit="a" * 40)
    calls = []
    monkeypatch.setattr(sharding, "_run_group", lambda *args, **kwargs: calls.append(args[1]))
    report = execute_shard(
        mini_config, plan=plan, shard_index=1, repository_root=REPO_ROOT,
        output_root=tmp_path / "shard-001", requested_device="cuda",
    )
    assert report["status"] == "completed"
    assert {group.group_id for group in calls} == set(plan["shards"][1]["group_ids"])
    assert not (tmp_path / "shard-001" / "matrix_execution.json").exists()
```

- [ ] **Step 2: Run worker tests and confirm RED**

Run:

```powershell
pytest tests/validation_v2/test_sharding.py -k shard_execution -q
```

Expected: failure because `execute_shard()` is absent.

- [ ] **Step 3: Implement worker markers and execution**

Implement `execute_shard()` and a small `_run_group()` wrapper. Marker fields must include schema version, plan hash, config hash, commit, device, shard index/count, group IDs, combination IDs, status, and run IDs. Before reuse, verify the existing marker matches every immutable field. Execute each group through the unchanged `run_smoke(group_execution_config(config, group))` path.

```python
ShardExecutor = Callable[
    [Mapping[str, Any], Mapping[str, Any], int, Path, Path, str],
    Mapping[str, Any],
]
```

Use atomic marker writes. Never write `matrix_execution.json`, `smoke_summary.json`, or a full summary in a shard root.

- [ ] **Step 4: Run worker tests**

Run:

```powershell
pytest tests/validation_v2/test_sharding.py -k "shard_execution or shard_resume" -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```powershell
git add validation_v2/experiments/sharding.py validation_v2/experiments/__init__.py tests/validation_v2/test_sharding.py
git commit -m "feat: execute isolated validation shards"
```

### Task 4: Strict atomic shard merge

**Files:**
- Modify: `validation_v2/experiments/sharding.py`
- Test: `tests/validation_v2/test_sharding.py`

- [ ] **Step 1: Write failing merge tests**

Create complete synthetic shard roots using existing artifact fixtures. Cover successful exhaustive merge, duplicate/missing group IDs, duplicate run IDs, foreign plan/config/commit/device, failed or started marker, tampered split/scaler assets, partial run artifacts, existing destination, and a forced promotion failure.

```python
def test_merge_rejects_missing_group_without_creating_destination(shard_fixture):
    shard_fixture.remove_last_group()
    with pytest.raises(ValueError, match="missing group"):
        merge_shards(
            config_path=shard_fixture.config_path,
            plan_path=shard_fixture.plan_path,
            shards_root=shard_fixture.shards_root,
            output_root=shard_fixture.final_root,
        )
    assert not shard_fixture.final_root.exists()
```

- [ ] **Step 2: Run merge tests and confirm RED**

Run:

```powershell
pytest tests/validation_v2/test_sharding.py -k merge -q
```

Expected: failure because `merge_shards()` is absent.

- [ ] **Step 3: Implement read-only preflight**

Implement a preflight function that returns an immutable promotion manifest only after verifying all shard markers, exact plan coverage, unique run IDs, complete run-file sets, matching content-addressed assets, and no foreign run directories.

```python
FORMAL_RUN_FILES = (
    "run.json", "history.json", "best.pt", "checkpoint.json",
    "test_evaluation.json", "per_record_metrics.csv",
)

ShardPreflight = Callable[
    [Mapping[str, Any], Mapping[str, Any], Path], Mapping[str, Any]
]
```

- [ ] **Step 4: Implement temporary-root promotion and validation**

Implement `merge_shards()` so the final destination must not exist. Create a unique sibling temporary directory, copy files with `shutil.copy2`, verify SHA-256 after copy, write the complete formal marker atomically, call `validate_artifacts(temp_root, config=config_path)`, and rename the temporary directory to the final destination only after validation succeeds. On failure, preserve shard roots and rename the temporary root with a `.failed-merge-*` suffix.

```python
ShardMerger = Callable[[Path, Path, Path, Path], Mapping[str, Any]]
```

- [ ] **Step 5: Run merge and validator tests**

Run:

```powershell
pytest tests/validation_v2/test_sharding.py tests/validation_v2/test_server_handoff.py -q
```

Expected: all tests pass; tampered and partial cases fail before a formal root appears.

- [ ] **Step 6: Commit**

```powershell
git add validation_v2/experiments/sharding.py tests/validation_v2/test_sharding.py
git commit -m "feat: strictly merge validation shards"
```

### Task 5: CLI integration

**Files:**
- Modify: `validation_v2/cli.py`
- Modify: `tests/validation_v2/test_cli_smoke.py`

- [ ] **Step 1: Write failing CLI tests**

Add subprocess tests for `shard-plan`, invalid shard indices, a two-shard mini execution, and `merge-shards`. Assert JSON stdout, nonzero exit codes with concise stderr, and that merged output passes `validate_artifacts`.

- [ ] **Step 2: Run CLI tests and confirm RED**

Run:

```powershell
pytest tests/validation_v2/test_cli_smoke.py -k shard -q
```

Expected: argparse rejects unknown shard commands.

- [ ] **Step 3: Add CLI parsers and handlers**

Add:

```text
shard-plan --config --shard-count --output --device
shard --config --plan --shard-index --output-root --device
merge-shards --config --plan --shards-root --output-root
```

Resolve the current git commit through the shared provenance helper or a narrow public wrapper. Print one canonical JSON result to stdout. Catch `OSError`, `TypeError`, `ValueError`, and YAML errors, print `validation-v2: <message>` to stderr, and return exit code 2.

- [ ] **Step 4: Run CLI integration tests**

Run:

```powershell
pytest tests/validation_v2/test_cli_smoke.py -q
```

Expected: all CLI tests pass.

- [ ] **Step 5: Commit**

```powershell
git add validation_v2/cli.py tests/validation_v2/test_cli_smoke.py
git commit -m "feat: expose validation shard commands"
```

### Task 6: Server runbook and operator commands

**Files:**
- Modify: `docs/validation_v2_server_runbook.md`
- Test: `tests/validation_v2/test_cli_smoke.py`

- [ ] **Step 1: Add documentation contract tests**

Assert the runbook contains the three new commands, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, independent shard roots, 2→4→8 pilot thresholds, process/status commands, merge into a fresh root, strict validation, and no server password.

- [ ] **Step 2: Run documentation tests and confirm RED**

Run:

```powershell
pytest tests/validation_v2/test_cli_smoke.py -k runbook -q
```

Expected: missing-command assertions fail.

- [ ] **Step 3: Document exact server workflow**

Add copy-paste commands that:

1. fetch and checkout the validated commit;
2. activate `pinn_imu` and optionally source `/etc/network_turbo` only for network operations;
3. run all tests and a dry plan;
4. generate an eight-shard plan;
5. start two pilot shard workers with separate logs and roots;
6. inspect GPU/process/marker state;
7. scale to four/eight workers by starting remaining indices;
8. wait for all shard markers to complete;
9. merge into a fresh final root;
10. run strict validation and formal summarization.

Include a loop based on explicit shard indices, not shared-root matrix processes. State that the old serial result root must not be mixed with new-commit shards.

- [ ] **Step 4: Run documentation and CLI tests**

Run:

```powershell
pytest tests/validation_v2/test_cli_smoke.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add docs/validation_v2_server_runbook.md tests/validation_v2/test_cli_smoke.py
git commit -m "docs: add sharded server validation workflow"
```

### Task 7: Full verification and independent review

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run static and focused verification**

```powershell
python -m compileall -q validation_v2 tests/validation_v2
pytest tests/validation_v2/test_groups.py tests/validation_v2/test_sharding.py -q
git diff --check
```

Expected: exit code 0 for every command.

- [ ] **Step 2: Run the full suite**

```powershell
pytest -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 3: Verify formal plan without training**

```powershell
python -m validation_v2.cli shard-plan `
  --config configs/validation_v2/server_full.yaml `
  --shard-count 8 `
  --output .artifacts/validation-v2-plan.json `
  --device cuda
```

Expected: JSON reports 175 groups, 4,095 cells, eight disjoint shards, and one plan SHA-256.

- [ ] **Step 4: Inspect repository state**

```powershell
git status --short --branch
git log --oneline --decorate -8
```

Expected: only intentional commits and no uncommitted changes.

- [ ] **Step 5: Request independent code and spec review**

The reviewer must check serial compatibility, no shared shard writers, exact coverage, atomic merge semantics, validator invocation, plan/config/commit/device binding, Python 3.9 compatibility, and runbook command correctness. Address every high- or medium-severity finding and rerun Steps 1–4.
