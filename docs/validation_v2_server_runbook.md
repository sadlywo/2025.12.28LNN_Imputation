# Validation v2 sharded server runbook

This is the operator contract for the paper-grade `server_full` campaign. Run
it on Linux from the validated revision. The formal campaign consists of one
immutable eight-shard plan, eight isolated shard roots, and one newly published
final root. A smoke run, a partial run, or artifacts from another commit are
diagnostic only.

## Current supported execution path

The supported runtime is Linux with generic CPython 3.10–3.12 and an RTX 4090
series GPU (RTX 4090 or RTX 4090D). Every supported Python minor uses the same
locked CUDA build, `torch==2.3.1+cu121`, and the same validation dependency
lock. Python 3.12 and RTX 4090D are supported configurations, not exclusive
requirements.

The generic runner is the executable scientific contract. Given an exact
40-character commit, it checks the clean worktree, performs the complete
preflight, executes a full immutable eight-shard campaign, and controls the
maximum concurrent worker count. It records provenance, runs the Linux atomic
race and complete pytest suite, creates and verifies the 175-group/4,095-cell
plan, trains every assigned shard, then merges, validates, and summarizes all
five seeds. The MatPool launcher below is the current operator wrapper around
that generic runner; it does not replace or weaken any of these checks.

### MatPool: shortest clean-checkout operation

From an already cloned repository, bind the reviewed 40-character commit before
any `start`. The explicit equality and clean-worktree gates below are mandatory;
the launcher repeats them and fails closed if either condition changes.

```bash
cd /2025.12.28LNN_Imputation
VALIDATED_COMMIT="<40-HEX-VALIDATED-COMMIT>"
git checkout --detach "$VALIDATED_COMMIT"
test "$(git rev-parse HEAD)" = "$VALIDATED_COMMIT"
test -z "$(git status --porcelain)"
bash scripts/run_validation_v2_matpool.sh start
bash scripts/run_validation_v2_matpool.sh status
bash scripts/run_validation_v2_matpool.sh logs
```

`start` returns after creating a detached tmux session. Inside that session the
generic runner executes its complete preflight before any training begins, so
a successful background launch is not evidence that preflight or training has
completed. A formal run can take multiple days. Use `status` to report the tmux
state and campaign paths, and `logs` to follow the current campaign log. The
launcher state is private under `REPO/.validation-v2-matpool/`: `current.json`
records the current session and artifact paths, while `run-*.log` and
`run-*.exit` preserve combined output and the eventual exit status. Audit data
is stored in the commit-qualified `validation-v2-audit-*` sibling directory;
shard and final roots are the paths printed by `status`.

There is deliberately no `stop` command. On failure, preserve the state file,
log, audit directory, shard roots, and exit-status evidence. Diagnose the first
preflight or runner error from `logs` and `status`; do not delete a campaign
seal, kill an unverified PID, or overwrite an existing root to force a retry.

The MatPool default max-workers value is 4, but the complete plan still runs
all 8 shards; the limit controls concurrency, not campaign coverage. Only after
reviewing the four-worker GPU-memory, throughput, PID, marker, and audit
evidence may an operator opt in to eight concurrent workers. After repeating
the exact-commit and clean-worktree gate in the main block, use
`bash scripts/run_validation_v2_matpool.sh start --max-workers 8`.

### Generic runner and dependency reuse

For direct diagnostics or a non-MatPool Linux host, invoke the generic runner
with the exact current commit. Choose exactly one of these paths: (A) run full
directly, or (B) run preflight and then full. In both paths `--max-workers` caps
concurrency while retaining the complete eight-shard plan.

Path A — direct full, without a separate preflight invocation:

```bash
DIRECT_SUFFIX="formal-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/run_validation_v2_server.sh --commit "$(git rev-parse HEAD)" --mode full \
  --campaign-suffix "$DIRECT_SUFFIX" --max-workers 4
```

Path B — diagnostic preflight followed by full. Preflight creates an immutable
campaign seal, so the formal run uses a separately named suffix. Dependency
installation may be reused only after preflight successfully provisioned and
verified `.venv-server/bin/python` on this same host:

```bash
PREFLIGHT_SUFFIX="preflight-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/run_validation_v2_server.sh --commit "$(git rev-parse HEAD)" --mode preflight \
  --campaign-suffix "$PREFLIGHT_SUFFIX"

FORMAL_SUFFIX="formal-$(date -u +%Y%m%dT%H%M%SZ)"
test "$PREFLIGHT_SUFFIX" != "$FORMAL_SUFFIX"
bash scripts/run_validation_v2_server.sh --commit "$(git rev-parse HEAD)" --mode full \
  --campaign-suffix "$FORMAL_SUFFIX" --max-workers 4 --skip-dependency-install
```

The reuse option never skips Git, runtime, test, plan, or full training checks.
For the MatPool wrapper, the equivalent reuse form remains
`bash scripts/run_validation_v2_matpool.sh start --skip-dependency-install`, but
it is valid only after a successful dependency-provisioning run on this host.

Completion requires an inactive tmux session, a zero value in the reported
`run-*.exit`, eight completed `shard_execution.json` markers, and a new final
root whose `validation_report.json` is `complete` and whose summaries cover
seeds 2026–2030. Partial shards, a zero `start` status, or a quiet log are not
completion evidence.

## Historical implementation reference

The remaining sections preserve the previous manual operator contract for
auditability and incident analysis. Everything below this heading is
**Historical only**: it is not the current execution path and must not be mixed
with the MatPool or generic-runner commands above. In particular, old
platform-specific paths, Conda commands, runtime assertions, and hand-copied
helper functions are preserved evidence rather than prerequisites for a new
campaign.

## 1. Quarantine the legacy single-process run

The old `fcf81f8` single-process `matrix` root and the new commit's shard roots
must not be mixed, copied together, used as merge inputs, or summarized as one
campaign. Never delete the old root: it is immutable diagnostic evidence. Give
the new campaign a different, commit-qualified root.

Stopping the old process is optional. The block below discovers every process
that is both the legacy matrix command and the `fcf81f8` root. Zero matches is a
no-op, exactly one is eligible for `SIGINT`, and multiple matches abort without
choosing a PID.

```bash
set -Eeuo pipefail
export OLD_ROOT="/root/autodl-tmp/2025.12.28LNN_Imputation/results/validation_v2/server_full-fcf81f8"
export PROC_ROOT=/proc
test -d "$OLD_ROOT"
# Legacy command signature: python -m validation_v2.cli matrix
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mapfile -t OLD_MATCHES < <(
  pgrep -af '[p]ython -m validation_v2\.cli matrix' \
    | grep -F -- "$OLD_ROOT" || true
)
printf '%s\n' "${OLD_MATCHES[@]}"
case "${#OLD_MATCHES[@]}" in
  0)
    echo 'no legacy process; not sending a signal'
    ;;
  1)
    export OLD_PID="${OLD_MATCHES[0]%% *}"
    [[ "$OLD_PID" =~ ^[0-9]+$ ]] || {
      echo "non-numeric legacy PID: $OLD_PID" >&2
      exit 2
    }
    if test ! -r "$PROC_ROOT/$OLD_PID/stat" \
        -o ! -r "$PROC_ROOT/$OLD_PID/cmdline"; then
      echo "cannot verify legacy process identity in /proc: $OLD_PID" >&2
      exit 2
    fi
    OLD_STARTTIME="$(awk '{print $22}' "$PROC_ROOT/$OLD_PID/stat")"
    OLD_CMDLINE="$(tr '\0' ' ' < "$PROC_ROOT/$OLD_PID/cmdline")"
    if ! [[ "$OLD_STARTTIME" =~ ^[0-9]+$ \
        && "$OLD_CMDLINE" == *"python -m validation_v2.cli matrix"* \
        && "$OLD_CMDLINE" == *"$OLD_ROOT"* ]]; then
      echo "legacy process identity re-check failed: pid=$OLD_PID cmd=$OLD_CMDLINE" >&2
      exit 2
    fi
    ps -ww -p "$OLD_PID" -o pid=,lstart=,args=
    kill -INT "$OLD_PID"
    OLD_STOP_DEADLINE=$((SECONDS + 300))
    while test -r "$PROC_ROOT/$OLD_PID/stat"; do
      CURRENT_STARTTIME="$(awk '{print $22}' "$PROC_ROOT/$OLD_PID/stat")"
      if test "$CURRENT_STARTTIME" != "$OLD_STARTTIME"; then
        echo "legacy process exited and PID was reused; not signalling replacement: $OLD_PID"
        break
      fi
      if test "$SECONDS" -ge "$OLD_STOP_DEADLINE"; then
        echo "legacy PID did not stop after SIGINT: $OLD_PID" >&2
        exit 3
      fi
      ps -ww -p "$OLD_PID" -o pid=,etime=,args=
      sleep 5
    done
    echo "legacy PID stopped: $OLD_PID"
    ;;
  *)
    printf 'ambiguous legacy matches; refusing to choose a PID:\n%s\n' \
      "${OLD_MATCHES[*]}" >&2
    exit 2
    ;;
esac
test -d "$OLD_ROOT"
```

Zero matches is a safe no-op. One strictly numeric PID is shown with its full
command before `SIGINT`; multiple matches abort without choosing one. The
bounded conditional wait makes failure to stop explicit. Do not use `kill -9`,
remove the old root, or point a shard at it.

## 2. Freeze the validated source without leaking network state

Network Turbo is allowed only in short-lived clone, fetch, or dependency
installation subshells. It is not sourced in the training shell. Training
requires no network; equivalently, training requires no network access.

For an initial clone, supply the repository URL interactively or through the
server's existing credential helper; never paste a credential into this file,
the repository, shell history, or a log.

```bash
cd /root/autodl-tmp
(
  source /etc/network_turbo
  git clone "$REPOSITORY_URL" 2025.12.28LNN_Imputation
)
```

For an existing checkout, choose exactly one validated-ref workflow. A
validated branch may be fast-forwarded:

```bash
export REPO="/root/autodl-tmp/2025.12.28LNN_Imputation"
cd "$REPO"
(
  source /etc/network_turbo
  git fetch --all --tags --prune
  git checkout codex/validation-v2
  git pull --ff-only origin codex/validation-v2
)
```

Alternatively, fetch and detach at an exact validated commit:

```bash
export REPO="/root/autodl-tmp/2025.12.28LNN_Imputation"
cd "$REPO"
(
  source /etc/network_turbo
  git fetch --all --tags --prune
)
export VALIDATED_COMMIT="<40-HEX-VALIDATED-COMMIT>"
git checkout --detach "$VALIDATED_COMMIT"
```

In a clean shell that has not sourced Network Turbo, freeze and record the
actual commit. A branch name alone is not provenance.

```bash
export REPO="/root/autodl-tmp/2025.12.28LNN_Imputation"
cd "$REPO"
test -z "$(git status --porcelain)"
export COMMIT="$(git rev-parse HEAD)"
test "$(printf '%s' "$COMMIT" | wc -c)" -eq 40
git show -s --format='%H %cI %s' "$COMMIT"
```

## 3. Re-enter offline, pin the commit, and verify the runtime

Dependency installation is the final permitted network operation. Skip it when
the pinned environment is already complete.

```bash
(
  source /etc/network_turbo
  /root/miniconda3/envs/pinn_imu/bin/python -m pip install -r \
    /root/autodl-tmp/2025.12.28LNN_Imputation/requirements-validation-v2.txt
)
```

Open a new offline shell. Re-activate the environment, re-enter the repository,
and only then derive `COMMIT`. Construct every commit-qualified path after that
assignment; do not rely on variables inherited from the network shell.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/miniconda3/envs/pinn_imu
set -Eeuo pipefail
export REPO="/root/autodl-tmp/2025.12.28LNN_Imputation"
cd "$REPO"
export COMMIT="$(git rev-parse HEAD)"
export PREFLIGHT_DIR="/root/autodl-tmp/validation-v2-preflight-${COMMIT}-sharded-v2"
export AUDIT_DIR="/root/autodl-tmp/validation-v2-audit-${COMMIT}-sharded-v2"
export PLAN="$AUDIT_DIR/server-full-8-shards-${COMMIT}.json"
export SHARDS_ROOT="$REPO/results/validation_v2/server-full-shards-${COMMIT}-sharded-v2"
export FINAL_ROOT="$REPO/results/validation_v2/server-full-final-${COMMIT}-sharded-v2"
export CONFIG="/root/autodl-tmp/2025.12.28LNN_Imputation/configs/validation_v2/server_full.yaml"
export SHARD_LOG_PATTERN="$AUDIT_DIR/shard-NNN.log"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
test "$(printf '%s' "$COMMIT" | wc -c)" -eq 40
test -z "$(git status --porcelain)"
test -f "$CONFIG"
test ! -e "$PREFLIGHT_DIR"
test ! -e "$AUDIT_DIR"
test ! -e "$SHARDS_ROOT"
test ! -e "$FINAL_ROOT"
mkdir -p "$PREFLIGHT_DIR" "$AUDIT_DIR" "$SHARDS_ROOT"
printf '%s\n' "$COMMIT" > "$AUDIT_DIR/COMMIT"
git status --porcelain > "$AUDIT_DIR/git-status-porcelain.txt"
test ! -s "$AUDIT_DIR/git-status-porcelain.txt"
```

The following check is intentionally exact. Stop if Python, any dependency,
Torch's local CUDA build, CUDA availability, or the 4090 D model differs.

```bash
python - <<'PY'
import importlib.metadata as md
import os
import platform
import torch

expected = {
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "scipy": "1.13.1",
    "PyYAML": "6.0.3",
    "pytest": "8.4.2",
    "torch": "2.3.1+cu121",
    "ncps": "1.0.1",
}
actual = {name: md.version(name) for name in expected}
assert platform.system() == "Linux", platform.system()
assert platform.python_version().startswith("3.9."), platform.python_version()
assert os.environ["CONDA_PREFIX"] == "/root/miniconda3/envs/pinn_imu"
assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
assert actual == expected, (actual, expected)
assert torch.__version__ == "2.3.1+cu121", torch.__version__
assert torch.cuda.is_available()
assert torch.version.cuda == "12.1", torch.version.cuda
assert "4090 D" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print(platform.python_version(), actual, torch.cuda.get_device_name(0))
PY
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv
df -h "$REPO" /root/autodl-tmp
df -Pk /root/autodl-tmp | awk 'NR==2 { if ($4 < 104857600) exit 1 }'
```

Run the real Linux no-replace race separately and require `1 passed`; a
`skipped` result is a hard stop. Then run the complete suite.

```bash
test "$(uname -s)" = Linux
python -m pytest -q \
  tests/validation_v2/test_sharding.py::test_linux_rename_noreplace_survives_real_directory_race \
  -rs | tee "$PREFLIGHT_DIR/linux-renameat2.txt"
test "${PIPESTATUS[0]}" -eq 0
grep -F '1 passed' "$PREFLIGHT_DIR/linux-renameat2.txt"
! grep -qi 'skipped' "$PREFLIGHT_DIR/linux-renameat2.txt"
python -m pytest -q | tee "$PREFLIGHT_DIR/pytest-full.txt"
test "${PIPESTATUS[0]}" -eq 0
cp "$PREFLIGHT_DIR/linux-renameat2.txt" "$PREFLIGHT_DIR/pytest-full.txt" "$AUDIT_DIR/"
```

## 4. Freeze and verify the eight-shard plan

The dry-run has one header plus 4,095 cells. The plan must say 175 training
groups, 4,095 cells, and 8 shards.

```bash
python -m validation_v2.cli matrix --config "$CONFIG" --dry-run \
  > "$AUDIT_DIR/matrix-dry-run.jsonl"
python - "$AUDIT_DIR/matrix-dry-run.jsonl" <<'PY'
import json
import sys

lines = open(sys.argv[1], encoding="utf-8").read().splitlines()
header = json.loads(lines[0])
assert header["combination_count"] == 4095, header
assert len(lines) == 4096, len(lines)
PY
python -m validation_v2.cli shard-plan \
  --config "$CONFIG" --shard-count 8 --output "$PLAN" --device cuda \
  | tee "$AUDIT_DIR/shard-plan.stdout.json"
python - "$PLAN" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
assert plan["total_groups"] == 175, plan
assert plan["total_cells"] == 4095, plan
assert plan["shard_count"] == 8, plan
assert len(plan["shards"]) == 8, plan
PY
```

## 5. Install the auditable rollout helpers

Each command runs its shard's full assigned group list, not one group. Each
shard has an independent `$SHARDS_ROOT/NNN` root and
`$AUDIT_DIR/shard-NNN.log`. The `.shard_execution.lock` is the final duplicate
guard, but the operator still checks roots and PIDs before launch. Training can
be quiet; stdout contains the final JSON only when the shard command returns.

Define these copy-pasteable Bash functions in the offline shell:

`wait_until_groups` and `wait_stage_metrics` are four-hour rollout-sampling
waits. They are deliberately separate from formal completion waits:
`wait_shard` defaults to a seven-day total deadline and a six-hour no-progress
deadline, while `wait_all_shards` defaults to fourteen days total and six hours
without any new completed group. Override the corresponding
`SHARD_WAIT_*`/`ALL_SHARDS_WAIT_*` variables only with a recorded operational
reason; a healthy 25-hour shard must not be treated as a timeout.

```bash
audit_active() {
  local now last=0 stamp="$AUDIT_DIR/.last-60s-audit"
  now="$(date +%s)"
  test ! -f "$stamp" || last="$(cat "$stamp")"
  test "$((now - last))" -ge 60 || return 0
  printf '%s\n' "$now" > "$stamp"
  {
    date -Is
    local shard marker status groups pid alive
    for shard in "$@"; do
      marker="$SHARDS_ROOT/$shard/shard_execution.json"
      status=missing
      groups=0
      if test -f "$marker"; then
        read -r status groups < <(python -c \
          'import json,sys; marker=json.load(open(sys.argv[1])); print(marker["status"], len(marker["group_runs"]))' \
          "$marker")
      fi
      pid=missing
      alive=no
      if test -f "$AUDIT_DIR/shard-$shard.pid"; then
        pid="$(cat "$AUDIT_DIR/shard-$shard.pid")"
        kill -0 "$pid" 2>/dev/null && alive=yes || true
      fi
      printf 'shard=%s marker=%s group_runs=%s pid=%s alive=%s\n' \
        "$shard" "$status" "$groups" "$pid" "$alive"
    done
    pgrep -af 'validation_v2\.cli shard' || true
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
  } | tee -a "$AUDIT_DIR/monitor-60s.log"
}

launch_shard() {
  local shard="${1-}"
  if ! [[ "$shard" =~ ^[0-7]{3}$ ]]; then
    echo "invalid zero-padded shard index: $shard" >&2
    return 2
  fi
  local index=$((10#$shard))
  if test "$index" -lt 0 -o "$index" -ge 8; then
    echo "shard index outside formal plan: $shard" >&2
    return 2
  fi
  if test -z "${CONFIG-}" -o ! -r "${CONFIG-}"; then
    echo "CONFIG is missing or unreadable: ${CONFIG-}" >&2
    return 2
  fi
  if test -z "${PLAN-}" -o ! -r "${PLAN-}"; then
    echo "PLAN is missing or unreadable: ${PLAN-}" >&2
    return 2
  fi
  if test -z "${SHARDS_ROOT-}" -o ! -d "${SHARDS_ROOT-}" \
      -o -z "${AUDIT_DIR-}" -o ! -d "${AUDIT_DIR-}"; then
    echo 'SHARDS_ROOT and AUDIT_DIR must already be directories' >&2
    return 2
  fi
  local output_root="$SHARDS_ROOT/$shard"
  local pid_file="$AUDIT_DIR/shard-$shard.pid"
  if [[ -e "$output_root" || -L "$output_root" ]]; then
    echo "shard root already exists or is linked: $output_root" >&2
    return 2
  fi
  if [[ -e "$pid_file" || -L "$pid_file" ]]; then
    local old_pid=""
    if test -r "$pid_file"; then
      read -r old_pid < "$pid_file" || true
    fi
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "recorded shard PID is alive: $shard pid=$old_pid" >&2
    else
      echo "stale/invalid shard PID file must be diagnosed, not overwritten: $pid_file" >&2
    fi
    return 2
  fi
  if pgrep -af "validation_v2\.cli shard.*--shard-index $index" \
      | grep -F -- "$PLAN"; then
    echo "duplicate shard process: $shard" >&2
    return 2
  fi
  nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    /root/miniconda3/envs/pinn_imu/bin/python -m validation_v2.cli shard \
    --config "$CONFIG" --plan "$PLAN" --shard-index "$index" \
    --output-root "$output_root" --device cuda \
    > "$AUDIT_DIR/shard-$shard.log" 2>&1 &
  local pid="$!"
  if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
    echo "nohup did not return a numeric PID: $pid" >&2
    return 3
  fi
  if ! printf '%s\n' "$pid" > "$pid_file"; then
    kill "$pid" 2>/dev/null || true
    return 3
  fi
}

wait_shard() {
  local shard="$1"
  local marker="$SHARDS_ROOT/$shard/shard_execution.json"
  local max_seconds="${SHARD_WAIT_MAX_SECONDS:-604800}"
  local max_idle_seconds="${SHARD_WAIT_MAX_IDLE_SECONDS:-21600}"
  local poll_seconds="${SHARD_WAIT_POLL_SECONDS:-60}"
  if ! [[ "$max_seconds" =~ ^[1-9][0-9]*$ \
      && "$max_idle_seconds" =~ ^[1-9][0-9]*$ \
      && "$poll_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo 'invalid full-shard wait timeout/idle/poll settings' >&2
    return 2
  fi
  local deadline=$((SECONDS + max_seconds))
  local last_progress="$SECONDS"
  local last_groups=-1
  while :; do
    audit_active "$shard" || return $?
    local groups=0
    if test -f "$marker"; then
      if STATE="$(python - "$marker" <<'PY'
import json
import sys

marker = json.load(open(sys.argv[1], encoding="utf-8"))
print(marker["status"], len(marker["group_runs"]))
if marker["status"] == "failed":
    print(json.dumps(marker, sort_keys=True), file=sys.stderr)
    raise SystemExit(2)
if marker["status"] == "completed":
    raise SystemExit(0)
if marker["status"] != "started":
    raise SystemExit(3)
raise SystemExit(4)
PY
)"; then
        printf '%s %s\n' "$(date -Is)" "$shard $STATE" \
          | tee -a "$AUDIT_DIR/wait-shards.log"
        return 0
      else
        local rc=$?
        if test "$rc" -eq 2 -o "$rc" -eq 3; then
          tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
          return "$rc"
        fi
        local pid
        test -f "$AUDIT_DIR/shard-$shard.pid" || return 3
        pid="$(cat "$AUDIT_DIR/shard-$shard.pid")"
        if ! kill -0 "$pid" 2>/dev/null; then
          echo "started marker but shard PID is gone: $shard pid=$pid" \
            | tee -a "$AUDIT_DIR/wait-shards.log" >&2
          return 3
        fi
        groups="${STATE##* }"
        if ! [[ "$groups" =~ ^[0-9]+$ ]]; then
          echo "invalid group count while waiting for shard: $shard state=$STATE" >&2
          return 3
        fi
      fi
    elif test -f "$AUDIT_DIR/shard-$shard.pid"; then
      local pid
      pid="$(cat "$AUDIT_DIR/shard-$shard.pid")"
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "missing marker and shard PID is gone: $shard pid=$pid" \
          | tee -a "$AUDIT_DIR/wait-shards.log" >&2
        tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
        return 3
      fi
    else
      echo "missing marker and PID file: $shard" >&2
      tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
      return 3
    fi
    if test "$groups" -ne "$last_groups"; then
      last_groups="$groups"
      last_progress="$SECONDS"
    fi
    if test "$SECONDS" -ge "$deadline"; then
      echo "full-shard total timeout: shard=$shard SHARD_WAIT_MAX_SECONDS=$max_seconds" >&2
      tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
      return 4
    fi
    if test "$((SECONDS - last_progress))" -ge "$max_idle_seconds"; then
      echo "full-shard no-progress timeout: shard=$shard SHARD_WAIT_MAX_IDLE_SECONDS=$max_idle_seconds groups=$groups" >&2
      tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
      return 4
    fi
    sleep "$poll_seconds"
  done
}

wait_all_shards() {
  local -a shards=("$@")
  local max_seconds="${ALL_SHARDS_WAIT_MAX_SECONDS:-1209600}"
  local max_idle_seconds="${ALL_SHARDS_WAIT_MAX_IDLE_SECONDS:-21600}"
  local poll_seconds="${ALL_SHARDS_WAIT_POLL_SECONDS:-60}"
  if ! [[ "$max_seconds" =~ ^[1-9][0-9]*$ \
      && "$max_idle_seconds" =~ ^[1-9][0-9]*$ \
      && "$poll_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo 'invalid all-shard wait timeout/idle/poll settings' >&2
    return 2
  fi
  local deadline=$((SECONDS + max_seconds))
  local last_progress="$SECONDS"
  local last_total_groups=-1
  while :; do
    local completed=0 outcome=0 total_groups=0
    local shard marker pid_file pid alive status groups
    for shard in "${shards[@]}"; do
      marker="$SHARDS_ROOT/$shard/shard_execution.json"
      pid_file="$AUDIT_DIR/shard-$shard.pid"
      pid=missing
      alive=no
      status=missing
      groups=0
      if [[ -f "$pid_file" && -r "$pid_file" ]]; then
        read -r pid < "$pid_file"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
          alive=yes
        fi
      fi
      if test -f "$marker"; then
        read -r status groups < <(python -c \
          'import json,sys; marker=json.load(open(sys.argv[1])); print(marker["status"], len(marker["group_runs"]))' \
          "$marker")
      fi
      if [[ "$groups" =~ ^[0-9]+$ ]]; then
        total_groups=$((total_groups + groups))
      else
        echo "invalid group count: $shard groups=$groups" \
          | tee -a "$AUDIT_DIR/wait-all-shards.log" >&2
        outcome=3
      fi
      printf '%s shard=%s status=%s group_runs=%s pid=%s alive=%s\n' \
        "$(date -Is)" "$shard" "$status" "$groups" "$pid" "$alive" \
        | tee -a "$AUDIT_DIR/wait-all-shards.log"
      case "$status" in
        completed)
          completed=$((completed + 1))
          ;;
        failed)
          cat "$marker" | tee -a "$AUDIT_DIR/wait-all-shards.log" >&2
          tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
          outcome=2
          ;;
        missing|started)
          if test "$alive" != yes -a "$outcome" -ne 2; then
            echo "nonterminal shard has no live PID: $shard status=$status pid=$pid" \
              | tee -a "$AUDIT_DIR/wait-all-shards.log" >&2
            tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
            outcome=3
          fi
          ;;
        *)
          echo "invalid shard status: $shard status=$status" \
            | tee -a "$AUDIT_DIR/wait-all-shards.log" >&2
          test "$outcome" -eq 2 || outcome=3
          ;;
      esac
    done
    pgrep -af 'validation_v2\.cli shard' \
      | tee -a "$AUDIT_DIR/wait-all-shards.log" || true
    if ! nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits \
        | tee -a "$AUDIT_DIR/wait-all-shards.log"; then
      echo 'GPU audit command failed' >&2
      return 3
    fi
    case "$outcome" in
      2) return 2 ;;
      3) return 3 ;;
    esac
    test "$completed" -ne "${#shards[@]}" || return 0
    if test "$total_groups" -ne "$last_total_groups"; then
      last_total_groups="$total_groups"
      last_progress="$SECONDS"
    fi
    if test "$SECONDS" -ge "$deadline"; then
      echo "all-shard total timeout: completed=$completed/${#shards[@]} groups=$total_groups ALL_SHARDS_WAIT_MAX_SECONDS=$max_seconds" \
        | tee -a "$AUDIT_DIR/wait-all-shards.log" >&2
      return 4
    fi
    if test "$((SECONDS - last_progress))" -ge "$max_idle_seconds"; then
      echo "all-shard no-progress timeout: completed=$completed/${#shards[@]} groups=$total_groups ALL_SHARDS_WAIT_MAX_IDLE_SECONDS=$max_idle_seconds" \
        | tee -a "$AUDIT_DIR/wait-all-shards.log" >&2
      return 4
    fi
    sleep "$poll_seconds"
  done
}

run_queue() {
  if test "$#" -lt 2; then
    echo 'run_queue requires MAX_PARALLEL and at least one shard index' >&2
    return 2
  fi
  local max_parallel="${1-}"
  shift || return 2
  if ! [[ "$max_parallel" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid queue parallelism: $max_parallel" >&2
    return 2
  fi
  local queue_max_seconds="${QUEUE_MAX_SECONDS:-1209600}"
  local queue_max_idle_seconds="${QUEUE_MAX_IDLE_SECONDS:-21600}"
  local queue_poll_seconds="${QUEUE_POLL_SECONDS:-10}"
  if ! [[ "$queue_max_seconds" =~ ^[1-9][0-9]*$ \
      && "$queue_max_idle_seconds" =~ ^[1-9][0-9]*$ \
      && "$queue_poll_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo 'invalid queue timeout/idle/poll settings' >&2
    return 2
  fi
  local deadline=$((SECONDS + queue_max_seconds))
  local last_progress="$SECONDS"
  local last_completed_groups=-1
  local -a pending=("$@")
  local -a all_shards=("$@")
  local -a active=()
  while test "${#pending[@]}" -gt 0 -o "${#active[@]}" -gt 0; do
    while test "${#pending[@]}" -gt 0 -a "${#active[@]}" -lt "$max_parallel"; do
      local shard="${pending[0]}"
      pending=("${pending[@]:1}")
      launch_shard "$shard" || return $?
      active+=("$shard")
    done
    local -a next=()
    for shard in "${active[@]}"; do
      local marker="$SHARDS_ROOT/$shard/shard_execution.json"
      if test -f "$marker"; then
        local state groups
        read -r state groups < <(python -c \
          'import json,sys; marker=json.load(open(sys.argv[1])); print(marker["status"], len(marker["group_runs"]))' \
          "$marker")
        if test "$state" = failed; then
          cat "$marker" >&2
          tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
          return 2
        fi
        if test "$state" != started -a "$state" != completed; then
          echo "invalid queue marker status: $shard status=$state" >&2
          return 3
        fi
        if test "$state" != completed; then
          local pid
          pid="$(cat "$AUDIT_DIR/shard-$shard.pid")"
          if ! kill -0 "$pid" 2>/dev/null; then
            echo "nonterminal marker but shard PID is gone: $shard" >&2
            return 3
          fi
          next+=("$shard")
        fi
      else
        local pid
        pid="$(cat "$AUDIT_DIR/shard-$shard.pid")"
        if ! kill -0 "$pid" 2>/dev/null; then
          echo "missing marker and shard PID is gone: $shard" >&2
          return 3
        fi
        next+=("$shard")
      fi
    done
    active=("${next[@]}")
    audit_active "${all_shards[@]}" || return $?
    local completed_groups=0
    for shard in "${all_shards[@]}"; do
      local marker="$SHARDS_ROOT/$shard/shard_execution.json"
      if test -f "$marker"; then
        local state groups
        read -r state groups < <(python -c \
          'import json,sys; marker=json.load(open(sys.argv[1])); print(marker["status"], len(marker["group_runs"]))' \
          "$marker")
        if test "$state" = failed; then
          cat "$marker" >&2
          return 2
        fi
        completed_groups=$((completed_groups + groups))
      fi
    done
    if test "$completed_groups" -ne "$last_completed_groups"; then
      last_completed_groups="$completed_groups"
      last_progress="$SECONDS"
    fi
    printf '%s max=%s pending=%s active=%s\n' \
      "$(date -Is)" "$max_parallel" "${pending[*]}" "${active[*]}" \
      | tee -a "$AUDIT_DIR/queue.log"
    if test "${#pending[@]}" -eq 0 -a "${#active[@]}" -eq 0; then
      return 0
    fi
    if test "$SECONDS" -ge "$deadline"; then
      echo "queue deadline exceeded: QUEUE_MAX_SECONDS=$queue_max_seconds" \
        | tee -a "$AUDIT_DIR/queue.log" >&2
      return 4
    fi
    if test "$((SECONDS - last_progress))" -ge "$queue_max_idle_seconds"; then
      echo "queue idle timeout: QUEUE_MAX_IDLE_SECONDS=$queue_max_idle_seconds" \
        | tee -a "$AUDIT_DIR/queue.log" >&2
      return 4
    fi
    test "${#pending[@]}" -eq 0 -a "${#active[@]}" -eq 0 || \
      sleep "$queue_poll_seconds"
  done
}

declare -Ag GPU_SAMPLER_JOBS=()

start_gpu_sampler() {
  local label="${1-}"
  if ! [[ "$label" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
    echo "invalid GPU sampler label: $label" >&2
    return 2
  fi
  local csv="$AUDIT_DIR/gpu-$label.csv"
  local pid_file="$AUDIT_DIR/gpu-$label.pid"
  local identity="validation-v2-gpu-sampler-$label"
  local proc_root="${PROC_ROOT:-/proc}"
  if [[ -e "$pid_file" || -L "$pid_file" ]]; then
    echo "GPU sampler PID file already exists: $pid_file" >&2
    return 2
  fi
  printf 'timestamp_utc,memory_used_mib,memory_total_mib,utilization_percent\n' > "$csv" \
    || return 3
  bash -c '
set -Eeuo pipefail
csv="$1"
while :; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits \
    | awk -F, -v timestamp="$timestamp" \
      '\''{gsub(/ /, ""); print timestamp "," $1 "," $2 "," $3}'\'' \
    | tee -a "$csv"
  sleep 10
done
' "$identity" "$csv" &
  local pid="$!"
  GPU_SAMPLER_JOBS["$label"]="$pid"
  local attempt
  for attempt in $(seq 1 50); do
    test -r "$proc_root/$pid/stat" -a -r "$proc_root/$pid/cmdline" && break
    sleep 0.1
  done
  if ! [[ "$pid" =~ ^[0-9]+$ ]] \
      || test ! -r "$proc_root/$pid/stat" \
      || test ! -r "$proc_root/$pid/cmdline"; then
    echo "unable to establish GPU sampler process identity: $pid" >&2
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    return 3
  fi
  local starttime cmdline
  starttime="$(awk '{print $22}' "$proc_root/$pid/stat")"
  cmdline="$(tr '\0' ' ' < "$proc_root/$pid/cmdline")"
  if ! [[ "$starttime" =~ ^[0-9]+$ \
      && "$cmdline" == *"$identity"* \
      && "$cmdline" == *"nvidia-smi"* ]]; then
    echo "GPU sampler identity verification failed immediately: $pid" >&2
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    return 3
  fi
  printf '%s %s\n' "$pid" "$starttime" > "$pid_file" || {
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    return 3
  }
}

stop_gpu_sampler() {
  local label="${1-}"
  local pid_file="$AUDIT_DIR/gpu-$label.pid"
  local proc_root="${PROC_ROOT:-/proc}"
  local identity="validation-v2-gpu-sampler-$label"
  if ! [[ -f "$pid_file" && -r "$pid_file" ]]; then
    echo "missing/unreadable GPU sampler PID file: $pid_file" >&2
    return 3
  fi
  local pid expected_start extra=""
  read -r pid expected_start extra < "$pid_file" || {
    echo "invalid GPU sampler PID record: $pid_file" >&2
    return 3
  }
  if ! [[ "$pid" =~ ^[0-9]+$ && "$expected_start" =~ ^[0-9]+$ \
      && -z "$extra" ]]; then
    echo "GPU sampler PID record must contain exactly two numeric fields" >&2
    return 3
  fi
  if test "${GPU_SAMPLER_JOBS[$label]-}" != "$pid"; then
    echo "GPU sampler is not the current shell job: label=$label pid=$pid" >&2
    return 3
  fi
  if test ! -r "$proc_root/$pid/stat" -o ! -r "$proc_root/$pid/cmdline"; then
    echo "GPU sampler process identity is unavailable; refusing kill: $pid" >&2
    return 3
  fi
  local actual_start cmdline
  actual_start="$(awk '{print $22}' "$proc_root/$pid/stat")"
  cmdline="$(tr '\0' ' ' < "$proc_root/$pid/cmdline")"
  if test "$actual_start" != "$expected_start" \
      || [[ "$cmdline" != *"$identity"* ]] \
      || [[ "$cmdline" != *"nvidia-smi"* ]]; then
    echo "GPU sampler identity mismatch; refusing kill: label=$label pid=$pid" >&2
    return 3
  fi
  kill "$pid" || return 3
  wait "$pid" 2>/dev/null || true
  unset 'GPU_SAMPLER_JOBS[$label]'
}

wait_until_groups() {
  local shard="$1"
  local required="$2"
  local marker="$SHARDS_ROOT/$shard/shard_execution.json"
  local pid_file="$AUDIT_DIR/shard-$shard.pid"
  local deadline=$((SECONDS + 14400))
  while :; do
    audit_active "$shard" || return $?
    local pid=""
    if test -f "$pid_file"; then
      pid="$(cat "$pid_file")"
    fi
    if test -f "$marker"; then
      local status groups
      read -r status groups < <(python - "$marker" <<'PY'
import json
import sys

marker = json.load(open(sys.argv[1], encoding="utf-8"))
print(marker["status"], len(marker["group_runs"]))
PY
)
      if test "$status" = failed; then
        cat "$marker" >&2
        tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
        return 2
      fi
      if test "$groups" -ge "$required"; then
        return 0
      fi
      if test "$status" = completed; then
        echo "completed before required group count: shard=$shard groups=$groups required=$required" >&2
        tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
        return 4
      fi
      if test "$status" != started; then
        echo "invalid marker status: shard=$shard status=$status" >&2
        return 3
      fi
    fi
    if test -z "$pid" || ! kill -0 "$pid" 2>/dev/null; then
      echo "marker missing/started but shard PID is gone: shard=$shard pid=${pid:-missing}" >&2
      tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
      return 3
    fi
    if test "$SECONDS" -ge "$deadline"; then
      echo "baseline group wait timeout after 14400 seconds: shard=$shard" >&2
      tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
      return 4
    fi
    sleep 10
  done
}
```

Install one audit helper that computes real, ordered group durations. For each
shard, the first duration is the first run ledger's completion minus
`marker.started_at`; later durations are adjacent
`test_evaluation.completed_at` differences in `marker.group_runs` order. GPU
peak is the maximum observed CSV sample, not the last sample or a cumulative
average. Its `groups_per_hour` field is the aggregate completed groups/hour for
the stage window, while `per_shard_new_groups` records the same window by active
shard. Every listed active shard must contribute at least one completion after
`stage_start`; aggregate progress from a subset cannot open the gate. A still
active but not-yet-large-enough window exits 4 and continues sampling. A shard
that already completed without contributing after `stage_start` makes the
stage contract impossible and exits 3 immediately. A failed marker exits 2; a
performance/resource assertion exits 10 after writing the metrics JSON. Only
exit 10 is eligible for a lower-concurrency fallback.

```bash
cat > "$AUDIT_DIR/collect_stage_metrics.py" <<'PY'
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


parser = argparse.ArgumentParser()
parser.add_argument("--shards-root", type=Path, required=True)
parser.add_argument("--indices", nargs="+", required=True)
parser.add_argument("--stage-start", required=True)
parser.add_argument("--gpu-csv", type=Path, required=True)
parser.add_argument("--minimum-groups", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--baseline", type=Path)
args = parser.parse_args()

stage_start = timestamp(args.stage_start)
durations = []
completion_times = []
marker_statuses = {}
per_shard_new_groups = {}
for shard in args.indices:
    marker_path = args.shards_root / shard / "shard_execution.json"
    if not marker_path.is_file():
        raise SystemExit(4)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker_statuses[shard] = marker["status"]
    if marker["status"] == "failed":
        print(json.dumps(marker, sort_keys=True))
        raise SystemExit(2)
    assert marker["status"] in {"started", "completed"}, marker
    previous = timestamp(marker["started_at"])
    new_groups = 0
    for binding in marker["group_runs"]:
        run_id = binding["run_ids"][0]
        ledger_path = args.shards_root / shard / run_id / "test_evaluation.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        completed = timestamp(ledger["completed_at"])
        duration = (completed - previous).total_seconds()
        assert duration > 0, (shard, run_id, duration)
        previous = completed
        if completed >= stage_start:
            durations.append(duration)
            completion_times.append(completed)
            new_groups += 1
    per_shard_new_groups[shard] = new_groups

missing_shard_progress = [
    shard for shard, count in per_shard_new_groups.items() if count < 1
]
completed_without_progress = [
    shard
    for shard in missing_shard_progress
    if marker_statuses[shard] == "completed"
]
if completed_without_progress:
    print(
        "completed active shards cannot contribute after stage_start: "
        f"{completed_without_progress}; per_shard_new_groups="
        f"{per_shard_new_groups}"
    )
    raise SystemExit(3)
if missing_shard_progress:
    print(
        "waiting: every active shard needs one new group after stage_start; "
        f"per_shard_new_groups={per_shard_new_groups}"
    )
    raise SystemExit(4)
if len(durations) < args.minimum_groups:
    print(f"waiting: {len(durations)}/{args.minimum_groups} new groups")
    raise SystemExit(4)

with args.gpu_csv.open(newline="", encoding="utf-8") as handle:
    samples = list(csv.DictReader(handle))
assert samples, "GPU sampler produced no rows"
used_samples = [float(row["memory_used_mib"]) for row in samples]
total_samples = [float(row["memory_total_mib"]) for row in samples]
peak_used = max(used_samples)
peak_gpu_memory_ratio = max(
    used / total for used, total in zip(used_samples, total_samples)
)
elapsed_hours = (max(completion_times) - stage_start).total_seconds() / 3600
assert elapsed_hours > 0, elapsed_hours
metrics = {
    "stage_start": stage_start.isoformat(),
    "indices": args.indices,
    "marker_statuses": marker_statuses,
    "per_shard_new_groups": per_shard_new_groups,
    "new_group_count": len(durations),
    "group_durations_seconds": durations,
    "groups_per_hour": len(durations) / elapsed_hours,
    "median_group_seconds": statistics.median(durations),
    "peak_gpu_memory_mib": peak_used,
    "peak_gpu_memory_ratio": peak_gpu_memory_ratio,
}
baseline = None
if args.baseline:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    metrics["baseline"] = str(args.baseline)
    metrics["gate_passed"] = (
        metrics["groups_per_hour"] >= baseline["groups_per_hour"] * 1.5
        and metrics["median_group_seconds"] < baseline["median_group_seconds"] * 1.8
        and metrics["peak_gpu_memory_ratio"] < 0.8
    )
else:
    metrics["gate_passed"] = metrics["peak_gpu_memory_ratio"] < 0.8
args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps(metrics, sort_keys=True))
try:
    assert all(count >= 1 for count in metrics["per_shard_new_groups"].values())
    if baseline:
        assert metrics["groups_per_hour"] >= baseline["groups_per_hour"] * 1.5
        assert metrics["median_group_seconds"] < baseline["median_group_seconds"] * 1.8
    assert metrics["peak_gpu_memory_ratio"] < 0.8
except AssertionError as error:
    print(f"performance/resource gate failed: {error}")
    raise SystemExit(10) from error
PY

wait_stage_metrics() {
  local label="$1"
  local stage_start="$2"
  local baseline="$3"
  local output="$4"
  local minimum_groups="$5"
  shift 5
  local -a active_indices=("$@")
  local deadline=$((SECONDS + 14400))
  while :; do
    audit_active "${active_indices[@]}" || return $?
    local all_completed=yes
    local shard marker pid status groups
    for shard in "${active_indices[@]}"; do
      marker="$SHARDS_ROOT/$shard/shard_execution.json"
      pid=""
      test ! -f "$AUDIT_DIR/shard-$shard.pid" || \
        pid="$(cat "$AUDIT_DIR/shard-$shard.pid")"
      if test -f "$marker"; then
        read -r status groups < <(python -c \
          'import json,sys; marker=json.load(open(sys.argv[1])); print(marker["status"], len(marker["group_runs"]))' \
          "$marker")
        if test "$status" = failed; then
          cat "$marker" >&2
          tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
          return 2
        fi
        if test "$status" = completed; then
          continue
        fi
        if test "$status" != started; then
          echo "invalid marker status: shard=$shard status=$status" >&2
          return 3
        fi
      fi
      all_completed=no
      if test -z "$pid" || ! kill -0 "$pid" 2>/dev/null; then
        echo "active stage shard PID is gone: shard=$shard pid=${pid:-missing}" >&2
        tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
        return 3
      fi
    done
    local -a command=(
      /root/miniconda3/envs/pinn_imu/bin/python "$AUDIT_DIR/collect_stage_metrics.py"
      --shards-root "$SHARDS_ROOT" --indices "${active_indices[@]}" --stage-start "$stage_start"
      --gpu-csv "$AUDIT_DIR/gpu-$label.csv" --minimum-groups "$minimum_groups"
      --output "$output"
    )
    test -z "$baseline" || command+=(--baseline "$baseline")
    local rc
    if "${command[@]}" 2>&1 | tee -a "$AUDIT_DIR/gate-$label.log"; then
      rc=0
    else
      rc="${PIPESTATUS[0]}"
    fi
    case "$rc" in
      0|10) return "$rc" ;;
      2) return 2 ;;
      4) ;;
      *) return 3 ;;
    esac
    if test "$all_completed" = yes; then
      echo "all active shards completed before minimum new stage groups: $minimum_groups" >&2
      for shard in "${active_indices[@]}"; do
        tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
      done
      return 4
    fi
    if test "$SECONDS" -ge "$deadline"; then
      echo "stage metrics timeout after 14400 seconds: $label" >&2
      for shard in "${active_indices[@]}"; do
        tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
      done
      return 4
    fi
    sleep 10
  done
}
```

## 6. Establish the executable one-worker baseline

Start only shard 000. Wait for two completed `group_runs` when possible (the
minimum is explicitly two here), then compute and save the real duration
sequence and baseline JSON. Shard 000 continues its full assignment afterward.

```bash
start_gpu_sampler baseline-1worker
launch_shard 000
wait_until_groups 000 2 || exit $?
BASELINE_START="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["started_at"])' \
  "$SHARDS_ROOT/000/shard_execution.json")"
wait_stage_metrics baseline-1worker "$BASELINE_START" "" \
  "$AUDIT_DIR/baseline-1worker.json" 2 000 || exit $?
stop_gpu_sampler baseline-1worker
```

## 7. Execute the 2 -> 4 -> 8 rollout and its fallback queues

The two-worker stage starts when shard 001 is launched while shard 000 keeps
running. The UTC stage start, 10-second GPU samples, gate log, and metrics JSON
are all retained. The gate waits for new ledger completions after the stage
start from every active shard (`000..001`, then `000..003`, then `000..007`),
with at least one completion per shard, and actually asserts throughput >= 1.5x
the prior stable stage, median group time < 1.8x baseline, peak memory < 80%,
and no failed marker. In percentage terms, throughput improvement must be >=
50%, median group-time growth must be < 80%, and GPU memory must remain < 80%.

If the two-worker gate fails only its performance/resource assertions, let 000
and 001 finish, then run 002..007 serially with `MAX_PARALLEL=1`. A failed shard
marker is not a performance fallback and exits immediately.

```bash
date -u +%Y-%m-%dT%H:%M:%S+00:00 | tee "$AUDIT_DIR/stage-2worker-start.txt"
STAGE2_START="$(cat "$AUDIT_DIR/stage-2worker-start.txt")"
start_gpu_sampler stage-2worker
launch_shard 001
if wait_stage_metrics stage-2worker "$STAGE2_START" \
    "$AUDIT_DIR/baseline-1worker.json" \
    "$AUDIT_DIR/stage-2worker-metrics.json" 2 000 001; then
  STAGE2_RC=0
else
  STAGE2_RC=$?
fi
stop_gpu_sampler stage-2worker
case "$STAGE2_RC" in
  0) ;;
  10) echo 'two-worker performance/resource gate failed' | tee -a "$AUDIT_DIR/rollout.log" ;;
  2|3|4) exit "$STAGE2_RC" ;;
  *) echo "unexpected two-worker gate status: $STAGE2_RC" >&2; exit 3 ;;
esac

if test "$STAGE2_RC" -eq 10; then
  echo 'two-worker gate failed; finish 000/001, then MAX_PARALLEL=1' \
    | tee -a "$AUDIT_DIR/rollout.log"
  wait_shard 000 || exit $?
  wait_shard 001 || exit $?
  run_queue 1 002 003 004 005 006 007 || exit $?
else
  date -u +%Y-%m-%dT%H:%M:%S+00:00 | tee "$AUDIT_DIR/stage-4worker-start.txt"
  STAGE4_START="$(cat "$AUDIT_DIR/stage-4worker-start.txt")"
  start_gpu_sampler stage-4worker
  launch_shard 002
  launch_shard 003
  if wait_stage_metrics stage-4worker "$STAGE4_START" \
      "$AUDIT_DIR/stage-2worker-metrics.json" \
      "$AUDIT_DIR/stage-4worker-metrics.json" 2 000 001 002 003; then
    STAGE4_RC=0
  else
    STAGE4_RC=$?
  fi
  stop_gpu_sampler stage-4worker
  case "$STAGE4_RC" in
    0) ;;
    10) echo 'four-worker performance/resource gate failed' | tee -a "$AUDIT_DIR/rollout.log" ;;
    2|3|4) exit "$STAGE4_RC" ;;
    *) echo "unexpected four-worker gate status: $STAGE4_RC" >&2; exit 3 ;;
  esac

  if test "$STAGE4_RC" -eq 10; then
    echo 'four-worker gate failed; finish 000..003, then MAX_PARALLEL=2' \
      | tee -a "$AUDIT_DIR/rollout.log"
    wait_shard 000 || exit $?
    wait_shard 001 || exit $?
    wait_shard 002 || exit $?
    wait_shard 003 || exit $?
    run_queue 2 004 005 006 007 || exit $?
  else
    date -u +%Y-%m-%dT%H:%M:%S+00:00 | tee "$AUDIT_DIR/stage-8worker-start.txt"
    STAGE8_START="$(cat "$AUDIT_DIR/stage-8worker-start.txt")"
    start_gpu_sampler stage-8worker
    launch_shard 004
    launch_shard 005
    launch_shard 006
    launch_shard 007
    if wait_stage_metrics stage-8worker "$STAGE8_START" \
        "$AUDIT_DIR/stage-4worker-metrics.json" \
        "$AUDIT_DIR/stage-8worker-metrics.json" 2 \
        000 001 002 003 004 005 006 007; then
      STAGE8_RC=0
    else
      STAGE8_RC=$?
    fi
    stop_gpu_sampler stage-8worker
    case "$STAGE8_RC" in
      0) ;;
      10) ;;
      2|3|4) exit "$STAGE8_RC" ;;
      *) echo "unexpected eight-worker monitor status: $STAGE8_RC" >&2; exit 3 ;;
    esac
    if test "$STAGE8_RC" -eq 10; then
      echo 'eight-worker resource/performance anomaly: launch no new campaign; diagnose; do not kill a group mid-write' \
        | tee -a "$AUDIT_DIR/rollout.log"
    fi
  fi
fi
```

The four-worker gate uses the last stable two-worker metrics as its baseline and
the same executable 1.5x/1.8x/0.8 assertions. The eight-worker measurement is
not a pre-launch gate: shards 004..007 already own formal plan directories and
must reach a terminal marker. On an eight-worker anomaly, do not start another
campaign or kill a group mid-write; retain diagnostics and wait for safe formal
completion. Whichever branch was taken, one condition loop must audit all eight
workers together and return only after every marker is completed.

```bash
wait_all_shards 000 001 002 003 004 005 006 007 || exit $?
```

## 8. Resume and failure policy

A clean `"started"` shard may be rerun with the same commit, config, plan, device, shard index, and shard root. The marker advances only at complete group
boundaries, so resume starts after the last completely recorded group. A
`"completed"` shard rerun is idempotent and returns its existing final JSON.
The lock still applies; first prove the old PID is gone.

```bash
set -Eeuo pipefail
export SHARD=003
export PID_FILE="$AUDIT_DIR/shard-$SHARD.pid"
if ! [[ -f "$PID_FILE" && -r "$PID_FILE" ]]; then
  echo "missing or unreadable shard PID file: $PID_FILE" >&2
  exit 2
fi
read -r OLD_SHARD_PID < "$PID_FILE"
if ! [[ "$OLD_SHARD_PID" =~ ^[0-9]+$ ]]; then
  echo "non-numeric shard PID: $OLD_SHARD_PID" >&2
  exit 2
fi
if kill -0 "$OLD_SHARD_PID" 2>/dev/null; then
  echo "recorded shard PID is still alive; refusing resume: $OLD_SHARD_PID" >&2
  exit 2
fi
if ps -p "$OLD_SHARD_PID" >/dev/null 2>&1; then
  echo "recorded PID still exists but cannot be signalled: $OLD_SHARD_PID" >&2
  exit 2
fi
echo "recorded shard PID is explicitly absent: $OLD_SHARD_PID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/root/miniconda3/envs/pinn_imu/bin/python -m validation_v2.cli shard \
  --config "$CONFIG" --plan "$PLAN" \
  --shard-index "$((10#$SHARD))" \
  --output-root "$SHARDS_ROOT/$SHARD" --device cuda \
  | tee -a "$AUDIT_DIR/shard-$SHARD.log"
```

Do not resume a `"failed"` marker or a root containing an uncommitted partial
group. Preserve it for diagnosis. Do not rename a replacement root to `NNN` or
mix it into the original plan: merge requires the plan's fixed directory names
`000` through `007`, and a replacement would falsify the formal campaign. The
safest recovery is a new formal campaign: increment the campaign suffix, create
a new `AUDIT_DIR`, `PLAN`, `SHARDS_ROOT`, and `FINAL_ROOT`, regenerate the plan,
and rerun all eight shards. Never reuse the failed/partial root.

## 9. Merge, strictly validate, and summarize five seeds

Merge only after all eight markers are `"completed"`. `FINAL_ROOT` must still
be absent. `merge-shards` publishes atomically to that fresh root and runs its
own strict checks; run the validator explicitly again as the formal handoff
gate.

```bash
test ! -e "$FINAL_ROOT"
python -m validation_v2.cli merge-shards \
  --config "$CONFIG" \
  --plan "$PLAN" \
  --shards-root "$SHARDS_ROOT" \
  --output-root "$FINAL_ROOT" \
  | tee "$AUDIT_DIR/merge.stdout.json"
python -m validation_v2.experiments.validate_artifacts \
  --root "$FINAL_ROOT" \
  --config "$CONFIG" \
  | tee "$AUDIT_DIR/validate.stdout.json"
python - "$FINAL_ROOT/validation_report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["status"] == "complete", report
PY
python -m validation_v2.cli summarize \
  --root "$FINAL_ROOT" \
  --config "$CONFIG" \
  --required-seeds 2026 2027 2028 2029 2030 \
  --baseline linear \
  | tee "$AUDIT_DIR/summarize.stdout.json"
```

The formal handoff contains:

- the recorded `COMMIT`, clean status, executed `CONFIG`, and immutable `PLAN`;
- shard logs and each isolated shard's `shard_execution.json`;
- final `matrix_execution.json` and `validation_report.json`;
- content-addressed split manifests and scalers;
- each run's `run.json`, `history.json`, `best.pt`, `checkpoint.json`,
  `test_evaluation.json`, and `per_record_metrics.csv`; and
- final `summary.csv` and `summary.json` covering all five required seeds.

Before packaging, require a clean repository and scan both tracked files and
audit logs for secret-shaped assignments. Any match must be reviewed and
removed before handoff. Do not put SSH credentials, tokens, repository URLs,
environment dumps, or network setup output in the repository or logs.

```bash
cd "$REPO"
test -z "$(git status --porcelain)"
if git grep -nEi '(password|passwd|token|secret)[[:space:]]*[:=]' -- .; then
  echo 'secret-shaped assignment found in tracked files' >&2
  exit 1
fi
if grep -RInE '(password|passwd|token|secret)[[:space:]]*[:=]' "$AUDIT_DIR"; then
  echo 'secret-shaped assignment found in audit logs' >&2
  exit 1
fi
```
