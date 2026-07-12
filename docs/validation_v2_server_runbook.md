# Validation v2 sharded server runbook

This is the operator contract for the paper-grade `server_full` campaign. Run
it on Linux from the validated revision. The formal campaign consists of one
immutable eight-shard plan, eight isolated shard roots, and one newly published
final root. A smoke run, a partial run, or artifacts from another commit are
diagnostic only.

## 1. Quarantine the legacy single-process run

The old `fcf81f8` single-process `matrix` root and the new commit's shard roots
must not be mixed, copied together, used as merge inputs, or summarized as one
campaign. Never delete the old root: it is immutable diagnostic evidence. Give
the new campaign a different, commit-qualified root.

Stopping the old process is optional. If it is still consuming the GPU, first
confirm that exactly one PID is both the legacy matrix command and the
`fcf81f8` root, then send `SIGINT` and wait for that PID to disappear. Do not run
this block when the match is absent or ambiguous.

```bash
export OLD_ROOT="/root/autodl-tmp/2025.12.28LNN_Imputation/results/validation_v2/server_full-fcf81f8"
test -d "$OLD_ROOT"
# Legacy command signature: python -m validation_v2.cli matrix
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mapfile -t OLD_MATCHES < <(pgrep -af '[p]ython -m validation_v2\.cli matrix' | grep 'fcf81f8')
printf '%s\n' "${OLD_MATCHES[@]}"
test "${#OLD_MATCHES[@]}" -eq 1
export OLD_PID="${OLD_MATCHES[0]%% *}"
ps -p "$OLD_PID" -o pid=,lstart=,cmd=
kill -INT "$OLD_PID"
timeout 300 tail --pid="$OLD_PID" -f /dev/null
! kill -0 "$OLD_PID" 2>/dev/null
test -d "$OLD_ROOT"
```

`timeout` makes failure to stop explicit. Escalate manually after inspection;
do not use `kill -9`, remove the old root, or point a shard at it.

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
  git checkout validation-v2-server
  git pull --ff-only origin validation-v2-server
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
  local shard="$1"
  local index=$((10#$shard))
  test "$shard" = "$(printf '%03d' "$index")"
  test ! -e "$SHARDS_ROOT/$shard"
  if pgrep -af "validation_v2\.cli shard.*--shard-index $index" \
      | grep -F -- "$PLAN"; then
    echo "duplicate shard process: $shard" >&2
    return 2
  fi
  nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    /root/miniconda3/envs/pinn_imu/bin/python -m validation_v2.cli shard \
    --config "$CONFIG" --plan "$PLAN" --shard-index "$index" \
    --output-root "$SHARDS_ROOT/$shard" --device cuda \
    > "$AUDIT_DIR/shard-$shard.log" 2>&1 &
  echo "$!" > "$AUDIT_DIR/shard-$shard.pid"
}

wait_shard() {
  local shard="$1"
  local marker="$SHARDS_ROOT/$shard/shard_execution.json"
  local deadline=$((SECONDS + 14400))
  while :; do
    audit_active "$shard" || return $?
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
    if test "$SECONDS" -ge "$deadline"; then
      echo "wait_shard timeout after 14400 seconds: $shard" >&2
      tail -n 50 "$AUDIT_DIR/shard-$shard.log" >&2 || true
      return 4
    fi
    sleep 10
  done
}

run_queue() {
  local max_parallel="$1"
  shift
  local -a pending=("$@")
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
        local state
        state="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$marker")"
        if test "$state" = failed; then
          cat "$marker" >&2
          return 2
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
    if test "${#active[@]}" -gt 0; then
      audit_active "${active[@]}" || return $?
    fi
    printf '%s max=%s pending=%s active=%s\n' \
      "$(date -Is)" "$max_parallel" "${pending[*]}" "${active[*]}" \
      | tee -a "$AUDIT_DIR/queue.log"
    test "${#pending[@]}" -eq 0 -a "${#active[@]}" -eq 0 || sleep 10
  done
}

start_gpu_sampler() {
  local label="$1"
  local csv="$AUDIT_DIR/gpu-$label.csv"
  printf 'timestamp_utc,memory_used_mib,memory_total_mib,utilization_percent\n' > "$csv"
  (
    while :; do
      local timestamp
      timestamp="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
      nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits \
        | awk -F, -v timestamp="$timestamp" \
          '{gsub(/ /, ""); print timestamp "," $1 "," $2 "," $3}' \
        | tee -a "$csv"
      sleep 10
    done
  ) &
  echo "$!" > "$AUDIT_DIR/gpu-$label.pid"
}

stop_gpu_sampler() {
  local label="$1"
  local pid
  pid="$(cat "$AUDIT_DIR/gpu-$label.pid")"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
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
the stage window. A not-yet-large-enough window exits 4; a failed marker exits
2; a performance/resource assertion exits 10 after writing the metrics JSON.
Only exit 10 is eligible for a lower-concurrency fallback.

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
are all retained. The gate waits for at least two new ledger completions after
the stage start and actually asserts throughput >= 1.5x the one-worker
baseline, median group time < 1.8x baseline, peak memory < 80%, and no failed
marker. In percentage terms, throughput improvement must be >= 50%, median
group-time growth must be < 80%, and GPU memory must remain < 80%.

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
completion. Whichever branch was taken, the following condition-based waits
must finish, then the explicit all-eight assertion must pass before merge.

```bash
for SHARD in 000 001 002 003 004 005 006 007; do
  wait_shard "$SHARD" || exit $?
done
python - "$SHARDS_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
completed = 0
for index in range(8):
    marker_path = root / f"{index:03d}" / "shard_execution.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["status"] == "completed", (marker_path, marker)
    completed += 1
assert completed == 8, completed
print("all eight shard markers completed")
PY
```

## 8. Resume and failure policy

A clean `"started"` shard may be rerun with the same commit, config, plan, device, shard index, and shard root. The marker advances only at complete group
boundaries, so resume starts after the last completely recorded group. A
`"completed"` shard rerun is idempotent and returns its existing final JSON.
The lock still applies; first prove the old PID is gone.

```bash
export SHARD=003
! kill -0 "$(cat "$AUDIT_DIR/shard-$SHARD.pid")" 2>/dev/null
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
