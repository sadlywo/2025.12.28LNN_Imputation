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

## 3. Activate and verify the pinned Linux/GPU environment

Dependency installation is the final permitted network operation. Skip it when
the pinned environment is already complete.

```bash
(
  source /etc/network_turbo
  /root/miniconda3/envs/pinn_imu/bin/python -m pip install -r \
    /root/autodl-tmp/2025.12.28LNN_Imputation/requirements-validation-v2.txt
)
```

Open a new offline shell, activate the exact environment, and pin deterministic
CUDA workspace behavior. Stop on a Python, Torch, CUDA, GPU, or disk failure.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/miniconda3/envs/pinn_imu
export REPO="/root/autodl-tmp/2025.12.28LNN_Imputation"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$REPO"
python - <<'PY'
import os
import platform
import torch

assert platform.system() == "Linux", platform.system()
assert os.environ["CONDA_PREFIX"] == "/root/miniconda3/envs/pinn_imu"
assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
assert torch.cuda.is_available()
assert torch.version.cuda is not None
name = torch.cuda.get_device_name(0)
assert "4090 D" in name, name
print(platform.python_version(), torch.__version__, torch.version.cuda, name)
PY
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv
df -h "$REPO" /root/autodl-tmp
df -Pk /root/autodl-tmp | awk 'NR==2 { if ($4 < 104857600) exit 1 }'
```

Run the real Linux no-replace race separately and require `1 passed`; a
`skipped` result is a hard stop. Then run the complete suite. `PIPESTATUS`
preserves pytest's status through `tee`.

```bash
export PREFLIGHT_DIR="/root/autodl-tmp/validation-v2-preflight-${COMMIT}"
test ! -e "$PREFLIGHT_DIR"
mkdir -p "$PREFLIGHT_DIR"
test "$(uname -s)" = Linux
set -o pipefail
python -m pytest -q \
  tests/validation_v2/test_sharding.py::test_linux_rename_noreplace_survives_real_directory_race \
  -rs | tee "$PREFLIGHT_DIR/linux-renameat2.txt"
test "${PIPESTATUS[0]}" -eq 0
grep -F '1 passed' "$PREFLIGHT_DIR/linux-renameat2.txt"
! grep -qi 'skipped' "$PREFLIGHT_DIR/linux-renameat2.txt"
python -m pytest -q | tee "$PREFLIGHT_DIR/pytest-full.txt"
test "${PIPESTATUS[0]}" -eq 0
```

## 4. Create one fresh, commit-qualified campaign

Set these exact variables once. `CONFIG` is absolute. Both roots are fresh and
contain the full commit. If `sharded-v1` was ever used with this commit, choose
`sharded-v2`; never reuse an old commit, plan, or root.

```bash
export REPO="/root/autodl-tmp/2025.12.28LNN_Imputation"
export CONFIG="/root/autodl-tmp/2025.12.28LNN_Imputation/configs/validation_v2/server_full.yaml"
export COMMIT="$(git -C "$REPO" rev-parse HEAD)"
export CAMPAIGN="${COMMIT}-sharded-v1"
export AUDIT_DIR="/root/autodl-tmp/validation-v2-audit-${CAMPAIGN}"
export PLAN="$AUDIT_DIR/server-full-8-shards-${COMMIT}.json"
export SHARDS_ROOT="$REPO/results/validation_v2/server-full-shards-${CAMPAIGN}"
export FINAL_ROOT="$REPO/results/validation_v2/server-full-final-${CAMPAIGN}"
export SHARD_LOG_PATTERN="$AUDIT_DIR/shard-NNN.log"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$REPO"
test -f "$CONFIG"
test -z "$(git status --porcelain)"
test ! -e "$AUDIT_DIR"
test ! -e "$SHARDS_ROOT"
test ! -e "$FINAL_ROOT"
mkdir -p "$AUDIT_DIR" "$SHARDS_ROOT"
printf '%s\n' "$COMMIT" > "$AUDIT_DIR/COMMIT"
git status --porcelain > "$AUDIT_DIR/git-status-porcelain.txt"
test ! -s "$AUDIT_DIR/git-status-porcelain.txt"
cp "$PREFLIGHT_DIR/linux-renameat2.txt" "$PREFLIGHT_DIR/pytest-full.txt" "$AUDIT_DIR/"
```

Freeze both the ordinary matrix dry-run and the eight-shard plan. The dry-run
has one header plus 4,095 cells. The plan must say 175 training groups, 4,095
cells, and 8 shards.

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
  --config "$CONFIG" \
  --shard-count 8 \
  --output "$PLAN" \
  --device cuda | tee "$AUDIT_DIR/shard-plan.stdout.json"
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

## 5. Launch the two-worker pilot

Each shard has an independent `$SHARDS_ROOT/NNN` root and
`$AUDIT_DIR/shard-NNN.log`. The `.shard_execution.lock` prevents concurrent
reuse of one root, but the operator must still use `pgrep` and inspect the root
before launch. A shard command runs its full assigned group list; it is not a
one-group command. Training may be quiet. The shard's stdout ends with its final
JSON report, so redirect both streams to the shard log and do not parse an
incomplete log as a result.

Use this single-shard command after setting `SHARD` to a zero-padded index:

```bash
export SHARD=000
test "$SHARD" = "$(printf '%03d' "$((10#$SHARD))")"
test ! -e "$SHARDS_ROOT/$SHARD"
! pgrep -af "validation_v2\.cli shard.*--shard-index $((10#$SHARD))" \
  | grep -F -- "$PLAN"
nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /root/miniconda3/envs/pinn_imu/bin/python -m validation_v2.cli shard \
  --config "$CONFIG" \
  --plan "$PLAN" \
  --shard-index "$((10#$SHARD))" \
  --output-root "$SHARDS_ROOT/$SHARD" \
  --device cuda \
  > "$AUDIT_DIR/shard-$SHARD.log" 2>&1 &
echo "$!" > "$AUDIT_DIR/shard-$SHARD.pid"
```

First launch shard 000 and shard 001 as two workers:

```bash
for SHARD in 000 001; do
  test ! -e "$SHARDS_ROOT/$SHARD"
  ! pgrep -af "validation_v2\.cli shard.*--shard-index $((10#$SHARD))" \
    | grep -F -- "$PLAN"
  nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    /root/miniconda3/envs/pinn_imu/bin/python -m validation_v2.cli shard \
    --config "$CONFIG" --plan "$PLAN" \
    --shard-index "$((10#$SHARD))" \
    --output-root "$SHARDS_ROOT/$SHARD" --device cuda \
    > "$AUDIT_DIR/shard-$SHARD.log" 2>&1 &
  echo "$!" > "$AUDIT_DIR/shard-$SHARD.pid"
done
pgrep -af 'validation_v2\.cli shard'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

Watch both markers until each has completed at least one group. The marker's
`completed_group_ids` is the authoritative group count; a quiet log is not a
hang.

```bash
while :; do
  export READY=0
  date -Is
  pgrep -af 'validation_v2\.cli shard' || true
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
  for SHARD in 000 001; do
    export MARKER="$SHARDS_ROOT/$SHARD/shard_execution.json"
    if test -f "$MARKER"; then
      python - "$MARKER" <<'PY'
import json
import sys

marker = json.load(open(sys.argv[1], encoding="utf-8"))
print(marker["status"], len(marker["completed_group_ids"]))
if marker["status"] == "failed":
    print(json.dumps(marker, sort_keys=True), file=sys.stderr)
    raise SystemExit(2)
PY
      test "$?" -ne 2 || exit 2
      GROUPS="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["completed_group_ids"]))' "$MARKER")"
      test "$GROUPS" -lt 1 || READY=$((READY + 1))
    else
      echo "$SHARD waiting-for-marker"
    fi
  done
  test "$READY" -eq 2 && break
  sleep 60
done
```

Before the pilot, put a same-commit/config/device one-worker reference in the
audit record as `ONE_WORKER_GROUPS_PER_HOUR` and
`ONE_WORKER_MEDIAN_GROUP_SECONDS`. If no trustworthy reference exists, retain
two workers and do not scale. After each pilot shard completes at least one
group, record aggregate completed groups/hour, median seconds per completed
group, peak GPU memory, and any failed marker. Scale only if all are true:

- aggregate throughput improvement is >= 50% over the one-worker reference;
- single-group median elapsed-time growth is < 80% over the reference;
- GPU memory remains < 80% of total; and
- neither marker nor log reports `failed`, CUDA OOM, or another error.

If any gate fails, keep the lower worker count. Do not change seeds, epochs,
config, plan, device, or batch size inside this campaign.

This snapshot reports the required aggregate completed groups/hour and the
median elapsed seconds per completed group from marker time. Keep the periodic
samples above in the audit log so the first transition for each group is
available rather than inferred later.

```bash
python - "$SHARDS_ROOT" <<'PY'
from datetime import datetime
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
counts = []
seconds_per_group = []
starts = []
finishes = []
for shard in ("000", "001"):
    marker_path = root / shard / "shard_execution.json"
    marker = json.loads(marker_path.read_text())
    assert marker["status"] != "failed", marker
    count = len(marker["completed_group_ids"])
    assert count >= 1, marker
    started = datetime.fromisoformat(marker["started_at"]).timestamp()
    finished = marker_path.stat().st_mtime
    starts.append(started)
    finishes.append(finished)
    counts.append(count)
    seconds_per_group.append((finished - started) / count)
hours = (max(finishes) - min(starts)) / 3600
print("aggregate completed groups/hour", sum(counts) / hours)
print("single-group median elapsed seconds", statistics.median(seconds_per_group))
PY
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits \
  | awk -F, '{ used=$1+0; total=$2+0; print "GPU memory", used "/" total; if (used/total >= 0.80) exit 1 }'
```

## 6. Scale in stages: 2 -> 4 -> 8

Only after the two-worker gate passes, launch `002 003`. Re-measure the same
throughput, per-group median, GPU memory, and failure gates. Only if the
four-worker gate passes, launch `004 005 006 007`. Otherwise keep the lower
worker count and let its commands finish their full assigned groups.

```bash
# Run once with STAGE="002 003", then only after approval with
# STAGE="004 005 006 007".
export STAGE="002 003"
for SHARD in $STAGE; do
  test ! -e "$SHARDS_ROOT/$SHARD"
  ! pgrep -af "validation_v2\.cli shard.*--shard-index $((10#$SHARD))" \
    | grep -F -- "$PLAN"
  nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    /root/miniconda3/envs/pinn_imu/bin/python -m validation_v2.cli shard \
    --config "$CONFIG" --plan "$PLAN" \
    --shard-index "$((10#$SHARD))" \
    --output-root "$SHARDS_ROOT/$SHARD" --device cuda \
    > "$AUDIT_DIR/shard-$SHARD.log" 2>&1 &
  echo "$!" > "$AUDIT_DIR/shard-$SHARD.pid"
done
```

## 7. Wait by marker state, not by elapsed time

After all eight shards have been launched, use this condition-based loop. It
reads every `shard_execution.json`: `"completed"` counts as done,
`"started"` or a missing marker continues waiting, and `"failed"` prints the
marker and exits immediately. Every 60 seconds it prints marker progress,
processes, and GPU state, so this is not an infinite blind sleep.

```bash
while :; do
  if python - "$SHARDS_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
completed = 0
progress = []
for index in range(8):
    marker_path = root / f"{index:03d}" / "shard_execution.json"
    if not marker_path.is_file():
        progress.append((f"{index:03d}", "waiting", 0))
        continue
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    status = marker.get("status")
    groups = len(marker.get("completed_group_ids", []))
    progress.append((f"{index:03d}", status, groups))
    if status == "failed":
        print(json.dumps(marker, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
    if status == "completed":
        completed += 1
    elif status != "started":
        print(f"invalid status in {marker_path}: {status!r}", file=sys.stderr)
        raise SystemExit(3)
print(f"completed_shards={completed}/8 progress={progress}")
raise SystemExit(0 if completed == 8 else 1)
PY
  then
    STATUS=0
  else
    STATUS=$?
  fi
  test "$STATUS" -ne 2 -a "$STATUS" -ne 3 || exit "$STATUS"
  test "$STATUS" -eq 0 && break
  date -Is
  pgrep -af 'validation_v2\.cli shard' || true
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
  sleep 60
done
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
