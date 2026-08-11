#!/usr/bin/env bash
# Shared, fail-closed server-rollout helpers. Source only after PYTHON_BIN exists.

if [[ -z "${PYTHON_BIN:-}" || ! -x "$PYTHON_BIN" ]]; then
  echo 'PYTHON_BIN must name an executable Python interpreter' >&2
  return 2 2>/dev/null || exit 2
fi

# This library has pipelines in its audit and sampler paths. Enabling pipefail
# makes a failed producer or logger visible to the caller instead of allowing a
# subsequent consumer to mask it.
set -o pipefail

validation_v2_require_shard() {
  local shard="${1-}"
  if ! [[ "$shard" =~ ^[0-7]{3}$ ]]; then
    echo "invalid zero-padded shard index: $shard" >&2
    return 2
  fi
}

validation_v2_require_label() {
  local label="${1-}"
  if ! [[ "$label" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
    echo "invalid GPU sampler label: $label" >&2
    return 2
  fi
}

audit_active() {
  local now last=0 stamp="$AUDIT_DIR/.last-60s-audit"
  local shard
  for shard in "$@"; do
    validation_v2_require_shard "$shard" || return $?
  done
  if ! command -v pgrep >/dev/null 2>&1; then
    echo 'pgrep is required for fail-closed shard auditing' >&2
    return 3
  fi
  now="$(date +%s)"
  test ! -f "$stamp" || last="$(cat "$stamp")"
  test "$((now - last))" -ge 60 || return 0
  printf '%s\n' "$now" > "$stamp"
  local process_listing="" pgrep_rc=0
  process_listing="$(pgrep -af 'validation_v2\.cli shard')" || pgrep_rc=$?
  if test "$pgrep_rc" -ne 0 -a "$pgrep_rc" -ne 1; then
    echo "cannot inspect shard processes: rc=$pgrep_rc" >&2
    return 3
  fi
  {
    date -Is
    local marker status groups pid alive
    for shard in "$@"; do
      marker="$SHARDS_ROOT/$shard/shard_execution.json"
      status=missing
      groups=0
      if test -f "$marker"; then
        read -r status groups < <("$PYTHON_BIN" -c \
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
    test -z "$process_listing" || printf '%s\n' "$process_listing"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
  } | tee -a "$AUDIT_DIR/monitor-60s.log"
}

launch_shard() {
  local shard="${1-}"
  validation_v2_require_shard "$shard" || return $?
  local index=$((10#$shard))
  if test "$index" -lt 0 -o "$index" -ge 8; then
    echo "shard index outside formal plan: $shard" >&2
    return 2
  fi
  local gpu_count="${VALIDATION_V2_GPU_COUNT:-1}"
  if ! [[ "$gpu_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid VALIDATION_V2_GPU_COUNT: $gpu_count" >&2
    return 2
  fi
  local gpu_index=$((index % gpu_count))
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
  local reservation="$AUDIT_DIR/shard-$shard.ownership.lock"
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
  if [[ -e "$reservation" || -L "$reservation" ]] \
      || ! mkdir "$reservation" 2>/dev/null; then
    echo "shard ownership is already reserved or linked: $shard" >&2
    return 2
  fi
  if ! command -v pgrep >/dev/null 2>&1; then
    echo 'pgrep is required for fail-closed shard launch' >&2
    rmdir "$reservation" 2>/dev/null || true
    return 3
  fi
  local process_listing="" pgrep_rc=0
  process_listing="$(pgrep -af "validation_v2\.cli shard.*--shard-index $index")" \
    || pgrep_rc=$?
  if test "$pgrep_rc" -ne 0 -a "$pgrep_rc" -ne 1; then
    echo "cannot inspect existing shard processes: rc=$pgrep_rc" >&2
    rmdir "$reservation" 2>/dev/null || true
    return 3
  fi
  if [[ -n "$process_listing" ]] && grep -F -- "$PLAN" <<< "$process_listing"; then
    echo "duplicate shard process: $shard" >&2
    rmdir "$reservation" 2>/dev/null || true
    return 2
  fi
  nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="$gpu_index" \
    "$PYTHON_BIN" -m validation_v2.cli shard \
    --config "$CONFIG" --plan "$PLAN" --shard-index "$index" \
    --output-root "$output_root" --device cuda \
    > "$AUDIT_DIR/shard-$shard.log" 2>&1 &
  local pid="$!"
  if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
    echo "nohup did not return a numeric PID: $pid" >&2
    rmdir "$reservation" 2>/dev/null || true
    return 3
  fi
  if ! printf '%s\n' "$pid" > "$pid_file"; then
    kill "$pid" 2>/dev/null || true
    rmdir "$reservation" 2>/dev/null || true
    return 3
  fi
}

wait_shard() {
  local shard="$1"
  validation_v2_require_shard "$shard" || return $?
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
      if STATE="$("$PYTHON_BIN" - "$marker" <<'PY'
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
  local shard
  if test "${#shards[@]}" -eq 0; then
    echo 'wait_all_shards requires at least one shard index' >&2
    return 2
  fi
  for shard in "${shards[@]}"; do
    validation_v2_require_shard "$shard" || return $?
  done
  if ! command -v pgrep >/dev/null 2>&1; then
    echo 'pgrep is required for fail-closed all-shard waiting' >&2
    return 3
  fi
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
        read -r status groups < <("$PYTHON_BIN" -c \
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
        completed) completed=$((completed + 1)) ;;
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
    local process_listing="" pgrep_rc=0
    process_listing="$(pgrep -af 'validation_v2\.cli shard')" || pgrep_rc=$?
    if test "$pgrep_rc" -ne 0 -a "$pgrep_rc" -ne 1; then
      echo "cannot inspect shard processes: rc=$pgrep_rc" >&2
      return 3
    fi
    if [[ -n "$process_listing" ]]; then
      tee -a "$AUDIT_DIR/wait-all-shards.log" <<< "$process_listing" || return 3
    fi
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
  local shard
  for shard in "${all_shards[@]}"; do
    validation_v2_require_shard "$shard" || return $?
  done
  while test "${#pending[@]}" -gt 0 -o "${#active[@]}" -gt 0; do
    while test "${#pending[@]}" -gt 0 -a "${#active[@]}" -lt "$max_parallel"; do
      shard="${pending[0]}"
      pending=("${pending[@]:1}")
      launch_shard "$shard" || return $?
      active+=("$shard")
    done
    local -a next=()
    for shard in "${active[@]}"; do
      local marker="$SHARDS_ROOT/$shard/shard_execution.json"
      if test -f "$marker"; then
        local state groups
        read -r state groups < <("$PYTHON_BIN" -c \
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
        read -r state groups < <("$PYTHON_BIN" -c \
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
    test "${#pending[@]}" -eq 0 -a "${#active[@]}" -eq 0 || sleep "$queue_poll_seconds"
  done
}

declare -Ag GPU_SAMPLER_JOBS=()

validation_v2_sampler_has_data() {
  local csv="$1"
  "$PYTHON_BIN" - "$csv" <<'PY'
import csv
import math
import sys

try:
    with open(sys.argv[1], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
except (OSError, UnicodeDecodeError, csv.Error):
    raise SystemExit(1)
for row in rows:
    try:
        values = [
            float(row[field])
            for field in (
                "memory_used_mib", "memory_total_mib", "utilization_percent"
            )
        ]
    except (KeyError, TypeError, ValueError):
        continue
    if all(math.isfinite(value) for value in values) and values[0] >= 0 and values[1] > 0:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

validation_v2_verify_sampler() {
  local label="${1-}"
  validation_v2_require_label "$label" || return $?
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
  if ! [[ "$pid" =~ ^[0-9]+$ && "$expected_start" =~ ^[0-9]+$ && -z "$extra" ]]; then
    echo "GPU sampler PID record must contain exactly two numeric fields" >&2
    return 3
  fi
  if test "${GPU_SAMPLER_JOBS[$label]-}" != "$pid"; then
    echo "GPU sampler is not the current shell job: label=$label pid=$pid" >&2
    return 3
  fi
  if test ! -r "$proc_root/$pid/stat" -o ! -r "$proc_root/$pid/cmdline"; then
    echo "GPU sampler process identity is unavailable; refusing use: $pid" >&2
    return 3
  fi
  local actual_start cmdline
  actual_start="$(awk '{print $22}' "$proc_root/$pid/stat")"
  cmdline="$(tr '\0' ' ' < "$proc_root/$pid/cmdline")"
  if test "$actual_start" != "$expected_start" \
      || [[ "$cmdline" != *"$identity"* ]] \
      || [[ "$cmdline" != *"nvidia-smi"* ]]; then
    echo "GPU sampler identity mismatch: label=$label pid=$pid" >&2
    return 3
  fi
}

validation_v2_stop_sampler_after_start_failure() {
  local pid="$1"
  local csv="$2"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  rm -f -- "$csv"
}

start_gpu_sampler() {
  local label="${1-}"
  validation_v2_require_label "$label" || return $?
  local csv="$AUDIT_DIR/gpu-$label.csv"
  local pid_file="$AUDIT_DIR/gpu-$label.pid"
  local identity="validation-v2-gpu-sampler-$label"
  local proc_root="${PROC_ROOT:-/proc}"
  local ready_max_seconds="${SAMPLER_READY_MAX_SECONDS:-30}"
  if ! [[ "$ready_max_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo 'invalid SAMPLER_READY_MAX_SECONDS' >&2
    return 2
  fi
  if [[ -e "$pid_file" || -L "$pid_file" ]]; then
    echo "GPU sampler PID file already exists: $pid_file" >&2
    return 2
  fi
  if [[ -e "$csv" || -L "$csv" ]]; then
    echo "GPU sampler CSV already exists or is linked: $csv" >&2
    return 2
  fi
  printf 'timestamp_utc,memory_used_mib,memory_total_mib,utilization_percent\n' > "$csv" || return 3
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
    validation_v2_stop_sampler_after_start_failure "$pid" "$csv"
    return 3
  fi
  local starttime cmdline
  starttime="$(awk '{print $22}' "$proc_root/$pid/stat")"
  cmdline="$(tr '\0' ' ' < "$proc_root/$pid/cmdline")"
  if ! [[ "$starttime" =~ ^[0-9]+$ \
      && "$cmdline" == *"$identity"* \
      && "$cmdline" == *"nvidia-smi"* ]]; then
    echo "GPU sampler identity verification failed immediately: $pid" >&2
    validation_v2_stop_sampler_after_start_failure "$pid" "$csv"
    return 3
  fi
  local ready_deadline=$((SECONDS + ready_max_seconds))
  while ! validation_v2_sampler_has_data "$csv"; do
    if test ! -r "$proc_root/$pid/stat" -o ! -r "$proc_root/$pid/cmdline"; then
      echo "GPU sampler exited before a valid data row: $pid" >&2
      validation_v2_stop_sampler_after_start_failure "$pid" "$csv"
      return 3
    fi
    if test "$SECONDS" -ge "$ready_deadline"; then
      echo "GPU sampler produced no valid data row within ${ready_max_seconds}s" >&2
      validation_v2_stop_sampler_after_start_failure "$pid" "$csv"
      return 3
    fi
    sleep 0.1
  done
  printf '%s %s\n' "$pid" "$starttime" > "$pid_file" || {
    validation_v2_stop_sampler_after_start_failure "$pid" "$csv"
    return 3
  }
}

stop_gpu_sampler() {
  local label="${1-}"
  validation_v2_verify_sampler "$label" || return $?
  local pid
  read -r pid _ < "$AUDIT_DIR/gpu-$label.pid"
  kill "$pid" || return 3
  wait "$pid" 2>/dev/null || true
  unset 'GPU_SAMPLER_JOBS[$label]'
}

wait_until_groups() {
  local shard="$1"
  local required="$2"
  validation_v2_require_shard "$shard" || return $?
  if ! [[ "$required" =~ ^[1-9][0-9]*$ ]]; then
    echo "required group count must be positive: $required" >&2
    return 2
  fi
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
      read -r status groups < <("$PYTHON_BIN" - "$marker" <<'PY'
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

wait_stage_metrics() {
  local label="$1"
  local stage_start="$2"
  local baseline="$3"
  local output="$4"
  local minimum_groups="$5"
  shift 5
  local -a active_indices=("$@")
  validation_v2_require_label "$label" || return $?
  if ! [[ "$minimum_groups" =~ ^[1-9][0-9]*$ ]]; then
    echo "minimum group count must be positive: $minimum_groups" >&2
    return 2
  fi
  if test "${#active_indices[@]}" -eq 0; then
    echo 'wait_stage_metrics requires at least one active shard index' >&2
    return 2
  fi
  local shard
  for shard in "${active_indices[@]}"; do
    validation_v2_require_shard "$shard" || return $?
  done
  local deadline=$((SECONDS + 14400))
  while :; do
    audit_active "${active_indices[@]}" || return $?
    validation_v2_verify_sampler "$label" || return $?
    local all_completed=yes
    local marker pid status groups
    for shard in "${active_indices[@]}"; do
      marker="$SHARDS_ROOT/$shard/shard_execution.json"
      pid=""
      test ! -f "$AUDIT_DIR/shard-$shard.pid" || pid="$(cat "$AUDIT_DIR/shard-$shard.pid")"
      if test -f "$marker"; then
        read -r status groups < <("$PYTHON_BIN" -c \
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
      "$PYTHON_BIN" "$REPO/scripts/collect_validation_v2_stage_metrics.py"
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
