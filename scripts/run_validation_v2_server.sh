#!/usr/bin/env bash
# Execute the immutable Validation v2 campaign on a supported Linux RTX 4090 host.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_validation_v2_server.sh --commit COMMIT --mode preflight|full [options]

Required:
  --commit COMMIT             Exact 40-character lowercase Git commit SHA.
  --mode preflight|full       Create and verify a campaign, or also train and merge it.

Options:
  --repo PATH                 Repository root (default: directory containing this script/..).
  --campaign-suffix NAME      New campaign suffix (default: sharded-v2-py310/py311/py312).
  --skip-dependency-install   Reuse an already-provisioned .venv-server, but still verify it.
  --help                      Show this help and exit.
EOF
}

die() {
  printf '%s\n' "$*" >&2
  exit 2
}

ensure_shard_output_parent() {
  local directory="$1"
  if [[ -L "$directory" ]]; then
    die "shard output parent is linked: $directory"
  fi
  if [[ -e "$directory" ]]; then
    [[ -d "$directory" ]] || die "shard output parent is not a directory: $directory"
  else
    mkdir "$directory" || die "cannot create shard output parent: $directory"
  fi
  [[ -d "$directory" && ! -L "$directory" ]] \
    || die "shard output parent is not a real directory: $directory"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd -P)"
COMMIT=""
MODE=""
CAMPAIGN_SUFFIX=""
CAMPAIGN_SUFFIX_EXPLICIT=0
SKIP_DEPENDENCY_INSTALL=0

while (($#)); do
  case "$1" in
    --help)
      usage
      exit 0
      ;;
    --commit)
      (($# >= 2)) || die '--commit requires COMMIT'
      COMMIT="$2"
      shift 2
      ;;
    --mode)
      (($# >= 2)) || die '--mode requires preflight or full'
      MODE="$2"
      shift 2
      ;;
    --repo)
      (($# >= 2)) || die '--repo requires PATH'
      REPO="$2"
      shift 2
      ;;
    --campaign-suffix)
      (($# >= 2)) || die '--campaign-suffix requires NAME'
      CAMPAIGN_SUFFIX="$2"
      CAMPAIGN_SUFFIX_EXPLICIT=1
      shift 2
      ;;
    --skip-dependency-install)
      SKIP_DEPENDENCY_INSTALL=1
      shift
      ;;
    *)
      usage >&2
      die "unknown option: $1"
      ;;
  esac
done

[[ "$MODE" == preflight || "$MODE" == full ]] || die '--mode must be preflight or full'
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || die '--commit must be a 40-character lowercase hexadecimal SHA'
[[ -d "$REPO" ]] || die "repository does not exist: $REPO"
REPO="$(cd "$REPO" && pwd -P)"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "not a Git worktree: $REPO"

# These must remain the first Python actions: failures must precede the audit
# seal, venv creation, package installation, and all other campaign writes.
PYTHON3_BIN="${PYTHON3_BIN:-python3}"
PYTHON3_RUNTIME="$("$PYTHON3_BIN" -c \
  "import platform, sys; print(platform.python_implementation(), '{}.{}'.format(*sys.version_info[:2]))" 2>&1)" \
  || die "cannot run PYTHON3_BIN: $PYTHON3_BIN"
[[ "$PYTHON3_RUNTIME" =~ ^CPython[[:space:]]3\.(10|11|12)$ ]] \
  || die "CPython 3.10, 3.11, or 3.12 is required; found: $PYTHON3_RUNTIME"
PYTHON_MINOR="${BASH_REMATCH[1]}"
if ! "$PYTHON3_BIN" -m venv --help >/dev/null 2>&1; then
  die "Python venv support is unavailable; install the matching python3.${PYTHON_MINOR}-venv package"
fi
if (( ! CAMPAIGN_SUFFIX_EXPLICIT )); then
  CAMPAIGN_SUFFIX="sharded-v2-py3${PYTHON_MINOR}"
fi
[[ "$CAMPAIGN_SUFFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die 'invalid --campaign-suffix'

HEAD_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
[[ "$HEAD_COMMIT" == "$COMMIT" ]] || die "HEAD does not match --commit: $HEAD_COMMIT != $COMMIT"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || die 'Git worktree must be clean'

PARENT_DIR="$(dirname "$REPO")"
PREFLIGHT_DIR="$PARENT_DIR/validation-v2-preflight-${COMMIT}-${CAMPAIGN_SUFFIX}"
AUDIT_DIR="$PARENT_DIR/validation-v2-audit-${COMMIT}-${CAMPAIGN_SUFFIX}"
PLAN="$AUDIT_DIR/server-full-8-shards-${COMMIT}.json"
SHARDS_ROOT="$REPO/results/validation_v2/server-full-shards-${COMMIT}-${CAMPAIGN_SUFFIX}"
FINAL_ROOT="$REPO/results/validation_v2/server-full-final-${COMMIT}-${CAMPAIGN_SUFFIX}"
if [[ -e "$AUDIT_DIR" || -L "$AUDIT_DIR" ]] \
    || ! mkdir "$AUDIT_DIR" 2>/dev/null; then
  die "AUDIT_DIR campaign seal already exists or is linked: $AUDIT_DIR"
fi

GPU_SAMPLER_LABELS=()
SHARDS_LAUNCHED=0
cleanup_runner() {
  local status="${1:-$?}"
  trap - EXIT INT TERM
  set +e
  local label
  for label in "${GPU_SAMPLER_LABELS[@]}"; do
    if declare -F stop_gpu_sampler >/dev/null 2>&1; then
      stop_gpu_sampler "$label" || true
    fi
  done
  if [[ -d "$AUDIT_DIR" && ! -L "$AUDIT_DIR" ]]; then
    {
      printf 'runner exit status=%s shards_launched=%s\n' "$status" "$SHARDS_LAUNCHED"
      if (( SHARDS_LAUNCHED )); then
        printf '%s\n' 'shard processes and the campaign seal were preserved for diagnosis; automatic re-entry is refused.'
      else
        printf '%s\n' 'campaign seal was preserved; use a new suffix for any retry.'
      fi
    } >> "$AUDIT_DIR/runner-exit-note.txt" || true
  fi
  exit "$status"
}
trap cleanup_runner EXIT
trap 'cleanup_runner 130' INT
trap 'cleanup_runner 143' TERM
[[ "$(uname -s)" == Linux ]] || die 'this formal runner requires Linux'

PYTHON_BIN="$REPO/.venv-server/bin/python"
if (( SKIP_DEPENDENCY_INSTALL )) && [[ ! -x "$PYTHON_BIN" ]]; then
  die '--skip-dependency-install requires an existing .venv-server Python interpreter'
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  "$PYTHON3_BIN" -m venv "$REPO/.venv-server"
fi
[[ -x "$PYTHON_BIN" ]] || die "venv Python is missing: $PYTHON_BIN"

install_dependencies() {
  (
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1
    "$PYTHON_BIN" -m pip install -r "$REPO/requirements-validation-v2.txt"
  )
}

if (( ! SKIP_DEPENDENCY_INSTALL )); then
  install_dependencies
fi

CONFIG="$REPO/configs/validation_v2/server_full.yaml"
export REPO COMMIT PREFLIGHT_DIR AUDIT_DIR PLAN SHARDS_ROOT FINAL_ROOT CONFIG PYTHON_BIN
export CUBLAS_WORKSPACE_CONFIG=:4096:8

[[ -f "$CONFIG" ]] || die "missing formal configuration: $CONFIG"
ensure_shard_output_parent "$REPO/results"
ensure_shard_output_parent "$REPO/results/validation_v2"
for campaign_path in "$PREFLIGHT_DIR" "$SHARDS_ROOT" "$FINAL_ROOT"; do
  [[ ! -e "$campaign_path" && ! -L "$campaign_path" ]] \
    || die "campaign path already exists or is linked: $campaign_path"
done
mkdir "$PREFLIGHT_DIR"
mkdir "$SHARDS_ROOT"
printf '%s\n' "$COMMIT" > "$AUDIT_DIR/COMMIT"

verify_runtime() {
  "$PYTHON_BIN" - "$AUDIT_DIR/environment.json" <<'PY'
import json
import os
import platform
import shutil
import subprocess
import sys

import torch

def require(condition, message):
    if not condition:
        raise SystemExit(message)

require(platform.python_implementation() == "CPython", platform.python_implementation())
require(sys.version_info[:2] in ((3, 10), (3, 11), (3, 12)), sys.version)
require(torch.__version__ == "2.3.1+cu121", torch.__version__)
require(torch.cuda.is_available(), "CUDA is unavailable")
require(torch.version.cuda == "12.1", torch.version.cuda)
name = torch.cuda.get_device_name(0)
require("4090" in name.upper(), name)
gpu_memory_bytes = int(torch.cuda.get_device_properties(0).total_memory)
require(
    gpu_memory_bytes >= 23 * 1024**3,
    "GPU memory is below 23 GiB: {}".format(gpu_memory_bytes),
)
compute_capability = torch.cuda.get_device_capability(0)
require(
    isinstance(compute_capability, (tuple, list)) and len(compute_capability) == 2,
    "invalid compute capability: {!r}".format(compute_capability),
)
try:
    nvidia_smi = shutil.which("nvidia-smi")
    require(nvidia_smi is not None, "nvidia-smi is unavailable")
    driver_query = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
except OSError as error:
    raise SystemExit("cannot query NVIDIA driver version: {}".format(error))
require(
    driver_query.returncode == 0,
    "nvidia-smi driver query failed: {}".format(driver_query.stderr.strip()),
)
driver_lines = [line.strip() for line in driver_query.stdout.splitlines() if line.strip()]
require(len(driver_lines) == 1, "nvidia-smi returned an invalid driver version")
driver_version = driver_lines[0]
if hasattr(os, "sched_getaffinity"):
    cpu_affinity_count = len(os.sched_getaffinity(0))
else:
    cpu_affinity_count = os.cpu_count()
require(
    isinstance(cpu_affinity_count, int) and cpu_affinity_count > 0,
    "cannot determine CPU affinity count",
)
try:
    host_memory_bytes = int(os.sysconf("SC_PAGE_SIZE")) * int(
        os.sysconf("SC_PHYS_PAGES")
    )
except (AttributeError, OSError, ValueError) as error:
    raise SystemExit("cannot determine host memory: {}".format(error))
require(host_memory_bytes > 0, "host memory must be positive")
with open(sys.argv[1], "x", encoding="utf-8") as handle:
    json.dump(
        {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_0": name,
            "gpu_memory_bytes": gpu_memory_bytes,
            "compute_capability": compute_capability,
            "driver_version": driver_version,
            "cpu_affinity_count": cpu_affinity_count,
            "host_memory_bytes": host_memory_bytes,
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY
}

run_preflight() {
  verify_runtime
  "$PYTHON_BIN" -m pytest -q \
    tests/validation_v2/test_sharding.py::test_linux_rename_noreplace_survives_real_directory_race \
    -rs | tee "$PREFLIGHT_DIR/linux-renameat2.txt"
  grep -F '1 passed' "$PREFLIGHT_DIR/linux-renameat2.txt"
  ! grep -qi 'skipped' "$PREFLIGHT_DIR/linux-renameat2.txt"

  "$PYTHON_BIN" -m pytest -q | tee "$PREFLIGHT_DIR/pytest-full.txt"
  cp "$PREFLIGHT_DIR/linux-renameat2.txt" "$PREFLIGHT_DIR/pytest-full.txt" "$AUDIT_DIR/"

  "$PYTHON_BIN" -m validation_v2.cli matrix --config "$CONFIG" --dry-run \
    > "$AUDIT_DIR/matrix-dry-run.jsonl"
  "$PYTHON_BIN" - "$AUDIT_DIR/matrix-dry-run.jsonl" <<'PY'
import json
import sys

lines = open(sys.argv[1], encoding="utf-8").read().splitlines()
if len(lines) != 4096:
    raise SystemExit("matrix dry-run must contain 4096 JSONL records: {}".format(len(lines)))
if json.loads(lines[0]).get("combination_count") != 4095:
    raise SystemExit("matrix dry-run must report 4095 combinations")
PY

  "$PYTHON_BIN" -m validation_v2.cli shard-plan \
    --config "$CONFIG" --shard-count 8 --output "$PLAN" --device cuda \
    | tee "$AUDIT_DIR/shard-plan.stdout.json"
  "$PYTHON_BIN" - "$PLAN" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
if plan.get("schema_version") != 2:
    raise SystemExit("unexpected shard plan schema")
if plan.get("total_groups") != 175 or plan.get("total_cells") != 4095:
    raise SystemExit("unexpected formal shard plan totals")
if plan.get("shard_count") != 8 or plan.get("dirty_state_digest") != "":
    raise SystemExit("shard plan is not a clean formal eight-shard plan")
try:
    group_counts = [len(shard["group_ids"]) for shard in plan["shards"]]
except (KeyError, TypeError):
    raise SystemExit("shard plan has invalid shard group identifiers")
if group_counts != [22, 22, 22, 22, 22, 22, 22, 21]:
    raise SystemExit("unexpected formal shard distribution")
PY
}

source "$REPO/scripts/lib/validation_v2_server_helpers.sh"

start_managed_sampler() {
  local label="$1"
  start_gpu_sampler "$label" || return $?
  GPU_SAMPLER_LABELS+=("$label")
}

stop_managed_sampler() {
  local label="$1"
  stop_gpu_sampler "$label" || return $?
  local -a remaining=()
  local active_label
  for active_label in "${GPU_SAMPLER_LABELS[@]}"; do
    if [[ "$active_label" != "$label" ]]; then
      remaining+=("$active_label")
    fi
  done
  GPU_SAMPLER_LABELS=("${remaining[@]}")
}

launch_formal_shard() {
  launch_shard "$1" || return $?
  SHARDS_LAUNCHED=1
}

run_formal_campaign() {
  local requested_mode="$1"
  [[ "$requested_mode" == full ]] || return 0

  start_managed_sampler baseline-1worker
  launch_formal_shard 000
  wait_until_groups 000 2
  local baseline_start
  baseline_start="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["started_at"])' "$SHARDS_ROOT/000/shard_execution.json")"
  wait_stage_metrics baseline-1worker "$baseline_start" "" \
    "$AUDIT_DIR/baseline-1worker.json" 2 000
  stop_managed_sampler baseline-1worker

  date -u +%Y-%m-%dT%H:%M:%S+00:00 | tee "$AUDIT_DIR/stage-2worker-start.txt"
  local stage2_start stage2_rc
  stage2_start="$(cat "$AUDIT_DIR/stage-2worker-start.txt")"
  start_managed_sampler stage-2worker
  launch_formal_shard 001
  if wait_stage_metrics stage-2worker "$stage2_start" \
      "$AUDIT_DIR/baseline-1worker.json" "$AUDIT_DIR/stage-2worker-metrics.json" 2 000 001; then
    stage2_rc=0
  else
    stage2_rc=$?
  fi
  stop_managed_sampler stage-2worker
  case "$stage2_rc" in
    0) ;;
    10) echo 'two-worker performance/resource gate failed' | tee -a "$AUDIT_DIR/rollout.log" ;;
    2|3|4) return "$stage2_rc" ;;
    *) echo "unexpected two-worker gate status: $stage2_rc" >&2; return 3 ;;
  esac
  if (( stage2_rc == 10 )); then
    wait_shard 000
    wait_shard 001
    run_queue 1 002 003 004 005 006 007
  else
    date -u +%Y-%m-%dT%H:%M:%S+00:00 | tee "$AUDIT_DIR/stage-4worker-start.txt"
    local stage4_start stage4_rc
    stage4_start="$(cat "$AUDIT_DIR/stage-4worker-start.txt")"
    start_managed_sampler stage-4worker
    launch_formal_shard 002
    launch_formal_shard 003
    if wait_stage_metrics stage-4worker "$stage4_start" \
        "$AUDIT_DIR/stage-2worker-metrics.json" "$AUDIT_DIR/stage-4worker-metrics.json" \
        2 000 001 002 003; then
      stage4_rc=0
    else
      stage4_rc=$?
    fi
    stop_managed_sampler stage-4worker
    case "$stage4_rc" in
      0) ;;
      10) echo 'four-worker performance/resource gate failed' | tee -a "$AUDIT_DIR/rollout.log" ;;
      2|3|4) return "$stage4_rc" ;;
      *) echo "unexpected four-worker gate status: $stage4_rc" >&2; return 3 ;;
    esac
    if (( stage4_rc == 10 )); then
      wait_shard 000
      wait_shard 001
      wait_shard 002
      wait_shard 003
      run_queue 2 004 005 006 007
    else
      date -u +%Y-%m-%dT%H:%M:%S+00:00 | tee "$AUDIT_DIR/stage-8worker-start.txt"
      local stage8_start stage8_rc
      stage8_start="$(cat "$AUDIT_DIR/stage-8worker-start.txt")"
      start_managed_sampler stage-8worker
      launch_formal_shard 004
      launch_formal_shard 005
      launch_formal_shard 006
      launch_formal_shard 007
      if wait_stage_metrics stage-8worker "$stage8_start" \
          "$AUDIT_DIR/stage-4worker-metrics.json" "$AUDIT_DIR/stage-8worker-metrics.json" \
          2 000 001 002 003 004 005 006 007; then
        stage8_rc=0
      else
        stage8_rc=$?
      fi
      stop_managed_sampler stage-8worker
      case "$stage8_rc" in
        0) ;;
        10) echo 'eight-worker resource/performance anomaly; retaining diagnostics' | tee -a "$AUDIT_DIR/rollout.log" ;;
        2|3|4) return "$stage8_rc" ;;
        *) echo "unexpected eight-worker monitor status: $stage8_rc" >&2; return 3 ;;
      esac
    fi
  fi

  wait_all_shards 000 001 002 003 004 005 006 007
  [[ ! -e "$FINAL_ROOT" && ! -L "$FINAL_ROOT" ]] || return 2
  "$PYTHON_BIN" -m validation_v2.cli merge-shards \
    --config "$CONFIG" --plan "$PLAN" --shards-root "$SHARDS_ROOT" --output-root "$FINAL_ROOT" \
    | tee "$AUDIT_DIR/merge.stdout.json"
  "$PYTHON_BIN" -m validation_v2.experiments.validate_artifacts \
    --root "$FINAL_ROOT" --config "$CONFIG" | tee "$AUDIT_DIR/validate.stdout.json"
  "$PYTHON_BIN" - "$FINAL_ROOT/validation_report.json" <<'PY'
import json
import sys
if json.load(open(sys.argv[1], encoding="utf-8")).get("status") != "complete":
    raise SystemExit("merged validation report is not complete")
PY
  "$PYTHON_BIN" -m validation_v2.cli summarize \
    --root "$FINAL_ROOT" --config "$CONFIG" \
    --required-seeds 2026 2027 2028 2029 2030 --baseline linear \
    | tee "$AUDIT_DIR/summarize.stdout.json"
}

cd "$REPO"
run_preflight
run_formal_campaign "$MODE"
