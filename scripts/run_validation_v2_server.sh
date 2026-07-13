#!/usr/bin/env bash
# Execute the immutable Validation v2 formal campaign on a Linux RTX 4090D host.

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
  --campaign-suffix NAME      New campaign suffix (default: sharded-v2-py312).
  --skip-dependency-install   Reuse an already-provisioned .venv-server, but still verify it.
  --help                      Show this help and exit.
EOF
}

die() {
  printf '%s\n' "$*" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd -P)"
COMMIT=""
MODE=""
CAMPAIGN_SUFFIX="sharded-v2-py312"
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
[[ "$CAMPAIGN_SUFFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die 'invalid --campaign-suffix'
[[ -d "$REPO" ]] || die "repository does not exist: $REPO"
REPO="$(cd "$REPO" && pwd -P)"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "not a Git worktree: $REPO"

# This must remain the first Python action: failures must not create a venv or
# attempt any package installation.
PYTHON3_BIN="${PYTHON3_BIN:-python3}"
PYTHON3_VERSION="$("$PYTHON3_BIN" --version 2>&1)" || die "cannot run PYTHON3_BIN: $PYTHON3_BIN"
[[ "$PYTHON3_VERSION" =~ ^Python[[:space:]]3\.12(\.|[[:space:]]) ]] \
  || die "Python 3.12 is required; found: $PYTHON3_VERSION"
[[ "$(uname -s)" == Linux ]] || die 'this formal runner requires Linux'

HEAD_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
[[ "$HEAD_COMMIT" == "$COMMIT" ]] || die "HEAD does not match --commit: $HEAD_COMMIT != $COMMIT"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || die 'Git worktree must be clean'

PYTHON_BIN="$REPO/.venv-server/bin/python"
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

PARENT_DIR="$(dirname "$REPO")"
PREFLIGHT_DIR="$PARENT_DIR/validation-v2-preflight-${COMMIT}-${CAMPAIGN_SUFFIX}"
AUDIT_DIR="$PARENT_DIR/validation-v2-audit-${COMMIT}-${CAMPAIGN_SUFFIX}"
PLAN="$AUDIT_DIR/server-full-8-shards-${COMMIT}.json"
SHARDS_ROOT="$REPO/results/validation_v2/server-full-shards-${COMMIT}-${CAMPAIGN_SUFFIX}"
FINAL_ROOT="$REPO/results/validation_v2/server-full-final-${COMMIT}-${CAMPAIGN_SUFFIX}"
CONFIG="$REPO/configs/validation_v2/server_full.yaml"
export REPO COMMIT PREFLIGHT_DIR AUDIT_DIR PLAN SHARDS_ROOT FINAL_ROOT CONFIG PYTHON_BIN
export CUBLAS_WORKSPACE_CONFIG=:4096:8

[[ -f "$CONFIG" ]] || die "missing formal configuration: $CONFIG"
for campaign_path in "$PREFLIGHT_DIR" "$AUDIT_DIR" "$SHARDS_ROOT" "$FINAL_ROOT"; do
  [[ ! -e "$campaign_path" && ! -L "$campaign_path" ]] \
    || die "campaign path already exists or is linked: $campaign_path"
done
mkdir -p "$PREFLIGHT_DIR" "$AUDIT_DIR" "$SHARDS_ROOT"
printf '%s\n' "$COMMIT" > "$AUDIT_DIR/COMMIT"

verify_runtime() {
  "$PYTHON_BIN" - "$AUDIT_DIR/environment.json" <<'PY'
import json
import platform
import sys

import torch

assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__ == "2.3.1+cu121", torch.__version__
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.version.cuda == "12.1", torch.version.cuda
name = torch.cuda.get_device_name(0)
assert "4090 D" in name.upper(), name
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
assert len(lines) == 4096, len(lines)
assert json.loads(lines[0])["combination_count"] == 4095
PY

  "$PYTHON_BIN" -m validation_v2.cli shard-plan \
    --config "$CONFIG" --shard-count 8 --output "$PLAN" --device cuda \
    | tee "$AUDIT_DIR/shard-plan.stdout.json"
  "$PYTHON_BIN" - "$PLAN" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
assert plan["schema_version"] == 2
assert plan["total_groups"] == 175
assert plan["total_cells"] == 4095
assert plan["shard_count"] == 8
assert plan["dirty_state_digest"] == ""
assert [len(shard["group_ids"]) for shard in plan["shards"]] == [22, 22, 22, 22, 22, 22, 22, 21]
PY
}

source "$REPO/scripts/lib/validation_v2_server_helpers.sh"

run_formal_campaign() {
  local requested_mode="$1"
  [[ "$requested_mode" == full ]] || return 0

  start_gpu_sampler baseline-1worker
  launch_shard 000
  wait_until_groups 000 2
  local baseline_start
  baseline_start="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["started_at"])' "$SHARDS_ROOT/000/shard_execution.json")"
  wait_stage_metrics baseline-1worker "$baseline_start" "" \
    "$AUDIT_DIR/baseline-1worker.json" 2 000
  stop_gpu_sampler baseline-1worker

  date -u +%Y-%m-%dT%H:%M:%S+00:00 | tee "$AUDIT_DIR/stage-2worker-start.txt"
  local stage2_start stage2_rc
  stage2_start="$(cat "$AUDIT_DIR/stage-2worker-start.txt")"
  start_gpu_sampler stage-2worker
  launch_shard 001
  if wait_stage_metrics stage-2worker "$stage2_start" \
      "$AUDIT_DIR/baseline-1worker.json" "$AUDIT_DIR/stage-2worker-metrics.json" 2 000 001; then
    stage2_rc=0
  else
    stage2_rc=$?
  fi
  stop_gpu_sampler stage-2worker
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
    start_gpu_sampler stage-4worker
    launch_shard 002
    launch_shard 003
    if wait_stage_metrics stage-4worker "$stage4_start" \
        "$AUDIT_DIR/stage-2worker-metrics.json" "$AUDIT_DIR/stage-4worker-metrics.json" \
        2 000 001 002 003; then
      stage4_rc=0
    else
      stage4_rc=$?
    fi
    stop_gpu_sampler stage-4worker
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
      start_gpu_sampler stage-8worker
      launch_shard 004
      launch_shard 005
      launch_shard 006
      launch_shard 007
      if wait_stage_metrics stage-8worker "$stage8_start" \
          "$AUDIT_DIR/stage-4worker-metrics.json" "$AUDIT_DIR/stage-8worker-metrics.json" \
          2 000 001 002 003 004 005 006 007; then
        stage8_rc=0
      else
        stage8_rc=$?
      fi
      stop_gpu_sampler stage-8worker
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
assert json.load(open(sys.argv[1], encoding="utf-8"))["status"] == "complete"
PY
  "$PYTHON_BIN" -m validation_v2.cli summarize \
    --root "$FINAL_ROOT" --config "$CONFIG" \
    --required-seeds 2026 2027 2028 2029 2030 --baseline linear \
    | tee "$AUDIT_DIR/summarize.stdout.json"
}

cd "$REPO"
run_preflight
run_formal_campaign "$MODE"
