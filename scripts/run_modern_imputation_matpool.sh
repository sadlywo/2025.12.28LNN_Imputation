#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_modern_imputation_matpool.sh COMMAND [CONFIG]

Commands:
  prepare          verify host, create both environments, clone pinned SSSD, run preflights
  start            start the complete campaign in a detached tmux session
  status           show campaign seal, tmux state, and last lifecycle marker
  logs             follow the campaign log
  resume           resume the sealed campaign using the exact recorded commit
  package-results  create summary and full result archives after validation
EOF
}

[[ ${1:-} == --help || ${1:-} == -h ]] && { usage; exit 0; }
COMMAND=${1:-}
[[ -n "$COMMAND" ]] || { usage >&2; exit 2; }
if [[ $COMMAND == package-results ]]; then
  PACKAGE_MODE=${2:-summary}
  CONFIG=${3:-configs/validation_v2/modern_stage_a.yaml}
else
  PACKAGE_MODE=""
  CONFIG=${2:-configs/validation_v2/modern_stage_a.yaml}
fi
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
COMMIT=$(git rev-parse HEAD)
[[ $COMMIT =~ ^[0-9a-f]{40}$ ]] || { echo "invalid git commit" >&2; exit 2; }
STATE="$ROOT/.modern-campaign"
SESSION="modern-imputation-${COMMIT:0:12}"
LOG="$STATE/campaign.log"
PY_MAIN="$ROOT/.venv-modern-pypots/bin/python"
PY_SSSD="$ROOT/.venv-modern-sssd/bin/python"
SSSD_SOURCE="$ROOT/third_party/sssd/source"
SSSD_COMMIT="4d3b7a51c54b658945c0ba0bbb26e5ee1f763bed"
SSSD_MARKER="$SSSD_SOURCE/.pinned-commit"

require_clean() {
  [[ -z $(git status --porcelain --untracked-files=all -- . ':(exclude).modern-campaign' ':(exclude).venv-modern-pypots' ':(exclude).venv-modern-sssd' ':(exclude)third_party/sssd/source' ':(exclude)repository.bundle' ':(exclude)bootstrap.sh') ]] || {
    echo "worktree must be clean before campaign operations" >&2; exit 2;
  }
}

prepare() {
  [[ $(uname -s) == Linux ]] || { echo "prepare requires Linux" >&2; exit 2; }
  require_clean
  command -v nvidia-smi >/dev/null
  command -v python3.10 >/dev/null
  command -v tmux >/dev/null
  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -n1)
  [[ $gpu == *4090* ]] || { echo "RTX 4090 is required: $gpu" >&2; exit 2; }
  memory=$(printf '%s' "$gpu" | awk -F, '{gsub(/ /,"",$2); print $2}')
  (( memory >= 23000 )) || { echo "at least 23 GiB GPU memory is required" >&2; exit 2; }
  mkdir -m 700 -p "$STATE"
  [[ ! -L "$STATE" ]] || { echo "state directory must not be a symlink" >&2; exit 2; }
  python3.10 -m venv "$ROOT/.venv-modern-pypots"
  "$PY_MAIN" -m pip install --upgrade pip
  "$PY_MAIN" -m pip install -r requirements-modern-pypots.txt -r requirements-validation-v2.txt
  python3.10 -m venv "$ROOT/.venv-modern-sssd"
  # pytorch-lightning 1.8.2 has legacy metadata rejected by pip 24.1+.
  "$PY_SSSD" -m pip install --upgrade "pip==24.0"
  # CUDA 11.7 cuFFT fails on RTX 4090; use the official CUDA 11.8 wheel.
  "$PY_SSSD" -m pip install --index-url https://download.pytorch.org/whl/cu118 "torch==2.0.1"
  "$PY_SSSD" -m pip install -r requirements-modern-sssd.txt
  if [[ -d "$SSSD_SOURCE/.git" ]]; then
    git -C "$SSSD_SOURCE" fetch --depth 1 origin "$SSSD_COMMIT"
    git -C "$SSSD_SOURCE" checkout --detach "$SSSD_COMMIT"
    [[ $(git -C "$SSSD_SOURCE" rev-parse HEAD) == "$SSSD_COMMIT" ]]
  elif [[ -f "$SSSD_MARKER" ]]; then
    [[ $(tr -d '\r\n' < "$SSSD_MARKER") == "$SSSD_COMMIT" ]] || {
      echo "packaged SSSD source marker does not match the pinned commit" >&2; exit 2;
    }
    [[ -f "$SSSD_SOURCE/src/imputers/SSSDS4Imputer.py" ]] || {
      echo "packaged SSSD source is incomplete" >&2; exit 2;
    }
  else
    if [[ -e "$SSSD_SOURCE" && -n $(find "$SSSD_SOURCE" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
      echo "unverified non-empty SSSD source directory: $SSSD_SOURCE" >&2; exit 2;
    fi
    git clone https://github.com/AI4HealthUOL/SSSD.git "$SSSD_SOURCE"
    git -C "$SSSD_SOURCE" fetch --depth 1 origin "$SSSD_COMMIT"
    git -C "$SSSD_SOURCE" checkout --detach "$SSSD_COMMIT"
    [[ $(git -C "$SSSD_SOURCE" rev-parse HEAD) == "$SSSD_COMMIT" ]]
  fi
  "$PY_MAIN" -m pytest tests/validation_v2 tests/validation_v2/modern -q
  "$PY_MAIN" -m validation_v2.modern.pypots_worker preflight --output "$STATE/pypots-environment.json"
  "$PY_SSSD" -m validation_v2.modern.sssd_worker preflight --source "$SSSD_SOURCE" --output "$STATE/sssd-environment.json"
  printf '{"commit":"%s","config":"%s","status":"prepared"}\n' "$COMMIT" "$CONFIG" > "$STATE/prepared.json"
  echo "prepared $COMMIT"
}

pipeline() {
  local action=$1
  "$PY_MAIN" -m validation_v2.modern.cli plan --config "$CONFIG" --output "$STATE/results"
  "$PY_MAIN" -m validation_v2.modern.cli export --config "$CONFIG" --output "$STATE/results"
  if [[ ! -f "$STATE/results/selected_hyperparameters.json" ]]; then
    "$PY_MAIN" -m validation_v2.modern.cli tune --config "$CONFIG" --output "$STATE/results" --pypots-python "$PY_MAIN" --sssd-python "$PY_SSSD"
  fi
  "$PY_MAIN" -m validation_v2.modern.cli "$action" --config "$CONFIG" --output "$STATE/results" --pypots-python "$PY_MAIN" --sssd-python "$PY_SSSD"
  "$PY_MAIN" -m validation_v2.modern.cli summarize --config "$CONFIG" --output "$STATE/results"
  "$PY_MAIN" -m validation_v2.modern.cli validate --config "$CONFIG" --output "$STATE/results"
}

case "$COMMAND" in
  prepare) prepare ;;
  start)
    require_clean; [[ -f "$STATE/prepared.json" ]] || { echo "run prepare first" >&2; exit 2; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "campaign session already active" >&2; exit 2; }
    tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash -lc 'bash scripts/run_modern_imputation_matpool.sh _run \"$CONFIG\"'"
    echo "started $SESSION"
    ;;
  _run) pipeline run >>"$LOG" 2>&1 ;;
  resume)
    [[ -f "$STATE/prepared.json" ]] || { echo "sealed preparation not found" >&2; exit 2; }
    grep -q "\"commit\":\"$COMMIT\"" "$STATE/prepared.json" || { echo "commit differs from sealed campaign" >&2; exit 2; }
    pipeline resume >>"$LOG" 2>&1
    ;;
  status) cat "$STATE/prepared.json" 2>/dev/null || true; tmux has-session -t "$SESSION" 2>/dev/null && echo active || echo inactive ;;
  logs) touch "$LOG"; tail -n 200 -f "$LOG" ;;
  package-results)
    [[ -f "$STATE/results/validation-report.json" ]] || { echo "validated results required" >&2; exit 2; }
    [[ $PACKAGE_MODE == summary || $PACKAGE_MODE == full ]] || { echo "mode must be summary or full" >&2; exit 2; }
    "$PY_MAIN" -m validation_v2.modern.cli package-results --config "$CONFIG" --output "$STATE/results" --mode "$PACKAGE_MODE"
    ;;
  *) usage >&2; exit 2 ;;
esac
