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
CONFIG=${2:-configs/validation_v2/modern_stage_a.yaml}
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
COMMIT=$(git rev-parse HEAD)
[[ $COMMIT =~ ^[0-9a-f]{40}$ ]] || { echo "invalid git commit" >&2; exit 2; }
STATE="$ROOT/.modern-campaign"
SESSION="modern-imputation-${COMMIT:0:12}"
LOG="$STATE/campaign.log"
PY_MAIN="$ROOT/.venv-modern-pypots/bin/python"
PY_SSSD="$ROOT/.venv-modern-sssd/bin/python"

require_clean() {
  [[ -z $(git status --porcelain --untracked-files=all -- . ':(exclude).modern-campaign' ':(exclude).venv-modern-pypots' ':(exclude).venv-modern-sssd' ':(exclude)third_party/sssd/source') ]] || {
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
  "$PY_SSSD" -m pip install --upgrade pip
  "$PY_SSSD" -m pip install -r requirements-modern-sssd.txt
  if [[ ! -d third_party/sssd/source/.git ]]; then
    git clone https://github.com/AI4HealthUOL/SSSD.git third_party/sssd/source
  fi
  git -C third_party/sssd/source fetch --depth 1 origin 4d3b7a51c54b658945c0ba0bbb26e5ee1f763bed
  git -C third_party/sssd/source checkout --detach 4d3b7a51c54b658945c0ba0bbb26e5ee1f763bed
  [[ $(git -C third_party/sssd/source rev-parse HEAD) == 4d3b7a51c54b658945c0ba0bbb26e5ee1f763bed ]]
  "$PY_MAIN" -m pytest tests/validation_v2 tests/validation_v2/modern -q
  "$PY_MAIN" -m validation_v2.modern.pypots_worker preflight --output "$STATE/pypots-environment.json"
  "$PY_SSSD" -m validation_v2.modern.sssd_worker preflight --source third_party/sssd/source --output "$STATE/sssd-environment.json"
  printf '{"commit":"%s","config":"%s","status":"prepared"}\n' "$COMMIT" "$CONFIG" > "$STATE/prepared.json"
  echo "prepared $COMMIT"
}

pipeline() {
  local action=$1
  "$PY_MAIN" -m validation_v2.modern.cli plan --config "$CONFIG" --output "$STATE/results"
  "$PY_MAIN" -m validation_v2.modern.cli export --config "$CONFIG" --output "$STATE/results"
  "$PY_MAIN" -m validation_v2.modern.cli "$action" --config "$CONFIG" --output "$STATE/results" --pypots-python "$PY_MAIN" --sssd-python "$PY_SSSD"
  "$PY_MAIN" -m validation_v2.modern.cli validate --config "$CONFIG" --output "$STATE/results"
}

case "$COMMAND" in
  prepare) prepare ;;
  start)
    require_clean; [[ -f "$STATE/prepared.json" ]] || { echo "run prepare first" >&2; exit 2; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "campaign session already active" >&2; exit 2; }
    tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash -lc 'scripts/run_modern_imputation_matpool.sh _run "$CONFIG"'"
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
    "$PY_MAIN" -m validation_v2.modern.cli package-results --config "$CONFIG" --output "$STATE/results" --mode summary
    "$PY_MAIN" -m validation_v2.modern.cli package-results --config "$CONFIG" --output "$STATE/results" --mode full
    ;;
  *) usage >&2; exit 2 ;;
esac
