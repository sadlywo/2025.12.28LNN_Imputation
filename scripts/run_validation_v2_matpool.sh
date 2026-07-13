#!/usr/bin/env bash
# Launch an immutable Validation v2 campaign in a detached MatPool tmux session.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_validation_v2_matpool.sh start [--max-workers 1|2|4|8] [--skip-dependency-install]
  bash scripts/run_validation_v2_matpool.sh status
  bash scripts/run_validation_v2_matpool.sh logs
  bash scripts/run_validation_v2_matpool.sh --help
EOF
}

die() {
  printf '%s\n' "$*" >&2
  exit 2
}

require_linux_tools() {
  [[ "$(uname -s 2>/dev/null)" == Linux ]] || die 'this launcher requires Linux'
  local tool
  for tool in bash git python3 tmux tail; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool is unavailable: $tool"
  done
}

inspect_session() {
  local session="$1"
  local result
  if tmux has-session -t "$session" >/dev/null 2>&1; then
    result=0
  else
    result=$?
  fi
  case "$result" in
    0) SESSION_STATE=active ;;
    1) SESSION_STATE=inactive ;;
    *) die "tmux inspection failed: rc=$result" ;;
  esac
}

load_state() {
  [[ -f "$STATE_FILE" && ! -L "$STATE_FILE" ]] || die "no current state: $STATE_FILE"
  local decoded
  if ! decoded="$(python3 - "$STATE_FILE" <<'PY'
import json
import posixpath
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(newline="\n")


def reject(message):
    raise SystemExit(message)


def object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            reject("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        state = json.load(handle, object_pairs_hook=object_without_duplicates)
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    reject("cannot parse state: {}".format(error))

keys = {
    "schema_version",
    "commit",
    "campaign_suffix",
    "session",
    "command_file",
    "log_path",
    "exit_status_path",
    "audit_dir",
    "shards_root",
    "final_root",
    "max_workers",
    "skip_dependency_install",
    "created_at",
}
if not isinstance(state, dict) or set(state) != keys:
    reject("unexpected state schema")
if state["schema_version"] != 1 or isinstance(state["schema_version"], bool):
    reject("unsupported state schema version")
if not isinstance(state["commit"], str) or not re.fullmatch(
    r"[0-9a-f]{40}", state["commit"]
):
    reject("invalid state commit")
if not isinstance(state["campaign_suffix"], str) or not re.fullmatch(
    r"matpool-[0-9]{8}T[0-9]{6}Z", state["campaign_suffix"]
):
    reject("invalid state campaign suffix")
if not isinstance(state["session"], str) or not re.fullmatch(
    r"[A-Za-z0-9][A-Za-z0-9._-]*", state["session"]
):
    reject("invalid state session")
if type(state["max_workers"]) is not int or state["max_workers"] not in (1, 2, 4, 8):
    reject("invalid state max_workers")
if type(state["skip_dependency_install"]) is not bool:
    reject("invalid state skip_dependency_install")
if not isinstance(state["created_at"], str) or not re.fullmatch(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
    state["created_at"],
):
    reject("invalid state created_at")

for key, value in state.items():
    if isinstance(value, str) and ("\n" in value or "\r" in value or "\0" in value):
        reject("unsafe characters in state field: {}".format(key))
for key in (
    "command_file",
    "log_path",
    "exit_status_path",
    "audit_dir",
    "shards_root",
    "final_root",
):
    value = state[key]
    if not isinstance(value, str) or not (
        posixpath.isabs(value) or re.fullmatch(r"[A-Za-z]:/.*", value)
    ):
        reject("state path is not absolute: {}".format(key))
    if posixpath.normpath(value) != value:
        reject("state path is not normalized: {}".format(key))

ordered = (
    "commit",
    "campaign_suffix",
    "session",
    "command_file",
    "log_path",
    "exit_status_path",
    "audit_dir",
    "shards_root",
    "final_root",
    "max_workers",
    "skip_dependency_install",
    "created_at",
)
for key in ordered:
    value = state[key]
    if isinstance(value, bool):
        value = "true" if value else "false"
    print(value)
PY
)"; then
    die "malformed state: strict JSON validation failed: $STATE_FILE"
  fi

  local -a fields=()
  mapfile -t fields <<< "$decoded"
  [[ "${#fields[@]}" -eq 12 ]] || die "malformed state: invalid field count: $STATE_FILE"
  STATE_COMMIT="${fields[0]}"
  STATE_SUFFIX="${fields[1]}"
  STATE_SESSION="${fields[2]}"
  STATE_COMMAND_FILE="${fields[3]}"
  STATE_LOG_PATH="${fields[4]}"
  STATE_EXIT_STATUS_PATH="${fields[5]}"
  STATE_AUDIT_DIR="${fields[6]}"
  STATE_SHARDS_ROOT="${fields[7]}"
  STATE_FINAL_ROOT="${fields[8]}"
  STATE_MAX_WORKERS="${fields[9]}"
  STATE_SKIP_DEPENDENCY_INSTALL="${fields[10]}"
  STATE_CREATED_AT="${fields[11]}"

  [[ "$STATE_SESSION" == "validation-v2-${STATE_COMMIT:0:12}-${STATE_SUFFIX}" ]] \
    || die "malformed state: session does not match campaign: $STATE_FILE"
  [[ "$STATE_COMMAND_FILE" == "$STATE_DIR/run-${STATE_SUFFIX}.sh" ]] \
    || die "malformed state: command path is outside its campaign: $STATE_FILE"
  [[ "$STATE_LOG_PATH" == "$STATE_DIR/run-${STATE_SUFFIX}.log" ]] \
    || die "malformed state: log path is outside its campaign: $STATE_FILE"
  [[ "$STATE_EXIT_STATUS_PATH" == "$STATE_DIR/run-${STATE_SUFFIX}.exit" ]] \
    || die "malformed state: exit path is outside its campaign: $STATE_FILE"
  [[ "$STATE_AUDIT_DIR" == "$(dirname "$REPO")/validation-v2-audit-${STATE_COMMIT}-${STATE_SUFFIX}" ]] \
    || die "malformed state: audit path does not match campaign: $STATE_FILE"
  [[ "$STATE_SHARDS_ROOT" == "$REPO/results/validation_v2/server-full-shards-${STATE_COMMIT}-${STATE_SUFFIX}" ]] \
    || die "malformed state: shards path does not match campaign: $STATE_FILE"
  [[ "$STATE_FINAL_ROOT" == "$REPO/results/validation_v2/server-full-final-${STATE_COMMIT}-${STATE_SUFFIX}" ]] \
    || die "malformed state: final path does not match campaign: $STATE_FILE"
}

publish_state() {
  local temporary="$STATE_DIR/.current.json.${CAMPAIGN_SUFFIX}.$$.$RANDOM"
  export MATPOOL_STATE_FILE="$STATE_FILE"
  export MATPOOL_STATE_TEMP="$temporary"
  export MATPOOL_COMMIT="$COMMIT"
  export MATPOOL_CAMPAIGN_SUFFIX="$CAMPAIGN_SUFFIX"
  export MATPOOL_SESSION="$SESSION"
  export MATPOOL_COMMAND_FILE="$COMMAND_FILE"
  export MATPOOL_LOG_PATH="$LOG_PATH"
  export MATPOOL_EXIT_STATUS_PATH="$EXIT_STATUS_PATH"
  export MATPOOL_AUDIT_DIR="$AUDIT_DIR"
  export MATPOOL_SHARDS_ROOT="$SHARDS_ROOT"
  export MATPOOL_FINAL_ROOT="$FINAL_ROOT"
  export MATPOOL_MAX_WORKERS="$MAX_WORKERS"
  export MATPOOL_SKIP_DEPENDENCY_INSTALL="$SKIP_DEPENDENCY_INSTALL"
  export MATPOOL_CREATED_AT="$CREATED_AT"
  python3 - <<'PY'
import json
import os

state = {
    "schema_version": 1,
    "commit": os.environ["MATPOOL_COMMIT"],
    "campaign_suffix": os.environ["MATPOOL_CAMPAIGN_SUFFIX"],
    "session": os.environ["MATPOOL_SESSION"],
    "command_file": os.environ["MATPOOL_COMMAND_FILE"],
    "log_path": os.environ["MATPOOL_LOG_PATH"],
    "exit_status_path": os.environ["MATPOOL_EXIT_STATUS_PATH"],
    "audit_dir": os.environ["MATPOOL_AUDIT_DIR"],
    "shards_root": os.environ["MATPOOL_SHARDS_ROOT"],
    "final_root": os.environ["MATPOOL_FINAL_ROOT"],
    "max_workers": int(os.environ["MATPOOL_MAX_WORKERS"]),
    "skip_dependency_install": os.environ["MATPOOL_SKIP_DEPENDENCY_INSTALL"] == "true",
    "created_at": os.environ["MATPOOL_CREATED_AT"],
}
temporary = os.environ["MATPOOL_STATE_TEMP"]
with open(temporary, "x", encoding="utf-8") as handle:
    json.dump(state, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, os.environ["MATPOOL_STATE_FILE"])
PY
}

write_command_file() {
  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'set -o pipefail'
    printf 'cd -- %q\n' "$REPO"
    printf '%s\n' 'set +e'
    printf '%q ' "$GENERIC_RUNNER" --commit "$COMMIT" --mode full \
      --campaign-suffix "$CAMPAIGN_SUFFIX" --max-workers "$MAX_WORKERS"
    if [[ "$SKIP_DEPENDENCY_INSTALL" == true ]]; then
      printf '%q ' --skip-dependency-install
    fi
    printf '2>&1 | tee -a -- %q\n' "$LOG_PATH"
    printf '%s\n' 'runner_status=${PIPESTATUS[0]}'
    printf 'exit_status_path=%q\n' "$EXIT_STATUS_PATH"
    printf '%s\n' 'status_tmp="${exit_status_path}.tmp.$$"'
    printf '%s\n' '(umask 077; set -o noclobber; : > "$status_tmp") || exit 125'
    printf '%s\n' 'printf "%s\n" "$runner_status" > "$status_tmp"'
    printf '%s\n' 'mv -- "$status_tmp" "$exit_status_path"'
    printf '%s\n' 'exit "$runner_status"'
  } > "$COMMAND_FILE"
}

start_campaign() {
  require_linux_tools
  [[ -L "$STATE_DIR" ]] && die "launcher state directory must not be linked: $STATE_DIR"
  if [[ -e "$STATE_DIR" ]]; then
    [[ -d "$STATE_DIR" ]] || die "launcher state path is not a directory: $STATE_DIR"
  fi

  COMMIT="$(git -C "$REPO" rev-parse --verify HEAD 2>/dev/null)" \
    || die "cannot resolve repository HEAD: $REPO"
  [[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] \
    || die "repository HEAD is not an exact 40-character lowercase commit: $COMMIT"
  [[ -z "$(git -C "$REPO" status --porcelain)" ]] || die 'Git worktree must be clean'

  if [[ -e "$STATE_FILE" || -L "$STATE_FILE" ]]; then
    load_state
    inspect_session "$STATE_SESSION"
    [[ "$SESSION_STATE" != active ]] || die "campaign is already active: $STATE_SESSION"
  fi

  local timestamp
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  [[ "$timestamp" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || die "cannot create UTC campaign timestamp"
  CAMPAIGN_SUFFIX="matpool-$timestamp"
  CREATED_AT="${timestamp:0:4}-${timestamp:4:2}-${timestamp:6:2}T${timestamp:9:2}:${timestamp:11:2}:${timestamp:13:2}Z"
  SESSION="validation-v2-${COMMIT:0:12}-${CAMPAIGN_SUFFIX}"
  COMMAND_FILE="$STATE_DIR/run-${CAMPAIGN_SUFFIX}.sh"
  LOG_PATH="$STATE_DIR/run-${CAMPAIGN_SUFFIX}.log"
  EXIT_STATUS_PATH="$STATE_DIR/run-${CAMPAIGN_SUFFIX}.exit"
  AUDIT_DIR="$(dirname "$REPO")/validation-v2-audit-${COMMIT}-${CAMPAIGN_SUFFIX}"
  SHARDS_ROOT="$REPO/results/validation_v2/server-full-shards-${COMMIT}-${CAMPAIGN_SUFFIX}"
  FINAL_ROOT="$REPO/results/validation_v2/server-full-final-${COMMIT}-${CAMPAIGN_SUFFIX}"

  local target
  for target in "$COMMAND_FILE" "$LOG_PATH" "$EXIT_STATUS_PATH" \
    "$AUDIT_DIR" "$SHARDS_ROOT" "$FINAL_ROOT"; do
    [[ ! -e "$target" && ! -L "$target" ]] || die "campaign target already exists: $target"
  done

  if [[ ! -e "$STATE_DIR" ]]; then
    mkdir "$STATE_DIR" || die "cannot create launcher state directory: $STATE_DIR"
  fi
  umask 077
  (set -o noclobber; : > "$COMMAND_FILE") \
    || die "cannot reserve unique command file: $COMMAND_FILE"
  chmod 700 "$COMMAND_FILE" || die "cannot secure command file: $COMMAND_FILE"
  write_command_file
  (set -o noclobber; : > "$LOG_PATH") \
    || die "cannot reserve unique log file: $LOG_PATH"
  publish_state || die "cannot atomically publish launcher state: $STATE_FILE"

  local tmux_status
  if tmux new-session -d -s "$SESSION" bash "$COMMAND_FILE"; then
    printf 'started session: %s\n' "$SESSION"
    printf 'log: %s\n' "$LOG_PATH"
  else
    tmux_status=$?
    printf 'tmux start failed: rc=%s session=%s\n' "$tmux_status" "$SESSION" \
      | tee -a -- "$LOG_PATH" >&2
    exit "$tmux_status"
  fi
}

show_status() {
  require_linux_tools
  [[ -L "$STATE_DIR" ]] && die "launcher state directory must not be linked: $STATE_DIR"
  [[ -d "$STATE_DIR" ]] || die "no current state: $STATE_FILE"
  load_state
  inspect_session "$STATE_SESSION"
  printf 'state: %s\n' "$SESSION_STATE"
  printf 'commit: %s\n' "$STATE_COMMIT"
  printf 'campaign_suffix: %s\n' "$STATE_SUFFIX"
  printf 'session: %s\n' "$STATE_SESSION"
  printf 'max_workers: %s\n' "$STATE_MAX_WORKERS"
  printf 'audit_dir: %s\n' "$STATE_AUDIT_DIR"
  printf 'shards_root: %s\n' "$STATE_SHARDS_ROOT"
  printf 'final_root: %s\n' "$STATE_FINAL_ROOT"
  printf 'log_path: %s\n' "$STATE_LOG_PATH"
  printf 'exit_status_path: %s\n' "$STATE_EXIT_STATUS_PATH"
  if [[ -f "$STATE_LOG_PATH" && ! -L "$STATE_LOG_PATH" ]]; then
    printf '%s\n' '--- log tail (up to 20 lines) ---'
    tail -n 20 -- "$STATE_LOG_PATH"
  fi
}

follow_logs() {
  require_linux_tools
  [[ -L "$STATE_DIR" ]] && die "launcher state directory must not be linked: $STATE_DIR"
  [[ -d "$STATE_DIR" ]] || die "no current state: $STATE_FILE"
  load_state
  [[ -f "$STATE_LOG_PATH" && ! -L "$STATE_LOG_PATH" ]] \
    || die "campaign log does not exist: $STATE_LOG_PATH"
  tail -F -- "$STATE_LOG_PATH"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd -P)"
STATE_DIR="$REPO/.validation-v2-matpool"
STATE_FILE="$STATE_DIR/current.json"
GENERIC_RUNNER="$REPO/scripts/run_validation_v2_server.sh"
MAX_WORKERS=4
SKIP_DEPENDENCY_INSTALL=false

[[ "$#" -gt 0 ]] || die 'command is required; use --help for usage'
case "$1" in
  --help)
    [[ "$#" -eq 1 ]] || die '--help does not accept arguments'
    usage
    ;;
  start)
    shift
    while [[ "$#" -gt 0 ]]; do
      case "$1" in
        --max-workers)
          [[ "$#" -ge 2 ]] || die '--max-workers requires 1, 2, 4, or 8'
          MAX_WORKERS="$2"
          shift 2
          ;;
        --skip-dependency-install)
          SKIP_DEPENDENCY_INSTALL=true
          shift
          ;;
        *) die "unknown option for start: $1" ;;
      esac
    done
    [[ "$MAX_WORKERS" == 1 || "$MAX_WORKERS" == 2 || "$MAX_WORKERS" == 4 || "$MAX_WORKERS" == 8 ]] \
      || die '--max-workers must be one of 1, 2, 4, or 8'
    [[ -f "$GENERIC_RUNNER" && ! -L "$GENERIC_RUNNER" ]] \
      || die "generic runner is unavailable: $GENERIC_RUNNER"
    start_campaign
    ;;
  status)
    [[ "$#" -eq 1 ]] || die 'status does not accept arguments'
    show_status
    ;;
  logs)
    [[ "$#" -eq 1 ]] || die 'logs does not accept arguments'
    follow_logs
    ;;
  *) die "unknown command: $1" ;;
esac
