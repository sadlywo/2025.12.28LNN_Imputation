# Python 3.12 server runner design

## Purpose

Provide one executable entry point for the formal Validation v2 campaign on a
Linux RTX 4090D server that has only a system Python 3.12 installation. The
entry point must avoid the historical conda environment, create a repository
local virtual environment, install the pinned CUDA runtime, and preserve the
existing formal campaign's safety and provenance gates.

## Command interface

The new executable is:

```bash
bash scripts/run_validation_v2_server.sh --commit <40-hex-sha> --mode full
```

It also supports `--mode preflight` for environment setup, tests, and immutable
plan creation without starting training. Both modes accept `--repo <path>`,
`--campaign-suffix <name>`, and `--skip-dependency-install`. The script
rejects unknown options, a non-40-hex commit, a non-Linux host, a dirty
worktree, a non-Python-3.12 interpreter, and pre-existing campaign paths.

## Runtime installation

The script uses `python3 -m venv .venv-server`; it never activates or modifies
`pinn_imu`, conda, or the system site-packages. It installs:

1. `torch==2.3.1` from `https://download.pytorch.org/whl/cu121`, then
2. the remaining exact packages from `requirements-validation-v2.txt`.

Before work begins, it verifies that the venv runs Python 3.12, reports
`torch.__version__ == 2.3.1+cu121`, has CUDA available, and sees a GPU whose
name contains `4090 D`. The plan and run manifests record this Python 3.12
runtime fingerprint; no result is presented as identical to the earlier Python
3.9 validation environment.

## Formal workflow

The script records commit-qualified `AUDIT_DIR`, `PLAN`, `SHARDS_ROOT`, and
`FINAL_ROOT` paths. It performs, in order:

1. Linux atomic no-replace test and the complete pytest suite;
2. dry-run matrix validation (4,095 cells) and immutable plan creation
   (175 groups, eight shards);
3. one-worker baseline, then 2, 4, and 8 worker staged rollout;
4. per-active-shard progress gate after each stage start, failure/PID checks,
   GPU memory and throughput gates, safe fallback queues, and bounded waits;
5. strict merge, artifact validation, and five-seed summary.

The old `server_full-fcf81f8` root is never modified or consumed. Every shard
gets an isolated `000`--`007` root. A failed marker or provenance mismatch
is terminal for that campaign and is not silently resumed.

## Implementation boundaries

The script is the sourceable implementation of the existing runbook's server
orchestration. The English and Chinese runbooks will link to it and retain
operator explanations, but no executable behavior is extracted dynamically
from Markdown. The script uses a venv-local Python variable for every Python
command and never sources Network Turbo except inside short dependency-install
subshells.

## Testing

Tests are added before implementation. They exercise the script's `--help`
and option validation, reject a non-3.12 interpreter before installation,
assert the CUDA 12.1 wheel index and venv-local interpreter are used, and
verify that the full-mode orchestration contains the immutable plan, staged
shard gate, strict merge, validation, and summary calls. Bash syntax validation
is mandatory. Existing validation-v2 tests continue to run unchanged.

## Non-goals

- Changing model algorithms, experiment configuration, or the formal
  175-group/4,095-cell matrix.
- Installing or modifying a system Python, CUDA driver, or conda environment.
- Automatically retrying a failed formal campaign under the same output roots.
