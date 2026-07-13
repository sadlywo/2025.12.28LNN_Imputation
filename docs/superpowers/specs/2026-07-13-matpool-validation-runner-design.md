# MatPool Validation v2 runner design

**Date:** 2026-07-13
**Status:** Proposed for user review
**Scope:** Execution-layer compatibility and operation only. The scientific
configuration, models, losses, data splits, metrics, shard plan, and merge
rules remain unchanged.

## 1. Context

The existing formal runner targets a Linux RTX 4090 D host with CPython 3.12.
It deliberately rejects any other Python minor version or GPU model. The new
MatPool host has:

- repository `/2025.12.28LNN_Imputation`, clean `main`, including Validation v2;
- Debian Linux, `/usr/bin/python3` 3.10.12, and working `venv` support;
- one NVIDIA GeForce RTX 4090 with 24,564 MiB, driver 565.77;
- 10 CPUs visible to the container, 50 GiB RAM, and about 299 GiB free disk;
- direct access to GitHub, PyPI, and the PyTorch CUDA 12.1 wheel index;
- `bash`, `git`, `nvidia-smi`, and `tmux` available.

The current runner therefore fails before dependency installation even though
PyTorch 2.3.1+cu121 provides a CPython 3.10 Linux wheel and the GPU can execute
the unchanged experiment.

## 2. Goals

1. Preserve the exact formal experiment and its immutable provenance rules.
2. Support CPython 3.10, 3.11, and 3.12 without Conda.
3. Support both RTX 4090 and RTX 4090 D 24 GB hosts.
4. Give the MatPool user one short command to start and short commands to check
   status and follow logs.
5. Run all eight formal shards while limiting MatPool's default simultaneous
   worker count to four.
6. Keep the previous Python 3.12/4090 D invocation valid.

## 3. Non-goals

- No change to `server_full.yaml`, the 175 training groups, the 4,095
  experiment cells, or seeds 2026--2030.
- No change to model architecture, loss functions, missingness generation,
  training, evaluation, statistics, artifact validation, or shard merging.
- No automatic `git pull`, branch switching, system package installation, or
  mutation of the host's Conda installation.
- No SSH credentials, host passwords, or platform tokens in source, tests,
  logs, documentation, commits, or provenance artifacts.
- No launcher `stop` command. Stopping the parent runner does not safely imply
  stopping already launched shards, so the interface must not suggest that it
  does.

## 4. Chosen architecture

### 4.1 Generic formal runner

`scripts/run_validation_v2_server.sh` remains the authoritative execution
engine. Its environment policy becomes platform-neutral:

- The system interpreter must be CPython 3.10, 3.11, or 3.12. Versions 3.9 and
  3.13 or later fail before virtual-environment creation or package install.
- The repository-local `.venv-server` is still the only runtime environment.
- PyTorch remains pinned to `2.3.1+cu121`; CUDA reported by Torch must remain
  exactly 12.1 and CUDA must be available.
- GPU 0 must contain `4090` in its normalized device name and report at least
  23 GiB of total memory. This accepts both 4090 and 4090 D while rejecting
  materially different hardware.
- The runtime manifest records the precise Python version, Torch version, CUDA
  version, GPU name, GPU memory, compute capability, driver version, visible
  CPU-affinity count, and host memory. These are provenance fields, not
  scientific inputs.
- If no campaign suffix is supplied, it is derived from the actual runtime as
  `sharded-v2-py310`, `sharded-v2-py311`, or `sharded-v2-py312`. An explicit
  suffix remains authoritative.
- A read-only `venv`/tool availability probe occurs before the immutable
  campaign seal is consumed. Missing host prerequisites produce a concrete
  installation hint and make no repository or campaign writes.

The runner adds `--max-workers 1|2|4|8`, defaulting to 8 for backward
compatibility. The option changes scheduling only, never the shard plan:

- `1`: run the remaining shard queue one at a time after the baseline gate;
- `2`: stop expansion after the two-worker gate and queue the remainder with
  two workers;
- `4`: stop expansion after the four-worker gate and queue the remainder with
  four workers;
- `8`: retain the complete staged 1 -> 2 -> 4 -> 8 rollout.

If a two- or four-worker gate fails, the existing lower-concurrency fallback
continues to apply. Every path waits for and validates all eight shards before
merge and summary publication.

### 4.2 MatPool convenience launcher

Add `scripts/run_validation_v2_matpool.sh` with this public interface:

```bash
bash scripts/run_validation_v2_matpool.sh start [--max-workers 1|2|4|8] [--skip-dependency-install]
bash scripts/run_validation_v2_matpool.sh status
bash scripts/run_validation_v2_matpool.sh logs
```

`start` performs only launcher-level checks, then creates a detached `tmux`
session that invokes the generic runner in `full` mode with:

- the current exact 40-character `HEAD` commit;
- a unique UTC suffix beginning with `matpool-`;
- `--max-workers 4` by default;
- the repository discovered from the launcher's own location.

The first start creates `.venv-server`, installs the pinned dependencies, runs
the complete preflight test suite, and proceeds to formal training only if all
preflight gates pass. A later start may explicitly request dependency reuse,
but the generic runner still verifies the environment.

Launcher state and logs live under the ignored directory
`.validation-v2-matpool/`. State includes the commit, campaign suffix, tmux
session name, audit directory, shard root, final root, and log path. State is
published atomically. `start` refuses a dirty worktree, an existing live
launcher session, a malformed state file, or a reused campaign suffix.

`status` never changes process state. It reports whether the tmux session is
active and prints the recorded campaign paths plus the latest concise log
lines. `logs` follows the recorded log and fails clearly when no campaign has
been started. Neither command infers scientific completion solely from the
tmux session; completion is established by the final Validation v2 artifact
validator and report.

The launcher accepts an explicit `--max-workers 1|2|4|8` override. Four is the
MatPool default because this container exposes 10 CPUs and 50 GiB RAM. Users
may opt into eight only after reviewing the four-worker GPU, throughput, and
host-resource evidence.

## 5. Data and control flow

```text
MatPool start command
  -> launcher checks repository, tmux, state, and commit
  -> detached tmux session + append-only launcher log
  -> generic full runner
       -> system Python and host prerequisite checks
       -> repository-local venv and pinned dependency install
       -> runtime verification and immutable campaign seal
       -> Linux atomic test + full pytest + matrix/shard-plan audit
       -> baseline -> 2-worker -> 4-worker staged rollout
       -> bounded queue completes all eight shards
       -> strict atomic merge -> artifact validation -> five-seed summary
  -> status/logs provide read-only observation
```

## 6. Failure and recovery behavior

- Environment, Git, or prerequisite failures occur before training and are
  written to the launcher log.
- Once the generic runner creates an audit seal, that suffix is never reused.
  A retry always receives a new UTC suffix.
- A launcher or terminal disconnect does not terminate work because the runner
  is owned by the detached tmux session.
- Signals retain the existing fail-closed behavior: sampler processes are
  cleaned up, but launched shards and the campaign seal are preserved for
  diagnosis.
- The launcher never automatically deletes artifacts, kills shards, switches
  commits, or resumes an ambiguous campaign.
- Disk, GPU, CUDA, test, stage-gate, shard, merge, or validation errors yield a
  nonzero runner status and an inspectable audit trail.

## 7. Test strategy

Implementation follows test-first development. Automated tests must cover:

1. acceptance of Python 3.10/3.11/3.12 and rejection of 3.9/3.13 before writes;
2. dynamic default campaign suffix and explicit-suffix compatibility;
3. acceptance of RTX 4090 and RTX 4090 D, memory threshold enforcement, and
   rejection of non-4090 GPUs;
4. exact Torch/CUDA pins and expanded environment provenance;
5. `--max-workers` validation and the 1/2/4/8 scheduling boundaries while
   still accounting for all eight shards;
6. launcher `start`, duplicate-start refusal, clean-worktree enforcement,
   atomic state publication, exact-commit forwarding, and default four-worker
   forwarding using fake process boundaries rather than real training;
7. read-only `status` and `logs` behavior, including missing or malformed state;
8. absence of SSH secrets from tracked files;
9. shell syntax, Python compilation, focused tests, and the complete pytest
   suite.

The Linux preflight on MatPool is the final integration test for the real
Python 3.10, CUDA, RTX 4090, atomic filesystem, and dependency-wheel path.

## 8. Documentation and rollout

Update both server runbooks so the generic Python 3.10--3.12/4090-series path
is current. Add a short MatPool section using repository path
`/2025.12.28LNN_Imputation` and the three launcher commands. Preserve the old
manual blocks only as clearly historical reference.

After local implementation, independent specification review, code-quality
review, full verification, commit, and push:

1. remotely fetch and check out the reviewed commit in the clean MatPool repo;
2. run the generic runner in a unique `preflight` campaign to verify the real
   host without starting formal training;
3. report the preflight evidence and the exact one-command `start` handoff;
4. do not start the paid multi-day formal campaign during rollout unless the
   user explicitly requests that final action after seeing preflight results.
