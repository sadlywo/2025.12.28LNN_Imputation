# Validation v2 Training-Group Sharding and Merge Design

## Status

Approved direction: replace the single-writer, serial formal matrix execution
with reproducible training-group shards and a strict, atomic merge. The existing
`fcf81f8` serial server run remains untouched until a sharded real-data pilot
passes.

## Problem

The formal matrix contains 4,095 reporting cells but only 175 independent
training groups. A group is identified by:

```text
(training_family, seed, protocol, objective)
```

The current runner executes those groups serially. On the RTX 4090 D server, a
group takes about 72 minutes while GPU utilization is about 5%. The workload is
dominated by small recurrent kernels, repeated CPU-to-GPU batch transfers,
per-batch synchronization, and full-record evaluation. Raw GPU peak throughput
therefore does not translate into high single-process throughput.

Running multiple matrix processes against one result root is unsafe because
they overwrite one `matrix_execution.json`, may race on shared assets, and may
attempt the same checkpoint or one-time test ledger.

## Goals

- Partition only at complete training-group boundaries.
- Preserve all scientific inputs: models, families, seeds, protocols,
  missingness conditions, epochs, window counts, splits, metrics, and device.
- Give every shard one exclusive output root and one exclusive set of groups.
- Make interrupted shards resumable only at completed group boundaries.
- Reject duplicate, missing, foreign, partial, or provenance-inconsistent shard
  inputs before merging.
- Produce a fresh final root accepted by the existing strict artifact validator.
- Preserve the serial execution path and its current behavior.
- Support a controlled 4090 D concurrency pilot before scaling worker count.

## Non-goals

- Epoch-level checkpoint/resume.
- Changing the training loop, floating-point reduction order, model definition,
  hyperparameters, evaluation protocol, or statistical analysis.
- Sharing one mutable result directory between workers.
- Treating `--max-combinations` output as a formal shard.
- Deleting or mutating the existing `server_full-fcf81f8` result root.

## Alternatives considered

### Continue the serial runner

This has the smallest engineering risk but requires roughly eight to nine days
at the measured rate and leaves most GPU capacity unused.

### Ad hoc staging workers with manual promotion

This preserves the current commit and can reuse existing serial results, but it
depends on an undocumented orchestration script and manual directory promotion.
It is difficult to reproduce and too easy to merge an incomplete group.

### First-class group shards and strict merge

This requires implementation and a fresh formal root, but it gives isolated
writers, deterministic coverage, auditable manifests, resumability, and a
validator-backed final result. This is the selected design.

## Architecture

### Stable group enumeration

Move or expose the current grouping logic as one shared function used by serial
execution, shard planning, shard execution, and merge validation. Its output is
an ordered sequence of immutable group plans containing:

- zero-based `group_index` and stable `group_id`;
- `training_family`, `training_model`, `reported_models`;
- `seed`, `protocol`, and `objective`;
- the complete ordered condition list and its combination IDs.

The stable `group_id` is the SHA-256 digest of canonical JSON containing the
group key and combination IDs. It does not depend on shard count or shard
index.

The full config must still enumerate exactly 175 groups and 4,095 unique
combination IDs for the current formal experiment.

### Deterministic shard plan

Add a planning command:

```bash
python -m validation_v2.cli shard-plan \
  --config configs/validation_v2/server_full.yaml \
  --shard-count 8 \
  --output plan.json
```

Assignment is deterministic round-robin by ordered group index:

```text
shard_index = group_index mod shard_count
```

Round-robin assignment balances families and protocols better than contiguous
ranges while remaining simple to audit. The plan records:

- schema version and creation timestamp;
- canonical source-config SHA-256;
- expected git commit and resolved device class;
- total groups and total cells;
- shard count;
- every shard's group IDs, group keys, and combination IDs;
- the digest of the complete plan.

The timestamp is excluded from the plan digest.

### Shard execution

Add a command:

```bash
python -m validation_v2.cli shard \
  --config configs/validation_v2/server_full.yaml \
  --plan plan.json \
  --shard-index 0 \
  --output-root results/validation_v2/server_full-<commit>/shards/000 \
  --device cuda
```

Each shard root is exclusive to one process. Before training, the worker checks
that the config hash, commit, shard index, shard count, group IDs, and device
match the plan. It writes `shard_execution.json` with `status: started`, then
executes only its assigned complete groups using the existing `run_smoke`
contract.

Completed groups may be reused only when all formal training and one-time test
artifacts pass the current content checks. A partial group remains a hard
failure; it is never silently restarted in place. A shard command may be rerun
only when its marker is `started` and every existing run directory is complete,
or when its marker is `completed`. Failed/partial shard roots are preserved for
diagnosis and replaced with fresh roots.

On success, the worker atomically replaces its marker with `status: completed`
and records the exact run IDs. No shard writes a full-matrix marker or a shared
summary.

### Merge

Add a command:

```bash
python -m validation_v2.cli merge-shards \
  --config configs/validation_v2/server_full.yaml \
  --plan plan.json \
  --shards-root results/validation_v2/server_full-<commit>/shards \
  --output-root results/validation_v2/server_full-<commit>-merged
```

The merge target must not exist. Merge first performs a read-only preflight:

- every planned shard exists and is completed;
- all shard plan/config/commit/device digests agree;
- group IDs and combination IDs are disjoint and cover the full plan exactly;
- run IDs are unique and match their directories;
- every run has all formal artifacts and passes existing provenance, checkpoint,
  test-ledger, metric, split, and scaler checks;
- content-addressed split/scaler files with the same name have identical bytes;
- no unrecognized run directory or partial marker is present.

After preflight, merge copies or hard-links artifacts into a temporary sibling
directory. Hard links may be used only when source and destination are on the
same filesystem; otherwise byte copies are used. Every promoted file is hashed
after placement. The merge builds a full `matrix_execution.json` containing all
175 run IDs and 4,095 selected combination IDs, with `partial: false` and
`status: completed`.

The existing strict artifact validator runs against the temporary root. Only a
zero exit result permits an atomic directory rename to the requested final
root. A failed merge leaves the shard roots unchanged and preserves the
temporary root under an explicitly diagnostic name.

### Serial compatibility

`python -m validation_v2.cli matrix` continues to use the same shared group
enumerator and executes every group in order. Its output schema and scientific
behavior remain unchanged. The old `--max-combinations` diagnostic remains
partial-only and is not accepted as a shard.

## Concurrency policy for the RTX 4090 D server

The first real-data pilot uses two concurrent shards. Monitor aggregate GPU
utilization, GPU memory, host memory, load average, group wall time, and failures
for at least one completed group per worker. Scale to four workers only if:

- no OOM, CUDA determinism, ledger, or artifact error occurs;
- aggregate completed groups per hour improves by at least 50%;
- median per-group wall time grows by less than 80%;
- GPU memory remains below 80% and host resources remain healthy.

Scale from four to eight workers using the same criteria. If throughput stops
improving, retain the lower worker count. Worker count affects scheduling only;
it is excluded from model provenance but recorded in shard execution metadata.

`CUBLAS_WORKSPACE_CONFIG=:4096:8` remains mandatory for every worker. Network
acceleration is useful for cloning or dependency downloads but does not affect
training throughput.

## Failure handling

- Duplicate group or combination ID: reject before any merge write.
- Missing shard or group: reject and list exact missing IDs.
- Config, commit, device, or plan mismatch: reject.
- Partial run artifacts: reject; never infer completion from `best.pt` alone.
- Existing merge destination: reject without modification.
- Interrupted merge: shard roots remain authoritative and unchanged.
- Worker failure: mark only that shard failed; other shards continue.
- CUDA OOM: stop scaling concurrency. Do not change scientific axes or epochs.

## Test strategy

### Unit tests

- Stable enumeration produces 175 unique groups and 4,095 unique cells.
- Group IDs are invariant to shard count.
- Round-robin plans are deterministic, disjoint, and exhaustive.
- Invalid shard counts/indices and plan/config mismatches are rejected.
- Serial and sharded group configurations are byte-equivalent after canonical
  serialization.

### Shard execution tests

- Two shards execute disjoint complete groups.
- Completed shard rerun is idempotent.
- Partial checkpoint and partial test ledger are rejected.
- A shard never writes `matrix_execution.json` or a shared summary.

### Merge tests

- A complete two-shard smoke matrix merges and passes the strict validator.
- Duplicate, missing, foreign, failed, partial, or tampered inputs are rejected.
- Split/scaler filename collisions with different bytes are rejected.
- Existing destination is never overwritten.
- A forced copy failure leaves source shards unchanged and no formal final root.

### Server acceptance

1. Run the full local test suite.
2. Run a real OxIOD two-shard CUDA pilot with one complete group per worker.
3. Validate each shard and the merged pilot.
4. Compare one serial and one sharded group for identical run ID, config hash,
   split hash, scaler hash, checkpoint hash, and metrics-file hash.
5. Scale concurrency gradually according to the 4090 D policy.
6. Start the fresh formal sharded root only after all acceptance checks pass.

## Rollout and preservation

- Do not alter or delete `server_full-fcf81f8`.
- Develop shard support on a new commit and use commit-qualified shard and final
  roots.
- Keep the serial process running during implementation and pilot validation.
- Once the sharded pilot passes, stop the serial process without deleting its
  completed artifacts; record its final status in the audit directory.
- The paper may use only the fresh merged root after strict validation and
  formal summarization complete successfully.

## Success criteria

- All local tests pass.
- The two-shard real-data CUDA pilot passes and produces identical artifacts for
  an equivalent serial group.
- The formal plan covers exactly 175 groups and 4,095 cells.
- All formal shards complete without scientific configuration changes.
- Merge and strict artifact validation exit zero.
- Aggregate wall time is materially lower than the measured serial estimate.
