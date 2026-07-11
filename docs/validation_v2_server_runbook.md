# Validation v2 server runbook

This runbook is the handoff contract for the paper-grade `server_full` matrix. A
partial matrix or smoke run is diagnostic only and must never be summarized or
used in the paper.

## 1. Connect and verify the host

Connect from the local machine:

```bash
ssh-keyscan -p 10274 connect.westb.seetacloud.com 2>/dev/null | ssh-keygen -lf -
ssh -p 10274 root@connect.westb.seetacloud.com
```

`ssh-keyscan` is not itself a trusted authentication channel. In OpenSSH,
`ssh-keygen -lf -` must report
`256 SHA256:liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc ... (ED25519)`; the
middle key comment may identify the host. PuTTY/plink displays the same fixed
hash as
`ssh-ed25519 255 SHA256:liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc`.
The OpenSSH `256` and PuTTY/plink `255` are tool-specific displays, so the full
strings are not expected to be identical; the SHA-256 hash must match exactly.
Accept the first-host prompt only when that hash matches, and stop otherwise.
When SSH asks for the password, use
`<enter interactively at the SSH prompt; do not save>`. Never place the
password in this file, shell history, logs, a config, or the result archive.

## 2. Freeze the source revision

The repository path on the server is
`/root/autodl-tmp/2025.12.28LNN_Imputation`. Network Turbo is for the Git
clone/fetch step only. It can interfere with or slow other `pip` traffic, so
source it in the Git shell and use a new shell (or disable it) before installing
packages.

For a missing checkout:

```bash
cd /root/autodl-tmp
source /etc/network_turbo
git clone <REPOSITORY_URL> 2025.12.28LNN_Imputation
exit
```

For an existing checkout, start a temporary Git shell and fetch:

```bash
cd /root/autodl-tmp/2025.12.28LNN_Imputation
source /etc/network_turbo
git fetch --all --tags --prune
exit
```

Reconnect without sourcing Network Turbo, then check out the exact validated
commit. Replace the placeholder once and retain the value for every later
command:

```bash
cd /root/autodl-tmp/2025.12.28LNN_Imputation
export COMMIT=<VALIDATED_COMMIT>
git checkout --detach "$COMMIT"
test "$(git rev-parse HEAD)" = "$COMMIT"
test -z "$(git status --porcelain)"
```

Do not run from `main`, and do not copy, merge, or summarize results produced by
an older `main` checkout. Use the commit-qualified output root below so old
results remain outside this execution.

## 3. Verify the pinned environment and GPU

Use the existing conda environment at `/root/miniconda3/envs/pinn_imu`:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/miniconda3/envs/pinn_imu
cd /root/autodl-tmp/2025.12.28LNN_Imputation
python --version
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1
python -m pip install -r requirements-validation-v2.txt
python - <<'PY'
import importlib.metadata as md
import platform
import torch

expected = {
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "scipy": "1.13.1",
    "PyYAML": "6.0.3",
    "pytest": "8.4.2",
    "torch": "2.3.1+cu121",
    "ncps": "1.0.1",
}
actual = {name: md.version(name) for name in expected}
assert platform.python_version().startswith("3.9."), platform.python_version()
assert actual == expected, (actual, expected)
assert torch.cuda.is_available()
assert torch.version.cuda == "12.1", torch.version.cuda
assert "4090 D" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print(platform.python_version(), actual)
print(torch.version.cuda, torch.cuda.get_device_name(0))
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

The expected platform is Python 3.9, PyTorch `2.3.1+cu121`, CUDA 12.1, and an
NVIDIA RTX 4090 D. Stop if Python, any pin, CUDA availability, or the GPU model
does not match. If Network Turbo is still active, close that shell and install
from a new shell.

## 4. Discover data and freeze the dry-run plan

The OxIOD data directory is `Oxford Dataset` below the repository. Discovery is
read-only and must find exactly 45 IMU/Vicon pairs:

```bash
cd /root/autodl-tmp/2025.12.28LNN_Imputation
python - <<'PY'
from pathlib import Path
from validation_v2.experiments.runner import discover_oxiod_pairs

pairs = discover_oxiod_pairs(Path("Oxford Dataset"))
print(f"discovered_pairs={len(pairs)}")
assert len(pairs) == 45, "expected exactly 45 OxIOD IMU/Vicon pairs"
PY
```

Freeze the full matrix plan and its hash before training. Keep audit files
outside the repository so the checked-out source remains clean:

```bash
export AUDIT_DIR="/root/autodl-tmp/validation-v2-audit-${COMMIT}"
export ACTIVE_CONFIG="$PWD/configs/validation_v2/server_full.yaml"
mkdir -p "$AUDIT_DIR"
python -m validation_v2.cli matrix \
  --config "$ACTIVE_CONFIG" \
  --dry-run | tee "$AUDIT_DIR/matrix_plan.txt"
sha256sum "$AUDIT_DIR/matrix_plan.txt" | tee "$AUDIT_DIR/matrix_plan.sha256"
head -n 1 "$AUDIT_DIR/matrix_plan.txt"
```

The current config is expected to enumerate 4,095 cells: 3,780 missingness
cells plus 315 timestamp-irregularity cells. Gate variants share training, so
the current runner expects 175 training groups. Treat the dry-run output at the
validated commit as authoritative; investigate rather than editing axes if its
counts differ.

## 5. Run and safely resume the full matrix

The validated config already uses the conservative `batch_size: 32`. Do not
change seeds, rates, topologies, protocols, epochs, or `--max-combinations`.
Define one fresh, commit-qualified output root:

```bash
export RESULT_ROOT="results/validation_v2/server_full-${COMMIT}"
export AUDIT_DIR="/root/autodl-tmp/validation-v2-audit-${COMMIT}"
export RUN_LOG="$AUDIT_DIR/server_full-${COMMIT}.log"
export ACTIVE_CONFIG="$PWD/configs/validation_v2/server_full.yaml"
test ! -e "$RESULT_ROOT"
```

Run inside `tmux` (recommended):

```bash
tmux new -s validation-v2
cd /root/autodl-tmp/2025.12.28LNN_Imputation
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/miniconda3/envs/pinn_imu
export COMMIT=<VALIDATED_COMMIT>
export RESULT_ROOT="results/validation_v2/server_full-${COMMIT}"
export AUDIT_DIR="/root/autodl-tmp/validation-v2-audit-${COMMIT}"
export RUN_LOG="$AUDIT_DIR/server_full-${COMMIT}.log"
export ACTIVE_CONFIG="$PWD/configs/validation_v2/server_full.yaml"
set -o pipefail
python -m validation_v2.cli matrix \
  --config "$ACTIVE_CONFIG" \
  --output-root "$RESULT_ROOT" \
  --device cuda 2>&1 | tee -a "$RUN_LOG"
```

Detach with `Ctrl-b d`. A `nohup` alternative is:

```bash
cd /root/autodl-tmp/2025.12.28LNN_Imputation
export COMMIT=<VALIDATED_COMMIT>
export ACTIVE_CONFIG="$PWD/configs/validation_v2/server_full.yaml"
export RESULT_ROOT="results/validation_v2/server_full-${COMMIT}"
export AUDIT_DIR="/root/autodl-tmp/validation-v2-audit-${COMMIT}"
export RUN_LOG="$AUDIT_DIR/server_full-${COMMIT}.log"
mkdir -p "$AUDIT_DIR"
nohup env ACTIVE_CONFIG="$ACTIVE_CONFIG" RESULT_ROOT="$RESULT_ROOT" RUN_LOG="$RUN_LOG" \
  bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/miniconda3/envs/pinn_imu && cd /root/autodl-tmp/2025.12.28LNN_Imputation && set -o pipefail && python -m validation_v2.cli matrix --config "$ACTIVE_CONFIG" --output-root "$RESULT_ROOT" --device cuda 2>&1 | tee -a "$RUN_LOG"' \
  >"$AUDIT_DIR/nohup.out" 2>&1 &
```

After an interruption, safe resume means running the exact same full command
with the same commit, config, output root, and device. The runner reuses only
content-matching completed groups. Never add `--max-combinations` to a resume.

Monitor without modifying the run:

```bash
watch -n 2 nvidia-smi
tail -n 100 -f "$RUN_LOG"
watch -n 30 'df -h /root/autodl-tmp; du -sh "$RESULT_ROOT"'
```

If CUDA reports OOM, do not alter seeds, rates, topologies, protocols, or
epochs. The only permitted fallback is a lower `batch_size` in a copied config.
That changes provenance and run IDs, so start a new output root and never mix it
with the original config's artifacts. Resume within a group requires the same
resolved config; this is why beginning with batch 32 is strongly preferred.

```bash
export ACTIVE_CONFIG="/root/autodl-tmp/server_full-${COMMIT}.yaml"
cp "$PWD/configs/validation_v2/server_full.yaml" "$ACTIVE_CONFIG"
sed -i 's/^batch_size: 32$/batch_size: 16/' "$ACTIVE_CONFIG"
grep '^batch_size:' "$ACTIVE_CONFIG"
export RESULT_ROOT="results/validation_v2/server_full-${COMMIT}-batch16"
export RUN_LOG="$AUDIT_DIR/server_full-${COMMIT}-batch16.log"
# Preserve the original root and run/resume this exact replacement command.
set -o pipefail
python -m validation_v2.cli matrix \
  --config "$ACTIVE_CONFIG" \
  --output-root "$RESULT_ROOT" \
  --device cuda 2>&1 | tee -a "$RUN_LOG"
```

## 6. Validate before formal summarization

Never summarize a partial marker. A run using `--max-combinations 1` is only a
diagnostic smoke/partial run and cannot support the paper. For the full root,
run the artifact validator first:

```bash
python -m validation_v2.experiments.validate_artifacts \
  --root "$RESULT_ROOT" \
  --config "$ACTIVE_CONFIG"
test -f "$RESULT_ROOT/validation_report.json"
```

Only after the validator exits zero and writes a `status: complete` report may
the formal summary run:

```bash
python -m validation_v2.cli summarize \
  --root "$RESULT_ROOT" \
  --config "$ACTIVE_CONFIG" \
  --baseline linear
```

For a deliberately bounded smoke root without `matrix_execution.json`, the
validator requires `--allow-smoke` and a `smoke_summary.json` declaring
`descriptive_only: true`; that report remains descriptive and is not a formal
summary.

The expected full tree is:

```text
server_full-<VALIDATED_COMMIT>/
├── matrix_execution.json
├── split_manifest-<sha256>.csv
├── scaler-<sha256>.json
├── <run_id>/
│   ├── run.json
│   ├── history.json
│   ├── best.pt
│   ├── checkpoint.json
│   ├── test_evaluation.json
│   └── per_record_metrics.csv
├── validation_report.json
├── summary.csv
└── summary.json
```

## 7. Package and download the audited handoff

Build the handoff directory outside both the repository and results root.
Before creating it, capture the repository status and require it to be empty.
Include results, the exact config, commit identity, frozen plan/hash, validation
report, and log. Do not include `Oxford Dataset`, any SSH material, passwords,
tokens, or other secrets.

```bash
export REPO="/root/autodl-tmp/2025.12.28LNN_Imputation"
cd "$REPO"
SOURCE_STATUS="$(git status --porcelain)"
test -z "$SOURCE_STATUS"
export HANDOFF="/root/autodl-tmp/validation-v2-handoff-${COMMIT}"
test ! -e "$HANDOFF"
mkdir -p "$HANDOFF/config"
printf '%s' "$SOURCE_STATUS" > "$HANDOFF/git_status_porcelain.txt"
cp "$ACTIVE_CONFIG" "$HANDOFF/config/executed_config.yaml"
cp "$AUDIT_DIR/matrix_plan.txt" "$AUDIT_DIR/matrix_plan.sha256" "$RUN_LOG" "$HANDOFF/"
git rev-parse HEAD > "$HANDOFF/validated_commit.txt"
cp "$REPO/$RESULT_ROOT/validation_report.json" "$HANDOFF/"
cp -a "$REPO/$RESULT_ROOT" "$HANDOFF/results"
export ARCHIVE="$(basename "${HANDOFF}.tar.gz")"
tar -C /root/autodl-tmp -czf "/root/autodl-tmp/$ARCHIVE" "$(basename "$HANDOFF")"
(
  cd /root/autodl-tmp
  sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
)
```

Download from the local machine:

```bash
scp -P 10274 \
  root@connect.westb.seetacloud.com:/root/autodl-tmp/validation-v2-handoff-<VALIDATED_COMMIT>.tar.gz \
  .
scp -P 10274 \
  root@connect.westb.seetacloud.com:/root/autodl-tmp/validation-v2-handoff-<VALIDATED_COMMIT>.tar.gz.sha256 \
  .
sha256sum -c validation-v2-handoff-<VALIDATED_COMMIT>.tar.gz.sha256
```

Enter `<enter interactively at the SSH prompt; do not save>` only at the
interactive SSH prompt. The final command must report `OK` before analysis.
