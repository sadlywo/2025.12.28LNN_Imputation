# Validation v2 服务器正式验证中文操作手册

本文用于在 Linux + RTX 4090D 服务器上运行论文级 server_full 正式验证。
它是面向操作者的中文指南；可执行 Bash helper 的唯一维护版本仍是
[英文运行手册](validation_v2_server_runbook.md)。两份文档发生冲突时，以英文
运行手册中的命令与检查为准。

> 不要把本文所有代码块一次性粘贴执行。必须按编号逐段执行；前一段的断言
> 失败时，先解决问题，不要跳到下一段。

## 0. 本次运行做什么

正式验证不是一次普通训练，而是：

1. 在干净、固定的 Git 提交和固定 Python 环境中生成一个不可变的 8 分片计划；
2. 执行 175 个训练组，共 4,095 个实验单元；每个分片有 21 或 22 组；
3. 先以 1 个、再以 2 个、4 个、8 个 worker 分阶段放量；
4. 仅在所有分片成功完成后，严格合并、复核并汇总五个随机种子。

旧的串行结果 results/validation_v2/server_full-fcf81f8 是诊断证据，绝不能
删除、复制到新分片目录、参与 merge，或和新结果一起 summarize。

## 1. 前置条件

- 服务器为 Linux，GPU 为 RTX 4090D，CUDA 可由 PyTorch 2.3.1+cu121 使用；
- Conda 环境位于 /root/miniconda3/envs/pinn_imu；
- 仓库位于 /root/autodl-tmp/2025.12.28LNN_Imputation；
- /root/autodl-tmp 至少有 100 GiB 可用空间；
- 当前要执行的 Git 提交已经过本地验证并由操作者明确指定；
- 网络加速只允许用于 git clone/fetch 或安装依赖的短子 shell；训练 shell 中
  不要执行 source /etc/network_turbo。

长时限是故障安全上限，不是耗时承诺：单 shard 最多 7 天、全部 shard 最多
14 天；任一 shard 6 小时没有新的完整训练组，或全部 shard 6 小时都无新完成组，
流程会失败退出以便诊断。

## 2. 获取并固定代码

以下命令中的 VALIDATED_COMMIT 必须替换为本次交付的 40 位提交 SHA。若仓库已经
存在，使用 fetch + detached checkout；不要用 git pull 把未验证修改混入正式运行。

~~~bash
export REPO=/root/autodl-tmp/2025.12.28LNN_Imputation
export VALIDATED_COMMIT='<40 位、已验证的提交 SHA>'

cd "$REPO"
(
  source /etc/network_turbo
  git fetch origin codex/validation-v2
)
git checkout --detach "$VALIDATED_COMMIT"
test "$(git rev-parse HEAD)" = "$VALIDATED_COMMIT"
test -z "$(git status --porcelain)"
git show -s --format='%H %cI %s' HEAD
~~~

如果服务器尚未有仓库，先在 /root/autodl-tmp 克隆 origin，再执行上面的 fetch
和 detached checkout。不要把凭据、密码或 token 粘贴到命令历史、日志或仓库文件中。

## 3. 进入离线训练 shell 并定义本次运行目录

从这里开始的 shell 不应再 source Network Turbo。

~~~bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/miniconda3/envs/pinn_imu
set -Eeuo pipefail

export REPO=/root/autodl-tmp/2025.12.28LNN_Imputation
cd "$REPO"
export COMMIT="$(git rev-parse HEAD)"
export PREFLIGHT_DIR="/root/autodl-tmp/validation-v2-preflight-${COMMIT}-sharded-v2"
export AUDIT_DIR="/root/autodl-tmp/validation-v2-audit-${COMMIT}-sharded-v2"
export PLAN="$AUDIT_DIR/server-full-8-shards-${COMMIT}.json"
export SHARDS_ROOT="$REPO/results/validation_v2/server-full-shards-${COMMIT}-sharded-v2"
export FINAL_ROOT="$REPO/results/validation_v2/server-full-final-${COMMIT}-sharded-v2"
export CONFIG="$REPO/configs/validation_v2/server_full.yaml"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

test -z "$(git status --porcelain)"
test -f "$CONFIG"
test ! -e "$PREFLIGHT_DIR"
test ! -e "$AUDIT_DIR"
test ! -e "$SHARDS_ROOT"
test ! -e "$FINAL_ROOT"
mkdir -p "$PREFLIGHT_DIR" "$AUDIT_DIR" "$SHARDS_ROOT"
printf '%s\n' "$COMMIT" > "$AUDIT_DIR/COMMIT"
~~~

若任何 test ! -e 失败，说明该提交已有同名正式活动或旧残留。不要复用、覆盖
或删除它；请更换新的 campaign 后缀并重新生成计划，或先诊断原活动。

## 4. 环境和测试门槛

完整执行英文运行手册第 3 节的环境校验块。它会检查 Python 3.9、包版本、
CUBLAS_WORKSPACE_CONFIG、CUDA 12.1、4090D、磁盘空间以及 GPU 可用性。

随后运行下面两个测试门槛：第一个必须在 Linux 上显示 1 passed，出现 skipped
也必须停止；第二个是完整测试套件。

~~~bash
test "$(uname -s)" = Linux
python -m pytest -q \
  tests/validation_v2/test_sharding.py::test_linux_rename_noreplace_survives_real_directory_race \
  -rs | tee "$PREFLIGHT_DIR/linux-renameat2.txt"
test "${PIPESTATUS[0]}" -eq 0
grep -F '1 passed' "$PREFLIGHT_DIR/linux-renameat2.txt"
! grep -qi 'skipped' "$PREFLIGHT_DIR/linux-renameat2.txt"

python -m pytest -q | tee "$PREFLIGHT_DIR/pytest-full.txt"
test "${PIPESTATUS[0]}" -eq 0
cp "$PREFLIGHT_DIR/linux-renameat2.txt" "$PREFLIGHT_DIR/pytest-full.txt" "$AUDIT_DIR/"
~~~

失败时不要开始训练。优先检查是否真的在 pinn_imu 环境、是否已固定到指定提交，
以及 GPU / CUDA / PyTorch 版本是否与英文手册一致。

## 5. 生成并核对不可变分片计划

~~~bash
python -m validation_v2.cli matrix --config "$CONFIG" --dry-run \
  > "$AUDIT_DIR/matrix-dry-run.jsonl"
python - "$AUDIT_DIR/matrix-dry-run.jsonl" <<'PY'
import json
import sys

lines = open(sys.argv[1], encoding="utf-8").read().splitlines()
assert json.loads(lines[0])["combination_count"] == 4095
assert len(lines) == 4096
PY

python -m validation_v2.cli shard-plan \
  --config "$CONFIG" --shard-count 8 --output "$PLAN" --device cuda \
  | tee "$AUDIT_DIR/shard-plan.stdout.json"
python - "$PLAN" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
assert plan["schema_version"] == 2
assert plan["total_groups"] == 175
assert plan["total_cells"] == 4095
assert plan["shard_count"] == 8
assert [len(item["group_ids"]) for item in plan["shards"]] == [22, 22, 22, 22, 22, 22, 22, 21]
assert plan["dirty_state_digest"] == ""
PY
~~~

此后不要修改已 checkout 的代码、Python 包或计划文件。计划会绑定 Git commit、
干净工作树摘要和运行时指纹；任一漂移都会使 shard 执行或 merge 失败。

## 6. 分阶段启动训练

英文运行手册第 5 节定义了 launch_shard、wait_until_groups、
wait_stage_metrics、wait_shard、wait_all_shards、run_queue 和 GPU 采样等 Bash
helper。完整复制该节的函数块到当前离线 shell，不要手写简化版。

随后按英文手册第 6、7 节顺序执行：

1. launch_shard 000，等它完成至少两个 group，建立单 worker 基线；
2. 启动 001，只有 000 和 001 都在本阶段开始后各完成至少一个 group，才能通过
   两 worker 门槛；
3. 若通过，启动 002、003；四个 active shard 都各完成一个新 group 后才能继续；
4. 若再通过，启动 004 到 007；八个 active shard 都必须各完成一个新 group；
5. 若仅性能或显存门槛失败，手册会进入安全的串行或两并发 fallback；若 marker
   失败、PID 异常或采样无进展，立即退出而不是强行扩容；
6. 最终用 wait_all_shards 000 001 002 003 004 005 006 007 等待全部完成。

放量门槛的要求是：吞吐相对前一稳定阶段至少 1.5 倍、组耗时中位数低于基线的
1.8 倍、峰值显存低于总显存的 80%，且没有失败 marker。它们是保护 4090D
单卡稳定性的门槛，不是为了盲目八进程并发。

## 7. 中断、恢复和失败处理

- started 的 shard 只能在原 commit、原 config、原 plan、原 device、原 shard
  index 与原 shard root 下恢复；恢复从最后一个完整 group 后开始；
- completed shard 可重跑，行为应为幂等；
- failed marker、部分写入的 group 或未知 PID 不能直接恢复；保留现场供诊断；
- 不要把替代目录重命名为 000–007，也不要把不同 campaign 的 shard 混入同一
  plan；发生失败时最安全的做法是使用新的 campaign 后缀，重新生成计划并重跑
  全部 8 个 shard；
- 不要 kill -9 正常训练进程。手册的 PID 身份检查和 SIGINT 流程会保留可恢复边界。

## 8. 合并、严格验证和汇总

只有八个 shard_execution.json 都显示 completed 时才可执行：

~~~bash
test ! -e "$FINAL_ROOT"
python -m validation_v2.cli merge-shards \
  --config "$CONFIG" \
  --plan "$PLAN" \
  --shards-root "$SHARDS_ROOT" \
  --output-root "$FINAL_ROOT" \
  | tee "$AUDIT_DIR/merge.stdout.json"

python -m validation_v2.experiments.validate_artifacts \
  --root "$FINAL_ROOT" \
  --config "$CONFIG" \
  | tee "$AUDIT_DIR/validate.stdout.json"

python - "$FINAL_ROOT/validation_report.json" <<'PY'
import json
import sys

assert json.load(open(sys.argv[1], encoding="utf-8"))["status"] == "complete"
PY

python -m validation_v2.cli summarize \
  --root "$FINAL_ROOT" \
  --config "$CONFIG" \
  --required-seeds 2026 2027 2028 2029 2030 \
  --baseline linear \
  | tee "$AUDIT_DIR/summarize.stdout.json"
~~~

merge 会拒绝不一致的计划、Git/dirty/runtime provenance、重复或缺失 group、
链接目录、已存在输出目录，以及预检到复制之间发生变化的文件。不要尝试手动
合并 CSV、复制 run 目录或覆盖 FINAL_ROOT。

## 9. 正式交付前检查

最终应保留：

- AUDIT_DIR 下的 COMMIT、干净 Git 状态、计划 JSON、GPU 采样和 shard 日志；
- 八个独立 shard 根目录及其 shard_execution.json；
- FINAL_ROOT 下状态为 complete 的 validation_report.json；
- FINAL_ROOT 下的 summary.csv 和 summary.json，覆盖种子 2026–2030；
- 内容寻址的 split manifest、scaler，以及每个 run 的 provenance、checkpoint、
  test ledger 和逐记录指标。

在打包前再次运行：

~~~bash
cd "$REPO"
test -z "$(git status --porcelain)"
test -f "$FINAL_ROOT/validation_report.json"
test -f "$FINAL_ROOT/summary.csv"
test -f "$FINAL_ROOT/summary.json"
~~~

不要在仓库、审计日志或打包结果中记录 SSH 密码、token、仓库凭据或完整环境转储。
