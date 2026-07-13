# Validation v2 服务器正式验证操作手册（Python 3.12）

本手册适用于 Linux + RTX 4090D 服务器上的论文级 `server_full` 正式验证。
当前唯一推荐入口是仓库内的 Python 3.12 运行器；不要把旧的手工 Conda、复制
helper 或逐条 shard 命令与它混用。英文完整历史契约见
[validation_v2_server_runbook.md](validation_v2_server_runbook.md)，但以本文件的
“当前执行路径”为准。

## 1. 运行器会做什么

运行器在 `REPO/.venv-server` 创建项目内 Python 3.12 虚拟环境，并显式安装
`torch==2.3.1+cu121` 和锁定的验证依赖。它随后会记录 Python、包版本、CUDA 和
GPU 的运行时 provenance，执行 Linux 原子写入竞态测试与完整 pytest，生成不可变的
8-shard 计划（175 个训练组、4,095 个实验单元），并在 full 模式执行
1 -> 2 -> 4 -> 8 分阶段放量、严格合并、产物验证和五随机种子汇总。

正式 full 运行可能持续数日。日志、`AUDIT_DIR`、分片目录、PID 和 GPU 采样文件
均是审计证据；失败时保留现场诊断，不要删除、覆盖或拼接结果。

旧的
`results/validation_v2/server_full-fcf81f8` 仅是历史诊断证据。不得将其复制到新
campaign，不得作为 merge 输入，也不得与新结果一起 summarize。

## 2. 克隆、固定提交并直接启动正式验证

网络加速只用于短暂的 clone/fetch/依赖安装操作，训练本身不需要网络。
克隆完成后，必须固定到本次交付的 40 位提交 SHA，且工作树必须干净：

```bash
export REPO=/root/autodl-tmp/2025.12.28LNN_Imputation
cd "$REPO"
git checkout --detach "<40-HEX-VALIDATED-COMMIT>"
test -z "$(git status --porcelain)"
git show -s --format='%H %cI %s' HEAD
```

对于正常的正式验证，直接运行 full。它自身包含全部 preflight，因此这是唯一
推荐的生产命令：

```bash
bash scripts/run_validation_v2_server.sh --commit "$(git rev-parse HEAD)" --mode full
```

不要在训练 shell 中执行 `source /etc/network_turbo`。如需在首次运行前加速下载
依赖，请只让安装命令处于一个短暂的 Network Turbo 子 shell；运行器本身会在
`.venv-server` 内完成 pip 安装。

`--skip-dependency-install` 仅可在同一服务器的 `.venv-server` 已创建且可执行的
`.venv-server/bin/python` 已成功通过运行器运行时校验后使用。若该解释器不存在，
运行器会以状态码 2 退出。它不会跳过 Git、CUDA、完整测试或计划校验：

```bash
bash scripts/run_validation_v2_server.sh --commit "$(git rev-parse HEAD)" --mode full \
  --skip-dependency-install
```

## 3. 可选诊断预检

如需先单独验证环境和计划，可执行：

```bash
bash scripts/run_validation_v2_server.sh --commit "$(git rev-parse HEAD)" --mode preflight
```

预检将创建不可变 campaign seal。预检失败时，不得启动 full；先修复环境、提交、
GPU 或测试问题。预检成功后也不能以同一后缀再次运行 full：每次运行都必须使用不同或新的
`--campaign-suffix`。因此，预检后的正式运行使用一个新的后缀，例如：

```bash
bash scripts/run_validation_v2_server.sh --commit "$(git rev-parse HEAD)" --mode full \
  --campaign-suffix "formal-$(date -u +%Y%m%dT%H%M%SZ)"
```

full 模式会再次执行完整 preflight；不应手工跳过 Linux 原子测试、pytest、计划
生成或 2 -> 4 -> 8 门槛。任一门槛、marker、PID、GPU 采样或 provenance 失败时，
运行器会停止并保留审计材料。

## 4. 结束后的核对

成功完成后，保留以下内容以支持论文审计：

- 父目录中 commit-qualified 的 preflight 与 audit 目录；
- 八个独立 shard 根目录和各自的 `shard_execution.json`；
- 新 final 根目录中的 `validation_report.json`、`summary.csv` 和 `summary.json`；
- runtime provenance、计划 JSON、GPU 采样、shard 日志以及 merge/validate/summarize 输出。

final 报告必须为 `complete`，summary 必须覆盖随机种子 2026、2027、2028、2029、
2030。不要手工合并 CSV、重命名分片目录，或用历史结果填充缺失分片。
