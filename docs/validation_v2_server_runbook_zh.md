# Validation v2 服务器正式验证操作手册（MatPool）

本手册是论文级 `server_full` 正式验证的当前中文运维入口。支持环境为 Linux、
通用 CPython 3.10–3.12，以及一张或多张 RTX 4090、4090D 或 5090 GPU。
各 Python 小版本均使用锁定的 `torch==2.11.0+cu128` 与同一组验证依赖。
当前矩池云目标配置是两张 RTX 5090；单张 RTX 4090 仍可用于预检和单 worker 执行。

`configs/validation_v2/server_full.yaml` 已切换为 OxIOD、EuRoC MAV 与 IDOL
三数据集联合正式协议。三套数据独立执行记录级 80/10/10 划分和训练集专属 robust
scaler，再按等额窗口预算进入同一共享模型。当前正式计划为 25 个训练组、585 个评估单元。

底层通用运行器负责科学执行合同：校验精确提交与干净工作树，执行完整预检，生成
不可变的 8-shard 计划，训练全部分片，再严格合并、验证并汇总五个随机种子。
MatPool 启动器只是该合同的后台运维封装，不会跳过测试、计划或完成检查。

## 1. clone 后直接启动

首次进入服务器时，只克隆当前开发分支；数据集和生成结果不会通过 Git 下载：

```bash
mkdir -p /root/workspace
cd /root/workspace
git clone --depth 1 --branch codex/physics-loss-refactor \
  git@github.com:sadlywo/2025.12.28LNN_Imputation.git lnn-imputation
cd /root/workspace/lnn-imputation
```

仓库已经 clone 到服务器后，必须在任何 `start` 之前绑定本次审阅通过的 40 位 commit。
下面的 HEAD 相等断言和干净工作树门禁不可省略；启动器还会再次执行相同检查。

**付费资源警告：**关闭或断开 SSH 不会停止后台 tmux、训练或计费；本启动器不提供
`stop` 命令。执行 `start` 前必须确认预算与 MatPool 平台租用时间。需要终止计费时，
必须使用平台核验的停机或实例释放流程，并在平台侧确认最终状态；不能把直接结束某个
shard 进程当作计费已经终止的证据。

```bash
set -Eeuo pipefail
cd /root/workspace/lnn-imputation
VALIDATED_COMMIT="<40-HEX-VALIDATED-COMMIT>"
git checkout --detach "$VALIDATED_COMMIT"
test "$(git rev-parse HEAD)" = "$VALIDATED_COMMIT"
test -z "$(git status --porcelain)"
bash scripts/run_validation_v2_matpool.sh start
bash scripts/run_validation_v2_matpool.sh status
bash scripts/run_validation_v2_matpool.sh logs
```

`start` 创建后台 tmux session 后即返回。tmux 中会先执行完整 preflight，成功后才
开始训练，因此启动命令返回 0 不代表预检或正式验证已经完成。完整运行可能持续数日。

默认最大同时 worker 数为 2，但完整计划仍会运行全部 8 个分片；这个参数只限制并发，
不会减少覆盖范围。每个 shard 进程通过 `CUDA_VISIBLE_DEVICES` 轮转绑定到独立 GPU，
因此两张卡各运行一个 worker。worker 数不能超过当前可见 GPU 数；单卡 4090 主机请使用
`bash scripts/run_validation_v2_matpool.sh start --max-workers 1`。

## 2. 状态、日志与证据位置

```bash
bash scripts/run_validation_v2_matpool.sh status
bash scripts/run_validation_v2_matpool.sh logs
```

`status` 报告 tmux 是 active 还是 inactive，并打印 commit、campaign suffix、
最大 worker 数、审计目录、8-shard 根目录、final 根目录、日志和退出状态文件。
`logs` 持续跟随当前 campaign 日志。

启动器的私有状态位于仓库内 `.validation-v2-matpool/`：`current.json` 记录当前
session 与各证据路径，`run-*.log` 保存合并输出，`run-*.exit` 保存最终退出码。
commit-qualified 的 `validation-v2-audit-*` 位于仓库同级目录；分片与 final 根目录
以 `status` 实际输出为准。启动器故意不提供 `stop` 命令。不要手工删除 state、
campaign seal、日志或分片根目录，也不要向未核验身份的进程发送信号。

## 3. 预检-only 与通用运行器

使用通用运行器时必须二选一：（A）直接 full；或（B）先做独立 preflight，再以
不同 campaign suffix 执行 full。两条路径都固定精确当前 commit；`--max-workers`
只控制并发，仍执行完整 8-shard 计划。

路径 A：不单独运行预检，直接 full：

```bash
DIRECT_SUFFIX="formal-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/run_validation_v2_server.sh --commit "$(git rev-parse HEAD)" --mode full \
  --campaign-suffix "$DIRECT_SUFFIX" --max-workers 2
```

路径 B：先做诊断预检，再启动 formal full。预检会创建不可变 campaign seal，
因此两个 suffix 必须分别命名。只有预检已在同一服务器成功创建并验证
`.venv-server/bin/python`，后续 full 才能重用依赖：

```bash
PREFLIGHT_SUFFIX="preflight-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/run_validation_v2_server.sh --commit "$(git rev-parse HEAD)" --mode preflight \
  --campaign-suffix "$PREFLIGHT_SUFFIX"

FORMAL_SUFFIX="formal-$(date -u +%Y%m%dT%H%M%SZ)"
test "$PREFLIGHT_SUFFIX" != "$FORMAL_SUFFIX"
bash scripts/run_validation_v2_server.sh --commit "$(git rev-parse HEAD)" --mode full \
  --campaign-suffix "$FORMAL_SUFFIX" --max-workers 2 --skip-dependency-install
```

依赖重用不会跳过精确提交、干净工作树、运行时、完整 pytest、计划、训练、合并或验证。
MatPool wrapper 的等价重用形式是
`bash scripts/run_validation_v2_matpool.sh start --skip-dependency-install`。它只允许在
同一主机成功 provision 依赖之后、并再次执行主代码块的 exact-commit 与干净工作树
门禁后使用；若解释器不存在或不合格，启动会失败关闭。

## 4. 失败诊断

先运行 `status`，再用 `logs` 定位第一条 preflight、依赖、测试、GPU 或 runner 错误。
如果 tmux 已退出，读取 `status` 报告的 `run-*.exit`，并保留 `run-*.log`、
`current.json`、审计目录、PID/GPU 采样、每个 `shard_execution.json` 和所有已有分片。
失败 campaign 的 seal 不得复用；修复原因后从干净提交启动一个新 campaign。

不要把启动成功、日志暂时无输出、部分 shard 完成或手工拼接 CSV 当作成功。不得用历史
结果补齐缺失分片，也不得覆盖已有 final 根目录。

## 5. 完成判据

只有以下条件全部成立，才能判定正式验证完成：

- `status` 显示 tmux 已 inactive，且 `run-*.exit` 内容为 `0`；
- 八个独立 shard 根目录均有状态为 `completed` 的 `shard_execution.json`；
- 新 final 根目录中的 `validation_report.json` 状态为 `complete`；
- `summary.csv` 与 `summary.json` 覆盖随机种子 2026、2027、2028、2029、2030；
- runtime provenance、计划 JSON、GPU 采样、shard 日志与 merge/validate/summarize
  输出均保留供审计。

## 历史资料（仅供审计）

旧的单进程结果和英文手册后半部分的手工命令仅用于审计与事故分析，属于
**Historical only**。它们不是当前入口，不得与上述 MatPool 或通用运行器命令混用。
英文历史合同见 [validation_v2_server_runbook.md](validation_v2_server_runbook.md)。
