# 现代时间序列插补模型对比实验设计

## 1. 目标

在现有 `validation_v2` 可信验证管线中加入 BRITS、SAITS、CSDI 和
SSSD-S4，并在 MatPool Linux 单机 GPU 服务器上完成可复现、可恢复、可审计的
阶段 A 对比实验。

本次交付必须形成完整闭环：在 Windows 本地打包代码和 OxIOD 数据，上传到
MatPool，创建两个隔离环境，后台调参和训练，统一评估现有模型与现代模型，最后
将摘要结果或完整结果封装下载。本次不复用历史结果 CSV 或旧检查点。

## 2. 已确认的范围

### 2.1 正式主表模型

同一 campaign 从头运行以下九个模型：

1. Linear interpolation；
2. LOCF；
3. BiLSTM；
4. BiLNN；
5. 当前 `HybridImputer`，即 BiLNN、BiLSTM 与 learned gate；
6. BRITS；
7. SAITS；
8. CSDI；
9. SSSD-S4。

`equal_average` 和 `fixed_gate_0/0.5/1` 继续作为现有机制消融保留，但不进入
本次现代模型主表。后续可以通过配置显式开启。

### 2.2 阶段 A 实验矩阵

- 协议：只运行 `strict_file`；
- 正式随机种子：2026、2027、2028、2029、2030；
- 缺失拓扑：point、block、channel；
- 请求缺失率：10%、20%、30%、40%；
- irregular-time case：1 个；
- 每个正式种子训练一次，每个检查点评估 12 个常规缺失条件和 1 个
  irregular-time condition；
- 训练条件固定为 30% point masking；
- 数据窗口长度固定为 `seq_len=30`。

scenario-holdout 不属于本次阶段 A。阶段 A 完成后，可以另行设计阶段 B，让表现
最好的现代模型与当前 Hybrid 进入全部 scenario-holdout 协议。

## 3. 选择的架构

采用“双环境 + 统一数据/预测契约”，由 `validation_v2` 作为唯一控制面：

```text
Validation V2 控制面
  |-- 固定划分、scaler、窗口、掩码和实验计划
  |-- 运行 Linear/LOCF/BiLSTM/BiLNN/Hybrid
  |-- 调用 PyPOTS 环境运行 BRITS/SAITS/CSDI
  |-- 调用 SSSD 环境运行官方 SSSD-S4
  `-- 校验预测包、统一评估、统计、绘图和打包
```

PyPOTS 当前统一提供 BRITS、SAITS 和 CSDI 的插补 API：
<https://docs.pypots.com/en/latest/pypots.imputation.html>。PyPOTS 锁定为 1.5 正式版，
安装包哈希也写入依赖锁。SSSD-S4 使用作者的 MIT 许可官方实现：
<https://github.com/AI4HealthUOL/SSSD>。SSSD 源码快照锁定为提交
`4d3b7a51c54b658945c0ba0bbb26e5ee1f763bed`，并随上传包携带提交身份、许可证和
第三方说明；服务器不会在正式运行时下载浮动的 `main` 分支。

### 3.1 代码和配置边界

新增内容位于以下边界内：

```text
validation_v2/modern/                    # 契约、导出、适配、导入、概率指标
configs/validation_v2/modern_smoke.yaml
configs/validation_v2/modern_tuning.yaml
configs/validation_v2/modern_stage_a.yaml
scripts/run_modern_imputation_matpool.sh # MatPool 统一入口
scripts/package_modern_experiment.ps1    # Windows 上传包
third_party/sssd/                        # 锁定的官方快照、许可证、兼容补丁说明
tests/validation_v2/modern/              # 新增测试
```

SSSD 兼容补丁必须保持最小化，并记录原文件哈希、补丁、补丁原因和补丁后哈希。
不可把不同算法重新实现为一个名称相同但语义不同的本地模型。

## 4. 数据契约与无泄漏约束

### 4.1 唯一数据来源

控制面继续使用现有 `validation_v2` 的：

- 录制文件级 `strict_file` 划分；
- 仅由训练录制拟合的 `RobustTrainScaler`；
- 六通道 IMU 目标；
- 显式时间和正 `dt`；
- 现有 point、block、channel 和 irregular-time 生成器。

所有模型共享完全相同的 split、scaler、窗口、随机种子和预生成 mask。mask 语义
固定为 `1 = observed`、`0 = missing`。外部库所需的缺失值用 `NaN` 表示，但数据包
始终同时保存显式二进制 mask，不能通过检查 `NaN` 重新推断实验身份。

### 4.2 信息预算

每个模型只能接收观测六通道数值、mask 和时间信息。模型可以使用其原生接口支持的
时间输入，但不能接收被遮挡目标或由被遮挡目标计算的统计量。SAITS 等不原生消费
连续时间的模型不会获得额外定制时间网络；这属于模型能力差异，而不是数据泄漏。

数据包至少包含：

- schema version、campaign ID、split hash、scaler hash；
- recording ID、scenario、原始样本索引和物理时间；
- normalized complete target，仅用于训练监督或统一评估；
- observed input、binary mask、`dt`；
- topology、requested fraction、realized fraction 和 mask seed；
- 每个数组的 shape、dtype 和 SHA-256。

训练监督可以使用训练集内人工遮挡位置的完整值。验证和测试目标只由控制面用于选参
或最终评估，不能作为模型输入。

### 4.3 泄漏不变性

保持 observed input、mask 和时间不变并修改所有 hidden targets 后：

- 导出的模型输入必须逐位相同；
- 模型配置和任务 ID 必须相同；
- 只有评估 target 的哈希可以改变。

不满足此性质的数据包不得进入训练或测试。

## 5. 训练、调参与检查点选择

### 5.1 小范围调参

调参只使用 `seed=2026`、`strict_file`、30% point masking 的验证集
missing-RMSE。测试集不参与超参数选择。

预注册网格如下，每个模型最多四个候选：

| 模型 | 候选空间 |
| --- | --- |
| BRITS | hidden size `{32, 64}` x learning rate `{1e-3, 5e-4}` |
| SAITS | `{1 layer, 64 dim}` 或 `{2 layers, 128 dim}` x learning rate `{1e-3, 5e-4}` |
| CSDI | channel width `{32, 64}` x learning rate `{1e-3, 5e-4}` |
| SSSD | residual width `{32, 64}` x learning rate `{1e-3, 5e-4}` |

CSDI 固定使用 50 个 diffusion steps、quadratic schedule、`beta_start=1e-4`、
`beta_end=0.5`。SSSD 固定使用官方 SSSD-S4 日程：`T=200`、`beta_0=1e-4`、
`beta_T=0.02`；同时固定 `num_res_layers=36`、`s4_d_state=64`、
`s4_lmax=30`、bidirectional S4、layer normalization 和零 S4 dropout。
`skip_channels` 与候选 residual width 相同。上述参数不进入网格，并在任何调参
任务启动前写入不可变 tuning plan。

### 5.2 共同训练规则

- 最多 100 epochs；
- early stopping patience 为 20；
- PyPOTS 模型使用锁定版本提供的原生训练流程；
- SSSD 使用官方损失和优化流程，并由适配器增加验证选择及检查点封存；
- CSDI 和 SSSD 调参验证使用 5 次采样均值；
- 正式概率评估使用 50 次采样；
- 主要选择量只有 normalized missing-RMSE。

RMSE 数值相同时，依次选择参数量更少、验证推理更快、configuration ID 字典序更小
的候选。CUDA OOM、非有限 loss 或无有效检查点会把候选标为失败；系统不会静默缩小
模型或更改 batch size 后把它当成原候选。

调参结束后生成带哈希的 `selected_hyperparameters.json`。正式 5-seed 任务只接受
该文件列出的配置。正式训练记录参数量、wall time、峰值显存、单窗口吞吐和完整录制
推理时间，但这些量不参与主要准确率排名。

## 6. 推理、拼接与概率输出

### 6.1 完整录制推理

固定长度神经模型使用 50% 重叠窗口推理。确定性模型对重叠位置求均值。所有模型的
最终 completed signal 必须逐位保留 observed input。

CSDI 和 SSSD 对每个测试条件生成 50 条样本。每个 sample index 独立完成全部窗口
拼接，形成 50 条完整录制序列；随后才计算均值、分位数和概率指标。禁止先在窗口内
求分位数再拼接。

### 6.2 统一指标

九个模型都报告：

- normalized 和 physical missing MSE、RMSE、MAE；
- 现有 measured-attitude trajectory diagnostics；
- 训练和推理成本。

CSDI 和 SSSD 额外报告：

- empirical CRPS；
- 95% prediction interval coverage；
- 95% prediction interval width。

概率指标只在人工缺失位置计算，并分别保存 normalized 与 physical 结果。CSDI/SSSD
样本均值作为确定性点预测进入 RMSE、MAE、MSE 和轨迹指标。确定性模型不伪造概率
区间，也不为其写入虚假的零宽区间。

## 7. 正式任务与统计汇总

调参锁定后，正式阶段共有 9 个模型 x 5 个种子，即 45 个训练任务。每个任务对
13 个条件执行完整测试录制评估。

独立统计单位保持为 recording。最终输出至少包含：

- 每录制、每种子、每条件原始指标；
- 均值、标准差、中位数和 IQR；
- paired bootstrap 95% confidence interval；
- 配对效应量；
- 对声明的主要比较执行 Holm multiplicity correction；
- 按 topology 和 missing rate 的性能曲线；
- CSDI/SSSD 概率校准图；
- 参数量、训练时间、显存与推理速度表。

主表直接比较当前 Hybrid 与四个现代模型，同时保留 Linear、LOCF、BiLSTM 和 BiLNN
作为参照。汇总器拒绝缺少模型、种子、条件、录制或检查点哈希的 campaign。

## 8. MatPool 运维闭环

### 8.1 Windows 本地打包

```powershell
.\scripts\package_modern_experiment.ps1 -IncludeData
```

上传包包含精确代码提交、约 0.6 GB 的当前 OxIOD 数据、锁定的 SSSD 快照、环境
清单、bootstrap 和包级 SHA-256。默认排除 `.git` 工作对象之外的不相关内容、历史
`results/`、`.worktrees/`、缓存和本地虚拟环境。打包器拒绝脏的实验代码提交；用户
已有但不属于本任务的工作树改动不会被自动纳入或覆盖。

### 8.2 服务器命令

```bash
bash scripts/run_modern_imputation_matpool.sh prepare
bash scripts/run_modern_imputation_matpool.sh start
bash scripts/run_modern_imputation_matpool.sh status
bash scripts/run_modern_imputation_matpool.sh logs
bash scripts/run_modern_imputation_matpool.sh resume
bash scripts/run_modern_imputation_matpool.sh package-results summary
bash scripts/run_modern_imputation_matpool.sh package-results full
```

`prepare` 创建并验证：

1. V2 + PyPOTS 环境，用于控制面、现有模型、BRITS、SAITS 和 CSDI；
2. SSSD 隔离环境，用于锁定的 SSSD-S4 运行时。

preflight 检查 Linux、RTX 4090 级 GPU、至少 23 GiB 显存、CUDA/驱动、Python、
依赖锁、源码哈希、数据哈希和两个环境的真实 smoke。任何一项失败都不得进入正式
付费训练。

### 8.3 状态机和恢复

```text
preflight -> data_export -> tuning -> hyperparameter_lock
          -> formal_training -> evaluation -> summary
          -> artifact_validation -> complete
```

单 GPU 同时只运行一个 GPU 训练或扩散推理任务。`start` 使用 `tmux` 后台运行；SSH
断开不终止任务。每个任务保存 manifest、日志、检查点、预测包和原子完成标记。

`resume` 只恢复未完成或明确失败的任务。完成且哈希匹配的任务不会覆盖；完成标记与
产物不一致时立即失败，要求先诊断。OOM、NaN、shape/dtype 不符、observed value
改变、概率样本数不足、检查点不匹配或环境漂移都会保留诊断证据并停止相关阶段。

## 9. 下载产物

`package-results summary` 生成小型结果包，包含：

- resolved configs、split/scaler/mask hashes；
- 调参候选和锁定结果；
- 每录制指标、汇总表、统计检验和图表；
- 日志、运行时信息、检查点身份和审计报告。

`package-results full` 在上述内容之外包含：

- 实际检查点；
- CSDI/SSSD 的 50 次压缩样本预测；
- 可在本地重新计算全部指标的输入索引和 schema。

两种包都附 SHA-256。下载并解压后，可在无 GPU 的本机重新运行汇总和绘图。失败的
campaign 可以生成单独的 diagnostic package，但不能冒充 `summary` 或 `full`
完成包。

## 10. 测试策略

实现遵循 test-first red-green-refactor。新增生产行为必须先有能够因功能缺失而失败
的测试。

### 10.1 单元和契约测试

- split/scaler/window/mask 身份与哈希；
- hidden-target mutation 下的输入泄漏不变性；
- 四个现代模型适配器的 shape、dtype、model ID 和 sample dimension；
- overlapping windows 的逐样本拼接；
- observed values 精确保留；
- empirical CRPS、95% coverage 和 interval width 的解析小样本；
- 任务 ID、manifest 和完成标记的确定性；
- 上传包与结果包重建和 SHA-256 校验。

### 10.2 集成和故障测试

- 模拟两个环境的成功、非零退出、中断、重复启动和恢复；
- 拒绝错误环境、错误 checkpoint、错误预测样本数和部分产物；
- PyPOTS 模型的极小真实数据 smoke；
- 现有 Hybrid 与现代模型进入同一条件矩阵；
- 现有 `tests/validation_v2` 全量回归通过。

### 10.3 服务器专属验收

Windows 本机验证 SSSD 的数据契约、命令、结果导入和恢复逻辑。MatPool `prepare`
必须在实际 SSSD CUDA/S4 环境中完成一个真实 batch 的 forward、backward、checkpoint
reload 和两次非退化采样。两次概率样本不能逐位相同，且都必须保持 observed values。

正式 campaign 只有在九个模型、五个种子、十三个条件和全部测试录制完整，所有哈希
闭合，统计汇总复算一致时才能标记为 `complete`。

## 11. 非目标

- 本阶段不运行 scenario-holdout；
- 不修改论文正文；
- 不把 measured-attitude trajectory diagnostic 描述为完整自主惯导系统；
- 不为追求结果而在测试集上选参；
- 不用历史 CSV 补齐失败或缺失的任务；
- 不要求 Windows 本机证明 SSSD 官方 CUDA 扩展在 Linux 服务器上的兼容性。

## 12. 验收标准

代码可以交付到服务器运行，必须满足：

1. 所有新增测试和现有 `tests/validation_v2` 通过；
2. 本地 smoke 完成现有 Hybrid 和可用的 PyPOTS 小流程；
3. Windows 上传包可重建、校验并包含明确服务器入口；
4. MatPool `prepare` 能创建两个锁定环境并通过真实 SSSD preflight；
5. 调参计划在任务启动前封存，正式运行只能使用锁定超参数；
6. `status`、`logs` 和 `resume` 能在中断场景中保留已完成证据；
7. 结果包能在本机无 GPU 条件下重新校验、汇总和绘图；
8. 正式完整性验证拒绝任何缺模型、种子、条件、录制或概率样本的结果。
