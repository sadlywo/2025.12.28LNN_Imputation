# Peer Review Report — Peer Reviewer 3（Continuous-Time Models & Edge Deployment）

## Manuscript Information

- **Title**: *Physics-Informed Neural Temporal Hybridization for Missing IMU Data Imputation: An Integrated Bidirectional LNN-LSTM Hybrid Model*
- **Manuscript ID**: 未提供
- **Review Date**: 2026-07-11
- **Review Round**: Round 1

## Reviewer Information

### Reviewer Role

Peer Reviewer 3（Perspective / Cross-disciplinary & Practical）

### Reviewer Identity

本人从事 continuous-time neural networks（LTC/CfC）、irregularly sampled sensor streams、streaming inference 与 edge benchmarking 研究，重点关注连续时间语义是否真实进入计算图、离线双向模型与在线部署边界，以及模型 profiling 是否足以支撑工程结论。作为跨学科审稿人，我不对惯性导航坐标物理或完整统计设计作全面裁决；本报告仅评估连续时间建模假设、门控机制解释和部署可行性。

### Review Focus

本报告将论文的三项实践性主张逐项映射至源码和保存结果：（i）BiLNN/CfC 是否真正以实际 `dt` 驱动连续时间状态演化；（ii）continuous channel loss 是否构成 irregular sampling；（iii）双向混合模型和 RTX 4090D batch-forward 测试是否能支持实时/嵌入式部署。门控部分仅审查其输入、权重语义与机制解释，不延伸为一般统计审计。

## Overall Assessment

### Recommendation

- [ ] Accept
- [ ] Minor Revision
- [x] **Major Revision**
- [ ] Reject

### Confidence Score

**5 / 5**。CfC 调用路径、bidirectional streaming 边界、router/gate 解释与 GPU inference profiling 均在本人核心专业范围内；评分仍应被理解为未经过 gold-set calibration 的序数判断。

### Summary Assessment

本文提出以 BiLNN/CfC 和 BiLSTM 为双分支、经逐时刻逐通道 gate 融合的 IMU 插补框架，并尝试以随机缺失、连续通道缺失、轨迹指标和推理速度说明其实践价值。从跨学科角度看，问题重要，残差插值锚点、双向上下文和轻量门控构成了一个具有工程潜力的离线修复器；作者也诚实承认当前 Hybrid 难以直接用于严格实时场景（正文 p. 11；`Manuscript.tex:411-414`）。然而，论文最有辨识度的“continuous-time / irregular sampling / edge deployment”证据链尚未成立：实际 CfC 调用没有把 `dt` 传入 `timespans`，连续通道掩码不改变采样时间，双向分支和双边线性插值需要未来信息，门控图不能证明分支功能因果分工，部署表亦无法由单一 CSV 完整重建。上述问题并不意味着模型没有插补效果，而意味着当前结果主要支持“离线、带未来上下文的残差混合插补”，尚不能支持“连续时间不规则采样建模”或“嵌入式部署可行性”。建议大修：重建时间语义、增加真实 timestamp irregularity 与 fixed-lag/causal 实验、重做可追溯的 single-stream edge benchmark，并把 gate 结论降级或经干预实验验证。

## Strengths

### S1：问题设置具有明确的跨学科实用价值

论文没有只停留在点误差，而是把 IMU 插补与轨迹重建、计算开销联系起来（正文 pp. 1-2、6、10-11；`Manuscript.tex:132-146,268-272,411-414`）。这种“signal quality → downstream utility → systems cost”的评价视角值得保留，适合 Measurement 或 IEEE Sensors Journal 的读者。

### S2：架构组件为离线缺口修复提供了合理工程组合

代码中的 `ResidualInputAdapter` 使用左右观测点构造线性基线和到边界的相对距离，双向 CfC/BiLSTM 再预测残差（`experiment_bidirectional_lnn_residual.py:115-185,188-238`）。对于允许等待未来观测的日志修复、运动数据清洗或离线重建，这是合理且可实现的 smoothing-style 方案，而非没有应用场景的纯结构拼接。

### S3：profiling 代码具备若干正确的基础动作

基准脚本加载真实 checkpoint、切换 `eval()`，设置 warm-up，并在计时前后执行 CUDA synchronization（`benchmark_downstream_models_inference_speed.py:136-215,218-238`）；同时报告 parameter count、state-dict size、checkpoint size、batch size、sequence length 与 input dimension（同文件 `:260-284`）。这些基础设施可直接扩展为合格的 profiling protocol。

### S4：作者没有完全回避速度-精度权衡

正文明确指出 Hybrid 的推理速度最慢，并承认未经优化时可能难以用于严格实时场景（正文 p. 11；`Manuscript.tex:412-414`）。这一克制判断比直接宣称“deployment-ready”更可信，也为修订时重新定位为 offline/on-device delayed imputation 留出了空间。

## Weaknesses

### W1：实际 CfC 未由真实时间间隔驱动，当前 BiLNN 不能被视为已验证的 continuous-time irregular-sampling 模型

**Problem**：论文称 CfC 的 time-dependent gate 显式使用时间，并据此适合 irregular sampling（正文 pp. 2-3；`Manuscript.tex:138,155,168-178,451`）。数据集确实计算了 `dt` 并把它拼进输入向量（`dataset.py:279-281,294-297,401-403`），但主模型调用是 `self.forward_lnn(enriched)` 和 `self.backward_lnn(rev_x)`，没有向 CfC 的 `timespans` 参数传入 `dt`（`experiment_bidirectional_lnn_residual.py:188-213`；trajectory 版本同样见 `experiment_bidirectional_lnn_residual_trajectory.py:182-202`）。安装的 `ncps 1.0.1` 中，`CfC.forward(input, hx=None, timespans=None)` 在未提供 `timespans` 时将每步 `ts` 固定为 `1.0`（`D:/Anaconda3/Lib/site-packages/ncps/torch/cfc.py:112-173`）。因此，`dt` 至多是一个普通协变量，未进入 CfC 闭式连续时间状态更新的指定路径。更进一步，反向分支直接翻转包含前向 `dt[t]=time[t]-time[t-1]` 的完整特征张量（`experiment_bidirectional_lnn_residual.py:211-213`），而没有为反向时间轴重新对齐间隔。论文还称采用 NCP sparse wiring（正文 p. 3；`Manuscript.tex:166-167`），代码却以整数 hidden units 构造 fully connected CfC，未使用 `AutoNCP` wiring（`experiment_bidirectional_lnn_residual.py:194-195`）。

**Why it matters**：这是论文从普通 time-aware recurrent imputer 区分于离散 RNN 的核心依据。原始 CfC 工作明确把 irregular sample 的时间变量设为其 timestamp/order 所对应的 elapsed time；仅把 `dt` 作为输入特征不能证明网络享有同样的连续时间语义。[Hasani et al., 2022](https://doi.org/10.1038/s42256-022-00556-7)

**Suggestion**：将 `dt` 明确拆出为 `timespans`，分别构造 forward 与 reverse 的正时间间隔；报告 `timespans=actual_dt`、`timespans=constant`、`dt-as-feature only`、`no-dt` 四组受控消融。对 timestamp 进行 jitter、random thinning 和 burst dropout，并评价对不同 gap duration 的性能。若继续采用 fully connected CfC，应删除“NCP sparse wiring 被采用”的表述；若确实采用 NCP，则提供 wiring 配置及相同参数/算力下对照。另需固定 `ncps` 精确版本，而不是仅给出 `ncps>=0.0.9`。

**Severity**：**Critical**

### W2：continuous channel loss 与 irregular timestamp sampling 被概念性混同

**Problem**：论文称连续通道缺失“aligns well with the irregular time sampling assumption”（正文 p. 6；`Manuscript.tex:251-252`）。实际 `missing_mode="channel"` 只是选择若干 feature channel 并把整段 mask 置零，时间戳及全局 `dt` 完全不变（`dataset.py:337-352`）；`block` 模式同样只改变每通道 mask（`:343-348`）。这模拟的是 feature-wise/structured missingness，而非 observation times 非均匀或 asynchronous sampling。现有 OxIOD 记录的主采样周期也大多约为 0.01 s；代码把异常/负间隔裁剪到 `[1e-4,1.0]`（`dataset.py:279-281`），但未报告真实时间间隔分布或专门构造 irregular-time test。

**Why it matters**：跨学科上，这两类问题需要不同机制。global `timespan` 描述相邻事件间隔；per-channel missingness 需要 mask、每变量 time-since-last-observed 或 asynchronous event representation。把二者合并会错误地把 channel dropout 下的优势归因于 CfC 的 continuous-time capability，也无法回答模型面对真正 packet timestamp gaps 时是否稳健。

**Suggestion**：将实验轴拆成（A）regular timestamps + value/channel missingness；（B）irregular timestamps + all-channel packet loss；（C）asynchronous per-channel sampling。分别报告 missing fraction、gap duration、inter-arrival distribution 与每通道 observation density，并增加 GRU-D 等 per-variable decay baseline。论文术语应将当前情景称为 continuous channel masking/loss，而不是 irregular sampling。

**Severity**：**Major**

### W3：双向架构、右边界插值与“实时部署”之间缺少明确的服务边界

**Problem**：BiLNN 明确翻转完整窗口以利用未来信息（`experiment_bidirectional_lnn_residual.py:208-214`），BiLSTM 同样是双向网络；残差基线在缺口两端之间做线性插值，并显式扫描右侧观测点（`:93-113,146-162`）。因此，对时刻 `t` 的输出通常需要 `t` 之后的数据。论文一方面强调 real-world deployment 和 embedded applications（正文 pp. 2、4、10-11；`Manuscript.tex:140,153,220,411-414`），另一方面没有定义这是离线修复、固定延迟 smoothing，还是 sample-by-sample online inference。以 `seq_len=30` 和约 100 Hz 数据为例，完整窗口本身可引入最高约 0.29 s 的 algorithmic look-ahead；这还未包括预处理和网络计算。

**Why it matters**：系统中的总响应时间是 algorithmic latency（等待未来样本）+ compute latency，而论文只测后者的 batch forward。双向模型可以适合离线或固定延迟任务，但不能在未说明 look-ahead budget 的情况下被外推到 closed-loop navigation、fault-tolerant sensing 或实时控制。

**Suggestion**：首先定义 deployment contract：输出时限、允许 look-ahead、batch size、设备、功耗和缺口何时被认为结束。至少增加三条曲线：causal unidirectional、fixed-lag BiLLSTM（look-ahead `k∈{1,5,10,15,29}`）、full offline bidirectional；横轴使用端到端 latency，纵轴使用 missing-point RMSE/轨迹指标。若不补实验，应把全文部署结论限定为“offline or delayed on-device post-processing”。

**Severity**：**Major**

### W4：gate 的实现、图注与机制解释不一致，现有热图不能证明“分支切换机制”

**Problem**：方法文字称 gate 输入包括两个分支 hidden states、原始 residual input 和两分支预测差的范数（正文 p. 5；`Manuscript.tex:234-238`）。主代码实际拼接的是 `lnn_pred, lstm_pred, lnn_unc, lstm_unc, x`，既没有 hidden states，也没有显式 prediction-difference norm（`experiment_bidirectional_lnn_residual.py:257-275`）。所谓 uncertainty 是两个 Softplus head 的输出，但主 reconstruction-only objective 并未校准或监督这些 head（同文件 `:64-77,201-217,264-275`）。此外，公式中 `g` 是 LNN 权重（`Manuscript.tex:235-238`），当前可视化代码也将 gate 标为 LNN weight（`visualize_hybrid_gate_bias_segments.py:145-169,203-239`），而论文 Fig. 8 图注却称 heatmap 是 BiLSTM weights、蓝线是 LNN bias（正文 p. 10；`Manuscript.tex:405-409`）。保存的 gate summary 中五段平均 LNN gate 仅约 0.055-0.089（`results/Downstream_Tra_Plot/gate_bias_visualization/hybrid_gate_segment_summary_20260325_145414.csv:2-6`），与论文四面板图没有可见的一一 provenance 说明。最后，gate 的 `x` 还包含由完整 `target_imu` 计算的窗口统计量（`dataset.py:366-401`），使“根据可观测局部状态自适应切换”的解释尤其需要谨慎。

**Why it matters**：router/gate 权重是预测计算中的混合系数，不天然等于专家重要性、因果贡献或已校准不确定性。仅观察权重随时间变化，无法区分真实 motion-dependent specialization、通道尺度偏置、residual anchor、完整窗口统计量或 gate saturation。当前摘要中“visualization confirms distinct switching mechanisms”的表述因此过强（正文 p. 1；`Manuscript.tex:104`）。

**Suggestion**：先统一 `g` 的分支语义、方法公式、绘图代码、caption 和 CSV；保存 figure manifest（checkpoint hash、sample index、mask seed、gate definition）。再做干预性验证：固定 gate=0/0.5/1、shuffle gate over time/channels、移除 uncertainty 输入、移除 window features、交换分支输出，并报告各运动强度/缺口长度下 conditional branch error、gate entropy、gate-error correlation。若 uncertainty 继续参与路由，应以 held-out calibration（如 NLL、coverage、reliability curve）证明其含义。完成前，将“confirms/explains”改为“suggests/descriptive visualization”。

**Severity**：**Major**

### W5：RTX 4090D、batch=32 的纯 forward 表不能支持 embedded deployment，且 Table 3 无法由单一保存结果重建

**Problem**：论文 Table 3 报告 Hybrid 为 1476.5212 ms/batch、21.6726 samples/s，并据此讨论嵌入式潜力（正文 pp. 10-11；`Manuscript.tex:411-431`）。脚本以 batch=32、seq=30 的随机张量作为输入，仅测 `model(sample_input)`，不包含数据传输、预处理、mask/residual construction 的真实输入分布、后处理、队列、功耗或内存峰值（`benchmark_downstream_models_inference_speed.py:241-284`）。虽然 warm-up、`eval()` 与同步均存在，但只保存单一平均值，没有独立重复、median、p95/p99 或 dispersion。尤其是 residual adapter 含逐 batch/逐通道 Python 循环和多次 `.item()`（`experiment_bidirectional_lnn_residual.py:134-162`），其 GPU 延迟会依赖输入 mask，并可能被 host-device synchronization 主导；用 Gaussian random input 不能代表真实二值 mask。

结果 provenance 亦不闭合：Table 3 的 BiLSTM/BiLNN/Hybrid 三行与 `inference_speed_benchmark_20260325_134753.csv:4-6` 基本精确对应；但同一 CSV 的 GRU/Transformer 参数与延迟是 80,844/424,268 和 0.9605/0.9811 ms，而表中为 92,364/428,108 和 110.7007/433.3724 ms。后一个 `inference_speed_benchmark_20260326_140536.csv:2-3` 的参数量与表一致，但延迟为 1.1070/2.3337 ms、throughput 为 28,906.8/13,712.0 samples/s，仍不等于表值。因此 Table 3 似乎混合了不同 checkpoint/run 或发生了 time-per-100-runs 与 time-per-run 的单位处理差异。

**Why it matters**：parameter count 和 FP32 state-dict size 只能说明静态模型体积；不能代表运行内存、能耗、deadline compliance 或 edge operator support。官方 MLPerf Inference 将 single-stream latency、offline throughput、系统描述、accuracy compliance 和（可选）full-system power 分开定义；RTX 4090D 上 batch throughput 不能外推到嵌入式 single-stream。[MLCommons Inference Edge](https://mlcommons.org/benchmarks/inference-edge/)；PyTorch 官方 benchmark 工具也强调 warm-up、同步和 run-to-run variation 的控制。[PyTorch Benchmark Utils](https://docs.pytorch.org/docs/stable/benchmark_utils.html)

**Suggestion**：重建唯一可追溯 benchmark manifest，记录 git commit、checkpoint SHA256、Python/PyTorch/CUDA/cuDNN/ncps 版本、OS、GPU/CPU、precision、clock/power mode、batch、seq、warm-up 和 repeats。分别报告 batch=1 single-stream latency（median/p95/p99）、batch throughput、host-to-device + preprocessing + inference + postprocessing 端到端时间、peak RAM/VRAM、energy/inference；至少在一个实际目标平台（如 Jetson/ARM CPU/NPU）上测试 FP32/FP16/INT8，并验证量化后精度。用真实 dataset batch 测时，消除 Python `.item()` 同步热点，再讨论 deployability。

**Severity**：**Major**

## Detailed Comments

### Assumption Audit

- **Explicit assumptions**：论文明确假定 BiLNN 捕捉 local irregular dynamics、BiLSTM 捕捉 long-range dependency，gate 可根据 motion condition 自适应融合（`Manuscript.tex:138-145,278-280,397-404`）。其中“长期依赖”与双向上下文是合理建模选择；但“irregular dynamics”目前既没有正确 `timespans` 路径，也没有 irregular timestamp 对照，不能由现有实验验证。
- **Implicit assumptions**：第一，feature/channel missingness 被假定等同于 irregular sampling；第二，未来上下文在部署时随时可用；第三，gate weight 被假定为分支功能贡献；第四，2.26 MB state dict 被假定为 embedded friendliness；第五，RTX 4090D batch forward 被假定能代表 single-stream edge inference。这五项都需要显式边界或实验支持。
- **Paradigmatic assumptions**：论文把“使用 continuous-time architecture”近似等同于“实现 continuous-time semantics”。从 dynamical-systems 角度，连续时间性质来自状态更新对实际 elapsed time 的显式依赖及其在不同采样网格下的行为；网络名称本身不是证据。类似地，从 systems 角度，模型体积和 kernel forward time 也不是部署可行性的充分统计量。

### Cross-Disciplinary Connections

- **Parallel research**：CfC/LTC 文献把 irregular sampling 表示为每个事件的实际时间距离，并在非等距序列上验证；missing-data 文献则常以 per-variable elapsed time 和 decay 处理异步观测。本文应把这两条研究线分开，而不是用一个 channel mask 同时代表二者。
- **Borrowing opportunities**：可从 fixed-lag smoothing 借用“性能-未来视野-算法延迟”三者曲线；从 mixture-of-experts 借用 router collapse、routing entropy 和 expert specialization 的干预检验；从 uncertainty calibration 借用 reliability/coverage 测试；从 real-time systems 借用 deadline、tail latency、WCET-like upper quantile 和 energy budget。
- **Methodological borrowing**：建议使用 sampling-grid invariance test：对同一物理轨迹以不同 timestamp grid 重采样，比较实际 `timespans` CfC 与 constant-step CfC 的退化幅度；使用 counterfactual gate intervention 检验路由，而不是仅画 heatmap；使用 MLPerf-style single-stream/offline 两种 scenario 区分 latency 与 throughput。

### Practical Impact

- **Real-world application**：当前架构最可信的用途是离线 IMU 日志修复、运动数据清洗、训练集补全和允许固定延迟的后处理。若在这些场景重新定位，双向性是优点而非缺点。
- **Implementation feasibility**：严格实时导航尚未被证明。除双向 look-ahead 外，Python 循环和 `.item()` 同步会显著影响 GPU/edge 性能；CfC 的 step-wise recurrent execution 也需要 operator-level profiling。参数剪枝只是一个方向，无法替代算法路径重构与目标硬件验证。
- **Stakeholders**：建议在 Discussion 中分别面向算法研究者、嵌入式工程师和导航系统使用者声明边界。对后两者，最关键的不是平均 batch throughput，而是缺口结束后多久能输出、最坏/尾部延迟、内存/能耗及异常时间戳时的失效方式。

### Broader Implications

- **Ethical / safety dimensions**：如果将离线双向补缺器描述为在线容错组件，使用者可能在 closed-loop 系统中低估未来信息依赖和 deadline miss 风险。高影响传感应用需要明确“不可在线使用”的边界和 fail-safe 行为。
- **Scientific communication impact**：把 gate heatmap 当作机制确认、把 channel loss 当作 irregular time、把 desktop GPU batch forward 当作 edge feasibility，容易造成跨领域术语迁移中的证据膨胀。修订后若能明确区分这些概念，论文反而可成为 measurement-AI 领域较有教育意义的范例。
- **Future directions**：最有价值的路线是 causal/fixed-lag hybrid、真实 asynchronous IMU stream、interval-aware CfC、校准 gate/uncertainty、量化与 target-hardware benchmark，以及性能-延迟-能耗 Pareto front。

## Cross-Disciplinary Reading Recommendations

1. **Hasani et al. (2022), “Closed-form continuous-time neural networks”**：CfC 原始论文，尤其应对照其 time-dependent gating、irregular timestamp encoding 与 physical dynamics 实验，明确 `timespans` 的语义。[Nature Machine Intelligence](https://doi.org/10.1038/s42256-022-00556-7)
2. **Hasani et al. (2021), “Liquid Time-constant Networks”**：用于校正 LTC 的 ODE、adaptive time constant、稳定性与 expressive-power 表述，避免把 CfC/LTC 的序列推理写成整体 O(1)。[AAAI Proceedings](https://doi.org/10.1609/aaai.v35i9.16936)
3. **Cao et al. (2018), “BRITS: Bidirectional Recurrent Imputation for Time Series”**：可帮助作者把 bidirectional imputation 正确定位为使用未来上下文的 smoothing-style 方法，并设计 causal/fixed-lag 对照。[NeurIPS Proceedings](https://papers.nips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html)
4. **MLCommons, MLPerf Inference: Edge**：用于重新设计 single-stream latency、offline throughput、系统描述、accuracy compliance 与 power 测量，避免将 desktop batch forward 外推为 edge deployment。[Official benchmark](https://mlcommons.org/benchmarks/inference-edge/)
5. **PyTorch `torch.utils.benchmark` documentation**：用于建立 warm-up、accelerator synchronization、重复测量和 dispersion 报告规范。[Official documentation](https://docs.pytorch.org/docs/stable/benchmark_utils.html)

## Questions for Authors

1. 作者是否有任何实验实际调用了 `CfC(..., timespans=dt)`？若有，请给出对应 commit、checkpoint 和配置；若没有，如何区分当前结果来自 CfC continuous-time update，还是来自把 `dt` 当作普通输入特征及残差插值基线？
2. 目标 deployment service 究竟是 offline repair、fixed-lag imputation，还是 causal real-time navigation？允许多少未来样本、总 deadline 和目标硬件/功耗预算分别是多少？
3. 请提供生成 Table 3 的唯一原始 CSV 和运行命令。为什么 GRU/Transformer 的参数量对应 2026-03-26 的 checkpoint，而 latency/throughput 与该 CSV 不一致；表内毫秒是 per run、per 100 runs 还是其他统计量？
4. Fig. 8 中 heatmap 究竟是 LNN 还是 BiLSTM 权重？为什么当前可视化 CSV 是五段且平均 LNN gate 约 0.055-0.089，而论文是四段并采用不同权重范围？作者能否用 fixed/shuffled gate 干预而非可视化相关性证明分支 specialization？

## Minor Issues

### Terminology / Model Specification

- 正文 p. 3 称 CfC 把计算复杂度降为“constant time complexity”（`Manuscript.tex:174`）。准确表述应是每次 recurrent update 无需 iterative ODE solver；长度为 `T` 的顺序处理仍为 O(T)。
- 正文 p. 4 称 BiLNN 每方向 64 hidden units（`Manuscript.tex:222`），主代码和超参结论则使用 128（`experiment_bidirectional_lnn_residual.py:559,668-673`; `Manuscript.tex:434`）。请统一。
- 摘要中的 “distinct switching mechanisms” 和正文中的 “confirm/explain” 应在干预性证据出现前改为描述性措辞。
- “BiLLSTM” 容易被误读为双向 liquid-LSTM 单体。建议固定使用 “BiLNN-BiLSTM hybrid” 或给出一次无歧义缩写定义。

### Figures / Tables / Layout

- Fig. 8 的图内标签、公式的 gate 方向、caption 和当前代码必须统一；当前颜色条字号偏小，且缺少 sample id、scenario、mask pattern 和 checkpoint provenance。
- Table 3 应增加 batch size、sequence length、precision、设备、重复次数和统计量定义；“samples/s”需说明 sample 是一个 30-step window 还是一个 IMU time step。
- PDF 未发现裁切或元素重叠；但 Fig. 8 和 Table 3 字号相对正文偏小，工程复核信息不足。

## Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|---|---:|---|---|
| Originality (20%) | 62 | Adequate | 双分支残差门控用于 IMU 插补有增量价值，但 continuous-time 与 gate 贡献尚未被隔离。 |
| Methodological Rigor (25%) | 44 | Insufficient / Weak | `timespans` 未接入、irregular sampling 测试缺失、双向部署 contract 未定义。 |
| Evidence Sufficiency (25%) | 47 | Weak | 插补数值丰富，但 gate mechanism、真实 irregularity 与 edge deployment 缺少直接证据；profiling provenance 不闭合。 |
| Argument Coherence (15%) | 53 | Weak | “CfC→irregular”“channel loss→irregular time”“batch GPU→embedded”存在跨层推断。 |
| Writing Quality (15%) | 63 | Adequate | 主线可读，但术语、复杂度、gate 方向和模型配置有多处不精确。 |
| Literature Integration (optional) | 58 | Weak | 引用了 LTC/CfC 与 GRU-D，但未把 elapsed-time、feature missingness 和 streaming boundary 整合进实验设计。 |
| Significance & Impact (R3 focus) | 61 | Adequate | 问题重要，离线应用价值明确；当前实时/边缘影响被高估。 |
| **Weighted Average** | **52.6** | **Major Revision** | 按 20/25/25/15/15 权重计算；核心问题可修复，但需要补实验而非文字润色。 |

## Final Recommendation Rationale

建议 **Major Revision**，而非 Reject：模型作为离线或 fixed-lag IMU imputer 具有明确工程潜力，代码也已有可继续扩展的 profiling 与 component-output 基础。但接受前必须完成至少三项实质修订：（1）以真实 `timespans` 和 irregular timestamp 实验重新建立 continuous-time 证据；（2）明确 bidirectional look-ahead 的应用边界并增加 causal/fixed-lag 对照；（3）用可追溯、真实输入、single-stream、目标硬件的 benchmark 重做部署评价。gate 机制若不能经干预验证，则应从“机制确认”降级为“描述性观察”。

---

**Sprint-contract fallback note**：本次调用未提供 v3.6.2 reviewer sprint contract、paper-blind precommitment 或 `<phase1_output>`；因此依任务指示采用标准单阶段 perspective review，未执行 contract-specific dimension/failure-condition 协议。本报告严格停留在 academic-paper-reviewer Phase 1 的 Peer Reviewer 3 边界，未读取其他 Phase 1 报告，未进行编辑综合，未修改论文或源码。
