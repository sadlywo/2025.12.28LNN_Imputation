# Phase 0：领域识别与审稿人配置

> 状态：等待作者确认。本文档只完成 `academic-paper-reviewer` 的 Phase 0；尚未启动 5 位审稿人的独立评审，也未修改 `Manuscript.tex`。

## 1. 审查材料与基本信息

- 论文题目：*Physics-Informed Neural Temporal Hybridization for Missing IMU Data Imputation: An Integrated Bidirectional LNN-LSTM Hybrid Model*
- 摘要长度：约 203 词
- 正文长度：约 5,681 词（LaTeX 清理后的近似计数，不含主要图表环境）
- 参考文献：PDF 实际列出 30 条；`document.bib` 含 39 个条目
- PDF：14 个物理页面，其中 1 页 Highlights、13 页编号正文
- 数据与代码：当前仓库含 45 对 OxIOD IMU/Vicon 文件，以及训练、评估、消融、轨迹和部署基准脚本
- 辅助材料：已读取此前 GPT 问题清单，并把它作为审稿线索而非既定结论

## 2. 领域分析

| 维度 | 识别结果 |
|---|---|
| Primary Discipline | 惯性传感与测量数据处理（inertial sensing / measurement data processing） |
| Secondary Disciplines | 时间序列缺失值插补；连续时间神经网络与深度学习；惯性导航与边缘 AI |
| Research Paradigm | 定量计算实验研究（quantitative computational study） |
| Methodology Type | 机器学习建模 + 人工缺失机制实验 + 消融研究 + 下游轨迹评估 + 推理性能基准 |
| Target Journal Tier | 当前证据链更接近专业领域 Q2–Q3 稿件，且尚不具备可靠投稿状态；若完成数据管线、物理定义、统计设计和可复现性重建，可面向专业 Q1–Q2 期刊 |
| Paper Maturity | “修订稿外观、方法学未闭环”：结构和图表接近投稿稿，但论文—代码—结果之间仍存在需要正式核验的高风险不一致 |

## 3. 真实代码主链定位

Phase 1 将以以下文件为主要代码证据链：

1. `dataset.py`：文件级划分、归一化、缺失掩码、窗口特征、时间间隔、Vicon 对齐与 ATE/RTE。
2. `experiment_bidirectional_lnn_residual.py`：BiLNN、BiLSTM、Hybrid 的主重建实验；其多缺失率 CSV 与论文表 1 的多组 RMSE 数值精确对应。
3. `experiment_baseline_imputation_comparison.py`：Mean、LOCF、KNN、MICE、GRU、Transformer 基线与测试结果。
4. `experiment_bidirectional_lnn_residual_trajectory.py`：所谓 trajectory-aware 训练目标和轨迹评估。
5. `models.py` 与 `models_hybrid.py`：旧版 physics loss、CfC/LSTM 分支及门控实现。
6. `demo_baseline_downstream_trajectory_random5.py`、`demo_bidirectional_lnn_residual_imputation*.py`：论文轨迹图与案例可视化来源。
7. `benchmark_downstream_models_inference_speed.py`：参数量、模型大小、forward latency 与 throughput。

这些定位结果将用于配置审稿人分工；它们尚不是 Phase 1 的正式审稿结论。

## 4. 推荐目标期刊

1. **Measurement**（首选但有条件）：主题与 measurement data processing、performance evaluation 和 measurement-oriented algorithms 相符；但该刊明确要求 AI 稿件体现测量科学增量、充分可复现，并非简单套用模型。当前稿件必须先补齐测量学语境、误差建模和复现证据。
2. **IEEE Sensors Journal**（现实匹配）：其 scope 明确覆盖 sensor data processing、machine learning、accelerometers/gyroscopes、信号稳定性与传感器系统验证，和论文对象高度匹配。
3. **Information Fusion**（冲刺目标）：只在论文能把“双分支门控”提升为有理论与实验支撑的 imperfect/incomplete-information fusion 方法，并完成强基线、跨数据集和严格统计验证后才建议考虑；以当前版本直接投稿风险很高。

## 5. 五位审稿人配置卡

### 配置卡 #1：Editor-in-Chief

**Role**：EIC / journal-fit editor  
**Identity Description**：*Measurement* 的资深副主编型审稿人，研究 measurement data processing、sensor-system performance evaluation 与 AI-assisted measurement，熟悉面向测量学期刊的机器学习 desk-screening 标准。  
**Review Focus**：

1. 判断工作究竟推进了 measurement science，还是仅把现有 BiLNN/BiLSTM 应用于公开 IMU 数据。
2. 审核创新性、读者价值、题目/摘要/贡献与实际证据是否匹配。
3. 判断现有验证强度能否支撑高水平 SCI 投稿，并给出期刊层级与编辑决定。

**Will particularly care about**：是否有可复现、可泛化且具有测量意义的误差模型与评价框架；“state-of-the-art”“physics-informed”“deployment-ready”等定位是否被证据支持。  
**Possible blind spots**：不深入逐行核验损失、掩码和 ATE 实现；由 Reviewer 1 与 Reviewer 2 补偿。

### 配置卡 #2：Peer Reviewer 1 — Methodology & Reproducibility

**Role**：方法学审稿人  
**Identity Description**：专攻 multivariate time-series imputation、missing-data mechanisms 与机器学习评估设计的统计学习研究者，熟悉 MCAR/MAR/MNAR、mask-aware objectives、grouped splits、nested validation、重复实验和不确定性报告。  
**Review Focus**：

1. 逐项核验训练损失、缺失位置指标、掩码方向、时间索引与论文公式的一致性。
2. 审计 file split、normalization、window feature、checkpoint reuse、超参数选择及潜在 test/target leakage。
3. 检查五次重复、随机种子、均值/方差、统计显著性、基线公平性和复现材料是否真实存在。

**Will particularly care about**：输入特征是否暗含完整 target window；测试集是否参与归一化或模型/超参数选择；表格数值能否从保存的配置与原始结果唯一重建。  
**Possible blind spots**：不会主导判断惯性坐标系、重力补偿和导航物理是否正确；由 Reviewer 2 负责。

### 配置卡 #3：Peer Reviewer 2 — Inertial Navigation & Physical Modeling

**Role**：领域审稿人  
**Identity Description**：从事 strapdown inertial navigation、IMU stochastic error modeling、Vicon/IMU synchronization 与 pedestrian inertial odometry 的资深研究者，熟悉坐标系变换、gravity removal、bias/drift、double integration、ATE/RPE/RTE 与轨迹对齐。  
**Review Focus**：

1. 判断所谓 physics-informed loss 是否对应可成立的惯性运动学约束，变量、单位、坐标系与可观测量是否闭环。
2. 审核 acceleration-to-trajectory 管线，包括姿态旋转、重力、初始速度、积分离散化、Vicon 对齐及 ATE 定义。
3. 判断轨迹实验是在评价 Vicon 真实轨迹、完整 IMU 积分参考，还是由初始化/短窗口造成的指标压缩。

**Will particularly care about**：手机坐标系下的 user acceleration 能否直接作为世界坐标加速度积分；physics loss 是否实际使用 velocity；ATE 单位、定义和表格数值是否一致。  
**Possible blind spots**：不重点评估 CfC 架构的新颖性和嵌入式 profiling 规范；由 Reviewer 3 负责。

### 配置卡 #4：Peer Reviewer 3 — Continuous-Time Models & Edge Deployment

**Role**：跨学科/实践审稿人  
**Identity Description**：研究 continuous-time neural networks（LTC/CfC）、irregularly sampled sensor streams 与 edge inference benchmarking 的机器学习系统研究者，熟悉双向模型的离线/在线边界、模型压缩和 GPU/embedded profiling。  
**Review Focus**：

1. 核验代码中的 BiLNN 是否忠实实现论文宣称的连续时间/不规则采样能力，以及 `dt` 是否真正进入正确计算路径。
2. 审核门控输入、残差基线、不确定性特征和双向上下文，判断门控可视化能否支持分支“切换机制”的因果解释。
3. 复核参数量、model size、warm-up、同步、batch latency、throughput 与硬件环境，界定可部署性结论。

**Will particularly care about**：continuous channel masking 与 irregular timestamp sampling 是否被混为一谈；bidirectional architecture 是否与实时补缺场景矛盾；RTX 4090D batch-forward 是否可外推到 embedded deployment。  
**Possible blind spots**：不主导统计显著性和导航坐标物理；分别由 Reviewer 1、Reviewer 2 补偿。

### 配置卡 #5：Devil’s Advocate

**Role**：核心论证挑战者  
**Identity Description**：具有 sensor AI、scientific machine learning 与论文可证伪性经验的严格审稿人，专门寻找“结果成立但解释不成立”“同一数字来自不同实验管线”“强结论由弱证据支撑”等逻辑断裂。  
**Review Focus**：

1. 构造最强反论证：性能提升是否可能来自 target leakage、残差插值基线、未来信息或实验选择，而非 hybrid/physics 机制。
2. 挑战“physics-informed”“state-of-the-art”“robustness”“generalization”“real-world deployment”五个核心标签。
3. 检查结论、表格、图注、代码输出和问题清单之间是否存在 cherry-picking、事后解释或不可追溯数值。

**Will particularly care about**：如果移除完整窗口统计、使用严格 held-out scenario、采用真实导航物理和强插补基线，论文的主结论是否仍然成立。  
**Possible blind spots**：刻意偏向反驳，可能低估工程探索价值；最终由 EIC synthesis 平衡，但任何 CRITICAL 问题不得被忽略。

## 6. 审稿策略与边界

- 五位审稿人将在 Phase 1 独立评审，不互相读取对方报告。
- 重点互补轴线为：期刊定位；统计方法与泄漏；惯性物理与 ATE；连续时间模型与部署；核心论证反驳。
- Phase 1 将引用论文页码/LaTeX 行号和代码行号；数值结论必须能追溯到 CSV、checkpoint/config 或明确标记为“无法复现”。
- PDF 视觉核查已覆盖全部页面；正式版式意见将在独立评审中报告。
- 本阶段不改写论文，不修代码，不生成编辑综合决定。

## 7. Checkpoint

请作者确认这 5 位审稿人配置，或指定要替换/强化的视角。确认后再进入 Phase 1（五份独立审稿）与 Phase 2（编辑综合、决定和修订路线图）。
