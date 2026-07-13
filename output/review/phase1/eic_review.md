# EIC Review Report

## Manuscript Information

- **Title**: *Physics-Informed Neural Temporal Hybridization for Missing IMU Data Imputation: An Integrated Bidirectional LNN-LSTM Hybrid Model*
- **Target-journal lens**: *Measurement*（measurement data processing / AI-assisted measurement）
- **Review date**: 2026-07-11
- **Review round**: Pre-submission full audit, Round 1

## Reviewer Information

### Reviewer Role

Editor-in-Chief / journal-fit reviewer

### Reviewer Identity

以 *Measurement* 资深副主编视角评估，关注 measurement science 增量、国际读者价值、创新定位、整体证据强度以及稿件是否达到外审门槛。

### Review Focus

本报告只判断期刊适配度、原创性、重要性、论证主线和投稿成熟度，不替代方法学、惯性导航物理或部署审稿人的技术核查。

## Overall Assessment

### Recommendation

- [ ] Accept
- [ ] Minor Revision
- [ ] Major Revision
- [x] **Reject — encourage resubmission after complete reconstruction of the evidence chain**

### Confidence Score

**4/5**。论文主题与 sensor/measurement data processing 高度相关；部分 physics 与 navigation 细节需依赖专门领域审稿人。

### Summary Assessment

本文提出以 bidirectional CfC/LNN 和 BiLSTM 双分支、残差插补与可学习门控为核心的 IMU 缺失数据恢复框架，并尝试通过 physics-informed objective、ATE 和推理性能分析把点级恢复扩展到下游导航价值。该问题具有现实意义，论文也具备清晰的工程主线和较完整的图表框架。然而，以 *Measurement* 的编辑门槛衡量，当前稿件尚未证明其对 measurement science 的独立增量：混合架构的贡献、所谓 physical constraint 的真实含义以及 ATE 改善并未形成一条可追溯且相互一致的证据链。摘要和结果使用 “consistently outperforms”“state-of-the-art”“robustness”“real-world deployment”等强表述，但正文只展示单一公开数据集、人工缺失、有限基线和缺乏统计不确定性的点估计。更严重的是，论文描述的训练配置、损失和若干指标不能从当前仓库中的主实验输出唯一重建。由于修复需要重构数据管线并重跑核心实验，而非仅修改文字，我不建议当前版本进入常规外审；但主题和双分支探索仍值得在严格重建后重新投稿。

## Strengths

### S1：问题具有明确的测量与应用价值

引言指出 IMU 缺失会通过积分传播到 motion reconstruction（PDF pp.1–2；LaTeX 132–140），并将信号误差与下游轨迹误差联系起来。这比仅报告 normalized RMSE 更接近测量系统的实际效用。

### S2：架构动机易于理解

论文将 BiLNN 定位为 local/irregular dynamics 分支，将 BiLSTM 定位为 long-range dependency 分支，并通过 per-time-step、per-channel gate 融合（PDF pp.2、5；LaTeX 138–145、234–238）。即使其独立增量仍需证明，这一设计叙事对读者是清楚的。

### S3：没有回避计算代价

部署部分明确报告 Hybrid 参数最多、速度最慢，并承认当前版本难以满足 strict real-time use（PDF pp.10–11；LaTeX 411–432）。这种负面结果披露比仅宣传性能提升更可信。

### S4：稿件已具备完整实验章节的雏形

稿件包含 missing-rate comparison、loss ablation、gate visualization、trajectory visualization、hyperparameter study 与 forward profiling。若这些模块通过统一 protocol 重跑，可形成较有说服力的完整实验包。

## Weaknesses

### W1：尚未形成可识别的 measurement-science 贡献

**Problem**：核心贡献目前主要是已有 BiLNN、BiLSTM、residual interpolation 和 gating 的组合。稿件没有定义新的 measurement model、uncertainty model、calibration procedure 或可推广的物理估计理论。Introduction 的三条 contribution（LaTeX 142–147）更多描述“做了什么”，而不是“相对于最接近方法新在哪里”。  
**Why it matters**：*Measurement* 官方 scope 明确警告，仅将 AI 工具应用于已知问题、且缺少 measurement context 和 reproducibility 的稿件可能被 desk reject。  
**Suggestion**：以可检验命题重写贡献：分别隔离 residual baseline、BiLNN、BiLSTM、gate、physical term 和 window features；增加 measurement uncertainty、bias/drift 或 calibration 的明确理论/实验目标，并与最接近的 IMU imputation 方法直接比较。  
**Severity**：Critical

### W2：中心标签 “physics-informed” 与展示证据不成比例

**Problem**：标题、摘要、highlights 和结论均把 physics-informed 作为主创新（LaTeX 54–60、103–117、449–453），但方法只给出 acceleration–velocity differential equation（LaTeX 210–218），且全文没有说明 velocity 如何由模型产生或作为监督量进入训练。  
**Why it matters**：当标题中的核心术语无法由方法和消融唯一支撑时，会构成 over-promising；编辑无法判断论文是 scientific machine learning，还是普通 derivative/trajectory regularization。  
**Suggestion**：在重跑前先确定物理建模层级：若没有 velocity state 和坐标变换，应改称 kinematic/temporal consistency regularization；若坚持 physics-informed，应建立状态、单位、frame、观测与损失之间的完整定义并给出物理残差验证。  
**Severity**：Critical

### W3：主要结果缺少统一、可追溯的实验身份

**Problem**：论文称 50 epochs、sequence length 50、五次重复（LaTeX 220–224、244、433–446），而与表 1 多组 RMSE 精确对应的保存结果来自 `results/bidirectional_lnn_residual_tra/missing_rate_comparison_20260310_151546.csv` 和 `results/baseline_Imputation_Method/missing_rate_comparison_20260311_213735.csv`；相应主脚本默认 20 epochs、sequence length 30 和单一 seed 2026（`experiment_bidirectional_lnn_residual.py` 554–585；`experiment_baseline_imputation_comparison.py` 365–397）。仓库中没有与表格逐行绑定的 run manifest。  
**Why it matters**：编辑无法确定表格、消融、图和部署结果是否属于同一 protocol，也无法验证“五次重复”的声明。  
**Suggestion**：为每张表和图提供 immutable experiment ID、完整 config、commit hash、checkpoint hash、raw per-run metrics 和 aggregation script；只使用统一重跑结果撰稿。  
**Severity**：Critical

### W4：结论显著超出单数据集人工缺失实验

**Problem**：稿件将结果概括为 state-of-the-art、robustness、generalization 和 real-world deployment potential（LaTeX 278–280、411–414、449–453），但数据仅来自 OxIOD 八类场景，缺失由 Bernoulli 或整通道 mask 人工生成（LaTeX 241–252），没有真实 packet-loss logs、真实 sensor dropout、跨设备/跨数据集验证或 domain shift。  
**Why it matters**：这会误导读者把“对一个数据集的合成遮挡有效”理解为“真实 IMU 故障鲁棒”。  
**Suggestion**：将结论限定为 OxIOD synthetic masking；增加 held-out scenario/device、真实 dropout 或至少 distribution-shift stress tests，并移除未经验证的 SOTA 与部署措辞。  
**Severity**：Major

### W5：稿件的出版完成度仍不足

**Problem**：存在图注错配（PDF p.8 的 Figure 6 实为轨迹图，caption 却重复 missing-pattern 说明；LaTeX 344–352）、表 2 中 ATE 的 0.01 与百分数 1.xx 混用（LaTeX 378–393）、术语和英文语法问题，以及末页参考文献大面积留白。  
**Why it matters**：这些问题降低可信度，也提示图表可能来自不同版本拼接。  
**Suggestion**：完成证据重跑后再统一生成图表、caption、单位和 cross-reference，并进行专业英文编辑与最终 PDF QA。  
**Severity**：Major

## Detailed Comments

### Journal Fit

主题在 *Measurement* 的 measurement data processing、performance analysis 和 sensor-system algorithms 范围内，但当前 contribution 更像 architecture application。若不增加测量学增量，更现实的去向是 *IEEE Sensors Journal* 或 *Measurement: Sensors*；即使转投，数据泄漏、物理定义和结果可追溯问题仍必须解决。

### Originality

双向 LNN 与双向 LSTM 的 gated hybrid 具有一定组合新颖性，但论文没有与最接近的 hybrid time-series imputation、neural ODE/CfC imputation、BRITS/SAITS/TimesNet 类方法建立清楚差异。门控图只表明权重存在非均匀分布，不能独立证明分支学到了作者赋予的 “local vs global” 功能。

### Significance

若严格成立，研究可为 wearable/mobile inertial sensing 的 dropout recovery 提供价值。当前影响范围应限定为短窗口、离线双向插补；不能外推为在线导航或嵌入式实时恢复。

### Structural Coherence

标题—摘要—贡献—结果表面一致，但方法与实现、结果与保存输出、Vicon ground truth 与 full-IMU reference 之间存在多处身份切换。Discussion 主要重复动机，没有系统回答 validity threats、failure cases 与适用边界。

### Title & Abstract

标题过长，且 “Physics-Informed Neural Temporal Hybridization” 暗示了比当前实现更强的理论创新。摘要应给出数据集、测试划分、主要指标的单位与不确定性，同时删除无法由统计检验支持的 “consistently” 和无法由机制实验支持的 “confirms”。

### Conclusion

“maintain the original data distribution”（LaTeX 453）没有任何 distributional metric 支撑；“mitigates drift and noise”（LaTeX 449–451）也没有 drift/noise-controlled experiment。结论需要改为与实际实验严格对应，并明确 offline bidirectional、synthetic missingness 和 single-dataset 限制。

## Questions for Authors

1. 表 1、表 2、Figure 6–10 各自对应哪个 script、commit、config、checkpoint 和随机种子？能否提供逐项 manifest？
2. 论文所称“五次重复实验”的五个 raw run 文件在哪里？表格展示的是 mean、median 还是单次结果，为什么没有 SD/CI？
3. 作者希望把本文定位为新的 measurement method、IMU-specific scientific ML，还是 hybrid time-series architecture？不同定位需要不同的理论和基线。
4. 如果删除 physics-related claims，仅保留 gated BiLNN–BiLSTM，论文的哪项主要实验结论会改变？

## Minor Issues

- Short title 中出现 “BiLLSTM”/“BiLSTM”/“Hybrid BiLLSTM” 多种命名，应统一。
- Keyword “Physical Informed” 应为 “Physics-Informed” 或 “Physically Informed”。
- Abstract 第一句 `reconstruction.However` 缺空格（LaTeX 104）。
- “approximately 2000,000” 应改为 “approximately 2,000,000”（LaTeX 242）。
- Figure 6 caption 与内容不符；正文还错误引用 Figure 8 讨论 Figure 6 的轨迹位置（LaTeX 347）。
- Table 2 标题应为 “under different loss functions”，并统一 ATE 单位。

## Dimension Scores

| Dimension | Score | Descriptor | Notes |
|---|---:|---|---|
| Originality (20%) | 57 | Weak | 有组合新颖性，但相对最接近方法的增量未隔离 |
| Methodological Rigor (25%) | 34 | Insufficient | 核心 protocol 与结果身份无法统一追溯 |
| Evidence Sufficiency (25%) | 38 | Insufficient | 单数据集、人工缺失、无统计不确定性，强结论证据不足 |
| Argument Coherence (15%) | 46 | Weak | 论文叙事清楚，但 physics/ATE/deployment 证据链断裂 |
| Writing Quality (15%) | 48 | Weak | 可读但存在大量语言、术语、图注和单位问题 |
| **Weighted Average** | **42.1** | **Reject** | 需重构实验后重新投稿，而非文字性大修 |

## Protocol Note

安装的 `academic-paper-reviewer` 技能包缺少其声明的 `shared/contracts/reviewer/full.json`、schema 与 validator，因此本报告按该技能的标准 EIC 单阶段模板执行，未伪造 sprint-contract 分数。
