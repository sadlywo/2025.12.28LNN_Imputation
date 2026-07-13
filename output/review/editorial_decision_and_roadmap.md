# Editorial Decision Package

## 1. 审稿材料与协议边界

本综合完整读取并仅依据以下五份 Phase 1 独立报告：`eic_review.md`、`reviewer1_methodology.md`、`reviewer2_domain.md`、`reviewer3_perspective.md`、`devils_advocate.md`。本文档是 Phase 2 editorial synthesis，不是第六份审稿；未新增报告之外的缺陷，也未修改 manuscript 或源码。

四位标准审稿人（EIC、R1、R2、R3）参与 consensus 计算；Devil’s Advocate（DA）不计入票数，其 CRITICAL findings 独立处理。

## 2. Report Inventory

| Reviewer | Recommendation | Confidence | Weighted score | 核心优点 | 核心弱点 | Questions / Minor items |
|---|---|---:|---:|---|---|---:|
| EIC | Reject — encourage resubmission after complete reconstruction | 4/5 | 42.1 | 问题具有 measurement/application value；双分支叙事清楚；主动披露速度代价；实验章节框架较完整 | measurement-science 增量未建立；physics-informed 证据不足；结果身份不可追溯；单数据集 synthetic masking 支撑不了 SOTA/generalization/deployment；出版完成度不足 | 4 / 6 |
| R1 Methodology | Reject — 方法重建与全量重跑后新稿重投 | 5/5 | 39.0 | 文件级 split 的意图正确；代码 reconstruction loss 针对 missing positions；固定评估 mask 与 validation checkpoint 有初步可重复性；同时评价 RMSE 与 downstream consequence | target leakage；physics loss 身份和 `dt` 错误；短窗 ATE construct invalidity；无 five-seed/CI/provenance；split、missingness、test tuning 与正文配置不一致 | 8 / 9 |
| R2 Domain | Reject — complete trajectory-pipeline revalidation 后重投 | 5/5 | 43.6 | downstream trajectory 问题重要；OxIOD 适合作为起点；保留 `dt` 的方向合理；尝试检验导航代价 | Vicon nanosecond/IMU second 未换算；缺 body/world frame mechanization；Eq. (8) 无匹配实现；短窗 oracle initialization 与非标准 ATE；近邻 imputation/inertial literature 缺失 | 4 / 12 |
| R3 Perspective | Major Revision | 5/5 | 52.6 | 离线 residual hybrid 有工程潜力；profiling 基础动作合理；承认实时限制；signal→downstream→systems cost 视角有价值 | CfC 未传 `timespans`；channel masking 不等于 irregular sampling；bidirectional look-ahead 与实时叙事冲突；gate 解释无干预证据；4090D batch-forward 不能证明 edge deployment 且 Table 3 provenance 不闭合 | 4 / 7 |

四位标准审稿人的共同评分结构非常一致：Originality 尚有增量价值，但 Methodological Rigor、Evidence Sufficiency 与 Argument Coherence 是主要失分项。R3 分数较高并非否认缺陷，而是把模型视为可通过大修保留的 offline/fixed-lag imputer。

## 3. Consensus Analysis

### 3.1 Points of Agreement

1. **[CONSENSUS-4] 研究问题与 downstream-oriented evaluation 方向有价值。** EIC、R1、R2、R3 均认可 IMU gap imputation 以及同时考察 signal error、trajectory consequence 和 systems cost 的研究动机。保留方向：继续以 missing-point fidelity 与 downstream utility 为双主线，但须使用有效指标。

2. **[CONSENSUS-4] 当前强结论超出证据边界。** 四人均认为现有证据不能支持 `state-of-the-art`、`generalization`、`real-world robustness`、`physics-informed` 或 `embedded/real-time deployment` 等强措辞。共同要求是：要么补足对应实验，要么把结论限定为 single-dataset、synthetic masking、offline bidirectional imputation。

3. **[CONSENSUS-4] 结果与 artifact provenance 不闭合。** EIC、R1、R2 指出 Tables 1–2 与 CSV/config/run identity 不一致；R3 对 Table 3 得出同样结论。共同要求：由 immutable experiment manifest 和统一脚本自动生成所有表图，逐项关联 commit、config、seed、checkpoint 与 raw per-run/per-record metrics。

4. **[CONSENSUS-4] `irregular sampling`/continuous-time 叙事没有由当前实验验证。** EIC 认为 local/irregular mechanism 未被隔离；R1、R2、R3 均指出 value/channel masking 保留规则时间网格，且 R1/R3 明确指出 CfC 未把实际 `dt` 传入 `timespans`。共同要求：区分 missing values、block/channel outage 与 irregular timestamps；若保留 continuous-time claim，必须接入真实 elapsed time 并做受控实验。

5. **[CONSENSUS-4] 需要更强、输入公平的 baselines 与 mechanism ablations。** 四人均认为现有比较不足以把增益归因于 BiLNN–BiLSTM complementarity、gate 或 physics term。共同要求包括 closest time-series imputers、linear interpolation anchor、单分支/等容量/fixed-gate 等对照；具体组合以可复现且公平的 input budget 为前提。

6. **[CONSENSUS-4] offline/online service boundary 与 deployment evidence 必须重写。** 四人均拒绝从双向未来信息和 desktop GPU batch-forward 直接外推实时/嵌入式能力。共同要求：明确 offline、fixed-lag 或 causal contract；不补目标硬件与端到端 benchmark 时，只能声称 offline/delayed post-processing。

7. **[CONSENSUS-3] physics/trajectory evidence chain 不成立。** EIC、R1、R2 一致认为 published physics loss、实现、变量、单位与 ATE evidence 无法闭环；R3 未对惯性导航物理作专业裁决。R3 不是反对者，而是主动限定了评审范围。按专业优先原则，采用 R2（domain, confidence 5）与 R1（methodology, confidence 5）的判断：该链路必须作为 P1 重建。

8. **[CONSENSUS-3] five-repeat、uncertainty 与 independent-unit statistical reporting 不足。** EIC、R1、R2 均指出单一 point estimate、无可追溯 five seeds、SD/CI/effect size 或 per-record statistics；R3 仅在 deployment profiling 中提出重复与 tail statistics，未全面评价主实验统计。作者必须以 recording 为分析单位重跑并报告不确定性。

9. **[CONSENSUS-3] 当前短窗 ATE 不能证明 long-term drift suppression。** EIC、R1、R2 均认为轨迹证据不足；R1/R2 具体指出 overlapping short windows、Vicon-derived reset、frame/integration 问题。R3 未裁决导航指标。编辑采用 R2 的领域判断。

### 3.2 Disagreements and Editorial Resolution

本轮不存在真正的 2:2 或 2-1-1 实质方向冲突，因此不人为制造 **[SPLIT]**。实际分歧均为 severity 或 perspective 差异：

#### Disagreement 1：Reject 还是 Major Revision

- **EIC / R1 / R2**：建议 Reject，原因是修复会改变数据管线、loss、trajectory evaluation 和全部核心结果，不属于维持当前 evidence chain 的常规大修。
- **R3**：建议 Major Revision，认为模型作为 offline/fixed-lag imputer 仍有明确工程潜力，continuous-time、gate 与 deployment 部分可以通过重做实验修复。
- **Type**：Severity disagreement（3:1，属 **[CONSENSUS-3]**，R3 dissent）。
- **Editor’s Resolution**：**Reject — Resubmit Encouraged**。
- **Basis**：三位 Reject reviewers 的 confidence 为 4/5、5/5、5/5，且问题触及 input validity、ground-truth synchronization、physical construct 与 published numbers。根据 decision standard，“Major + Reject + Reject + Reject”应为 Reject。R3 关于可修复性的判断用于确定 `Resubmit Encouraged` 子类型，而非降低当前决定。

#### Disagreement 2：论文应保留 physics-informed identity，还是重定位为 offline hybrid imputer

- **EIC**：要求作者先决定 measurement method、IMU-specific scientific ML 或 hybrid architecture 的核心身份；physics 不成立时应降级命名。
- **R1 / R2**：认为当前 Eq. (8)、实现、`dt`、frame 与 trajectory evidence 不能支撑 physics-informed identity；只有完成 dimensionally valid reconstruction 后才能保留。
- **R3**：不裁决惯性物理，但认为 residual gated model 作为 offline/fixed-lag imputer 仍有独立价值。
- **Type**：Perspective difference，而非对事实相反判断。
- **Editor’s Resolution**：默认以 **offline bidirectional hybrid imputer** 作为重投稿件的最低可辩护身份；只有 P1 physical closure 全部验收后，才可恢复 `physics-informed` 到标题、摘要和贡献。
- **Basis**：专业问题服从 R2 的 domain confidence 5 和 R1 的 methodology confidence 5；R3 的价值判断保留为无 physics 标签时的替代投稿路径。

#### Disagreement 3：deployment claim 应删除还是补实验保留

- **EIC / R1 / R2**：在当前证据下删除或显著收缩 deployment/real-time/embedded claims。
- **R3**：给出可保留该方向的条件路线：causal/fixed-lag 对照、真实输入、single-stream、end-to-end、目标硬件、tail latency 与 energy。
- **Type**：Direction disagreement（立即删减 vs 条件性重建）。
- **Editor’s Resolution**：当前稿必须删减；重投稿件只有完成对应 P2 benchmark acceptance criteria 后才能恢复条件性 deployment claim。
- **Basis**：两条路线并不矛盾；在证据产生前采用保守措辞，产生后再按结果决定。

## 4. Devil’s Advocate CRITICAL Register

DA 的五项 CRITICAL findings 均逐项进入决定；任何一项未关闭，决定均不可能为 Accept。

| ID | DA argument | 其他审稿人佐证 | EIC assessment | Required author response |
|---|---|---|---|---|
| C1 | Table 1 宣称 Hybrid random-missing ATE 最优，但可追踪 CSV 中 Transformer（及 MICE）更低；正文数字来源不明 | R1 W4/Results Integrity；R2 W4；EIC W3 | EIC 将主结果缺乏统一、可追溯实验身份列为 Critical，支持该质疑 | 从原始 checkpoint 按冻结 protocol 重算；提供每个表格单元格的 unique experiment ID、raw metric 与生成脚本；无法追溯的数字全部删除 |
| C2 | Eq. (8) 是 velocity–acceleration constraint，但实现为 cumulative target matching；无 velocity state，且 `inputs[:,:,-1:]` 实际读取 target-derived statistic 而非 `dt` | EIC W2；R1 W2；R2 W3；R3 W1 对真实 elapsed-time 路径提供侧面佐证 | EIC 明确认定 physics-informed 标签与证据不成比例，并要求重建状态、单位、frame、observation 与 loss | 选择唯一 objective；给出 equation–code–tensor-index–unit–frame mapping 和 unit tests；用 true `dt`；在只改变 physics term 的条件下全量重跑消融 |
| C3 | CSV 对照显示 physics-aware variant 的三种模型 ATE 均略变差，部分 RMSE 也变差，与 Table 2“均改善”方向相反 | R1 W4/Results Integrity；R2 W4；EIC W2–W3 | EIC 不接受现有 physics ablation 作为中心证据，并要求统一重跑 | 发布 reconstruction-only 与 physics variant 的 paired raw results、CI/effect size；按真实结果重写结论，不得选择性保留改善方向 |
| C4 | 完整 target 的 window statistics 进入输入，同时 test RMSE/ATE 用于选 λ、L、hidden size，形成 input-side target leakage 与 selection-side test leakage | R1 W1、W5、Hyperparameter audit；R3 W4 对 gate 输入含 target-derived features 的部分佐证 | EIC 未单列 24-D leakage，但其 Critical provenance/reproducibility 判断要求重构数据管线；综合按 R1 的 methodology confidence 5 采纳 | 删除 target-derived input；normalizer 只在 train recordings 拟合；超参数只用 validation；添加“改变 masked target 不改变 input”单元测试；全部结果从头重跑 |
| C5 | ATE 忽略 frame/attitude、短窗用 GT 初始化、积分模型在 observed points 的改写值；定量/定性 reference 不同；gate 约 5%–9% LNN weight，不能证明 switching，增益可由 smoothing/residual/capacity/BiLSTM dominance 解释 | EIC W2 与 gate 解释限制；R1 W3、Loss Audit；R2 W1–W4；R3 W4 | EIC 认为 physics/ATE/deployment evidence chain 断裂，且 gate visualization 不能证明 local/global functional split | 重建同步、frame-aware full-record trajectory metric，并显式 preserve observed values；做 fixed/shuffled gate、linear-only、equal-capacity 与 branch ablations；把 visualization 降级为 descriptive，除非 intervention 支持机制解释 |

## 5. Editorial Decision Letter

Dear Author(s),

感谢提交题为 *Physics-Informed Neural Temporal Hybridization for Missing IMU Data Imputation: An Integrated Bidirectional LNN-LSTM Hybrid Model* 的稿件。审稿组一致认可该问题的应用价值：将 missing-point reconstruction 与 downstream trajectory consequence、systems cost 同时评价，是值得继续发展的方向；双分支 residual hybrid 也可能成为有效的 offline/fixed-lag imputer。

### Decision: Reject — Resubmit Encouraged

本决定不是因为研究问题缺乏价值，而是因为当前版本的核心 evidence chain 无法维持。四位标准审稿人中 EIC、R1、R2 均建议 Reject，R3 建议 Major Revision；三位 Reject reviewers 的关键判断分别覆盖 journal fit、methodology/reproducibility 与 inertial-navigation physics，且 confidence 为 4/5、5/5、5/5。当前问题包括 target-derived inputs 与 test tuning、Vicon/IMU timestamp mismatch、frame/integration/ATE construct failure、published physics loss 与实现不一致、实际 `dt` 路径错误，以及 Tables 1–3 与保存 artifacts 无法统一追溯。这些缺陷会改变训练输入、模型排序、physics ablation 与 downstream conclusion，不能通过文字澄清或局部补实验解决。

我们同时采纳 R3 的积极判断：模型作为离线或固定延迟的 hybrid imputer 仍可能成立。因此决定采用 `Resubmit Encouraged`，而非否定研究路线。重投前应按照下述 P1 顺序先关闭数据与物理有效性，再全量重跑并重建 provenance；只有在新结果支持时，才恢复 physics-informed、continuous-time、SOTA、generalization 或 deployment claims。新稿应被视为证据链重建后的新 submission，并接受完整外审。

Sincerely,

Managing Editor / Editorial Synthesizer

## 6. Executable Revision Roadmap

### P1 — Structural / Validity Revisions（11 项，预计总计 10–16 周；存在并行空间）

| ID | Executable revision | Sources | Acceptance criteria | Estimated effort |
|---|---|---|---|---:|
| P1-1 | 消除 target leakage 与 split-specific preprocessing leakage | R1 W1/W5；R3 W4；DA C4 | 输入仅由 observed values、mask、elapsed time 计算；scaler 只 fit train；val/test 仅 transform；单元测试证明改变 masked target 不会改变 model input | 4–7 days |
| P1-2 | 修复 Vicon/IMU timestamp unit 与 association | R2 W1；DA C5 | Vicon nanoseconds 正确换算；每个文件报告 overlap、offset/association；无 silent extrapolation；测试确认 interpolated Vicon trajectory 非常量且与 source displacement/velocity 一致 | 4–7 days |
| P1-3 | 建立 frame-aware inertial mechanization | R1 W3；R2 W2/Theoretical Framework；DA C5 | 明确 body/device/world/Vicon frames、quaternion convention、gravity、attitude source、extrinsics、bias 与 units；stationary/known-motion tests 通过；离散积分公式与代码一致 | 2–4 weeks |
| P1-4 | 冻结唯一、可审计、量纲一致的 reconstruction + physics/kinematic objective | EIC W2；R1 W2/Loss Audit；R2 W3；DA C2/C3 | Eqs. (7)–(9) 与唯一实现一一映射；`1-M`/同刻目标/denominator 正确；true `dt` index；定义 velocity/state/frame/normalization；unit/dimension tests 通过 | 1–2 weeks |
| P1-5 | 重设计 split、missingness 与 validation-only model selection | EIC W4；R1 W5；R2 W5；R3 W2；DA C4/M1–M3 | 发布固定 train/val/test manifest；scenario/user/device 隔离策略明确；区分 MCAR/MAR/MNAR 与 point/block/channel/timestamp topology；报告 realized missing fraction；λ/L/hidden/epoch 仅用 validation，test 一次性 | 1–2 weeks |
| P1-6 | 重建 full-record downstream evaluation | EIC W2–W4；R1 W3/Evaluation；R2 W4；DA C5 | held-out complete recordings 单次初始化、无滑窗 GT reset；observed values 保持原样，仅填 missing points；报告明确定义的 ATE-RMSE、RPE/RTE、endpoint drift、velocity error 与 imputation-induced Δmetric，单位统一 | 2–3 weeks |
| P1-7 | 建立 immutable experiment provenance 并全量重跑 | EIC W3；R1 W4/Reproducibility；R2 W4；R3 W5；DA C1/C3 | 每个 run 记录 commit、environment、config、seed、checkpoint hash、split/mask manifest、per-record outputs；Tables 1–3 与 Figures 由单一 versioned pipeline 自动生成；任意单元格可反向追踪 | 2–4 weeks |
| P1-8 | 以 independent recording 为单位完成重复与统计推断 | EIC W3–W4；R1 Statistical Remedy；R2 W4 | ≥5 个预设 seeds；per-record paired values；mean±SD、median/IQR、95% paired CI、effect size；预设 primary contrast 并处理多重比较；禁止把 overlapping windows 当独立样本 | 1–2 weeks（可随 P1-7 并行） |
| P1-9 | 补足公平 baseline 与 attribution ablations | EIC W1/Originality；R1 Baseline Fairness；R2 W5/Missing References；R3 W4；DA Alternatives | 至少包含 linear interpolation only、适当传统 baseline、BRITS/SAITS/CSDI 或书面合理排除；equal-capacity single branch、fixed gate 0/0.5/1、simple averaging、no-window-feature；统一 input information budget 和 tuning protocol | 2–4 weeks |
| P1-10 | 真实接入 continuous-time semantics 并分离 irregular-time experiments | EIC Originality；R1 W5；R2 W5；R3 W1–W2；DA M1 | CfC 使用 forward/reverse 正确 `timespans`；完成 actual-dt、constant-step、dt-as-feature、no-dt 消融；独立测试 timestamp jitter/thinning/burst/asynchronous cases；若不做则删除 irregular/continuous-time performance claim 与 NCP sparse-wiring 表述 | 2–3 weeks |
| P1-11 | 重定论文 identity 与 claim envelope | EIC W1/W2/W4；R1 Overgeneralization；R2 Contribution；R3 W3/W5；DA C1–C5 | 标题、摘要、highlights、contributions、Discussion、Conclusion 与重跑证据一致；未验收 P1-2/3/4/6 不得使用 physics-informed；未验收 P1-10 不得声称 irregular-time advantage；未完成目标平台证据不得声称 real-time/embedded；删除无统计支撑的 SOTA/consistently/significantly | 3–5 days（在结果冻结后） |

### P2 — Evidence Strengthening（建议完成，预计 4–8 周，可与后期 P1 并行）

| ID | Executable revision | Sources | Acceptance criteria | Estimated effort |
|---|---|---|---|---:|
| P2-1 | 定义 offline/causal/fixed-lag deployment contract 与性能–look-ahead 曲线 | EIC W4；R1 Baseline/Deployment；R2 W5；R3 W3 | 报告 causal、fixed-lag `k`、full bidirectional 的 accuracy、algorithmic latency 与 compute latency；明确服务时限、未来样本预算和适用场景 | 1–2 weeks |
| P2-2 | 重做 target-hardware profiling | EIC S3/W4；R1 Baseline Fairness；R2 W5；R3 W5 | 唯一 benchmark manifest；真实 dataset batch；batch=1 median/p95/p99、throughput、end-to-end preprocessing/inference/postprocessing、peak memory、energy；至少一个实际 target platform；量化时验证精度 | 2–3 weeks |
| P2-3 | 以干预而非 heatmap 验证 gate/uncertainty | EIC Originality；R1 Hypotheses；R3 W4；DA C5 | fixed/shuffled/swapped gate、移除 uncertainty/window features；conditional branch error、gate entropy 与 error correlation；若 uncertainty 保留则给 held-out calibration；figure manifest 完整 | 1–2 weeks |
| P2-4 | 重构 closest-literature positioning | EIC W1/Originality；R1 Baselines；R2 W5/Missing References；R3 Reading Recommendations | Related Work 分为 time-series imputation、inertial mechanization/learned odometry、continuous-time models；准确比较 BRITS/SAITS/CSDI、RIDI/IONet/RoNIN/TLIO、CfC/LTC；每项 contribution 对应 closest comparator | 4–7 days |
| P2-5 | 增加 external-validity 与真实故障 stress tests | EIC W4；R1 W5；R2 W5；R3 W2；DA M2–M3 | 至少 held-out scenario/user/device 或跨数据集；如无真实 dropout logs，明确 synthetic limitation，并测试 distribution shift、gap duration 与 channel outage；不得把 synthetic MCAR 外推为真实 malfunction | 2–4 weeks |
| P2-6 | 扩展 failure-case 与 physical-unit reporting | R1 Evaluation/Statistical Remedy；R2 W4；DA Alternative 4–5 | accelerometer/gyro、各轴、gap length、scenario 分层；同时给 normalized 与 physical-unit metrics；公开 smoothing 改善 ATE 但损害 signal fidelity 的失败案例 | 1–2 weeks |
| P2-7 | 固化可复现环境与 artifact policy | EIC W3；R1 Reproducibility；R2 W4；R3 W5 | 提供 lockfile/environment export、exact `ncps`/PyTorch/CUDA versions、deterministic settings、checkpoint reuse policy、data/table generation commands | 2–4 days |

### P3 — Text, Figures, and Publication QA（预计 1–2 周）

| ID | Executable revision | Sources | Acceptance criteria | Estimated effort |
|---|---|---|---|---:|
| P3-1 | 统一术语、模型名与方法配置 | EIC Minor；R1 Minor；R2 Minor；R3 Minor | 全文统一 `BiLNN–BiLSTM hybrid`、`physics-informed`、actual epochs/seq_len/hidden/optimizer；不再混用 BiLLSTM/GRU-D/MICE/Z-score 等与实现不符名称 | 1–2 days |
| P3-2 | 修复 caption、cross-reference、gate direction 与单位 | EIC W5/Minor；R1 Minor；R2 Minor；R3 W4/Minor；DA m2 | Figure 6 caption 与轨迹内容匹配；Figure 8 明确 gate 是 LNN 或 BiLSTM weight；Table 1–3 单位/统计量/batch/sample 定义一致；所有引用指向正确图表 | 2–3 days |
| P3-3 | 统一 ATE/RMSE/%/m/cm 与增幅公式 | EIC W5；R1 Results Integrity；R2 W4；DA C1/C3 | 每个 metric 只使用一个明确定义和单位；Table 1 增幅列公式与标签一致；Table 2 不再把 meter-like values 标为 percent | 1 day |
| P3-4 | 完成专业英文与 claim-level copyedit | EIC W5/Minor；R1 Minor；R2 Minor；R3 Minor；DA m3 | 修复 grammar、`reconstruction.However`、`2,000,000`、constant-time complexity、bias terminology；所有 `confirms/ensures/mitigates` 均有对应证据或降级 | 2–4 days |
| P3-5 | 最终 PDF visual QA | EIC W5；R2 Results/PDF；R3 Figures/Layout | 表图字号可读；无裁切/重叠；dense tables、colorbars、末页留白和 references layout 经逐页检查；figure/table provenance 随 supplementary materials 提供 | 1–2 days |

### Recommended execution order

`P1-1/2/3/4/5 → P1-6 → P1-7/8/9/10 → P1-11 → P2 evidence strengthening → P3 publication QA`。

任何新稿均应附 point-by-point response 和 change/provenance matrix。由于本次决定为 Reject—Resubmit Encouraged，不设常规 revision deadline；建议按 10–16 周的 P1 重建周期规划，未经 P1 acceptance audit 不进入文字改写阶段。

## 7. Reviewer Report Summaries

### EIC Summary

- **Recommendation / Confidence / Score**：Reject—Resubmit Encouraged；4/5；42.1。
- **Summary**：主题符合 sensor/measurement data processing，但当前贡献更像 architecture application；physics、ATE、deployment 与结果身份没有形成 Measurement 所需的 measurement-science evidence chain。优点是问题重要、架构动机清楚、实验模块齐备且没有回避计算代价。

### Reviewer 1 Methodology Summary

- **Recommendation / Confidence / Score**：Reject；5/5；39.0。
- **Summary**：完整 target-derived window features、test-set tuning、loss identity mismatch、错误 `dt`、短窗 ATE、无 five-seed/statistics 以及 table–CSV 不一致共同使主结果无效。必须先消除 leakage、冻结 protocol，再从头重跑。

### Reviewer 2 Domain Summary

- **Recommendation / Confidence / Score**：Reject；5/5；43.6。
- **Summary**：Vicon nanoseconds 与 IMU seconds 未转换导致 interpolated ground truth collapse；device-frame acceleration 未旋转到 world frame，短窗 oracle initialization 与非标准 ATE 不能支持 navigation/physics claims。建议完整重建 synchronization、mechanization 与 trajectory evaluation。

### Reviewer 3 Perspective Summary

- **Recommendation / Confidence / Score**：Major Revision；5/5；52.6。
- **Summary**：残差双向 hybrid 作为 offline/fixed-lag imputer 有工程潜力，但真实 `timespans` 未进入 CfC、channel masking 不等于 irregular sampling、future look-ahead 未计入 latency、gate heatmap 无因果证据、4090D batch-forward 不能证明 embedded feasibility。其 Major 判断构成重投可行性的主要正面依据。

### Devil’s Advocate Summary

- **Recommendation/score**：DA 不参与 recommendation voting 或 rubric scoring。
- **Summary**：最强替代解释是 target leakage、linear interpolation residual、capacity/ensemble、smoothing 与 short-window GT reset，而非 physics 或 branch complementarity。其 C1–C5 分别针对表格反证、physics foundation collapse、physics benefit 方向相反、双重 leakage、ATE/gate attribution；五项均已纳入 P1 acceptance criteria。

## 8. Protocol Fallback Note

五份 Phase 1 报告一致声明：本次材料未提供技能文档所称 v3.6.2 executable sprint contract、paper-blind precommitment，以及对应 schema/validator。因而本轮不能诚实执行 mechanical contract aggregation，也没有伪造 `block/warn/pass` matrix、failure-condition arithmetic 或 contract-derived decision。

本综合采用 academic-paper-reviewer 的标准 fallback：完整 inventory 四位标准审稿人；只在 EIC/R1/R2/R3 中计算 consensus；按 evidence、expertise、confidence 与 conservative principle 仲裁；独立处理五项 DA CRITICAL；按 editorial decision standards 作出 Reject—Resubmit Encouraged；所有 roadmap items 均可追溯至具体 Phase 1 报告。该 fallback 不改变 read-only 边界，也不构成 manuscript revision。
