# Peer Review Report

## Manuscript Information

- **Title**: *Physics-Informed Neural Temporal Hybridization for Missing IMU Data Imputation: An Integrated Bidirectional LNN-LSTM Hybrid Model*
- **Manuscript ID**: 未提供
- **Review Date**: 2026-07-11
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role

Peer Reviewer 1 (Methodology & Reproducibility)

### Reviewer Identity

时间序列插补、missingness mechanisms、机器学习评估与可复现性专家，重点审查人工遮挡协议、数据泄漏、训练/验证/测试隔离、指标实现、统计推断和代码—论文—结果的一致性。

### Review Focus

本报告仅评价研究设计、数据处理、损失函数、实验比较、统计报告和可复现性。审查以论文 PDF/LaTeX、当前代码仓库及可用 CSV 为相互独立的证据层；问题清单只用于末端交叉检查，不作为预设结论来源。

---

## Overall Assessment

### Recommendation

- [ ] **Accept**
- [ ] **Minor Revision**
- [ ] **Major Revision**
- [x] **Reject — 鼓励完成方法重建与全量重跑后作为新稿重投**

### Confidence Score

**5/5**。论文所采用的 artificial masking、time-series imputation、deep-learning evaluation 和代码审计均在本审稿人的专长范围内；轨迹积分中的惯性坐标变换问题也可由实现直接核查。

### Summary Assessment

本文提出一个以双向 CfC/LNN 与 BiLSTM 为分支、通过逐通道 gate 融合的 IMU 插补模型，并以 masked RMSE 和由加速度积分得到的 ATE 评价随机缺失与连续缺失。研究问题具有应用价值，文件级划分、固定评估 mask、missing-only reconstruction objective 及下游轨迹评价的设计意图值得肯定。然而，现有证据不能支持论文的主要定量结论：数据管道把由完整 target 计算的 24 维 window statistics 直接输入模型，构成确定性的 target leakage；论文所定义的 velocity–acceleration physics loss 在主结果代码中没有对应实现，被标为 physics-informed 的结果实际来自另一种 trajectory-aware target-matching loss，且该 loss 读取了错误的 `dt` 通道。ATE 又在每个短滑窗中用 Vicon 真值重置位置和速度，未做姿态旋转/导航坐标转换，因此不能证明 long-horizon drift suppression。论文声称五次重复取均值，但代码只有单一 seed，CSV 无重复、方差或置信区间；表 1、表 2 与现存 CSV 也无法一致追溯。鉴于修复 target leakage 后全部模型、消融、超参数与轨迹实验均须重跑，本稿不适合以常规大修维持现有证据链，建议方法重建后新稿重投。

---

## Strengths

### S1: 采用文件级而非窗口级划分的正确意图

论文明确称按 recording file 做 80/10/10 划分（PDF printed p.6；`Manuscript.tex:244`），实现也在切滑窗之前选择文件（`E:\\VSCode_Study\\2025.12.28LNN_Imputation\\dataset.py:103-141`）。这一顺序原则上能够避免同一 recording 的重叠窗口同时进入 train/test，是合理的基本设计。

### S2: 代码中的核心 reconstruction objective 确实针对人工缺失位置

尽管论文式 (7) 写成 `M` 作用于 observed positions 并带有无法解释的 `t-1` 偏移（PDF printed p.4；`Manuscript.tex:199-203`），主实验代码使用 `1-mask` 且预测与同一时刻 target 对齐（`E:\\VSCode_Study\\2025.12.28LNN_Imputation\\experiment_bidirectional_lnn_residual.py:64-77`；`...\\experiment_baseline_imputation_comparison.py:178-205`）。这表明该处至少有一部分是严重的 manuscript–code documentation error，而不是主实验按错误公式训练。

### S3: 评估 mask 和 checkpoint selection 具备初步可重复性

评估阶段以 window index 固定随机种子并恢复 RNG state（`E:\\VSCode_Study\\2025.12.28LNN_Imputation\\dataset.py:332-355`），checkpoint 则按 validation missing-RMSE 保存（`E:\\VSCode_Study\\2025.12.28LNN_Imputation\\experiment_bidirectional_lnn_residual_trajectory.py:334-365`）。这两点优于直接在 test set 上选 epoch。

### S4: 同时考察 missing-point error 与 downstream consequence 的方向合理

论文没有只报告全序列 MSE，而是试图区分 missing-position RMSE 与 trajectory error（PDF printed pp.6–7；`Manuscript.tex:259-272`）。代码也保留 `test_rmse_missing`、ATE 和 RTE 字段（`E:\\VSCode_Study\\2025.12.28LNN_Imputation\\experiment_bidirectional_lnn_residual_trajectory.py:310-330`）。若物理坐标、积分时域和统计单位被重建，这可形成有价值的 task-oriented evaluation。

---

## Weaknesses

### W1: 完整 target 统计量进入模型，造成直接 target leakage

**Problem**: `CfCIMUDataset.__getitem__` 在生成 mask 后，用未遮挡的 `target_imu` 计算每个窗口的 absolute mean、difference energy、energy 和 variance，并把 24 维统计量重复到每个 time step 作为输入（`E:\\VSCode_Study\\2025.12.28LNN_Imputation\\dataset.py:366-403`）。主实验显式开启 `include_window_features=True`（`...\\experiment_bidirectional_lnn_residual.py:566-585,604-638`；trajectory 版本 `...\\experiment_bidirectional_lnn_residual_trajectory.py:434-472`）。此外，每个 train/val/test 文件的 MAD normalization 由该文件完整值独立计算（`dataset.py:283-287`），不是仅由 train split 拟合并冻结。

**Why it matters**: 模型在推理时获得了只有完整序列才可计算的信息；尤其 variance、energy 与差分能量直接描述 missing region 的真实动态。这会系统性低估插补误差，并使 learned models 与只读取 masked values 的 deterministic baselines 不公平。该缺陷足以使当前全部主结果失效。

**Suggestion**: 删除所有由 `target_imu` 计算的输入特征，或严格改为只由 observed values、mask 和 elapsed time 计算并进行 missing-aware normalization；所有 scaling statistics 只在 training recordings 上拟合，val/test 只 transform。修复后从头重跑全部模型、消融和图表，并提供 leakage unit test，验证改变 missing target 不会改变 model input。

**Severity**: **Critical**

### W2: 论文的 physics-informed loss 与用于生成结果的代码不是同一方法

**Problem**: 论文定义 `||(v_t-v_{t-1})/Δt-a_t||²`（PDF printed p.4；`Manuscript.tex:210-218`），但被称为 physics-informed 的主结果目录对应 `TrajectoryAwareReconstructionLoss`：它比较 predicted acceleration 与 complete target acceleration 的累计和，并不包含 velocity state 或论文式 (8)（`E:\\VSCode_Study\\2025.12.28LNN_Imputation\\experiment_bidirectional_lnn_residual_trajectory.py:66-100,472-476`）。更严重的是，该 criterion 收到的 `dt=inputs[:,:,-1:]`（同文件 `:232-242`），而数据格式中真实 `dt` 位于 index 12，最后 24 维是 window features（`dataset.py:90-91,400-403`）；因此 trajectory loss 实际以最后一个 target-derived window statistic 充当时间步长。仓库中的另一 `PhysicsInformedLoss` 又约束 gyro derivative 与 acceleration magnitude，而非论文公式（`E:\\VSCode_Study\\2025.12.28LNN_Imputation\\models.py:405-523`）。

**Why it matters**: 方法身份、量纲与训练信号均不一致，读者无法知道表 2 验证了哪一种 loss。现有消融不能支持“acceleration–velocity kinematic constraint”或“physics-informed”的因果结论。

**Suggestion**: 冻结一个唯一、形式化定义且有 unit test 的 loss；明确每个 state/label 的来源、coordinate frame、units、mask support 和 normalization。若使用 complete target acceleration/pose，应称为 kinematic auxiliary supervision 或 trajectory-consistency supervision。逐项记录 loss components，并用同一代码路径重跑 reconstruction-only 与 physics variants。

**Severity**: **Critical**

### W3: ATE 协议测量的是短窗、真值初始化的局部偏差，不能证明长期漂移抑制

**Problem**: 数据先以 50% overlap 切成长度 30 的窗口（默认主实验），每个窗口单独积分；`compute_ate` 用该窗口第一个 Vicon 位置初始化，并用前两帧 Vicon/`dt` 计算真值初速度（`E:\\VSCode_Study\\2025.12.28LNN_Imputation\\dataset.py:289-304,475-503`）。它直接积分 device-channel `user_acc`，只做 G→m/s² 转换，没有使用 attitude/quaternion 将 acceleration 转到 navigation/world frame（`dataset.py:225-243,456-468`）。位置更新还使用更新后的 `vel`，并非正文声称的完整 trapezoidal position integration（`dataset.py:497-503`）。

**Why it matters**: 对约 0.3 s 的高度重叠窗口反复用 ground truth reset，会把 drift 截断并产生 pseudo-replication；不经姿态旋转就与 world-frame Vicon 比较，缺乏物理意义。所得约厘米级数值不能外推到 long-horizon navigation robustness。

**Suggestion**: 在完整 recording 上连续积分，不在滑窗边界重置；用姿态将 specific force 转到 navigation frame并处理 gravity/bias，或明确 OxIOD `user_acc` 的 frame 与预处理。分别报告 0.5/1/2/5/10 s 与 full-record ATE-RMSE、RPE、endpoint drift、velocity error；所有指标先按独立 recording 计算，再进行 paired statistics。若使用 GT initialization，应单独命名为 local short-horizon error。

**Severity**: **Critical**

### W4: 数值表无法由现有 CSV 一致复现，且“五次重复”没有运行证据

**Problem**: 论文称所有指标为五次重复平均（PDF printed p.6；`Manuscript.tex:244`），但三个主脚本只设置单一 `seed=2026`，没有 run/seed loop（例如 `...\\experiment_bidirectional_lnn_residual_trajectory.py:434-455,479-518`）；CSV 也只含一个 point estimate。表 1 的 random RMSE 多数可追溯，但 ATE 不一致：例如 Transformer 30% 的 CSV 为 `0.0114511`（`E:\\VSCode_Study\\2025.12.28LNN_Imputation\\results\\baseline_Imputation_Method\\missing_rate_comparison_20260311_213735.csv:24`），论文为 `0.0125`；Hybrid 30% 的 CSV 为 `0.0193931`（`...\\results\\bidirectional_lnn_residual_tra\\missing_rate_comparison_20260310_151546.csv:12`），论文为 `0.0124`。表 2 中 “physics-informed” RMSE 与 trajectory-aware CSV 近似对应（`...\\summary_bidirectional_lnn_traj_20260310_151546.csv:2-4`），但其 ATE 均约 `0.0194`，不是表中的 `1.24–1.27%`，且 reconstruction-only 数值也无法由现存 summary 唯一复现（`...\\results\\bidirectional_lnn_residual\\summary_bidirectional_lnn_20260309_100957.csv:2-4`）。

**Why it matters**: 缺少 run provenance、dispersion 和统一生成脚本，无法判断表中数据来自何种模型、mask、checkpoint、loss 或后处理；这属于 results integrity 与 reproducibility 的核心缺陷。

**Suggestion**: 用 immutable experiment ID 关联 git commit、config、seed、checkpoint、raw per-record predictions 和 derived tables；所有 LaTeX tables 由同一 tidy CSV 自动生成。至少 5 个预先指定 seeds，报告 mean±SD、95% bootstrap CI、paired per-record test 与 effect size。作者须解释每个现存表格单元格对应的源文件与公式；无法追溯者应删除并重跑。

**Severity**: **Critical**

### W5: split、missingness 与 hyperparameter protocol 不能支持“八场景泛化”和 irregular-sampling 结论

**Problem**: 文件列表按固定 scenario 顺序 append 后直接切前 80%、中间 10%、末尾 10%，没有 shuffle/stratification（`E:\\VSCode_Study\\2025.12.28LNN_Imputation\\dataset.py:103-135`）。在当前 45 个 file pairs 中，这使 test set 仅为 `user-2/imu2.csv` 至 `imu6.csv`，并非八场景 test。`channel` 模式用 `int(6×rate)` 选择整通道（`dataset.py:349-352`），因此 nominal 10%、20%、30%、40% 实际为 16.7%、16.7%、16.7%、33.3%；`block` 是每通道固定长度块（`:343-348`），而 random point mask 才是 Bernoulli。所有时间戳仍保留在规则网格，CfC forward 也没有传入 library 的 `timespans` 参数，而是把 `dt` 当普通 feature（`...\\experiment_bidirectional_lnn_residual.py:188-218`）。论文又用 Test RMSE/ATE 选择 `L`、hidden size 与 λ（PDF printed pp.9,11；`Manuscript.tex:361,433-446`），主代码实际默认 `seq_len=30, epochs=20, lnn_hidden=128, AdamW+OneCycleLR`，与正文 `L=50, 50 epochs, Adam` 不符（`...\\experiment_bidirectional_lnn_residual_trajectory.py:422-453,334-336` 对比 `Manuscript.tex:220-224`）。

**Why it matters**: test set 与场景完全混杂，外部有效性无法判断；missing rate 标签不准确，structured missingness 被误写为 irregular sampling；使用 test 指标选超参数造成 test leakage，且正文无法复现实验。

**Suggestion**: 采用 stratified/leave-one-recording-mode-out、leave-one-user-out 或 leave-one-device-out split，公开 manifest；区分 missingness mechanism (MCAR/MAR/MNAR) 与 mask topology (point/block/channel outage)，报告 realized missing fraction。所有超参数仅在 validation set 选择，冻结后一次性 test；主表必须使用冻结 config 并附完整 YAML/JSON。

**Severity**: **Critical**

---

## Detailed Methodological Audit

### 1. Research Questions & Hypotheses

- 论文的目标可概括为：hybrid architecture 与 kinematic regularization 是否改善 missing IMU reconstruction 及 downstream trajectory fidelity。但没有预先定义 primary endpoint、最强 baseline、最小有意义效应或可证伪 hypotheses（`Manuscript.tex:140-147`）。
- “gate 捕捉 local irregular dynamics”“physics loss 抑制 drift”均是机制性因果解释，却没有 fixed-gate、no-window-feature、no-`dt`、oracle gate、coordinate-frame 等必要消融。
- 建议把 primary outcome 预设为 per-record masked RMSE，secondary outcomes 为 long-horizon ATE/RPE，并将 gate/physics claims 对应到具体 ablation contrasts。

### 2. Research Design & Internal Validity

#### 2.1 Data split

- 文件级划分顺序正确，但 deterministic slicing 没有 randomization 或 scenario stratification。当前数据顺序导致训练覆盖前 36 files、validation 主要为 trolley/user-2、test 仅 user-2 5 个 recordings。论文的 “generalization across eight scenarios” 因此是对 training coverage 的描述，不是独立 test evidence。
- 50% overlapping windows（`dataset.py:289-304`）不能当作独立样本用于统计推断；分析单位必须是 recording/user/device，而不是 window。

#### 2.2 Preprocessing leakage

- 论文声称 Z-score（`Manuscript.tex:244`），代码为 per-file median/MAD（`dataset.py:283-287`）。两者不仅命名不同，而且 test recording 使用自身完整 target distribution 做 normalization。
- target-derived window features 是直接泄漏，必须先消除，不能通过增加 baselines 或统计检验补救。

#### 2.3 Model-input comparability

- neural baselines 接收 43-D input，包括 target-derived statistics；Mean/LOCF/KNN/MICE wrapper 只取 masked IMU 与 mask（`experiment_baseline_imputation_comparison.py:66-155`）。因此 learned-vs-deterministic 的输入信息预算不一致。
- Transformer 声称使用 `dt-aware positional encoding`，但 `PositionalEncoding` 假定最后一维是 `dt`（`E:\\VSCode_Study\\2025.12.28LNN_Imputation\\models.py:240-258`）；实际 43-D input 最后一维是 window feature，所以 positional encoding 不使用真实 `dt`。

### 3. Missingness Protocol

- random mode 的 point mask 可视为与数值独立的 artificial MCAR topology；block/channel 是结构性 mask，不是 Bernoulli point missing。论文将二者统称 “MCAR implemented using Bernoulli” 不准确（PDF printed p.6；`Manuscript.tex:251-252`）。
- continuous channel loss 与 irregular sampling 不等价：时间戳仍存在、`dt` 未因丢包重算。要检验 irregular sampling，需删除/抖动 timestamps 或改变 inter-arrival times，并将 elapsed time 正确传入 CfC。
- 当前提供的主 CSV 仅能识别 default `missing_mode=random` 运行；表 1 continuous 部分没有可定位的对应 CSV/config/checkpoint。作者应提供完整 provenance，不能以手工表格替代。

### 4. Loss Function Audit

#### 4.1 Reconstruction loss

- 论文式 (7) 的 mask support 与 time index 均错误：按文中 `M=1 observed` 定义，它优化 observed positions，且比较 `Y(t)` 与 `Ŷ(t-1)`（`Manuscript.tex:199-203`）。
- 主代码则是缺失位置同刻 MSE（`experiment_bidirectional_lnn_residual.py:64-77`），这一实现更合理；论文必须按代码更正。
- neural model 的最终输出没有显式 `X_imp=M⊙X+(1-M)⊙Ŷ` observed-value preservation。RMSE 只在 missing positions 可避开此问题，但 trajectory evaluation 直接积分 model `pred`（同文件 `:349-366`），因此 observed points 也可能被模型改写，与 deterministic methods 的行为不同。

#### 4.2 “Physics” loss

- 论文没有定义 `v_t` 的来源、frame、supervision 或 scale；代码中的实际 loss 更不是该公式。
- trajectory-aware code 的累计量只有一次 `a·dt` 累积，维度是 velocity increment，不是 displacement；变量名 `pred_disp` 和 “trajectory-aware displacement” 不正确（`experiment_bidirectional_lnn_residual_trajectory.py:79-91`）。
- loss 在 normalized acceleration 上计算，`dt` 又索引错误，无法赋予稳定物理量纲。

### 5. Evaluation Metrics

#### 5.1 RMSE

- masked RMSE support 正确，但 `_evaluate` 先对每个 batch 计算 RMSE，再对 batches 做未加权平均（`experiment_bidirectional_lnn_residual_trajectory.py:258-281`），并非对全部 missing elements 汇总后开根号；最后一个小 batch 与完整 batch 权重相同。
- 六个 normalized channels 混合成单一 RMSE，未分别报告 gyro/accelerometer 或 physical units；由于 MAD normalization 还按每个文件变化，跨文件平均难以解释。

#### 5.2 ATE/RTE

- 实现的 `ate` 是 mean Euclidean position error（`dataset.py:505-516`），与正文式 (12) 基本一致，但函数 docstring 称 “RMSE”，命名内部不一致。常见 ATE-RMSE 应另行报告并明确是否 SE(2)/SE(3) alignment。
- quantitative code 与 Vicon 比较，而 trajectory 图正文以 complete-IMU integrated trajectory 为 reference（PDF printed p.7；`Manuscript.tex:347`）；二者回答不同问题，必须命名为 `ATE_Vicon` 与 imputation-induced trajectory deviation。
- 论文 Figure 4 的 speed colorbar 达约 120 m/s（PDF printed p.6），与手持/跑步场景明显不符，应审计时间戳单位、位置差分和速度单位后再使用任何速度/轨迹结论。

### 6. Baseline Fairness

- Methodology 先称 GRU-D，表格与实现是普通 GRU；MICE 是 batch-flattened、2-iteration pseudo-inverse regression，而非标准 multiple imputation with uncertainty（`experiment_baseline_imputation_comparison.py:122-140`）。名称会误导公平性判断。
- 缺少 linear interpolation、spline、Kalman/state-space smoother、BRITS/SAITS 等直接时序插补 baselines；但在消除 leakage 之前，增加 baselines 不会修复有效性。
- 各模型参数、input dimension 和 checkpoint provenance 不统一。部署表中的 BiLSTM/BiLNN/Hybrid latency 可追溯到 `inference_speed_benchmark_20260325_134753.csv:4-6`，但该 CSV 的 GRU/Transformer 是 13-D block checkpoints且约 0.96/0.98 ms；论文表却列 110.7/433.4 ms。当前 43-D benchmark `inference_speed_benchmark_20260326_140536.csv:2-6` 又给出 GRU 1.107 ms、Transformer 2.334 ms，不支持论文数字。吞吐单位实际上是 windows/s，而非 IMU samples/s。

### 7. Hyperparameter Selection & Test Leakage

- 论文明确用 **Test RMSE/Test ATE** 选择 λ、sequence length 和 hidden units（`Manuscript.tex:361,433-446`），违反 test set 一次性最终评价原则。
- 主结果实际默认 config 与论文选择结果冲突：代码/CSV 指向 `seq_len=30, epochs=20, lnn_hidden=128`，论文方法称 50 epochs、BiLNN 每方向 64 units，后文又称 `L=50`。optimizer 也从论文 Adam 变为代码 AdamW+OneCycleLR。
- 建议 nested protocol：train 拟合；validation 选择 architecture/λ/L/epoch；冻结；test 只评一次。任何根据 test 曲线选择的设置都应废弃并重新评估。

### 8. Results Integrity and Claim Alignment

- Table 1 最后两列标签/数据互换。以 LOCF 10% 为例，RMSE 相对增幅约为 `(0.3975-0.3406)/0.3406≈16.7%`，却出现在 “Average ATE Increase” 列；相邻约 6–7% 的数值对应 ATE。且逐行值并非 average。
- 正文称所有方法 RMSE/ATE 随 missing rate 增加而上升，但表内 MICE random ATE 从 0.0183 降至 0.0153，Transformer 与 Bi models 也非单调（PDF printed pp.7–8；`Manuscript.tex:278-280`）。
- “Hybrid consistently best in RMSE and ATE” 不受表格支持：例如 random 20%/40% 和 continuous 20%/30% 的 ATE 均有其他模型更低。
- Table 2 把 ATE 标为 `%`，而 Table 1 是无单位小数；`1.2403` 更像 `0.012403 m ×100` 的厘米数，不是百分比。

### 9. Reproducibility

- `requirements.txt` 仅给下界（如 `torch>=2.0.0`, `ncps>=0.0.9`），没有 lockfile、CUDA/cuDNN、GPU driver、ncps exact version 或 git commit；CfC behavior 可能随版本变化。
- 结果目录缺少与关键 CSV 同 timestamp 的 config JSON 和部分 checkpoints；代码默认允许 reuse latest checkpoint，且按 modification time 选择（`experiment_bidirectional_lnn_residual_trajectory.py:52-54,368-381`），容易把不同 config 的 checkpoint 混入新实验。
- 没有 deterministic algorithm flags、worker seeding manifest、raw per-record metrics、five-seed outputs、data split manifest 或 table-generation script。

---

## Statistical Reporting Completeness

### Completeness Grade

**Unacceptable（18/100）**

| Dimension | Score | Evidence |
|---|---:|---|
| Descriptive statistics completeness | 6/15 | 有部分模型/缺失率 point estimates，但无 per-record `n`、distribution、range、SD |
| Effect size reporting | 0/20 | 无 paired effect size 或相对改进 CI |
| Confidence intervals | 0/15 | 全文无 95% CI |
| Assumption/robustness testing | 3/15 | 有多 missing-rate 扫描，但无跨 recording heterogeneity、mask realization、outlier/sensitivity analysis |
| Statistical power / seed design | 0/10 | 无 a priori run design；“5 repeats”不可追溯 |
| Missing data protocol reporting | 5/10 | nominal rates 和两类 topology 有描述，但 realized rates/channel quantization 未报告 |
| Format/unit correctness | 4/10 | 基本表格结构存在，但 ATE decimal/%/cm 混用，Table 1 增幅列错位 |
| No red flags | 0/5 | 存在 target leakage、test-set tuning、表格与 CSV 不一致、不可追溯报告和 pseudo-replication |

### Required Statistical Remedy

1. 以 independent recording 为分析单位，预先固定 5–10 seeds；报告每模型每 setting 的 per-record values。
2. 报告 mean±SD、median/IQR、95% paired bootstrap CI；主比较采用 paired permutation/Wilcoxon，并报告 rank-biserial correlation 或 paired standardized effect。
3. 对多模型×多缺失率比较做 Holm/FDR correction，并预先指定 primary contrast。
4. 输出 accelerometer/gyro 分层指标、各轴 physical-unit metrics、不同 gap length 与场景分层结果。
5. 所有表格由 raw long-format CSV 自动生成，并保留 config/checkpoint/commit hash。

---

## Methodological Fallacies Detected

1. **Target leakage / data snooping**：complete-target window statistics 和 per-test-file scaling 进入输入。
2. **Test-set leakage**：用 Test RMSE/ATE 选 λ、L 和 hidden units。
3. **Pseudo-replication**：大量重叠短窗被平均，未以 recording 作为独立样本。
4. **Confirmation bias / selective reporting risk**：五次重复无证据，表格 ATE 与 CSV 不一致，且文字只强调支持性方向。
5. **Overgeneralization**：test 实际仅 user-2 recordings，却推断八场景 generalizability、real-world robustness 与 state-of-the-art。
6. **Post hoc causal attribution**：仅因 Hybrid RMSE 较低便归因于 gate 的 local/global complementarity，缺少关键消融。
7. **Construct validity error**：structured value missing 被等同于 irregular sampling；短窗真值初始化误作 long-term drift robustness。
8. **Metric/unit equivocation**：ATE 在 meter-like decimal、percentage 与 centimeter-like values 间切换。

---

## Questions for Authors

1. 请逐列解释 `dataset.py:368-397` 的 24 个 window features 在真实缺失部署时如何获得；这些特征是否全部由完整 target 计算？若是，请确认是否同意当前结果需全部重跑。
2. 表 2 所谓 physics-informed checkpoints 究竟由哪一个 loss class、哪一个 git commit 和 config 生成？请给出 `v_t` 的具体来源，并解释为何现存 trajectory-aware CSV 的 ATE 约 0.0194，而表中为 0.0124/1.24%。
3. 论文所称 five repeated experiments 的五个 seeds、五个 checkpoints、per-run metrics 和聚合脚本在哪里？为什么当前 CSV 只有单一结果且脚本没有 repeat loop？
4. 请提供 Table 1 continuous missing 的原始 CSV/config，并说明使用 `block` 还是 `channel`。若为 `channel`，如何处理 6 通道下 nominal 10/20/30/40% 的量化误差？
5. ATE 是否在每个长度 30/50 的窗口中用 Vicon position/velocity 重置？若是，如何据此支持 long-term drift suppression？为何 acceleration 未用 attitude/quaternion 转到 world frame？
6. 请提供固定的 train/val/test manifest。作者是否知晓当前 deterministic order 使 test set 仅包含 user-2 的 5 个 recordings？
7. 主表最终使用的 `seq_len`、epochs、BiLNN hidden units、optimizer 和 scheduler 分别是什么？为什么与 `Manuscript.tex:220-224,433-446` 及主脚本默认值不一致？
8. 表 1/2/3 是否由脚本自动生成？若否，请给出每个单元格到 raw result 的映射，并解释 Table 1 增幅列、Table 2 ATE 单位和 Table 3 GRU/Transformer latency 的转换公式。

---

## Minor Issues

### Method notation and terminology

- 式 (7) 应使用 `1-M` 且删除 `t-1`，分母应为 missing count 而不是 `N_obs`（PDF printed p.4；`Manuscript.tex:199-203`）。
- “Z-score normalization” 与实际 MAD normalization 不一致（`Manuscript.tex:244` vs `dataset.py:283-287`）。
- “GRU-D” 与实际普通 GRU 不一致（`Manuscript.tex:153,224` vs `models.py:215-237`）。
- “MICE” 应改为 custom lightweight chained linear regression，除非使用并验证标准 MICE implementation。

### Figures and tables

- Figure 6 是 trajectory plot，但 caption 误复制 missing-pattern caption（PDF printed p.9；`Manuscript.tex:347-352`）。正文又把 Figure 6 panels 错引为 Figure 8。
- Table 1 `\\label` 位于 `\\caption` 之前（`Manuscript.tex:281-284`），可能造成交叉引用异常。
- Table 2 的 “Physic-Informed” 应为 “Physics-Informed”，ATE 单位必须统一为 m/cm 或 path-normalized %。
- Figure 4 的 speed scale 需核查；若不是 m/s，应更正 colorbar label。

### Reporting

- 论文声称使用 Python 3.12/PyTorch 2.3.0，但 `requirements.txt` 未冻结版本；建议提供 lockfile/environment export。
- “significantly”, “state-of-the-art”, “consistently outperforms”, “mitigates noise and drift” 在无统计检验、强 baseline 和 noise/drift experiment 时应删除。

---

## Dimension Scores

| Dimension | Score (0–100) | Descriptor | Notes |
|---|---:|---|---|
| Originality (20%) | 58 | Weak | hybrid continuous/discrete branches 有一定组合价值，但 gate 与所谓 physics contribution 仍偏增量且未被有效消融 |
| Methodological Rigor (25%) | 24 | Insufficient | target leakage、loss identity mismatch、test tuning、ATE construct invalidity直接破坏主结论 |
| Evidence Sufficiency (25%) | 28 | Insufficient | 单数据集且 test 仅单 scenario；无多 seed/CI；continuous/physics 表格缺可追溯原始证据 |
| Argument Coherence (15%) | 42 | Insufficient | 论文公式、代码、CSV 和结论多处断裂；“irregular/physics/drift”证据链不闭合 |
| Writing Quality (15%) | 54 | Weak | 总体可读，但术语、单位、图注、方法配置和结果叙述存在大量实质性不一致 |
| Literature Integration (optional) | 55 | Weak | 非本 reviewer 主项；基线协议缺少关键时序插补方法 |
| Significance & Impact (optional) | 62 | Adequate | IMU gap imputation 有应用意义，但当前证据不能支持部署或导航鲁棒性影响 |
| **Weighted Average** | **39.0** | **Reject** | `58×0.20 + 24×0.25 + 28×0.25 + 42×0.15 + 54×0.15 = 39.0` |

---

## Minimum Conditions for a New Submission

1. 移除 target-derived inputs 与 test-file normalization，建立 leakage tests。
2. 冻结唯一的 physics/kinematic loss 定义和正确 `dt` indexing，并证明 coordinate-frame/units 正确。
3. 重新设计 split、missingness、validation-only tuning 和 per-record long-horizon trajectory evaluation。
4. 在无 test reuse 的条件下全量重训全部 baselines/ablations，至少 5 seeds，提供 CI/effect size。
5. 用自动化 provenance pipeline 重建 Table 1–3，并保证每个数字可追溯到 raw result、config、checkpoint 与 commit。

---

## Protocol Note

本报告按 `academic-paper-reviewer` 的 Phase 1 methodology reviewer 标准执行。技能包所述 v3.6.2 sprint contract 的可用模板/预承诺包未随本次委派提供，且上游已明确要求采用 standard single-stage report；因此使用 **protocol fallback**：完整遵循独立审稿、read-only、问题—影响—建议—severity、统计完整性与 0–100 rubric，但不输出机械 contract dimension/failure-condition sections。报告未读取 `output/review/phase1` 下任何其他审稿人文件，未修改 manuscript 或源码。
