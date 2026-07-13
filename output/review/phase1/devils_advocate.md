# Devil’s Advocate Stress-Test Report

本文把信号重建、轨迹影响和模型开销放在同一条证据链中评估，并提供了可追踪的源码与结果文件；这种任务导向值得保留。双分支残差建模也具有工程探索价值，尤其适合进一步做严格的机制消融。

## Strongest Counter-Argument

对本文核心结论最强的反论证是：现有结果更符合“带有目标信息泄漏的双向线性插值残差集成，在一个失真的短窗轨迹指标上取得较低误差”，而不是“物理约束与 BiLNN–BiLSTM 互补机制提高了真实缺失 IMU 的物理一致性”。首先，数据集把完整目标窗口计算出的 24 维统计量直接送入模型，测试输入因此包含被遮挡真值的摘要；超参数又由 Test RMSE/ATE 选定。其次，论文的 physics loss 与实际代码不一致：实现没有速度状态或论文所写的加速度—速度微分约束，且训练时把输入最后一列（实际为由完整目标计算的窗口方差）当作 `dt`。更关键的是，仓库 CSV 直接反驳正文：所谓 physics-aware 训练使三种模型的 ATE 全部略微变差；随机缺失下 Transformer 的 ATE 在四个缺失率上均低于 Hybrid，而论文表格却把 Hybrid 写成最优。轨迹评估还把设备坐标加速度直接积分到 Vicon 世界坐标，每个仅 30 帧的重叠窗口都用 Vicon 真值初始化位置和速度，并积分模型在观测点上的改写值；这更容易奖励平滑，而非正确恢复。门控数据则显示 LNN 平均权重只有约 5%–9%，没有论文声称的分支“switching”。因此，在排除泄漏、重建正确轨迹基准、核对原始表格并加入线性插值、等参数集成与固定门控对照之前，physics-informed、hybrid mechanism、SOTA、generalization 和 deployment 结论均不能由现有证据推出。

## Issue List

### CRITICAL

| # | Challenge dimension | 具体问题 | 论文/代码/CSV location |
|---|---|---|---|
| C1 | Data–conclusion mismatch / SOTA | 表 1 宣称 Hybrid 在随机缺失下取得最低 ATE（0.0114–0.0125），但可追踪 CSV 中 Hybrid 为 0.019441、0.019426、0.019393、0.019352；Transformer 为 0.012033、0.011770、0.011451、0.011195，在四个缺失率上均更低。MICE 也在四点均低于 Hybrid。现有数据直接反驳“Hybrid consistently best / state-of-the-art trajectory reconstruction”。必须从原始 checkpoint 重算统一表格，并说明正文数字的唯一来源。 | `D:/OneDrive/临时/2026.7.7修改BiLSTM补缺模型/els-cas-templates/Manuscript.tex` lines 276–280、281–343；`E:/VSCode_Study/2025.12.28LNN_Imputation/results/bidirectional_lnn_residual_tra/missing_rate_comparison_20260310_151546.csv`；`E:/VSCode_Study/2025.12.28LNN_Imputation/results/baseline_Imputation_Method/missing_rate_comparison_20260311_213735.csv` |
| C2 | Foundation collapse / physics-informed claim | 论文式 (8) 定义为速度差分与加速度的一致性，但实际实现只比较 `cumsum(acc*dt)` 的预测与目标，不存在速度状态，也不是位移双积分。更严重的是训练/验证取 `inputs[:, :, -1:]` 作为 `dt`；在当前 43 维输入布局中，真实 `dt` 位于索引 12，最后一列是用完整目标窗口计算的统计特征。所谓 physics loss 因而在量纲和对象上都不等于论文定义。 | `D:/OneDrive/临时/2026.7.7修改BiLSTM补缺模型/els-cas-templates/Manuscript.tex` lines 196–218；`E:/VSCode_Study/2025.12.28LNN_Imputation/experiment_bidirectional_lnn_residual_trajectory.py` lines 66–100、232–242、258–269；`E:/VSCode_Study/2025.12.28LNN_Imputation/dataset.py` lines 88–91、366–403 |
| C3 | Data–conclusion mismatch / physics benefit | 论文表 2 称加入物理损失后三种模型 RMSE/ATE 均改善，且 Hybrid 增益最大；仓库中 reconstruction-only 与 trajectory-aware 两份 summary 对照却显示 ATE 三者全部变差：BiLNN 0.019268→0.019375、BiLSTM 0.019351→0.019389、Hybrid 0.019327→0.019393；RMSE 还在 BiLNN 和 Hybrid 上变差。该证据不仅“不显著”，而是方向与结论相反。 | `D:/OneDrive/临时/2026.7.7修改BiLSTM补缺模型/els-cas-templates/Manuscript.tex` lines 354–359、368–396；`E:/VSCode_Study/2025.12.28LNN_Imputation/results/bidirectional_lnn_residual/summary_bidirectional_lnn_20260309_100957.csv`；`E:/VSCode_Study/2025.12.28LNN_Imputation/results/bidirectional_lnn_residual_tra/summary_bidirectional_lnn_traj_20260310_151546.csv` |
| C4 | Leakage / evidence-chain contamination | 完整、未遮挡的 `target_imu` 被用于计算 24 维窗口均值、能量、差分能量和方差，再拼接进模型输入；整份测试文件的中位数/MAD也在遮挡前由完整数据计算。与此同时，正文明确用 Test RMSE/ATE 选择 λ、序列长度和隐藏维度，之后仍在同一 test 上作最终性能主张。这是输入侧 target leakage 与选择侧 test leakage 的叠加，足以使泛化和排序失去独立性。 | `E:/VSCode_Study/2025.12.28LNN_Imputation/dataset.py` lines 283–287、318–323、366–403；`E:/VSCode_Study/2025.12.28LNN_Imputation/experiment_bidirectional_lnn_residual.py` lines 566–585、604–638；`D:/OneDrive/临时/2026.7.7修改BiLSTM补缺模型/els-cas-templates/Manuscript.tex` lines 361、433–446 |
| C5 | Logic-chain break / ATE and hybrid attribution | ATE 代码忽略姿态/四元数，不把设备坐标加速度旋转到 Vicon 世界坐标；每个 30 帧、50% 重叠窗口都用 Vicon 前两帧初始化位置和速度；随后积分模型对全部时刻的输出，而非“观测值保持原样、仅填缺失点”的完成序列。正文定量定义以 Vicon 为真值，定性图却改用完整 IMU 积分为 reference。与此同时，门控 CSV 显示 LNN 平均权重仅 0.0547–0.0894、BiLSTM 权重 0.9106–0.9453，并不存在跨越 0.5 的分支切换。更简约的解释是平滑、线性插值残差、额外参数和近似 BiLSTM 主导的集成，而不是被验证的物理互补机制。 | `E:/VSCode_Study/2025.12.28LNN_Imputation/dataset.py` lines 417–510；`E:/VSCode_Study/2025.12.28LNN_Imputation/experiment_bidirectional_lnn_residual.py` lines 146–185、241–275、349–377；`D:/OneDrive/临时/2026.7.7修改BiLSTM补缺模型/els-cas-templates/Manuscript.tex` lines 268–272、344–353、397–410；`E:/VSCode_Study/2025.12.28LNN_Imputation/results/Downstream_Tra_Plot/gate_bias_visualization/hybrid_gate_segment_summary_20260325_145414.csv` |

### MAJOR

| # | Challenge dimension | 具体问题 | 论文/代码/CSV location |
|---|---|---|---|
| M1 | Overgeneralization / irregular sampling | “continuous channel loss”只遮蔽数值，不删除时间戳或改变采样间隔，因此不是 irregular sampling。CfC 调用也未传 `timespans`，真实 `dt`仅作为普通输入特征；库在 `timespans=None` 时使用 `ts=1.0`。此外，6 通道下 `max(1,int(6r))` 使 10%、20%、30% 均只遮一个通道，所称四档 missing-rate 并非四档实际缺失率。 | `D:/OneDrive/临时/2026.7.7修改BiLSTM补缺模型/els-cas-templates/Manuscript.tex` lines 138、155、178、251–252；`E:/VSCode_Study/2025.12.28LNN_Imputation/dataset.py` lines 279–281、340–352；`E:/VSCode_Study/2025.12.28LNN_Imputation/experiment_bidirectional_lnn_residual.py` lines 208–216；`D:/Anaconda3/Lib/site-packages/ncps/torch/cfc.py` lines 112、162–173 |
| M2 | Overgeneralization / split semantics | 文件没有随机化或按场景分层，而是按固定场景顺序取前 80%、中间 10%、后 10%。按现有 45 个文件，测试集实际仅为 `user-2/imu2.csv` 至 `imu6.csv`，无法支持“跨八场景 generalization”或对 Running、Handheld 等场景的独立测试结论。 | `E:/VSCode_Study/2025.12.28LNN_Imputation/dataset.py` lines 103–141；`E:/VSCode_Study/2025.12.28LNN_Imputation/Oxford Dataset/user-2/imu2.csv` 至 `imu6.csv`；`D:/OneDrive/临时/2026.7.7修改BiLSTM补缺模型/els-cas-templates/Manuscript.tex` lines 242–244、275–280 |
| M3 | Stronger counter-narrative / missingness | MCAR 被同时用于描述逐点随机缺失和整通道失效，但后者的通道选择并非 MCAR 逐项伯努利。现实故障更可能依赖运动、冲击、无线质量或设备状态（MNAR/MAR）。只在合成遮挡上训练并在相同生成器上测试，不能推出“real-world sensor malfunction robustness”。 | `D:/OneDrive/临时/2026.7.7修改BiLSTM补缺模型/els-cas-templates/Manuscript.tex` lines 251–257、278–280；`E:/VSCode_Study/2025.12.28LNN_Imputation/dataset.py` lines 332–352 |
| M4 | Deployment claim | “deployment simulation”只是在 RTX 4090D 上用 batch=32、长度 30 的随机张量测平均 forward；Hybrid 约 1.48 s/batch 且是双向非因果模型。它没有端到端预处理、内存峰值、能耗、P95/P99、移动/嵌入式硬件或在线等待时间。表 3 的 GRU/Transformer 延迟也无法由当前三份 benchmark CSV 一致追溯，因此最多是桌面 GPU microbenchmark。 | `D:/OneDrive/临时/2026.7.7修改BiLSTM补缺模型/els-cas-templates/Manuscript.tex` lines 411–432；`E:/VSCode_Study/2025.12.28LNN_Imputation/benchmark_downstream_models_inference_speed.py` lines 218–287；`E:/VSCode_Study/2025.12.28LNN_Imputation/results/Downstream_Tra_Plot/inference_speed_benchmark_20260325_134753.csv`；`E:/VSCode_Study/2025.12.28LNN_Imputation/results/Downstream_Tra_Plot/inference_speed_benchmark_20260326_140536.csv` |

### MINOR

| # | Challenge dimension | 具体问题 | 论文/代码/CSV location |
|---|---|---|---|
| m1 | Internal consistency | 式 (7) 用观测掩码 `M=1` 且预测写成 `Ŷ(t-1)`，正文称其拟合观测点；实际核心训练损失使用 `1-mask`、预测同一时刻 `t`。这不是符号小误差，而会让读者无法知道训练目标，但在修复核心实验后可直接校正。 | `D:/OneDrive/临时/2026.7.7修改BiLSTM补缺模型/els-cas-templates/Manuscript.tex` lines 199–203；`E:/VSCode_Study/2025.12.28LNN_Imputation/experiment_bidirectional_lnn_residual.py` lines 64–77 |
| m2 | Evidence labeling | Fig. 6 的 caption 仍是 missing-pattern 图注，与实际轨迹图不符；Fig. 8 caption 把 heatmap 称为 BiLSTM 权重，而生成代码把 gate 明确定义为 LNN branch weight。标签反转会进一步误导门控解释。 | `D:/OneDrive/临时/2026.7.7修改BiLSTM补缺模型/els-cas-templates/Manuscript.tex` lines 348–352、405–409；`E:/VSCode_Study/2025.12.28LNN_Imputation/visualize_hybrid_gate_bias_segments.py` lines 148–170、225–238 |
| m3 | Claim precision | 结论称“maintain the original data distribution”“mitigates noise and drift”，但主实验既无分布检验，也无噪声/漂移受控对照；这些句子应降格为未检验假设。 | `D:/OneDrive/临时/2026.7.7修改BiLSTM补缺模型/els-cas-templates/Manuscript.tex` lines 447–453 |

## Ignored Alternative Explanations/Paths

1. **目标摘要泄漏**：完整窗口的均值、能量、差分能量和方差已约束缺失区间分布，神经模型可能在“有答案摘要”的条件下恢复，而非从观测序列泛化。
2. **线性插值残差主导**：每个通道先使用左右观测值线性插值，再由网络预测残差；缺少“linear interpolation only”后，模型相对传统基线的增益无法归因于网络。
3. **容量与集成效应**：Hybrid 参数量约 591k，高于 BiLNN 416k 和 BiLSTM 173k；无等参数单分支、固定 0.5 门控、静态最优门控或 prediction averaging 对照。
4. **平滑降低积分误差**：更高缺失率下 MICE/Transformer ATE 反而下降，且 `downstream_baselines_metrics_20260323_181334.csv` 中多种补缺轨迹优于 full-IMU 轨迹；低 ATE 可由抑制高频加速度造成，而不代表恢复真实信号或物理规律。
5. **短窗真值初始化效应**：每个窗口重置到 Vicon 位置和速度，ATE 主要反映约 30 帧局部偏差，不能支持“long-term drift suppression”。
6. **离线未来信息效应**：双向网络和线性插值都使用缺口后的观测，适用于离线修复，但不能被直接叙述为实时部署优势。

## Missing Stakeholder Perspectives

- 需要因果、低延迟恢复的嵌入式/实时导航工程师。
- 需要保留观测值、坐标系和不确定性可审计性的安全关键惯导用户。
- 面对真实通信故障、设备冲击和整段断链的传感器系统维护者。
- 需要复现实验、核对 checkpoint—CSV—表格血缘的数据与模型审计者。

## Unexamined Premise

全文默认“更平滑、积分后更接近某个短窗参考的加速度”就等同于“更物理、更接近真实缺失 IMU”。这一前提并不成立：IMU 真值包含真实高频运动、噪声、偏置和坐标变换；平滑可以降低双积分漂移，却同时抹去真实动力学。若不把信号恢复、传感器误差建模和导航状态估计分开，physics-informed 的命题会被指标自身循环证明。

## Observations (Non-Defects)

- 文件级而非窗口级划分的方向是正确的；问题在于固定顺序导致测试场景单一，而不是“按文件划分”本身。
- 使用缺口两侧信息对离线 imputation 是允许的，但必须明确任务是 offline/smoothing，而非 online filtering。
- RMSE 与下游指标并列报告是有价值的；需要的是正确坐标处理、仅替换缺失点、连续轨迹评估及可追踪原始结果。
- 当前代码和 CSV 足以定位主要断点，也为后续建立可复现实验血缘提供了良好起点。

## Protocol Note

本次调用未提供可执行的 v3.6.2 sprint contract JSON 及 paper-blind 预承诺输出；依任务指令采用标准单阶段 Devil’s Advocate Phase 1 报告作为 fallback。报告未读取 Phase 1 其他审稿人输出，未给出期刊 fit、评分或编辑决定，且未修改论文与源码。
