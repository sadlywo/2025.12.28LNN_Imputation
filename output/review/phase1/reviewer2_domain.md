# Peer Review Report — Peer Reviewer 2 (Domain)

## Manuscript Information

- **Title**: *Physics-Informed Neural Temporal Hybridization for Missing IMU Data Imputation: An Integrated Bidirectional LNN-LSTM Hybrid Model*
- **Manuscript ID**: Not provided
- **Review date**: 2026-07-11
- **Review round**: Phase 1 independent review
- **Materials reviewed**: `Manuscript.tex`, rendered `Manuscript.pdf`, the OxIOD files in the current repository, relevant training/evaluation/visualization code, and result CSVs. No other Phase 1 reviewer report was read.

## Reviewer Information

### Reviewer Role

Peer Reviewer 2 (Inertial Navigation & Physical Modeling / Domain)

### Reviewer Identity

Senior researcher in strapdown inertial navigation, smartphone IMU stochastic error modeling, Vicon/IMU synchronization, coordinate-frame mechanization, gravity compensation, double integration, and trajectory evaluation (ATE/RPE/RTE).

### Review Focus

This review assesses whether the claimed physics-informed contribution closes a valid physical loop from measured IMU quantities to velocity and position, and whether the Vicon synchronization and trajectory metrics support the navigation claims. It also audits the accuracy of the manuscript's inertial-navigation, liquid-network, missingness, and trajectory-evaluation terminology and literature positioning.

## Overall Assessment

### Recommendation

- [ ] Accept
- [ ] Minor Revision
- [ ] Major Revision
- [x] **Reject in the present form; encourage resubmission after complete trajectory-pipeline revalidation**

This recommendation is driven by invalidated navigation evidence, not by lack of value in the research question. The signal-imputation model may remain scientifically promising, but the paper's defining “physics-informed” and trajectory-fidelity claims cannot be assessed from the present implementation/results.

### Confidence Score

**5/5 (high confidence).** The decisive findings concern timestamp units, coordinate frames, inertial mechanization, numerical integration, and trajectory metrics, all directly within the reviewer's expertise and directly verifiable in the supplied data and code.

### Summary Assessment

The manuscript proposes a gated bidirectional CfC/LSTM imputer for six-channel smartphone IMU data and evaluates reconstruction RMSE together with an acceleration-integrated trajectory metric on OxIOD. The topic is important: missing IMU samples can contaminate downstream state estimation, and separating pointwise imputation fidelity from navigation impact is a sound research direction. However, the domain evidence chain is presently broken at several load-bearing points. Most critically, the Vicon timestamps are stored in nanoseconds while IMU timestamps are in seconds, yet the code interpolates them without conversion; this collapses every interpolated Vicon sequence to its first pose. The trajectory code also integrates device-frame `user_acc` directly as world-frame acceleration, does not use the available attitude/quaternion, resets each short sliding window using Vicon-derived initial conditions, and reports mean position distance as “ATE/RMSE” without standard trajectory alignment. In addition, the implemented “physics” objectives do not match Eq. (8), and the CSV values do not reproduce the ATE values printed in Tables 1–2. These issues invalidate claims of physical consistency, reduced integration drift, high-fidelity Vicon trajectory recovery, and navigation robustness. A defensible resubmission requires corrected synchronization, explicit coordinate-frame mechanization, a dimensionally valid loss, sequence-level trajectory evaluation, and complete regeneration of all trajectory-dependent tables and figures.

## Strengths

### S1: Important task formulation beyond pointwise imputation

The manuscript correctly recognizes that small acceleration errors can accumulate in downstream motion reconstruction (PDF pp. 1–2; LaTeX lines 132–146). Evaluating both missing-point reconstruction and a downstream motion quantity is, in principle, more meaningful than reporting only normalized RMSE.

### S2: Choice of a diverse public inertial dataset

OxIOD contains varied phone carrying modes and motion regimes, and supplies motion-capture ground truth (PDF p. 5; LaTeX line 242). This is an appropriate starting point for evaluating smartphone IMU imputation, provided that timing, frames, and pose labels are handled correctly. The manuscript cites the primary OxIOD/DeepIO papers (`chen2019deep`, `chen2020deep`).

### S3: Awareness of actual sampling intervals

The dataset code retains per-sample `dt` (`dataset.py`, lines 279–281), and the manuscript motivates CfC using explicit time dependence (PDF pp. 3–4; LaTeX lines 168–178). This is a potentially useful architectural property if the experiment truly varies observation times and passes the correct `dt` tensor into CfC/loss calculations.

### S4: Attempt to expose the navigation cost of reconstruction error

The manuscript includes quantitative and qualitative trajectory sections (PDF pp. 6–9; LaTeX lines 268–272 and 344–359), rather than leaving “physical plausibility” as an untested assertion. This is good scientific instinct, although the current trajectory pipeline must be rebuilt before those sections can serve as evidence.

## Weaknesses

### W1: Vicon/IMU timestamp mismatch collapses the ground-truth trajectory

**Problem**: OxIOD IMU timestamps are in seconds (e.g., `1495462820.28`), while the paired Vicon timestamps are in nanoseconds (e.g., `1495462816462090000`). The local `Oxford Dataset/ReadMe.txt` documents the file formats. The loader reads both as raw floating-point values and calls `np.interp(imu_time, vicon_time, ...)` without dividing Vicon time by (10^9) (`dataset.py`, lines 245–274). A direct read-only check on `handheld-1/imu1.csv` and `vi1.csv` found an IMU range of (1.49546282028\times10^9)–(1.49546320330\times10^9), a raw Vicon range of (1.49546281646\times10^{18})–(1.49546321147\times10^{18}), and only **one unique interpolated Vicon position across the first 100 IMU samples**, even though the true Vicon position spans approximately ([3.55, 5.84, 0.55]) m over the file.

**Why it matters**: `np.interp` treats every IMU query time as earlier than the first Vicon timestamp and returns the first Vicon pose. Thus the “ground truth” passed to `compute_ate` is a constant point, the estimated initial velocity becomes zero, and all reported ATE/RTE values are measured relative to a synchronization artifact. This invalidates the ATE claims in the abstract (PDF p. 1; LaTeX line 104), the main result discussion (PDF pp. 7–8; lines 276–280), Table 1, Table 2, and any Vicon-based qualitative inference.

**Suggestion**: Convert Vicon timestamps to seconds before interpolation; crop to the true overlapping interval; quantify residual time offset and association error; never extrapolate silently; interpolate positions linearly and orientations with normalized quaternion SLERP; and add unit tests showing that interpolated Vicon displacement/velocity matches the source track. Regenerate every trajectory-dependent result after this correction.

**Severity**: **Critical**

### W2: The acceleration-to-position mechanization ignores coordinate frames and orientation

**Problem**: The model outputs `rotation_rate` and `user_acc` only (`dataset.py`, lines 10–14 and 186–239). Although OxIOD also provides attitude, gravity, and Vicon quaternion, the default dataset excludes them (`dataset.py`, lines 45–47 and 79–91), and `compute_ate` directly extracts channels 3–5 and integrates them as (x/y/z) world acceleration (`dataset.py`, lines 456–503). No (R^w_b) or (R^b_w) rotation is applied; the Vicon quaternion is discarded at lines 470–473; gyroscope data are not mechanized into attitude; and the manuscript never defines body/device/world/Vicon frames or axis conventions (PDF pp. 4–6; LaTeX lines 196–218 and 241–272).

**Why it matters**: A smartphone's `user_acc` is not automatically a common Vicon-world vector merely because gravity has been separated. When the phone rotates, device-frame acceleration components change even for the same world motion. Direct component-wise double integration therefore has no physically valid relation to Vicon translation. It cannot support claims of “kinematic consistency,” “navigation performance,” or “drift resistance.”

**Suggestion**: Define every frame and quaternion convention. If `user_acc` is gravity-removed device-frame acceleration, use (a^w_t=R^w_{b,t}a^b_t); if raw specific force is used, apply (a^w_t=R^w_{b,t}f^b_t+g^w). State how attitude is obtained at inference, calibrate sensor-to-Vicon extrinsics, handle accelerometer/gyro bias, and validate signs/axes with stationary and known-motion tests. If attitude is intentionally treated as oracle metadata, label the resulting evaluation accordingly and provide a non-oracle alternative.

**Severity**: **Critical**

### W3: The published physics loss has undefined variables, wrong supervision semantics, and no matching implementation

**Problem**: Equation (7) uses (M=1) for observed values, divides by (N_{obs}), and compares (Y(t)) with \hat{Y}(t-1) (PDF p. 4; LaTeX lines 199–203). This trains observed rather than artificially missing entries and introduces an unexplained one-step shift. Equation (8) requires (v_t), although the model outputs only gyro and acceleration and no velocity state/head is defined (PDF p. 4; LaTeX lines 210–218 and 222). The result-producing trajectory loss instead uses missing-point MSE plus weak observed-point MSE and a cumulative **normalized acceleration-times-dt** difference (`experiment_bidirectional_lnn_residual_trajectory.py`, lines 66–100); it neither constructs velocity (v_t) as written nor performs the claimed acceleration–velocity residual. Worse, training passes `inputs[:, :, -1:]` as `dt` (lines 232–242), but the experiment sets `include_window_features=True` (lines 434–452), so the actual `dt` is at index 12 and the last channel is a window statistic (`dataset.py`, lines 90–91 and 400–403). An alternative `PhysicsInformedLoss` constrains predicted versus target gyro derivatives and an acceleration-magnitude threshold (`models.py`, lines 405–520), again not Eq. (8).

**Why it matters**: The paper's central contribution cannot be reproduced from the equations, and the implemented terms do not have the claimed variables or physical dimensions. With normalized acceleration and a non-time feature supplied as `dt`, the trajectory term is not a kinematic law. Consequently, improved RMSE cannot be causally attributed to the published physical constraint.

**Suggestion**: Choose one auditable formulation. For example, denormalize (a^w) to m/s², integrate a defined velocity state using true seconds, and compare interval displacement/velocity against properly synchronized Vicon labels, or impose a residual on a fully specified discrete mechanization. Apply the reconstruction loss to artificially hidden entries using (1-M), preserve observed values at inference, remove the (t-1) shift unless explicitly forecasting, and state the units and mask domain of every term. Add unit/dimension tests and an ablation that changes only the physics term.

**Severity**: **Critical**

### W4: “ATE” is non-standard, oracle-initialized over very short windows, and numerically untraceable to the tables

**Problem**: The manuscript defines ATE as the time mean of Euclidean position distance (PDF p. 6; LaTeX lines 268–272), while the code documentation calls it RMSE but computes `sqrt(sum(error**2))` followed by a time/batch mean (`dataset.py`, lines 505–516). It performs no rigid alignment. Each 30-sample window is reset to the Vicon first position and estimates initial velocity from the first two Vicon positions (`dataset.py`, lines 475–503); windows overlap with a 50% stride (`dataset.py`, lines 289–304). At approximately 100 Hz, this measures only about 0.3 s per independently oracle-initialized window, not long-term inertial drift. Standard trajectory evaluation first associates timestamps, aligns trajectories, and then reports a clearly specified statistic; the TUM benchmark distinguishes ATE from RPE and uses trajectory association/alignment ([Sturm et al., 2012](https://jsturm.de/publications/data/sturm12iros.pdf); [official evaluation tools](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools)).

The paper's numbers also fail source traceability. For the hybrid at 30% missingness, the trajectory-result CSV reports RMSE `0.152848` and ATE `0.019393` (`results/bidirectional_lnn_residual_tra/missing_rate_comparison_20260310_151546.csv`, line 12), while Table 2 combines RMSE `0.1528` with ATE `1.2403%` / `0.0124` (PDF pp. 8–9; LaTeX lines 357 and 378–392). The summary CSV repeats `test_ate=0.019393` (`summary_bidirectional_lnn_traj_20260310_151546.csv`, line 4). Baseline ATEs likewise differ; for example, the baseline CSV gives Transformer ATE `0.012033` at 10% and `0.011195` at 40% (lines 22 and 25), whereas Table 1 prints `0.0130` and `0.0132` (PDF p. 8; LaTeX lines 314–318). Table 2 labels ATE as percent although the code operates on Vicon positions in meters.

**Why it matters**: The evaluation conditions strongly compress drift and leak ground-truth derivatives into every window. The reported metric is not standard ATE-RMSE, and the published values cannot be recovered from the supplied result artifacts. The conclusions that the hybrid has “high fidelity to the ground truth,” suppresses drift, and is state of the art are therefore unsupported.

**Suggestion**: Evaluate full non-overlapping files with a single initial condition, report trajectory duration and distance, use corrected timestamp association, and define whether alignment is SE(3), yaw-only, or origin-only. Report ATE-RMSE in meters, RPE/RTE over stated time/distance intervals, endpoint drift, and drift as a percentage of path length, with per-file distributions and confidence intervals. If the aim is only imputation-induced degradation, separately report ΔATE relative to a full-IMU mechanization and never call the full-IMU integral “ground truth.” Link every table cell to a machine-generated CSV.

**Severity**: **Critical**

### W5: Domain positioning and terminology omit the closest work and overstate the contribution

**Problem**: The related-work discussion relies heavily on distant applications (medical missingness, climate reconstruction, traffic, underwater networks, NMR temperature compensation) and does not critically position the method against the closest time-series imputers or learned inertial-odometry work (PDF pp. 1–3; LaTeX lines 132–178; `document.bib`). BRITS, SAITS, and CSDI are central imputation references/baselines; the bibliography includes CSDI but the manuscript does not use it in the argument. In inertial navigation, RIDI, IONet, RoNIN, and TLIO directly address why naive double integration fails and how frames/velocity/displacement/uncertainty are handled. The paper also equates block/channel missingness with irregular sampling (PDF p. 6; LaTeX line 252), although masking values on an unchanged timestamp grid does not create irregular observation times. The code's `block` mode masks an independent contiguous interval per channel, whereas `channel` masks whole channels (`dataset.py`, lines 340–352); neither is precisely documented in the manuscript.

**Why it matters**: Without the closest literature and exact missingness definitions, novelty and theoretical motivation are overstated. CfC's ability to accept time intervals does not itself demonstrate an advantage when timestamps remain regular and only values are masked. The phrase “state-of-the-art” (LaTeX line 280) is not justified by the current baseline set.

**Suggestion**: Rebuild the related work around (i) missing-value imputation with masks/time gaps, (ii) inertial mechanization and learned inertial odometry, and (iii) continuous-time neural models. Compare to at least BRITS/SAITS and CSDI or justify a deployment-constrained exclusion. Distinguish missing values, channel outage, block missingness, packet loss, asynchronous channels, and irregular timestamps. Test genuinely perturbed/irregular timestamps if irregular-time robustness is claimed.

**Severity**: **Major**

## Detailed Domain Comments

### Literature Review

- **Coverage**: The manuscript cites the CfC/LTC originals and OxIOD papers, which is appropriate. However, it omits the most relevant bidirectional imputation and learned inertial-navigation lineage. This makes the gap statement (“existing methods treat IMU as generic time series”) too broad.
- **Integration quality**: Much of the review is enumerative. Citations from unrelated domains are used to support IMU-specific failure mechanisms without explaining transferability. For example, an underwater network protocol and NMR temperature-drift compensation do not establish actual missingness mechanisms in smartphone inertial streams (`document.bib`, entries `jha2024novel` and `wang2025novel`; cited at LaTeX line 132).
- **Research gap**: A defensible gap would be narrower: missing-point IMU reconstruction under known timestamps, with explicit evaluation of *incremental degradation* in a correctly mechanized downstream estimator. The present paper instead claims general physical/navigational validity without the required state-estimation pipeline.

### Theoretical Framework and Physical Closure

- **Appropriateness**: A physics-regularized objective is reasonable, but acceleration and velocity must be expressed in the same frame and units, with velocity either estimated or labeled. Equation (8) currently treats velocity as available without defining its source.
- **Application depth**: The physical model is superficial. Gyroscope, attitude, gravity, bias, extrinsics, and frame transformations are absent from the loss/mechanization. These are not optional details in inertial navigation; they determine whether the equation is physically meaningful.
- **Discrete integration**: The code labels the position step “trapezoidal” but sets `vel_avg = vel` after updating velocity (`dataset.py`, lines 494–503). Correct trapezoidal position integration needs both (v_{t-1}) and (v_t), or an equivalent second-order update. This introduces another avoidable discretization inconsistency.
- **Alternative framework**: If full strapdown mechanization is outside scope, position the downstream test as a controlled *signal-level proxy* and evaluate changes relative to the full-IMU proxy under identical initialization. Do not present it as an absolute inertial-navigation trajectory.

### Academic Argument Quality

- **Factual accuracy**: “Continuous channel loss aligns well with irregular time sampling” (PDF p. 6; LaTeX line 252) is incorrect. Missing values at regular timestamps and irregular timestamps are distinct observation processes.
- **Logic**: Lower normalized RMSE plus a broken short-window position metric does not demonstrate enforcement of a physical law. Gate visualizations likewise show learned weights, not branch causality or physical interpretability.
- **Terminology**: Use “physics-informed,” not “physical informed.” Replace “zero-bias drift” (PDF p. 12; LaTeX line 451) with the intended stochastic error term (constant bias, bias instability, bias drift, or random walk). State ATE units; do not label meters as percent.
- **Internal contradiction**: The evaluation section says (p_t^{gt}) is Vicon ground truth (PDF p. 6; LaTeX lines 268–272), while the qualitative section says the integrated full-IMU trajectory is the reference because using “true ground-truth trajectory” would introduce drift (PDF p. 7; LaTeX line 347). Ground truth itself does not acquire inertial integration drift. These are two different evaluation questions and must be separated.

### Contribution to the Field

- **Incremental contribution**: The gated CfC/BiLSTM combination may be an incremental imputation architecture contribution. The current evidence does not establish a new physics-informed inertial-navigation method.
- **Positioning**: The closest learned inertial methods often regress velocity/displacement or fuse learned relative motion with a state estimator specifically to avoid naive double integration. The manuscript should explain why its approach is different and what assumptions it makes.
- **Overclaiming**: “State-of-the-art,” “ensures kinematic consistency,” “high fidelity to ground truth,” “reducing drift,” and “real-world deployment” should be removed until validated under corrected synchronization, mechanization, modern baselines, and a deployment-relevant benchmark.

### Results, Figures, and PDF Presentation

- The 13 numbered manuscript pages plus the separate highlights page render without clipping or broken equations. Formulae and tables are visually legible, though Table 1 is dense.
- Figure 6 is a trajectory plot but its caption repeats the missing-pattern caption (PDF p. 9; LaTeX lines 348–352). This prevents reliable interpretation.
- Figure 6's plotted coordinate spans are only centimeters and are consistent with independently reset 30-sample windows, not long navigation trajectories.
- Table 2 mixes decimal prose ATE with percentage-labeled table ATE and lacks units/provenance (PDF pp. 8–9).
- Figure 7's λ sweep (`0.05`–`0.30`) is not traceable to the supplied physics-weight scripts, which sweep separate integration/energy weights (`experiment_loss_functions_3_physics_weights_ablation.py`, lines 84–112) or much smaller stabilized weights (`experiment_physics_loss_weights_stable.py`, lines 86–117). Provide the exact generating script/CSV.

## Missing Key References

The following are not requests to inflate the bibliography; each closes a specific gap in the argument or evaluation protocol.

1. **Cao, W. et al. (2018), “BRITS: Bidirectional Recurrent Imputation for Time Series,” NeurIPS.** Closest foundational bidirectional recurrent imputation method; directly relevant to masks, time gaps, and bidirectional information. [Primary source](https://proceedings.neurips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html).
2. **Du, W., Côté, D., and Liu, Y. (2023), “SAITS: Self-Attention-based Imputation for Time Series,” Expert Systems with Applications.** Strong, efficient deterministic multivariate imputation baseline with missingness-aware weighted attention. [Preprint/primary manuscript](https://arxiv.org/abs/2202.08516).
3. **Tashiro, Y. et al. (2021), “CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation,” NeurIPS.** Already present in `document.bib` but absent from the paper's substantive positioning; relevant modern generative baseline. [Primary source](https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html).
4. **Yan, H., Shan, Q., and Furukawa, Y. (2018), “RIDI: Robust IMU Double Integration,” ECCV.** Directly addresses acceleration bias correction and double-integration failure in natural smartphone motion. [Primary source](https://openaccess.thecvf.com/content_ECCV_2018/html/Hang_Yan_RIDI_Robust_IMU_ECCV_2018_paper.html).
5. **Chen, C. et al. (2018), “IONet: Learning to Cure the Curse of Drift in Inertial Odometry,” AAAI.** Explains why direct low-cost IMU double integration becomes unusable and motivates window-level learned motion quantities. [Primary source](https://ojs.aaai.org/index.php/AAAI/article/view/12102).
6. **Herath, S., Yan, H., and Furukawa, Y. (2020), “RoNIN: Robust Neural Inertial Navigation in the Wild,” ICRA.** Provides relevant ATE/RTE conventions and robust frame handling for neural inertial navigation. [Project and citation](https://ronin.cs.sfu.ca/index.html).
7. **Liu, W. et al. (2020), “TLIO: Tight Learned Inertial Odometry,” IEEE Robotics and Automation Letters.** Important for learned displacement/uncertainty and state-estimator fusion rather than uncorrected double integration. [Primary preprint](https://arxiv.org/abs/2007.01867).
8. **Sturm, J. et al. (2012), “A Benchmark for the Evaluation of RGB-D SLAM Systems,” IROS.** Canonical distinction and implementation principles for ATE/RPE, timestamp association, and alignment. [Primary paper](https://jsturm.de/publications/data/sturm12iros.pdf).
9. **Hasani, R. et al. (2022), “Closed-form continuous-time neural networks,” Nature Machine Intelligence.** Already cited, but the manuscript should accurately state that CfC can be supplied true time intervals; this does not make value masking equivalent to irregular sampling. [Primary source](https://www.nature.com/articles/s42256-022-00556-7).

## Questions for Authors

1. What exact unit conversion and synchronization procedure was intended for Vicon timestamps, and can the authors provide a before/after plot showing associated IMU/Vicon time ranges and nonconstant interpolated Vicon trajectories for every file?
2. In which coordinate frame are OxIOD `user_acc` channels expressed in this work, and what rotation/extrinsic calibration maps them to the Vicon world frame before integration? If none was used, on what physical basis is Eq. (8) or ATE compared to Vicon position?
3. Where does (v_t) in Eq. (8) come from during training, and which supplied implementation exactly generates Table 2 and Figure 7? Please provide a one-to-one equation–code–CSV trace, including units, mask domain, `dt` index, and random seeds.
4. Why are Vicon position and the first two Vicon frames used to reset every overlapping 30-sample test window? Can the authors report corrected full-file trajectories with a single initialization and standard ATE/RPE/RTE, alongside an explicitly named imputation-induced Δtrajectory metric?

## Minor Issues

### Language and Terminology

- PDF p. 1 / LaTeX line 104: insert a space after “reconstruction.” and use “motion patterns.”
- PDF pp. 1 and 4 / LaTeX lines 116, 125, 196: use **physics-informed**, not “physical informed,” “Physical Informed,” or “Physic-Informed.”
- PDF p. 5 / LaTeX line 242: “approximately 2000,000” should be “approximately 2,000,000,” and the exact retained sample/file count should be generated from the loader.
- PDF p. 12 / LaTeX line 451: “zero-bias drift” is not standard; identify constant bias, bias instability, rate random walk, or another Allan-variance term.
- Use one model name consistently: Hybrid BiLLSTM, BiLLSTM, and Hybrid BiLNN-BiLSTM currently alternate.

### Data/Method Description

- PDF p. 5 / LaTeX line 244 says Z-score normalization, while the code uses per-file median/MAD scaling (`dataset.py`, lines 283–287).
- The manuscript states 50 epochs (LaTeX line 222), while the trajectory summary CSV records best epoch 20 and the trajectory script defaults to 20 epochs (`experiment_bidirectional_lnn_residual_trajectory.py`, lines 430–472). Report the actual executed configuration.
- Define whether continuous missingness is `block` or `channel` mode and whether the percentage means missing samples, duration, or channels.

### Figures and Tables

- PDF p. 9 / LaTeX lines 348–352: replace the duplicated missing-pattern caption under the trajectory plot.
- PDF p. 8: Table 1 should include units and uncertainty/replicate statistics, and should not report unexplained “average increase” columns without a stated formula.
- PDF p. 9: Table 2's ATE is labeled `%` although the code's positions are meters; correct the unit after rebuilding the metric.
- PDF p. 7 / LaTeX line 347 refers to Fig. 8 while discussing Fig. 6 trajectory panels; correct cross-references after clarifying which data each figure uses.

## Dimension Scores

Scores follow the `academic-paper-reviewer` 0–100 rubric and are calibrated to a high-level SCI measurement/sensors venue. They are ordinal quality judgments, not acceptance probabilities.

| Dimension | Score | Descriptor | Domain rationale |
|---|---:|---|---|
| Originality (20%) | 62 | Adequate | Gated CfC/BiLSTM for IMU imputation is a plausible incremental combination, but novelty relative to BRITS/SAITS/CSDI and learned inertial methods is not established. |
| Methodological Rigor (25%) | 30 | Insufficient | Timestamp-unit failure, absent frame transformation, incorrect `dt` indexing, oracle window initialization, and equation/code mismatch invalidate the physical/navigation experiment. |
| Evidence Sufficiency (25%) | 34 | Insufficient | Main ATE claims are invalid and published values do not trace to supplied CSVs; no corrected per-file trajectory statistics or uncertainty are given. |
| Argument Coherence (15%) | 43 | Insufficient | The central chain “physics loss → kinematic consistency → reduced drift → navigation fidelity” does not follow from the implemented variables or metrics. |
| Writing Quality (15%) | 58 | Weak | Generally readable structure, but terminology, units, figure captions, internal contradictions, and repeated grammatical problems remain below journal standard. |
| **Weighted Average** | **43.6** | **Reject** | Weighted rubric score: 43.55, rounded to 43.6. |
| Literature Integration (R2 focus) | 48 | Significant gaps | Primary CfC/OxIOD sources are present, but the closest imputation, inertial mechanization, learned odometry, and trajectory-evaluation literature is missing or not integrated. |
| Significance & Impact (optional) | 58 | Limited/conditional | The application matters, but impact depends on rebuilding and independently validating the entire physical/trajectory evidence chain. |

## Required Domain Revalidation Before Resubmission

1. Fix time units and Vicon/IMU association; add synchronization tests and nonconstant-ground-truth assertions.
2. Specify and implement body/device-to-world mechanization, gravity convention, attitude source, extrinsics, bias handling, and correct discrete integration.
3. Replace Eqs. (7)–(9) with the exact implemented, dimensionally consistent objective; ensure the true `dt` channel is used.
4. Evaluate complete held-out files with one initialization; report standard ATE/RPE/RTE plus a separately named imputation-induced degradation metric.
5. Regenerate Tables 1–2 and Figures 6–7 from versioned scripts/CSVs; report per-file variation and repeat statistics.
6. Add modern imputation and learned-inertial baselines or narrow the claims; remove “state-of-the-art,” “ensures,” and deployment claims until evidence supports them.

## Protocol Note

The sprint-contract template required by the current `academic-paper-reviewer` release was not supplied to this reviewer, and the delegation explicitly identified that template as missing. Accordingly, this is a standard single-stage, paper-visible Phase 1 Domain Review Card produced under the documented fallback. It remains independent, read-only, and confined to the Peer Reviewer 2 deliverable.
