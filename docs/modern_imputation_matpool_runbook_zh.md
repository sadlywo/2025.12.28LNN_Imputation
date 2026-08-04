# BRITS、SAITS、CSDI、SSSD 与现有 Hybrid 的矩池云实验手册

本流程在完全相同的 `strict_file` 划分、训练集 scaler、窗口和预生成 mask 上，重新训练并比较
`linear、locf、bilstm、bilnn、hybrid、brits、saits、csdi、sssd`。因此现有 Hybrid 会运行，且不是复用旧结果。

## 1. 本地生成上传包

在干净的精确 Git 提交上运行：

```powershell
.\scripts\package_modern_experiment.ps1 -IncludeData
```

生成 `dist/modern-imputation-upload-<commit>.tar.gz` 和同名 `.sha256`。不加 `-IncludeData`
时不复制 Oxford 数据，适合数据已在服务器上的情况。上传包包含 Git bundle、全部配置和脚本、
SSSD 固定提交源码、许可证及 `bootstrap.sh`，不包含本地虚拟环境、results、cache 或 `.worktrees`。

## 2. 矩池云实例与上传

选择 Linux、单张 RTX 4090（显存至少 23 GiB）、足够磁盘空间的按量实例。把 tar.gz 与 sha256
上传到同一目录，然后执行：

```bash
sha256sum -c modern-imputation-upload-<commit>.tar.gz.sha256
mkdir modern-imputation && tar -xzf modern-imputation-upload-<commit>.tar.gz -C modern-imputation
cd modern-imputation
bash bootstrap.sh
```

`prepare` 会安装两个隔离环境。它需要系统可用的 `python3.10`、`tmux`、Git、驱动和
`nvidia-smi`；若镜像缺少这些基础命令，应先换用带 Python 3.10 的 PyTorch 镜像。

## 3. 预检与开始实验

```bash
bash scripts/run_modern_imputation_matpool.sh prepare
bash scripts/run_modern_imputation_matpool.sh start
```

`prepare` 会检查 GPU/显存、干净提交、固定 SSSD 提交，创建 `.venv-modern-pypots` 与
`.venv-modern-sssd`，运行 V2/modern 测试和两个真实 preflight。任何门禁失败都不会启动正式
付费训练。`start` 在 tmux 中依次执行 plan、数据导出、小网格调参、5-seed 正式训练、评估、汇总和校验。

正式调参仅使用 seed 2026 的 validation point-30%；CSDI/SSSD 调参采样 5 次，正式采样 50 次。
正式矩阵为 5 个种子 × 9 个模型，每个检查点评估 12 个常规条件和 1 个 irregular 条件。

## 4. 查看状态、日志和恢复

```bash
bash scripts/run_modern_imputation_matpool.sh status
bash scripts/run_modern_imputation_matpool.sh logs
bash scripts/run_modern_imputation_matpool.sh resume
```

关闭 SSH 不会停止 tmux。只有 `validation-report.json` 的 `status` 为 `complete` 且
`campaign_complete=true` 才算正式完成。失败时保留 task JSON、stdout/stderr、checkpoint 和已完成预测；
`resume` 只接受 `prepared.json` 记录的同一提交，并复用已有 checkpoint/预测，不会覆盖完成证据。

主要路径：

- `.modern-campaign/prepared.json`：双环境预检封印；
- `.modern-campaign/campaign.log`：总日志；
- `.modern-campaign/results/selected_hyperparameters.json`：调参锁；
- `.modern-campaign/results/reference/`：现有五模型（含 Hybrid）重跑结果；
- `.modern-campaign/results/formal/`：BRITS/SAITS/CSDI/SSSD checkpoint 与预测；
- `.modern-campaign/results/summary/`：汇总表；
- `.modern-campaign/results/validation-report.json`：最终验收。

## 5. 打包并下载结果

```bash
bash scripts/run_modern_imputation_matpool.sh package-results summary
bash scripts/run_modern_imputation_matpool.sh package-results full
sha256sum -c <download-package>.sha256
```

`summary` 包含配置、指标、图表、日志、环境清单、哈希清单和调参锁，不含大 checkpoint/采样数组；
`full` 额外包含 checkpoint 与 50-sample 预测制品。建议先下载 summary，再按论文复核需要下载 full。

下载并解压后，可在本地重新汇总：

```powershell
conda run -n pinn_imu python -m validation_v2.modern.cli summarize `
  --config configs/validation_v2/modern_stage_a.yaml --output <解压后的结果目录>
```

按量实例持续计费。确认 `status` 完成、两个归档下载并通过 SHA-256 后，应在矩池云控制台主动关机或释放实例。
