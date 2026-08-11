# IMU Missing-Data Imputation (`validation_v2`)

本仓库现在以 `validation_v2` 和正式物理损失重构为唯一主开发路径。旧版根目录训练、demo、
消融和可视化脚本已经整体归档到 `legacy/pre_validation_v2/`，不会再与当前入口混用。

## 当前方案

- 任务：对 IMU 六通道时序缺失值进行插补，并评估信号误差、姿态/速度/位置物理误差和轨迹误差。
- 缺失协议：保留 point、block、channel 及原有缺失比例；另有 `interval_jitter` 不规则时间实验。
- 主模型：当前矩阵中的 `hybrid`（BiLNN/BiLSTM hybrid）。
- 目标函数：缺失位置重建损失 + 基于 SO(3) 旋转、速度、位置的物理损失。
- 安全约束：坐标系、单位和 IMU/Vicon 外参未验证前，只允许 `lambda_physics=0`；非零物理权重会自动拒绝运行。
- 数据集：Oxford Inertial Odometry Dataset、EuRoC MAV Vicon Room 1/2 和 IDOL Building 1/2/3；三者通过统一 adapter contract 接入。

完整设计和改动范围见 [PHYSICS_LOSS_REFACTOR_REPORT.md](PHYSICS_LOSS_REFACTOR_REPORT.md)，
新数据集字段约定见 [docs/physics_loss_refactor_dataset_contract.md](docs/physics_loss_refactor_dataset_contract.md)。

## 环境

本地建议使用独立 Python 3.10 环境：

```powershell
conda create -n lnn-imputation python=3.10 -y
conda activate lnn-imputation
python -m pip install -r requirements.txt
```

`requirements.txt` 只指向当前 `validation_v2` 的锁定依赖。PyPOTS/SSSD 等可选现代 baseline
仍分别使用 `requirements-modern-pypots.txt` 和 `requirements-modern-sssd.txt`。

RTX 5090 服务器使用 PyTorch 2.11.0 的 CUDA 12.8 构建。正式 MatPool 入口默认两个
GPU worker，每个 shard 进程通过 `CUDA_VISIBLE_DEVICES` 固定到一张卡，不使用 DDP。

## 当前运行入口

所有命令均从仓库根目录执行。

1. 先检查实验矩阵，不训练：

```powershell
python run.py matrix --config physics_refactor_smoke.yaml --dry-run
```

2. 执行单组合 smoke test：

```powershell
python run.py smoke --config physics_refactor_smoke.yaml --device cpu
```

3. 执行当前配置的完整矩阵：

```powershell
python run.py matrix --config physics_refactor_smoke.yaml --device cuda
```

当前 YAML 是结构和短流程验证配置（1 epoch、受限窗口数）。正式训练前应复制该配置，增加
`epochs`、训练窗口与种子数量，并保持缺失协议及物理字段不变。

4. 在启用非零物理权重前诊断 IMU/Vicon 机械化约定：

```powershell
python scripts/diagnose_imu_vicon_mechanization.py
```

诊断输出位于 `results/physics_loss_refactor/v1/diagnostics/`。人工确认坐标系、单位和固定外参后，
方可把配置中的 `frame_validation_status` 改为 `validated`。

5. 查看或执行物理权重消融：

```powershell
python scripts/run_physics_lambda_ablation.py --dry-run
python scripts/run_physics_lambda_ablation.py --device cuda
```

初始化并核验 EuRoC/IDOL 本地数据：

```powershell
python scripts/initialize_external_datasets.py
```

该命令会验证 IDOL 官方 MD5、逐条读取全部 130 个 Feather 文件，并从 EuRoC 的 6 个
序列包中只初始化 IMU、ground truth 和传感器 YAML（不重复解压相机图像）。结果清单写入
`datasets/manifests/external_datasets.json`。

初始化后可分别检查两个 adapter 的实验矩阵：

```powershell
python run.py matrix --config euroc_adapter_smoke.yaml --dry-run
python run.py matrix --config idol_adapter_smoke.yaml --dry-run
```

## 目录边界

```text
configs/validation_v2/             当前实验配置
datasets/                          EuRoC/IDOL 等新增数据集（原始数据不进 Git）
docs/                              当前运行手册与数据契约
legacy/pre_validation_v2/          旧版脚本、旧公共模块和旧依赖
results/legacy_archive/            旧结果范围索引
results/physics_loss_refactor/      新方案唯一结果目录
scripts/                            当前诊断、消融和服务器运行工具
tests/validation_v2/                当前测试
validation_v2/                      当前数据、模型、损失、训练和评估代码
run.py                              当前统一入口
```

## 数据目录

下载后的 EuRoC MAV 与 IDOL 原始文件放在：

```text
datasets/raw/euroc_mav/archives/
datasets/raw/euroc_mav/extracted/
datasets/raw/idol/archives/
datasets/raw/idol/extracted/
```

原始文件、解压数据、缓存和中间处理结果均被 Git 忽略；仅代码、数据契约和校验清单应提交。

## 旧版复现

旧脚本没有删除。需要复现旧实验时，请阅读
[legacy/pre_validation_v2/README.md](legacy/pre_validation_v2/README.md)。新功能和新实验不要再修改旧目录。
