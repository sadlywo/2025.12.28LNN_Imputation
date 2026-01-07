# CfC-IMU Imputation Project

基于 **Closed-form Continuous-time (CfC)** 神经网络的 IMU 数据补缺系统。

## 核心特性

### 1. CfC 优化架构
- ✅ **连续时间建模**：保留真实时间间隔（不规则采样）
- ✅ **物理先验嵌入**：分离的陀螺仪/加速度计处理头
- ✅ **不确定性估计**：自适应损失权重

### 2. 创新设计
- **MAD 归一化**：比 Z-score 更鲁棒的归一化方法
- **自适应损失函数**：
  - 加权重建损失（基于不确定性）
  - 时间一致性约束
  - 平滑度正则化
- **多模式评估**：random/block/channel 缺失模式

### 3. 简化架构
- ❌ 移除手动速度积分（CfC 自带 ODE solver）
- ❌ 移除复杂的物理损失（让网络学习物理规律）
- ✅ 专注于 CfC 的核心优势

## 项目结构

```
├── config.py          # 配置文件（数据/模型/训练）
├── dataset.py         # CfC 优化的数据集
├── models.py          # 物理先验模型 + 自适应损失
├── train.py           # 训练流程
├── main.py            # CLI 入口
└── Oxford Dataset/    # 数据集目录
    ├── handbag/
    ├── iPhone 5/
    └── pocket/
```

## 快速开始

### 基础训练
```bash
python main.py --root_dir "Oxford Dataset" --epochs 50 --hidden_units 64
```

### 自定义参数
```bash
python main.py \
    --root_dir "Oxford Dataset" \
    --seq_len 50 \
    --mask_rate 0.3 \
    --missing_mode random \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-3 \
    --hidden_units 64 \
    --w_recon 1.0 \
    --w_consistency 0.5 \
    --w_smooth 0.1
```

## 参数说明

### 数据参数
- `--root_dir`: 数据集根目录
- `--seq_len`: 序列长度（默认 50）
- `--mask_rate`: 缺失率 0-1（默认 0.3）
- `--missing_mode`: 缺失模式 random/block/channel

### 模型参数
- `--hidden_units`: CfC 隐藏单元数（默认 64）

### 训练参数
- `--epochs`: 训练轮数（默认 50）
- `--lr`: 学习率（默认 1e-3）
- `--batch_size`: 批大小（默认 16）
- `--device`: cuda/cpu

### 损失权重
- `--w_recon`: 重建损失（默认 1.0）
- `--w_consistency`: 时间一致性（默认 0.5）
- `--w_smooth`: 平滑度（默认 0.1）

## 输出文件

- `best_model.pt`: 最佳模型权重
- `training_results.pt`: 训练历史和多模式评估结果

## 核心改进

### vs 原始版本

| 方面 | 原版本 | 新版本 |
|-----|-------|-------|
| 时间处理 | 固定间隔 | **保留真实间隔** |
| 归一化 | Z-score | **MAD（更鲁棒）** |
| 物理约束 | 手动积分层 | **CfC 内置 ODE** |
| 损失函数 | 多项物理损失 | **自适应不确定性加权** |
| 架构 | 单一输出头 | **分离 gyro/acc 头** |
| 代码复杂度 | 高 | **低（简化 40%）** |

## 依赖

```bash
pip install torch numpy pandas tqdm ncps
```

## 理论基础

CfC 的核心优势：
1. **连续时间动力系统**：天然处理不规则采样
2. **闭式解**：数值稳定性好
3. **长期依赖建模**：优于 LSTM/GRU

物理先验设计：
- 陀螺仪（角速度）和加速度计（线性加速度）物理特性不同
- 分离处理头更符合物理直觉
- 不确定性估计帮助模型识别难以预测的区域

## License

MIT
