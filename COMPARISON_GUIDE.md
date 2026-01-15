# 对比实验使用指南

## 📊 完整对比实验

该脚本会自动训练并对比所有模型（CfC、GRU、Transformer），在多种缺失模式下评估性能。

### 🚀 快速开始

#### 1. **快速测试**（20 epochs，推荐先运行）
```powershell
python run_comparison.py --quick
```
预计时间：10-20分钟（取决于硬件）

#### 2. **完整对比实验**（50 epochs，标准配置）
```powershell
python run_comparison.py
```
预计时间：30-60分钟

#### 3. **自定义对比实验**
```powershell
python run_comparison.py --epochs 100 --batch_size 32 --hidden_units 128 --seq_len 100
```

---

## 📋 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--root_dir` | `"Oxford Dataset"` | 数据集路径 |
| `--seq_len` | 50 | 序列长度 |
| `--mask_rate` | 0.3 | 训练时缺失率（30%） |
| `--missing_mode` | `random` | 训练时缺失模式 |
| `--batch_size` | 16 | 批大小 |
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 1e-3 | 学习率 |
| `--hidden_units` | 64 | 隐藏单元数 |
| `--device` | `cuda` | 设备（cuda/cpu） |
| `--output_dir` | `comparison_results` | 结果保存目录 |
| `--quick` | - | 快速测试模式（20 epochs） |

---

## 📁 输出文件说明

实验完成后，会在 `comparison_results/` 目录生成以下文件：

### 1. **模型权重**
- `cfc_best_model.pt` - CfC 模型最佳权重
- `gru_best_model.pt` - GRU 模型最佳权重
- `transformer_best_model.pt` - Transformer 模型最佳权重

### 2. **对比报告**（带时间戳）
- `summary_YYYYMMDD_HHMMSS.csv` - 模型性能总结表
- `multi_pattern_YYYYMMDD_HHMMSS.csv` - 多模式评估详细结果
- `recommendation_YYYYMMDD_HHMMSS.txt` - 最佳模型推荐
- `raw_results_YYYYMMDD_HHMMSS.pt` - 完整原始结果（可用于进一步分析）

---

## 📊 对比维度

### 训练阶段对比
- ✅ 训练时间
- ✅ 收敛速度
- ✅ 最终验证损失
- ✅ 插补误差（MSE）

### 多模式评估（训练后）
对每个模型在以下场景评估：

**缺失模式**：
- `random` - 随机缺失
- `block` - 连续块缺失
- `channel` - 通道缺失

**缺失率**：
- 10%, 20%, 30%, 40%, 50%

**评估指标**：
- MSE (all) - 全序列均方误差
- MSE (masked) - 仅缺失位置均方误差

---

## 🎯 示例输出

### 终端输出示例
```
================================================================================
SUMMARY TABLE
================================================================================
  Model  Training Time (min)  Final MSE (all)  Final MSE (masked)  Best Val Loss
    CFC                 8.45            0.1234              0.1456          0.0987
    GRU                 6.23            0.1567              0.1890          0.1234
TRANSFORMER             9.12            0.1445              0.1678          0.1098

================================================================================
RECOMMENDATION
================================================================================
✅ BEST MODEL: CFC
   MSE (masked): 0.1456
   Training time: 8.45 min
```

### CSV 文件示例（summary）
```csv
Model,Training Time (min),Final MSE (all),Final MSE (masked),Best Val Loss
CFC,8.45,0.1234,0.1456,0.0987
GRU,6.23,0.1567,0.1890,0.1234
TRANSFORMER,9.12,0.1445,0.1678,0.1098
```

---

## 🔧 常见问题

### 1. CUDA 内存不足
```powershell
python run_comparison.py --batch_size 8 --seq_len 30
```

### 2. 仅在 CPU 上运行
```powershell
python run_comparison.py --device cpu --batch_size 8
```

### 3. 快速验证流程是否正常
```powershell
python run_comparison.py --quick --epochs 5
```

### 4. 只对比部分模型
修改 `run_comparison.py` 第 37 行：
```python
models_to_test = ["cfc", "gru"]  # 只对比 CfC 和 GRU
```

---

## 📈 后续分析

### 加载结果进行分析
```python
import torch
import pandas as pd

# 加载原始结果
results = torch.load("comparison_results/raw_results_20260112_143022.pt")

# 查看 CfC 的训练历史
cfc_history = results["cfc"]["history"]
print("CfC Training Loss:", cfc_history["train_loss"])

# 加载 CSV 进行可视化
df = pd.read_csv("comparison_results/summary_20260112_143022.csv")
print(df)
```

### 绘制对比图表
```python
import matplotlib.pyplot as plt

# 对比训练曲线
for model_name in ["cfc", "gru", "transformer"]:
    if model_name in results:
        plt.plot(results[model_name]["history"]["val_loss"], 
                label=model_name.upper())
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.savefig("comparison_loss_curves.png")
```

---

## ⏱️ 预计时间

| 配置 | 单模型时间 | 总时间（3模型） |
|------|------------|----------------|
| Quick (20 epochs) | 3-6 min | 10-20 min |
| Standard (50 epochs) | 8-15 min | 30-50 min |
| Full (100 epochs) | 15-30 min | 60-90 min |

*时间基于 GPU (RTX 3080) 估算，CPU 运行时间约为 5-10 倍*

---

## ✅ 实验完成检查清单

- [ ] 所有模型训练成功
- [ ] 生成了 summary CSV 文件
- [ ] 生成了 multi-pattern CSV 文件
- [ ] 生成了 recommendation 文件
- [ ] 所有模型权重已保存
- [ ] 查看了最佳模型推荐

---

## 🎓 实验建议

1. **首次运行**：使用 `--quick` 模式验证流程
2. **正式实验**：使用默认 50 epochs 或更高
3. **论文/报告**：运行多次取平均，固定随机种子
4. **调优**：基于对比结果调整超参数

现在可以运行了！建议从快速模式开始：
```powershell
python run_comparison.py --quick
```
