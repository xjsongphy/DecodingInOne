# Ising 实验目录

## 概述

包含 Ising 风格的神经网络解码器训练和推理实验。使用重构后的模块化架构，将数据处理、模型、训练、解码职责清晰分离。

## 1) 训练模型

**脚本：** `experiments/ising/ising_train_model.py`

**用途：** 提取 Ising 风格的**训练实验主链路**（仅训练，不做完整解码流程）

**流程：**
1. 配置加载（YAML + CLI 覆盖）
2. Stim 生成表面码 memory 电路
3. 采样训练/验证集（detectors → observables）
4. 数据预处理（transforms 转换为网络输入格式）
5. PyTorch 训练（Conv3D 网络）
6. 输出 checkpoint 和训练报告

### 配置参数说明

**配置文件：** `experiments/ising/ising_train.yaml`

#### 数据生成配置（`IsingDataConfig`）

| 参数 | 默认值 | 说明 | 推荐范围 | Ising-Decoding 项目 |
|------|--------|------|----------|-------------------|
| `distance` | 5 | 码距，决定电路大小 | 3-7（更大→更多物理比特） | R=9 (Model 1): 9<br>R=13 (Model 4): 13 |
| `rounds` | 5 | 测量轮数，决定时间深度 | 3-11（更多→更长历史） | R=9: 9<br>R=13: 13 |
| `basis` | "O1" | 表面码方向（O1-O4） | O1, O2, O3, O4 | O1, O2, O3, O4 (code_rotation) |
| `train_shots` | 80000 | 训练样本数 | 10000-100000（快速测试用更小值） | 67M/epoch (8 GPU) |
| `val_shots` | 20000 | 验证样本数 | train_shots 的 1/4 | - |

**25 参数噪声模型（与 Ising-Decoding 一致）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `p_prep_X/Z` | 0.002 | 态制备错误（2 个参数） |
| `p_meas_X/Z` | 0.002 | 测量错误（2 个参数） |
| `p_idle_cnot_X/Y/Z` | 0.001 | CNOT 层空闲错误（3 个参数） |
| `p_idle_spam_X/Y/Z` | 0.001996 | SPAM 窗口空闲错误（3 个参数） |
| `p_cnot_*` | 0.0002 | CNOT 两比特错误（15 个参数） |

> **注：** 本项目现已使用与 Ising-Decoding 一致的 25 参数电路级噪声模型。`basis` 参数使用 O1-O4 表示表面码旋转方向，映射关系：
> - O1 → XV（X 型稳定子优先，垂直边界）
> - O2 → XH（X 型稳定子优先，水平边界）
> - O3 → ZV（Z 型稳定子优先，垂直边界）
> - O4 → ZH（Z 型稳定子优先，水平边界）

#### 模型结构配置（`Conv3DModelConfig`）

| 参数 | 默认值 | 说明 | 推荐范围 | Ising-Decoding 项目 |
|------|--------|------|----------|-------------------|
| `input_channels` | 4 | 输入通道数（固定） | 4（dets→4通道） | 4 |
| `out_channels` | 4 | 输出通道数（固定） | 4（对应 4 个时空通道） | 4 |
| `num_filters` | [64,64,64,4] | 各层卷积核数 | 小模型：[32,32,32,4]，大模型：[128,128,128,4] | Model 1 (R=9): 自定义<br>Model 4 (R=13): 自定义 |
| `kernel_sizes` | [3,3,3,3] | 各层卷积核大小 | 3 或 5 | 3 |
| `activation` | "gelu" | 激活函数 | "relu", "gelu", "leakyrelu" | ReLU |
| `dropout` | 0.1 | Dropout 概率 | 0.0-0.3 | - |

**网络参数量估算：**
```
默认 [64,64,64,4]: ~0.5M 参数
[128,128,128,4]: ~2M 参数
```

#### 训练优化配置（`OptimConfig`）

| 参数 | 默认值 | 说明 | 推荐范围 | Ising-Decoding 项目 |
|------|--------|------|----------|-------------------|
| `batch_size` | 1024 | 批大小 | 64-512（GPU 内存受限时用更小值） | 自适应（按 GPU 数量） |
| `epochs` | 10 | 训练轮数 | 5-20 | 100+（Models 1,4,5 需至少 100） |
| `lr` | 0.001 | 学习率 | 0.0001-0.01 | 自适应调度（基于 milestones） |
| `weight_decay` | 0.00001 | 权重衰减（L2 正则化） | 1e-5-1e-3 | - |
| `device` | "auto" | 训练设备 | "auto", "cpu", "cuda" | 多 GPU DDP |
| `seed` | 0 | 随机种子 | 任意整数 | - |
| `out_dir` | "outputs/ising/train" | 输出目录 | 任意路径 | `$SHARED_OUTPUT_DIR/outputs/` |

### 运行方式

**使用配置文件：**
```bash
python experiments/ising/ising_train_model.py --config experiments/ising/ising_train.yaml
```

**命令行覆盖参数（推荐用于快速测试）：**
```bash
# 快速 smoke 测试（约 30 秒）
python experiments/ising/ising_train_model.py \
  --train-shots 1000 \
  --val-shots 500 \
  --epochs 1 \
  --batch-size 64 \
  --out-dir outputs/ising/test

# 中等规模测试（约 5 分钟）
python experiments/ising/ising_train_model.py \
  --train-shots 10000 \
  --val-shots 5000 \
  --epochs 2 \
  --batch-size 256

# 完整训练
python experiments/ising/ising_train_model.py
```

### 输出文件

训练完成后会生成：

- `best_model.pt` - 最佳模型 checkpoint
- `full_report.json` - 完整训练报告，包含：
  - 配置参数
  - 数据形状信息
  - 训练历史（每个 epoch 的 train_loss, val_loss）
  - 最佳验证损失

### 常见参数调整场景

**快速迭代调试：**
```yaml
distance: 3          # 更小的码
rounds: 3            # 更少的轮数
train_shots: 5000    # 更少的数据
num_filters: [32,32,4]  # 更小的网络
batch_size: 64       # 更小的 batch
epochs: 3
```

**最终高性能模型：**
```yaml
distance: 7          # 更大的码
rounds: 11           # 更多轮数
train_shots: 100000  # 更多数据
num_filters: [128,128,128,4]  # 更大的网络
batch_size: 512      # 更大的 batch
epochs: 20
lr: 0.0005
```

---

## 2) 推理流程

**脚本：** `experiments/ising/ising_full_pipeline.py`

**说明：** 端到端推理/评估流程，包含 PyMatching baseline 与神经网络解码器对比

**流程：**
1. 加载配置
2. 构建电路并采样
3. 加载训练好的 NeuralDecoder
4. 执行推理并计算逻辑错误率（LER）
5. 输出对比报告

### 配置参数

| 参数 | 默认值 | 说明 | Ising-Decoding 项目 |
|------|--------|------|-------------------|
| `checkpoint_path` | "outputs/ising/train/best_model.pt" | 模型 checkpoint 路径 | `$SHARED_OUTPUT_DIR/outputs/$EXPERIMENT_NAME/models/` |
| `shots` | 20000 | 推理样本数 | 用户定义 |
| `latency_samples` | 5000 | 延迟测试样本数 | - |

### 运行方式

```bash
# 使用训练好的模型进行推理
python experiments/ising/ising_full_pipeline.py \
  --checkpoint-path outputs/ising/train/best_model.pt \
  --shots 10000 \
  --out-dir outputs/ising/pipeline
```

---

## 模块架构

重构后的实验脚本调用以下核心模块：

```
decoding_in_one/
├── models/
│   ├── common/              # 可复用组件
│   │   ├── conv.py         # Conv3DBlock, 激活函数
│   │   └── heads.py        # PoolingHead
│   └── surface_code/        # 表面码专用
│       ├── conv3d_decoder.py  # SurfaceCodeConv3DDecoder
│       └── transforms.py      # 数据变换
├── training/                # 训练框架
│   ├── trainer.py         # 通用 Trainer
│   └── config.py           # OptimConfig
├── data/                    # 数据处理
│   ├── sampling.py        # Stim 采样
│   └── datasets.py        # DataLoader 构建
└── decoders/
    └── neural.py          # NeuralDecoder
```

---

## 向后兼容性

重构保留了向后兼容的别名：

```python
# 旧代码仍然可以工作
from decoding_in_one.decoders.predecoder_models import Conv3DPredecoder
from decoding_in_one.models import Conv3DNeuralDecoder
```
