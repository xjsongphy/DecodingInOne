# Ising 解码器集成设计文档

**日期**: 2026-05-30
**目标**: 将 Ising 相关代码从 `experiments/ising/` 整合到 `decoding_in_one` 核心模块

## 背景

当前 Ising 相关代码集中在 `experiments/ising/` 目录下，包含：
- `ising_train_model.py` - 训练脚本（数据处理、模型训练）
- `ising_full_pipeline.py` - 推理流程（PyMatching 集成）
- `decoding_in_one/decoders/predecoder_models.py` - Conv3DPredecoder 模型

这种结构导致：
1. experiments 目录包含过多实现逻辑
2. 数据处理、模型、训练耦合在一起
3. 难以复用和扩展

## 设计目标

1. **职责分离** - 模型、数据变换、训练、解码各自独立
2. **可复用性** - 核心功能可在 experiments 脚本中复用
3. **可扩展性** - 支持未来添加 Transformer、GNN 等其他网络架构
4. **清晰边界** - 避免实验逻辑固化到核心包中

## 架构设计

### 目录结构

```
decoding_in_one/
├── models/                          # 神经网络模型
│   ├── __init__.py
│   ├── base.py                      # DecodingModel 基类
│   ├── conv3d.py                    # Conv3DNeuralDecoder 网络
│   └── transforms.py                # 数据变换函数
├── training/                        # 训练框架
│   ├── __init__.py
│   ├── base.py                      # Trainer 基类
│   ├── config.py                    # OptimConfig（仅训练/优化配置）
│   └── trainer.py                   # 通用训练器
├── data/                            # 数据处理（新增）
│   ├── __init__.py
│   ├── sampling.py                  # Stim 采样
│   └── datasets.py                  # TensorDataset/DataLoader helper
├── decoders/
│   ├── __init__.py
│   ├── base.py
│   ├── pymatching.py                # PyMatching 解码器
│   ├── neural.py                    # NeuralDecoder
│   └── predecoder_models.py        # 向后兼容 re-export
└── (现有) circuits/, codes/, noise/, evaluation/
```

### 职责划分

#### models/ - 神经网络模型

- **base.py** - `DecodingModel` 抽象基类（继承 `nn.Module`）
- **conv3d.py** - `Conv3DNeuralDecoder` 网络结构
- **transforms.py** - 数据变换函数（`dets_to_conv3d_input`, `obs_to_conv3d_target`）

#### training/ - 训练框架

- **base.py** - `Trainer` 抽象基类
- **config.py** - `OptimConfig` 仅包含训练/优化配置
- **trainer.py** - 通用 PyTorch 训练循环（只接收 dataloader）

#### data/ - 数据处理（新增）

- **sampling.py** - Stim 采样工具
- **datasets.py** - TensorDataset/DataLoader 构建助手

#### decoders/ - 解码器接口

- **neural.py** - `NeuralDecoder` 使用训练好的模型进行推理
- **predecoder_models.py** - 向后兼容层，re-export `Conv3DNeuralDecoder`

## 模块详细设计

### 1. models/base.py

```python
from abc import ABC, abstractmethod
import torch.nn as nn

class DecodingModel(nn.Module, ABC):
    """神经网络解码模型的抽象基类"""

    @abstractmethod
    def get_input_channels(self) -> int:
        """返回输入通道数"""
        pass

    @abstractmethod
    def expected_input_rank(self) -> int:
        """返回期望的输入张量维度（如 5 表示 B,C,T,D,D）"""
        pass
```

### 2. models/conv3d.py

```python
import torch
import torch.nn as nn

class Conv3DNeuralDecoder(DecodingModel):
    """Conv3D 神经网络解码器"""

    def __init__(
        self,
        input_channels: int = 4,
        out_channels: int = 4,
        num_filters: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        dropout_p: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        # Conv3D 网络定义（迁移自 predecoder_models.py）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输出 logical observable 预测"""
        return self.net(x)

    def get_input_channels(self) -> int:
        return self.input_channels

    def expected_input_rank(self) -> int:
        return 5  # (B, C, T, D, D)
```

### 3. models/transforms.py

```python
import numpy as np

def dets_to_conv3d_input(dets: np.ndarray, rounds: int, distance: int) -> np.ndarray:
    """将检测器数据转换为 Conv3D 输入格式 (B,4,T,D,D)

    迁移自 ising_train_model.py 的 _dets_to_trainx
    """

def obs_to_conv3d_target(obs: np.ndarray, rounds: int, distance: int) -> np.ndarray:
    """将观测量广播为 Conv3D 目标格式 (B,4,T,D,D)

    迁移自 ising_train_model.py 的 _obs_to_target4
    """
```

### 4. training/config.py

```python
from dataclasses import dataclass

@dataclass
class OptimConfig:
    """训练优化配置（仅包含训练循环相关参数）"""
    batch_size: int = 512
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "auto"
    seed: int = 0
    out_dir: str = "experiments/ising/train_output"
```

### 5. data/config.py（或 experiments/ising/config.py）

```python
from dataclasses import dataclass

@dataclass
class IsingDataConfig:
    """Ising 实验数据配置（数据生成相关）"""
    distance: int = 5
    rounds: int = 5
    basis: str = "X"
    train_shots: int = 20000
    val_shots: int = 5000

    # 噪声参数
    p_after_clifford: float = 0.001
    p_before_round_data: float = 0.001
    p_before_measure_flip: float = 0.001
    p_after_reset_flip: float = 0.001

    # 模型参数
    num_filters: list[int] | None = None
    kernel_sizes: list[int] | None = None
    activation: str = "gelu"
    dropout: float = 0.1

    def __post_init__(self):
        if self.num_filters is None:
            self.num_filters = [64, 64, 64, 4]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 3, 3, 3]
```

### 6. training/trainer.py

```python
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Any
from .config import OptimConfig

class Trainer:
    """通用 PyTorch 训练器"""

    def __init__(self, model: nn.Module, config: OptimConfig):
        self.model = model
        self.config = config
        self.device = self._get_device()

    def train(
        self,
        train_loader,
        val_loader,
        criterion: nn.Module = None
    ) -> dict[str, Any]:
        """执行训练循环

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数（默认 BCEWithLogitsLoss）

        Returns:
            训练报告字典
        """
        # 训练循环逻辑（不涉及电路构建和采样）
```

### 7. data/sampling.py

```python
import numpy as np
import stim

def sample_detectors_observables(
    circuit: stim.Circuit,
    shots: int,
    seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """从电路采样检测器和观测量

    Returns:
        (dets, obs): 检测器数组和观测量数组
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    return np.asarray(dets, dtype=np.float32), np.asarray(obs, dtype=np.float32)
```

### 8. data/datasets.py

```python
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def build_dataloaders(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    batch_size: int,
) -> tuple:
    """构建训练和验证数据加载器"""
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader
```

### 9. decoders/neural.py

```python
from .base import Decoder, Correction
from ..models.transforms import dets_to_conv3d_input

class NeuralDecoder(Decoder):
    """使用训练好的神经网络模型进行 logical observable 预测

    注意：Correction.predictions 表示 predicted observables，不是 qubit correction
    """

    def __init__(
        self,
        model,
        checkpoint_path: str,
        rounds: int,
        distance: int,
        threshold: float = 0.5
    ):
        """
        Args:
            model: DecodingModel 实例
            checkpoint_path: 模型权重路径
            rounds: 电路轮数（用于数据变换）
            distance: 码距（用于数据变换）
            threshold: 预测阈值
        """
        self.model = model
        self.rounds = rounds
        self.distance = distance
        self.threshold = threshold
        self.load_checkpoint(checkpoint_path)

    def decode(self, syndrome, observables=None) -> Correction:
        """
        Args:
            syndrome: 检测器测量结果 (batch, n_detectors)
            observables: 观测量（可选，用于评估）

        Returns:
            Correction: logical observable 预测
        """
        # 1. 使用 transforms 转换 syndrome 为模型输入
        x = dets_to_conv3d_input(syndrome, self.rounds, self.distance)
        # 2. 模型推理
        logits = self.model(x)
        # 3. 应用阈值
        pred = (torch.sigmoid(logits) >= self.threshold).float()
        # 4. 返回 Correction（predictions 是 logical observable）
        return Correction(predictions=pred)
```

### 10. decoders/predecoder_models.py（向后兼容）

```python
# 向后兼容：re-export Conv3DNeuralDecoder
from decoding_in_one.models.conv3d import Conv3DNeuralDecoder as Conv3DPredecoder

__all__ = ['Conv3DPredecoder']
```

## experiments 脚本简化

重构后，`experiments/ising/ising_train_model.py` 将简化为：

```python
from decoding_in_one.models import Conv3DNeuralDecoder
from decoding_in_one.models.transforms import dets_to_conv3d_input, obs_to_conv3d_target
from decoding_in_one.training import Trainer, OptimConfig
from decoding_in_one.data import sample_detectors_observables, build_dataloaders
from decoding_in_one.circuits import MemoryCircuit
from decoding_in_one.codes import SurfaceCode

# 实验脚本仅负责：
# 1. 配置加载（IsingDataConfig + OptimConfig）
# 2. 电路构建（调用 circuits/）
# 3. 数据采样（调用 data/sampling）
# 4. 数据预处理（调用 transforms）
# 5. 训练（调用 Trainer.train）
```

## 迁移计划

### 阶段 1：创建新模块
1. 创建 `models/` 目录及文件（base.py, conv3d.py, transforms.py）
2. 创建 `training/` 目录及文件（base.py, config.py, trainer.py）
3. 创建 `data/` 目录及文件（sampling.py, datasets.py）
4. 更新 `decoders/neural.py`

### 阶段 2：迁移代码
1. 迁移 `Conv3DPredecoder` 到 `models/conv3d.py`
2. 迁移 `_dets_to_trainx`, `_obs_to_target4` 到 `models/transforms.py`
3. 拆分配置：`TrainConfig` → `OptimConfig` + `IsingDataConfig`
4. 迁移训练循环到 `training/trainer.py`
5. 迁移采样逻辑到 `data/sampling.py`
6. 创建 `decoders/neural.py`（携带 rounds/distance 参数）
7. 创建 `decoders/predecoder_models.py` 向后兼容层

### 阶段 3：更新 experiments
1. 重写 `experiments/ising/ising_train_model.py`
2. 重写 `experiments/ising/ising_full_pipeline.py`
3. 测试验证

### 阶段 4：清理
1. 保留 `decoders/predecoder_models.py` 作为向后兼容层
2. 更新文档

## 未来扩展

### 支持其他网络架构
- `models/transformer.py` - Transformer 解码器
- `models/gnn.py` - GNN 解码器
- 对应的 `transforms.py` 中的数据变换函数

### 支持其他量子纠错码
- 通过 `Trainer` 的灵活性支持不同码
- 数据变换函数可扩展支持不同码的几何结构

## 关键设计决策

1. **数据处理与模型分离** - `transforms.py` 独立存放数据变换，便于复用和测试
2. **训练器边界清晰** - `Trainer` 只负责训练循环，不包含电路构建和采样
3. **配置职责分离** - `OptimConfig` 仅含训练参数，`IsingDataConfig` 含数据生成参数
4. **电路构建职责** - 继续由 `circuits/` 模块负责，避免职责重叠
5. **模型输出明确** - `NeuralDecoder` 输出 logical observable 预测，是 standalone 解码器
6. **接口完整性** - `NeuralDecoder` 初始化时显式携带 `rounds/distance` 等变换参数
7. **向后兼容** - 保留 `predecoder_models.py` 作为兼容层，避免破坏性变更
