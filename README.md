# DecodingInOne

**基于 Ising-Decoding 架构的模块化量子纠错解码框架**

DecodingInOne 是一个用于量子纠错码研究的模块化 Python 框架。该框架基于 [Ising-Decoding](https://github.com/quantum-research/Ising-Decoding) 架构，提供了统一的接口来构建量子电路、应用 25 参数电路级噪声模型、执行解码算法，并评估解码性能。

## 核心特性

### 基于 Ising-Decoding 架构
- **25 参数电路级噪声**：完整的电路级噪声建模（态制备、测量、空闲、CNOT）
- **MemoryCircuit 电路构建**：手动构建 Stim 电路，支持精确的噪声控制
- **通用码接口**：支持表面码和未来的 QLDPC 等任意量子纠错码
- **距离依赖噪声接口**：为空间相关噪声预留接口

### 模块化架构
- **可扩展的码类型**：通过 `QuantumCode` 基类轻松添加新的量子纠错码
- **灵活的噪声模型**：支持 25 参数电路级噪声和距离相关噪声
- **可插拔的解码器**：统一接口集成多种解码算法
- **通用电路构建**：支持局部码（表面码）和非局部码（QLDPC）

### 支持的功能
- **表面码 (Surface Code)**：标准旋转表面码实现（O1-O4 基）
- **QLDPC 码**：支持非近邻比特间的 CNOT 操作（接口已预留）
- **25 参数电路级噪声**：完整的 CNOT、空闲、制备/测量误差模型
- **神经网络解码器**：基于机器学习的解码方法
- **Stim 电路生成**：与 Stim 电路模拟器无缝集成

## 安装

### 环境要求
- Python 3.10+
- uv（推荐的 Python 包管理器）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/xjsongphy/DecodingInOne.git
cd DecodingInOne

# 使用 uv 安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

### 核心依赖

- `stim` >= 1.9.0 — 量子电路模拟
- `torch` — 神经网络解码器支持
- `numpy` — 数值计算
- `pyyaml` — 配置文件管理

## 快速开始

### Ising 风格训练（推荐）

基于 Ising-Decoding 架构的训练流程，**完全独立运行，无需外部依赖**：

#### Conv3D 解码器训练

```bash
# 使用默认配置训练（自动从 Stim 电路提取 DEM）
python experiments/ising/ising_train_model.py

# 使用自定义配置
python experiments/ising/ising_train_model.py --config experiments/ising/ising_train.yaml

# 覆盖特定参数
python experiments/ising/ising_train_model.py \
    --distance 9 \
    --rounds 9 \
    --basis O1 \
    --train-shots 100000 \
    --epochs 32
```

#### GNN 解码器训练

基于图神经网络的解码器，使用与 Conv3D 相当的参数量（~1.3M）：

```bash
# 使用 GNN 配置训练
python experiments/ising/ising_train_model.py --config experiments/ising/ising_train_gnn.yaml

# GNN 特定参数覆盖
python experiments/ising/ising_train_model.py \
    --config experiments/ising/ising_train_gnn.yaml \
    --distance 9 \
    --rounds 9 \
    --train-shots 100000 \
    --epochs 64
```

**GNN 解码器特性**：
- 8 层图注意力网络（GAT）
- 256 隐藏通道，8 个注意力头
- 参数量：1.32M（与 Conv3D 的 1.36M 相当）
- 自动构建 syndrome 图结构（检测器节点 + 数据比特节点）
- 支持空间边、时间边和支撑边

### 并行采样配置

对于大批量数据生成（100,000+ 样本），可以启用并行采样加速：

**配置文件 (`ising_train.yaml`)**：
```yaml
# 并行采样配置（可选，默认关闭）
enable_parallel: true    # 启用并行采样
num_workers: 4          # 工作数（GPU 数量或 CPU 进程数）
parallel_device_ids: [0, 1]  # 指定使用的 GPU（null = 自动检测）
```

**何时启用并行**：
- ✅ `train_shots ≥ 100,000` - 推荐启用
- ✅ 有多张 GPU 或多核 CPU
- ❌ `train_shots < 10,000` - 不推荐（开销大于收益）

**性能提升**：
- 单 GPU → 双 GPU：~2x 加速
- 单进程 → 八进程 CPU：~7x 加速

**输出目录结构**：
```
outputs/ising/YYYYMMDD_HHMMSS/
├── best_model.pt       # 最佳模型检查点
├── train_history.json  # 每个 epoch 的历史
├── iter_history.json   # 每个迭代的历史
└── full_report.json    # 完整训练报告
```

**完全独立运行**：
- ✅ 自动从 MemoryCircuit 生成的 Stim 电路提取 DEM 矩阵
- ✅ 不依赖 Ising-Decoding 的预计算文件
- ✅ 可选使用预计算文件加速（`precomputed_frames_dir`）
- ✅ 可选启用并行采样加速（`enable_parallel`）

### 基础代码示例

```python
from decoding_in_one.codes import SurfaceCode
from decoding_in_one.noise import NoiseModel
from decoding_in_one.circuits import MemoryCircuit

# 1. 定义量子纠错码
code = SurfaceCode(distance=5, rotation='XV')
print(f"物理比特数: {code.get_n_physical()}")
print(f"逻辑比特数: {code.get_n_logical()}")

# 2. 创建 25 参数噪声模型
noise_model = NoiseModel(
    p_prep_X=0.002, p_prep_Z=0.002,
    p_meas_X=0.002, p_meas_Z=0.002,
    p_idle_cnot_X=0.001, p_idle_cnot_Y=0.001, p_idle_cnot_Z=0.001,
    p_idle_spam_X=0.001, p_idle_spam_Y=0.001, p_idle_spam_Z=0.001,
    # ... CNOT 参数
)

# 3. 构建 MemoryCircuit（通用化稳定子测量）
circuit = MemoryCircuit(
    code=code,
    n_rounds=5,
    basis='X',
    noise_model=noise_model,
)

# 4. 获取 Stim 电路
stim_circuit = circuit.compile_to_stim()
print(f"电路长度: {len(stim_circuit)}")
```

## 项目结构

```
DecodingInOne/
├── decoding_in_one/           # 核心包
│   ├── __init__.py
│   ├── codes/                # 量子纠错码模块
│   │   ├── base.py           # QuantumCode 抽象基类（通用接口）
│   │   └── surface_code.py   # 表面码实现
│   ├── noise/                # 噪声模型模块
│   │   ├── base.py           # NoiseModel 抽象基类
│   │   └── circuit_level.py  # 25参数电路级噪声（Ising-Decoding 迁移）
│   ├── circuits/             # 电路构建模块
│   │   ├── base.py           # CircuitBuilder 抽象基类
│   │   └── memory_circuit.py # MemoryCircuit（Ising-Decoding 迁移）
│   ├── models/               # 神经网络解码器
│   │   └── surface_code.py   # Conv3D 解码器
│   ├── training/             # 训练模块
│   │   └── optimizer.py     # 训练器
│   ├── data/                 # 数据处理
│   │   └── config.py         # 数据配置
│   └── sampling/             # 采样模块
│       └── generator.py      # CircuitDataGenerator
├── experiments/              # 实验脚本
│   └── ising/               # Ising 风格实验
│       ├── ising_train_model.py  # 训练脚本
│       └── ising_train.yaml      # 训练配置
├── test_new_interfaces.py   # 接口测试脚本
├── pyproject.toml           # uv 项目配置
└── README.md
```

## 架构说明

### 基于 Ising-Decoding 的数据流

```
NoiseModel (25p) + QuantumCode
    ↓
MemoryCircuit → Stim 电路（通用化稳定子测量）
    ↓
自动提取 DEM 矩阵 H, p（从 Stim 电路）
    ↓
CircuitDataGenerator.generate_batch()
    ↓
dem_sampling(H, p) → GPU/CPU 采样
    ↓
DataLoader → Trainer → 模型
```

**完全独立**：所有步骤在本地完成，无需外部预计算文件。

### 25 参数噪声模型

```python
# 态制备错误（2）
p_prep_X, p_prep_Z

# 测量错误（2）
p_meas_X, p_meas_Z

# CNOT 层空闲错误（3）
p_idle_cnot_X, p_idle_cnot_Y, p_idle_cnot_Z

# SPAM 窗口空闲错误（3）
p_idle_spam_X, p_idle_spam_Y, p_idle_spam_Z

# CNOT 两比特错误（15）
p_cnot_IX, p_cnot_IY, ..., p_cnot_ZZ
```

### 通用码接口

`QuantumCode` 基类提供了完整的通用接口：

```python
class QuantumCode(ABC):
    # 基础接口
    def get_n_physical(self) -> int
    def get_n_logical(self) -> int
    def get_stabilizers(self) -> List[PauliString]

    # H 矩阵和图结构（为 QLDPC 设计）
    def get_parity_check_matrices(self) -> Dict[str, np.ndarray]
    def get_stabilizer_graph(self) -> Dict[str, List[Tuple[int, int]]]
    def get_stabilizer_measurement_layers(self, stabilizer_type: str) -> List[List[Tuple[int, int]]]

    # 空间坐标和距离相关噪声
    def get_qubit_coordinates(self) -> Dict[int, QubitCoordinate]
    def get_spatial_distance(self, qubit1: int, qubit2: int) -> float
    def get_noise_scaling_factor(self, qubit1: int, qubit2: int) -> float
```

## 配置文件

训练配置示例（`experiments/ising/ising_train.yaml`）：

```yaml
# 数据生成配置
distance: 9
rounds: 9
basis: O1  # O1, O2, O3, O4
train_shots: 800000
val_shots: 200000
precomputed_frames_dir: path/to/frames  # DEM artifacts 目录

# 25 参数噪声模型
p_prep_X: 0.002
p_prep_Z: 0.002
p_meas_X: 0.002
p_meas_Z: 0.002
# ... 完整 25 个参数

# 模型配置
input_channels: 4
num_filters: [64, 64, 64, 4]

# 训练配置
batch_size: 1024
epochs: 64
lr: 0.001
```

## 扩展开发

### 添加 QLDPC 码

```python
from decoding_in_one.codes.base import QuantumCode

class QLDPCCode(QuantumCode):
    def __init__(self, hx: np.ndarray, hz: np.ndarray):
        self.hx = hx  # X 稳定子校验矩阵
        self.hz = hz  # Z 稳定子校验矩阵
        # 实现其他必需方法...

    def get_stabilizer_measurement_layers(self, stabilizer_type: str):
        # QLDPC 使用图着色处理非局部连接
        # 返回不冲突的 CNOT 层
        pass
```

### 添加距离依赖噪声

```python
class MyCode(QuantumCode):
    def get_noise_scaling_factor(self, qubit1: int, qubit2: int) -> float:
        distance = self.get_spatial_distance(qubit1, qubit2)
        # 实现距离-噪声关系，如指数衰减
        return np.exp(-distance / correlation_length)
```

## 测试

运行接口测试：

```bash
python test_new_interfaces.py
```

## 相关项目

- [Ising-Decoding](https://github.com/quantum-research/Ising-Decoding) — 神经网络辅助解码器（本项目架构来源）

## 架构对比

| 特性 | Ising-Decoding | DecodingInOne |
|------|----------------|---------------|
| 噪声模型 | 25 参数 | ✅ 25 参数（完全一致） |
| 电路构建 | MemoryCircuit | ✅ MemoryCircuit（迁移） |
| 码类型 | 表面码 | ✅ 表面码 + QLDPC 接口 |
| 距离噪声 | ❌ | ✅ 接口预留 |
| 通用性 | 表面码专用 | ✅ QuantumCode 抽象 |

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
