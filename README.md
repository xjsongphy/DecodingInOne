# DecoderInOne

**模块化量子纠错解码框架**

DecoderInOne 是一个用于量子纠错码研究的模块化 Python 框架。该框架提供了统一的接口来构建量子电路、应用噪声模型、执行解码算法，并评估解码性能。

## 核心特性

### 模块化架构
- **可扩展的码类型**：通过 `QuantumCode` 基类轻松添加新的量子纠错码
- **灵活的噪声模型**：支持从简单到复杂的电路级噪声
- **可插拔的解码器**：统一接口集成多种解码算法
- **灵活的电路构建**：支持不同的电路模式（存储、计算等）

### 支持的功能
- **表面码 (Surface Code)**：标准旋转表面码实现
- **QLDPC 码**：支持非近邻比特间的 CNOT 操作
- **25参数电路级噪声**：完整的 CNOT、空闲、制备/测量误差模型
- **PyMatching 集成**：最小权重完美匹配解码
- **神经网络解码器**：支持基于机器学习的解码方法
- **Stim 电路生成**：与 Stim 电路模拟器无缝集成

## 安装

### 环境要求
- Python 3.10+
- uv（推荐的 Python 包管理器）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/yourusername/DecoderInOne.git
cd DecoderInOne

# 使用 uv 安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

### 核心依赖

- `stim` >= 1.9.0 — 量子电路模拟
- `pymatching` — 最小权重完美匹配解码
- `torch` — 神经网络解码器支持
- `numpy` — 数值计算
- `pyyaml` — 配置文件管理
- `networkx` — 图算法支持

## 快速开始

### 基础示例：表面码存储实验

```python
from decoding_in_one.codes import SurfaceCode
from decoding_in_one.noise import CircuitLevelNoise
from decoding_in_one.circuits import MemoryCircuit

# 1. 定义量子纠错码
code = SurfaceCode(distance=5, rotation='XV')
print(f"物理比特数: {code.get_n_physical()}")
print(f"逻辑比特数: {code.get_n_logical()}")

# 2. 加载噪声模型配置
noise = CircuitLevelNoise.from_config('configs/noise_25p.yaml')

# 3. 构建电路
builder = MemoryCircuit(code=code, noise=noise)
circuit = builder.build_memory_circuit(
    code=code,
    n_rounds=5,
    measurement_basis='X'
)

# 4. 保存电路文件
with open('my_circuit.stim', 'w') as f:
    f.write(circuit)
```

运行示例实验：

```bash
python experiments/surface_code_basic.py
```

## 项目结构

```
DecoderInOne/
├── decoding_in_one/           # 核心包
│   ├── __init__.py
│   ├── utils/
│   │   └── types.py          # 数据类型定义
│   ├── codes/                # 量子纠错码模块
│   │   ├── base.py           # QuantumCode 抽象基类
│   │   └── surface_code.py   # 表面码实现
│   ├── noise/                # 噪声模型模块
│   │   ├── base.py           # NoiseModel 抽象基类
│   │   └── circuit_level.py  # 25参数电路级噪声
│   ├── circuits/             # 电路构建模块
│   │   ├── base.py           # CircuitBuilder 抽象基类
│   │   └── memory_circuit.py # 重复测量电路
│   ├── decoders/             # 解码器模块
│   │   ├── base.py           # Decoder 抽象基类
│   │   └── pymatching.py     # PyMatching 解码器
│   └── evaluation/           # 性能评估模块
│       └── metrics.py        # 指标计算器
├── configs/                  # 配置文件
│   └── noise_25p.yaml        # 25参数噪声配置
├── experiments/              # 实验脚本
│   └── surface_code_basic.py
├── tests/                   # 测试套件
│   ├── test_codes/
│   ├── test_noise/
│   ├── test_circuits/
│   ├── test_decoders/
│   └── test_evaluation/
├── pyproject.toml           # uv 项目配置
└── README.md
```

## 模块详解

### 码模块 (Codes)

`QuantumCode` 是所有量子纠错码的基类，定义了统一接口：

```python
from decoding_in_one.codes import QuantumCode

class MyCode(QuantumCode):
    def get_n_physical(self) -> int:
        """返回物理比特数"""
        ...
    
    def get_n_logical(self) -> int:
        """返回逻辑比特数"""
        ...
```

### 噪声模块 (Noise)

`NoiseModel` 基类支持参数化噪声配置：

```python
from decoding_in_one.noise import CircuitLevelNoise

# 从配置文件加载
noise = CircuitLevelNoise.from_config('configs/noise_25p.yaml')

# 或直接指定参数
noise = CircuitLevelNoise(
    p_prep_X=0.001,
    p_prep_Z=0.001,
    p_meas_X=0.001,
    p_meas_Z=0.001,
    p_idle_X=0.0001,
    p_idle_Y=0.0001,
    p_idle_Z=0.0001,
    # ... CNOT 参数
)
```

### 电路模块 (Circuits)

`CircuitBuilder` 生成 Stim 电路字符串：

```python
from decoding_in_one.circuits import MemoryCircuit

builder = MemoryCircuit(code=code, noise=noise)
circuit = builder.build_memory_circuit(
    code=code,
    n_rounds=10,
    measurement_basis='X'
)
```

### 解码器模块 (Decoders)

统一解码接口：

```python
from decoding_in_one.decoders import PyMatchingDecoder

decoder = PyMatchingDecoder()
correction = decoder.decode(syndrome, observables)
```

## 配置文件

噪声模型使用 YAML 配置文件，便于参数管理：

```yaml
# configs/noise_25p.yaml
p_prep_X: 0.002
p_prep_Z: 0.002
p_meas_X: 0.002
p_meas_Z: 0.002
p_idle_X: 0.0001
p_idle_Y: 0.0001
p_idle_Z: 0.0001
p_cnot_IX: 0.0005
# ... 完整 25 个参数
```

## 扩展开发

### 添加新的量子纠错码

1. 继承 `QuantumCode` 基类
2. 实现必需方法：`get_n_physical()`、`get_n_logical()`
3. 实现码特定的几何结构和稳定子连接

```python
from decoding_in_one.codes.base import QuantumCode

class QLDPCCode(QuantumCode):
    def __init__(self, hx, hz):
        self.hx = hx  # X 稳定子校验矩阵
        self.hz = hz  # Z 稳定子校验矩阵
        # ...
```

### 添加新的噪声模型

1. 继承 `NoiseModel` 基类
2. 实现参数管理和噪声应用逻辑

### 添加新的解码器

1. 继承 `Decoder` 基类
2. 实现 `decode()` 方法

## 测试

运行完整测试套件：

```bash
pytest tests/ -v
```

运行特定模块测试：

```bash
pytest tests/test_codes/ -v
pytest tests/test_noise/ -v
pytest tests/test_circuits/ -v
```

## 性能评估

使用 `MetricsCalculator` 计算标准指标：

```python
from decoding_in_one.evaluation import MetricsCalculator

# 逻辑错误率
ler = MetricsCalculator.logical_error_rate(
    predictions=predicted_observables,
    actual=actual_observables
)

# 症状密度
density = MetricsCalculator.syndrome_density(syndrome)

# 密度减少因子
reduction = MetricsCalculator.syndrome_density_reduction(
    before=raw_syndrome,
    after=processed_syndrome
)
```

## 开发指南

### TDD 工作流

本项目采用测试驱动开发：

1. 为新功能编写测试
2. 运行测试确认失败
3. 实现最小代码使测试通过
4. 重构优化
5. 提交代码

### Git 提交规范

- 使用清晰描述性消息
- 频繁提交，每次提交完成一个逻辑单元
- 提交前确保测试通过

### 代码风格

- 遵循 PEP 8 规范
- 使用类型注解
- 添加文档字符串

## 相关项目

- [Ising-Decoding](https://github.com/quantum-research/Ising-Decoding) — 神经网络辅助解码器

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 引用

如果您在研究中使用了 DecoderInOne，请引用：

```
@software{decoderinone,
  title={DecoderInOne: A Modular Quantum Error Correction Decoding Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/DecoderInOne}
}
```
