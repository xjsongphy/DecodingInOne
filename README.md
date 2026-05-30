# DecoderInOne

模块化量子纠错解码框架

## 特性

- 模块化设计，易于扩展
- 支持多种量子纠错码（表面码、QLDPC 等）
- 支持多种噪声模型
- 支持多种解码算法
- 简单的实验脚本接口

## 安装

```bash
uv sync
```

## 快速开始

```python
from decoding_in_one.codes import SurfaceCode
from decoding_in_one.noise import CircuitLevelNoise
from decoding_in_one.decoders import PyMatchingDecoder

code = SurfaceCode(distance=7)
noise = CircuitLevelNoise.from_config('configs/noise_25p.yaml')
decoder = PyMatchingDecoder()
# ... 完整示例见 experiments/
```
