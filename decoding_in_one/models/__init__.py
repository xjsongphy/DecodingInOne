"""神经网络模型模块

按码类型组织，common 包含可复用组件。
"""

# 向后兼容：从 surface_code 导出
from .surface_code import (
    SurfaceCodeConv3DDecoder,
    Conv3DNeuralDecoder,  # 别名
)

from .base import DecodingModel
from .config import Conv3DModelConfig

__all__ = ["DecodingModel", "Conv3DModelConfig", "SurfaceCodeConv3DDecoder", "Conv3DNeuralDecoder"]
