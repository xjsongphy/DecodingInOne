"""神经网络模型模块

按码类型组织，common 包含可复用组件。
"""

# 向后兼容：从 surface_code 导出
from .surface_code import (
    SurfaceCodeConv3DDecoder,
    Conv3DNeuralDecoder,  # 别名
)

# GNN 解码器
from .surface_code.gnn_decoder import SurfaceCodeGNNDecoder

from .base import DecodingModel
from .config import Conv3DModelConfig, GNNModelConfig

__all__ = [
    "DecodingModel",
    "Conv3DModelConfig",
    "GNNModelConfig",
    "SurfaceCodeConv3DDecoder",
    "Conv3DNeuralDecoder",
    "SurfaceCodeGNNDecoder",
]
