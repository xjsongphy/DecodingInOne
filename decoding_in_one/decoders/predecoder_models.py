# decoding_in_one/decoders/predecoder_models.py
"""向后兼容层

为现有代码提供 Conv3DPredecoder 别名。
"""

# 从新的 surface_code 模块导入
from decoding_in_one.models.surface_code import SurfaceCodeConv3DDecoder as Conv3DPredecoder

__all__ = ['Conv3DPredecoder']
