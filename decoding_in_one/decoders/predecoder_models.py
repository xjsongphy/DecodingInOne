# decoding_in_one/decoders/predecoder_models.py
"""向后兼容层

为现有代码提供 Conv3DPredecoder 别名。
"""

from decoding_in_one.models.conv3d import Conv3DNeuralDecoder as Conv3DPredecoder

__all__ = ['Conv3DPredecoder']
