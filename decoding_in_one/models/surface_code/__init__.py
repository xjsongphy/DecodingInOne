"""models/surface_code 模块

表面码专用的模型和数据变换。
"""

from .conv3d_decoder import SurfaceCodeConv3DDecoder
from .transforms import (
    dets_to_conv3d_input,
    obs_to_conv3d_target,
    reduce_conv3d_output,
)

# 向后兼容别名
Conv3DNeuralDecoder = SurfaceCodeConv3DDecoder

__all__ = [
    "SurfaceCodeConv3DDecoder",
    "Conv3DNeuralDecoder",  # 向后兼容
    "dets_to_conv3d_input",
    "obs_to_conv3d_target",
    "reduce_conv3d_output",
]
