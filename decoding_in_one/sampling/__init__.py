# decoding_in_one/sampling/__init__.py
"""DEM 矩阵采样模块

包含 GPU 加速的检测器误差模型采样、辅助函数和数据生成器。
"""

from decoding_in_one.sampling.dem import (
    dem_sampling,
    measure_from_stacked_frames,
    timelike_syndromes,
)

__all__ = [
    "dem_sampling",
    "measure_from_stacked_frames",
    "timelike_syndromes",
]
