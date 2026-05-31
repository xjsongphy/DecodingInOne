# decoding_in_one/noise/__init__.py
"""噪声模型模块

包含完整的 25 参数电路级噪声模型实现。
"""

from decoding_in_one.noise.base import NoiseModel as NoiseModelBase
from decoding_in_one.noise.circuit_level import (
    NoiseModel,
    CircuitLevelNoise,  # 向后兼容别名
    _single_p_mapping,
    CNOT_ERROR_TYPES,
    CNOT_ERROR_INDEX,
    get_grouped_totals,
    get_training_upscaled_noise_model,
    SURFACE_CODE_TRAINING_UPSCALE_TARGET,
    SURFACE_CODE_THRESHOLD_APPROX,
)

__all__ = [
    'NoiseModelBase',
    'NoiseModel',
    'CircuitLevelNoise',
    '_single_p_mapping',
    'CNOT_ERROR_TYPES',
    'CNOT_ERROR_INDEX',
    'get_grouped_totals',
    'get_training_upscaled_noise_model',
    'SURFACE_CODE_TRAINING_UPSCALE_TARGET',
    'SURFACE_CODE_THRESHOLD_APPROX',
]
