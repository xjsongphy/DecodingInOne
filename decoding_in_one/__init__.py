# decoding_in_one/__init__.py
"""
DecoderInOne - 模块化量子纠错解码框架
"""

__version__ = "0.1.0"

# 核心模块导出（随着实现逐步添加）
from decoding_in_one.utils.types import DecodingBatch, CodeSpec, CircuitSpec, CircuitArtifact
from decoding_in_one.codes import QuantumCode
from decoding_in_one.noise import NoiseModel
from decoding_in_one.circuits import CircuitBuilder
from decoding_in_one.decoders import Decoder, Correction

__all__ = [
    'DecodingBatch',
    'CodeSpec',
    'CircuitSpec',
    'CircuitArtifact',
    'QuantumCode',
    'NoiseModel',
    'CircuitBuilder',
    'Decoder',
    'Correction',
]
