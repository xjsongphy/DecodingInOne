# decoding_in_one/codes/__init__.py
from decoding_in_one.codes.base import QuantumCode, PauliString
from decoding_in_one.codes.surface_code import SurfaceCode

# 暴露 data_mapping 但不放在 __all__ 中
from decoding_in_one.codes.surface_code import data_mapping

__all__ = ['QuantumCode', 'PauliString', 'SurfaceCode', 'data_mapping']
