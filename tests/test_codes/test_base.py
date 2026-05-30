# tests/test_codes/test_base.py
import pytest
from decoding_in_one.codes.base import QuantumCode

def test_abstract_base_cannot_instantiate():
    with pytest.raises(TypeError):
        QuantumCode()
