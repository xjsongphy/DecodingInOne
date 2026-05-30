# tests/test_circuits/test_base.py
import pytest
from decoding_in_one.circuits.base import CircuitBuilder

def test_abstract_base_cannot_instantiate():
    with pytest.raises(TypeError):
        CircuitBuilder()
