# tests/test_noise/test_base.py
import pytest
from decoding_in_one.noise.base import NoiseModel

def test_abstract_base_cannot_instantiate():
    with pytest.raises(TypeError):
        NoiseModel()
