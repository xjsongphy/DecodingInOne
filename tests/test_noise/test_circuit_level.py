# tests/test_noise/test_circuit_level.py
import pytest
from decoding_in_one.noise import CircuitLevelNoise

def test_load_from_config():
    noise = CircuitLevelNoise.from_config('configs/noise_25p.yaml')
    assert noise.p_prep_X == 0.002
    assert noise.validate() == True

def test_get_parameters():
    noise = CircuitLevelNoise.from_config('configs/noise_25p.yaml')
    params = noise.get_parameters()
    assert 'p_prep_X' in params
    assert len(params) == 25
