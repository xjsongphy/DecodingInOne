# tests/test_noise/test_circuit_level.py
import pytest
from decoding_in_one.noise import CircuitLevelNoise
from decoding_in_one.utils.types import CircuitArtifact, CircuitSpec, CodeSpec

def test_load_from_config():
    noise = CircuitLevelNoise.from_config('configs/noise_25p.yaml')
    assert noise.p_prep_X == 0.002
    assert noise.validate() == True

def test_get_parameters():
    noise = CircuitLevelNoise.from_config('configs/noise_25p.yaml')
    params = noise.get_parameters()
    assert 'p_prep_X' in params
    assert len(params) == 25

def test_apply_noise_to_artifact():
    noise = CircuitLevelNoise.from_config('configs/noise_25p.yaml')
    artifact = CircuitArtifact(
        stim_circuit="M 0\n",
        code=CodeSpec(code_family="SurfaceCode", distance=3),
        spec=CircuitSpec(n_rounds=1),
    )
    out = noise.apply_to_circuit(artifact)
    assert isinstance(out, CircuitArtifact)
    assert "Circuit-level noise" in out.stim_circuit
    assert out.metadata["noise_model"] == "circuit_level_25p"
