# tests/test_codes/test_surface_code.py
import pytest
from decoding_in_one.codes import SurfaceCode

def test_surface_code_initialization():
    code = SurfaceCode(distance=5)
    assert code.get_n_physical() == 25
    assert code.get_n_logical() == 1
    assert len(code.get_check_qubits("X")) == 12
    assert len(code.get_check_qubits("Z")) == 12

def test_surface_code_distance_must_be_odd():
    with pytest.raises(ValueError):
        SurfaceCode(distance=4)

def test_surface_code_rotation_validation():
    with pytest.raises(ValueError):
        SurfaceCode(distance=5, rotation="bad")

def test_stabilizer_supports_have_valid_weights():
    code = SurfaceCode(distance=5, rotation="XV")
    for support in code.get_stabilizer_supports("X").values():
        assert len(support) in (2, 4)
    for support in code.get_stabilizer_supports("Z").values():
        assert len(support) in (2, 4)

def test_qubit_topology_complete():
    code = SurfaceCode(distance=3)
    topology = code.get_qubit_topology()
    assert len(topology) == 9
    assert topology[0] == (0, 0)
    assert topology[8] == (2, 2)
