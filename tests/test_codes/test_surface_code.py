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


@pytest.mark.parametrize(
    ("rotation", "expected_x", "expected_z"),
    [
        (
            "XV",
            [[1, 4, 0, 3], [-1, -1, 2, 5], [3, 6, -1, -1], [5, 8, 4, 7]],
            [[-1, -1, 1, 0], [4, 3, 7, 6], [2, 1, 5, 4], [8, 7, -1, -1]],
        ),
        (
            "XH",
            [[-1, -1, 2, 1], [1, 0, 4, 3], [5, 4, 8, 7], [7, 6, -1, -1]],
            [[0, 3, -1, -1], [4, 7, 3, 6], [2, 5, 1, 4], [-1, -1, 5, 8]],
        ),
        (
            "ZV",
            [[-1, -1, 1, 0], [2, 1, 5, 4], [4, 3, 7, 6], [8, 7, -1, -1]],
            [[3, 6, -1, -1], [1, 4, 0, 3], [5, 8, 4, 7], [-1, -1, 2, 5]],
        ),
        (
            "ZH",
            [[0, 3, -1, -1], [2, 5, 1, 4], [4, 7, 3, 6], [-1, -1, 5, 8]],
            [[1, 0, 4, 3], [7, 6, -1, -1], [-1, -1, 2, 1], [5, 4, 8, 7]],
        ),
    ],
)
def test_surface_code_matches_expected_plaquette_order(rotation, expected_x, expected_z):
    code = SurfaceCode(distance=3, rotation=rotation)
    actual_x = [code.xcheck_qubits_dict[q]["plaquette"]["qubit_id"] for q in code.get_check_qubits("X")]
    actual_z = [code.zcheck_qubits_dict[q]["plaquette"]["qubit_id"] for q in code.get_check_qubits("Z")]
    assert actual_x == expected_x
    assert actual_z == expected_z
