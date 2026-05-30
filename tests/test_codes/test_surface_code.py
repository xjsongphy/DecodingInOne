# tests/test_codes/test_surface_code.py
import pytest
from decoding_in_one.codes import SurfaceCode

def test_surface_code_initialization():
    code = SurfaceCode(distance=5)
    assert code.get_n_physical() == 25
    assert code.get_n_logical() == 1

def test_surface_code_distance_must_be_odd():
    with pytest.raises(ValueError):
        SurfaceCode(distance=4)
