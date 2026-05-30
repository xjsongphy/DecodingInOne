# tests/test_circuits/test_memory_circuit.py
import pytest
from decoding_in_one.codes import SurfaceCode
from decoding_in_one.circuits import MemoryCircuit

def test_build_basic_circuit():
    code = SurfaceCode(distance=3)
    builder = MemoryCircuit(code)
    circuit = builder.build_memory_circuit(code, n_rounds=3, measurement_basis='X')

    assert 'REPEAT' in circuit
    assert 'CNOT' in circuit or 'CX' in circuit
    assert 'M' in circuit
