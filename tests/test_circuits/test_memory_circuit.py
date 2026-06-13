# tests/test_circuits/test_memory_circuit.py
import pytest
from decoding_in_one.codes import SurfaceCode
from decoding_in_one.circuits import MemoryCircuit
from decoding_in_one.utils.types import CircuitSpec

def test_build_basic_circuit():
    code = SurfaceCode(distance=3)
    builder = MemoryCircuit(code=code, n_rounds=3, basis="X")
    circuit = builder.build_memory_circuit(code, n_rounds=3, measurement_basis='X')

    assert 'REPEAT' in circuit
    assert 'CNOT' in circuit or 'CX' in circuit
    assert 'M' in circuit

def test_build_memory_artifact():
    code = SurfaceCode(distance=3)
    builder = MemoryCircuit(code=code, n_rounds=2, basis="Z")
    artifact = builder.build_memory_artifact(code, CircuitSpec(n_rounds=2, measurement_basis='Z'))

    assert artifact.code.distance == 3
    assert artifact.spec.n_rounds == 2
    assert 'MX' in artifact.stim_circuit or 'MZ' in artifact.stim_circuit
