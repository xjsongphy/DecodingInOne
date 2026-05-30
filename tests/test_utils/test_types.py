# tests/test_utils/test_types.py
import pytest
import torch
from decoding_in_one.utils.types import DecodingBatch, CodeSpec, CircuitSpec, CircuitArtifact

def test_decoding_batch_creation():
    detectors = torch.zeros((10, 100))
    observables = torch.zeros((10, 2))

    batch = DecodingBatch(
        detectors=detectors,
        observables=observables,
        syndrome_grid=None,
        metadata={'code': 'surface', 'distance': 7}
    )

    assert batch.detectors.shape == (10, 100)
    assert batch.observables.shape == (10, 2)
    assert batch.metadata['distance'] == 7

def test_circuit_artifact_creation():
    artifact = CircuitArtifact(
        stim_circuit="M 0\n",
        code=CodeSpec(code_family="SurfaceCode", distance=5, n_physical=25, n_logical=1),
        spec=CircuitSpec(n_rounds=5, measurement_basis="X"),
    )
    assert artifact.code.distance == 5
    assert artifact.spec.measurement_basis == "X"
