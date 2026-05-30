# tests/test_utils/test_types.py
import pytest
import torch
from decoding_in_one.utils.types import DecodingBatch

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
