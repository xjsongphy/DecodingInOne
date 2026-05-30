# tests/test_decoders/test_base.py
import pytest
import torch
from decoding_in_one.decoders.base import Decoder

class MockDecoder(Decoder):
    def decode(self, syndrome, observables=None):
        return torch.zeros(syndrome.shape[0], 10)

    def get_name(self):
        return "Mock"

def test_decoder_interface():
    decoder = MockDecoder()
    syndrome = torch.zeros((10, 100))
    result = decoder.decode(syndrome)
    assert result.shape == (10, 10)
    assert decoder.get_name() == "Mock"
