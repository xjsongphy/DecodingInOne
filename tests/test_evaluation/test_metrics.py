# tests/test_evaluation/test_metrics.py
import pytest
import torch
from decoding_in_one.evaluation.metrics import MetricsCalculator

def test_logical_error_rate():
    predictions = torch.tensor([[0, 0], [1, 0], [0, 1]])
    actual = torch.tensor([[0, 0], [0, 0], [0, 1]])

    ler = MetricsCalculator.logical_error_rate(predictions, actual)
    assert abs(ler - 1.0 / 3) < 1e-6  # 浮点精度容差
