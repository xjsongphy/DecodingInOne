# tests/test_decoders/test_neural.py
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from decoding_in_one.decoders.neural import NeuralDecoder
from decoding_in_one.models import SurfaceCodeConv3DDecoder, Conv3DModelConfig

def test_neural_decoder_initialization():
    """NeuralDecoder 应该正确初始化"""
    model = SurfaceCodeConv3DDecoder(Conv3DModelConfig(num_filters=[16, 16, 4], kernel_sizes=[3, 3, 3]))

    # 创建临时 checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name
        torch.save({"model_state_dict": model.state_dict()}, f.name)

    try:
        decoder = NeuralDecoder(
            model=model,
            checkpoint_path=checkpoint_path,
            rounds=3,
            distance=5
        )

        assert decoder.rounds == 3
        assert decoder.distance == 5
        assert decoder.threshold == 0.5
    finally:
        Path(checkpoint_path).unlink()

def test_neural_decoder_decode():
    """NeuralDecoder 应该正确解码"""
    model = SurfaceCodeConv3DDecoder(Conv3DModelConfig(num_filters=[16, 16, 4], kernel_sizes=[3, 3, 3]))

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name
        torch.save({"model_state_dict": model.state_dict()}, f.name)

    try:
        decoder = NeuralDecoder(
            model=model,
            checkpoint_path=checkpoint_path,
            rounds=3,
            distance=5
        )

        # 创建输入：2 个样本，75 个检测器
        syndrome = np.random.randint(0, 2, size=(2, 75)).astype(np.float32)

        correction = decoder.decode(syndrome)

        # 检查输出
        assert correction.predictions.shape[0] == 2  # batch size
        # 输出应该是 dense field (B, 4, T, D, D) 或 (B, n_observables)
    finally:
        Path(checkpoint_path).unlink()

def test_neural_decoder_with_tensor_input():
    """NeuralDecoder 应该支持 tensor 输入"""
    model = SurfaceCodeConv3DDecoder(Conv3DModelConfig(num_filters=[16, 16, 4], kernel_sizes=[3, 3, 3]))

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name
        torch.save({"model_state_dict": model.state_dict()}, f.name)

    try:
        decoder = NeuralDecoder(
            model=model,
            checkpoint_path=checkpoint_path,
            rounds=3,
            distance=5
        )

        # tensor 输入
        syndrome = torch.randint(0, 2, size=(2, 75), dtype=torch.float32)

        correction = decoder.decode(syndrome)
        assert correction.predictions is not None
    finally:
        Path(checkpoint_path).unlink()

def test_neural_decoder_reduce_output():
    """NeuralDecoder 应该支持输出聚合"""
    model = SurfaceCodeConv3DDecoder(Conv3DModelConfig(num_filters=[16, 16, 4], kernel_sizes=[3, 3, 3]))

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name
        torch.save({"model_state_dict": model.state_dict()}, f.name)

    try:
        decoder = NeuralDecoder(
            model=model,
            checkpoint_path=checkpoint_path,
            rounds=3,
            distance=5,
            reduce_output=True,
            reduce_method="mean"
        )

        syndrome = np.random.randint(0, 2, size=(2, 75)).astype(np.float32)
        correction = decoder.decode(syndrome)

        # 聚合后输出应该是 (B, n_observables)
        assert correction.predictions.ndim == 2
    finally:
        Path(checkpoint_path).unlink()
