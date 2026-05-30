# tests/test_models/test_conv3d.py
import pytest
import torch
import numpy as np
from decoding_in_one.models.conv3d import Conv3DNeuralDecoder
from decoding_in_one.models.config import Conv3DModelConfig

def test_conv3d_decoder_initialization():
    """Conv3DNeuralDecoder 应该正确初始化"""
    config = Conv3DModelConfig()
    model = Conv3DNeuralDecoder(config)

    assert model.input_channels == 4
    assert model.out_channels == 4

def test_conv3d_decoder_forward():
    """Conv3DNeuralDecoder 应该正确执行前向传播"""
    config = Conv3DModelConfig(num_filters=[16, 16, 4], kernel_sizes=[3, 3, 3])
    model = Conv3DNeuralDecoder(config)

    # 输入 (B=2, C=4, T=3, D=5, D=5)
    x = torch.randn(2, 4, 3, 5, 5)
    output = model(x)

    # 输出形状应该匹配输入的空间维度
    assert output.shape == (2, 4, 3, 5, 5)

def test_conv3d_decoder_get_input_channels():
    """get_input_channels 应该返回正确的通道数"""
    config = Conv3DModelConfig(input_channels=8)
    model = Conv3DNeuralDecoder(config)

    assert model.get_input_channels() == 8

def test_conv3d_decoder_expected_input_rank():
    """expected_input_rank 应该返回 5"""
    model = Conv3DNeuralDecoder(Conv3DModelConfig())

    assert model.expected_input_rank() == 5

def test_conv3d_decoder_output_shape():
    """output_shape 应该返回正确格式"""
    config = Conv3DModelConfig(out_channels=4)
    model = Conv3DNeuralDecoder(config)

    shape = model.output_shape()
    assert shape == (4, None, None, None)

def test_conv3d_activation_options():
    """应该支持不同的激活函数"""
    for activation in ["relu", "gelu", "leakyrelu"]:
        config = Conv3DModelConfig(activation=activation)
        model = Conv3DNeuralDecoder(config)

        x = torch.randn(1, 4, 2, 3, 3)
        output = model(x)
        assert output is not None
