# tests/test_models/test_transforms.py
import numpy as np
import pytest
from decoding_in_one.models.surface_code.transforms import dets_to_conv3d_input, obs_to_conv3d_target, reduce_conv3d_output

def test_dets_to_conv3d_input_basic():
    """基本功能测试"""
    # 输入：2 个样本，每个有 75 个检测器 (distance=5, rounds=3 -> 3*5*5=75)
    dets = np.random.randint(0, 2, size=(2, 75)).astype(np.float32)
    result = dets_to_conv3d_input(dets, rounds=3, distance=5)

    # 输出应该是 (B, 4, T, D, D) = (2, 4, 3, 5, 5)
    assert result.shape == (2, 4, 3, 5, 5)

def test_dets_to_conv3d_input_padding():
    """当检测器数不足时应该填充零"""
    # 只有 50 个检测器，但目标需要 75
    dets = np.ones((2, 50), dtype=np.float32)  # 使用全 1 确保有数据
    result = dets_to_conv3d_input(dets, rounds=3, distance=5)

    assert result.shape == (2, 4, 3, 5, 5)
    # 应该有非零值（来自原始数据）
    assert result[:, :1].sum() > 0

def test_dets_to_conv3d_input_channels():
    """应该创建 4 个通道"""
    dets = np.random.randint(0, 2, size=(1, 12)).astype(np.float32)  # distance=2, rounds=3
    result = dets_to_conv3d_input(dets, rounds=3, distance=2)

    assert result.shape[1] == 4

def test_obs_to_conv3d_target_single_observable():
    """单个观测量应该广播到所有 4 个通道"""
    obs = np.array([[1], [0], [1]], dtype=np.float32)
    result = obs_to_conv3d_target(obs, rounds=3, distance=5)

    assert result.shape == (3, 4, 3, 5, 5)
    # 所有 4 个通道应该相同
    np.testing.assert_array_equal(result[:, 0, 0, 0, 0], result[:, 1, 0, 0, 0])

def test_obs_to_conv3d_target_multiple_observables():
    """多个观测量应该分配到不同通道"""
    obs = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    result = obs_to_conv3d_target(obs, rounds=3, distance=5)

    assert result.shape == (3, 4, 3, 5, 5)
    # 检查前两个通道不同
    assert not np.array_equal(result[:, 0, 0, 0, 0], result[:, 1, 0, 0, 0])

def test_reduce_conv3d_output_mean():
    """mean 聚合应该正确计算"""
    # 创建简单的测试数据
    conv_output = np.ones((2, 4, 3, 5, 5), dtype=np.float32)
    result = reduce_conv3d_output(conv_output, method="mean")

    # 结果应该是 (B, n_observables)
    # 对于表面码 X 基，n_observables = 1
    assert result.ndim == 2
    assert result.shape[0] == 2

def test_reduce_conv3d_output_with_tensor():
    """应该支持 torch.Tensor 输入"""
    import torch

    conv_output = torch.ones(2, 4, 3, 5, 5)
    result = reduce_conv3d_output(conv_output, method="mean")

    assert result.ndim == 2
    assert isinstance(result, np.ndarray)
