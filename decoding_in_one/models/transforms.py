# decoding_in_one/models/transforms.py
import numpy as np
import torch
from typing import Literal

def dets_to_conv3d_input(dets: np.ndarray, rounds: int, distance: int) -> np.ndarray:
    """将检测器数据转换为 Conv3D 输入格式 (B,4,T,D,D)

    迁移自 ising_train_model.py 的 _dets_to_trainx

    Args:
        dets: 检测器数组 (batch, n_detectors)
        rounds: 电路轮数
        distance: 码距

    Returns:
        np.ndarray: shape (batch, 4, rounds, distance, distance)
    """
    bsz, n_det = dets.shape
    t = rounds
    d = distance
    target = t * d * d
    padded = np.zeros((bsz, target), dtype=np.float32)
    use = min(n_det, target)
    padded[:, :use] = dets[:, :use]
    ch0 = padded.reshape(bsz, t, d, d)
    ch1 = ch0.copy()
    ch2 = np.ones_like(ch0, dtype=np.float32)
    ch3 = np.ones_like(ch0, dtype=np.float32)
    return np.stack([ch0, ch1, ch2, ch3], axis=1)


def obs_to_conv3d_target(obs: np.ndarray, rounds: int, distance: int) -> np.ndarray:
    """将观测量广播为 Conv3D 目标格式 (B,4,T,D,D)

    迁移自 ising_train_model.py 的 _obs_to_target4

    注意：这是 dense 训练目标，不是最终的 observable 向量

    Args:
        obs: 观测量数组 (batch, n_observables)
        rounds: 电路轮数
        distance: 码距

    Returns:
        np.ndarray: shape (batch, 4, rounds, distance, distance)
    """
    bsz, n_obs = obs.shape
    base = np.zeros((bsz, 4, rounds, distance, distance), dtype=np.float32)
    if n_obs == 1:
        for i in range(4):
            base[:, i, :, :, :] = obs[:, 0][:, None, None, None]
        return base
    for i in range(min(4, n_obs)):
        base[:, i, :, :, :] = obs[:, i][:, None, None, None]
    return base


def reduce_conv3d_output(
    conv_output: np.ndarray | torch.Tensor,
    method: Literal["mean", "max", "vote"] = "mean"
) -> np.ndarray:
    """将 dense logical field 聚合为 logical observable 向量

    Args:
        conv_output: Conv3D 模型输出 (B,4,T,D,D)
        method: 聚合方法 ("mean", "max", "vote")

    Returns:
        np.ndarray: logical observable 预测 (B, n_observables)
    """
    if torch.is_tensor(conv_output):
        conv_output = conv_output.cpu().numpy()

    bsz = conv_output.shape[0]

    if method == "mean":
        # 对空间和时间维度求平均
        reduced = conv_output.mean(axis=(2, 3, 4))  # (B, 4)
        return reduced
    elif method == "max":
        # 对空间和时间维度取最大值
        reduced = conv_output.max(axis=(2, 3, 4))  # (B, 4)
        return reduced
    elif method == "vote":
        # 投票：多数决定
        spatial_sum = conv_output.sum(axis=(2, 3, 4))  # (B, 4)
        total_elements = conv_output.shape[2] * conv_output.shape[3] * conv_output.shape[4]
        return (spatial_sum > total_elements / 2).astype(np.float32)
    else:
        raise ValueError(f"Unknown method: {method}")
