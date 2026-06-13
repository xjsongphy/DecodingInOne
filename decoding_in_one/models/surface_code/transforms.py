# decoding_in_one/models/surface_code/transforms.py
"""
推理阶段的数据转换函数（忠实迁移自 Ising-Decoding datapipe_stim.py）

这些函数用于将 Stim 采样结果转换为 Conv3D 模型所需的输入格式，
以及将模型输出聚合为逻辑可观测量的预测。

关键逻辑：
- dets_to_conv3d_input: 原始测量 → (B, 4, T, D, D) trainX
  1. 拆分 X/Z 稳定子测量
  2. 时间方向 XOR 差分
  3. scatter-permutation 映射到 D×D 网格
  4. 应用 presence mask 和基相关边界屏蔽

- reduce_conv3d_output: (B, 4, T, D, D) → 逻辑 observable 预测
  使用 parity-check 矩阵做正确的 XOR 约化
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Literal, Optional

from decoding_in_one.codes.surface_code.data_mapping import (
    compute_stabX_to_data_index_map,
    compute_stabZ_to_data_index_map,
    normalized_weight_mapping_Xstab_memory,
    normalized_weight_mapping_Zstab_memory,
)


def dets_to_conv3d_input(
    dets: np.ndarray,
    rounds: int,
    distance: int,
    *,
    basis: str = "X",
    code_rotation: str = "XV",
) -> np.ndarray:
    """将原始测量数据转换为 Conv3D 输入格式 (B, 4, T, D, D)

    迁移自 Ising-Decoding 的 datapipe_stim.py._precompute_transformations

    输入 dets 是 Stim 采样的原始 ancilla 测量结果，形状为
    (batch, T, D²-1) 或 (batch, T*(D²-1))，其中前 half 列是 X 型
    稳定子测量，后 half 列是 Z 型稳定子测量。

    Args:
        dets: 原始测量数组。
              形状 (batch, T, D²-1) 或 (batch, T*(D²-1)) 或 (batch, n_detectors)
        rounds: 电路轮数 T
        distance: 码距 D
        basis: 测量基 ('X' 或 'Z')，影响边界屏蔽
        code_rotation: 码旋转方向 ('XV', 'XH', 'ZV', 'ZH')

    Returns:
        np.ndarray: shape (batch, 4, T, D, D)
                    通道 0: x_syn_diff (X syndrome 差分)
                    通道 1: z_syn_diff (Z syndrome 差分)
                    通道 2: x_pres (X presence mask)
                    通道 3: z_pres (Z presence mask)
    """
    basis = basis.upper()
    code_rotation = code_rotation.upper()
    D = distance
    T = rounds
    half = (D * D - 1) // 2  # 每种稳定子的数量

    # --- 处理不同输入形状 ---
    if dets.ndim == 2:
        # (batch, T * (D²-1)) → (batch, T, D²-1)
        if dets.shape[1] == T * (D * D - 1):
            dets = dets.reshape(dets.shape[0], T, D * D - 1)
        elif dets.shape[1] == T * half * 2:
            # 可能是交错格式
            dets = dets.reshape(dets.shape[0], T, half * 2)
        else:
            # 尝试直接 reshape
            dets = dets.reshape(dets.shape[0], T, -1)
    bsz = dets.shape[0]

    # --- 分离 X/Z 稳定子测量 ---
    if dets.shape[2] == D * D - 1:
        # 标准格式：前 half 是 X，后 half 是 Z
        x_raw = dets[:, :, :half]   # (B, T, half)
        z_raw = dets[:, :, half:]   # (B, T, half)
    elif dets.shape[2] == half * 2:
        x_raw = dets[:, :, :half]
        z_raw = dets[:, :, half:]
    else:
        # 回退：尝试均分
        mid = dets.shape[2] // 2
        x_raw = dets[:, :, :mid]
        z_raw = dets[:, :, mid:]
        half = mid

    # 转为 uint8 用于 XOR 运算
    x_raw = x_raw.astype(np.uint8)
    z_raw = z_raw.astype(np.uint8)

    # --- 时间方向 XOR 差分 ---
    # 加零帧做 diff: x_aug = [0, t0, t1, ..., t_{T-1}]，然后 diff = aug[:,1:] ^ aug[:,:-1]
    x_raw_t = x_raw.transpose(0, 2, 1)  # (B, half, T)
    z_raw_t = z_raw.transpose(0, 2, 1)  # (B, half, T)

    zero_frame = np.zeros((bsz, half, 1), dtype=np.uint8)
    x_aug = np.concatenate([zero_frame, x_raw_t], axis=2)  # (B, half, T+1)
    z_aug = np.concatenate([zero_frame, z_raw_t], axis=2)  # (B, half, T+1)

    x_syn_diff = (x_aug[:, :, 1:] ^ x_aug[:, :, :-1]).astype(np.float32)  # (B, half, T)
    z_syn_diff = (z_aug[:, :, 1:] ^ z_aug[:, :, :-1]).astype(np.float32)  # (B, half, T)

    # --- 基相关边界屏蔽 ---
    if basis == "X":
        z_syn_diff[:, :, 0] = 0
        z_syn_diff[:, :, -1] = 0
    else:  # "Z"
        x_syn_diff[:, :, 0] = 0
        x_syn_diff[:, :, -1] = 0

    # --- scatter-permutation 映射到 D×D 网格 ---
    idx_map_x = compute_stabX_to_data_index_map(D, code_rotation)
    idx_map_z = compute_stabZ_to_data_index_map(D, code_rotation)
    n_stab_x = len(idx_map_x)
    n_stab_z = len(idx_map_z)

    x_syn_stab = x_syn_diff[:, :n_stab_x, :]  # (B, n_stab_x, T)
    z_syn_stab = z_syn_diff[:, :n_stab_z, :]  # (B, n_stab_z, T)

    # scatter 到 (B, D*D, T)
    x_grid = np.zeros((bsz, D * D, T), dtype=np.float32)
    z_grid = np.zeros((bsz, D * D, T), dtype=np.float32)

    idx_x = np.array(idx_map_x, dtype=np.intp)
    idx_z = np.array(idx_map_z, dtype=np.intp)

    x_grid[:, idx_x, :] = x_syn_stab
    z_grid[:, idx_z, :] = z_syn_stab

    # reshape 到 (B, T, D, D) 并 permute
    x_type = x_grid.reshape(bsz, D, D, T).transpose(0, 3, 1, 2)  # (B, T, D, D)
    z_type = z_grid.reshape(bsz, D, D, T).transpose(0, 3, 1, 2)  # (B, T, D, D)

    # --- presence mask ---
    w_mapX = normalized_weight_mapping_Xstab_memory(D, code_rotation).reshape(D, D).numpy()
    w_mapZ = normalized_weight_mapping_Zstab_memory(D, code_rotation).reshape(D, D).numpy()

    x_pres = np.broadcast_to(w_mapX[None, None, :, :], (bsz, T, D, D)).copy().astype(np.float32)
    z_pres = np.broadcast_to(w_mapZ[None, None, :, :], (bsz, T, D, D)).copy().astype(np.float32)

    # 基相关 presence mask
    if basis == "X":
        z_pres[:, 0, :, :] = 0
        z_pres[:, -1, :, :] = 0
    else:  # "Z"
        x_pres[:, 0, :, :] = 0
        x_pres[:, -1, :, :] = 0

    # --- 堆叠为 (B, 4, T, D, D) ---
    trainX = np.stack([
        x_type,   # channel 0: x_syn_diff
        z_type,   # channel 1: z_syn_diff
        x_pres,   # channel 2: x_pres
        z_pres,   # channel 3: z_pres
    ], axis=1)

    return trainX


def obs_to_conv3d_target(
    obs: np.ndarray,
    rounds: int,
    distance: int,
) -> np.ndarray:
    """将逻辑 observable 转换为 Conv3D 目标格式 (B, 4, T, D, D)

    注意：此函数仅用于简单的 loss 监控场景。
    在 Ising-Decoding 的训练管线中，dense target (trainY) 是由
    CircuitDataGenerator._format_for_model() 生成的，包含 per-qubit 的
    错误预测和稳定子校正，而非简单的 logical observable 广播。

    对于正式的训练和评估，请使用 CircuitDataGenerator 生成的 trainY。

    Args:
        obs: 逻辑 observable 数组 (batch, n_observables)
        rounds: 电路轮数
        distance: 码距

    Returns:
        np.ndarray: shape (batch, 4, rounds, distance, distance)
    """
    bsz, n_obs = obs.shape
    base = np.zeros((bsz, 4, rounds, distance, distance), dtype=np.float32)
    # 只填充 channel 0 (z_err 通道) 作为 logical Z observable
    # 其他通道保持为 0
    if n_obs >= 1:
        base[:, 0, :, :, :] = obs[:, 0][:, None, None, None]
    return base


def reduce_conv3d_output(
    conv_output: np.ndarray | torch.Tensor,
    *,
    distance: int,
    code_rotation: str = "XV",
    basis: str = "X",
    method: Literal["parity", "mean", "max", "vote"] = "parity",
) -> np.ndarray:
    """将 dense logical field 聚合为 logical observable 向量

    迁移自 Ising-Decoding 的 post_matrices 约化逻辑。

    "parity" 方法（推荐）:
      对 channel 0 (z_err) 和 channel 1 (x_err) 在数据比特位置上做
      XOR 约化。使用 logical operator 矩阵 (lx/lz) 将 per-qubit 预测
      映射到 logical observable。

    "mean"/"max"/"vote" 方法:
      简单的空间-时间聚合，不保证正确性，仅用于快速测试。

    Args:
        conv_output: Conv3D 模型输出 (B, 4, T, D, D)
        distance: 码距
        code_rotation: 码旋转方向
        basis: 测量基 ('X' 或 'Z')
        method: 聚合方法 ("parity" 推荐, "mean", "max", "vote")

    Returns:
        np.ndarray: logical observable 预测 (B,) 或 (B, n_obs)
    """
    if torch.is_tensor(conv_output):
        conv_output = conv_output.cpu().numpy()

    bsz = conv_output.shape[0]
    D = distance

    if method == "parity":
        from decoding_in_one.codes.surface_code import SurfaceCode

        code = SurfaceCode(D, rotation=code_rotation)

        # conv_output: (B, 4, T, D, D)
        # channel 0: z_err predictions, channel 1: x_err predictions
        # 使用最后一轮的预测做 logical observable 判定
        z_err = conv_output[:, 0, -1, :, :]  # (B, D, D)
        x_err = conv_output[:, 1, -1, :, :]  # (B, D, D)

        # 二值化 (>0.5 → 1)
        z_err_bin = (z_err > 0.5).astype(np.float32)
        x_err_bin = (x_err > 0.5).astype(np.float32)

        # 用 logical operator 做 XOR 约化
        # lx: (1, D²) → reshape 为 (D, D)
        # lz: (1, D²) → reshape 为 (D, D)
        lx = code.lx.flatten().reshape(D, D).astype(np.float32)
        lz = code.lz.flatten().reshape(D, D).astype(np.float32)

        # Logical X = sum(x_err * lx) mod 2
        # Logical Z = sum(z_err * lz) mod 2
        if basis == "X":
            # X-basis: logical observable is the X logical operator
            logical_pred = np.sum(x_err_bin * lx[None, :, :], axis=(1, 2)) % 2  # (B,)
        else:
            # Z-basis: logical observable is the Z logical operator
            logical_pred = np.sum(z_err_bin * lz[None, :, :], axis=(1, 2)) % 2  # (B,)

        return logical_pred

    elif method == "mean":
        reduced = conv_output.mean(axis=(2, 3, 4))  # (B, 4)
        return reduced
    elif method == "max":
        reduced = conv_output.max(axis=(2, 3, 4))  # (B, 4)
        return reduced
    elif method == "vote":
        spatial_sum = conv_output.sum(axis=(2, 3, 4))  # (B, 4)
        total_elements = conv_output.shape[2] * conv_output.shape[3] * conv_output.shape[4]
        return (spatial_sum > total_elements / 2).astype(np.float32)
    else:
        raise ValueError(f"Unknown method: {method}")
