# decoding_in_one/surface_code/data_mapping.py
"""
Surface code stabilizer-to-data qubit mappings and weight functions.

源码参考: D:\\Develop\\Ising-Decoding\\code\\qec\\surface_code\\data_mapping.py

将 X/Z 稳定子映射到数据比特网格，支持四种旋转方向: XV, XH, ZV, ZH。
"""

from __future__ import annotations

import torch


# ===================================================================
# 内部辅助：从校验矩阵计算映射（通用旋转方向）
# ===================================================================


def _compute_stab_to_data_from_parity_X_boundary_aware(
    parity_matrix: torch.Tensor, distance: int
) -> torch.Tensor:
    """
    X 稳定子到数据比特映射（边界感知）。

    边界约定:
    - 水平对 (上/下边界) → 选左 (最小列)
    - 垂直对 (左/右边界) → 选上 (最小行)
    - 体内 (weight-4) → 选左上
    """
    num_stabs = parity_matrix.shape[0]
    stab_to_data = torch.empty(num_stabs, dtype=torch.int32)

    for stab_idx in range(num_stabs):
        support = torch.nonzero(parity_matrix[stab_idx], as_tuple=True)[0].tolist()
        positions = [(idx // distance, idx % distance, idx) for idx in support]

        if len(support) == 2:
            rows = [p[0] for p in positions]
            if rows[0] == rows[1]:
                positions.sort(key=lambda x: x[1])
            else:
                positions.sort(key=lambda x: x[0])
        else:
            positions.sort(key=lambda x: (x[0], x[1]))

        stab_to_data[stab_idx] = positions[0][2]

    return stab_to_data


def _compute_stab_to_data_from_parity_Z_boundary_aware(
    parity_matrix: torch.Tensor, distance: int
) -> torch.Tensor:
    """
    Z 稳定子到数据比特映射（边界感知）。

    边界约定:
    - 垂直对 (左/右边界) → 选上 (最小行)
    - 水平对 (上/下边界) → 选右 (最大列)
    - 体内 (weight-4) → 选右上
    """
    num_stabs = parity_matrix.shape[0]
    stab_to_data = torch.empty(num_stabs, dtype=torch.int32)

    for stab_idx in range(num_stabs):
        support = torch.nonzero(parity_matrix[stab_idx], as_tuple=True)[0].tolist()
        positions = [(idx // distance, idx % distance, idx) for idx in support]

        if len(support) == 2:
            cols = [p[1] for p in positions]
            if cols[0] == cols[1]:
                positions.sort(key=lambda x: x[0])
            else:
                positions.sort(key=lambda x: -x[1])
        else:
            positions.sort(key=lambda x: (x[0], -x[1]))

        stab_to_data[stab_idx] = positions[0][2]

    return stab_to_data


def _compute_normalized_weight_from_parity(
    parity_matrix: torch.Tensor, stab_to_data: torch.Tensor, distance: int
) -> torch.Tensor:
    """从校验矩阵计算归一化权重。边界 (weight-2) 得 0.5, 体内 (weight-4) 得 1.0。"""
    num_stabs = parity_matrix.shape[0]
    out = torch.zeros(distance * distance)

    for stab_idx in range(num_stabs):
        support_size = parity_matrix[stab_idx].sum().item()
        data_idx = int(stab_to_data[stab_idx])
        out[data_idx] = 0.5 if support_size == 2 else 1.0

    return out


# ===================================================================
# XV 方向优化版映射
# ===================================================================


def _compute_stabX_to_data_XV(distance):
    cols = (distance + 1) // 2
    rows = distance - 1
    total_stabs = cols * rows
    stab_to_data_index_map = torch.empty(total_stabs, dtype=torch.int32)
    data_qubit_index = 0
    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            stab_to_data_index_map[idx] = data_qubit_index
            idx += 1
            if rr % 2 == 0:
                if cc == cols - 1:
                    data_qubit_index += 1
                else:
                    data_qubit_index += 2
            else:
                if cc == 0:
                    data_qubit_index += 1
                else:
                    data_qubit_index += 2
    return stab_to_data_index_map


def _compute_stabZ_to_data_XV(distance):
    cols = distance - 1
    rows = (distance + 1) // 2
    total_stabs = cols * rows
    stab_to_data_index_map = torch.empty(total_stabs, dtype=torch.int32)
    data_qubit_index = 1
    stab_idx = 0
    for cc in range(cols):
        data_qubit_top = data_qubit_index
        for rr in range(rows):
            stab_to_data_index_map[stab_idx] = data_qubit_index
            stab_idx += 1
            if cc % 2 == 0:
                if rr == 0:
                    data_qubit_index += distance
                elif rr == rows - 1:
                    data_qubit_index = data_qubit_top + 1
                else:
                    data_qubit_index += 2 * distance
            else:
                if rr == rows - 1:
                    data_qubit_index = data_qubit_top + 1
                else:
                    data_qubit_index += 2 * distance
    return stab_to_data_index_map


def _compute_stabZ_to_data_XH(distance):
    cols = (distance + 1) // 2
    rows = distance - 1
    total_stabs = cols * rows
    stab_to_data_index_map = torch.empty(total_stabs, dtype=torch.int32)
    data_qubit_index = 1
    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            stab_to_data_index_map[idx] = data_qubit_index
            idx += 1
            if rr % 2 == 0:
                if cc == 0:
                    data_qubit_index += 1
                else:
                    data_qubit_index += 2
            else:
                if cc == cols - 1:
                    data_qubit_index += 1
                else:
                    data_qubit_index += 2
    return stab_to_data_index_map


# ===================================================================
# 公共 API: compute_stabX/Z_to_data_index_map
# ===================================================================


def _get_surface_code_for_rotation(distance, rotation):
    """根据 rotation 获取 SurfaceCode 的 hx/hz 矩阵。"""
    from decoding_in_one.codes.surface_code import SurfaceCode

    first_bulk = rotation[0]
    rotated = rotation[1]
    code = SurfaceCode(distance, rotation=rotation)
    # 直接用 code 的内部数据构建 hx / hz
    n_data = distance ** 2
    xcheck = code.get_check_qubits('X')
    zcheck = code.get_check_qubits('Z')
    x_supps = code.get_stabilizer_supports('X')
    z_supps = code.get_stabilizer_supports('Z')

    hx = torch.zeros(len(xcheck), n_data, dtype=torch.int32)
    for i, q in enumerate(xcheck):
        for dq in x_supps[q]:
            hx[i, dq] = 1

    hz = torch.zeros(len(zcheck), n_data, dtype=torch.int32)
    for i, q in enumerate(zcheck):
        for dq in z_supps[q]:
            hz[i, dq] = 1

    return hx, hz


def compute_stabX_to_data_index_map(distance, rotation='XV'):
    """X 稳定子索引 → 数据比特索引映射。"""
    rotation = rotation.upper()
    if rotation == 'XV':
        return _compute_stabX_to_data_XV(distance)
    elif rotation in ('XH', 'ZV', 'ZH'):
        hx, _ = _get_surface_code_for_rotation(distance, rotation)
        return _compute_stab_to_data_from_parity_X_boundary_aware(hx, distance)
    else:
        raise ValueError(f"Invalid rotation '{rotation}'. Must be one of: XV, XH, ZV, ZH")


def compute_stabZ_to_data_index_map(distance, rotation='XV'):
    """Z 稳定子索引 → 数据比特索引映射。"""
    rotation = rotation.upper()
    if rotation == 'XV':
        return _compute_stabZ_to_data_XV(distance)
    elif rotation == 'XH':
        return _compute_stabZ_to_data_XH(distance)
    elif rotation in ('ZV', 'ZH'):
        _, hz = _get_surface_code_for_rotation(distance, rotation)
        return _compute_stab_to_data_from_parity_Z_boundary_aware(hz, distance)
    else:
        raise ValueError(f"Invalid rotation '{rotation}'. Must be one of: XV, XH, ZV, ZH")


# ===================================================================
# 公共 API: 归一化权重映射
# ===================================================================


def _normalized_weight_mapping_Xstab_XV(distance):
    cols = (distance + 1) // 2
    rows = distance - 1
    data_qubit_index = 0
    out = torch.zeros(distance * distance)
    for rr in range(rows):
        for cc in range(cols):
            if rr % 2 == 0:
                if cc == cols - 1:
                    out[data_qubit_index] = 0.5
                    data_qubit_index += 1
                else:
                    out[data_qubit_index] = 1
                    data_qubit_index += 2
            else:
                if cc == 0:
                    out[data_qubit_index] = 0.5
                    data_qubit_index += 1
                else:
                    out[data_qubit_index] = 1
                    data_qubit_index += 2
    return out


def _normalized_weight_mapping_Xstab_XH(distance):
    cols = distance - 1
    rows = (distance + 1) // 2
    data_qubit_index = 0
    out = torch.zeros(distance * distance)
    for cc in range(cols):
        data_qubit_top = data_qubit_index
        for rr in range(rows):
            if cc % 2 == 0:
                if rr == rows - 1:
                    out[data_qubit_index] = 0.5
                    data_qubit_index = data_qubit_top + 1
                else:
                    out[data_qubit_index] = 1
                    data_qubit_index += 2 * distance
            else:
                if rr == 0:
                    out[data_qubit_index] = 0.5
                    data_qubit_index += distance
                elif rr == rows - 1:
                    out[data_qubit_index] = 1
                    data_qubit_index = data_qubit_top + 1
                else:
                    out[data_qubit_index] = 1
                    data_qubit_index += 2 * distance
    return out


def _normalized_weight_mapping_Zstab_XV(distance):
    cols = distance - 1
    rows = (distance + 1) // 2
    data_qubit_index = 1
    out = torch.zeros(distance * distance)
    for cc in range(cols):
        data_qubit_top = data_qubit_index
        for rr in range(rows):
            if cc % 2 == 0:
                if rr == 0:
                    out[data_qubit_index] = 0.5
                    data_qubit_index += distance
                elif rr == rows - 1:
                    out[data_qubit_index] = 1
                    data_qubit_index = data_qubit_top + 1
                else:
                    out[data_qubit_index] = 1
                    data_qubit_index += 2 * distance
            else:
                if rr == rows - 1:
                    out[data_qubit_index] = 0.5
                    data_qubit_index = data_qubit_top + 1
                else:
                    out[data_qubit_index] = 1
                    data_qubit_index += 2 * distance
    return out


def _normalized_weight_mapping_Zstab_XH(distance):
    cols = (distance + 1) // 2
    rows = distance - 1
    data_qubit_index = 1
    out = torch.zeros(distance * distance)
    for rr in range(rows):
        for cc in range(cols):
            if rr % 2 == 0:
                if cc == 0:
                    out[data_qubit_index] = 0.5
                    data_qubit_index += 1
                else:
                    out[data_qubit_index] = 1
                    data_qubit_index += 2
            else:
                if cc == cols - 1:
                    out[data_qubit_index] = 0.5
                    data_qubit_index += 1
                else:
                    out[data_qubit_index] = 1
                    data_qubit_index += 2
    return out


def normalized_weight_mapping_Xstab_memory(distance, rotation='XV'):
    """X 稳定子的归一化权重映射到数据比特。"""
    rotation = rotation.upper()
    if rotation == 'XV':
        return _normalized_weight_mapping_Xstab_XV(distance)
    elif rotation in ('XH', 'ZV', 'ZH'):
        hx, _ = _get_surface_code_for_rotation(distance, rotation)
        stab_to_data = _compute_stab_to_data_from_parity_X_boundary_aware(hx, distance)
        return _compute_normalized_weight_from_parity(hx, stab_to_data, distance)
    else:
        raise ValueError(f"Invalid rotation '{rotation}'. Must be one of: XV, XH, ZV, ZH")


def normalized_weight_mapping_Zstab_memory(distance, rotation='XV'):
    """Z 稳定子的归一化权重映射到数据比特。"""
    rotation = rotation.upper()
    if rotation == 'XV':
        return _normalized_weight_mapping_Zstab_XV(distance)
    elif rotation == 'XH':
        return _normalized_weight_mapping_Zstab_XH(distance)
    elif rotation in ('ZV', 'ZH'):
        _, hz = _get_surface_code_for_rotation(distance, rotation)
        stab_to_data = _compute_stab_to_data_from_parity_Z_boundary_aware(hz, distance)
        return _compute_normalized_weight_from_parity(hz, stab_to_data, distance)
    else:
        raise ValueError(f"Invalid rotation '{rotation}'. Must be one of: XV, XH, ZV, ZH")


# ===================================================================
# 公共 API: 网格重塑
# ===================================================================


def reshape_Xstabilizers_to_grid_vectorized(stab_tensor, distance, rotation='XV'):
    """X 稳定子张量 → D×D 网格。输入 (B, num_stabs, T)，输出 (B, D², T)。"""
    if stab_tensor.ndim == 2:
        stab_tensor = stab_tensor.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, num_stabs, T = stab_tensor.shape
    idx_map = torch.as_tensor(
        compute_stabX_to_data_index_map(distance, rotation),
        dtype=torch.long, device=stab_tensor.device,
    )

    out = torch.zeros(B, distance * distance, T, device=stab_tensor.device, dtype=stab_tensor.dtype)
    out[:, idx_map, :] = stab_tensor
    return out[0] if squeeze_output else out


def reshape_Zstabilizers_to_grid_vectorized(stab_tensor, distance, rotation='XV'):
    """Z 稳定子张量 → D×D 网格。输入 (B, num_stabs, T)，输出 (B, D², T)。"""
    if stab_tensor.ndim == 2:
        stab_tensor = stab_tensor.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, num_stabs, T = stab_tensor.shape
    idx_map = torch.as_tensor(
        compute_stabZ_to_data_index_map(distance, rotation),
        dtype=torch.long, device=stab_tensor.device,
    )

    out = torch.zeros(B, distance * distance, T, device=stab_tensor.device, dtype=stab_tensor.dtype)
    out[:, idx_map, :] = stab_tensor
    return out[0] if squeeze_output else out
