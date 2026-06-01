# decoding_in_one/models/common/graph_builder.py
"""
图构建器：将 syndrome 张量转换为图结构

输入：syndrome 张量 (B, 4, T, D, D)
输出：图结构（节点特征 + 边索引）

图结构设计：
- 节点：检测器节点 (D-1) × T + 数据比特节点 (D²)
- 边：
  - 空间边：相邻数据比特之间
  - 时间边：同一检测器的不同时间步
  - 校正子边：检测器 ↔ 其支撑的数据比特
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class GraphStructure:
    """图结构数据类"""
    x: torch.Tensor              # 节点特征 (N, node_features)
    edge_index: torch.Tensor      # 边索引 (2, E)
    edge_type: Optional[torch.Tensor] = None  # 边类型 (E,)，可选
    node_types: Optional[torch.Tensor] = None  # 节点类型 (N,)，可选
    batch_ptr: Optional[torch.Tensor] = None  # batch 指针 (B+1,)，用于批处理


class SyndromeGraphBuilder:
    """
    Syndrome 张量到图结构的构建器

    将 (B, 4, T, D, D) 的 syndrome 张量转换为图：
    - x_syn: X 型 syndrome (B, T, D-1, D-1)
    - z_syn: Z 型 syndrome (B, T, D-1, D-1)
    - x_pres, z_pres: presence 掩码
    """

    def __init__(
        self,
        include_spatial_edges: bool = True,
        include_temporal_edges: bool = True,
        include_support_edges: bool = True,
        normalize_node_features: bool = True,
    ):
        """
        Args:
            include_spatial_edges: 是否包含空间边（相邻数据比特）
            include_temporal_edges: 是否包含时间边（同检测器不同时间步）
            include_support_edges: 是否包含支撑边（检测器↔数据比特）
            normalize_node_features: 是否归一化节点特征
        """
        self.include_spatial_edges = include_spatial_edges
        self.include_temporal_edges = include_temporal_edges
        self.include_support_edges = include_support_edges
        self.normalize_node_features = normalize_node_features

    def build_graph(
        self,
        syndrome: torch.Tensor,
        distance: int,
        n_rounds: int,
    ) -> GraphStructure:
        """
        构建 GNN 输入的图结构

        Args:
            syndrome: (B, 4, T, D, D) syndrome 张量
                - 通道 0: x_syn (X 型差分)
                - 通道 1: z_syn (Z 型差分)
                - 通道 2: x_pres (X presence)
                - 通道 3: z_pres (Z presence)
            distance: 码距 D
            n_rounds: 时间步数 T

        Returns:
            GraphStructure 包含节点特征和边索引
        """
        batch_size, _, T, D, D = syndrome.shape

        # 1. 创建节点
        node_features, node_types = self._create_nodes(syndrome, distance, n_rounds)

        # 2. 创建边
        edge_index, edge_types = self._create_edges(distance, n_rounds)

        return GraphStructure(
            x=node_features,
            edge_index=edge_index,
            edge_type=edge_types,
            node_types=node_types,
        )

    def _create_nodes(
        self,
        syndrome: torch.Tensor,
        distance: int,
        n_rounds: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建节点特征

        节点类型：
        - type 0: 检测器节点 (X 和 Z) × T 时间步
        - type 1: 数据比特节点

        重要：syndrome 张量是 (B, 4, T, D, D)，但检测器位于 interior (D-1)×(D-1) 位置
        需要从 [1:D, 1:D] 提取检测器测量值
        """
        batch_size = syndrome.size(0)
        device = syndrome.device

        # 分解 syndrome - 每个都是 (B, T, D, D)
        x_syn = syndrome[:, 0]
        z_syn = syndrome[:, 1]
        x_pres = syndrome[:, 2]
        z_pres = syndrome[:, 3]

        # 检测器节点特征：从 interior (1:D, 1:D) 提取
        # 表面码中，检测器位于数据比特网格的 interior 位置
        x_syn_det = x_syn[:, :, 1:distance, 1:distance]  # (B, T, D-1, D-1)
        z_syn_det = z_syn[:, :, 1:distance, 1:distance]

        # 提取对应的 presence 掩码
        x_pres_det = x_pres[:, :, 1:distance, 1:distance]
        z_pres_det = z_pres[:, :, 1:distance, 1:distance]

        # 重塑为 (B, 2, T, (D-1)^2)，然后 transpose 到 (B, T, 2, (D-1)^2)
        n_detectors_per_type = (distance - 1) ** 2

        x_det_flat = x_syn_det.reshape(batch_size, n_rounds, n_detectors_per_type)
        z_det_flat = z_syn_det.reshape(batch_size, n_rounds, n_detectors_per_type)
        x_pres_flat = x_pres_det.reshape(batch_size, n_rounds, n_detectors_per_type)
        z_pres_flat = z_pres_det.reshape(batch_size, n_rounds, n_detectors_per_type)

        # 交错 X 和 Z 检测器：在每个时间步，先 X 后 Z
        # (B, T, 2, (D-1)^2) -> (B, 2*T, (D-1)^2) -> (B, 2*T*(D-1)^2)
        x_z_stack = torch.stack([x_det_flat, z_det_flat], dim=2)  # (B, T, 2, (D-1)^2)
        x_z_flat = x_z_stack.reshape(batch_size, -1)  # (B, 2*T*(D-1)^2)

        x_z_pres_stack = torch.stack([x_pres_flat, z_pres_flat], dim=2)
        x_z_pres_flat = x_z_pres_stack.reshape(batch_size, -1)

        # 检测器节点特征：(B, 4, 2*T*(D-1)^2)
        det_features = torch.stack([
            x_z_flat,  # X 和 Z 的 syndrome 交错
            x_z_pres_flat,  # X 和 Z 的 presence 交错
        ], dim=1)  # (B, 2, 2*T*(D-1)^2)

        # 扩展为 4 个通道 (通过复制)
        det_features = det_features.repeat(1, 2, 1)  # (B, 4, 2*T*(D-1)^2)

        # 分离为独立的特征通道
        det_features = torch.stack([
            x_z_flat,  # channel 0: syndrome (X和Z交错)
            x_z_flat,  # channel 1: syndrome (相同数据，用于简化)
            x_z_pres_flat,  # channel 2: presence (X和Z交错)
            x_z_pres_flat,  # channel 3: presence (相同数据，用于简化)
        ], dim=1)  # (B, 4, 2*T*(D-1)^2)

        # 数据比特节点特征：(D^2, 4) - 全零或 learned embedding
        n_data_nodes = distance ** 2
        data_features = torch.zeros(batch_size, n_data_nodes, 4, device=device)

        # 时间编码（可选）：为每个时间步添加位置信息
        if self.normalize_node_features:
            max_val = det_features.abs().max(dim=-1, keepdim=True).values
            det_features = det_features / max_val.clamp(min=1e-6)

        # 交换维度以匹配预期的格式: (B, N, 4)
        # det_features 是 (B, 4, 2*T*(D-1)^2)，需要转为 (B, 2*T*(D-1)^2, 4)
        det_features = det_features.permute(0, 2, 1).contiguous()  # (B, N_detectors, 4)

        # 合并节点特征
        node_features = torch.cat([det_features, data_features], dim=1)  # (B, N_nodes, 4)

        # 创建节点类型标记（用于区分检测器和数据比特）
        n_detector_nodes = det_features.size(1)
        n_total_nodes = n_detector_nodes + n_data_nodes

        node_types = torch.cat([
            torch.zeros(n_detector_nodes, device=device, dtype=torch.long),  # type 0: 检测器
            torch.ones(n_data_nodes, device=device, dtype=torch.long),      # type 1: 数据比特
        ])

        return node_features, node_types

        # 数据比特节点特征：(D^2, 4) - 全零或 learned embedding
        n_data_nodes = distance ** 2
        data_features = torch.zeros(batch_size, n_data_nodes, 4, device=device)

        # 时间编码（可选）：为每个时间步添加位置信息
        if self.normalize_node_features:
            max_val = det_features.abs().max(dim=-1, keepdim=True).values
            det_features = det_features / max_val.clamp(min=1e-6)

        # 合并节点特征
        node_features = torch.cat([det_features, data_features], dim=1)  # (B, N_nodes, 4)

        # 创建节点类型标记（用于区分检测器和数据比特）
        n_detector_nodes = det_features.size(1)
        n_total_nodes = n_detector_nodes + n_data_nodes

        node_types = torch.cat([
            torch.zeros(n_detector_nodes, device=device),  # type 0: 检测器
            torch.ones(n_data_nodes, device=device),      # type 1: 数据比特
        ]).long()

        return node_features, node_types

    def _create_edges(
        self,
        distance: int,
        n_rounds: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建边索引和边类型

        边类型：
        - type 0: 空间边（相邻数据比特）
        - type 1: 时间边（同检测器不同时间步）
        - type 2: 支撑边（检测器 ↔ 数据比特）
        """
        n_data_nodes = distance ** 2
        n_detectors_per_type = (distance - 1) ** 2
        n_detector_nodes = n_detectors_per_type * 2 * n_rounds  # X + Z, 所有时间步
        n_total_nodes = n_detector_nodes + n_data_nodes

        edges_list = []
        edge_types_list = []

        # 1. 空间边：相邻数据比特
        if self.include_spatial_edges:
            offset = n_detector_nodes  # 数据比特节点的起始索引
            for r in range(distance):
                for c in range(distance):
                    node_id = offset + (r * distance + c)
                    # 右邻居
                    if c < distance - 1:
                        neighbor = offset + (r * distance + (c + 1))
                        edges_list.append([node_id, neighbor])
                        edges_list.append([neighbor, node_id])  # 无向边
                        edge_types_list.extend([0, 0])
                    # 下邻居
                    if r < distance - 1:
                        neighbor = offset + ((r + 1) * distance + c)
                        edges_list.append([node_id, neighbor])
                        edges_list.append([neighbor, node_id])
                        edge_types_list.extend([0, 0])

        # 2. 时间边：同一检测器的不同时间步
        if self.include_temporal_edges:
            det_size = n_detectors_per_type * 2  # X + Z 检测器（单个时间步）
            for t in range(n_rounds - 1):
                current_start = t * det_size
                next_start = (t + 1) * det_size
                for i in range(det_size):
                    edges_list.append([current_start + i, next_start + i])
                    edges_list.append([next_start + i, current_start + i])
                    edge_types_list.extend([1, 1])

        # 3. 支撑边：检测器 ↔ 其支撑的数据比特
        # 简化版：每个检测器连接到其对应的 4 个数据比特
        if self.include_support_edges:
            # 修正节点索引计算
            n_data_nodes = distance ** 2

            for t in range(n_rounds):
                # 当前时间步的检测器索引偏移
                det_offset = t * (n_detectors_per_type * 2)

                for r in range(distance - 1):
                    for c in range(distance - 1):
                        # X 检测器
                        x_det_idx = det_offset + (r * (distance - 1) + c)

                        # 对应的数据比特：检测器 (r,c) 支撑数据比特 (r,c), (r+1,c), (r,c+1)
                        data_indices = [
                            r * distance + c,
                            (r + 1) * distance + c,
                            r * distance + (c + 1),
                            (r + 1) * distance + (c + 1),
                        ]

                        for data_idx in data_indices:
                            if data_idx < n_data_nodes:  # 边界检查
                                edges_list.append([x_det_idx, n_detector_nodes + data_idx])
                                edge_types_list.append(2)

                        # Z 检测器
                        z_det_idx = det_offset + n_detectors_per_type + (r * (distance - 1) + c)

                        # 对应的数据比特：检测器 (r,c) 支撑数据比特 (r,c), (r,c+1), (r+1,c), (r+1,c+1)
                        data_indices = [
                            r * distance + c,
                            r * distance + (c + 1),
                            (r + 1) * distance + c,
                            (r + 1) * distance + (c + 1),
                        ]

                        for data_idx in data_indices:
                            if data_idx < n_data_nodes:  # 边界检查
                                edges_list.append([z_det_idx, n_detector_nodes + data_idx])
                                edge_types_list.append(2)

        # 转换为张量
        if len(edges_list) > 0:
            edge_index = torch.tensor(edges_list, dtype=torch.long).t()
            edge_type = torch.tensor(edge_types_list, dtype=torch.long)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_type = torch.empty(0, dtype=torch.long)

        return edge_index, edge_type


def build_syndrome_graph(
    syndrome: torch.Tensor,
    distance: int,
    n_rounds: int,
    builder: Optional[SyndromeGraphBuilder] = None,
) -> GraphStructure:
    """
    便捷函数：从 syndrome 构建图

    Args:
        syndrome: (B, 4, T, D, D) syndrome 张量
        distance: 码距
        n_rounds: 时间步数
        builder: 可选的图构建器

    Returns:
        GraphStructure
    """
    if builder is None:
        builder = SyndromeGraphBuilder()

    return builder.build_graph(syndrome, distance, n_rounds)
