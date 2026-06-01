# decoding_in_one/models/surface_code/gnn_decoder.py
"""
表面码 GNN 解码器

使用图神经网络处理 syndrome 数据，预测错误位置。
参数量设计：与 Conv3D 解码器相当（~1-2M 参数）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from decoding_in_one.models.base import DecodingModel
from decoding_in_one.models.common.gnn_layers import GCNConv, GATConv, MLPReadout
from decoding_in_one.models.common.graph_builder import SyndromeGraphBuilder, build_syndrome_graph


class SurfaceCodeGNNDecoder(DecodingModel):
    """
    表面码 GNN 解码器

    架构：
    1. 图构建器：syndrome → 图结构
    2. GNN 层：多层图卷积聚合邻居信息
    3. 读出层：节点特征 → 预测错误

    输入：(B, 4, T, D, D) syndrome 张量
    输出：(B, 4, T, D, D) dense logical field

    参数量设计：
    - GNN 层数：3-4 层
    - 隐藏维度：64-128
    - 总参数：~1-2M（与 Conv3D 相当）
    """

    def __init__(self, config):
        """
        Args:
            config: GNNModelConfig 配置对象
        """
        super().__init__()
        self.config = config

        # 图构建器
        self.graph_builder = SyndromeGraphBuilder(
            include_spatial_edges=True,
            include_temporal_edges=config.use_temporal_edges,
            include_support_edges=True,
        )

        # GNN 层
        self.gnn_layers = nn.ModuleList()
        in_channels = config.node_features  # syndrome 通道数 = 4
        hidden_channels = config.hidden_channels
        out_channels = config.hidden_channels

        for i in range(config.num_layers):
            if i == 0:
                layer_in_channels = in_channels
            else:
                layer_in_channels = hidden_channels

            if config.layer_type == 'GCN':
                layer = GCNConv(layer_in_channels, hidden_channels)
            elif config.layer_type == 'GAT':
                layer = GATConv(
                    layer_in_channels,
                    hidden_channels,
                    heads=config.attention_heads,
                    dropout=config.dropout,
                )
            else:
                raise ValueError(f"Unknown layer_type: {config.layer_type}")

            self.gnn_layers.append(layer)
            self.gnn_layers.append(nn.Dropout(config.dropout))

        # 读出层：节点特征 → 预测
        self.readout = MLPReadout(
            in_channels=hidden_channels,
            hidden_channels=config.readout_hidden,
            out_channels=config.output_channels,  # 4 (z_err, x_err, s1x, s1z)
            num_layers=config.readout_layers,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor, distance: int, n_rounds: int) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (B, 4, T, D, D) syndrome 张量
            distance: 码距 D
            n_rounds: 时间步数 T

        Returns:
            (B, 4, T, D, D) 预测的错误场
        """
        batch_size = x.size(0)
        device = x.device

        # 1. 构建图结构（对每个样本独立构建）
        # 为简化，先对第一个样本构建，然后批量处理
        # TODO: 实现高效的批量图构建
        graph = build_syndrome_graph(x[0:1], distance, n_rounds)

        # 2. GNN 消息传递
        # graph.x 是 (B, N, F)，需要 squeeze 为 (N, F) 用于 GNN 层
        node_features = graph.x.squeeze(0)  # (N, node_features)
        edge_index = graph.edge_index  # (2, E)

        for layer in self.gnn_layers:
            if isinstance(layer, nn.Dropout):
                node_features = layer(node_features)
            else:
                node_features = layer(node_features, edge_index)

        # 3. 读出：节点特征 → 预测
        predictions = self.readout(node_features)  # (N, 4)

        # 4. 重塑回 (B, 4, T, D, D) 格式
        # 需要按照节点类型和位置重新组织
        # 简化实现：这里只处理单个样本的情况
        output = self._reshape_to_output(predictions, distance, n_rounds)

        # 扩展到 batch
        if batch_size > 1:
            output = output.expand(batch_size, -1, -1, -1, -1)

        return output

    def _reshape_to_output(self, node_features: torch.Tensor, distance: int, n_rounds: int) -> torch.Tensor:
        """
        将节点特征重塑回 (1, 4, T, D, D) 格式

        Args:
            node_features: (N, 4) 节点特征
            distance: 码距
            n_rounds: 时间步数

        Returns:
            (1, 4, T, D, D) 输出张量
        """
        # 简化实现：直接使用检测器节点特征
        # 实际实现需要根据节点类型和位置正确映射

        n_detectors_per_type = (distance - 1) ** 2
        n_detector_nodes = n_detectors_per_type * 2 * n_rounds

        # 只取检测器节点特征
        det_features = node_features[:n_detector_nodes]  # (N_det, 4)

        # 重塑为 (T, 2, D-1, D-1, 4)
        det_features = det_features.reshape(n_rounds, 2, distance - 1, distance - 1, 4)

        # 分离 X 和 Z 特征
        x_features = det_features[:, 0]  # (T, D-1, D-1, 4)
        z_features = det_features[:, 1]  # (T, D-1, D-1, 4)

        # 扩展到完整网格（填充边界）
        output = torch.zeros(1, 4, n_rounds, distance, distance, device=node_features.device)

        # 填充内部区域
        output[0, 0, :, 1:distance, 1:distance] = x_features[:, :, 0]  # channel 0: z_err from x_syn
        output[0, 1, :, 1:distance, 1:distance] = z_features[:, :, 1]  # channel 1: x_err from z_syn

        return output

    def get_input_channels(self) -> int:
        return 4  # syndrome 通道数

    def expected_input_rank(self) -> int:
        return 5  # (B, C, T, D, D)

    def output_shape(self) -> tuple:
        return (4, None, None, None)
