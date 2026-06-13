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

        # 构建图结构（与样本无关的拓扑结构）
        graph = build_syndrome_graph(x[0:1], distance, n_rounds)
        edge_index = graph.edge_index  # (2, E)

        # 对 batch 中每个样本独立处理
        outputs = []
        for b in range(batch_size):
            # 提取第 b 个样本的节点特征
            # graph_builder 返回 (B=1, N, 4)，这里直接从 syndrome 构建节点特征
            node_features = self._extract_node_features(
                x[b:b+1], distance, n_rounds, device
            )  # (N, 4)

            # GNN 消息传递
            for layer in self.gnn_layers:
                if isinstance(layer, nn.Dropout):
                    node_features = layer(node_features)
                else:
                    node_features = layer(node_features, edge_index)

            # 读出：节点特征 → 预测
            predictions = self.readout(node_features)  # (N, 4)

            # 重塑回 (1, 4, T, D, D) 格式
            output = self._reshape_to_output(predictions, distance, n_rounds, device)
            outputs.append(output)

        return torch.cat(outputs, dim=0)  # (B, 4, T, D, D)

    def _extract_node_features(
        self,
        syndrome: torch.Tensor,
        distance: int,
        n_rounds: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        从 syndrome 张量提取节点特征（与图构建器的节点排列一致）。

        Args:
            syndrome: (1, 4, T, D, D)
            distance: 码距
            n_rounds: 时间步数
            device: 设备

        Returns:
            (N, node_features) 节点特征
        """
        # 使用 graph_builder 来构建节点特征
        graph = self.graph_builder.build_graph(syndrome, distance, n_rounds)
        # graph.x 是 (1, N, 4)，squeeze 为 (N, 4)
        return graph.x.squeeze(0)

    def _reshape_to_output(
        self,
        node_features: torch.Tensor,
        distance: int,
        n_rounds: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        将节点特征重塑回 (1, 4, T, D, D) 格式。

        输出 4 个通道：[z_err, x_err, s1x, s1z]
        与 Ising-Decoding 的 trainY 格式一致。

        Args:
            node_features: (N, 4) 节点特征（4 = output_channels）
            distance: 码距
            n_rounds: 时间步数
            device: 设备

        Returns:
            (1, 4, T, D, D) 输出张量
        """
        n_detectors_per_type = (distance - 1) ** 2
        n_detector_nodes = n_detectors_per_type * 2 * n_rounds
        n_data_nodes = distance ** 2

        # 初始化输出
        output = torch.zeros(1, 4, n_rounds, distance, distance, device=device)

        # ---- 检测器节点 → s1x, s1z 通道 ----
        # 节点排列: [t0_x_dets, t0_z_dets, t1_x_dets, t1_z_dets, ...]
        # 每个时间步: X 检测器 (n_detectors_per_type) + Z 检测器 (n_detectors_per_type)
        det_features = node_features[:n_detector_nodes]  # (N_det, 4)

        for t in range(n_rounds):
            det_offset = t * n_detectors_per_type * 2

            # X 检测器 → s1x 通道 (channel 2)
            x_det_feat = det_features[det_offset:det_offset + n_detectors_per_type]  # ((D-1)^2, 4)
            x_det_reshaped = x_det_feat[:, 2].reshape(distance - 1, distance - 1)
            output[0, 2, t, 1:distance, 1:distance] = x_det_reshaped

            # Z 检测器 → s1z 通道 (channel 3)
            z_det_feat = det_features[
                det_offset + n_detectors_per_type:det_offset + n_detectors_per_type * 2
            ]  # ((D-1)^2, 4)
            z_det_reshaped = z_det_feat[:, 3].reshape(distance - 1, distance - 1)
            output[0, 3, t, 1:distance, 1:distance] = z_det_reshaped

        # ---- 数据比特节点 → z_err, x_err 通道 ----
        # 数据节点在检测器节点之后
        data_features = node_features[n_detector_nodes:n_detector_nodes + n_data_nodes]  # (D^2, 4)

        for t in range(n_rounds):
            # z_err (channel 0) 和 x_err (channel 1) 从数据节点的读出获取
            # 使用数据节点的 channel 0 → z_err, channel 1 → x_err
            z_err = data_features[:, 0].reshape(distance, distance)
            x_err = data_features[:, 1].reshape(distance, distance)
            output[0, 0, t] = z_err
            output[0, 1, t] = x_err

        return output

    def get_input_channels(self) -> int:
        return 4  # syndrome 通道数

    def expected_input_rank(self) -> int:
        return 5  # (B, C, T, D, D)

    def output_shape(self) -> tuple:
        return (4, None, None, None)
