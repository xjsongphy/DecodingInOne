# decoding_in_one/models/common/gnn_layers.py
"""
图神经网络层实现

提供常用的 GNN 层：
- GCNConv: 图卷积层 (Kipf & Welling)
- GATConv: 图注意力层 (Veličković et al.)
- MessagePassingBase: 消息传递基类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MessagePassingBase(nn.Module):
    """消息传递基类"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征 (N, in_channels)
            edge_index: 边索引 (2, E)，格式 [source, target]

        Returns:
            更新后的节点特征 (N, out_channels)
        """
        raise NotImplementedError


class GCNConv(MessagePassingBase):
    """
    图卷积层 (GCN - Kipf & Welling)

    公式: D^-0.5 A D^-0.5 X W
    其中 D 是度矩阵，A 是邻接矩阵（含自环）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        normalize: bool = True,
    ):
        super().__init__(in_channels, out_channels)
        self.normalize = normalize

        # 线性变换
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征 (N, in_channels)
            edge_index: 边索引 (2, E)
        """
        # 添加自环（每个节点连向自己）
        num_nodes = x.size(0)
        edge_index, _ = self._add_self_loops(edge_index, num_nodes)

        # 计算度矩阵 D^-0.5
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

        # 消息传递: gather features from neighbors
        out = torch.zeros_like(x)
        for i in range(x.size(1)):
            # 聚合邻居特征（包含自环）
            src_features = x[row] * deg_inv_sqrt[row].view(-1, 1)
            dst_features = src_features * deg_inv_sqrt[col].view(-1, 1)
            out.scatter_add_(0, col.unsqueeze(1).expand(-1, dst_features.size(1)), dst_features)

        # 线性变换
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            out = out + self.bias

        return out

    def _add_self_loops(self, edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, int]:
        """添加自环边"""
        loop_index = torch.arange(0, num_nodes, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        return edge_index, num_nodes


class GATConv(MessagePassingBase):
    """
    图注意力层 (GAT - Veličković et al.)

    使用多头注意力机制聚合邻居信息。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__(in_channels, out_channels)
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # 每个头的维度
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        self.per_head_channels = out_channels // heads

        # 线性变换：a = W_l x
        self.lin_l = nn.Linear(in_channels, heads * self.per_head_channels, bias=False)
        self.lin_r = nn.Linear(in_channels, heads * self.per_head_channels, bias=False)

        # 注意力参数
        self.att_l = nn.Parameter(torch.Tensor(1, heads, self.per_head_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, heads, self.per_head_channels))

        if bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征 (N, in_channels)
            edge_index: 边索引 (2, E)
        """
        num_nodes = x.size(0)

        # 添加自环
        edge_index, _ = self._add_self_loops(edge_index, num_nodes)

        # 线性变换
        x_l = self.lin_l(x).view(-1, self.heads, self.per_head_channels)  # (N, heads, C')
        x_r = self.lin_r(x).view(-1, self.heads, self.per_head_channels)  # (N, heads, C')

        # 计算注意力系数
        row, col = edge_index  # (E,), (E,)

        # alpha_l = a_l^T (W_l x_i), alpha_r = a_r^T (W_r x_j)
        alpha_l = (x_l[row] * self.att_l).sum(dim=-1)  # (E, heads)
        alpha_r = (x_r[col] * self.att_r).sum(dim=-1)  # (E, heads)
        alpha = F.leaky_relu(alpha_l + alpha_r, 0.2)  # (E, heads)

        # Softmax 归一化
        alpha = alpha.softmax(dim=0)  # 对每个目标节点的所有边
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 消息传递: out_i = sum(alpha_ij * x_r_j)
        out = torch.zeros(num_nodes, self.heads, self.per_head_channels, device=x.device)

        for head in range(self.heads):
            out_features = x_r[col, head] * alpha[:, head].unsqueeze(1)
            out[:, head, :].scatter_add_(0, col.unsqueeze(1).expand(-1, out_features.size(1)), out_features)

        # 合并多头
        if self.concat:
            out = out.view(-1, self.out_channels)  # (N, out_channels)
        else:
            out = out.mean(dim=1)  # (N, C')

        if self.bias is not None:
            out = out + self.bias

        return out

    def _add_self_loops(self, edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, int]:
        """添加自环边"""
        loop_index = torch.arange(0, num_nodes, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        return edge_index, num_nodes


class MLPReadout(nn.Module):
    """
    MLP 读出层，用于从节点特征预测输出

    常用于 GNN 的最后一步，将聚合后的节点特征映射到输出空间。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()
        self.num_layers = num_layers

        layers = []
        in_ch = in_channels
        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(nn.Linear(in_ch, out_channels))
            else:
                layers.append(nn.Linear(in_ch, hidden_channels))
                layers.append(self._get_activation(activation))
                layers.append(nn.Dropout(dropout))
                in_ch = hidden_channels

        self.mlp = nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        if activation == 'gelu':
            return nn.GELU()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
