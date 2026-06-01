# decoding_in_one/models/config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Conv3DModelConfig:
    """Conv3D 模型结构配置"""
    input_channels: int = 4
    out_channels: int = 4
    num_filters: Optional[list[int]] = None
    kernel_sizes: Optional[list[int]] = None
    activation: str = "gelu"
    dropout: float = 0.1

    def __post_init__(self):
        if self.num_filters is None:
            self.num_filters = [64, 64, 64, self.out_channels]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3] * len(self.num_filters)

        # 验证
        if len(self.num_filters) != len(self.kernel_sizes):
            raise ValueError("num_filters and kernel_sizes must have the same length")

        if self.num_filters[-1] != self.out_channels:
            raise ValueError("num_filters[-1] must equal out_channels")


@dataclass
class GNNModelConfig:
    """GNN 模型结构配置"""
    # 图构建参数
    use_temporal_edges: bool = True  # 是否使用时间边
    node_features: int = 4  # 节点特征维度（syndrome 通道数）

    # GNN 层参数
    layer_type: str = "GCN"  # 'GCN' 或 'GAT'
    num_layers: int = 4  # GNN 层数
    hidden_channels: int = 128  # 隐藏维度
    attention_heads: int = 4  # GAT 的注意力头数
    dropout: float = 0.1  # Dropout 概率
    activation: str = "gelu"  # 激活函数

    # 读出层参数
    readout_hidden: int = 128  # 读出层隐藏维度
    readout_layers: int = 2  # 读出层数
    output_channels: int = 4  # 输出通道数（与 Conv3D 一致）

    def __post_init__(self):
        # 验证
        if self.layer_type not in ("GCN", "GAT"):
            raise ValueError("layer_type must be 'GCN' or 'GAT'")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if self.hidden_channels % self.attention_heads != 0:
            raise ValueError("hidden_channels must be divisible by attention_heads")
