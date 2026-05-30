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
