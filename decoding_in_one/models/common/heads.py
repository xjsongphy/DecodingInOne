# decoding_in_one/models/common/heads.py
"""输出头相关的通用组件"""

import torch
import torch.nn as nn
from typing import Literal


class PoolingHead(nn.Module):
    """池化输出头

    将空间/时间维度池化为单一值或向量。
    """

    def __init__(
        self,
        method: Literal["mean", "max", "sum"] = "mean",
        keepdim: bool = False,
    ):
        """
        Args:
            method: 池化方法
            keepdim: 是否保持维度
        """
        super().__init__()
        self.method = method
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，假设空间/时间维度在后几维

        Returns:
            池化后的张量
        """
        if self.method == "mean":
            return x.mean(dim=(2, 3, 4), keepdim=self.keepdim)
        elif self.method == "max":
            return x.amax(dim=(2, 3, 4), keepdim=self.keepdim)
        elif self.method == "sum":
            return x.sum(dim=(2, 3, 4), keepdim=self.keepdim)
        else:
            raise ValueError(f"Unknown pooling method: {self.method}")
