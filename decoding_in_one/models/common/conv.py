# decoding_in_one/models/common/conv.py
"""Conv3D 相关的通用构建块"""

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """获取激活函数

    Args:
        name: 激活函数名称 ("relu", "gelu", "leakyrelu")

    Returns:
        对应的激活函数模块
    """
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU(approximate="tanh")
    if key == "leakyrelu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation: {name}")


class Conv3DBlock(nn.Module):
    """Conv3D 块：Conv3d + Dropout + Activation

    可复用的网络组件。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout_p: float = 0.1,
        activation: str = "gelu",
        use_dropout: bool = True,
        use_activation: bool = True,
    ):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            dropout_p: Dropout 概率
            activation: 激活函数名称
            use_dropout: 是否使用 Dropout
            use_activation: 是否使用激活函数
        """
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        layers = []
        if use_dropout:
            layers.append(nn.Dropout3d(p=dropout_p))
        if use_activation:
            layers.append(get_activation(activation))

        self.post = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post(self.conv(x))
