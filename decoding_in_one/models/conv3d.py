# decoding_in_one/models/conv3d.py
import torch
import torch.nn as nn
from .base import DecodingModel
from .config import Conv3DModelConfig

class Conv3DNeuralDecoder(DecodingModel):
    """Conv3D 神经网络解码器

    输出：dense logical field (B,4,T,D,D)
    注意：不是最终的 logical observable 向量，而是时空密集预测。
    """

    def __init__(self, config: Conv3DModelConfig):
        super().__init__()
        self.config = config
        self.input_channels = config.input_channels
        self.out_channels = config.out_channels

        self.net = self._build_network(config)

    def _build_network(self, config: Conv3DModelConfig) -> nn.Sequential:
        """构建 Conv3D 网络"""
        act = self._get_activation(config.activation)
        layers: list[nn.Module] = []
        in_channels = config.input_channels

        for i, (filt, k) in enumerate(zip(config.num_filters, config.kernel_sizes)):
            layers.append(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=filt,
                    kernel_size=k,
                    padding=k // 2,
                )
            )
            if i < len(config.num_filters) - 1:
                layers.append(nn.Dropout3d(p=config.dropout))
                layers.append(act)
            in_channels = filt

        return nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """获取激活函数"""
        key = name.lower()
        if key == "relu":
            return nn.ReLU()
        if key == "gelu":
            return nn.GELU(approximate="tanh")
        if key == "leakyrelu":
            return nn.LeakyReLU()
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输出 dense logical field"""
        return self.net(x)

    def get_input_channels(self) -> int:
        return self.input_channels

    def expected_input_rank(self) -> int:
        return 5  # (B, C, T, D, D)

    def output_shape(self) -> tuple:
        return (self.out_channels, None, None, None)
