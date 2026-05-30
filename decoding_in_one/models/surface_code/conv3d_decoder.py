# decoding_in_one/models/surface_code/conv3d_decoder.py
"""表面码专用 Conv3D 解码器

使用通用构建块组装，适用于表面码的 2D 网格结构。
"""

import torch
import torch.nn as nn
from decoding_in_one.models.common.conv import Conv3DBlock
from decoding_in_one.models.base import DecodingModel
from decoding_in_one.models.config import Conv3DModelConfig


class SurfaceCodeConv3DDecoder(DecodingModel):
    """表面码 Conv3D 解码器

    专为表面码设计，利用其 2D 网格几何结构。
    输入形状：(B, C, T, D, D)，其中 T 是测量轮数，D 是码距。
    输出：dense logical field (B, C, T, D, D)。
    """

    def __init__(self, config: Conv3DModelConfig):
        super().__init__()
        self.config = config
        self.input_channels = config.input_channels
        self.out_channels = config.out_channels

        self.net = self._build_network(config)

    def _build_network(self, config: Conv3DModelConfig) -> nn.Sequential:
        """构建 Conv3D 网络

        使用通用 Conv3DBlock 组件。
        """
        layers: list[nn.Module] = []
        in_channels = config.input_channels

        for i, (filt, k) in enumerate(zip(config.num_filters, config.kernel_sizes)):
            # 最后一层不添加 dropout 和激活
            is_last = (i == len(config.num_filters) - 1)
            layers.append(
                Conv3DBlock(
                    in_channels=in_channels,
                    out_channels=filt,
                    kernel_size=k,
                    dropout_p=config.dropout,
                    activation=config.activation,
                    use_dropout=not is_last,
                    use_activation=not is_last,
                )
            )
            in_channels = filt

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输出 dense logical field"""
        return self.net(x)

    def get_input_channels(self) -> int:
        return self.input_channels

    def expected_input_rank(self) -> int:
        return 5  # (B, C, T, D, D)

    def output_shape(self) -> tuple:
        return (self.out_channels, None, None, None)
