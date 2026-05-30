import torch
import torch.nn as nn


class Conv3DPredecoder(nn.Module):
    """Ising-style predecoder backbone: stacked Conv3d blocks."""

    def __init__(
        self,
        input_channels: int = 4,
        out_channels: int = 4,
        num_filters: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        dropout_p: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        if num_filters is None:
            num_filters = [64, 64, 64, out_channels]
        if kernel_sizes is None:
            kernel_sizes = [3] * len(num_filters)
        if len(num_filters) != len(kernel_sizes):
            raise ValueError("num_filters and kernel_sizes must have the same length")
        if num_filters[-1] != out_channels:
            raise ValueError("num_filters[-1] must equal out_channels")

        act = self._get_activation(activation)
        layers: list[nn.Module] = []
        in_channels = input_channels
        for i, (filt, k) in enumerate(zip(num_filters, kernel_sizes)):
            layers.append(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=filt,
                    kernel_size=k,
                    padding=k // 2,
                )
            )
            if i < len(num_filters) - 1:
                layers.append(nn.Dropout3d(p=dropout_p))
                layers.append(act)
            in_channels = filt
        self.net = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        key = name.lower()
        if key == "relu":
            return nn.ReLU()
        if key == "gelu":
            return nn.GELU(approximate="tanh")
        if key == "leakyrelu":
            return nn.LeakyReLU()
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
