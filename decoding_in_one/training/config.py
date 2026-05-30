# decoding_in_one/training/config.py
from dataclasses import dataclass

@dataclass
class OptimConfig:
    """训练优化配置（仅包含训练循环相关参数）"""
    batch_size: int = 512
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "auto"
    seed: int = 0
    out_dir: str = "experiments/ising/train_output"
