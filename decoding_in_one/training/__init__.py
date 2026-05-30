"""训练框架模块

包含训练器抽象基类和通用 PyTorch 训练器。
"""

from decoding_in_one.training.config import OptimConfig
# 从 trainer.py 导入通用 Trainer
from decoding_in_one.training.trainer import Trainer

__all__ = ["OptimConfig", "Trainer"]
