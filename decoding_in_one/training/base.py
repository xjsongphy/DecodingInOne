# decoding_in_one/training/base.py
from abc import ABC, abstractmethod
from typing import Any

class Trainer(ABC):
    """训练器抽象基类"""

    @abstractmethod
    def train(self, train_loader, val_loader, **kwargs) -> dict[str, Any]:
        """执行训练

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器

        Returns:
            训练报告字典
        """
        pass
