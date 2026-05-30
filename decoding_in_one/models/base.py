# decoding_in_one/models/base.py
from abc import ABC, abstractmethod
import torch.nn as nn

class DecodingModel(nn.Module, ABC):
    """神经网络解码模型的抽象基类

    所有解码神经网络应继承此类并实现抽象方法。
    """

    @abstractmethod
    def get_input_channels(self) -> int:
        """返回输入通道数"""
        pass

    @abstractmethod
    def expected_input_rank(self) -> int:
        """返回期望的输入张量维度（如 5 表示 B,C,T,D,D）"""
        pass

    @abstractmethod
    def output_shape(self) -> tuple:
        """返回输出形状（不含 batch 维度）

        对于动态维度，使用 None 占位。
        例如 (4, None, None, None) 表示 (C, T, D, D)，其中 T 和 D 取决于输入。
        """
        pass
