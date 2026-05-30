# decoding_in_one/decoders/base.py
from abc import ABC, abstractmethod
from typing import Optional
import torch

class Correction:
    """解码结果（校正或预测）"""
    def __init__(self, predictions: torch.Tensor):
        """
        Args:
            predictions: 预测的校正或逻辑错误 (batch, n_qubits或 n_observables)
        """
        self.predictions = predictions

    @property
    def shape(self) -> tuple:
        return self.predictions.shape

class Decoder(ABC):
    """解码器抽象基类"""

    @abstractmethod
    def decode(
        self,
        syndrome: torch.Tensor,
        observables: Optional[torch.Tensor] = None
    ) -> Correction:
        """
        解码接口

        Args:
            syndrome: 检测器测量结果 (batch, n_detectors)
            observables: 观测量 (batch, n_observables)，可选

        Returns:
            Correction: 预测的校正
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """返回解码器名称"""
        pass
