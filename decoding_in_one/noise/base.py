# decoding_in_one/noise/base.py
from abc import ABC, abstractmethod
from typing import Dict
from decoding_in_one.utils.types import CircuitArtifact

class NoiseModel(ABC):
    """噪声模型抽象基类"""

    @abstractmethod
    def apply_to_circuit(self, circuit: str | CircuitArtifact) -> str | CircuitArtifact:
        """
        将噪声应用到 Stim 电路

        Args:
            circuit: Stim 电路字符串或结构化电路对象

        Returns:
            带噪声的电路（保持输入类型）
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        """返回噪声参数字典"""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """验证参数有效性"""
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config_path: str) -> 'NoiseModel':
        """从 YAML 配置文件加载"""
        pass
