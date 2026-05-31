# decoding_in_one/noise/base.py
from abc import ABC, abstractmethod
from typing import Dict, Optional, TYPE_CHECKING
from decoding_in_one.utils.types import CircuitArtifact

if TYPE_CHECKING:
    from decoding_in_one.codes import QuantumCode


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

    # ---- 距离依赖噪声接口 ----

    def get_distance_scaled_error_rate(
        self,
        base_error_rate: float,
        qubit1: int,
        qubit2: int,
        code: Optional["QuantumCode"] = None,
    ) -> float:
        """基于比特间距离的缩放错误率（为距离依赖噪声预留）

        允许噪声模型根据比特间的空间距离调整错误率。
        这对于建模非均匀噪声或串扰噪声非常重要。

        Args:
            base_error_rate: 基础错误率
            qubit1: 第一个比特 ID
            qubit2: 第二个比特 ID
            code: QuantumCode 实例（用于计算距离）

        Returns:
            缩放后的错误率

        Note:
            默认实现返回基础错误率（无距离依赖）。
            子类可以覆盖以实现具体的距离-噪声关系。
        """
        if code is None:
            return base_error_rate

        # 获取距离缩放因子
        scaling_factor = code.get_noise_scaling_factor(qubit1, qubit2)
        return base_error_rate * scaling_factor
