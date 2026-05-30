# decoding_in_one/circuits/base.py
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from decoding_in_one.utils.types import CircuitArtifact, CircuitSpec

if TYPE_CHECKING:
    from decoding_in_one.codes import QuantumCode

class CircuitBuilder(ABC):
    """量子电路构建器抽象基类"""

    @abstractmethod
    def build_stabilizer_measurement(
        self,
        code: 'QuantumCode',
        stabilizer_type: str,
        stabilizer_idx: int
    ) -> str:
        """
        构建单个稳定子测量的 Stim 电路片段

        Args:
            code: 量子纠错码对象
            stabilizer_type: 'X' 或 'Z'
            stabilizer_idx: 稳定子索引

        Returns:
            Stim 电路字符串片段
        """
        pass

    @abstractmethod
    def build_memory_circuit(
        self,
        code: 'QuantumCode',
        n_rounds: int,
        measurement_basis: str
    ) -> str:
        """
        构建完整的重复测量电路

        Args:
            code: 量子纠错码对象
            n_rounds: 测量轮数
            measurement_basis: 测量基 'X' 或 'Z'

        Returns:
            完整的 Stim 电路字符串
        """
        pass

    @abstractmethod
    def build_memory_artifact(
        self,
        code: 'QuantumCode',
        spec: CircuitSpec
    ) -> CircuitArtifact:
        """构建结构化电路对象。"""
        pass
