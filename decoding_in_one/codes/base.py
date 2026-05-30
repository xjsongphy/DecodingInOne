# decoding_in_one/codes/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class PauliString:
    """Pauli 算符表示（简化版）"""
    def __init__(self, operators: str):
        """
        Args:
            operators: Pauli 算符字符串，如 "XZIY"
        """
        if not all(c in 'XYZI' for c in operators):
            raise ValueError("Pauli operators must be X, Y, Z, or I")
        self.operators = operators

class QuantumCode(ABC):
    """量子纠错码抽象基类"""

    @abstractmethod
    def get_n_physical(self) -> int:
        """返回物理比特数"""
        pass

    @abstractmethod
    def get_n_logical(self) -> int:
        """返回逻辑比特数"""
        pass

    @abstractmethod
    def get_stabilizers(self) -> List[PauliString]:
        """返回所有稳定子生成元"""
        pass

    @abstractmethod
    def get_logical_operators(self) -> Dict[str, PauliString]:
        """返回逻辑算符 {'X': ..., 'Z': ...}"""
        pass

    @abstractmethod
    def get_qubit_topology(self) -> Dict[int, Tuple[int, int]]:
        """返回比特到 2D 坐标的映射"""
        pass
