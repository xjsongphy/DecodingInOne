# decoding_in_one/codes/surface_code.py
"""
从 Ising-Decoding 迁移的表面码实现
源码参考: Ising-Decoding/code/qec/surface_code/memory_circuit.py
"""

import numpy as np
from typing import Dict, List, Tuple
from decoding_in_one.codes.base import QuantumCode, PauliString

class SurfaceCode(QuantumCode):
    """
    旋转表面码实现

    Args:
        distance: 码距（必须是奇数）
        rotation: 电路方向 'XV', 'XH', 'ZV', 'ZH'
    """

    def __init__(self, distance: int, rotation: str = 'XV'):
        if distance % 2 == 0:
            raise ValueError("Distance must be odd")

        self.distance = distance
        self.rotation = rotation

        # 构建码结构
        self._build_code()

    def _build_code(self):
        """构建表面码结构（从 Ising 迁移）"""
        # 数据比特
        n_data = self.distance ** 2
        self._data_qubits = list(range(n_data))

        # X 型稳定子（面内）
        n_x_checks = (self.distance ** 2 - 1) // 2
        self._xcheck_qubits = list(range(n_data, n_data + n_x_checks))

        # Z 型稳定子（面内）
        n_z_checks = (self.distance ** 2 - 1) // 2
        self._zcheck_qubits = list(range(
            n_data + n_x_checks,
            n_data + n_x_checks + n_z_checks
        ))

        # 构建稳定子-数据比特连接关系
        self._build_stabilizer_connections()

    def _build_stabilizer_connections(self):
        """构建每个稳定子连接的数据比特"""
        self._x_connections = {}
        self._z_connections = {}

        # 为每个 X 型稳定子分配连接的数据比特（简化版：按顺序分配4个）
        for i, xcheck in enumerate(self._xcheck_qubits):
            start_idx = i * 4
            connections = []
            for j in range(4):
                data_idx = (start_idx + j) % len(self._data_qubits)
                connections.append(data_idx)
            self._x_connections[xcheck] = connections

        # 为每个 Z 型稳定子分配连接的数据比特（简化版）
        for i, zcheck in enumerate(self._zcheck_qubits):
            start_idx = (i * 4 + 2) % len(self._data_qubits)
            connections = []
            for j in range(4):
                data_idx = (start_idx + j) % len(self._data_qubits)
                connections.append(data_idx)
            self._z_connections[zcheck] = connections

    def get_n_physical(self) -> int:
        return len(self._data_qubits)

    def get_n_logical(self) -> int:
        return 1

    def get_stabilizers(self) -> List[PauliString]:
        """返回所有稳定子（简化版）"""
        stabilizers = []

        # X 型稳定子
        for _ in self._xcheck_qubits:
            stabilizers.append(PauliString("X" * self.get_n_physical()))

        # Z 型稳定子
        for _ in self._zcheck_qubits:
            stabilizers.append(PauliString("Z" * self.get_n_physical()))

        return stabilizers

    def get_logical_operators(self) -> Dict[str, PauliString]:
        """返回逻辑算符"""
        # 简化版：返回逻辑 X 和 Z
        return {
            'X': PauliString("X" * self.distance + "I" * (self.get_n_physical() - self.distance)),
            'Z': PauliString("Z" * self.distance + "I" * (self.get_n_physical() - self.distance)),
        }

    def get_qubit_topology(self) -> Dict[int, Tuple[int, int]]:
        """返回比特到 2D 坐标的映射"""
        topology = {}

        # 数据比特排列在 distance × distance 网格的奇数位置
        idx = 0
        for i in range(self.distance):
            for j in range(self.distance):
                if (i + j) % 2 == 0:  # 数据比特位置
                    topology[idx] = (2 * i + 1, 2 * j + 1)
                    idx += 1

        return topology
