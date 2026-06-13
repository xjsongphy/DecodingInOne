# decoding_in_one/codes/base.py
"""量子纠错码抽象基类与通用数据类型

扩展支持任意维度空间坐标，为空间距离相关噪声打基础。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
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

    def __repr__(self) -> str:
        return f"PauliString('{self.operators}')"


@dataclass(frozen=True)
class Stabilizer:
    """稳定子描述。"""
    check_qubit: int
    stabilizer_type: str
    data_qubits: Tuple[int, ...]


@dataclass(frozen=True)
class QubitCoordinate:
    """通用比特坐标（支持 1D/2D/3D/任意维度）

    Attributes:
        coords: 坐标元组，如 (0, 1) 表示 2D，(0, 1, 0) 表示 3D
        qubit_id: 比特 ID
        qubit_type: 'data', 'check_X', 'check_Z'
    """
    coords: Tuple[float, ...]
    qubit_id: int
    qubit_type: str  # 'data', 'check_X', 'check_Z'

    @property
    def dim(self) -> int:
        """坐标维度"""
        return len(self.coords)


class QuantumCode(ABC):
    """量子纠错码抽象基类

    子类必须实现所有抽象方法。
    distance 属性由子类在 __init__ 中设置。
    """

    # 子类应在 __init__ 中设置此属性
    distance: int

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

    @abstractmethod
    def get_data_qubits(self) -> List[int]:
        """返回数据比特 ID 列表。"""
        pass

    @abstractmethod
    def get_check_qubits(self, stabilizer_type: str) -> List[int]:
        """返回给定类型稳定子测量比特 ID 列表（'X' 或 'Z'）。"""
        pass

    @abstractmethod
    def get_stabilizer_supports(self, stabilizer_type: str) -> Dict[int, Tuple[int, ...]]:
        """返回稳定子到其支撑数据比特的映射。"""
        pass

    # ---- 新增：空间坐标系统 ----

    def get_qubit_coordinates(self) -> Dict[int, QubitCoordinate]:
        """返回所有比特的完整坐标信息

        默认实现从 get_qubit_topology() 转换。
        子类可以覆盖以提供更高维度的坐标。

        Returns:
            Dict[qubit_id, QubitCoordinate]
        """
        topology = self.get_qubit_topology()
        data_qubits = set(self.get_data_qubits())
        xcheck = set(self.get_check_qubits('X'))
        zcheck = set(self.get_check_qubits('Z'))

        result: Dict[int, QubitCoordinate] = {}
        for qid, coord in topology.items():
            if qid in data_qubits:
                qtype = 'data'
            elif qid in xcheck:
                qtype = 'check_X'
            elif qid in zcheck:
                qtype = 'check_Z'
            else:
                qtype = 'data'  # fallback
            result[qid] = QubitCoordinate(
                coords=tuple(float(c) for c in coord),
                qubit_id=qid,
                qubit_type=qtype,
            )
        return result

    def get_spatial_distance(self, qubit1: int, qubit2: int) -> float:
        """计算两个比特间的空间距离（欧氏距离）

        基于 get_qubit_coordinates() 返回的坐标。
        子类可以覆盖以使用其他距离度量（如 Manhattan 距离）。

        Args:
            qubit1: 第一个比特 ID
            qubit2: 第二个比特 ID

        Returns:
            欧氏距离
        """
        coords = self.get_qubit_coordinates()
        c1 = coords[qubit1].coords
        c2 = coords[qubit2].coords
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

    # ---- 新增：H 矩阵和图结构接口（为 QLDPC 通用化） ----

    def get_parity_check_matrices(self) -> dict[str, np.ndarray]:
        """返回校验矩阵 H_X 和 H_Z（为 QLDPC 等通用码设计）

        Returns:
            dict 包含:
                - 'H_X': X 型稳定子的校验矩阵 (n_X_stabilizers × n_data_qubits)
                - 'H_Z': Z 型稳定子的校验矩阵 (n_Z_stabilizers × n_data_qubits)

        Note:
            默认实现从 get_stabilizer_supports() 构建。
            对于 QLDPC 等稀疏码，子类应直接返回稀疏矩阵。
        """
        import numpy as np

        data_qubits = self.get_data_qubits()
        n_data = len(data_qubits)
        xcheck = self.get_check_qubits('X')
        zcheck = self.get_check_qubits('Z')
        n_x = len(xcheck)
        n_z = len(zcheck)

        # 构建 H_X 和 H_Z
        H_X = np.zeros((n_x, n_data), dtype=np.int8)
        H_Z = np.zeros((n_z, n_data), dtype=np.int8)

        x_supports = self.get_stabilizer_supports('X')
        z_supports = self.get_stabilizer_supports('Z')

        # 映射 qubit_id 到矩阵索引
        data_id_to_idx = {qid: i for i, qid in enumerate(data_qubits)}
        xcheck_id_to_idx = {qid: i for i, qid in enumerate(xcheck)}
        zcheck_id_to_idx = {qid: i for i, qid in enumerate(zcheck)}

        for check_qubit, support in x_supports.items():
            if check_qubit in xcheck_id_to_idx:
                row_idx = xcheck_id_to_idx[check_qubit]
                for data_q in support:
                    if data_q in data_id_to_idx:
                        H_X[row_idx, data_id_to_idx[data_q]] = 1

        for check_qubit, support in z_supports.items():
            if check_qubit in zcheck_id_to_idx:
                row_idx = zcheck_id_to_idx[check_qubit]
                for data_q in support:
                    if data_q in data_id_to_idx:
                        H_Z[row_idx, data_id_to_idx[data_q]] = 1

        return {'H_X': H_X, 'H_Z': H_Z}

    def get_stabilizer_graph(self) -> dict[str, list[tuple[int, int]]]:
        """返回稳定子到数据比特的图结构（支持非局部连接，为 QLDPC 设计）

        Returns:
            dict 包含:
                - 'X_edges': [(check_qubit, data_qubit), ...] 所有 X 型稳定子的边
                - 'Z_edges': [(check_qubit, data_qubit), ...] 所有 Z 型稳定子的边

        Note:
            默认实现从 get_stabilizer_supports() 构建。
            对于 QLDPC，这允许任意的非局部连接图。
        """
        x_supports = self.get_stabilizer_supports('X')
        z_supports = self.get_stabilizer_supports('Z')

        x_edges = []
        for check, data_list in x_supports.items():
            for data_q in data_list:
                x_edges.append((check, data_q))

        z_edges = []
        for check, data_list in z_supports.items():
            for data_q in data_list:
                z_edges.append((check, data_q))

        return {'X_edges': x_edges, 'Z_edges': z_edges}

    def get_stabilizer_measurement_layers(
        self, stabilizer_type: str
    ) -> list[list[tuple[int, int]]]:
        """返回稳定子测量的 CNOT 层（支持任意图结构，不限于表面码四层）

        对于非局部图（如 QLDPC），需要将边分层使得同一层内无冲突。
        默认实现使用简单的贪心着色算法。

        CNOT 方向约定（与 Ising-Decoding 一致）：
          - X 稳定子: CNOT(control=check_qubit, target=data_qubit)
            即 ancilla → data（H 门前后包裹 CNOT 实现 X 型测量）
          - Z 稳定子: CNOT(control=data_qubit, target=check_qubit)
            即 data → ancilla（直接 CNOT 实现 Z 型测量）

        Args:
            stabilizer_type: 'X' 或 'Z'

        Returns:
            List of CNOT layers，每层是 [(control, target), ...]

        Note:
            - 对于局部码（如表面码），可以返回优化的固定层数
            - 对于 QLDPC，使用图着色算法处理冲突
        """
        supports = self.get_stabilizer_supports(stabilizer_type)
        check_qubits = self.get_check_qubits(stabilizer_type)

        # 构建 CNOT 边：注意 X 和 Z 的 control/target 方向不同
        control_first = (stabilizer_type.upper() == 'X')
        edges = []
        for check_q in check_qubits:
            for data_q in supports.get(check_q, []):
                if control_first:
                    edges.append((check_q, data_q))   # X: ancilla → data
                else:
                    edges.append((data_q, check_q))    # Z: data → ancilla

        # 贪心图着色：返回不冲突的层
        layers = []
        used_qubits = set()  # 当前层已使用的比特

        for edge in edges:
            control, target = edge
            # 检查是否与当前层冲突
            if control in used_qubits or target in used_qubits:
                # 需要新层
                layers.append([])
                used_qubits = {control, target}
                layers[-1].append(edge)
            else:
                # 加入当前层
                if not layers:
                    layers.append([])
                layers[-1].append(edge)
                used_qubits.add(control)
                used_qubits.add(target)

        return layers

    # ---- 距离依赖噪声接口 ----

    def get_noise_scaling_factor(self, qubit1: int, qubit2: int) -> float:
        """基于距离返回噪声缩放因子（为距离依赖噪声预留）

        允许噪声模型根据比特间的空间距离调整错误率。
        这对于建模非均匀噪声或串扰噪声非常重要。

        Args:
            qubit1: 第一个比特 ID
            qubit2: 第二个比特 ID

        Returns:
            噪声缩放因子，1.0 表示无距离依赖

        Note:
            默认实现返回 1.0（无距离依赖）。
            子类可以覆盖以实现具体的距离-噪声关系，例如：
                - 指数衰减：exp(-distance / correlation_length)
                - 幂律：1 / (1 + distance^alpha)
        """
        return 1.0  # 默认无距离依赖
