# decoding_in_one/codes/surface_code.py
"""
从 Ising-Decoding 迁移的表面码实现
源码参考: Ising-Decoding/code/qec/surface_code/memory_circuit.py
"""

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
        rotation = rotation.upper()
        if rotation not in ("XV", "XH", "ZV", "ZH"):
            raise ValueError("rotation must be one of: XV, XH, ZV, ZH")

        self.distance = distance
        self.rotation = rotation

        # 构建码结构
        self._build_code()

    def _build_code(self):
        """构建表面码结构。"""
        n_data = self.distance ** 2
        self._data_qubits = list(range(n_data))
        self._data_coords = {
            idx: (idx // self.distance, idx % self.distance) for idx in self._data_qubits
        }

        x_supports: List[Tuple[int, ...]] = []
        z_supports: List[Tuple[int, ...]] = []
        boundary_supports: List[Tuple[int, ...]] = []

        x_prefer_even = self.rotation in ("XV", "ZH")
        keep_even_boundary = self.rotation in ("XV", "XH")

        max_coord = 2 * self.distance
        for x in range(0, max_coord + 1, 2):
            for y in range(0, max_coord + 1, 2):
                support = self._support_from_check_coord(x, y)
                if len(support) not in (2, 4):
                    continue

                parity = ((x // 2) + (y // 2)) % 2
                if len(support) == 2:
                    if (parity == 0) != keep_even_boundary:
                        continue
                    boundary_supports.append(support)
                    continue

                is_x = parity == 0 if x_prefer_even else parity == 1
                if is_x:
                    x_supports.append(support)
                else:
                    z_supports.append(support)

        boundary_supports = sorted(boundary_supports, key=lambda s: s[0])
        boundary_x_first = self.rotation[0] == "X"
        for i, support in enumerate(boundary_supports):
            if (i % 2 == 0) == boundary_x_first:
                x_supports.append(support)
            else:
                z_supports.append(support)

        expected = (self.distance ** 2 - 1) // 2
        if len(x_supports) != expected or len(z_supports) != expected:
            raise RuntimeError(
                f"Invalid stabilizer counts for distance={self.distance}, rotation={self.rotation}: "
                f"X={len(x_supports)}, Z={len(z_supports)}, expected={expected}"
            )

        x_supports = sorted(x_supports, key=lambda s: s[0])
        z_supports = sorted(z_supports, key=lambda s: s[0])

        self._xcheck_qubits = list(range(n_data, n_data + len(x_supports)))
        self._zcheck_qubits = list(range(
            n_data + len(x_supports),
            n_data + len(x_supports) + len(z_supports),
        ))
        self._x_connections = {
            check: list(support) for check, support in zip(self._xcheck_qubits, x_supports)
        }
        self._z_connections = {
            check: list(support) for check, support in zip(self._zcheck_qubits, z_supports)
        }

    def _support_from_check_coord(self, x: int, y: int) -> Tuple[int, ...]:
        """根据格点坐标生成该稳定子的支撑数据比特。"""
        support: List[int] = []
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            nx, ny = x + dx, y + dy
            if 1 <= nx <= 2 * self.distance - 1 and 1 <= ny <= 2 * self.distance - 1:
                if nx % 2 == 1 and ny % 2 == 1:
                    r = (nx - 1) // 2
                    c = (ny - 1) // 2
                    support.append(r * self.distance + c)
        return tuple(sorted(support))

    def get_n_physical(self) -> int:
        return len(self._data_qubits)

    def get_n_logical(self) -> int:
        return 1

    def get_stabilizers(self) -> List[PauliString]:
        """返回所有稳定子生成元。"""
        stabilizers = []
        n = self.get_n_physical()

        for support in self._x_connections.values():
            ops = ["I"] * n
            for q in support:
                ops[q] = "X"
            stabilizers.append(PauliString("".join(ops)))

        for support in self._z_connections.values():
            ops = ["I"] * n
            for q in support:
                ops[q] = "Z"
            stabilizers.append(PauliString("".join(ops)))

        return stabilizers

    def get_logical_operators(self) -> Dict[str, PauliString]:
        """返回一组标准逻辑算符。"""
        n = self.get_n_physical()
        ops_x = ["I"] * n
        ops_z = ["I"] * n
        if self.rotation in ("XV", "ZH"):
            line = [r * self.distance for r in range(self.distance)]
            cross = [c for c in range(self.distance)]
        else:
            line = [c for c in range(self.distance)]
            cross = [r * self.distance for r in range(self.distance)]
        for q in line:
            ops_x[q] = "X"
        for q in cross:
            ops_z[q] = "Z"
        return {
            'X': PauliString("".join(ops_x)),
            'Z': PauliString("".join(ops_z)),
        }

    def get_qubit_topology(self) -> Dict[int, Tuple[int, int]]:
        """返回数据比特到 2D 坐标 (row, col) 的映射。"""
        return dict(self._data_coords)

    def get_data_qubits(self) -> List[int]:
        return list(self._data_qubits)

    def get_check_qubits(self, stabilizer_type: str) -> List[int]:
        stabilizer_type = stabilizer_type.upper()
        if stabilizer_type == "X":
            return list(self._xcheck_qubits)
        if stabilizer_type == "Z":
            return list(self._zcheck_qubits)
        raise ValueError("stabilizer_type must be 'X' or 'Z'")

    def get_stabilizer_supports(self, stabilizer_type: str) -> Dict[int, Tuple[int, ...]]:
        stabilizer_type = stabilizer_type.upper()
        if stabilizer_type == "X":
            return {k: tuple(v) for k, v in self._x_connections.items()}
        if stabilizer_type == "Z":
            return {k: tuple(v) for k, v in self._z_connections.items()}
        raise ValueError("stabilizer_type must be 'X' or 'Z'")
