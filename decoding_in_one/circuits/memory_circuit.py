# decoding_in_one/circuits/memory_circuit.py
"""
从 Ising-Decoding 迁移的重复测量电路构建器
源码参考: Ising-Decoding/code/qec/surface_code/memory_circuit.py
"""

from decoding_in_one.circuits.base import CircuitBuilder

class MemoryCircuit(CircuitBuilder):
    """
    表面码重复测量电路构建器

    Args:
        code: SurfaceCode 对象
        noise: 可选的噪声模型
    """

    def __init__(self, code, noise=None):
        self.code = code
        self.noise = noise

    def build_stabilizer_measurement(
        self,
        code,
        stabilizer_type: str,
        stabilizer_idx: int
    ) -> str:
        """
        构建单个稳定子测量电路

        简化实现：生成基本的 CNOT 结构
        """
        # 获取该稳定子连接的数据比特
        if stabilizer_type == 'X':
            # 使用实际的稳定子比特 ID
            xcheck_qubit = code._xcheck_qubits[stabilizer_idx]
            connections = code._x_connections.get(xcheck_qubit, [])
        else:
            zcheck_qubit = code._zcheck_qubits[stabilizer_idx]
            connections = code._z_connections.get(zcheck_qubit, [])

        circuit = f"# {stabilizer_type}-type stabilizer {stabilizer_idx}\n"

        # 对每个连接的数据比特执行 CNOT
        for data_qubit in connections:
            if stabilizer_type == 'Z':
                control = stabilizer_idx + len(code._data_qubits)
                target = data_qubit
            else:
                control = data_qubit
                target = stabilizer_idx + len(code._data_qubits)

            circuit += f"CX {control} {target}\n"

        return circuit

    def build_memory_circuit(
        self,
        code,
        n_rounds: int,
        measurement_basis: str
    ) -> str:
        """
        构建完整的重复测量电路

        Args:
            code: QuantumCode 对象
            n_rounds: 测量轮数
            measurement_basis: 测量基

        Returns:
            Stim 电路字符串
        """
        circuit = f"# Surface Code Memory Circuit\n"
        circuit += f"# Distance: {code.distance}, Rounds: {n_rounds}\n\n"

        # 重复测量轮
        circuit += f"REPEAT {n_rounds} {{\n"

        # X 型稳定子测量
        for i in range(len(code._xcheck_qubits)):
            circuit += self.build_stabilizer_measurement(code, 'X', i)

        # Z 型稳定子测量
        for i in range(len(code._zcheck_qubits)):
            circuit += self.build_stabilizer_measurement(code, 'Z', i)

        circuit += "}\n"

        # 最终数据比特测量
        circuit += "# Final data qubit measurements\n"
        for q in code._data_qubits:
            circuit += f"M {q}\n"

        # 应用噪声
        if self.noise:
            circuit = self.noise.apply_to_circuit(circuit)

        return circuit
