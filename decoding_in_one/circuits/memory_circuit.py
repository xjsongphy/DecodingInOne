# decoding_in_one/circuits/memory_circuit.py
"""
从 Ising-Decoding 迁移的重复测量电路构建器
源码参考: Ising-Decoding/code/qec/surface_code/memory_circuit.py
"""

from decoding_in_one.circuits.base import CircuitBuilder
from decoding_in_one.utils.types import CircuitArtifact, CircuitSpec, CodeSpec

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
            check_qubit = code.get_check_qubits('X')[stabilizer_idx]
            connections = code.get_stabilizer_supports('X').get(check_qubit, [])
        else:
            check_qubit = code.get_check_qubits('Z')[stabilizer_idx]
            connections = code.get_stabilizer_supports('Z').get(check_qubit, [])

        circuit = f"# {stabilizer_type}-type stabilizer {stabilizer_idx}\n"

        # 对每个连接的数据比特执行 CNOT
        for data_qubit in connections:
            if stabilizer_type == 'Z':
                control = check_qubit
                target = data_qubit
            else:
                control = data_qubit
                target = check_qubit

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
        artifact = self.build_memory_artifact(
            code=code,
            spec=CircuitSpec(n_rounds=n_rounds, measurement_basis=measurement_basis),
        )
        return artifact.stim_circuit

    def build_memory_artifact(
        self,
        code,
        spec: CircuitSpec
    ) -> CircuitArtifact:
        """构建结构化电路对象。"""
        circuit = f"# Surface Code Memory Circuit\n"
        circuit += f"# Distance: {code.distance}, Rounds: {spec.n_rounds}\n\n"

        # 重复测量轮
        circuit += f"REPEAT {spec.n_rounds} {{\n"

        # X 型稳定子测量
        for i in range(len(code.get_check_qubits('X'))):
            circuit += self.build_stabilizer_measurement(code, 'X', i)

        # Z 型稳定子测量
        for i in range(len(code.get_check_qubits('Z'))):
            circuit += self.build_stabilizer_measurement(code, 'Z', i)

        circuit += "}\n"

        # 最终数据比特测量
        circuit += "# Final data qubit measurements\n"
        for q in code.get_data_qubits():
            circuit += f"M {q}\n"

        artifact = CircuitArtifact(
            stim_circuit=circuit,
            code=CodeSpec(
                code_family=code.__class__.__name__,
                distance=code.distance,
                rotation=getattr(code, "rotation", "XV"),
                n_physical=code.get_n_physical(),
                n_logical=code.get_n_logical(),
            ),
            spec=spec,
            metadata={},
        )
        if self.noise:
            noisy = self.noise.apply_to_circuit(artifact)
            if isinstance(noisy, CircuitArtifact):
                return noisy
            artifact.stim_circuit = noisy
        return artifact
