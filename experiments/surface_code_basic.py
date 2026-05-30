# experiments/surface_code_basic.py
"""
表面码基础实验示例

演示如何使用 DecoderInOne 框架进行简单的解码实验
"""

from decoding_in_one.codes import SurfaceCode
from decoding_in_one.noise import CircuitLevelNoise
from decoding_in_one.circuits import MemoryCircuit

def main():
    print("=" * 60)
    print("Surface Code Basic Experiment")
    print("=" * 60)

    # 1. 定义码
    print("\n[Step 1] Defining Surface Code...")
    code = SurfaceCode(distance=5, rotation='XV')
    print(f"  Physical qubits: {code.get_n_physical()}")
    print(f"  Logical qubits: {code.get_n_logical()}")

    # 2. 加载噪声模型
    print("\n[Step 2] Loading noise model...")
    noise = CircuitLevelNoise.from_config('configs/noise_25p.yaml')
    print(f"  Parameters loaded: {len(noise.get_parameters())}")
    print(f"  Valid: {noise.validate()}")

    # 3. 构建电路
    print("\n[Step 3] Building memory circuit...")
    builder = MemoryCircuit(code=code, noise=noise)
    circuit = builder.build_memory_circuit(
        code=code,
        n_rounds=5,
        measurement_basis='X'
    )
    print(f"  Circuit length: {len(circuit)} characters")

    # 保存电路到文件
    output_path = 'experiments/output_circuit.stim'
    with open(output_path, 'w') as f:
        f.write(circuit)
    print(f"  Circuit saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()
