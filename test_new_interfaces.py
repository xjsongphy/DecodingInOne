#!/usr/bin/env python3
"""测试新的通用性接口"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from decoding_in_one.codes import SurfaceCode


def test_surface_code_new_interfaces():
    """测试 SurfaceCode 的新接口实现"""
    print("=" * 60)
    print("测试 SurfaceCode 新接口")
    print("=" * 60)

    # 创建一个距离为 3 的表面码
    code = SurfaceCode(distance=3, rotation='XV')

    # 1. 测试基础接口
    print(f"\n[基础信息]")
    print(f"  距离: {code.distance}")
    print(f"  旋转: {code.rotation}")
    print(f"  物理比特数: {code.get_n_physical()}")
    print(f"  逻辑比特数: {code.get_n_logical()}")

    # 2. 测试 H 矩阵接口
    print(f"\n[H 矩阵接口]")
    h_matrices = code.get_parity_check_matrices()
    print(f"  H_X 形状: {h_matrices['H_X'].shape}")
    print(f"  H_Z 形状: {h_matrices['H_Z'].shape}")
    print(f"  H_X 非零元素: {h_matrices['H_X'].sum()}")
    print(f"  H_Z 非零元素: {h_matrices['H_Z'].sum()}")

    # 3. 测试图结构接口
    print(f"\n[图结构接口]")
    graph = code.get_stabilizer_graph()
    print(f"  X 边数: {len(graph['X_edges'])}")
    print(f"  Z 边数: {len(graph['Z_edges'])}")
    print(f"  X 边样例: {graph['X_edges'][:2]}")
    print(f"  Z 边样例: {graph['Z_edges'][:2]}")

    # 4. 测试距离接口
    print(f"\n[空间距离接口]")
    coords = code.get_qubit_coordinates()
    print(f"  总比特数: {len(coords)}")
    print(f"  数据比特样例 (qubit 0): {coords[0]}")

    # 测试几个比特间的距离
    data_qubits = code.get_data_qubits()[:3]
    for i in range(len(data_qubits) - 1):
        dist = code.get_spatial_distance(data_qubits[i], data_qubits[i + 1])
        print(f"  距离({data_qubits[i]}, {data_qubits[i + 1]}): {dist:.2f}")

    # 5. 测试噪声缩放接口
    print(f"\n[距离依赖噪声接口]")
    for i in range(len(data_qubits) - 1):
        scaling = code.get_noise_scaling_factor(data_qubits[i], data_qubits[i + 1])
        print(f"  噪声缩放({data_qubits[i]}, {data_qubits[i + 1]}): {scaling:.2f}")

    # 6. 测试 CNOT 层接口（表面码优化的四层）
    print(f"\n[CNOT 层接口]")
    x_layers = code.get_stabilizer_measurement_layers('X')
    z_layers = code.get_stabilizer_measurement_layers('Z')
    print(f"  X 稳定子 CNOT 层数: {len(x_layers)}")
    print(f"  Z 稳定子 CNOT 层数: {len(z_layers)}")
    print(f"  X 第 0 层边数: {len(x_layers[0])}")
    print(f"  X 第 0 层样例: {x_layers[0][:2]}")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)


def test_generic_code_interface():
    """测试通用接口对 QLDPC 的支持"""
    print("\n" + "=" * 60)
    print("测试通用 QLDPC 接口（模拟）")
    print("=" * 60)

    # 这里我们使用 SurfaceCode 来验证接口的通用性
    # 对于真正的 QLDPC，需要实现 QLDPCCode 类
    code = SurfaceCode(distance=5, rotation='XV')

    print(f"\n[通用接口验证]")
    print(f"  H 矩阵: {code.get_parity_check_matrices()['H_X'].shape}, {code.get_parity_check_matrices()['H_Z'].shape}")
    print(f"  图边数: X={len(code.get_stabilizer_graph()['X_edges'])}, Z={len(code.get_stabilizer_graph()['Z_edges'])}")
    print(f"  空间坐标维度: {list(code.get_qubit_coordinates().values())[0].dim}D")
    print(f"  CNOT 层数: X={len(code.get_stabilizer_measurement_layers('X'))}, Z={len(code.get_stabilizer_measurement_layers('Z'))}")

    print("\n" + "=" * 60)
    print("通用接口验证通过！")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_surface_code_new_interfaces()
        test_generic_code_interface()
        print("\n✅ 所有测试成功完成！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
