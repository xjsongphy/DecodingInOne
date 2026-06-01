#!/usr/bin/env python3
"""测试并行采样是否正确工作"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from decoding_in_one.sampling.dem import (
    dem_sampling,
    dem_sampling_parallel,
    custab_available,
    _CUSTAB_AVAILABLE
)

print(f"=== 环境检查 ===")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
print(f"cuquantum available: {_CUSTAB_AVAILABLE}")
print(f"custab_available() function: {custab_available()}")
print()

# 创建小的测试数据
n_det = 100
n_errors = 50
H = torch.randint(0, 2, (2 * n_det, n_errors), dtype=torch.uint8)
p = torch.rand(n_errors, dtype=torch.float32) * 0.01

print(f"=== 测试数据 ===")
print(f"H shape: {H.shape}")
print(f"p shape: {p.shape}")
print()

# 测试单次采样
print(f"=== 测试 1: 单次采样（batch_size=500） ===")
try:
    result1 = dem_sampling(H, p, batch_size=500, seed=42)
    print(f"✓ 单次采样成功，输出 shape: {result1.shape}")
except Exception as e:
    print(f"✗ 单次采样失败: {e}")
print()

# 测试并行采样（单 GPU 场景）
print(f"=== 测试 2: 并行采样（batch_size=20000, num_workers=4） ===")
try:
    result2 = dem_sampling_parallel(H, p, batch_size=20000, num_workers=4, seed=42)
    print(f"✓ 并行采样成功，输出 shape: {result2.shape}")
except Exception as e:
    print(f"✗ 并行采样失败: {e}")
print()

# 测试不同的 worker 数量
print(f"=== 测试 3: 不同 workers 数量 ===")
for workers in [1, 2, 4, 8]:
    try:
        result = dem_sampling_parallel(H, p, batch_size=10000, num_workers=workers, seed=42)
        print(f"  workers={workers}: ✓ shape={result.shape}")
    except Exception as e:
        print(f"  workers={workers}: ✗ {e}")
print()

print(f"=== 测试完成 ===")
