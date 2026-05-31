# decoding_in_one/sampling/dem.py
"""
DEM (Detector Error Model) 矩阵采样核心函数

源码参考: D:\\Develop\\Ising-Decoding\\code\\qec\\dem_sampling.py

采样通过 cuQuantum 的 cuStabilizer BitMatrixSampler 在 GPU 上运行，
可选 CuPy zero-copy DLPack 传输。需要 cuquantum>=26.3.0。

若无 cuquantum，回退到基于 numpy 的 CPU 采样。

提供三个核心函数：
- dem_sampling(): 从 DEM 矩阵 (H, p) 采样误差帧
- measure_from_stacked_frames(): 从堆叠帧提取测量值
- timelike_syndromes(): 应用 A 矩阵做时间方向校正
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np
import torch

# ---------- 可选 GPU 加速依赖 ----------

try:
    from cuquantum.stabilizer.dem_sampling import BitMatrixSampler
    from cuquantum.stabilizer.simulator import Options

    _CUSTAB_AVAILABLE = True
except ImportError:
    BitMatrixSampler = None  # type: ignore[misc, assignment]
    Options = None  # type: ignore[misc, assignment]
    _CUSTAB_AVAILABLE = False

try:
    import cupy as _cp  # noqa: F401

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def custab_available() -> bool:
    """返回 cuquantum.stabilizer 是否可用"""
    return _CUSTAB_AVAILABLE


# ---------- 模块级缓存 ----------

_cached_sampler = None
_cached_H: Optional[torch.Tensor] = None
_cached_HT: Optional[torch.Tensor] = None
_cached_max_shots: int = 0
_cached_device_id: Optional[int] = None
_cached_seed: Optional[int] = None

_DEM_TIMINGS_S: deque[float] = deque(maxlen=200)
_custab_path_logged: bool = False

_MIN_MAX_SHOTS = 1024


def get_dem_sampling_avg_ms() -> float:
    """最近 dem_sampling 调用的平均耗时（毫秒）"""
    if not _DEM_TIMINGS_S:
        return 0.0
    return (sum(_DEM_TIMINGS_S) / len(_DEM_TIMINGS_S)) * 1000.0


def _reset_sampler_cache() -> None:
    """重置模块级采样器缓存"""
    global _cached_sampler, _cached_H, _cached_HT, _cached_max_shots
    global _cached_device_id, _cached_seed
    _cached_sampler = None
    _cached_H = None
    _cached_HT = None
    _cached_max_shots = 0
    _cached_device_id = None
    _cached_seed = None


# ===================================================================
# 核心：dem_sampling
# ===================================================================


def _dem_sampling_cpu(
    H: torch.Tensor,
    p: torch.Tensor,
    batch_size: int,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    CPU fallback：基于 numpy 的 DEM 采样。

    对每个误差独立地按概率 p_i 采样是否触发，然后叠加到检测器帧上。
    速度远不如 GPU 路径，但无 cuquantum 依赖时可用。
    """
    rng = np.random.default_rng(seed)
    H_np = H.cpu().numpy()  # (2*num_detectors, num_errors)
    p_np = p.cpu().numpy()  # (num_errors,)

    # (batch_size, num_errors) 布尔掩码：每个误差是否触发
    triggers = rng.random((batch_size, p_np.shape[0])) < p_np[np.newaxis, :]

    # GF(2) 矩阵乘法: frames = triggers @ H^T (mod 2)
    frames = (triggers.astype(np.uint8) @ H_np.T.astype(np.uint8)) % 2

    return torch.as_tensor(frames, dtype=torch.uint8, device=H.device)


def dem_sampling(
    H: torch.Tensor,
    p: torch.Tensor,
    batch_size: int,
    device_id: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    从 DEM 矩阵采样误差帧。

    若 cuquantum 可用则使用 GPU BitMatrixSampler，否则回退到 CPU numpy 路径。

    Args:
        H: (2*num_detectors, num_errors) uint8 — 检测器-误差关联矩阵
        p: (num_errors,) float32 — 每个误差的概率
        batch_size: 采样数量
        device_id: GPU 设备 ID（cuST 路径）
        seed: RNG 种子（重复调用相同种子产生相同输出）

    Returns:
        frames_xz: (batch_size, 2*num_detectors) uint8 — 检测器输出
    """
    global _cached_sampler, _cached_H, _cached_HT, _cached_max_shots
    global _cached_device_id, _cached_seed, _custab_path_logged

    if H.ndim != 2:
        raise ValueError(f"H must be 2-D, got ndim={H.ndim}")
    if p.ndim != 1:
        raise ValueError(f"p must be 1-D, got ndim={p.ndim}")
    if H.shape[1] != p.shape[0]:
        raise ValueError(f"H has {H.shape[1]} columns but p has {p.shape[0]} entries")

    # ---- CPU fallback ----
    if not _CUSTAB_AVAILABLE:
        return _dem_sampling_cpu(H, p, batch_size, seed=seed)

    # ---- GPU 路径 (cuST BitMatrixSampler) ----
    from cuquantum.stabilizer.dem_sampling import BitMatrixSampler
    from cuquantum.stabilizer.simulator import Options

    if device_id is None:
        if H.is_cuda:
            device_index = H.device.index
            device_id = int(torch.cuda.current_device() if device_index is None else device_index)
        else:
            device_id = 0

    gpu_native = _CUPY_AVAILABLE and H.is_cuda

    if _cached_H is not H:
        _cached_HT = H.T
        _cached_H = H
        _cached_sampler = None
        _cached_device_id = None
        _cached_seed = None

    need_new = (
        _cached_sampler is None
        or batch_size > _cached_max_shots
        or _cached_device_id != device_id
        or seed is not None
    )

    if need_new:
        max_shots = max(batch_size, _MIN_MAX_SHOTS)
        if gpu_native:
            import cupy as cp

            with cp.cuda.Device(device_id):
                H_in = cp.from_dlpack(_cached_HT.detach())
                p_in = cp.from_dlpack(p.detach().to(torch.float64))
            pkg = "cupy"
        else:
            H_in = _cached_HT.detach().cpu().numpy().astype(np.uint8)
            p_in = p.detach().cpu().numpy().astype(np.float64)
            pkg = "numpy"
        bms_kwargs: dict = {"package": pkg, "options": Options(device_id=device_id)}
        if seed is not None:
            bms_kwargs["seed"] = seed
        _cached_sampler = BitMatrixSampler(H_in, p_in, max_shots, **bms_kwargs)
        _cached_max_shots = max_shots
        _cached_device_id = device_id
        _cached_seed = seed

    t0 = time.perf_counter()
    if gpu_native:
        import cupy as cp

        with cp.cuda.Device(device_id):
            _cached_sampler.sample(batch_size)
            out = _cached_sampler.get_outcomes(bit_packed=False)
    else:
        _cached_sampler.sample(batch_size)
        out = _cached_sampler.get_outcomes(bit_packed=False)
    if isinstance(out, np.ndarray):
        out = torch.as_tensor(out, device=H.device).to(dtype=torch.uint8)
    else:
        out = torch.from_dlpack(out).to(dtype=torch.uint8)
    _DEM_TIMINGS_S.append(time.perf_counter() - t0)

    if not _custab_path_logged:
        print(
            f"[dem_sampling] cuST BitMatrixSampler "
            f"(max_shots={_cached_max_shots}, gpu_native={gpu_native}, device_id={device_id})"
        )
        _custab_path_logged = True

    return out


# ===================================================================
# 辅助：measure_from_stacked_frames
# ===================================================================


def measure_from_stacked_frames(
    frames_xz: torch.Tensor,
    meas_qubits: torch.Tensor,
    meas_bases: torch.Tensor,
    nq: int,
) -> torch.Tensor:
    """
    从堆叠帧数据中提取测量结果。

    约定：Z 基测量读取 X 分量（反对易），X 基测量读取 Z 分量。

    Args:
        frames_xz: (batch_size, 2*num_detectors) uint8 — 堆叠 [X|Z] 检测器帧
        meas_qubits: (num_meas,) long — 测量比特索引
        meas_bases: (num_meas,) long — 基（0=X, 1=Z）
        nq: 总比特数

    Returns:
        meas_old: (batch_size, n_rounds, num_meas) uint8
    """
    meas_qubits = torch.as_tensor(meas_qubits, device=frames_xz.device, dtype=torch.long).reshape(-1)
    meas_bases = torch.as_tensor(meas_bases, device=frames_xz.device, dtype=torch.long).reshape(-1)
    D = frames_xz.shape[1] // 2
    R = D // int(nq)
    assert D == R * int(nq), f"Detector count {D} must be divisible by nq={nq}"

    idx = (torch.arange(R, device=frames_xz.device)[:, None] * int(nq) + meas_qubits[None, :]).reshape(-1)
    x = frames_xz[:, :D].index_select(1, idx).reshape(frames_xz.shape[0], R, -1)
    z = frames_xz[:, D:].index_select(1, idx).reshape(frames_xz.shape[0], R, -1)
    return torch.where(meas_bases[None, None, :] == 1, x, z).to(torch.uint8)


# ===================================================================
# 辅助：timelike_syndromes
# ===================================================================


def timelike_syndromes(
    frames_xz: torch.Tensor,
    A: torch.Tensor,
    meas_old: torch.Tensor,
) -> torch.Tensor:
    """
    应用 A 矩阵对测量做时间方向校正。

    A 是 GF(2) 上的线性映射，从 frames_xz 产生 s2；
    meas_new = s2 ^ meas_old。

    Args:
        frames_xz: (batch_size, 2*num_detectors) uint8
        A: (n_rounds*num_meas, 2*num_detectors) uint8
        meas_old: (batch_size, n_rounds, num_meas) uint8

    Returns:
        meas_new: (batch_size, n_rounds, num_meas) uint8
    """
    s2 = torch.remainder(frames_xz.float() @ A.t().float(), 2).to(torch.uint8)
    return (s2 ^ meas_old.reshape(meas_old.shape[0], -1)).reshape_as(meas_old)
