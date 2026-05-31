# decoding_in_one/sampling/generator.py
"""
通用电路数据生成器（迁移自 Ising-Decoding MemoryCircuitTorch）

源码参考: D:\\Develop\\Ising-Decoding\\code\\qec\\surface_code\\memory_circuit_torch.py

核心类 CircuitDataGenerator:
- 从 Stim 电路自动提取 DEM 矩阵 (H, p)，或从预计算 .npz 加载
- 用 GPU/CPU 采样 (dem_sampling)
- 用 Ising-Decoding 的 _format_for_model 逻辑生成 (trainX, trainY)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import stim
import torch

from decoding_in_one.sampling.dem import dem_sampling
from decoding_in_one.surface_code.data_mapping import (
    normalized_weight_mapping_Xstab_memory,
    normalized_weight_mapping_Zstab_memory,
)


def _npz1(path: Path):
    """从 .npz 文件加载第一个非标量数组"""
    z = np.load(path)
    for k in ("p", "arr_0"):
        if k in z.files:
            return z[k]
    for k in z.files:
        a = z[k]
        if getattr(a, "ndim", 0) > 0 and a.size > 1:
            return a
    return z[z.files[0]]


# ===================================================================
# 从 Stim 电路提取 DEM 矩阵
# ===================================================================

def extract_dem_from_stim_circuit(
    circuit: stim.Circuit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 Stim 电路提取检测器-误差关联矩阵 H、可观测量矩阵 Hz、概率向量 p。

    Args:
        circuit: stim.Circuit 对象

    Returns:
        H: (n_detectors, n_errors) uint8 — 检测器-误差关联
        Hz: (n_observables, n_errors) uint8 — 观测量-误差关联
        p: (n_errors,) float32 — 每个误差的概率
    """
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    n_det = dem.num_detectors
    n_obs = dem.num_observables

    error_re = re.compile(r"error\(([^)]+)\)\s+(.*)")
    det_re = re.compile(r"D(\d+)")
    obs_re = re.compile(r"L(\d+)")

    H_rows, Hz_rows, p_list = [], [], []

    for line in str(dem).strip().split("\n"):
        line = line.strip()
        m = error_re.match(line)
        if not m:
            continue
        prob = float(m.group(1))
        targets_str = m.group(2)

        dets = [int(x) for x in det_re.findall(targets_str)]
        obs = [int(x) for x in obs_re.findall(targets_str)]

        det_row = np.zeros(n_det, dtype=np.uint8)
        for d in dets:
            det_row[d] = 1

        obs_row = np.zeros(n_obs, dtype=np.uint8)
        for o in obs:
            obs_row[o] = 1

        H_rows.append(det_row)
        Hz_rows.append(obs_row)
        p_list.append(prob)

    H = np.array(H_rows, dtype=np.uint8).T   # (n_det, n_err)
    Hz = np.array(Hz_rows, dtype=np.uint8).T  # (n_obs, n_err)
    p = np.array(p_list, dtype=np.float32)
    return H, Hz, p


# ===================================================================
# CircuitDataGenerator
# ===================================================================


class CircuitDataGenerator:
    """
    通用电路数据生成器（Ising-Decoding 方式）

    数据来源优先级:
    1. stim_circuit — 从 Stim 电路自动提取 DEM 矩阵
    2. precomputed_frames_dir — 从 .npz 文件加载预计算矩阵
    3. 内存中的 (H, p) 张量

    Args:
        distance: 码距
        n_rounds: 测量轮数
        basis: 测量基 ('X' 或 'Z')
        code_rotation: 电路方向 ('XV', 'XH', 'ZV', 'ZH')
        stim_circuit: Stim 电路对象（自动提取 DEM）
        precomputed_frames_dir: 预计算 .npz 文件目录
        H/p/A: 内存中的 DEM 矩阵
        p_override: 覆盖概率向量
        device: 计算设备
    """

    def __init__(
        self,
        *,
        distance: int,
        n_rounds: int,
        basis: str,
        code_rotation: str = "XV",
        # DEM 来源（三选一）
        stim_circuit: Optional[stim.Circuit] = None,
        allow_stim_dem_extraction: bool = False,
        precomputed_frames_dir: Optional[str] = None,
        H: Optional[torch.Tensor] = None,
        p: Optional[torch.Tensor] = None,
        A: Optional[torch.Tensor] = None,
        p_override: Optional[Union[torch.Tensor, np.ndarray]] = None,
        # 设备
        device: Optional[torch.device] = None,
    ):
        self.distance = int(distance)
        self.n_rounds = int(n_rounds)
        self.basis = str(basis).upper()
        self.code_rotation = str(code_rotation).upper()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- 加载 DEM 矩阵 ----------
        if H is not None and p is not None:
            # 内存张量
            self.H = H.to(device=self.device, dtype=torch.uint8)
            self.p = p.to(device=self.device, dtype=torch.float32)
            self.A = A.to(device=self.device, dtype=torch.uint8) if A is not None else None
        elif stim_circuit is not None and allow_stim_dem_extraction:
            # 从 Stim 电路自动提取
            self.H, self.Hz, self.p, self.A = self._extract_from_stim(stim_circuit)
        elif precomputed_frames_dir is not None:
            # 从预计算文件加载
            self.H, self.p, self.A = self._load_precomputed(precomputed_frames_dir, p_override)
            self.Hz = None
        else:
            raise ValueError(
                "Provide one of: precomputed_frames_dir or in-memory H/p. "
                "Stim DEM extraction is disabled by default."
            )

        # ---------- 权重映射（trainX presence 通道）----------
        self.w_mapXgrid = (
            normalized_weight_mapping_Xstab_memory(self.distance, rotation=self.code_rotation)
            .reshape(self.distance, self.distance)
            .to(self.device)
        )
        self.w_mapZgrid = (
            normalized_weight_mapping_Zstab_memory(self.distance, rotation=self.code_rotation)
            .reshape(self.distance, self.distance)
            .to(self.device)
        )

    # ------------------------------------------------------------------
    # DEM 来源
    # ------------------------------------------------------------------

    def _extract_from_stim(
        self, circuit: stim.Circuit
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """从 Stim 电路提取 DEM 并转为 torch 张量"""
        H_np, Hz_np, p_np = extract_dem_from_stim_circuit(circuit)
        H = torch.from_numpy(H_np).to(self.device, dtype=torch.uint8)
        Hz = torch.from_numpy(Hz_np).to(self.device, dtype=torch.uint8)
        p = torch.from_numpy(p_np).to(self.device, dtype=torch.float32)
        return H, Hz, p, None  # A 不从 Stim 自动提取

    def _load_precomputed(
        self, frames_dir: str, p_override=None
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """从 .npz 文件加载预计算的 H, p, A"""
        d = Path(frames_dir)
        prefix = f"surface_d{self.distance}_r{self.n_rounds}_{self.basis}_frame_predecoder"

        hx_path = d / f"{prefix}.X.npz"
        hz_path = d / f"{prefix}.Z.npz"
        p_path = d / f"{prefix}.p.npz"
        a_path = d / f"{prefix}.A.npz"

        if not (hx_path.exists() and hz_path.exists() and p_path.exists()):
            raise FileNotFoundError(
                f"Missing DEM artifacts in {d!r}. Expected:\n"
                f"  {hx_path.name}, {hz_path.name}, {p_path.name}"
            )

        hx = np.asarray(_npz1(hx_path), dtype=np.uint8)
        hz = np.asarray(_npz1(hz_path), dtype=np.uint8)
        p_arr = np.asarray(_npz1(p_path)).reshape(-1)

        if p_override is not None:
            if isinstance(p_override, torch.Tensor):
                p_arr = p_override.detach().cpu().numpy().reshape(-1)
            else:
                p_arr = np.asarray(p_override).reshape(-1)

        errors = p_arr.shape[0]
        hx = hx if hx.shape[1] == errors else hx.T
        hz = hz if hz.shape[1] == errors else hz.T

        H = torch.from_numpy(np.concatenate([hx, hz], axis=0)).to(self.device, dtype=torch.uint8)
        p_tensor = torch.from_numpy(p_arr).to(self.device, dtype=torch.float32)

        A = None
        if a_path.exists():
            A = torch.from_numpy(np.asarray(_npz1(a_path), dtype=np.uint8)).to(
                self.device, dtype=torch.uint8
            )

        return H, p_tensor, A

    # ------------------------------------------------------------------
    # 数据生成（Ising-Decoding 完整流程）
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        batch_size: int,
        seed: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """
        生成一个批次的训练数据（Ising-Decoding 方式）。

        流程:
        1. dem_sampling(H, p) → frames_xz (B, 2*n_det)
        2. 提取累积数据比特帧 → x_cum, z_cum
        3. 差分 → x_diff, z_diff
        4. _format_for_model → trainX (B,4,T,D,D), trainY (B,4,T,D,D)

        Args:
            batch_size: 批次大小
            seed: 可选随机种子

        Returns:
            dict with:
            - 'trainX': (B, 4, T, D, D) — [x_syn, z_syn, x_pres, z_pres]
            - 'trainY': (B, 4, T, D, D) — [z_err, x_err, s1x, s1z]
        """
        device_id = None
        if self.device.type == "cuda":
            device_index = self.device.index
            device_id = int(torch.cuda.current_device() if device_index is None else device_index)

        # 1. 采样误差帧
        frames_xz = dem_sampling(
            self.H, self.p, batch_size, device_id=device_id, seed=seed
        )  # (B, 2*n_det)

        # 2. 提取累积数据比特帧
        # frames_xz = [X_block | Z_block]，每个 block 有 (n_rounds * n_qubits) 列
        D = frames_xz.shape[1] // 2  # n_det（半边）
        nq = D // self.n_rounds      # 总比特数
        R = self.n_rounds
        DD = self.distance

        # 数据比特索引（按可用位数截断；不足时后续补零到 distance²）
        n_data = min(DD * DD, nq)
        data_idx = (
            torch.arange(R, device=self.device)[:, None] * nq
            + torch.arange(n_data, device=self.device)[None, :]
        ).reshape(-1)

        x_cum = frames_xz[:, :D].index_select(1, data_idx).reshape(batch_size, R, n_data)
        z_cum = frames_xz[:, D:].index_select(1, data_idx).reshape(batch_size, R, n_data)

        if n_data < DD * DD:
            pad = DD * DD - n_data
            x_cum = torch.nn.functional.pad(x_cum, (0, pad))
            z_cum = torch.nn.functional.pad(z_cum, (0, pad))

        # 3. 差分（当前帧 XOR 前一帧）
        xpad = torch.cat([torch.zeros_like(x_cum[:, :1, :]), x_cum], dim=1)
        zpad = torch.cat([torch.zeros_like(z_cum[:, :1, :]), z_cum], dim=1)
        x_diff = xpad[:, :-1, :] ^ xpad[:, 1:, :]  # (B, R, D²)
        z_diff = zpad[:, :-1, :] ^ zpad[:, 1:, :]

        # 4. Ising-Decoding 格式化
        trainX, trainY = self._format_for_model(x_diff, z_diff)

        return {"trainX": trainX, "trainY": trainY}

    def _format_for_model(
        self, x_diff: torch.Tensor, z_diff: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ising-Decoding 的 _format_for_model 逻辑

        trainX: (B, 4, R, D, D) = [x_syn_grid, z_syn_grid, x_pres, z_pres]
        trainY: (B, 4, R, D, D) = [z_err, x_err, s1x_grid, s1z_grid]

        其中:
        - x/z_syn_grid: 差分错误映射到 D×D 网格（stabilizer 空间）
        - x/z_pres: 归一化权重映射（presence 通道）
        - x/z_err: 差分错误 reshape 到网格
        - s1x/s1z: syndrome 占位（无 HE 时同 x/z_err）

        Args:
            x_diff: (B, R, D²) X 差分
            z_diff: (B, R, D²) Z 差分
        """
        B, R, D2 = x_diff.shape
        D = self.distance

        # 差分 reshape 到 D×D 网格
        x_err = x_diff.reshape(B, R, D, D)
        z_err = z_diff.reshape(B, R, D, D)
        # 注意：x_diff 来自 X-frame 差分，直接 reshape 即可

        # 权重映射（presence 通道）
        x_pres = self.w_mapXgrid.unsqueeze(0).unsqueeze(0).expand(B, R, D, D).clone()
        z_pres = self.w_mapZgrid.unsqueeze(0).unsqueeze(0).expand(B, R, D, D).clone()

        # basis 掩码（与 Ising-Decoding 一致）
        if self.basis == "X":
            z_pres[:, 0] = 0
            z_pres[:, -1] = 0
        else:
            x_pres[:, 0] = 0
            x_pres[:, -1] = 0

        # trainX: [x_syn, z_syn, x_pres, z_pres]
        # 简化版: x_syn = x_err（差分即 syndrome 近似）
        trainX = torch.stack(
            [x_err.float(), z_err.float(), x_pres.float(), z_pres.float()], dim=1
        ).contiguous()

        # trainY: [z_err, x_err, s1x, s1z]
        # 与 Ising-Decoding 一致：z_err 和 x_err 交换位置，s1x/s1z 用 x/z_err 近似
        trainY = torch.stack(
            [z_err.float(), x_err.float(), x_err.float(), z_err.float()], dim=1
        ).contiguous()

        return trainX, trainY
