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
from typing import Dict, List, Mapping, Optional, Union

import numpy as np
import stim
import torch

from decoding_in_one.codes import SurfaceCode
from decoding_in_one.sampling.dem import (
    dem_sampling,
    dem_sampling_parallel,
    measure_from_stacked_frames,
    timelike_syndromes,
)
from decoding_in_one.codes.surface_code.data_mapping import (
    normalized_weight_mapping_Xstab_memory,
    normalized_weight_mapping_Zstab_memory,
    reshape_Xstabilizers_to_grid_vectorized,
    reshape_Zstabilizers_to_grid_vectorized,
)
from decoding_in_one.codes.surface_code.homological_equivalence_torch import (
    apply_weight1_timelike_homological_equivalence_torch,
    build_spacelike_he_cache,
    build_timelike_he_cache,
    build_weight2_timelike_cache,
    warmup_he_compile,
)

# tqdm 进度条（可选依赖）
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None  # type: ignore
    TQDM_AVAILABLE = False


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

    dem_str = str(dem).strip().split("\n")
    n_errors = len(dem_str)

    # 显示进度条（DEM 提取可能很慢）
    if TQDM_AVAILABLE and n_errors > 100:
        print(f"[DEM] Extracting {n_errors} error mechanisms from Stim circuit...")
        iterator = tqdm(dem_str, desc="Extracting DEM", total=n_errors, unit="errors")
    else:
        iterator = dem_str
        if n_errors > 0:
            print(f"[DEM] Extracting {n_errors} error mechanisms...")

    for line in iterator:
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
        # 并行采样配置
        enable_parallel: bool = False,
        num_workers: int = 4,
        device_ids: Optional[List[int]] = None,
        # HE configuration
        timelike_he: bool = True,
        num_he_cycles: int = 1,
        max_passes_w1: int = 32,
        use_compile: bool = False,
        compile_chunk_size: int = 2,
        compute_dtype: Optional[torch.dtype] = None,
        use_weight2: bool = False,
        max_passes_w2: int = 4,
        use_coset_search: bool = False,
        coset_max_generators: int = 20,
        use_dense_overlap: bool = False,
        use_parallel_spacelike: bool = False,
        # 设备
        device: Optional[torch.device] = None,
    ):
        self.distance = int(distance)
        self.n_rounds = int(n_rounds)
        self.basis = str(basis).upper()
        self.code_rotation = str(code_rotation).upper()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mixed = self.basis in ("BOTH", "MIXED")
        self._mixed_generators: Dict[str, "CircuitDataGenerator"] = {}
        self._mixed_call_count = 0

        if self._mixed:
            if H is not None or p is not None or A is not None or p_override is not None:
                raise ValueError("Mixed-basis generator does not accept direct H/p/A inputs")
            if stim_circuit is not None and not isinstance(stim_circuit, Mapping):
                raise ValueError(
                    "Mixed-basis generator expects stim_circuit to be a mapping with X/Z entries"
                )
            stim_circuits = dict(stim_circuit or {})
            common_kwargs = dict(
                distance=self.distance,
                n_rounds=self.n_rounds,
                code_rotation=self.code_rotation,
                allow_stim_dem_extraction=allow_stim_dem_extraction,
                precomputed_frames_dir=precomputed_frames_dir,
                enable_parallel=enable_parallel,
                num_workers=num_workers,
                device_ids=device_ids,
                timelike_he=timelike_he,
                num_he_cycles=num_he_cycles,
                max_passes_w1=max_passes_w1,
                use_compile=use_compile,
                compile_chunk_size=compile_chunk_size,
                compute_dtype=compute_dtype,
                use_weight2=use_weight2,
                max_passes_w2=max_passes_w2,
                use_coset_search=use_coset_search,
                coset_max_generators=coset_max_generators,
                use_dense_overlap=use_dense_overlap,
                use_parallel_spacelike=use_parallel_spacelike,
                device=self.device,
            )
            self._mixed_generators["X"] = CircuitDataGenerator(
                basis="X",
                stim_circuit=stim_circuits.get("X"),
                **common_kwargs,
            )
            self._mixed_generators["Z"] = CircuitDataGenerator(
                basis="Z",
                stim_circuit=stim_circuits.get("Z"),
                **common_kwargs,
            )
            return

        self.code = SurfaceCode(self.distance, rotation=self.code_rotation)
        self.data_qubits = torch.as_tensor(
            self.code.get_data_qubits(), dtype=torch.long, device=self.device
        )
        self.xcheck_qubits = torch.as_tensor(
            self.code.get_check_qubits("X"), dtype=torch.long, device=self.device
        )
        self.zcheck_qubits = torch.as_tensor(
            self.code.get_check_qubits("Z"), dtype=torch.long, device=self.device
        )
        self.nq = int(
            len(self.code.get_data_qubits()) +
            len(self.code.get_check_qubits("X")) +
            len(self.code.get_check_qubits("Z"))
        )
        self.meas_qubits = torch.cat([self.xcheck_qubits, self.zcheck_qubits], dim=0)
        self.meas_bases = torch.cat(
            [
                torch.zeros(len(self.xcheck_qubits), dtype=torch.long, device=self.device),
                torch.ones(len(self.zcheck_qubits), dtype=torch.long, device=self.device),
            ],
            dim=0,
        )

        # 并行采样配置
        self.enable_parallel = enable_parallel
        self.num_workers = num_workers
        self.device_ids = device_ids
        self.timelike_he = bool(timelike_he)
        self.num_he_cycles = int(num_he_cycles)
        self.max_passes_w1 = int(max_passes_w1)
        self.use_compile = bool(use_compile)
        self.compile_chunk_size = int(compile_chunk_size)
        self.compute_dtype = compute_dtype
        self.use_weight2 = bool(use_weight2)
        self.max_passes_w2 = int(max_passes_w2)
        self.use_coset_search = bool(use_coset_search)
        self.coset_max_generators = int(coset_max_generators)
        self.use_dense_overlap = bool(use_dense_overlap)
        self.use_parallel_spacelike = bool(use_parallel_spacelike)

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
        self._frame_row_count = int(2 * self.n_rounds * self.nq)
        self._has_frame_predecoder_semantics = int(self.H.shape[0]) == self._frame_row_count
        self._fallback_warning_emitted = False
        self._compile_thread = None

        self.parity_X = torch.tensor(self.code.hx, dtype=torch.uint8, device=self.device)
        self.parity_Z = torch.tensor(self.code.hz, dtype=torch.uint8, device=self.device)
        self.cache_X_sp = None
        self.cache_Z_sp = None
        self.cache_X_tl = None
        self.cache_Z_tl = None
        self.cache_X_w2 = None
        self.cache_Z_w2 = None

        if self._has_frame_predecoder_semantics and self.timelike_he:
            self.cache_X_sp = build_spacelike_he_cache(
                self.parity_X, distance=self.distance, basis="X", device=self.device
            )
            self.cache_Z_sp = build_spacelike_he_cache(
                self.parity_Z, distance=self.distance, basis="Z", device=self.device
            )
            self.cache_X_tl = build_timelike_he_cache(self.parity_X)
            self.cache_Z_tl = build_timelike_he_cache(self.parity_Z)

            if self.use_weight2:
                self.cache_X_w2 = build_weight2_timelike_cache(
                    self.parity_Z, self.parity_Z, self.distance, "X", self.device
                )
                self.cache_Z_w2 = build_weight2_timelike_cache(
                    self.parity_X, self.parity_X, self.distance, "Z", self.device
                )

            if self.device.type == "cuda" and self.use_compile:
                import threading

                self._compile_thread = threading.Thread(
                    target=warmup_he_compile,
                    kwargs=dict(
                        distance=self.distance,
                        n_rounds=self.n_rounds,
                        basis=self.basis,
                        max_passes_w1=self.max_passes_w1,
                        use_weight2=self.use_weight2,
                        max_passes_w2=self.max_passes_w2,
                        use_parallel_spacelike=self.use_parallel_spacelike,
                    ),
                    daemon=True,
                )
                self._compile_thread.start()

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
        keep_on_cpu: bool = False,
        verbose: bool = True,
        step: Optional[int] = None,
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
            keep_on_cpu: 保持数据在 CPU 上
            verbose: 是否打印采样日志（分批采样时设为 False）

        Returns:
            dict with:
            - 'trainX': (B, 4, T, D, D) — [x_syn, z_syn, x_pres, z_pres]
            - 'trainY': (B, 4, T, D, D) — [z_err, x_err, s1x, s1z]
        """
        if self._mixed:
            if step is None:
                step = self._mixed_call_count
                self._mixed_call_count += 1
            branch_basis = "X" if (int(step) % 2 == 0) else "Z"
            return self._mixed_generators[branch_basis].generate_batch(
                batch_size=batch_size,
                seed=seed,
                keep_on_cpu=keep_on_cpu,
                verbose=verbose,
                step=step,
            )

        if self._compile_thread is not None:
            self._compile_thread.join(timeout=1200)
            if self._compile_thread.is_alive():
                raise RuntimeError("warmup_he_compile thread did not finish within 20 min")
            self._compile_thread = None

        device_id = None
        if self.device.type == "cuda":
            device_index = self.device.index
            device_id = int(torch.cuda.current_device() if device_index is None else device_index)

        # 1. 采样误差帧（支持并行）
        if self.enable_parallel and batch_size >= 10000:
            frames_xz = dem_sampling_parallel(
                self.H, self.p, batch_size,
                num_workers=self.num_workers,
                device_ids=self.device_ids,
                seed=seed,
                verbose=verbose,
            )
        else:
            frames_xz = dem_sampling(
                self.H, self.p, batch_size, device_id=device_id, seed=seed
            )  # (B, 2*n_det)

        trainX, trainY, device_orig = self._build_batch_from_frames(
            frames_xz, batch_size=batch_size, verbose=verbose
        )

        # 将最终结果移回原设备（如果需要）
        if device_orig.type != "cpu" and not keep_on_cpu:
            trainX = trainX.to(device_orig)
            trainY = trainY.to(device_orig)

        return {"trainX": trainX, "trainY": trainY}

    def _build_batch_from_frames(
        self,
        frames_xz: torch.Tensor,
        *,
        batch_size: int,
        verbose: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.device]:
        if self._has_frame_predecoder_semantics:
            D = frames_xz.shape[1] // 2
            idx_data = (
                torch.arange(self.n_rounds, device=self.device)[:, None] * self.nq +
                self.data_qubits[None, :]
            ).reshape(-1)
            x_cum = frames_xz[:, :D].index_select(1, idx_data).reshape(
                batch_size, self.n_rounds, -1
            )
            z_cum = frames_xz[:, D:].index_select(1, idx_data).reshape(
                batch_size, self.n_rounds, -1
            )
            meas_old = measure_from_stacked_frames(
                frames_xz, self.meas_qubits, self.meas_bases, nq=self.nq
            )
            meas_new = (
                timelike_syndromes(frames_xz, self.A, meas_old)
                if self.A is not None else meas_old.clone()
            )

            if self.timelike_he:
                num_x = int(self.xcheck_qubits.numel())
                s1s2x = meas_new[:, :, :num_x]
                s1s2z = meas_new[:, :, num_x:]
                mx = meas_old[:, :, :num_x]
                mz = meas_old[:, :, num_x:]
                mxp = torch.cat([torch.zeros_like(mx[:, :1, :]), mx], dim=1)
                mzp = torch.cat([torch.zeros_like(mz[:, :1, :]), mz], dim=1)
                trainX_x = mxp[:, :-1, :] ^ mxp[:, 1:, :]
                trainX_z = mzp[:, :-1, :] ^ mzp[:, 1:, :]
                z_diff, x_diff, s1s2x, s1s2z = apply_weight1_timelike_homological_equivalence_torch(
                    z_cum,
                    x_cum,
                    s1s2x,
                    s1s2z,
                    self.parity_Z,
                    self.parity_X,
                    self.distance,
                    self.num_he_cycles,
                    self.max_passes_w1,
                    self.basis,
                    True,
                    trainX_x=trainX_x,
                    trainX_z=trainX_z,
                    cache_Z_spacelike=self.cache_Z_sp,
                    cache_X_spacelike=self.cache_X_sp,
                    use_compile=self.use_compile,
                    compile_chunk_size=self.compile_chunk_size,
                    compute_dtype=self.compute_dtype,
                    use_weight2=self.use_weight2,
                    max_passes_w2=self.max_passes_w2,
                    cache_Z_w2=self.cache_Z_w2,
                    cache_X_w2=self.cache_X_w2,
                    use_coset_search=self.use_coset_search,
                    coset_max_generators=self.coset_max_generators,
                    use_dense_overlap=self.use_dense_overlap,
                    use_parallel_spacelike=self.use_parallel_spacelike,
                )
                meas_new = torch.cat([s1s2x, s1s2z], dim=2)
            else:
                xpad = torch.cat([torch.zeros_like(x_cum[:, :1, :]), x_cum], dim=1)
                zpad = torch.cat([torch.zeros_like(z_cum[:, :1, :]), z_cum], dim=1)
                x_diff = xpad[:, :-1, :] ^ xpad[:, 1:, :]
                z_diff = zpad[:, :-1, :] ^ zpad[:, 1:, :]

            device_orig = x_cum.device
            trainX, trainY = self._format_for_model(
                x_diff.to("cpu"),
                z_diff.to("cpu"),
                meas_old.to("cpu"),
                meas_new.to("cpu"),
            )
        else:
            if verbose and not self._fallback_warning_emitted:
                print(
                    "[CircuitDataGenerator] Warning: DEM source does not match "
                    "Ising frame_predecoder semantics; using approximate fallback formatting."
                )
                self._fallback_warning_emitted = True

            D = frames_xz.shape[1] // 2
            nq = max(D // self.n_rounds, 1)
            R = self.n_rounds
            DD = self.distance
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

            xpad = torch.cat([torch.zeros_like(x_cum[:, :1, :]), x_cum], dim=1)
            zpad = torch.cat([torch.zeros_like(z_cum[:, :1, :]), z_cum], dim=1)
            x_diff = xpad[:, :-1, :] ^ xpad[:, 1:, :]
            z_diff = zpad[:, :-1, :] ^ zpad[:, 1:, :]

            device_orig = x_diff.device
            trainX, trainY = self._format_for_model_legacy(x_diff.to("cpu"), z_diff.to("cpu"))

        return trainX, trainY, device_orig

    def _format_for_model(
        self,
        x_diff: torch.Tensor,
        z_diff: torch.Tensor,
        meas_old: torch.Tensor,
        meas_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Faithful Ising-style formatting for frame_predecoder samples."""
        B, R, D2 = x_diff.shape
        D = self.distance
        num_x = int(self.xcheck_qubits.numel())

        x_raw = meas_old[:, :, :num_x]
        z_raw = meas_old[:, :, num_x:]
        s1x = meas_new[:, :, :num_x]
        s1z = meas_new[:, :, num_x:]

        xp = torch.cat([torch.zeros_like(x_raw[:, :1, :]), x_raw], dim=1)
        zp = torch.cat([torch.zeros_like(z_raw[:, :1, :]), z_raw], dim=1)
        x_syn = (xp[:, :-1, :] ^ xp[:, 1:, :]).transpose(1, 2)
        z_syn = (zp[:, :-1, :] ^ zp[:, 1:, :]).transpose(1, 2)
        s1x = s1x.transpose(1, 2)
        s1z = s1z.transpose(1, 2)

        x_syn_g = (
            reshape_Xstabilizers_to_grid_vectorized(x_syn, D, rotation=self.code_rotation)
            .reshape(B, D, D, R).permute(0, 3, 1, 2).contiguous()
        )
        z_syn_g = (
            reshape_Zstabilizers_to_grid_vectorized(z_syn, D, rotation=self.code_rotation)
            .reshape(B, D, D, R).permute(0, 3, 1, 2).contiguous()
        )
        s1x_g = (
            reshape_Xstabilizers_to_grid_vectorized(s1x, D, rotation=self.code_rotation)
            .reshape(B, D, D, R).permute(0, 3, 1, 2).contiguous()
        )
        s1z_g = (
            reshape_Zstabilizers_to_grid_vectorized(s1z, D, rotation=self.code_rotation)
            .reshape(B, D, D, R).permute(0, 3, 1, 2).contiguous()
        )

        x_err = x_diff.reshape(B, R, D, D)
        z_err = z_diff.reshape(B, R, D, D)
        x_pres = self.w_mapXgrid.to(x_diff.device).unsqueeze(0).unsqueeze(0).expand(B, R, D, D).clone()
        z_pres = self.w_mapZgrid.to(x_diff.device).unsqueeze(0).unsqueeze(0).expand(B, R, D, D).clone()

        if self.basis == "X":
            z_pres[:, 0] = 0
            z_syn_g[:, 0] = 0
            z_pres[:, -1] = 0
            z_syn_g[:, -1] = 0
        else:
            x_pres[:, 0] = 0
            x_syn_g[:, 0] = 0
            x_pres[:, -1] = 0
            x_syn_g[:, -1] = 0

        trainX = torch.stack(
            [x_syn_g.float(), z_syn_g.float(), x_pres.float(), z_pres.float()], dim=1
        ).contiguous()
        trainY = torch.stack(
            [z_err.float(), x_err.float(), s1x_g.float(), s1z_g.float()], dim=1
        ).contiguous()

        return trainX, trainY

    def _format_for_model_legacy(
        self, x_diff: torch.Tensor, z_diff: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Approximate fallback for legacy DEM inputs without frame_predecoder semantics."""
        B, R, D2 = x_diff.shape
        D = self.distance
        device = x_diff.device

        x_err = x_diff.reshape(B, R, D, D)
        z_err = z_diff.reshape(B, R, D, D)
        x_pres = self.w_mapXgrid.to(device).unsqueeze(0).unsqueeze(0).expand(B, R, D, D).clone()
        z_pres = self.w_mapZgrid.to(device).unsqueeze(0).unsqueeze(0).expand(B, R, D, D).clone()

        if self.basis == "X":
            z_pres[:, 0] = 0
            z_pres[:, -1] = 0
        else:
            x_pres[:, 0] = 0
            x_pres[:, -1] = 0

        trainX = torch.stack(
            [x_err.float(), z_err.float(), x_pres.float(), z_pres.float()], dim=1
        ).contiguous()
        trainY = torch.stack(
            [z_err.float(), x_err.float(), x_err.float(), z_err.float()], dim=1
        ).contiguous()
        return trainX, trainY
