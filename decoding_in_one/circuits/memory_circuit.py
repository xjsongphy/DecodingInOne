# decoding_in_one/circuits/memory_circuit.py
"""
MemoryCircuit 电路构建器（迁移自 Ising-Decoding）

源码参考: D:\\Develop\\Ising-Decoding\\code\\qec\\surface_code\\memory_circuit.py

此模块提供手动构建 Stim 电路的能力，支持 25 参数噪声模型。
包含两层：
  - Circuit: 底层 Stim 电路字符串构建器（门操作 + 噪声注入）
  - MemoryCircuit: 高层表面码存储实验电路（完整的多轮稳定子测量）
"""

from __future__ import annotations

import numpy as np
import stim
from typing import Optional, TYPE_CHECKING

from decoding_in_one.noise import NoiseModel
from decoding_in_one.utils.types import CircuitArtifact, CircuitSpec

if TYPE_CHECKING:
    from decoding_in_one.codes import QuantumCode


# ---------------------------------------------------------------------------
# Circuit: 底层 Stim 电路字符串构建器
# ---------------------------------------------------------------------------

class Circuit:
    """
    Stim 电路字符串构建器

    支持两种噪声模式：
    1. 简单模式：单错误率 (idle_error, sqgate_error, tqgate_error, spam_error)
    2. NoiseModel 模式：25 参数显式噪声模型
    """

    def __init__(self, all_qubits, code=None):
        self.circuit = ""
        self.margin = ""
        self.all_qubits = all_qubits
        self.code = code  # 用于 hadamard_layer 等需要码结构的方法

        # 简单错误率
        self.idle_error = 0.0
        self.sqgate_error = 0.0
        self.tqgate_error = 0.0
        self.spam_error = 0.0

        # 25 参数噪声模型
        self.noise_model: Optional[NoiseModel] = None
        self.basis = "X"  # 当前测量基（逻辑测量轮特殊处理用）

    # ---- 噪声设置 --------------------------------------------------------

    def set_noise_model(self, noise_model: NoiseModel) -> None:
        """设置 25 参数噪声模型，同时更新简单错误率回退值"""
        self.noise_model = noise_model
        if noise_model is not None:
            self.spam_error = max(
                noise_model.p_prep_X, noise_model.p_prep_Z,
                noise_model.p_meas_X, noise_model.p_meas_Z,
            )
            self.idle_error = max(
                noise_model.get_total_idle_cnot_probability(),
                noise_model.get_total_idle_spam_probability(),
            )
            self.tqgate_error = noise_model.get_total_cnot_probability()

    def set_error_rates_simple(self, idle_error, sqgate_error, tqgate_error, spam_error):
        self.idle_error = idle_error
        self.sqgate_error = sqgate_error
        self.tqgate_error = tqgate_error
        self.spam_error = spam_error

    # ---- 电路结构 --------------------------------------------------------

    def start_loop(self, num_rounds):
        c = "REPEAT %d {\n" % num_rounds
        self.circuit += c
        self.margin = "    "
        return c

    def end_loop(self):
        c = "}\n"
        self.circuit += c
        self.margin = ""
        return c

    def add_tick(self):
        c = self.margin + "TICK\n"
        self.circuit += c
        return c

    # ---- 门操作 + 噪声 ---------------------------------------------------

    def add_reset(self, qubits, basis="Z"):
        """RESET 操作 + 制备错误"""
        basis = basis.upper()
        c = self.margin
        c += "RZ " if basis == "Z" else "RX "
        for q in qubits:
            c += "%d " % q
        c += "\n"

        if self.noise_model is not None:
            if basis == "Z" and self.noise_model.p_prep_Z > 0:
                c += self.margin + "X_ERROR(%.10f) " % self.noise_model.p_prep_Z
                for q in qubits:
                    c += "%d " % q
                c += "\n"
            elif basis == "X" and self.noise_model.p_prep_X > 0:
                c += self.margin + "Z_ERROR(%.10f) " % self.noise_model.p_prep_X
                for q in qubits:
                    c += "%d " % q
                c += "\n"
        elif self.spam_error > 0.0:
            c += self.margin
            c += ("X_ERROR" if basis == "Z" else "Z_ERROR") + "(%.10f) " % self.spam_error
            for q in qubits:
                c += "%d " % q
            c += "\n"

        self.circuit += c

    def add_single_error(self, qubits, error_type):
        """单量子比特错误（X 或 Z）"""
        if self.noise_model is not None:
            error_prob = self.noise_model.p_prep_Z if error_type == "X" else self.noise_model.p_prep_X
        else:
            error_prob = self.spam_error

        if error_prob == 0.0:
            return ""

        c = self.margin
        c += ("%s_ERROR(%.10f) " % (error_type, error_prob))
        for q in qubits:
            c += "%d " % q
        c += "\n"
        self.circuit += c
        return c

    def add_idle(self, qubits, logical_measurement=False, idle_kind: str = "cnot"):
        """空闲错误

        idle_kind: 'cnot' = bulk/CNOT 层; 'spam' = ancilla prep/reset 窗口
        """
        if self.noise_model is not None:
            if idle_kind == "spam":
                pX, pY, pZ = self.noise_model.to_stim_pauli_channel_1_args_spam()
            else:
                pX, pY, pZ = self.noise_model.to_stim_pauli_channel_1_args_cnot()
            if pX + pY + pZ == 0.0:
                return ""
            c = self.margin
            if not logical_measurement:
                c += "PAULI_CHANNEL_1(%.10f, %.10f, %.10f) " % (pX, pY, pZ)
            else:
                c += ("Z_ERROR(%.10f) " % pZ) if self.basis == "X" else ("X_ERROR(%.10f) " % pX)
            for q in qubits:
                c += "%d " % q
            c += "\n"
            self.circuit += c
            return c
        else:
            if self.idle_error == 0.0:
                return ""
            c = self.margin
            if not logical_measurement:
                c += "DEPOLARIZE1(%.10f) " % self.idle_error
            else:
                c += ("Z_ERROR(%.10f) " % self.idle_error) if self.basis == "X" else (
                    "X_ERROR(%.10f) " % self.idle_error
                )
            for q in qubits:
                c += "%d " % q
            c += "\n"
            self.circuit += c
            return c

    def add_cnot(self, qubits):
        """CNOT + 噪声"""
        c = self.margin + "CX "
        for q in qubits:
            c += "%d " % q
        c += "\n"

        if self.noise_model is not None:
            probs = self.noise_model.to_stim_pauli_channel_2_args()
            if sum(probs) > 0.0:
                c += self.margin + "PAULI_CHANNEL_2(%s) " % ", ".join(
                    "%.10f" % p for p in probs
                )
                for q in qubits:
                    c += "%d " % q
                c += "\n"
        elif self.tqgate_error > 0.0:
            c += self.margin + "DEPOLARIZE2(%.10f) " % self.tqgate_error
            for q in qubits:
                c += "%d " % q
            c += "\n"

        self.circuit += c

    def add_cnot_layer(self, qubits, add_tick=True):
        """CNOT 层（对其他比特加空闲错误）"""
        self.add_cnot(qubits)
        all_qubits = np.atleast_1d(np.asarray(self.all_qubits))
        active = np.atleast_1d(np.asarray(qubits))
        other = all_qubits[~np.isin(all_qubits, active)]
        self.add_idle(other)
        if add_tick:
            self.add_tick()

    # ---- 测量 ------------------------------------------------------------

    def add_measure(self, qubits, basis="Z", include_reset=False):
        """测量 + 测量前错误"""
        basis = basis.upper()
        c = ""

        # 测量前错误
        if self.noise_model is not None:
            if basis == "Z" and self.noise_model.p_meas_Z > 0:
                c += self.margin + "X_ERROR(%.10f) " % self.noise_model.p_meas_Z
                for q in qubits:
                    c += "%d " % q
                c += "\n"
            elif basis == "X" and self.noise_model.p_meas_X > 0:
                c += self.margin + "Z_ERROR(%.10f) " % self.noise_model.p_meas_X
                for q in qubits:
                    c += "%d " % q
                c += "\n"
        elif self.spam_error > 0.0:
            c += self.margin
            c += ("X_ERROR" if basis == "Z" else "Z_ERROR") + "(%.10f) " % self.spam_error
            for q in qubits:
                c += "%d " % q
            c += "\n"

        c += self.margin
        op = {"Z": "MRZ" if include_reset else "MZ", "X": "MRX" if include_reset else "MX"}[basis]
        c += op + " "
        for q in qubits:
            c += "%d " % q
        c += "\n"
        self.circuit += c

    # ---- 注解 ------------------------------------------------------------

    def add_detector(self, inds):
        c = self.margin + "DETECTOR "
        for ind in inds:
            c += "rec[-%d] " % ind
        c += "\n"
        self.circuit += c

    def add_observable(self, observable_no, inds):
        c = self.margin + "OBSERVABLE_INCLUDE(%d) " % observable_no
        for ind in inds:
            c += "rec[-%d] " % ind
        c += "\n"
        self.circuit += c

    # ---- 编译 ------------------------------------------------------------

    def compile_to_stim(self) -> stim.Circuit:
        return stim.Circuit(self.circuit)


# ---------------------------------------------------------------------------
# MemoryCircuit: 表面码存储实验电路
# ---------------------------------------------------------------------------

class MemoryCircuit(Circuit):
    """
    表面码存储电路（迁移自 Ising-Decoding MemoryCircuit）

    完整的多轮稳定子测量电路，支持 25 参数噪声模型。
    接受 QuantumCode 实例（通用化），但电路排布目前针对表面码。

    Args:
        code: QuantumCode 实例
        n_rounds: 稳定子测量轮数
        basis: 逻辑基 ('X' 或 'Z')
        noise_model: 可选的 25 参数噪声模型
        add_tick: 是否添加时序标记
        add_detectors: 是否添加检测器注释
        add_boundary_detectors: 是否添加边界检测器
    """

    def __init__(
        self,
        code: "QuantumCode",
        n_rounds: int,
        basis: str = "X",
        noise_model: Optional[NoiseModel] = None,
        add_tick: bool = True,
        add_detectors: bool = True,
        add_boundary_detectors: bool = False,
    ):
        self.distance = code.distance
        self.n_rounds = n_rounds
        self.basis = basis.upper()
        self._add_tick = add_tick
        self._add_detectors = add_detectors
        self._add_boundary_detectors = add_boundary_detectors
        self.code = code

        # 初始化底层 Circuit
        super().__init__(code.get_n_physical(), code=code)

        # 设置噪声
        if noise_model is not None:
            self.set_noise_model(noise_model)

        get_Z_detectors = self.basis == "Z" or add_detectors
        get_X_detectors = self.basis == "X" or add_detectors

        self._build(get_X_detectors, get_Z_detectors)
        self.stim_circuit = self.compile_to_stim()

    # ------------------------------------------------------------------
    # 电路组装
    # ------------------------------------------------------------------

    def _build(self, get_X_detectors, get_Z_detectors):
        code = self.code
        n_x = len(code.get_check_qubits("X"))
        n_z = len(code.get_check_qubits("Z"))

        # ---- 逻辑态制备 ----
        self.add_reset(code.get_data_qubits(), self.basis)
        self._add_stabilizer_round(state_prep=True, combine_reset_and_measure=True)

        # ---- 第一轮检测器 ----
        if self.basis == "X" and get_X_detectors:
            for i in range(1, n_x + 1)[::-1]:
                self.add_detector([n_z + i])
        elif self.basis == "Z" and get_Z_detectors:
            for i in range(1, n_z + 1)[::-1]:
                self.add_detector([i])

        # ---- 逻辑存储（带噪声）----
        if self.n_rounds - 2 > 0:
            self.start_loop(self.n_rounds - 2)
            self._add_stabilizer_round(combine_reset_and_measure=True)

            if self._add_detectors:
                if get_Z_detectors:
                    for i in range(1, n_z + 1)[::-1]:
                        ind = n_x + i
                        self.add_detector([ind, ind + n_x + n_z])
                if get_X_detectors:
                    for i in range(1, n_x + 1)[::-1]:
                        self.add_detector([i, i + n_x + n_z])

            self.end_loop()

        # ---- 逻辑测量 ----
        self._add_stabilizer_round(logical_measurement=True, combine_reset_and_measure=True)

        if self._add_detectors:
            if get_Z_detectors:
                for i in range(1, n_z + 1)[::-1]:
                    ind = n_x + i
                    self.add_detector([ind, ind + n_x + n_z])
            if get_X_detectors:
                for i in range(1, n_x + 1)[::-1]:
                    self.add_detector([i, i + n_x + n_z])

        # ---- 最终数据比特测量（无噪声）----
        orig_rates = (self.idle_error, self.sqgate_error, self.tqgate_error, self.spam_error)
        orig_nm = self.noise_model
        self.set_error_rates_simple(0, 0, 0, 0)
        self.noise_model = None
        self.add_measure(code.get_data_qubits(), basis=self.basis)
        self.set_error_rates_simple(*orig_rates)
        self.noise_model = orig_nm

        # ---- 边界检测器 ----
        if self._add_boundary_detectors:
            self._add_boundary_detectors_to_circuit()

        # ---- 逻辑可观测量 ----
        if self._add_detectors:
            self._add_logical_observable()

    # ------------------------------------------------------------------
    # 稳定子轮（通用化：支持任意图结构）
    # ------------------------------------------------------------------

    def _add_stabilizer_round(
        self, logical_measurement=False, state_prep=False, combine_reset_and_measure=False
    ):
        code = self.code
        xcheck = code.get_check_qubits("X")
        zcheck = code.get_check_qubits("Z")

        # 通用化：使用 get_stabilizer_measurement_layers 获取 CNOT 层
        # 对于表面码，这返回优化的四层；对于 QLDPC，使用图着色
        x_layers = code.get_stabilizer_measurement_layers("X")
        z_layers = code.get_stabilizer_measurement_layers("Z")

        # --- 逻辑测量轮：保存 / 临时清除噪声 ---
        if logical_measurement:
            orig = (self.idle_error, self.sqgate_error, self.tqgate_error, self.spam_error)
            orig_nm = self.noise_model
            self.noise_model = None
            self.set_error_rates_simple(0, 0, 0, 0)

        # --- Reset / MR 之间选择性错误 ---
        if not combine_reset_and_measure:
            self.add_reset(xcheck, basis="X")
            self.add_reset(zcheck, basis="Z")
        else:
            if state_prep:
                self.add_reset(xcheck, basis="X")
                self.add_reset(zcheck, basis="Z")
            else:
                self.add_single_error(xcheck, "Z")
                self.add_single_error(zcheck, "X")

        # --- 数据比特 idle / fake SPAM ---
        if not state_prep:
            if logical_measurement and orig_nm is not None:
                # 注入 "fake data-measurement SPAM" 错误
                if self.basis == "X":
                    p_fake = float(orig_nm.p_meas_X)
                    if p_fake > 0:
                        c = self.margin + "Z_ERROR(%.10f) " % p_fake
                        for q in code.get_data_qubits():
                            c += "%d " % q
                        c += "\n"
                        self.circuit += c
                else:
                    p_fake = float(orig_nm.p_meas_Z)
                    if p_fake > 0:
                        c = self.margin + "X_ERROR(%.10f) " % p_fake
                        for q in code.get_data_qubits():
                            c += "%d " % q
                        c += "\n"
                        self.circuit += c
            else:
                if self.noise_model is not None:
                    pass  # NoiseModel 模式忽略此处 data-idle
                else:
                    self.add_idle(code.get_data_qubits(), logical_measurement=logical_measurement)

        if logical_measurement:
            self.noise_model = orig_nm
            self.set_error_rates_simple(*orig)

        # --- TICK ---
        if self._add_tick:
            self.add_tick()

        # --- 通用 CNOT 层：支持任意图结构 ---
        # 对于表面码，x_layers/z_layers 返回优化的四层
        # 对于 QLDPC，使用图着色处理非局部连接
        all_layers = []
        for x_layer in x_layers:
            all_layers.append(x_layer)
        for z_layer in z_layers:
            all_layers.append(z_layer)

        for layer in all_layers:
            qubits = []
            for control, target in layer:
                qubits.extend([control, target])
            if qubits:
                self.add_cnot_layer(qubits, add_tick=self._add_tick)

        # --- 测量前 TICK ---
        if self._add_tick:
            self.add_tick()

        # --- 测量辅助比特 ---
        self.add_measure(xcheck, basis="X", include_reset=combine_reset_and_measure)
        self.add_measure(zcheck, basis="Z", include_reset=combine_reset_and_measure)

        # --- 数据比特 idle（测量窗口）---
        if self.noise_model is None:
            self.add_idle(code.get_data_qubits(), logical_measurement=logical_measurement)
        else:
            self.add_idle(code.get_data_qubits(), logical_measurement=logical_measurement, idle_kind="spam")

    # ------------------------------------------------------------------
    # 边界检测器 & 逻辑可观测量
    # ------------------------------------------------------------------

    def _add_boundary_detectors_to_circuit(self):
        """添加边界检测器（比较最终数据比特测量与最后一次辅助比特测量）"""
        code = self.code
        num_data = len(code.get_data_qubits())
        num_x = len(code.get_check_qubits("X"))
        num_z = len(code.get_check_qubits("Z"))

        if self.basis == "X":
            parity = self.code.hx if hasattr(self.code, "hx") else None
            ancilla_base = num_data + num_z
            num_ancillas = num_x
        else:
            parity = self.code.hz if hasattr(self.code, "hz") else None
            ancilla_base = num_data
            num_ancillas = num_z

        if parity is None:
            return

        for stab_idx in range(parity.shape[0]):
            support = [i for i in range(num_data) if parity[stab_idx, i] == 1]
            if not support:
                continue
            data_rec = [num_data - i for i in support]
            ancilla_rec = ancilla_base + (num_ancillas - stab_idx)
            self.add_detector(data_rec + [ancilla_rec])

    def _add_logical_observable(self):
        """添加逻辑可观测量"""
        code = self.code
        data_qubits = code.get_data_qubits()
        logical_ops = code.get_logical_operators()
        num_data = len(data_qubits)

        key = self.basis.upper()
        if key in logical_ops:
            pauli_str = logical_ops[key]
            inds = [num_data - i for i, op in enumerate(pauli_str.operators) if op == key]
            if inds:
                self.add_observable(0, inds)

    # ------------------------------------------------------------------
    # CircuitBuilder 接口兼容
    # ------------------------------------------------------------------

    def build_memory_circuit(self, code, n_rounds: int, measurement_basis: str) -> str:
        """CircuitBuilder 接口兼容方法"""
        return self.circuit

    def build_memory_artifact(self, code, spec: CircuitSpec) -> CircuitArtifact:
        """CircuitBuilder 接口兼容方法"""
        from decoding_in_one.utils.types import CodeSpec
        return CircuitArtifact(
            stim_circuit=self.circuit,
            code=CodeSpec(
                code_family=code.__class__.__name__,
                distance=code.distance,
                rotation=getattr(code, "rotation", "XV"),
                n_physical=code.get_n_physical(),
                n_logical=code.get_n_logical(),
            ),
            spec=spec,
        )

    def build_stabilizer_measurement(self, code, stabilizer_type: str, stabilizer_idx: int) -> str:
        """CircuitBuilder 接口兼容方法"""
        return ""
