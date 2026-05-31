# decoding_in_one/data/config.py
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class IsingDataConfig:
    """Ising 实验数据配置（数据生成相关）"""
    distance: int = 5
    rounds: int = 5
    basis: str = "O1"  # O1, O2, O3, O4 (对应 Ising-Decoding 的 code_rotation)
    train_shots: int = 20000
    val_shots: int = 5000

    # 噪声模型配置路径（25 参数电路级噪声）
    noise_model_path: Optional[str] = None
    # 预计算 DEM 矩阵目录（Ising-Decoding 同款 frame_predecoder 产物）
    precomputed_frames_dir: Optional[str] = None

    # 兼容旧版 4 参数接口（已废弃，保留用于向后兼容）
    p_after_clifford: float = 0.001
    p_before_round_data: float = 0.001
    p_before_measure_flip: float = 0.001
    p_after_reset_flip: float = 0.001

    # 25 参数噪声模型字段（可选直接指定）
    p_prep_X: float = 0.002
    p_prep_Z: float = 0.002
    p_meas_X: float = 0.002
    p_meas_Z: float = 0.002
    p_idle_cnot_X: float = 0.001
    p_idle_cnot_Y: float = 0.001
    p_idle_cnot_Z: float = 0.001
    p_idle_spam_X: float = 0.001996
    p_idle_spam_Y: float = 0.001996
    p_idle_spam_Z: float = 0.001996
    p_cnot_IX: float = 0.0002
    p_cnot_IY: float = 0.0002
    p_cnot_IZ: float = 0.0002
    p_cnot_XI: float = 0.0002
    p_cnot_XX: float = 0.0002
    p_cnot_XY: float = 0.0002
    p_cnot_XZ: float = 0.0002
    p_cnot_YI: float = 0.0002
    p_cnot_YX: float = 0.0002
    p_cnot_YY: float = 0.0002
    p_cnot_YZ: float = 0.0002
    p_cnot_ZI: float = 0.0002
    p_cnot_ZX: float = 0.0002
    p_cnot_ZY: float = 0.0002
    p_cnot_ZZ: float = 0.0002

    def get_rotation(self) -> str:
        """将 basis (O1-O4) 映射到 code_rotation (XV, XH, ZV, ZH)"""
        mapping = {"O1": "XV", "O2": "XH", "O3": "ZV", "O4": "ZH"}
        b = self.basis.upper()
        if b in mapping:
            return mapping[b]
        # 兼容旧的 X/Z 基
        if b == "X":
            return "XV"
        if b == "Z":
            return "ZH"
        raise ValueError(f"basis must be O1-O4 or X/Z, got {self.basis}")

    def get_25p_noise_params(self) -> Dict[str, float]:
        """返回所有 25 个噪声参数"""
        return {
            "p_prep_X": self.p_prep_X,
            "p_prep_Z": self.p_prep_Z,
            "p_meas_X": self.p_meas_X,
            "p_meas_Z": self.p_meas_Z,
            "p_idle_cnot_X": self.p_idle_cnot_X,
            "p_idle_cnot_Y": self.p_idle_cnot_Y,
            "p_idle_cnot_Z": self.p_idle_cnot_Z,
            "p_idle_spam_X": self.p_idle_spam_X,
            "p_idle_spam_Y": self.p_idle_spam_Y,
            "p_idle_spam_Z": self.p_idle_spam_Z,
            "p_cnot_IX": self.p_cnot_IX,
            "p_cnot_IY": self.p_cnot_IY,
            "p_cnot_IZ": self.p_cnot_IZ,
            "p_cnot_XI": self.p_cnot_XI,
            "p_cnot_XX": self.p_cnot_XX,
            "p_cnot_XY": self.p_cnot_XY,
            "p_cnot_XZ": self.p_cnot_XZ,
            "p_cnot_YI": self.p_cnot_YI,
            "p_cnot_YX": self.p_cnot_YX,
            "p_cnot_YY": self.p_cnot_YY,
            "p_cnot_YZ": self.p_cnot_YZ,
            "p_cnot_ZI": self.p_cnot_ZI,
            "p_cnot_ZX": self.p_cnot_ZX,
            "p_cnot_ZY": self.p_cnot_ZY,
            "p_cnot_ZZ": self.p_cnot_ZZ,
        }
