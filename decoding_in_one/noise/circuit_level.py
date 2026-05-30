# decoding_in_one/noise/circuit_level.py
"""
从 Ising-Decoding 迁移的 25 参数电路级噪声模型
源码参考: Ising-Decoding/code/qec/noise_model.py
"""

import yaml
from pathlib import Path
from typing import Dict, List
from decoding_in_one.noise.base import NoiseModel

# CNOT 错误类型（15 个，排除 II）
CNOT_ERROR_TYPES = [
    'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ',
    'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ'
]

class CircuitLevelNoise(NoiseModel):
    """
    25 参数电路级噪声模型

    参数分类：
    - 态制备（2）：p_prep_X, p_prep_Z
    - 测量（2）：p_meas_X, p_meas_Z
    - CNOT 层空闲（3）：p_idle_cnot_X, p_idle_cnot_Y, p_idle_cnot_Z
    - SPAM 窗口空闲（3）：p_idle_spam_X, p_idle_spam_Y, p_idle_spam_Z
    - CNOT 两比特（15）：p_cnot_XX, p_cnot_XY, ..., p_cnot_ZZ
    """

    def __init__(
        self,
        p_prep_X: float = 0.0,
        p_prep_Z: float = 0.0,
        p_meas_X: float = 0.0,
        p_meas_Z: float = 0.0,
        p_idle_cnot_X: float = 0.0,
        p_idle_cnot_Y: float = 0.0,
        p_idle_cnot_Z: float = 0.0,
        p_idle_spam_X: float = 0.0,
        p_idle_spam_Y: float = 0.0,
        p_idle_spam_Z: float = 0.0,
        **kwargs  # 接受所有 CNOT 参数
    ):
        # 态制备
        self.p_prep_X = p_prep_X
        self.p_prep_Z = p_prep_Z

        # 测量
        self.p_meas_X = p_meas_X
        self.p_meas_Z = p_meas_Z

        # CNOT 层空闲
        self.p_idle_cnot_X = p_idle_cnot_X
        self.p_idle_cnot_Y = p_idle_cnot_Y
        self.p_idle_cnot_Z = p_idle_cnot_Z

        # SPAM 窗口空闲
        self.p_idle_spam_X = p_idle_spam_X
        self.p_idle_spam_Y = p_idle_spam_Y
        self.p_idle_spam_Z = p_idle_spam_Z

        # CNOT 两比特参数（15 个）
        for error_type in CNOT_ERROR_TYPES:
            key = f'p_cnot_{error_type}'
            setattr(self, key, kwargs.get(key, 0.0))

    def apply_to_circuit(self, circuit: str) -> str:
        """
        将噪声应用到 Stim 电路

        简化实现：返回带噪声注释的电路
        完整实现需要解析 Stim 电路并插入噪声操作
        """
        # 简化版：在电路开头添加噪声说明
        header = f"# Circuit-level noise (25p model)\n"
        header += f"# p_prep={self.p_prep_X}, p_idle_cnot={self.p_idle_cnot_X}\n"
        return header + circuit

    def get_parameters(self) -> Dict[str, float]:
        """返回所有 25 个参数"""
        params = {}

        # 态制备和测量
        for key in ['p_prep_X', 'p_prep_Z', 'p_meas_X', 'p_meas_Z']:
            params[key] = getattr(self, key)

        # 空闲
        for key in ['p_idle_cnot_X', 'p_idle_cnot_Y', 'p_idle_cnot_Z',
                    'p_idle_spam_X', 'p_idle_spam_Y', 'p_idle_spam_Z']:
            params[key] = getattr(self, key)

        # CNOT
        for error_type in CNOT_ERROR_TYPES:
            key = f'p_cnot_{error_type}'
            params[key] = getattr(self, key, 0.0)

        return params

    def validate(self) -> bool:
        """验证参数有效性"""
        # 检查所有概率在 [0, 1] 范围内
        params = self.get_parameters()
        for key, value in params.items():
            if not (0 <= value <= 1):
                return False

        # 检查 CNOT 总概率不超过 1
        cnot_total = sum(getattr(self, k, 0) for k in params.keys() if k.startswith('p_cnot_'))
        if cnot_total > 1:
            return False

        return True

    @classmethod
    def from_config(cls, config_path: str) -> 'CircuitLevelNoise':
        """从 YAML 配置文件加载"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return cls(**config)
