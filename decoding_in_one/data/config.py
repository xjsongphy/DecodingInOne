# decoding_in_one/data/config.py
from dataclasses import dataclass

@dataclass
class IsingDataConfig:
    """Ising 实验数据配置（数据生成相关）"""
    distance: int = 5
    rounds: int = 5
    basis: str = "X"
    train_shots: int = 20000
    val_shots: int = 5000

    # 噪声参数
    p_after_clifford: float = 0.001
    p_before_round_data: float = 0.001
    p_before_measure_flip: float = 0.001
    p_after_reset_flip: float = 0.001
