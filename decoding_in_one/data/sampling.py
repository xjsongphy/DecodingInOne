# decoding_in_one/data/sampling.py
import numpy as np
import stim

def sample_detectors_observables(
    circuit: stim.Circuit,
    shots: int,
    seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """从电路采样检测器和观测量

    Args:
        circuit: Stim 电路
        shots: 采样次数
        seed: 随机种子

    Returns:
        (dets, obs): 检测器数组和观测量数组
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    return np.asarray(dets, dtype=np.float32), np.asarray(obs, dtype=np.float32)
