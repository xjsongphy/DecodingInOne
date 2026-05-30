# decoding_in_one/utils/types.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch

@dataclass
class DecodingBatch:
    """解码批次数据"""
    detectors: torch.Tensor           # (batch, n_detectors)
    observables: torch.Tensor         # (batch, n_observables)
    syndrome_grid: Optional[torch.Tensor] = None  # (batch, n_rounds, d, d)
    metadata: Dict[str, Any] = None    # 噪声参数、代码参数等

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
