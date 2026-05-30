# decoding_in_one/utils/types.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch


@dataclass(frozen=True)
class CodeSpec:
    """码参数快照，用于模块间传递。"""
    code_family: str
    distance: int
    rotation: str = "XV"
    n_physical: int = 0
    n_logical: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CircuitSpec:
    """电路构建参数。"""
    n_rounds: int
    measurement_basis: str = "X"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitArtifact:
    """结构化电路产物。"""
    stim_circuit: str
    code: CodeSpec
    spec: CircuitSpec
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodingBatch:
    """解码批次数据"""
    detectors: torch.Tensor           # (batch, n_detectors)
    observables: torch.Tensor         # (batch, n_observables)
    syndrome_grid: Optional[torch.Tensor] = None  # (batch, n_rounds, d, d)
    metadata: Dict[str, Any] = field(default_factory=dict)    # 噪声参数、代码参数等

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
