from dataclasses import dataclass, field


@dataclass
class OptimConfig:
    """Training hyperparameters for the local Ising-style trainer."""

    batch_size: int = 512
    val_batch_size: int = 2048
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-7
    optimizer_type: str = "Lion"
    beta2: float = 0.95
    accumulate_steps: int = 2
    lr_scheduler_type: str = "warmup_then_decay"
    warmup_steps: int = 100
    lr_milestones: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])
    lr_gamma: float = 0.7
    min_lr: float = 1e-6
    batch_schedule_enabled: bool = True
    batch_size_initial: int = 512
    batch_size_final: int = 2048
    batch_schedule_start_epoch: int = 0
    batch_schedule_end_epoch: int = 0
    device: str = "auto"
    seed: int = 0
    out_dir: str = "outputs/ising"
