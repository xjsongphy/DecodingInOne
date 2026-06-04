import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class Lion(Optimizer):
    """Implements the Lion optimizer used by Ising-Decoding."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


def build_warmup_then_decay_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    milestones: list[float],
    gamma: float,
):
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if warmup_steps >= total_steps:
        warmup_steps = max(0, total_steps - 1)

    milestone_steps = [int(total_steps * frac) for frac in milestones]

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        decay_factor = 1.0
        for milestone_step in milestone_steps:
            if current_step >= milestone_step:
                decay_factor *= gamma
        return decay_factor

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_cosine_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float,
):
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if base_lr <= 0:
        raise ValueError(f"base_lr must be > 0, got {base_lr}")
    if warmup_steps >= total_steps:
        warmup_steps = max(0, total_steps - 1)

    min_lr_ratio = min_lr / base_lr

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return cosine_decay * (1.0 - min_lr_ratio) + min_lr_ratio

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
