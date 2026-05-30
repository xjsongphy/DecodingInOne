#!/usr/bin/env python3
"""Ising-style training experiment (training only).

- Experiment script stays in experiments/
- Model architecture lives in decoding_in_one/decoders/
- Uses a smaller Conv3d setup by default for quick iteration
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import stim
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

from decoding_in_one.decoders.predecoder_models import Conv3DPredecoder


@dataclass
class TrainConfig:
    distance: int = 5
    rounds: int = 5
    basis: str = "X"

    train_shots: int = 20000
    val_shots: int = 5000

    p_after_clifford: float = 0.001
    p_before_round_data: float = 0.001
    p_before_measure_flip: float = 0.001
    p_after_reset_flip: float = 0.001

    num_filters: list[int] | None = None
    kernel_sizes: list[int] | None = None
    activation: str = "gelu"
    dropout: float = 0.1

    batch_size: int = 512
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5

    device: str = "auto"
    seed: int = 0
    out_dir: str = "experiments/ising/train_output"

    def __post_init__(self):
        if self.num_filters is None:
            self.num_filters = [64, 64, 64, 4]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 3, 3, 3]


def _task_from_basis(basis: str) -> str:
    b = basis.strip().upper()
    if b not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")
    return f"surface_code:rotated_memory_{b.lower()}"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(config_path: str | None) -> TrainConfig:
    if not config_path:
        return TrainConfig()
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    defaults = asdict(TrainConfig())
    for k, v in raw.items():
        if k in defaults:
            defaults[k] = v
    return TrainConfig(**defaults)


def _build_stim_circuit(cfg: TrainConfig) -> stim.Circuit:
    return stim.Circuit.generated(
        _task_from_basis(cfg.basis),
        distance=cfg.distance,
        rounds=cfg.rounds,
        after_clifford_depolarization=cfg.p_after_clifford,
        before_round_data_depolarization=cfg.p_before_round_data,
        before_measure_flip_probability=cfg.p_before_measure_flip,
        after_reset_flip_probability=cfg.p_after_reset_flip,
    )


def _sample_dataset(circuit: stim.Circuit, shots: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    return np.asarray(dets, dtype=np.float32), np.asarray(obs, dtype=np.float32)


def _dets_to_trainx(dets: np.ndarray, rounds: int, distance: int) -> np.ndarray:
    """Map detector vectors to Ising-like 4-channel tensors (B,4,T,D,D)."""
    bsz, n_det = dets.shape
    t = rounds
    d = distance
    target = t * d * d
    padded = np.zeros((bsz, target), dtype=np.float32)
    use = min(n_det, target)
    padded[:, :use] = dets[:, :use]
    ch0 = padded.reshape(bsz, t, d, d)
    ch1 = ch0.copy()
    ch2 = np.ones_like(ch0, dtype=np.float32)
    ch3 = np.ones_like(ch0, dtype=np.float32)
    return np.stack([ch0, ch1, ch2, ch3], axis=1)


def _obs_to_target4(obs: np.ndarray, rounds: int, distance: int) -> np.ndarray:
    """Broadcast observables to 4-channel (B,4,T,D,D) targets."""
    bsz, n_obs = obs.shape
    base = np.zeros((bsz, 4, rounds, distance, distance), dtype=np.float32)
    if n_obs == 1:
        for i in range(4):
            base[:, i, :, :, :] = obs[:, 0][:, None, None, None]
        return base
    for i in range(min(4, n_obs)):
        base[:, i, :, :, :] = obs[:, i][:, None, None, None]
    return base


def _build_loaders(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)

            pred = (torch.sigmoid(logits) >= 0.5).float()
            exact = (pred == y).view(y.size(0), -1).all(dim=1).sum().item()
            correct += int(exact)
            total += x.size(0)

    avg_loss = total_loss / max(total, 1)
    exact_match_acc = correct / max(total, 1)
    return avg_loss, exact_match_acc


def run_training(cfg: TrainConfig) -> dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(cfg.seed)

    device = torch.device(
        "cuda" if (cfg.device == "auto" and torch.cuda.is_available())
        else cfg.device if cfg.device != "auto"
        else "cpu"
    )

    circuit = _build_stim_circuit(cfg)
    train_dets, train_obs = _sample_dataset(circuit, cfg.train_shots, cfg.seed)
    val_dets, val_obs = _sample_dataset(circuit, cfg.val_shots, cfg.seed + 1)

    train_x = _dets_to_trainx(train_dets, cfg.rounds, cfg.distance)
    val_x = _dets_to_trainx(val_dets, cfg.rounds, cfg.distance)
    train_y = _obs_to_target4(train_obs, cfg.rounds, cfg.distance)
    val_y = _obs_to_target4(val_obs, cfg.rounds, cfg.distance)

    train_loader, val_loader = _build_loaders(train_x, train_y, val_x, val_y, cfg.batch_size)

    model = Conv3DPredecoder(
        input_channels=4,
        out_channels=4,
        num_filters=cfg.num_filters,
        kernel_sizes=cfg.kernel_sizes,
        dropout_p=cfg.dropout,
        activation=cfg.activation,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_ckpt = out_dir / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * x.size(0)
            seen += x.size(0)

        train_loss = running_loss / max(seen, 1)
        val_loss, val_exact = _evaluate(model, val_loader, device, criterion)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_exact_match_acc": float(val_exact),
        }
        history.append(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "input_shape": list(train_x.shape[1:]),
                    "output_shape": list(train_y.shape[1:]),
                },
                best_ckpt,
            )

        print(
            f"[Train] epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_exact_match_acc={val_exact:.6f}"
        )

    final_report = {
        "config": asdict(cfg),
        "device": str(device),
        "data": {
            "input_shape": list(train_x.shape[1:]),
            "output_shape": list(train_y.shape[1:]),
            "train_shots": int(cfg.train_shots),
            "val_shots": int(cfg.val_shots),
        },
        "best_val_loss": float(best_val_loss),
        "best_checkpoint": str(best_ckpt),
        "history": history,
    }

    (out_dir / "train_report.json").write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return final_report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ising-style training experiment (model only)")
    p.add_argument("--config", type=str, default="experiments/ising/ising_train.yaml")
    p.add_argument("--distance", type=int)
    p.add_argument("--rounds", type=int)
    p.add_argument("--basis", type=str)
    p.add_argument("--train-shots", type=int)
    p.add_argument("--val-shots", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--out-dir", type=str)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config if args.config and Path(args.config).exists() else None)

    override_map = {
        "distance": "distance",
        "rounds": "rounds",
        "basis": "basis",
        "train_shots": "train_shots",
        "val_shots": "val_shots",
        "epochs": "epochs",
        "batch_size": "batch_size",
        "lr": "lr",
        "seed": "seed",
        "out_dir": "out_dir",
    }
    for arg_key, cfg_key in override_map.items():
        val = getattr(args, arg_key)
        if val is not None:
            setattr(cfg, cfg_key, val)

    report = run_training(cfg)
    print(
        f"[Train] done. best_val_loss={report['best_val_loss']:.6f}, "
        f"ckpt={report['best_checkpoint']}"
    )


if __name__ == "__main__":
    main()
