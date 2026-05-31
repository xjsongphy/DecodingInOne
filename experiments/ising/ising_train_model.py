#!/usr/bin/env python3
"""Ising-style training experiment (使用新模块).

- 实验脚本调用 decoding_in_one 核心模块
- 模型、数据、训练职责分离
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
import yaml

from decoding_in_one.models import SurfaceCodeConv3DDecoder, Conv3DModelConfig
from decoding_in_one.models.surface_code import dets_to_conv3d_input, obs_to_conv3d_target
from decoding_in_one.training import Trainer, OptimConfig
from decoding_in_one.data import IsingDataConfig, sample_detectors_observables, build_dataloaders


@dataclass
class ExperimentConfig:
    """实验配置（组合各类配置）"""
    # 数据配置
    data: IsingDataConfig

    # 模型配置
    model: Conv3DModelConfig

    # 训练配置
    training: OptimConfig


def _load_config(config_path: str | None) -> ExperimentConfig:
    """加载配置文件"""
    if not config_path:
        # 使用默认配置
        return ExperimentConfig(
            data=IsingDataConfig(),
            model=Conv3DModelConfig(),
            training=OptimConfig()
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # 解析各类配置
    data_config = IsingDataConfig(**{k: v for k, v in raw.items() if k in IsingDataConfig.__dataclass_fields__})
    model_config = Conv3DModelConfig(**{k: v for k, v in raw.items() if k in Conv3DModelConfig.__dataclass_fields__})
    training_config = OptimConfig(**{k: v for k, v in raw.items() if k in OptimConfig.__dataclass_fields__})

    return ExperimentConfig(data=data_config, model=model_config, training=training_config)


def _task_from_basis(basis: str) -> str:
    """从测量基获取 Stim 任务名称

    支持:
    - O1-O4: 表面码旋转方向 (O1=XV, O2=XH, O3=ZV, O4=ZH)
    - X/Z: 兼容旧格式
    """
    b = basis.strip().upper()

    # O1-O4 映射到测量基
    if b in ("O1", "O2"):
        return "surface_code:rotated_memory_x"
    if b in ("O3", "O4"):
        return "surface_code:rotated_memory_z"

    # 兼容旧格式 X/Z
    if b == "X":
        return "surface_code:rotated_memory_x"
    if b == "Z":
        return "surface_code:rotated_memory_z"

    raise ValueError("basis must be O1-O4 or X/Z")


def _build_stim_circuit(cfg: IsingDataConfig) -> stim.Circuit:
    """构建 Stim 电路"""
    return stim.Circuit.generated(
        _task_from_basis(cfg.basis),
        distance=cfg.distance,
        rounds=cfg.rounds,
        after_clifford_depolarization=cfg.p_after_clifford,
        before_round_data_depolarization=cfg.p_before_round_data,
        before_measure_flip_probability=cfg.p_before_measure_flip,
        after_reset_flip_probability=cfg.p_after_reset_flip,
    )


def run_training(cfg: ExperimentConfig) -> dict[str, Any]:
    """执行训练实验"""
    # 设置随机种子
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    # 构建电路
    circuit = _build_stim_circuit(cfg.data)

    # 采样数据
    train_dets, train_obs = sample_detectors_observables(
        circuit, cfg.data.train_shots, cfg.training.seed
    )
    val_dets, val_obs = sample_detectors_observables(
        circuit, cfg.data.val_shots, cfg.training.seed + 1
    )

    # 数据预处理
    train_x = dets_to_conv3d_input(train_dets, cfg.data.rounds, cfg.data.distance)
    val_x = dets_to_conv3d_input(val_dets, cfg.data.rounds, cfg.data.distance)
    train_y = obs_to_conv3d_target(train_obs, cfg.data.rounds, cfg.data.distance)
    val_y = obs_to_conv3d_target(val_obs, cfg.data.rounds, cfg.data.distance)

    # 构建数据加载器
    train_loader, val_loader = build_dataloaders(
        train_x, train_y, val_x, val_y, cfg.training.batch_size
    )

    # 创建模型
    model = SurfaceCodeConv3DDecoder(cfg.model)

    # 训练
    trainer = Trainer(model, cfg.training)
    report = trainer.train(train_loader, val_loader)

    # 添加数据配置信息到报告
    report["data_config"] = asdict(cfg.data)
    report["model_config"] = asdict(cfg.model)
    report["data_shape"] = {
        "input_shape": list(train_x.shape[1:]),
        "output_shape": list(train_y.shape[1:]),
    }

    # 保存完整报告
    out_dir = Path(cfg.training.out_dir)
    (out_dir / "full_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report


def _parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="Ising-style training experiment (new modules)")
    p.add_argument("--config", type=str, default="experiments/ising/ising_train.yaml")
    # 允许覆盖部分参数
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


def _override_config(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """用命令行参数覆盖配置"""
    # 数据配置覆盖
    if args.distance is not None:
        cfg.data.distance = args.distance
    if args.rounds is not None:
        cfg.data.rounds = args.rounds
    if args.basis is not None:
        cfg.data.basis = args.basis
    if args.train_shots is not None:
        cfg.data.train_shots = args.train_shots
    if args.val_shots is not None:
        cfg.data.val_shots = args.val_shots

    # 训练配置覆盖
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.out_dir is not None:
        cfg.training.out_dir = args.out_dir

    return cfg


def main() -> None:
    """主函数"""
    args = _parse_args()
    cfg = _load_config(args.config if args.config and Path(args.config).exists() else None)
    cfg = _override_config(cfg, args)

    report = run_training(cfg)

    print(
        f"[Train] done. best_val_loss={report['best_val_loss']:.6f}, "
        f"ckpt={report['best_checkpoint']}"
    )


if __name__ == "__main__":
    main()
