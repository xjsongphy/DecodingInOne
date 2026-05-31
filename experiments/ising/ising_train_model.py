#!/usr/bin/env python3
"""Ising-style training experiment（基于 Ising-Decoding 架构）

数据流:
  NoiseModel (25p) + QuantumCode (通用化，支持 SurfaceCode/QLDPC)
      ↓
  MemoryCircuit → Stim 电路字符串（通用化稳定子测量）
      ↓
  预计算 → H, p, A 矩阵 (.npz)  [如果可用]
      ↓
  CircuitDataGenerator.generate_batch()
      ↓
  dem_sampling(H, p) → GPU/CPU 采样
      ↓
  _format_for_model() → trainX, trainY
      ↓
  DataLoader → Trainer → 模型

架构说明:
  - 移除了 legacy stim.Circuit.generated() 方式
  - 统一使用 MemoryCircuit + NoiseModel (25参数)
  - 支持任意 QuantumCode 实现（SurfaceCode、QLDPC等）
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# 新架构
from decoding_in_one.codes import SurfaceCode
from decoding_in_one.noise import NoiseModel
from decoding_in_one.circuits import MemoryCircuit
from decoding_in_one.sampling.generator import CircuitDataGenerator

# 训练相关
from decoding_in_one.models import SurfaceCodeConv3DDecoder, Conv3DModelConfig
from decoding_in_one.training import Trainer, OptimConfig
from decoding_in_one.data import IsingDataConfig, build_dataloaders


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
        return ExperimentConfig(
            data=IsingDataConfig(),
            model=Conv3DModelConfig(),
            training=OptimConfig()
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    data_config = IsingDataConfig(**{k: v for k, v in raw.items() if k in IsingDataConfig.__dataclass_fields__})
    model_config = Conv3DModelConfig(**{k: v for k, v in raw.items() if k in Conv3DModelConfig.__dataclass_fields__})
    training_config = OptimConfig(**{k: v for k, v in raw.items() if k in OptimConfig.__dataclass_fields__})

    return ExperimentConfig(data=data_config, model=model_config, training=training_config)


def _build_noise_model(cfg: IsingDataConfig) -> NoiseModel:
    """从配置构建 25 参数噪声模型"""
    if cfg.noise_model_path:
        return NoiseModel.from_config(cfg.noise_model_path)
    return NoiseModel.from_config_dict(cfg.get_25p_noise_params())


def _basis_to_circuit_basis(basis: str) -> str:
    """将 O1-O4/X/Z 映射到电路测量基 ('X' 或 'Z')"""
    b = basis.strip().upper()
    if b in ("O1", "O2", "X"):
        return "X"
    if b in ("O3", "O4", "Z"):
        return "Z"
    raise ValueError(f"basis must be O1-O4 or X/Z, got {basis}")


def run_training(cfg: ExperimentConfig) -> dict[str, Any]:
    """使用 Ising-Decoding 架构执行训练

    步骤:
    1. 从 IsingDataConfig 构建 QuantumCode + NoiseModel
    2. 用 MemoryCircuit 构建 Stim 电路（通用化稳定子测量）
    3. 用 CircuitDataGenerator 采样训练数据
    4. 训练模型
    """
    # 设置随机种子
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    data_cfg = cfg.data
    rotation = data_cfg.get_rotation()  # O1→XV, O2→XH, ...
    circuit_basis = _basis_to_circuit_basis(data_cfg.basis)

    # 1. 构建码和噪声模型（支持任意 QuantumCode 实现）
    code = SurfaceCode(distance=data_cfg.distance, rotation=rotation)
    noise_model = _build_noise_model(data_cfg)

    print(f"[Train] QuantumCode: SurfaceCode(distance={data_cfg.distance}, rotation={rotation})")
    print(f"[Train] NoiseModel: {noise_model}")

    # 2. 构建 MemoryCircuit（验证电路正确性，使用通用化稳定子测量）
    mem_circuit = MemoryCircuit(
        code=code,
        n_rounds=data_cfg.rounds,
        basis=circuit_basis,
        noise_model=noise_model,
    )
    print(f"[Train] MemoryCircuit built: {len(mem_circuit.circuit)} chars")

    # 3. 从 MemoryCircuit 获取 Stim 电路
    stim_circuit = mem_circuit.compile_to_stim()
    print(f"[Train] Stim circuit compiled: {len(stim_circuit)} instructions")

    # 4. 用 CircuitDataGenerator 从 Stim 电路自动提取 DEM（完全独立）
    # 优先级: stim_circuit > precomputed_frames_dir
    use_precomputed = data_cfg.precomputed_frames_dir is not None

    if use_precomputed:
        print(f"[Train] Using precomputed DEM from: {data_cfg.precomputed_frames_dir}")

    generator = CircuitDataGenerator(
        distance=data_cfg.distance,
        n_rounds=data_cfg.rounds,
        basis=circuit_basis,
        code_rotation=rotation,
        stim_circuit=stim_circuit,
        allow_stim_dem_extraction=True,  # 启用从 Stim 电路自动提取 DEM
        precomputed_frames_dir=data_cfg.precomputed_frames_dir if use_precomputed else None,
        device=torch.device("cpu"),
    )

    if not use_precomputed:
        print(f"[Train] DEM extracted from Stim circuit (no external dependency)")
    train_batch = generator.generate_batch(
        batch_size=data_cfg.train_shots,
        seed=cfg.training.seed,
    )
    val_batch = generator.generate_batch(
        batch_size=data_cfg.val_shots,
        seed=cfg.training.seed + 1,
    )

    # 4. 数据预处理（转为 dataloader 需要的 numpy）
    train_x = train_batch["trainX"].cpu().numpy().astype(np.float32, copy=False)
    train_y = train_batch["trainY"].cpu().numpy().astype(np.float32, copy=False)
    val_x = val_batch["trainX"].cpu().numpy().astype(np.float32, copy=False)
    val_y = val_batch["trainY"].cpu().numpy().astype(np.float32, copy=False)

    print(f"[Train] Data shapes: train_x={train_x.shape}, train_y={train_y.shape}")

    # 5. 构建数据加载器
    train_loader, val_loader = build_dataloaders(
        train_x, train_y, val_x, val_y, cfg.training.batch_size
    )

    # 6. 创建模型
    model = SurfaceCodeConv3DDecoder(cfg.model)

    # 7. 训练
    trainer = Trainer(model, cfg.training)
    report = trainer.train(train_loader, val_loader)

    # 添加配置信息到 Trainer 的报告中
    report["data_config"] = asdict(data_cfg)
    report["model_config"] = asdict(cfg.model)
    report["data_shape"] = {
        "input_shape": list(train_x.shape[1:]),
        "output_shape": list(train_y.shape[1:]),
    }
    report["architecture"] = "ising_decoding"  # 统一使用 Ising-Decoding 架构

    # 保存完整报告到 Trainer 的时间戳目录
    run_dir = Path(report["run_dir"])
    (run_dir / "full_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[Train] All outputs saved to: {run_dir}")

    return report


def _parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="Ising-style training experiment")
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
