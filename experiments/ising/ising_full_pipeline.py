#!/usr/bin/env python3
"""Ising-style 推理流程（使用新模块）.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pymatching
import stim
import yaml

from decoding_in_one.decoders import NeuralDecoder
from decoding_in_one.models import SurfaceCodeConv3DDecoder, Conv3DModelConfig
from decoding_in_one.data import IsingDataConfig, sample_detectors_observables


@dataclass
class PipelineConfig:
    """推理流程配置"""
    distance: int = 5
    rounds: int = 5
    basis: str = "O1"  # O1, O2, O3, O4
    shots: int = 20000
    p_after_clifford: float = 0.001
    p_before_round_data: float = 0.001
    p_before_measure_flip: float = 0.001
    p_after_reset_flip: float = 0.001
    seed: int = 0
    latency_samples: int = 5000
    out_dir: str = "outputs/ising/pipeline"  # 根目录下的 outputs
    checkpoint_path: str = "outputs/ising/train/best_model.pt"
    save_circuit: bool = True


def _load_config(config_path: str | None) -> PipelineConfig:
    """加载配置"""
    if not config_path:
        return PipelineConfig()
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    defaults = asdict(PipelineConfig())
    for k, v in raw.items():
        if k in defaults:
            defaults[k] = v
    return PipelineConfig(**defaults)


def _task_from_basis(basis: str) -> str:
    """从测量基获取 Stim 任务名称"""
    b = basis.strip().upper()
    if b not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")
    return f"surface_code:rotated_memory_{b.lower()}"


def _build_stim_circuit(cfg: PipelineConfig) -> stim.Circuit:
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


def _decode_batch(dem: stim.DetectorErrorModel, dets: np.ndarray) -> np.ndarray:
    """PyMatching 批量解码"""
    matcher = pymatching.Matching.from_detector_error_model(dem)
    return matcher.decode_batch(dets)


def _latency_us_per_shot(dem: stim.DetectorErrorModel, dets: np.ndarray, n_samples: int) -> float:
    """计算 PyMatching 延迟"""
    matcher = pymatching.Matching.from_detector_error_model(dem)
    n = min(n_samples, len(dets))
    if n <= 0:
        return float("nan")
    start = time.perf_counter()
    for i in range(n):
        matcher.decode(dets[i])
    elapsed = time.perf_counter() - start
    return elapsed / n * 1e6


def _logical_error_rate(pred_obs: np.ndarray, true_obs: np.ndarray) -> float:
    """计算逻辑错误率"""
    mismatches = np.any(pred_obs != true_obs, axis=1)
    return float(np.mean(mismatches))


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    """执行推理流程"""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 构建电路
    circuit = _build_stim_circuit(cfg)
    if cfg.save_circuit:
        (out_dir / "generated_surface_code.stim").write_text(str(circuit), encoding="utf-8")

    # 采样数据
    dets, obs = sample_detectors_observables(circuit, shots=cfg.shots, seed=cfg.seed)
    dem = circuit.detector_error_model(decompose_errors=True)

    # 创建 NeuralDecoder
    model = SurfaceCodeConv3DDecoder(Conv3DModelConfig())
    decoder = NeuralDecoder(
        model=model,
        checkpoint_path=cfg.checkpoint_path,
        rounds=cfg.rounds,
        distance=cfg.distance,
        reduce_output=True,  # 聚合为 observable 向量
        reduce_method="mean"
    )

    # NeuralDecoder 推理
    neural_pred = decoder.decode(dets)
    neural_ler = _logical_error_rate(neural_pred.predictions.cpu().numpy(), obs)

    # PyMatching baseline
    baseline_pred = _decode_batch(dem, dets)
    baseline_ler = _logical_error_rate(baseline_pred, obs)

    # 延迟测试
    baseline_latency = _latency_us_per_shot(dem, dets, cfg.latency_samples)

    report = {
        "config": asdict(cfg),
        "stats": {
            "num_detectors": int(dets.shape[1]),
            "num_observables": int(obs.shape[1]),
            "shots": int(cfg.shots),
        },
        "metrics": {
            "ler_baseline": baseline_ler,
            "ler_neural": neural_ler,
            "pymatching_latency_baseline_us_per_shot": baseline_latency,
        },
    }

    (out_dir / "pipeline_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="Ising-style inference pipeline (new modules)")
    p.add_argument("--config", type=str, default="experiments/ising/ising_pipeline.yaml")
    p.add_argument("--distance", type=int)
    p.add_argument("--rounds", type=int)
    p.add_argument("--basis", type=str)
    p.add_argument("--shots", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--checkpoint-path", type=str)
    p.add_argument("--out-dir", type=str)
    return p.parse_args()


def main() -> None:
    """主函数"""
    args = _parse_args()
    cfg = _load_config(args.config if args.config and Path(args.config).exists() else None)

    for key in ["distance", "rounds", "basis", "shots", "seed", "checkpoint_path", "out_dir"]:
        val = getattr(args, key)
        if val is not None:
            setattr(cfg, key, val)

    report = run_pipeline(cfg)

    m = report["metrics"]
    print(f"[Pipeline] shots={report['stats']['shots']}, dets={report['stats']['num_detectors']}, obs={report['stats']['num_observables']}")
    print(f"[Pipeline] LER baseline={m['ler_baseline']:.6f}, neural={m['ler_neural']:.6f}")
    print(f"[Pipeline] PyMatching latency={m['pymatching_latency_baseline_us_per_shot']:.3f} us/shot")


if __name__ == "__main__":
    main()
