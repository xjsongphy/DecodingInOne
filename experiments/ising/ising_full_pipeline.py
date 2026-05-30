#!/usr/bin/env python3
"""NVIDIA Ising-style end-to-end inference pipeline (extracted minimal flow).

Pipeline stages:
1) Load config (YAML + CLI overrides)
2) Build code/noise metadata
3) Generate surface-code memory circuit (Stim)
4) Sample detectors/observables from circuit
5) Baseline decode with PyMatching
6) Predecoder hook + decode-after-predecoder
7) Report LER/latency/speedup and save artifacts
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

from decoding_in_one.codes import SurfaceCode
from decoding_in_one.noise import CircuitLevelNoise


@dataclass
class PipelineConfig:
    distance: int = 5
    rounds: int = 5
    basis: str = "X"  # X or Z
    shots: int = 20000
    p_after_clifford: float = 0.001
    p_before_round_data: float = 0.001
    p_before_measure_flip: float = 0.001
    p_after_reset_flip: float = 0.001
    seed: int = 0
    latency_samples: int = 5000
    out_dir: str = "exp/output"
    save_circuit: bool = True


def _load_config(config_path: str | None) -> PipelineConfig:
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
    b = basis.strip().upper()
    if b not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")
    return f"surface_code:rotated_memory_{b.lower()}"


def _build_stim_circuit(cfg: PipelineConfig) -> stim.Circuit:
    return stim.Circuit.generated(
        _task_from_basis(cfg.basis),
        distance=cfg.distance,
        rounds=cfg.rounds,
        after_clifford_depolarization=cfg.p_after_clifford,
        before_round_data_depolarization=cfg.p_before_round_data,
        before_measure_flip_probability=cfg.p_before_measure_flip,
        after_reset_flip_probability=cfg.p_after_reset_flip,
    )


def _sample(circuit: stim.Circuit, shots: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    return np.asarray(dets, dtype=np.uint8), np.asarray(obs, dtype=np.uint8)


def _decode_batch(dem: stim.DetectorErrorModel, dets: np.ndarray) -> np.ndarray:
    matcher = pymatching.Matching.from_detector_error_model(dem)
    return matcher.decode_batch(dets)


def _latency_us_per_shot(dem: stim.DetectorErrorModel, dets: np.ndarray, n_samples: int) -> float:
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
    mismatches = np.any(pred_obs != true_obs, axis=1)
    return float(np.mean(mismatches))


def _identity_predecoder(dets: np.ndarray) -> np.ndarray:
    """Hook for replacing with learned predecoder later."""
    return dets


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep framework-side metadata aligned with existing modules.
    _ = SurfaceCode(distance=cfg.distance, rotation="XV")
    _ = CircuitLevelNoise(
        p_prep_X=cfg.p_after_reset_flip,
        p_prep_Z=cfg.p_after_reset_flip,
        p_meas_X=cfg.p_before_measure_flip,
        p_meas_Z=cfg.p_before_measure_flip,
        p_idle_cnot_X=cfg.p_before_round_data / 3,
        p_idle_cnot_Y=cfg.p_before_round_data / 3,
        p_idle_cnot_Z=cfg.p_before_round_data / 3,
    )

    circuit = _build_stim_circuit(cfg)
    if cfg.save_circuit:
        (out_dir / "generated_surface_code.stim").write_text(str(circuit), encoding="utf-8")

    dets, obs = _sample(circuit, shots=cfg.shots, seed=cfg.seed)
    dem = circuit.detector_error_model(decompose_errors=True)

    baseline_pred = _decode_batch(dem, dets)
    baseline_ler = _logical_error_rate(baseline_pred, obs)

    dets_after_predecoder = _identity_predecoder(dets)
    after_pred = _decode_batch(dem, dets_after_predecoder)
    after_ler = _logical_error_rate(after_pred, obs)

    baseline_latency = _latency_us_per_shot(dem, dets, cfg.latency_samples)
    after_latency = _latency_us_per_shot(dem, dets_after_predecoder, cfg.latency_samples)
    speedup = baseline_latency / after_latency if after_latency > 0 else float("nan")

    report = {
        "config": asdict(cfg),
        "stats": {
            "num_detectors": int(dets.shape[1]),
            "num_observables": int(obs.shape[1]),
            "shots": int(cfg.shots),
        },
        "metrics": {
            "ler_baseline": baseline_ler,
            "ler_after_predecoder": after_ler,
            "pymatching_latency_baseline_us_per_shot": baseline_latency,
            "pymatching_latency_after_predecoder_us_per_shot": after_latency,
            "pymatching_speedup": speedup,
        },
    }

    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extracted NVIDIA Ising-style full pipeline")
    p.add_argument("--config", type=str, default="experiments/ising/ising_pipeline.yaml")
    p.add_argument("--distance", type=int)
    p.add_argument("--rounds", type=int)
    p.add_argument("--basis", type=str)
    p.add_argument("--shots", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--latency-samples", type=int)
    p.add_argument("--out-dir", type=str)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config if args.config and Path(args.config).exists() else None)

    for key in ["distance", "rounds", "basis", "shots", "seed", "latency_samples", "out_dir"]:
        val = getattr(args, key)
        if val is not None:
            setattr(cfg, key, val)

    report = run_pipeline(cfg)

    m = report["metrics"]
    print(f"[Pipeline] shots={report['stats']['shots']}, dets={report['stats']['num_detectors']}, obs={report['stats']['num_observables']}")
    print(f"[Pipeline] LER baseline={m['ler_baseline']:.6f}, after_predecoder={m['ler_after_predecoder']:.6f}")
    print(
        "[Pipeline] PyMatching latency baseline/after "
        f"= {m['pymatching_latency_baseline_us_per_shot']:.3f} / "
        f"{m['pymatching_latency_after_predecoder_us_per_shot']:.3f} us/shot, "
        f"speedup={m['pymatching_speedup']:.3f}x"
    )


if __name__ == "__main__":
    main()
