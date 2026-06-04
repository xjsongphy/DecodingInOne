#!/usr/bin/env python3
"""Ising-style training experiment."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from decoding_in_one.circuits import MemoryCircuit
from decoding_in_one.codes import SurfaceCode
from decoding_in_one.data import IsingDataConfig
from decoding_in_one.models import Conv3DModelConfig, SurfaceCodeConv3DDecoder
from decoding_in_one.noise import NoiseModel
from decoding_in_one.sampling.dem import custab_available
from decoding_in_one.sampling.generator import CircuitDataGenerator
from decoding_in_one.training import OptimConfig, Trainer


@dataclass
class ExperimentConfig:
    data: IsingDataConfig
    model: Conv3DModelConfig
    training: OptimConfig


class MemmapDataset(Dataset):
    """Disk-backed dataset that reads .npy arrays lazily."""

    def __init__(self, x_path: Path, y_path: Path) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.asarray(self.x[idx], dtype=np.float32))
        y = torch.from_numpy(np.asarray(self.y[idx], dtype=np.float32))
        return x, y


DATA_CONFIG_FILENAME = "data_config.json"
DATA_INFO_FILENAME = "dataset_info.json"


def _load_config(config_path: str | None) -> ExperimentConfig:
    if not config_path:
        return ExperimentConfig(
            data=IsingDataConfig(),
            model=Conv3DModelConfig(),
            training=OptimConfig(),
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    data_config = IsingDataConfig(
        **{k: v for k, v in raw.items() if k in IsingDataConfig.__dataclass_fields__}
    )
    model_config = Conv3DModelConfig(
        **{k: v for k, v in raw.items() if k in Conv3DModelConfig.__dataclass_fields__}
    )
    training_config = OptimConfig(
        **{k: v for k, v in raw.items() if k in OptimConfig.__dataclass_fields__}
    )
    return ExperimentConfig(data=data_config, model=model_config, training=training_config)


def _build_noise_model(cfg: IsingDataConfig) -> NoiseModel:
    if cfg.noise_model_path:
        return NoiseModel.from_config(cfg.noise_model_path)
    return NoiseModel.from_config_dict(cfg.get_25p_noise_params())


def _basis_to_circuit_basis(basis: str) -> str:
    b = basis.strip().upper()
    if b in ("O1", "O2", "X"):
        return "X"
    if b in ("O3", "O4", "Z"):
        return "Z"
    raise ValueError(f"basis must be O1-O4 or X/Z, got {basis}")


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


def _normalize_optional_path(path_str: str | None) -> str | None:
    if not path_str:
        return None
    return str(Path(path_str).expanduser().resolve())


def _build_data_cache_config(
    data_cfg: IsingDataConfig,
    noise_model: NoiseModel,
    *,
    seed: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "code_type": "surface_code",
        "distance": int(data_cfg.distance),
        "rounds": int(data_cfg.rounds),
        "basis": data_cfg.basis.upper(),
        "rotation": data_cfg.get_rotation(),
        "circuit_basis": _basis_to_circuit_basis(data_cfg.basis),
        "train_shots": int(data_cfg.train_shots),
        "val_shots": int(data_cfg.val_shots),
        "train_seed": int(seed),
        "val_seed": int(seed + 1_000_000),
        "precomputed_frames_dir": _normalize_optional_path(data_cfg.precomputed_frames_dir),
        "noise_model_path": _normalize_optional_path(data_cfg.noise_model_path),
        "noise_model": noise_model.canonical_parameters(),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _dataset_files_exist(dataset_dir: Path) -> bool:
    required = ["trainX.npy", "trainY.npy", "valX.npy", "valY.npy", DATA_CONFIG_FILENAME]
    return all((dataset_dir / name).exists() for name in required)


def _find_matching_dataset_dir(cache_root: Path, expected_config: dict[str, Any]) -> Path | None:
    if not cache_root.exists():
        return None

    expected_json = _canonical_json(expected_config)
    candidates = sorted((p for p in cache_root.iterdir() if p.is_dir()), reverse=True)
    for dataset_dir in candidates:
        if not _dataset_files_exist(dataset_dir):
            continue
        cached_config = _read_json(dataset_dir / DATA_CONFIG_FILENAME)
        if cached_config is None:
            continue
        if _canonical_json(cached_config) == expected_json:
            return dataset_dir
    return None


def _create_dataset_dir(cache_root: Path) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = cache_root / timestamp
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=False)
            return dataset_dir


def _write_dataset_metadata(
    dataset_dir: Path,
    *,
    data_config: dict[str, Any],
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> None:
    (dataset_dir / DATA_CONFIG_FILENAME).write_text(
        _canonical_json(data_config),
        encoding="utf-8",
    )
    dataset_info = {
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "files": {
            "train_x": "trainX.npy",
            "train_y": "trainY.npy",
            "val_x": "valX.npy",
            "val_y": "valY.npy",
        },
    }
    (dataset_dir / DATA_INFO_FILENAME).write_text(
        _canonical_json(dataset_info),
        encoding="utf-8",
    )


def _prepare_cached_datasets(
    *,
    generator: CircuitDataGenerator,
    data_cfg: IsingDataConfig,
    training_seed: int,
    batch_size: int,
    cache_root: Path,
    noise_model: NoiseModel,
) -> tuple[Path, Path, Path, Path, tuple[int, ...], tuple[int, ...], Path, bool]:
    cache_config = _build_data_cache_config(data_cfg, noise_model, seed=training_seed)
    matched_dir = _find_matching_dataset_dir(cache_root, cache_config)
    if matched_dir is not None:
        print(f"[Train] Reusing cached dataset: {matched_dir}")
        train_x_path = matched_dir / "trainX.npy"
        train_y_path = matched_dir / "trainY.npy"
        val_x_path = matched_dir / "valX.npy"
        val_y_path = matched_dir / "valY.npy"
        input_shape = tuple(np.load(train_x_path, mmap_mode="r").shape[1:])
        output_shape = tuple(np.load(train_y_path, mmap_mode="r").shape[1:])
        return (
            train_x_path,
            train_y_path,
            val_x_path,
            val_y_path,
            input_shape,
            output_shape,
            matched_dir,
            True,
        )

    dataset_dir = _create_dataset_dir(cache_root)
    print(f"[Train] No matching cached dataset found; generating into: {dataset_dir}")

    chunk_size = max(int(data_cfg.sampling_chunk_size), int(batch_size))
    print(
        f"[Train] Pre-generating datasets in chunks: "
        f"train_samples={data_cfg.train_shots}, val_samples={data_cfg.val_shots}, "
        f"chunk_size={chunk_size}"
    )
    if data_cfg.enable_parallel:
        print(
            f"[Train] Parallel sampling requested: workers={data_cfg.num_workers}, "
            f"chunk_size={chunk_size}"
        )

    train_x_path = dataset_dir / "trainX.npy"
    train_y_path = dataset_dir / "trainY.npy"
    val_x_path = dataset_dir / "valX.npy"
    val_y_path = dataset_dir / "valY.npy"

    train_x_path, train_y_path, input_shape, output_shape = _pregenerate_to_memmap(
        generator=generator,
        total_samples=data_cfg.train_shots,
        chunk_size=chunk_size,
        seed_base=training_seed,
        out_x=train_x_path,
        out_y=train_y_path,
        label="train",
    )
    _pregenerate_to_memmap(
        generator=generator,
        total_samples=data_cfg.val_shots,
        chunk_size=chunk_size,
        seed_base=training_seed + 1_000_000,
        out_x=val_x_path,
        out_y=val_y_path,
        label="val",
    )
    _write_dataset_metadata(
        dataset_dir,
        data_config=cache_config,
        input_shape=input_shape,
        output_shape=output_shape,
    )
    return (
        train_x_path,
        train_y_path,
        val_x_path,
        val_y_path,
        input_shape,
        output_shape,
        dataset_dir,
        False,
    )


def _pregenerate_to_memmap(
    *,
    generator: CircuitDataGenerator,
    total_samples: int,
    chunk_size: int,
    seed_base: int,
    out_x: Path,
    out_y: Path,
    label: str,
) -> tuple[Path, Path, tuple[int, ...], tuple[int, ...]]:
    probe = generator.generate_batch(
        batch_size=min(chunk_size, total_samples, 4),
        seed=seed_base,
        keep_on_cpu=True,
    )
    sample_x = probe["trainX"].cpu().numpy().astype(np.float16, copy=False)
    sample_y = probe["trainY"].cpu().numpy().astype(np.float16, copy=False)
    x_shape = (total_samples, *sample_x.shape[1:])
    y_shape = (total_samples, *sample_y.shape[1:])

    x_mm = np.lib.format.open_memmap(out_x, mode="w+", dtype=np.float16, shape=x_shape)
    y_mm = np.lib.format.open_memmap(out_y, mode="w+", dtype=np.float16, shape=y_shape)

    written = 0
    chunk_idx = 0
    num_chunks = math.ceil(total_samples / chunk_size)
    pbar = tqdm(total=total_samples, desc=f"Sampling {label}", unit="samples", leave=True)
    while written < total_samples:
        current = min(chunk_size, total_samples - written)
        batch = generator.generate_batch(
            batch_size=current,
            seed=seed_base + chunk_idx,
            keep_on_cpu=True,
            verbose=False,  # 由 tqdm 进度条管理输出
        )
        batch_x = batch["trainX"].cpu().numpy().astype(np.float16, copy=False)
        batch_y = batch["trainY"].cpu().numpy().astype(np.float16, copy=False)
        x_mm[written:written + current] = batch_x
        y_mm[written:written + current] = batch_y
        written += current
        chunk_idx += 1
        x_mm.flush()
        y_mm.flush()
        pbar.update(current)
        pbar.set_postfix({"chunk": f"{chunk_idx}/{num_chunks}"})

    pbar.close()
    del x_mm
    del y_mm
    return out_x, out_y, x_shape[1:], y_shape[1:]


def run_training(cfg: ExperimentConfig) -> dict[str, Any]:
    run_dir = Path(cfg.training.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Train] Output dir: {run_dir}")

    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    data_cfg = cfg.data
    rotation = data_cfg.get_rotation()
    circuit_basis = _basis_to_circuit_basis(data_cfg.basis)

    code = SurfaceCode(distance=data_cfg.distance, rotation=rotation)
    noise_model = _build_noise_model(data_cfg)
    print(f"[Train] QuantumCode: SurfaceCode(distance={data_cfg.distance}, rotation={rotation})")
    print(f"[Train] NoiseModel: {noise_model}")

    mem_circuit = MemoryCircuit(
        code=code,
        n_rounds=data_cfg.rounds,
        basis=circuit_basis,
        noise_model=noise_model,
    )
    print(f"[Train] MemoryCircuit built: {len(mem_circuit.circuit)} chars")

    stim_circuit = mem_circuit.compile_to_stim()
    print(f"[Train] Stim circuit compiled: {len(stim_circuit)} instructions")

    use_precomputed = data_cfg.precomputed_frames_dir is not None
    if use_precomputed:
        print(f"[Train] Using precomputed DEM from: {data_cfg.precomputed_frames_dir}")

    sampling_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if sampling_device.type == "cuda":
        print("[Train] Using CUDA device for sampling")
        if custab_available():
            print("[Train] cuQuantum detected; DEM sampling can run on GPU")
        else:
            print("[Train] cuQuantum not detected; DEM sampling will fall back to CPU")

    generator = CircuitDataGenerator(
        distance=data_cfg.distance,
        n_rounds=data_cfg.rounds,
        basis=circuit_basis,
        code_rotation=rotation,
        stim_circuit=stim_circuit,
        allow_stim_dem_extraction=True,
        precomputed_frames_dir=data_cfg.precomputed_frames_dir if use_precomputed else None,
        enable_parallel=data_cfg.enable_parallel,
        num_workers=data_cfg.num_workers,
        device_ids=data_cfg.parallel_device_ids,
        device=sampling_device,
    )

    if not use_precomputed:
        print("[Train] DEM extracted from Stim circuit (no external dependency)")
    data_cache_root = Path(cfg.training.out_dir) / "data"
    print(f"[Train] Dataset cache root: {data_cache_root}")
    (
        train_x_path,
        train_y_path,
        val_x_path,
        val_y_path,
        input_shape,
        output_shape,
        dataset_dir,
        reused_dataset,
    ) = _prepare_cached_datasets(
        generator=generator,
        data_cfg=data_cfg,
        training_seed=cfg.training.seed,
        batch_size=cfg.training.batch_size,
        cache_root=data_cache_root,
        noise_model=noise_model,
    )
    if not data_cfg.save_samples:
        print("[Train] save_samples=false is ignored because dataset caching is always enabled")

    train_ds = MemmapDataset(train_x_path, train_y_path)
    val_ds = MemmapDataset(val_x_path, val_y_path)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.val_batch_size, shuffle=False, drop_last=False)

    model = SurfaceCodeConv3DDecoder(cfg.model)
    trainer = Trainer(model, cfg.training)
    trainer.run_dir = str(run_dir)
    report = trainer.train(train_loader, val_loader)

    report["data_config"] = asdict(data_cfg)
    report["model_config"] = asdict(cfg.model)
    report["data_shape"] = {
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
    }
    report["architecture"] = "ising_decoding"
    report["dataset_dir"] = str(dataset_dir)
    report["dataset_reused"] = reused_dataset

    (run_dir / "full_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Train] All outputs saved to: {run_dir}")
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ising-style training experiment")
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


def _override_config(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
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
