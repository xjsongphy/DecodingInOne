from pathlib import Path

from decoding_in_one.data import IsingDataConfig
from decoding_in_one.noise import NoiseModel
from experiments.ising.ising_train_model import (
    DATA_CONFIG_FILENAME,
    _build_data_cache_config,
    _find_matching_dataset_dir,
    _write_dataset_metadata,
)


def _make_dataset_dir(root: Path, name: str) -> Path:
    dataset_dir = root / name
    dataset_dir.mkdir(parents=True)
    for filename in ("trainX.npy", "trainY.npy", "valX.npy", "valY.npy"):
        (dataset_dir / filename).write_bytes(b"placeholder")
    return dataset_dir


def test_build_data_cache_config_resolves_noise_and_seed() -> None:
    cfg = IsingDataConfig(distance=7, rounds=9, basis="O3", train_shots=12, val_shots=4)
    noise_model = NoiseModel.from_config_dict(cfg.get_25p_noise_params())

    cache_config = _build_data_cache_config(cfg, noise_model, seed=123)

    assert cache_config["distance"] == 7
    assert cache_config["rounds"] == 9
    assert cache_config["basis"] == "O3"
    assert cache_config["rotation"] == "ZV"
    assert cache_config["circuit_basis"] == "Z"
    assert cache_config["train_shots"] == 12
    assert cache_config["val_shots"] == 4
    assert cache_config["train_seed"] == 123
    assert cache_config["val_seed"] == 1_000_123
    assert cache_config["noise_model"] == noise_model.canonical_parameters()


def test_find_matching_dataset_dir_uses_saved_config(tmp_path: Path) -> None:
    cfg = IsingDataConfig(train_shots=16, val_shots=8)
    noise_model = NoiseModel.from_config_dict(cfg.get_25p_noise_params())
    expected_config = _build_data_cache_config(cfg, noise_model, seed=5)

    older_dir = _make_dataset_dir(tmp_path, "20240101_000000")
    newer_dir = _make_dataset_dir(tmp_path, "20240101_000001")

    (older_dir / DATA_CONFIG_FILENAME).write_text("{}", encoding="utf-8")
    _write_dataset_metadata(
        newer_dir,
        data_config=expected_config,
        input_shape=(4, 5, 5, 5),
        output_shape=(4, 5, 5, 5),
    )

    matched = _find_matching_dataset_dir(tmp_path, expected_config)

    assert matched == newer_dir
