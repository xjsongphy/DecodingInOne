from pathlib import Path
import tempfile

from decoding_in_one.data import IsingDataConfig
from decoding_in_one.noise import NoiseModel
from experiments.ising.ising_train_model import (
    DATA_CONFIG_FILENAME,
    _basis_to_circuit_basis,
    _build_data_cache_config,
    _find_matching_dataset_dir,
    _load_config,
    _write_dataset_metadata,
    run_training,
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
    assert cache_config["measurement_basis"] == "Z"
    assert cache_config["rotation"] == "ZV"
    assert cache_config["circuit_basis"] == "Z"
    assert cache_config["train_shots"] == 12
    assert cache_config["val_shots"] == 4
    assert cache_config["train_seed"] == 123
    assert cache_config["val_seed"] == 1_000_123
    assert cache_config["noise_model"] == noise_model.canonical_parameters()


def test_build_data_cache_config_supports_both_basis() -> None:
    cfg = IsingDataConfig(
        distance=9,
        rounds=9,
        basis="both",
        rotation="XV",
        train_shots=32,
        val_shots=8,
    )
    noise_model = NoiseModel.from_config_dict(cfg.get_25p_noise_params())

    cache_config = _build_data_cache_config(cfg, noise_model, seed=7)

    assert cache_config["basis"] == "BOTH"
    assert cache_config["measurement_basis"] == "BOTH"
    assert cache_config["rotation"] == "XV"
    assert cache_config["circuit_basis"] == "both"


def test_find_matching_dataset_dir_uses_saved_config() -> None:
    with tempfile.TemporaryDirectory(prefix="dio_cache_test_") as tmpdir:
        tmp_path = Path(tmpdir)
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


def test_build_data_cache_config_tracks_current_semantics_version() -> None:
    cfg = IsingDataConfig()
    noise_model = NoiseModel.from_config_dict(cfg.get_25p_noise_params())
    cache_config = _build_data_cache_config(cfg, noise_model, seed=0)
    assert cache_config["format_version"] == 2
    assert cache_config["data_semantics"] == "ising_frame_predecoder_he_v1"
    assert cache_config["allow_stim_dem_extraction"] is False


def test_run_training_requires_precomputed_frames_by_default() -> None:
    cfg = _load_config(None)
    cfg.data.precomputed_frames_dir = None
    cfg.data.allow_stim_dem_extraction = False
    try:
        run_training(cfg)
    except ValueError as exc:
        assert "precomputed frame_predecoder DEM artifacts" in str(exc)
    else:
        raise AssertionError("run_training should reject missing precomputed_frames_dir by default")


def test_basis_to_circuit_basis_supports_both() -> None:
    assert _basis_to_circuit_basis("both") == "both"
    assert _basis_to_circuit_basis("mixed") == "both"
