from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class IsingDataConfig:
    """Configuration for local Ising-style data generation."""

    distance: int = 5
    rounds: int = 5
    basis: str = "O1"  # O1, O2, O3, O4, X, Z, both, mixed
    rotation: Optional[str] = None  # XV, XH, ZV, ZH; required for both/mixed
    train_shots: int = 20000
    val_shots: int = 5000

    noise_model_path: Optional[str] = None
    precomputed_frames_dir: Optional[str] = None
    # Debug-only fallback: Stim DEM extraction does not reproduce
    # Ising-Decoding's frame_predecoder semantics.
    allow_stim_dem_extraction: bool = False

    # Legacy 4-parameter interface retained for compatibility.
    p_after_clifford: float = 0.001
    p_before_round_data: float = 0.001
    p_before_measure_flip: float = 0.001
    p_after_reset_flip: float = 0.001

    # Full 25-parameter circuit-level noise model.
    p_prep_X: float = 0.002
    p_prep_Z: float = 0.002
    p_meas_X: float = 0.002
    p_meas_Z: float = 0.002
    p_idle_cnot_X: float = 0.001
    p_idle_cnot_Y: float = 0.001
    p_idle_cnot_Z: float = 0.001
    p_idle_spam_X: float = 0.001996
    p_idle_spam_Y: float = 0.001996
    p_idle_spam_Z: float = 0.001996
    p_cnot_IX: float = 0.0002
    p_cnot_IY: float = 0.0002
    p_cnot_IZ: float = 0.0002
    p_cnot_XI: float = 0.0002
    p_cnot_XX: float = 0.0002
    p_cnot_XY: float = 0.0002
    p_cnot_XZ: float = 0.0002
    p_cnot_YI: float = 0.0002
    p_cnot_YX: float = 0.0002
    p_cnot_YY: float = 0.0002
    p_cnot_YZ: float = 0.0002
    p_cnot_ZI: float = 0.0002
    p_cnot_ZX: float = 0.0002
    p_cnot_ZY: float = 0.0002
    p_cnot_ZZ: float = 0.0002

    enable_parallel: bool = False
    num_workers: int = 4
    parallel_device_ids: Optional[list[int]] = None
    sampling_chunk_size: int = 16384
    save_samples: bool = False

    def get_measurement_basis(self) -> str:
        """Map the user-facing basis selector to X, Z, or both."""
        basis = str(self.basis).upper()
        if basis in ("O1", "O2", "X"):
            return "X"
        if basis in ("O3", "O4", "Z"):
            return "Z"
        if basis in ("BOTH", "MIXED"):
            return "both"
        raise ValueError(f"basis must be O1-O4, X, Z, both, or mixed, got {self.basis}")

    def get_rotation(self) -> str:
        """Resolve the surface-code rotation."""
        mapping = {"O1": "XV", "O2": "XH", "O3": "ZV", "O4": "ZH"}
        basis = str(self.basis).upper()

        if self.rotation is not None:
            rotation = str(self.rotation).upper()
            if rotation not in ("XV", "XH", "ZV", "ZH"):
                raise ValueError("rotation must be one of: XV, XH, ZV, ZH")
            if basis in mapping and mapping[basis] != rotation:
                raise ValueError(
                    f"basis {self.basis} implies rotation {mapping[basis]}, got explicit rotation {rotation}"
                )
            return rotation

        if basis in mapping:
            return mapping[basis]
        if basis == "X":
            return "XV"
        if basis == "Z":
            return "ZH"
        if basis in ("BOTH", "MIXED"):
            raise ValueError("rotation must be set when basis is both/mixed")
        raise ValueError(f"basis must be O1-O4, X, Z, both, or mixed, got {self.basis}")

    def get_25p_noise_params(self) -> Dict[str, float]:
        """Return the full 25-parameter noise model."""
        return {
            "p_prep_X": self.p_prep_X,
            "p_prep_Z": self.p_prep_Z,
            "p_meas_X": self.p_meas_X,
            "p_meas_Z": self.p_meas_Z,
            "p_idle_cnot_X": self.p_idle_cnot_X,
            "p_idle_cnot_Y": self.p_idle_cnot_Y,
            "p_idle_cnot_Z": self.p_idle_cnot_Z,
            "p_idle_spam_X": self.p_idle_spam_X,
            "p_idle_spam_Y": self.p_idle_spam_Y,
            "p_idle_spam_Z": self.p_idle_spam_Z,
            "p_cnot_IX": self.p_cnot_IX,
            "p_cnot_IY": self.p_cnot_IY,
            "p_cnot_IZ": self.p_cnot_IZ,
            "p_cnot_XI": self.p_cnot_XI,
            "p_cnot_XX": self.p_cnot_XX,
            "p_cnot_XY": self.p_cnot_XY,
            "p_cnot_XZ": self.p_cnot_XZ,
            "p_cnot_YI": self.p_cnot_YI,
            "p_cnot_YX": self.p_cnot_YX,
            "p_cnot_YY": self.p_cnot_YY,
            "p_cnot_YZ": self.p_cnot_YZ,
            "p_cnot_ZI": self.p_cnot_ZI,
            "p_cnot_ZX": self.p_cnot_ZX,
            "p_cnot_ZY": self.p_cnot_ZY,
            "p_cnot_ZZ": self.p_cnot_ZZ,
        }
