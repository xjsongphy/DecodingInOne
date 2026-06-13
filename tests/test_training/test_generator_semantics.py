import torch
import numpy as np
from pathlib import Path
import tempfile

from decoding_in_one.sampling.dem import measure_from_stacked_frames, timelike_syndromes
from decoding_in_one.sampling.generator import CircuitDataGenerator
from decoding_in_one.codes.surface_code.homological_equivalence_torch import (
    apply_weight1_timelike_homological_equivalence_torch,
)


def test_build_batch_from_frames_matches_manual_he_pipeline() -> None:
    distance = 3
    n_rounds = 3
    basis = "X"
    nq = 2 * distance * distance - 1
    num_detectors = n_rounds * nq
    num_meas = nq - distance * distance

    H = torch.zeros((2 * num_detectors, 1), dtype=torch.uint8)
    p = torch.zeros((1,), dtype=torch.float32)
    A = torch.zeros((n_rounds * num_meas, 2 * num_detectors), dtype=torch.uint8)

    gen = CircuitDataGenerator(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        code_rotation="XV",
        H=H,
        p=p,
        A=A,
        device=torch.device("cpu"),
    )

    rng = torch.Generator().manual_seed(123)
    frames_xz = torch.randint(
        0,
        2,
        (4, 2 * num_detectors),
        generator=rng,
        dtype=torch.uint8,
    )

    built_x, built_y, _ = gen._build_batch_from_frames(frames_xz, batch_size=4, verbose=False)

    D = frames_xz.shape[1] // 2
    idx_data = (
        torch.arange(gen.n_rounds)[:, None] * gen.nq + gen.data_qubits[None, :]
    ).reshape(-1)
    x_cum = frames_xz[:, :D].index_select(1, idx_data).reshape(4, gen.n_rounds, -1)
    z_cum = frames_xz[:, D:].index_select(1, idx_data).reshape(4, gen.n_rounds, -1)
    meas_old = measure_from_stacked_frames(frames_xz, gen.meas_qubits, gen.meas_bases, nq=gen.nq)
    meas_new = timelike_syndromes(frames_xz, gen.A, meas_old)

    num_x = int(gen.xcheck_qubits.numel())
    s1s2x = meas_new[:, :, :num_x]
    s1s2z = meas_new[:, :, num_x:]
    mx = meas_old[:, :, :num_x]
    mz = meas_old[:, :, num_x:]
    mxp = torch.cat([torch.zeros_like(mx[:, :1, :]), mx], dim=1)
    mzp = torch.cat([torch.zeros_like(mz[:, :1, :]), mz], dim=1)
    trainX_x = mxp[:, :-1, :] ^ mxp[:, 1:, :]
    trainX_z = mzp[:, :-1, :] ^ mzp[:, 1:, :]

    z_diff, x_diff, s1s2x, s1s2z = apply_weight1_timelike_homological_equivalence_torch(
        z_cum,
        x_cum,
        s1s2x,
        s1s2z,
        gen.parity_Z,
        gen.parity_X,
        gen.distance,
        gen.num_he_cycles,
        gen.max_passes_w1,
        gen.basis,
        True,
        trainX_x=trainX_x,
        trainX_z=trainX_z,
        cache_Z_spacelike=gen.cache_Z_sp,
        cache_X_spacelike=gen.cache_X_sp,
        use_compile=gen.use_compile,
        compile_chunk_size=gen.compile_chunk_size,
        compute_dtype=gen.compute_dtype,
        use_weight2=gen.use_weight2,
        max_passes_w2=gen.max_passes_w2,
        cache_Z_w2=gen.cache_Z_w2,
        cache_X_w2=gen.cache_X_w2,
        use_coset_search=gen.use_coset_search,
        coset_max_generators=gen.coset_max_generators,
        use_dense_overlap=gen.use_dense_overlap,
        use_parallel_spacelike=gen.use_parallel_spacelike,
    )
    manual_meas_new = torch.cat([s1s2x, s1s2z], dim=2)
    manual_x, manual_y = gen._format_for_model(
        x_diff.to("cpu"),
        z_diff.to("cpu"),
        meas_old.to("cpu"),
        manual_meas_new.to("cpu"),
    )

    assert torch.equal(built_x, manual_x)
    assert torch.equal(built_y, manual_y)


def test_mixed_basis_generator_alternates_x_and_z_branches() -> None:
    distance = 3
    n_rounds = 3
    nq = 2 * distance * distance - 1
    num_detectors = n_rounds * nq
    num_meas = nq - distance * distance

    hx = np.zeros((num_detectors, 1), dtype=np.uint8)
    hz = np.zeros((num_detectors, 1), dtype=np.uint8)
    p = np.zeros((1,), dtype=np.float32)
    a = np.zeros((n_rounds * num_meas, 2 * num_detectors), dtype=np.uint8)

    with tempfile.TemporaryDirectory(prefix="dio_mixed_gen_") as tmpdir:
        root = Path(tmpdir)
        for basis in ("X", "Z"):
            prefix = root / f"surface_d{distance}_r{n_rounds}_{basis}_frame_predecoder"
            np.savez(prefix.with_suffix(".X.npz"), hx)
            np.savez(prefix.with_suffix(".Z.npz"), hz)
            np.savez(prefix.with_suffix(".p.npz"), p)
            np.savez(prefix.with_suffix(".A.npz"), a)

        gen = CircuitDataGenerator(
            distance=distance,
            n_rounds=n_rounds,
            basis="both",
            code_rotation="XV",
            precomputed_frames_dir=str(root),
            device=torch.device("cpu"),
        )

        batch_x = gen.generate_batch(batch_size=2, seed=11, keep_on_cpu=True, step=0)
        batch_z = gen.generate_batch(batch_size=2, seed=12, keep_on_cpu=True, step=1)

    assert batch_x["trainX"].shape == (2, 4, n_rounds, distance, distance)
    assert batch_z["trainX"].shape == (2, 4, n_rounds, distance, distance)
    assert not torch.equal(batch_x["trainX"], batch_z["trainX"])
