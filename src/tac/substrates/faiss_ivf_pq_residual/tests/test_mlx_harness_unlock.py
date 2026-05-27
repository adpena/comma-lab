# SPDX-License-Identifier: MIT
"""MLX-SCORE-AWARE-HARNESS-WAVE unlock tests for faiss_ivf_pq_residual.

The HONEST case: faiss IVF-PQ's training is K-MEANS CODEBOOK FITTING, NOT
gradient descent — so it does NOT route through the gradient-reachable MLX
score-aware harness (PQ codebook assignment is argmin, non-differentiable).
These tests verify the codebook-fitting pipeline (the substrate's actual
training algorithm) that unblocked ``_full_main``: train_pq_codebook (k-means)
-> encode -> reconstruct round-trip. numpy-only; runs everywhere.
"""
from __future__ import annotations

import numpy as np


def _tiny_cfg():
    from tac.substrates.faiss_ivf_pq_residual import FaissIVFPQResidualConfig

    return FaissIVFPQResidualConfig(
        m_sub_quantizers=2,
        ksub_codebook_size=8,
        tile_h=192,
        tile_w=256,
        num_pairs=4,
    )


def test_codebook_fitting_reduces_reconstruction_error() -> None:
    """K-means codebook fit must reconstruct residual tiles better than random."""
    from tac.substrates.faiss_ivf_pq_residual.numpy_reference import (
        encode_per_pair_residual,
        pq_reconstruct_tile_vectors,
        train_pq_codebook,
    )

    cfg = _tiny_cfg()
    rng = np.random.default_rng(0)
    # Structured residual tiles (clustered so k-means has signal to fit).
    centers = rng.standard_normal((cfg.ksub_codebook_size, cfg.tile_dim)).astype(
        np.float32
    )
    assign = rng.integers(0, cfg.ksub_codebook_size, size=64)
    tiles = centers[assign] + 0.01 * rng.standard_normal(
        (64, cfg.tile_dim)
    ).astype(np.float32)

    codebook = train_pq_codebook(
        tiles,
        m_sub_quantizers=cfg.m_sub_quantizers,
        ksub_codebook_size=cfg.ksub_codebook_size,
        num_kmeans_iters=10,
        seed=0,
    )
    assert codebook.shape == (cfg.m_sub_quantizers, cfg.ksub_codebook_size, cfg.sub_dim)

    indices = encode_per_pair_residual(tiles, codebook)
    recon = pq_reconstruct_tile_vectors(codebook, indices.astype(np.int64))
    fitted_mse = float(np.mean((recon - tiles) ** 2))

    # A random codebook reconstructs the same tiles much worse.
    rand_codebook = rng.standard_normal(codebook.shape).astype(np.float32)
    rand_indices = encode_per_pair_residual(tiles, rand_codebook)
    rand_recon = pq_reconstruct_tile_vectors(rand_codebook, rand_indices.astype(np.int64))
    rand_mse = float(np.mean((rand_recon - tiles) ** 2))

    assert fitted_mse < rand_mse


def test_codebook_fit_is_deterministic_given_seed() -> None:
    from tac.substrates.faiss_ivf_pq_residual.numpy_reference import (
        train_pq_codebook,
    )

    cfg = _tiny_cfg()
    rng = np.random.default_rng(1)
    tiles = rng.standard_normal((40, cfg.tile_dim)).astype(np.float32)
    cb_a = train_pq_codebook(
        tiles,
        m_sub_quantizers=cfg.m_sub_quantizers,
        ksub_codebook_size=cfg.ksub_codebook_size,
        num_kmeans_iters=5,
        seed=7,
    )
    cb_b = train_pq_codebook(
        tiles,
        m_sub_quantizers=cfg.m_sub_quantizers,
        ksub_codebook_size=cfg.ksub_codebook_size,
        num_kmeans_iters=5,
        seed=7,
    )
    assert np.array_equal(cb_a, cb_b)


def test_archive_roundtrip_from_fitted_codebook() -> None:
    from tac.substrates.faiss_ivf_pq_residual import (
        build_archive_bytes,
        parse_archive,
    )
    from tac.substrates.faiss_ivf_pq_residual.numpy_reference import (
        encode_per_pair_residual,
        train_pq_codebook,
    )

    cfg = _tiny_cfg()
    rng = np.random.default_rng(2)
    tiles = rng.standard_normal((cfg.tiles_per_pair, cfg.tile_dim)).astype(np.float32)
    codebook = train_pq_codebook(
        tiles,
        m_sub_quantizers=cfg.m_sub_quantizers,
        ksub_codebook_size=cfg.ksub_codebook_size,
        num_kmeans_iters=3,
        seed=0,
    )
    codewords = np.stack(
        [encode_per_pair_residual(tiles, codebook) for _ in range(cfg.num_pairs)],
        axis=0,
    )
    data = build_archive_bytes(
        codebook, codewords, tile_h=cfg.tile_h, tile_w=cfg.tile_w
    )
    arch = parse_archive(data)
    assert np.array_equal(arch.codebook, codebook)
    assert np.array_equal(arch.per_pair_codewords, codewords)
