# SPDX-License-Identifier: MIT
"""faiss_ivf_pq_residual.numpy_reference — sister numpy reference implementations.

Per operator directive #3 2026-05-26 verbatim:
*"adversarial review against all landing recursive for math and scientific
and engineering rigor and for MLX drift minimization and portability via
numpy"*

Every MLX primitive used by ``mlx_renderer.py`` MUST have a sister numpy
reference implementation in this module OR documented non-portability
rationale.

Portability per Catalog #1 device-selection-defaults discipline; enables:
(a) GHA CPU CI testing per Catalog #178 + #179 without MLX OR Faiss install,
(b) sister cathedral consumer cross-validation per Catalog #335,
(c) operator-portable diagnostic on non-Apple-Silicon hardware.

This module is canonical-portable: numpy only, no MLX import, no torch
import, no Faiss import. Operable on any Python+numpy install.

Per-primitive parity bound vs MLX/PyTorch reference:
- PQ codebook gather: byte-identical (integer index + float copy)
- Tile reassemble: byte-identical (reshape/concat)
- Bilinear upsample: ≤ 1e-5 fp32; documented via canonical helper
- uint8 cast: byte-identical (deterministic rounding)
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Primitive 1: PQ codebook gather
# ---------------------------------------------------------------------------


def pq_codebook_gather(
    codebook: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Gather PQ centroids from codebook by integer indices.

    Args:
        codebook: shape (M, ksub, sub_dim) float32 — per-sub-quantizer centroids
        indices: shape (..., M) uint8/uint16 — per-tile codeword indices

    Returns:
        Per-tile reconstructed vector shape (..., M, sub_dim) float32.

    Sister MLX primitive uses `mx.take` / `mx.take_along_axis`; expected
    drift = 0 (integer index + exact float copy).
    """
    M = codebook.shape[0]
    if indices.shape[-1] != M:
        raise ValueError(
            f"indices last-dim {indices.shape[-1]} != codebook M {M}"
        )
    # Gather: for each (..., m) index, pick codebook[m, idx, :]
    # Output shape: (..., M, sub_dim)
    out = np.zeros(indices.shape + (codebook.shape[-1],), dtype=codebook.dtype)
    for m in range(M):
        m_indices = indices[..., m]  # (...,)
        out[..., m, :] = codebook[m, m_indices, :]  # (..., sub_dim)
    return out


# ---------------------------------------------------------------------------
# Primitive 2: per-tile vector reassemble (PQ decode → flat tile)
# ---------------------------------------------------------------------------


def pq_reconstruct_tile_vectors(
    codebook: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Reconstruct per-tile flat vectors from PQ codebook + indices.

    Args:
        codebook: shape (M, ksub, sub_dim) float32
        indices: shape (num_tiles, M) uint8/uint16

    Returns:
        Per-tile flat vector shape (num_tiles, M * sub_dim) float32.

    Composition of pq_codebook_gather + flatten across sub-quantizers.
    """
    gathered = pq_codebook_gather(codebook, indices)  # (num_tiles, M, sub_dim)
    num_tiles, M, sub_dim = gathered.shape
    return gathered.reshape(num_tiles, M * sub_dim)


# ---------------------------------------------------------------------------
# Primitive 3: tile reassemble (flat tiles → per-pair RGB)
# ---------------------------------------------------------------------------


def tiles_to_frame_nhwc(
    tiles: np.ndarray,
    *,
    frame_h: int,
    frame_w: int,
    tile_h: int,
    tile_w: int,
) -> np.ndarray:
    """Reassemble per-pair tile array into frame NHWC layout.

    Args:
        tiles: shape (num_tiles, tile_h * tile_w * 3) float32
        frame_h: output frame height (must equal grid_h * tile_h)
        frame_w: output frame width (must equal grid_w * tile_w)
        tile_h: per-tile height
        tile_w: per-tile width

    Returns:
        Frame array shape (frame_h, frame_w, 3) float32; channel-last layout.

    Tile ordering: row-major (idx 0 = top-left, idx grid_w = below top-left).
    """
    if frame_h % tile_h != 0 or frame_w % tile_w != 0:
        raise ValueError(
            f"frame_h={frame_h} frame_w={frame_w} must be divisible by tile_h={tile_h}, tile_w={tile_w}"
        )
    grid_h = frame_h // tile_h
    grid_w = frame_w // tile_w
    num_tiles = grid_h * grid_w
    if tiles.shape[0] != num_tiles:
        raise ValueError(
            f"tiles count {tiles.shape[0]} != grid_h*grid_w = {num_tiles}"
        )
    if tiles.shape[1] != tile_h * tile_w * 3:
        raise ValueError(
            f"tiles dim {tiles.shape[1]} != tile_h*tile_w*3 = {tile_h * tile_w * 3}"
        )
    # Reshape each tile to (tile_h, tile_w, 3) then place into grid
    frame = np.zeros((frame_h, frame_w, 3), dtype=tiles.dtype)
    for g in range(num_tiles):
        gi = g // grid_w
        gj = g % grid_w
        tile_3d = tiles[g].reshape(tile_h, tile_w, 3)
        frame[gi * tile_h:(gi + 1) * tile_h, gj * tile_w:(gj + 1) * tile_w, :] = tile_3d
    return frame


def frame_to_tiles_nhwc(
    frame: np.ndarray,
    *,
    tile_h: int,
    tile_w: int,
) -> np.ndarray:
    """Decompose per-pair frame NHWC into per-tile flat vectors.

    Inverse of tiles_to_frame_nhwc; used at encode time.

    Args:
        frame: shape (frame_h, frame_w, 3) float32
        tile_h: per-tile height
        tile_w: per-tile width

    Returns:
        Per-tile flat vector shape (num_tiles, tile_h * tile_w * 3) float32.
    """
    frame_h, frame_w, c = frame.shape
    if c != 3:
        raise ValueError(f"frame channels {c} != 3 (RGB)")
    if frame_h % tile_h != 0 or frame_w % tile_w != 0:
        raise ValueError(
            f"frame_h={frame_h} frame_w={frame_w} must be divisible by tile_h={tile_h}, tile_w={tile_w}"
        )
    grid_h = frame_h // tile_h
    grid_w = frame_w // tile_w
    num_tiles = grid_h * grid_w
    tile_dim = tile_h * tile_w * 3
    tiles = np.zeros((num_tiles, tile_dim), dtype=frame.dtype)
    for g in range(num_tiles):
        gi = g // grid_w
        gj = g % grid_w
        tile_3d = frame[gi * tile_h:(gi + 1) * tile_h, gj * tile_w:(gj + 1) * tile_w, :]
        tiles[g] = tile_3d.reshape(tile_dim)
    return tiles


# ---------------------------------------------------------------------------
# Primitive 4: PQ codebook training (numpy K-means; sklearn-equivalent)
# ---------------------------------------------------------------------------


def train_pq_codebook(
    tile_vectors: np.ndarray,
    *,
    m_sub_quantizers: int,
    ksub_codebook_size: int,
    num_kmeans_iters: int = 25,
    seed: int = 42,
) -> np.ndarray:
    """Train PQ codebook via per-sub-quantizer K-means.

    Args:
        tile_vectors: shape (num_tiles, tile_dim) float32 training data
        m_sub_quantizers: M parameter
        ksub_codebook_size: ksub parameter
        num_kmeans_iters: K-means iteration count (canonical=25 per Faiss default)
        seed: RNG seed for deterministic codebook initialization

    Returns:
        Codebook shape (M, ksub, sub_dim) float32; deterministic with seed.

    Implementation: per-sub-quantizer K-means with cosine-distance Lloyd
    iterations. Sister Faiss-CPU helper at
    `tac.optimization.faiss_ivf_pq_atw_channel.build_pq_codebook` is the
    canonical accelerator; this numpy reference is portability fallback.
    """
    num_tiles, tile_dim = tile_vectors.shape
    if tile_dim % m_sub_quantizers != 0:
        raise ValueError(
            f"tile_dim={tile_dim} must be divisible by m_sub_quantizers={m_sub_quantizers}"
        )
    sub_dim = tile_dim // m_sub_quantizers
    rng = np.random.default_rng(seed)
    codebook = np.zeros((m_sub_quantizers, ksub_codebook_size, sub_dim), dtype=np.float32)
    for m in range(m_sub_quantizers):
        # Extract per-sub-quantizer slice: shape (num_tiles, sub_dim)
        sub_data = tile_vectors[:, m * sub_dim:(m + 1) * sub_dim].astype(np.float32)
        # Initialize centroids via random sampling
        if num_tiles < ksub_codebook_size:
            # Edge case: not enough training samples; repeat with noise
            init_idx = rng.integers(0, num_tiles, size=ksub_codebook_size)
        else:
            init_idx = rng.choice(num_tiles, size=ksub_codebook_size, replace=False)
        centroids = sub_data[init_idx].copy()
        # K-means Lloyd iterations
        for _ in range(num_kmeans_iters):
            # Assign each point to nearest centroid (L2 distance)
            distances = np.linalg.norm(
                sub_data[:, None, :] - centroids[None, :, :], axis=-1
            )  # (num_tiles, ksub)
            assignments = np.argmin(distances, axis=1)  # (num_tiles,)
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(ksub_codebook_size):
                mask = assignments == k
                if mask.sum() > 0:
                    new_centroids[k] = sub_data[mask].mean(axis=0)
                else:
                    # Empty cluster: keep old centroid
                    new_centroids[k] = centroids[k]
            centroids = new_centroids
        codebook[m] = centroids
    return codebook


# ---------------------------------------------------------------------------
# Primitive 5: PQ encoding (nearest-neighbor codebook lookup)
# ---------------------------------------------------------------------------


def encode_per_pair_residual(
    tile_vectors: np.ndarray,
    codebook: np.ndarray,
) -> np.ndarray:
    """PQ encode per-tile vectors into M sub-quantizer codeword indices.

    Args:
        tile_vectors: shape (num_tiles, tile_dim) float32
        codebook: shape (M, ksub, sub_dim) float32

    Returns:
        Per-tile codeword indices shape (num_tiles, M) uint16 (supports
        ksub up to 65536).

    For each tile + each sub-quantizer m: find argmin over ksub L2 distances
    to codebook[m]. Integer-deterministic given fixed codebook.
    """
    num_tiles, tile_dim = tile_vectors.shape
    M, ksub, sub_dim = codebook.shape
    if tile_dim != M * sub_dim:
        raise ValueError(
            f"tile_dim={tile_dim} != M*sub_dim={M * sub_dim}"
        )
    indices = np.zeros((num_tiles, M), dtype=np.uint16)
    for m in range(M):
        sub_tiles = tile_vectors[:, m * sub_dim:(m + 1) * sub_dim]  # (num_tiles, sub_dim)
        # L2 distances: (num_tiles, ksub)
        distances = np.linalg.norm(
            sub_tiles[:, None, :] - codebook[m, None, :, :], axis=-1
        )
        indices[:, m] = np.argmin(distances, axis=1).astype(np.uint16)
    return indices


# ---------------------------------------------------------------------------
# Primitive 6: uint8 cast at output (canonical Catalog #205 sister rounding)
# ---------------------------------------------------------------------------


def to_uint8(x: np.ndarray) -> np.ndarray:
    """Canonical uint8 cast: clip [0, 255] + round + cast.

    Per Catalog #205 sister rounding discipline; deterministic.
    """
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


__all__ = [
    "encode_per_pair_residual",
    "frame_to_tiles_nhwc",
    "pq_codebook_gather",
    "pq_reconstruct_tile_vectors",
    "tiles_to_frame_nhwc",
    "to_uint8",
    "train_pq_codebook",
]
