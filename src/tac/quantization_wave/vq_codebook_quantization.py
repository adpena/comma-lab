# SPDX-License-Identifier: MIT
"""VQ-codebook quantization — van den Oord / VQ-VAE pattern.

Per the T4 SYMPOSIUM Priority 1 BOLT-ON #3 (van den Oord voice):
"discrete-token codebook latents are the canonical winning compression
primitive in modern neural compression... A Rule #6 BOLT-ON on A1 that
adds VQ-codebook encoding of A1's per-pair latent would be a stronger
version of the PR101 bolt-on."

This module ships VQ codebook quantization specifically targeted at
A1's 28-dimensional per-pair latent (the existing 15,387-byte
``latent_blob`` in ``submissions/a1/src/codec.py``).

The codebook trades:
    - Codebook overhead: ``K * D * fp16`` bytes (K codewords × D dims × 2 bytes)
    - Index overhead: ``N * log2(K) / 8`` bytes (N samples × log2(K) bits)

vs the existing PR101 LZMA-compressed temporal-delta encoding.

For K=256, D=28, N=600 pairs:
    codebook: 256 × 28 × 2 = 14,336 bytes
    indices:  600 × 8 / 8 = 600 bytes
    total: 14,936 bytes
    (vs PR101 ~15,387 bytes — neutral on raw size; advantage comes
     from BROTLI/LZMA-compressibility of repeating indices + lower
     quantization error for clustered latents)

For K=64 (4× smaller codebook):
    codebook: 64 × 28 × 2 = 3,584 bytes
    indices:  600 × 6 / 8 = 450 bytes
    total: 4,034 bytes (74% reduction vs PR101 15,387 bytes)

[prediction]
[verified-against:VQ-VAE paper (van den Oord 2017) + WaveNet
quantization + PR101 ``latent_blob`` layout]
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import torch


@dataclass(frozen=True)
class VQCodebookEncoded:
    """VQ-codebook encoded latents."""
    codebook: torch.Tensor  # (K, D) fp16
    indices: torch.Tensor  # (N,) uint8 or uint16 depending on K
    bits_per_index: int
    n_samples: int


def encode_vq_codebook_latent(
    latent: torch.Tensor,
    *,
    n_codebook_entries: int = 256,
    n_iter: int = 50,
    seed: int = 0,
) -> VQCodebookEncoded:
    """Encode a (N, D) latent matrix via VQ codebook quantization.

    Uses k-means++ initialization + Lloyd's iterations to fit a
    codebook of ``n_codebook_entries`` entries.

    Args:
        latent: (N, D) latent matrix (e.g. A1's (600, 28) per-pair latent)
        n_codebook_entries: K (codebook size; 64 / 128 / 256 typical)
        n_iter: Lloyd's iterations (50 is canonical for convergence)
        seed: RNG seed for deterministic k-means++ init

    [verified-against:VQ-VAE codebook fitting + k-means++ canonical
    (Arthur & Vassilvitskii 2007)]
    """
    if latent.ndim != 2:
        raise ValueError(
            f"encode_vq_codebook_latent requires 2D (N, D) latent; got shape {tuple(latent.shape)}"
        )
    x = latent.detach().contiguous().float()
    N, D = x.shape
    K = n_codebook_entries
    if K > N:
        raise ValueError(f"n_codebook_entries {K} > n_samples {N}")
    gen = torch.Generator().manual_seed(seed)
    # k-means++ init
    first_idx = torch.randint(0, N, (1,), generator=gen).item()
    centroids = [x[first_idx]]
    for _ in range(K - 1):
        # Compute min distance to existing centroids
        cmat = torch.stack(centroids)  # (k_so_far, D)
        dists = torch.cdist(x, cmat).min(dim=1).values
        probs = dists / dists.sum().clamp(min=1e-10)
        # Sample new centroid weighted by squared distance (k-means++)
        cdf = (probs**2).cumsum(0)
        cdf = cdf / cdf[-1].clamp(min=1e-10)
        r = torch.rand((), generator=gen).item()
        new_idx = int((cdf >= r).nonzero()[0].item())
        centroids.append(x[new_idx])
    centroids = torch.stack(centroids)  # (K, D)
    # Lloyd's iterations
    for _ in range(n_iter):
        dists = torch.cdist(x, centroids)  # (N, K)
        assignments = dists.argmin(dim=1)
        new_centroids = torch.zeros_like(centroids)
        for k in range(K):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = x[mask].mean(dim=0)
            else:
                # Re-seed empty cluster with random sample
                new_centroids[k] = x[torch.randint(0, N, (1,), generator=gen).item()]
        if torch.allclose(new_centroids, centroids, rtol=1e-5):
            break
        centroids = new_centroids
    # Final assignments
    dists = torch.cdist(x, centroids)
    indices = dists.argmin(dim=1)
    bits_per_index = max(1, math.ceil(math.log2(K)))
    if bits_per_index <= 8:
        indices_dtype = torch.uint8
    elif bits_per_index <= 16:
        indices_dtype = torch.int16
    else:
        indices_dtype = torch.int32
    return VQCodebookEncoded(
        codebook=centroids.to(torch.float16).cpu(),
        indices=indices.to(indices_dtype).cpu(),
        bits_per_index=bits_per_index,
        n_samples=N,
    )


def decode_vq_codebook_latent(encoded: VQCodebookEncoded) -> torch.Tensor:
    """Inverse of :func:`encode_vq_codebook_latent`."""
    codebook = encoded.codebook.float()
    indices = encoded.indices.long()
    return codebook[indices]


class VQCodebookQuantizer:
    """Wrapper that fits + encodes + estimates archive bytes for the
    canonical T4 SYMPOSIUM Priority 1 BOLT-ON #3 (van den Oord)."""

    def __init__(self, *, n_codebook_entries: int = 256, n_iter: int = 50, seed: int = 0):
        self.n_codebook_entries = n_codebook_entries
        self.n_iter = n_iter
        self.seed = seed
        self._encoded: VQCodebookEncoded | None = None

    def fit_and_encode(self, latent: torch.Tensor) -> VQCodebookEncoded:
        self._encoded = encode_vq_codebook_latent(
            latent,
            n_codebook_entries=self.n_codebook_entries,
            n_iter=self.n_iter,
            seed=self.seed,
        )
        return self._encoded

    def estimate_archive_bytes(self, *, brotli_compress_indices: bool = True) -> int:
        """Return archive byte estimate for the encoded codebook + indices.

        With ``brotli_compress_indices=True``, the indices are passed
        through Brotli (the canonical PR101 final-step compression).
        """
        if self._encoded is None:
            raise RuntimeError("must call fit_and_encode first")
        # Codebook: K * D * 2 bytes (fp16)
        codebook_bytes = self._encoded.codebook.numpy().tobytes()
        indices_bytes = self._encoded.indices.numpy().tobytes()
        if brotli_compress_indices:
            try:
                import brotli
                indices_bytes = brotli.compress(indices_bytes, quality=11)
            except Exception:
                pass
        return len(codebook_bytes) + len(indices_bytes)
