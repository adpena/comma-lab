# SPDX-License-Identifier: MIT
"""Tests for ``tac.codec_pipeline_kl_latent.Op_KLLatent``.

Verifies the same contract as test_codec_pipeline_kl_pose but generalized
to arbitrary 2D latent matrices (not just (N, 6) poses). Emphasis on:
  - CodecOp Protocol satisfaction
  - encode/decode roundtrip with bounded reconstruction error
  - byte-deterministic encode (same input → identical bytes)
  - substrate-adaptive basis (different inputs → different blobs)
  - smooth low-rank latent compresses below raw int16 baseline
  - validate() catches schema violations
  - JSON-serializable op_state (CPL1 wire-format discipline)
  - PR101-shape (600, 28) empirical smoke
"""
from __future__ import annotations

import json

import pytest
import torch

from tac.codec_pipeline import CodecOp, CodecPipeline
from tac.codec_pipeline_kl_latent import (
    LATENT_KEY,
    Op_KLLatent,
    estimate_truncation_rms,
)


def _smooth_low_rank_latents(
    n_frames: int = 600, latent_dim: int = 28, rank: int = 4, seed: int = 0
) -> torch.Tensor:
    """Synthetic (N, D) latent matrix with effective rank ≪ D + small noise.

    The smooth low-rank case is the regime KL exploits: top-`rank` components
    capture ~100% of variance, so KL with k ≥ rank reconstructs near-perfectly.
    """
    g = torch.Generator().manual_seed(seed)
    # Random row coefficients (N x rank) and column basis (rank x D), unit norm
    row_coefs = torch.randn(n_frames, rank, generator=g)
    col_basis = torch.randn(rank, latent_dim, generator=g)
    col_basis = col_basis / col_basis.norm(dim=1, keepdim=True)
    latents = row_coefs @ col_basis  # (N, D), rank-`rank` exactly
    # Add small full-rank noise so SVD doesn't drop all but `rank` singular values
    latents += 0.001 * torch.randn(n_frames, latent_dim, generator=g)
    return latents


def _random_full_rank_latents(
    n_frames: int = 600, latent_dim: int = 28, seed: int = 1
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_frames, latent_dim, generator=g) * 0.3


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def test_op_satisfies_codec_op_protocol() -> None:
    op = Op_KLLatent()
    assert isinstance(op, CodecOp)
    assert op.name == "kl_latent"


def test_op_default_n_components_is_eight() -> None:
    op = Op_KLLatent()
    assert op.n_components == 8


# ---------------------------------------------------------------------------
# Encode / decode roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_low_rank_latents_within_quant_tolerance() -> None:
    """Smooth low-rank latents through k=4 KL basis must reconstruct
    within int16-quantization + tiny-noise tolerance."""
    latents = _smooth_low_rank_latents(rank=4)
    op = Op_KLLatent(n_components=4, brotli_quality=1)
    sd = {LATENT_KEY: latents}
    result = op.encode(sd, context={})
    decoded = op.decode(result.blob, op_state=result.op_state, context={})
    recon = decoded[LATENT_KEY]
    assert recon.shape == latents.shape
    # Truncation residual ~0.001 (the added noise), int16 quant ~max_abs/32767.
    # Generous 1e-2 absolute tolerance covers both for synthetic ranges.
    max_err = (recon.float() - latents.float()).abs().max().item()
    assert max_err < 1e-2, f"max abs error {max_err} exceeded tolerance"


def test_roundtrip_full_rank_keeps_all_components() -> None:
    """When rank == D, k=D keeps everything; only int16 quantization
    contributes to error."""
    latents = _random_full_rank_latents(latent_dim=28)
    op = Op_KLLatent(n_components=28, brotli_quality=1)
    sd = {LATENT_KEY: latents}
    result = op.encode(sd, context={})
    decoded = op.decode(result.blob, op_state=result.op_state, context={})
    recon = decoded[LATENT_KEY]
    max_abs = latents.abs().max().item()
    grid_bound = 2.0 * max_abs / 32767.0 * 28
    max_err = (recon.float() - latents.float()).abs().max().item()
    assert max_err < grid_bound, (
        f"full-rank error {max_err} > expected quant bound {grid_bound}"
    )


def test_truncation_diagnostic_high_variance_for_low_rank() -> None:
    """Low-rank synthetic: top-4 captures > 99% of variance."""
    latents = _smooth_low_rank_latents(rank=4)
    diag = estimate_truncation_rms(latents, n_components=4)
    assert diag["cumulative_variance_ratio"] > 0.99, diag


def test_truncation_diagnostic_balanced_for_random() -> None:
    """Random iid: top-4 captures ~4/D of variance (uniform spectrum)."""
    latents = _random_full_rank_latents(latent_dim=28)
    diag = estimate_truncation_rms(latents, n_components=4)
    # Top-4 of 28 dims ~14% of variance, but with 600 rows the empirical
    # spectrum is biased toward the top. Allow a wide window.
    assert 0.05 < diag["cumulative_variance_ratio"] < 0.45


# ---------------------------------------------------------------------------
# Byte determinism + substrate-adaptive basis
# ---------------------------------------------------------------------------


def test_encode_byte_deterministic() -> None:
    latents = _smooth_low_rank_latents()
    op = Op_KLLatent(n_components=4, brotli_quality=11)
    sd = {LATENT_KEY: latents}
    blob_a = op.encode(sd, context={}).blob
    blob_b = op.encode(sd, context={}).blob
    assert blob_a == blob_b


def test_substrate_adaptive_different_latents_different_basis() -> None:
    op = Op_KLLatent(n_components=4, brotli_quality=1)
    a = _smooth_low_rank_latents(seed=0, rank=4)
    b = _random_full_rank_latents(seed=1)
    res_a = op.encode({LATENT_KEY: a}, context={})
    res_b = op.encode({LATENT_KEY: b}, context={})
    assert res_a.blob != res_b.blob


# ---------------------------------------------------------------------------
# Compression empirical
# ---------------------------------------------------------------------------


def test_low_rank_latents_compress_below_raw_int16() -> None:
    """KL with k=2 on a rank-2 600×28 matrix must beat raw 600×28×2 = 33,600 B."""
    latents = _smooth_low_rank_latents(rank=2)
    raw_int16_bytes = latents.size(0) * latents.size(1) * 2  # 33,600
    op = Op_KLLatent(n_components=2, brotli_quality=11)
    result = op.encode({LATENT_KEY: latents}, context={})
    assert result.bytes_out < raw_int16_bytes, (
        f"k=2 KL ({result.bytes_out} B) did not beat raw int16 ({raw_int16_bytes} B)"
    )


# ---------------------------------------------------------------------------
# CodecPipeline integration
# ---------------------------------------------------------------------------


def test_op_in_codec_pipeline() -> None:
    latents = _smooth_low_rank_latents(rank=4)
    op = Op_KLLatent(n_components=4, brotli_quality=1)
    pipeline = CodecPipeline([op])
    blob, manifest = pipeline.encode({LATENT_KEY: latents})
    assert blob[:4] in (b"CPL1", b"CPL2")  # CPL2 is canonical default 2026-05-08
    decoded, replayed = pipeline.decode(blob)
    assert replayed == ["kl_latent"]
    assert LATENT_KEY in decoded
    assert decoded[LATENT_KEY].shape == latents.shape


def test_op_state_is_json_serializable() -> None:
    latents = _smooth_low_rank_latents(rank=4)
    op = Op_KLLatent(n_components=4, brotli_quality=1)
    result = op.encode({LATENT_KEY: latents}, context={})
    encoded = json.dumps(result.op_state)
    decoded = json.loads(encoded)
    assert decoded["n_components"] == 4
    assert decoded["latent_dim"] == 28


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def test_validate_rejects_missing_latent_key() -> None:
    op = Op_KLLatent()
    rep = op.validate({"not_latents": torch.zeros(100, 28)}, context={})
    assert not rep.passed
    assert any("missing required" in f for f in rep.findings)


def test_validate_rejects_non_2d_latents() -> None:
    op = Op_KLLatent()
    rep = op.validate({LATENT_KEY: torch.zeros(100)}, context={})
    assert not rep.passed
    assert any("must be 2D" in f for f in rep.findings)


def test_validate_rejects_zero_n_components() -> None:
    op = Op_KLLatent(n_components=0)
    rep = op.validate({LATENT_KEY: torch.zeros(100, 28)}, context={})
    assert not rep.passed
    assert any("must be >= 1" in f for f in rep.findings)


# ---------------------------------------------------------------------------
# Encode error handling
# ---------------------------------------------------------------------------


def test_encode_raises_on_missing_latent_key() -> None:
    op = Op_KLLatent()
    with pytest.raises(ValueError, match="missing required key"):
        op.encode({"other": torch.zeros(100, 28)}, context={})


def test_encode_raises_on_non_tensor_latents() -> None:
    op = Op_KLLatent()
    with pytest.raises(TypeError, match=r"must be torch\.Tensor"):
        op.encode({LATENT_KEY: [[0.0] * 28] * 100}, context={})  # type: ignore[dict-item]


def test_decode_raises_on_bad_magic() -> None:
    op = Op_KLLatent()
    with pytest.raises(ValueError, match="bad magic"):
        op.decode(b"NOPE" + b"\x00" * 100, op_state={}, context={})


# ---------------------------------------------------------------------------
# PR101-shape empirical smoke
# ---------------------------------------------------------------------------


def test_pr101_shape_smoke_k_sweep() -> None:
    """Empirical: on synthetic (600, 28) low-rank latents matching PR101's
    shape, sweep k = {2, 4, 8, 16, 28} and verify monotone bytes-out."""
    latents = _smooth_low_rank_latents(n_frames=600, latent_dim=28, rank=4)
    bytes_by_k = {}
    for k in (2, 4, 8, 16, 28):
        op = Op_KLLatent(n_components=k, brotli_quality=11)
        result = op.encode({LATENT_KEY: latents}, context={})
        bytes_by_k[k] = result.bytes_out
    # Bytes should monotonically grow with k (more coefficients to store)
    sorted_ks = sorted(bytes_by_k.keys())
    for i in range(1, len(sorted_ks)):
        assert bytes_by_k[sorted_ks[i]] >= bytes_by_k[sorted_ks[i - 1]] - 50, (
            f"non-monotone at k={sorted_ks[i]}: "
            f"{bytes_by_k}"
        )
    # k=28 should be larger than k=2 by a meaningful margin
    assert bytes_by_k[28] > bytes_by_k[2] + 100
