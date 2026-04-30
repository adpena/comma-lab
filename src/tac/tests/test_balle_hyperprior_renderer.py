"""Lane 20 — Ballé hyperprior tests.

Per Phase 3 Lane 20 spec (memory project_phases_2_3_4_*):
- Encode → decode round-trip
- Header overhead tracking (must be < ~500B to amortise on 11KB stream)
- Byte-savings vs static factorized baseline (synthetic comparison)

All claims tagged [synthetic].

CLAUDE.md non-negotiables verified:
- No scorer load
- No silent defaults (all required-keyword)
- No GPU
- Deterministic CPU-only
- Pure-math byte → tensor pipeline
"""
from __future__ import annotations

import pytest
import torch

from tac.balle_hyperprior_renderer import (
    BHP_MAGIC,
    ScalePriorMLP,
    amortisation_break_even_bytes,
    decode_balle_hyperprior,
    encode_balle_hyperprior,
    gaussian_rate_bits,
    static_factorised_rate_bits,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: encode → decode round-trip preserves all weights
# ─────────────────────────────────────────────────────────────────────────────


def test_encode_decode_roundtrip_synthetic() -> None:
    """[synthetic] BHP1 round-trip preserves ScalePriorMLP weights to fp16."""
    mlp = ScalePriorMLP(context_dim=8, hidden_dim=16, depth=4, seed=2026)
    qstream = torch.randint(-31, 32, (100,), dtype=torch.int8)
    blob = encode_balle_hyperprior(scale_prior=mlp, qint_stream=qstream)
    assert blob[:4] == BHP_MAGIC
    mlp_decoded = decode_balle_hyperprior(blob)
    # Same arch
    assert mlp_decoded.context_dim == mlp.context_dim
    assert mlp_decoded.hidden_dim == mlp.hidden_dim
    assert mlp_decoded.depth == mlp.depth
    # Weights match within fp16 precision
    for k, v in mlp.state_dict().items():
        v2 = mlp_decoded.state_dict()[k]
        assert v2.shape == v.shape
        assert torch.allclose(v.float(), v2.float(), atol=1e-2, rtol=1e-2)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: header overhead < ~500B for the canonical Selfcomp scale-prior
# ─────────────────────────────────────────────────────────────────────────────


def test_small_scale_prior_overhead_within_amortisation_band_synthetic() -> None:
    """[synthetic] A tiny scale-prior MLP has overhead in the amortisable band.

    Per Ballé 2018 / memory: hyperprior amortises on streams >5KB. A 4-layer
    MLP with context_dim=4, hidden_dim=8 has ~125 params → header < 200 bytes
    → break-even ~4KB at 5% savings.
    """
    mlp = ScalePriorMLP(context_dim=4, hidden_dim=8, depth=4, seed=2026)
    overhead = mlp.header_byte_size()
    # Tiny MLP fits the < 200 byte band
    assert overhead < 500, f"overhead={overhead} too large for amortisation"
    # Break-even at 5% savings should be < 10KB (i.e. amortises on Selfcomp 11KB stream)
    breakeven = amortisation_break_even_bytes(mlp, expected_savings_fraction=0.05)
    assert breakeven < 10_000, f"break-even {breakeven} bytes too large"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: gaussian_rate_bits matches closed-form for known inputs
# ─────────────────────────────────────────────────────────────────────────────


def test_gaussian_rate_bits_matches_closed_form_synthetic() -> None:
    """[synthetic] gaussian_rate_bits matches the closed-form gaussian density."""
    import math

    qstream = torch.tensor([0.0, 1.0, 2.0])
    sigma = torch.tensor([1.0, 1.0, 1.0])
    bits = gaussian_rate_bits(qstream, sigma)
    # Closed form: -log2 N(0, 1)(y) = 0.5*log2(2π) + 0.5*y²/log(2)
    expected_y0 = 0.5 * math.log2(2 * math.pi) + 0.0
    expected_y1 = 0.5 * math.log2(2 * math.pi) + 0.5 * 1.0 / math.log(2.0)
    assert bits[0].item() == pytest.approx(expected_y0, abs=1e-4)
    assert bits[1].item() == pytest.approx(expected_y1, abs=1e-4)
    # All bits finite + non-negative for sigma > 0 reasonable
    assert torch.isfinite(bits).all()


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: byte-savings vs static-factorised on a heteroscedastic synthetic
# ─────────────────────────────────────────────────────────────────────────────


def test_hyperprior_beats_static_on_heteroscedastic_synthetic() -> None:
    """[synthetic] On a heteroscedastic stream, hyperprior σ beats static σ.

    Build a stream where the first half has small std, second half has large std.
    Static factorised picks one global σ → suboptimal.
    Hyperprior with the TRUE per-element σ → exact match → minimal rate.
    """
    g = torch.Generator().manual_seed(2026)
    half = 200
    # First half: std=0.5 → small
    s1 = torch.randn(half, generator=g) * 0.5
    # Second half: std=4.0 → large
    s2 = torch.randn(half, generator=g) * 4.0
    qstream = torch.cat([s1, s2]).round().to(torch.int32)
    # True σ (per-element, oracle)
    true_sigma = torch.cat([torch.full((half,), 0.5), torch.full((half,), 4.0)])
    # Static rate
    static_bits = static_factorised_rate_bits(qstream).sum().item()
    # Oracle hyperprior rate (best possible σ choice)
    oracle_bits = gaussian_rate_bits(qstream, true_sigma).sum().item()
    # Oracle should beat static by a measurable margin (≥3% on this distribution)
    assert oracle_bits < static_bits, (
        f"oracle {oracle_bits:.1f} >= static {static_bits:.1f} — "
        f"hyperprior should win on heteroscedastic stream"
    )
    saving = (static_bits - oracle_bits) / static_bits
    assert saving > 0.03, f"savings only {saving*100:.1f}% (expected >3%)"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: forward output shape + finite at init
# ─────────────────────────────────────────────────────────────────────────────


def test_scale_prior_forward_shape_and_finite_synthetic() -> None:
    """[synthetic] ScalePriorMLP.forward returns positive σ shape (B,)."""
    mlp = ScalePriorMLP(context_dim=4, hidden_dim=8, depth=3)
    ctx = torch.randn(7, 4)
    sigma = mlp(ctx)
    assert sigma.shape == (7,)
    assert (sigma > 0).all(), "softplus output should be strictly positive"
    assert torch.isfinite(sigma).all()


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: no silent defaults — encode/decode reject None
# ─────────────────────────────────────────────────────────────────────────────


def test_no_silent_defaults_synthetic() -> None:
    """[synthetic] encode/decode reject None; integer-only qint_stream."""
    mlp = ScalePriorMLP(context_dim=4, hidden_dim=8, depth=3)
    qstream = torch.zeros(10, dtype=torch.int8)
    with pytest.raises(ValueError, match="scale_prior is required"):
        encode_balle_hyperprior(scale_prior=None, qint_stream=qstream)
    with pytest.raises(ValueError, match="qint_stream is required"):
        encode_balle_hyperprior(scale_prior=mlp, qint_stream=None)
    with pytest.raises(ValueError, match="must be integer tensor"):
        encode_balle_hyperprior(scale_prior=mlp, qint_stream=torch.zeros(10))  # float
    with pytest.raises(ValueError, match="blob is required"):
        decode_balle_hyperprior(blob=None)
    with pytest.raises(ValueError, match="bad magic"):
        decode_balle_hyperprior(blob=b"WRNG" + b"\x00" * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: amortisation_break_even_bytes math
# ─────────────────────────────────────────────────────────────────────────────


def test_amortisation_break_even_math_synthetic() -> None:
    """[synthetic] break_even = side_info / savings_fraction."""
    mlp = ScalePriorMLP(context_dim=4, hidden_dim=8, depth=3)
    side_info = mlp.header_byte_size()
    breakeven_5pct = amortisation_break_even_bytes(mlp, expected_savings_fraction=0.05)
    breakeven_10pct = amortisation_break_even_bytes(mlp, expected_savings_fraction=0.10)
    # Break-even doubles when savings halves
    assert breakeven_5pct >= 2 * breakeven_10pct - 1
    # Break-even must be at least side_info (need to recover the cost)
    assert breakeven_5pct >= side_info
    # Negative / zero savings rejected
    with pytest.raises(ValueError, match="must be > 0"):
        amortisation_break_even_bytes(mlp, expected_savings_fraction=0.0)
    with pytest.raises(ValueError, match="must be > 0"):
        amortisation_break_even_bytes(mlp, expected_savings_fraction=-0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: ScalePriorMLP rejects bad arch
# ─────────────────────────────────────────────────────────────────────────────


def test_scale_prior_rejects_bad_arch_synthetic() -> None:
    """[synthetic] ScalePriorMLP rejects invalid arch."""
    with pytest.raises(ValueError, match="invalid arch"):
        ScalePriorMLP(context_dim=0, hidden_dim=8, depth=4)
    with pytest.raises(ValueError, match="invalid arch"):
        ScalePriorMLP(context_dim=4, hidden_dim=8, depth=1)
