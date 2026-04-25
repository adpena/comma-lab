"""Tests for FP4 quantization robustness fixes (R-FP4-fix 2026-04-25).

Trend report observation: float SegNet 0.54 → 0.28 over 15 epochs while the
FP4 path stayed at 93.44 from epoch 5. Diagnosis: the original FP4 codebook
+ max-based per-block scale collapses small-magnitude weights (rounds them
to zero), preventing the residual head from training through quantization.

Three fixes pinned by these tests:
  1. RESIDUAL_CODEBOOK has finer entries near zero (0.125 vs 0.5 first nonzero)
  2. robust_scale uses p99.5 quantile instead of max(|w|) — protects against
     outliers that push the small-magnitude tail past the rounding boundary.
  3. stochastic rounding gives unbiased dither at training time

Plus regression: train↔export round-trip MUST agree to within fp16 precision
(scales are stored as fp16 on disk by design).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from tac.fp4_quantize import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CODEBOOK,
    QATRendererFP4,
    RESIDUAL_CODEBOOK,
    _quantize_block,
    dequantize_fp4,
    fake_quant_fp4,
    quantize_fp4,
)


# ── Codebook properties ───────────────────────────────────────────────────


def test_residual_codebook_denser_near_zero() -> None:
    """RESIDUAL_CODEBOOK's smallest nonzero must be < DEFAULT's smallest nonzero."""
    assert RESIDUAL_CODEBOOK[1].item() < DEFAULT_CODEBOOK[1].item()
    # Both must end at the same max (6.0) so dynamic range matches
    assert DEFAULT_CODEBOOK[-1].item() == RESIDUAL_CODEBOOK[-1].item()
    # Both monotonically increasing (required for stochastic rounding bucketize)
    assert torch.all(DEFAULT_CODEBOOK[1:] > DEFAULT_CODEBOOK[:-1])
    assert torch.all(RESIDUAL_CODEBOOK[1:] > RESIDUAL_CODEBOOK[:-1])


def test_residual_codebook_preserves_smaller_magnitudes() -> None:
    """A weight that DEFAULT rounds to zero, RESIDUAL preserves nonzero.

    This is the entire point of the RESIDUAL codebook — small-magnitude
    weights (which DOMINATE residual / correction heads) survive quantization
    instead of being clipped to zero.
    """
    # Construct a block with [outlier=1.0, small=0.04 × 31].
    # max_mag = 1.0, scale = 1/6 = 0.1667, normalized small = 0.04/0.1667 = 0.24.
    # DEFAULT midpoints: midpoint(0, 0.5) = 0.25 → 0.24 < 0.25 → rounds to idx 0 (zero).
    # RESIDUAL midpoints: midpoint(0.125, 0.25) = 0.1875 → 0.24 > 0.1875 → rounds to idx 2 (0.25).
    block = torch.tensor([1.0] + [0.04] * 31, dtype=torch.float32)
    idx_default, _, _ = _quantize_block(block, DEFAULT_CODEBOOK)
    idx_residual, _, _ = _quantize_block(block, RESIDUAL_CODEBOOK)

    # DEFAULT rounds the small weights to zero (catastrophic for residual heads)
    assert idx_default[1].item() == 0, (
        f"DEFAULT_CODEBOOK should round 0.04*scale to zero (got {idx_default[1].item()})"
    )
    # RESIDUAL rounds them to a nonzero codebook entry — the whole point
    assert idx_residual[1].item() > 0, (
        f"RESIDUAL_CODEBOOK should preserve 0.04*scale as nonzero "
        f"(got idx={idx_residual[1].item()})"
    )


# ── Robust scale ──────────────────────────────────────────────────────────


def test_robust_scale_returns_smaller_value_when_outlier_present() -> None:
    """With outliers in a large block, p99.5 scale < max-based scale.

    For block_size=32 and a single outlier, the percentile interpolates and
    barely beats max. Robust scale's win shines on multi-outlier large blocks
    or when a learned-step optimizer drives the residual head's weights toward
    a small-magnitude regime — that regime is where DEFAULT bites hardest.
    """
    # 8x larger block to make percentile meaningfully ignore outliers
    block = torch.tensor([10.0, 8.0] + [0.05] * 254, dtype=torch.float32)
    block = block.reshape(1, -1)  # (1, 256) for vectorized path
    cb = DEFAULT_CODEBOOK
    from tac.fp4_quantize import _block_scales
    scale_max = _block_scales(block.abs(), cb[-1], robust=False)
    scale_p995 = _block_scales(block.abs(), cb[-1], robust=True)
    assert scale_p995.item() < scale_max.item(), (
        f"Robust scale {scale_p995.item():.4f} should be smaller than "
        f"max-based {scale_max.item():.4f} when outliers present"
    )


def test_robust_scale_handles_all_zero_block() -> None:
    """All-zero block (e.g. padding) must not crash or produce NaN."""
    block = torch.zeros(32, dtype=torch.float32)
    out = fake_quant_fp4(block, DEFAULT_CODEBOOK, 32, robust_scale=True)
    assert torch.all(torch.isfinite(out))
    assert torch.all(out == 0)


# ── Stochastic rounding ───────────────────────────────────────────────────


def test_stochastic_rounding_is_unbiased() -> None:
    """Mean of stochastic rounding over many samples ≈ exact value.

    Deterministic argmin biases hard for any value not exactly at a codebook
    midpoint. Stochastic rounds floor/ceil with prob ∝ fractional position,
    which is unbiased in expectation. Verifying expectation = input is the
    foundational property that lets gradients flow correctly.
    """
    block_size = 32
    # Block of 32 weights, all equal to 0.3, plus one outlier=6.0 to fix scale=1.0.
    block = torch.full((block_size,), 0.3, dtype=torch.float32)
    block[0] = 6.0  # sets scale = 6/6 = 1.0

    # Deterministic — should round 0.3 to 0.5 (closer than 0)
    out_det = fake_quant_fp4(block, DEFAULT_CODEBOOK, block_size, stochastic=False)
    assert out_det[1].item() == 0.5, f"deterministic should round 0.3 → 0.5 (got {out_det[1].item()})"

    # Stochastic average over 1000 samples
    samples = []
    for seed in range(1000):
        torch.manual_seed(seed)
        out = fake_quant_fp4(block, DEFAULT_CODEBOOK, block_size, stochastic=True)
        samples.append(out[1].item())
    mean_stoch = sum(samples) / len(samples)
    # Expected mean ≈ 0.3 (unbiased). Allow 5% slack for finite-sample noise.
    assert abs(mean_stoch - 0.3) < 0.05, (
        f"stochastic mean {mean_stoch:.4f} should ≈ 0.3 (input value)"
    )


def test_stochastic_disabled_at_eval() -> None:
    """FP4Parametrize with stochastic=True must be deterministic in .eval() mode.

    This is non-negotiable: any randomness leaking into eval mode means the
    inflate-time model output is non-deterministic, which violates the contest
    rule that inflate must produce the same .raw given the same archive.
    """
    from tac.fp4_quantize import FP4Parametrize

    p = FP4Parametrize(DEFAULT_CODEBOOK.clone(), DEFAULT_BLOCK_SIZE,
                       stochastic=True, robust_scale=False)
    weight = torch.full((32,), 0.3, dtype=torch.float32)
    weight[0] = 6.0

    p.train()
    torch.manual_seed(0)
    out_train_a = p(weight)
    torch.manual_seed(1)
    out_train_b = p(weight)
    # Different seeds → different stochastic samples
    assert not torch.allclose(out_train_a, out_train_b)

    p.eval()
    out_eval_a = p(weight)
    out_eval_b = p(weight)
    # Eval mode → deterministic
    assert torch.allclose(out_eval_a, out_eval_b)


# ── Train ↔ export consistency ────────────────────────────────────────────

# Round-trip tolerance: scales are stored as fp16 on disk (DESIGNED 2x saving),
# so the dequantized weight will differ from the in-memory FP4 result by
# fp16-precision quantization noise on the per-block scale. ~1e-3 on weights
# scaled to magnitude ~1 is the practical floor.
ROUND_TRIP_ATOL = 5e-3


def _roundtrip_diff(robust_scale: bool) -> float:
    """Compare FakeQuantFP4 output (eval mode) to quantize_fp4 + dequantize_fp4."""
    torch.manual_seed(0)
    base = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 4, 1))
    base.eval()

    # Capture pre-export state dict (the fp32 weights)
    pre_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    # In-memory parametrize (eval mode, deterministic) gives the "training-time
    # round-trip" value the loss sees in eval.
    qat = QATRendererFP4(base, robust_scale=robust_scale, stochastic=False)
    qat.eval()
    in_memory_w = base[0].weight.detach().clone()

    # Strip parametrizations to get original floats back
    qat.remove_hooks()
    # Restore the same fp32 weights for export (parametrize machinery may have
    # stored the original differently; force them explicitly to match)
    base.load_state_dict(pre_state)

    # Export path
    packed = quantize_fp4(base.state_dict(), robust_scale=robust_scale)
    restored = dequantize_fp4(packed)

    # Both paths use the same codebook + scale logic; only the storage of
    # scales as fp16 distinguishes them.
    return (restored["0.weight"] - in_memory_w).abs().max().item()


def test_train_export_round_trip_default_scale() -> None:
    """Robust_scale=False (legacy): train↔export agree within fp16 scale precision."""
    diff = _roundtrip_diff(robust_scale=False)
    assert diff < ROUND_TRIP_ATOL, (
        f"train↔export drift {diff:.6f} > tolerance {ROUND_TRIP_ATOL} — "
        f"silent corruption between QAT and inflate"
    )


def test_train_export_round_trip_robust_scale() -> None:
    """Robust_scale=True: train↔export agree (matched percentile path)."""
    diff = _roundtrip_diff(robust_scale=True)
    assert diff < ROUND_TRIP_ATOL, (
        f"train↔export drift {diff:.6f} with robust_scale=True"
    )


# ── Backward compat ───────────────────────────────────────────────────────


def test_default_args_match_legacy_codebook_and_scale() -> None:
    """fake_quant_fp4 with default args = pre-fix behaviour (DEFAULT codebook,
    max-based scale, deterministic rounding). This guards against silent
    regression in models trained before R-FP4-fix landed."""
    torch.manual_seed(123)
    w = torch.randn(64, 16)
    out_default = fake_quant_fp4(w)
    out_explicit = fake_quant_fp4(w, DEFAULT_CODEBOOK, DEFAULT_BLOCK_SIZE,
                                   stochastic=False, robust_scale=False)
    assert torch.allclose(out_default, out_explicit, atol=1e-6)


def test_qat_wrapper_default_codebook_unchanged() -> None:
    """QATRendererFP4() with no codebook arg must use DEFAULT_CODEBOOK
    (old models continue to work). RESIDUAL is opt-in only."""
    base = nn.Conv2d(3, 8, 3)
    qat = QATRendererFP4(base)
    assert torch.equal(qat.codebook, DEFAULT_CODEBOOK)
    assert qat.stochastic is False
    assert qat.robust_scale is False
