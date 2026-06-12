# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the Lever-1 differentiable brotli-rate surrogate
(``tac.losses.rate_surrogate``).

The module ACTUALLY computes the order-1 conditional weight entropy + the latent
temporal-delta entropy on the real weights/latents the codec compresses. These
tests verify the MECHANISM (not constants):

1. A smoother (longer-run) INT8 weight stream has LOWER conditional entropy than a
   high-entropy random stream — the lever rewards scan-order-smooth weights (long
   brotli LZ matches), which is its entire reason for existing.
2. The conditional entropy ``H(W|W_prev) <= H(W)`` marginal — the conditioning
   inequality, the true-bound property that makes it a conservative brotli proxy.
3. The gradient flows to the real weights/latents so the optimizer can descend it.
4. Temporally-smooth latents have LOWER delta entropy than noisy latents — the
   currently-unexploited byte lever.

If any of these were a constant (a FAKE ``return 0.0`` / ``return baseline``), the
ordering / gradient / bound assertions below would FAIL.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tac.losses.rate_surrogate import (
    brotli_rate_surrogate,
    conditional_weight_entropy,
    latent_delta_entropy,
)


def _conv_decoder(weight: torch.Tensor) -> nn.Module:
    """A 1-Conv decoder whose weight is exactly ``weight`` (so we control the symbol
    stream the surrogate reads)."""
    m = nn.Conv2d(1, 1, kernel_size=weight.shape[-1], bias=False)
    with torch.no_grad():
        m.weight.copy_(weight)
    return nn.Sequential(m)


# ---------------------------------------------------------------------------
# (1) smoother weights -> lower conditional entropy (the mechanism).
# ---------------------------------------------------------------------------
def test_smooth_weight_stream_has_lower_conditional_entropy_than_random():
    """A monotone ramp (long scan-order runs, small adjacent deltas) has LOWER
    H(W|W_prev) than a high-entropy random stream — the lever's core claim."""
    n = 1024
    # Smooth: a slow ramp (adjacent symbols nearly equal → low conditional entropy).
    smooth = torch.linspace(-1.0, 1.0, n).reshape(1, 1, 1, n)
    # Random: i.i.d. (adjacent symbols independent → high conditional entropy).
    g = torch.Generator().manual_seed(0)
    rand = torch.randn(1, 1, 1, n, generator=g)

    h_smooth = conditional_weight_entropy(_conv_decoder(smooth), device="cpu").item()
    h_rand = conditional_weight_entropy(_conv_decoder(rand), device="cpu").item()

    assert h_smooth < h_rand, (
        f"smooth-stream H(W|W_prev)={h_smooth:.4f} must be < random {h_rand:.4f} "
        "(the surrogate is not actually reading the scan-order structure — FAKE)"
    )


# ---------------------------------------------------------------------------
# (2) conditional <= marginal (the true-bound property).
# ---------------------------------------------------------------------------
def test_conditional_entropy_below_marginal_true_bound():
    """H(W|W_prev) <= H(W) marginal (conditioning never increases entropy). Compute
    the marginal on the SAME soft-INT8 grid the surrogate uses and assert the bound."""
    g = torch.Generator().manual_seed(1)
    w = torch.randn(1, 1, 1, 2048, generator=g)
    dec = _conv_decoder(w)

    h_cond = conditional_weight_entropy(dec, device="cpu").item()

    # Marginal H(W) on the same grid: reuse the surrogate's soft-bin machinery so the
    # comparison is apples-to-apples (same sigma, same bins).
    from tac.losses.rate_surrogate import RateSurrogateConfig, _soft_bin_assignment

    cfg = RateSurrogateConfig()
    bins = torch.arange(cfg.bin_min, cfg.bin_max + 1, dtype=torch.float32)
    wn = (w.flatten() / (w.abs().max() / 127.0))[: cfg.pair_sample_size + 1]
    sa = _soft_bin_assignment(wn, bins, cfg.sigma, cfg.eps)
    p = sa.mean(dim=0)
    p = p / p.sum()
    h_marg = -(p * torch.log2(p + cfg.eps)).sum().item()

    assert h_cond <= h_marg + 1e-6, (
        f"H(W|W_prev)={h_cond:.4f} must be <= H(W)={h_marg:.4f} (true-bound FAIL)"
    )


# ---------------------------------------------------------------------------
# (3) real gradient to the weights.
# ---------------------------------------------------------------------------
def test_conditional_entropy_has_gradient_to_weights():
    """The surrogate is differentiable to the weights so the optimizer can descend it
    (a constant-returning fake would give zero/None grad)."""
    g = torch.Generator().manual_seed(2)
    w = torch.randn(1, 1, 1, 512, generator=g)
    m = nn.Conv2d(1, 1, kernel_size=512, bias=False)
    with torch.no_grad():
        m.weight.copy_(w)
    m.weight.requires_grad_(True)
    dec = nn.Sequential(m)

    h = conditional_weight_entropy(dec, device="cpu")
    h.backward()
    assert m.weight.grad is not None and m.weight.grad.abs().sum().item() > 0, (
        "conditional entropy produced no real gradient to the weights (FAKE)"
    )


# ---------------------------------------------------------------------------
# (4) latent temporal-delta entropy: smooth < noisy.
# ---------------------------------------------------------------------------
def test_latent_delta_entropy_smooth_below_noisy():
    """Temporally-smooth latents (small deltas) have LOWER delta entropy than noisy
    latents (large deltas) — the currently-unexploited byte lever (the codec
    delta-codes the latents; small deltas → tiny brotli)."""
    n_pairs, dim = 200, 8
    # Smooth: a slow drift per dim (small temporal deltas).
    t = torch.linspace(0, 1, n_pairs).unsqueeze(1)
    smooth = t.repeat(1, dim) + 0.01 * torch.randn(n_pairs, dim, generator=torch.Generator().manual_seed(3))
    # Noisy: i.i.d. per pair (large temporal deltas).
    noisy = torch.randn(n_pairs, dim, generator=torch.Generator().manual_seed(4))

    h_smooth = latent_delta_entropy(smooth, device="cpu").item()
    h_noisy = latent_delta_entropy(noisy, device="cpu").item()

    assert h_smooth < h_noisy, (
        f"smooth-latent delta entropy {h_smooth:.4f} must be < noisy {h_noisy:.4f} "
        "(the latent rate term is not reading the temporal structure — FAKE)"
    )


def test_latent_delta_entropy_has_gradient():
    """The latent delta entropy is differentiable to the latents (so the optimizer
    can reward temporal smoothness)."""
    z = torch.randn(50, 8, requires_grad=True)
    h = latent_delta_entropy(z, device="cpu")
    h.backward()
    assert z.grad is not None and z.grad.abs().sum().item() > 0, (
        "latent delta entropy produced no gradient to the latents (FAKE)"
    )


# ---------------------------------------------------------------------------
# brotli_rate_surrogate composition.
# ---------------------------------------------------------------------------
def test_brotli_rate_surrogate_returns_both_terms():
    """``brotli_rate_surrogate`` returns (H_cond_weights, R_lat); latents=None gives
    R_lat=0 (decoder-only)."""
    g = torch.Generator().manual_seed(5)
    dec = _conv_decoder(torch.randn(1, 1, 1, 256, generator=g))
    z = torch.randn(20, 4)

    h_cond, r_lat = brotli_rate_surrogate(dec, z, device="cpu")
    assert h_cond.item() > 0 and r_lat.item() > 0

    h_cond2, r_lat2 = brotli_rate_surrogate(dec, None, device="cpu")
    assert r_lat2.item() == 0.0, "latents=None must give R_lat=0 (decoder-only)"
    assert torch.allclose(h_cond, h_cond2), "weight term must not depend on the latent arg"


def test_rate_surrogate_score_claim_flags_are_false():
    """The module carries the non-promotable score-claim discipline flags (sister of
    cat_entropy_v2): it changes TRAINING DYNAMICS, asserts NO score."""
    from tac.losses import rate_surrogate

    assert rate_surrogate.SCORE_CLAIM is False
    assert rate_surrogate.PROMOTION_ELIGIBLE is False
    assert rate_surrogate.READY_FOR_EXACT_EVAL_DISPATCH is False
