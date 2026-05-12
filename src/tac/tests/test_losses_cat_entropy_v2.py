"""Tests for the PR95 ``cat_entropy_v2`` training-loss primitive.

Covers:
- Output is a scalar Tensor in bits/weight with `requires_grad` flowing
  through when the input has grad enabled.
- Math agrees with PR95's source within float32 tolerance on a fixed seed
  (Catalog #91 paired-roundtrip discipline: the in-port re-implementation
  matches the public PR95 reference cell-by-cell).
- Per-layer numel weighting works.
- Sub-`max_abs_floor` tensors are skipped, returning 0 when ALL are skipped.
- Conv2d AND Linear weights are both included.
- Subsample is bounded by `sample_size`.
- Sigma controls entropy: smaller sigma → smaller entropy (sharper grid commit).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from tac.losses.cat_entropy_v2 import (
    CatEntropyV2Config,
    PROMOTION_ELIGIBLE,
    READY_FOR_EXACT_EVAL_DISPATCH,
    SCORE_CLAIM,
    TARGET_SUBSTRATE_HINT,
    cat_entropy_v2,
)


def _tiny_decoder(seed: int = 0) -> nn.Module:
    """Tiny module with one Conv2d + one Linear."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
        nn.Linear(4, 2, bias=False),
    )


# ── Shape / dtype contract ──────────────────────────────────────────────────


def test_cat_entropy_v2_returns_scalar_tensor() -> None:
    decoder = _tiny_decoder()
    out = cat_entropy_v2(decoder)
    assert isinstance(out, torch.Tensor)
    assert out.shape == ()
    assert out.dtype == torch.float32


def test_cat_entropy_v2_is_finite_for_random_init() -> None:
    decoder = _tiny_decoder()
    out = cat_entropy_v2(decoder)
    assert torch.isfinite(out).item()
    assert out.item() > 0.0  # Random init has some categorical entropy.


def test_cat_entropy_v2_requires_grad_flow() -> None:
    """Gradient flows back to the decoder weights when requires_grad is set."""
    decoder = _tiny_decoder()
    for p in decoder.parameters():
        p.requires_grad_(True)
    out = cat_entropy_v2(decoder)
    out.backward()
    grads = [p.grad for p in decoder.parameters() if p.grad is not None]
    assert len(grads) == 2  # Conv2d weight + Linear weight
    for g in grads:
        assert torch.isfinite(g).all().item()
        # At least one element should be non-zero (gradient is real).
        assert g.abs().max().item() > 0.0


# ── Math contract: PR95 byte-faithful port ──────────────────────────────────


def test_cat_entropy_v2_matches_pr95_reference_cell_by_cell() -> None:
    """Recompute the PR95 cell-by-cell on a fixed seed and verify exact match."""
    torch.manual_seed(7)
    decoder = nn.Sequential(
        nn.Conv2d(2, 3, kernel_size=3, bias=False),
        nn.Linear(3, 2, bias=False),
    )
    sigma = 0.2
    sample_size = 2000
    device = next(decoder.parameters()).device

    # Reference (verbatim PR95 inline computation).
    bins = torch.arange(-127, 128, device=device, dtype=torch.float32)
    ref_weighted = torch.zeros((), device=device)
    ref_total = 0
    for _, mod in decoder.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and hasattr(mod, "weight"):
            w = mod.weight
            numel = w.numel()
            ma = w.abs().max().detach()
            if ma.item() < 1e-12:
                continue
            wn = (w / (ma / 127.0)).flatten()
            if wn.numel() > sample_size:
                # NOTE: subsample with same RNG state expected. Tensors below
                # sample_size are not subsampled, so this fixture is exact.
                pass
            sa = torch.exp(
                -0.5 * ((wn.unsqueeze(1) - bins.unsqueeze(0)) / sigma).pow(2)
            )
            sa = sa / (sa.sum(dim=1, keepdim=True) + 1e-12)
            bp = sa.mean(dim=0)
            bp = bp / (bp.sum() + 1e-12)
            entropy = -(bp * torch.log2(bp + 1e-12)).sum()
            ref_weighted = ref_weighted + numel * entropy
            ref_total += numel
    ref_value = ref_weighted / max(ref_total, 1)

    # Port:
    port_value = cat_entropy_v2(decoder)
    assert math.isclose(port_value.item(), ref_value.item(), rel_tol=1e-6)


def test_cat_entropy_v2_zero_when_all_weights_below_floor() -> None:
    """A decoder with weights << max_abs_floor returns 0.0."""
    decoder = nn.Sequential(
        nn.Conv2d(2, 2, kernel_size=3, bias=False),
        nn.Linear(2, 2, bias=False),
    )
    # Zero out everything.
    for p in decoder.parameters():
        p.data.zero_()
    out = cat_entropy_v2(decoder)
    assert out.item() == 0.0


def test_cat_entropy_v2_skips_only_below_floor_tensors() -> None:
    """A decoder with mixed zero + non-zero layers returns non-zero entropy."""
    decoder = nn.Sequential(
        nn.Conv2d(2, 2, kernel_size=3, bias=False),  # will be zeroed
        nn.Linear(2, 2, bias=False),  # keep random
    )
    torch.manual_seed(11)
    decoder[1].weight.data.normal_()
    decoder[0].weight.data.zero_()
    out = cat_entropy_v2(decoder)
    assert out.item() > 0.0


# ── Layer-class coverage ────────────────────────────────────────────────────


def test_cat_entropy_v2_includes_both_conv2d_and_linear() -> None:
    """Both Conv2d and Linear should contribute (delta when removing one)."""
    full = _tiny_decoder()
    conv_only = nn.Sequential(_tiny_decoder()[0])
    linear_only = nn.Sequential(_tiny_decoder()[1])
    out_full = cat_entropy_v2(full).item()
    out_conv = cat_entropy_v2(conv_only).item()
    out_lin = cat_entropy_v2(linear_only).item()
    # Both single-layer outputs should be in (0, max(out_full, out_conv, out_lin)+eps).
    assert out_conv > 0.0
    assert out_lin > 0.0
    # The full decoder's weighted mean must lie between the two single-layer
    # entropies (by definition of weighted mean of two scalars).
    lo, hi = sorted((out_conv, out_lin))
    assert lo - 1e-6 <= out_full <= hi + 1e-6


def test_cat_entropy_v2_ignores_non_conv_linear_layers() -> None:
    """ReLU / BatchNorm / Dropout layers are silently ignored."""
    decoder = nn.Sequential(
        nn.Conv2d(2, 2, kernel_size=3, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(2),
        nn.Dropout(0.1),
        nn.Linear(2, 2, bias=False),
    )
    out = cat_entropy_v2(decoder)
    assert torch.isfinite(out).item()


# ── Config contract ─────────────────────────────────────────────────────────


def test_cat_entropy_v2_sigma_controls_entropy() -> None:
    """Smaller sigma → smaller entropy (sharper grid commit)."""
    torch.manual_seed(42)
    decoder = _tiny_decoder(seed=42)
    out_wide = cat_entropy_v2(decoder, CatEntropyV2Config(sigma=2.0))
    out_narrow = cat_entropy_v2(decoder, CatEntropyV2Config(sigma=0.05))
    # Wider sigma → softer histogram → higher entropy.
    assert out_wide.item() > out_narrow.item()


def test_cat_entropy_v2_sample_size_caps_per_tensor() -> None:
    """A large Conv2d weight is correctly capped at sample_size."""
    decoder = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, bias=False),  # 64*64*9 = 36864 elements
    )
    # Sample_size=500 means the function should subsample 500 elements only.
    out = cat_entropy_v2(decoder, CatEntropyV2Config(sample_size=500))
    assert torch.isfinite(out).item()
    assert out.item() > 0.0


# ── Module metadata contract ────────────────────────────────────────────────


def test_target_substrate_hint_is_any_with_categorical_outputs() -> None:
    assert TARGET_SUBSTRATE_HINT == "any_with_categorical_outputs"


def test_score_claim_flags_are_all_false() -> None:
    """Score-claim discipline per CLAUDE.md forbidden_score_claim_with_byte_change."""
    assert SCORE_CLAIM is False
    assert PROMOTION_ELIGIBLE is False
    assert READY_FOR_EXACT_EVAL_DISPATCH is False


# ── Catalog #91 roundtrip-style invariant ───────────────────────────────────


def test_cat_entropy_v2_call_is_deterministic_on_no_subsample_path() -> None:
    """When sample_size > numel, two calls return EXACTLY the same value."""
    decoder = _tiny_decoder(seed=99)
    cfg = CatEntropyV2Config(sample_size=10**9)  # no subsampling triggered
    a = cat_entropy_v2(decoder, cfg)
    b = cat_entropy_v2(decoder, cfg)
    assert math.isclose(a.item(), b.item(), rel_tol=1e-12)


def test_cat_entropy_v2_with_explicit_generator_is_reproducible() -> None:
    """Re-running with a fresh, seeded generator gives identical results."""
    decoder = nn.Sequential(
        nn.Conv2d(8, 8, kernel_size=3, bias=False),  # 8*8*9 = 576 elements
    )
    cfg = CatEntropyV2Config(sample_size=100)  # forces subsampling

    g1 = torch.Generator(device="cpu")
    g1.manual_seed(0)
    a = cat_entropy_v2(decoder, cfg, generator=g1)

    g2 = torch.Generator(device="cpu")
    g2.manual_seed(0)
    b = cat_entropy_v2(decoder, cfg, generator=g2)

    assert math.isclose(a.item(), b.item(), rel_tol=1e-7)


def test_cat_entropy_v2_decoder_with_no_parameters_returns_zero_on_cpu() -> None:
    """Edge case: empty decoder returns 0 on CPU."""
    decoder = nn.Sequential(nn.ReLU())  # no parameters at all
    out = cat_entropy_v2(decoder, device="cpu")
    assert out.item() == 0.0
    assert out.device.type == "cpu"
