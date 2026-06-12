# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the Lever-4 score-aware QAT
(``tac.torch_vehicle.score_aware_qat``).

The module ACTUALLY varies the per-tensor INT8 quantization grid by the supplied
score-sensitivity, on the real weights. These tests verify the MECHANISM (not
constants):

1. A HIGH-sensitivity tensor gets a FINER grid (more INT8 levels, smaller quant
   error) than a LOW-sensitivity tensor (the water-filling of the bit budget).
2. Uniform/absent sensitivity reproduces the vendored uniform 127-level quant
   BIT-IDENTICALLY (the default-preserving guard) — and ``n_levels=127`` matches
   the vendored ``fake_quantize`` exactly.
3. The STE gradient flows through (forward=quant, backward=identity), so a
   score-aware-QAT forward is trainable.
4. ``accumulate_tensor_sensitivity`` populates the EMA from real ``w.grad``
   magnitudes (decay=0 → ema == current norm exactly).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tac.torch_vehicle.score_aware_qat import (
    ScoreAwareQATConfig,
    _fake_quantize_n,
    accumulate_tensor_sensitivity,
    apply_score_aware_qat,
    per_tensor_levels_from_sensitivity,
    restore_score_aware_qat,
)


# ---------------------------------------------------------------------------
# (1) high sensitivity -> finer grid -> smaller quant error.
# ---------------------------------------------------------------------------
def test_high_sensitivity_tensor_gets_finer_grid_smaller_error():
    """The HIGH-sensitivity tensor gets MORE INT8 levels and a SMALLER quant error
    than the LOW-sensitivity tensor (the argmax boundary is protected)."""
    names = ["low", "high"]
    sens = {"low": 0.0, "high": 100.0}
    levels = per_tensor_levels_from_sensitivity(sens, names)
    assert levels["high"] > levels["low"], "high-sensitivity tensor must get a finer grid"

    g = torch.Generator().manual_seed(0)
    w = torch.randn(4096, generator=g)
    err_low = (_fake_quantize_n(w, levels["low"]) - w).abs().mean().item()
    err_high = (_fake_quantize_n(w, levels["high"]) - w).abs().mean().item()
    assert err_high < err_low, (
        f"finer grid (high-sens, {levels['high']} levels, err={err_high:.5f}) must have "
        f"smaller quant error than coarse (low-sens, {levels['low']}, err={err_low:.5f})"
    )


# ---------------------------------------------------------------------------
# (2) default-preserving: uniform/None == vendored uniform 127.
# ---------------------------------------------------------------------------
def test_none_sensitivity_uses_base_levels_everywhere():
    """``sensitivity is None`` returns the base 127 levels for EVERY tensor (the
    default-preserving fallback)."""
    names = ["a", "b", "c"]
    levels = per_tensor_levels_from_sensitivity(None, names)
    assert all(v == 127 for v in levels.values())


def test_uniform_sensitivity_falls_back_to_base():
    """UNIFORM sensitivity (all tensors equal) collapses to the base level count for
    every tensor — bit-identical to uniform QAT."""
    names = ["a", "b", "c"]
    sens = {"a": 5.0, "b": 5.0, "c": 5.0}
    levels = per_tensor_levels_from_sensitivity(sens, names)
    assert all(v == 127 for v in levels.values())


def test_n_levels_127_matches_a_reference_symmetric_quant():
    """``_fake_quantize_n(w, 127)`` is the canonical per-tensor symmetric INT8 STE — a
    hand-rolled reference at 127 levels matches it bit-for-bit."""
    g = torch.Generator().manual_seed(1)
    w = torch.randn(2000, generator=g)
    out = _fake_quantize_n(w, 127)
    ma = w.abs().max()
    scale = ma / 127
    q = (w / scale).round().clamp(-127, 127)
    ref = (q * scale - w).detach() + w
    assert torch.equal(out, ref), "127-level fake-quant diverged from the symmetric reference"


def test_apply_score_aware_qat_none_matches_uniform_on_real_decoder():
    """On a small Conv/Linear decoder, ``apply_score_aware_qat(decoder, None)``
    quantizes EVERY tensor at 127 levels — bit-identical to applying
    ``_fake_quantize_n(w, 127)`` to each weight directly."""
    torch.manual_seed(2)
    dec = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Conv2d(4, 2, 3), nn.Flatten(), nn.Linear(8, 5))
    # Make linear input match: use a dummy forward to size it — instead just test the
    # weight quant directly on the modules.
    ref = {}
    for name, m in dec.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            ref[name] = _fake_quantize_n(m.weight.data.clone(), 127)
    originals = apply_score_aware_qat(dec, None)
    for name, m in dec.named_modules():
        if name in ref:
            assert torch.equal(m.weight.data, ref[name]), (
                f"score-aware QAT (None) != uniform 127 at {name} (default-FAIL)"
            )
    restore_score_aware_qat(dec, originals)
    # restore puts the originals back.
    for name, m in dec.named_modules():
        if name in originals:
            assert torch.equal(m.weight.data, originals[name])


# ---------------------------------------------------------------------------
# (3) STE gradient flows through.
# ---------------------------------------------------------------------------
def test_fake_quantize_ste_gradient_is_identity():
    """The STE makes the backward an identity pass-through (so a quantized forward is
    trainable) — d(quant(w))/dw == 1."""
    w = torch.randn(100, requires_grad=True)
    out = _fake_quantize_n(w, 64)
    out.sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w)), "STE backward is not identity"


# ---------------------------------------------------------------------------
# (4) sensitivity EMA accumulates from grad.
# ---------------------------------------------------------------------------
def test_accumulate_sensitivity_decay0_equals_grad_norm():
    """``accumulate_tensor_sensitivity`` with decay=0 sets ema[name] == ||w.grad||
    exactly (the mechanism reads the real grad)."""
    torch.manual_seed(3)
    dec = nn.Sequential(nn.Conv2d(2, 3, 3), nn.Flatten(), nn.Linear(3 * 6 * 6, 4))
    x = torch.randn(1, 2, 8, 8)
    dec(x).pow(2).mean().backward()
    ema: dict[str, float] = {}
    accumulate_tensor_sensitivity(dec, ema, decay=0.0)
    assert ema, "no sensitivity accumulated"
    for name, m in dec.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight.grad is not None:
            assert abs(ema[name] - float(m.weight.grad.norm())) < 1e-5


def test_accumulate_sensitivity_ema_smooths():
    """The EMA smooths: a second update with a different grad moves the value toward
    the new norm by (1-decay) — not all the way (proving it's an EMA, not a replace)."""
    torch.manual_seed(4)
    dec = nn.Sequential(nn.Linear(4, 4))
    name = next(n for n, m in dec.named_modules() if isinstance(m, nn.Linear))
    # First grad.
    dec(torch.randn(1, 4)).sum().backward()
    ema: dict[str, float] = {}
    accumulate_tensor_sensitivity(dec, ema, decay=0.9)
    first = ema[name]
    # Second, different grad.
    dec.zero_grad()
    (dec(torch.randn(1, 4)).sum() * 10.0).backward()
    accumulate_tensor_sensitivity(dec, ema, decay=0.9)
    second = ema[name]
    cur = float(dict(dec.named_modules())[name].weight.grad.norm())
    expected = 0.9 * first + 0.1 * cur
    assert abs(second - expected) < 1e-5, "EMA update is not decay*prior + (1-decay)*cur"


def test_score_aware_qat_score_claim_flags_are_false():
    """Non-promotable score-claim discipline flags (changes TRAINING DYNAMICS, asserts
    NO score)."""
    from tac.torch_vehicle import score_aware_qat

    assert score_aware_qat.SCORE_CLAIM is False
    assert score_aware_qat.PROMOTION_ELIGIBLE is False
    assert score_aware_qat.READY_FOR_EXACT_EVAL_DISPATCH is False


def test_min_abs_levels_floor_protects_near_zero_sensitivity():
    """A near-zero-sensitivity tensor never degenerates below ``min_abs_levels`` (a
    1-level all-zero quant would destroy it)."""
    names = ["a", "b"]
    sens = {"a": 0.0, "b": 1e9}
    cfg = ScoreAwareQATConfig(min_abs_levels=16)
    levels = per_tensor_levels_from_sensitivity(sens, names, cfg)
    assert levels["a"] >= 16
