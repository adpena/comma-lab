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


# ===========================================================================
# MED-2 codec mechanism (2026-06-12): score-aware grid → SMALLER real brotli blob.
# The R1 audit flagged Lever-4's byte-win as an UNPROVEN indirect effect (the codec
# always quantizes at 127). The MED-2 probe
# (``experiments/probe_lever4_qat_brotli_blob_delta.py``) measured a -3263 B real
# decoder-blob delta (70264 vs 73527) at equal advisory d_seg from a REAL frozen-scorer
# sensitivity. This test pins the CODEC-side mechanism the probe proved: applying the
# score-aware per-tensor grid (low-sensitivity tensors coarsened) produces a strictly
# SMALLER vendored-codec brotli decoder blob than uniform-127, AND the coarse snap
# SURVIVES the codec's own 127-level requant (fewer distinct symbols). A fake that left
# the grid uniform (or that the codec's 127-requant erased) would FAIL.
# ===========================================================================
def test_score_aware_grid_yields_smaller_real_brotli_blob_than_uniform():
    pytest = __import__("pytest")
    try:
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parents[4]
        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))
        from tac.torch_vehicle.vendored_imports import import_vendored

        codec = import_vendored("codec")
        model = import_vendored("model")
    except Exception as exc:  # pragma: no cover - vendored clone may be absent in CI
        pytest.skip(f"vendored codec/model unavailable: {exc}")

    # A real basin-sized decoder with random (high-entropy) weights — the regime where
    # the codec uses most of the 127-symbol alphabet (so coarsening has room to help).
    torch.manual_seed(0)
    dec = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(384, 512))
    base_sd = {k: v.detach().float() for k, v in dec.state_dict().items()}

    names = [
        n
        for n, m in dec.named_modules()
        if isinstance(m, (nn.Conv2d, nn.Linear)) and getattr(m, "weight", None) is not None
    ]
    # Deterministic NON-uniform sensitivity: first half low, second half high → the low
    # tensors get a coarser grid (the score-aware allocation).
    sens = {n: (0.01 if i < len(names) // 2 else 100.0) for i, n in enumerate(names)}

    def _blob(state_dict):
        return len(codec.encode_decoder(codec.quantize_state_dict(state_dict)))

    # Uniform-127 arm (default-preserving): sensitivity=None.
    dec_u = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(384, 512))
    dec_u.load_state_dict(base_sd)
    o_u = apply_score_aware_qat(dec_u, sensitivity=None)
    uniform_blob = _blob(dec_u.state_dict())
    restore_score_aware_qat(dec_u, o_u)

    # Score-aware arm: low-sensitivity tensors coarsened.
    dec_s = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(384, 512))
    dec_s.load_state_dict(base_sd)
    apply_score_aware_qat(dec_s, sensitivity=sens)
    score_blob = _blob(dec_s.state_dict())

    assert score_blob < uniform_blob, (
        f"score-aware blob {score_blob} must be < uniform-127 blob {uniform_blob} "
        "(coarsening low-sensitivity tensors must shrink the REAL brotli decoder blob — "
        "the MED-2 indirect-win mechanism); equal bytes would mean the grid never "
        "actually changed the deployed encoding (FAKE)"
    )


def test_uniform_score_aware_blob_equals_vendored_uniform_blob():
    """The default-preserving guard at the CODEC byte level: sensitivity=None produces
    a brotli decoder blob BIT-IDENTICAL to the vendored uniform-127 codec path (so an
    unconditioned score-aware-QAT run ships the exact same bytes as today)."""
    pytest = __import__("pytest")
    try:
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parents[4]
        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))
        from tac.torch_vehicle.vendored_imports import import_vendored

        codec = import_vendored("codec")
        model = import_vendored("model")
        losses = import_vendored("losses")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"vendored codec/model/losses unavailable: {exc}")

    torch.manual_seed(1)
    dec = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(384, 512))
    base_sd = {k: v.detach().float() for k, v in dec.state_dict().items()}

    # score-aware (sensitivity=None) snap.
    dec_a = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(384, 512))
    dec_a.load_state_dict(base_sd)
    apply_score_aware_qat(dec_a, sensitivity=None)
    blob_a = codec.encode_decoder(codec.quantize_state_dict(dec_a.state_dict()))

    # vendored uniform QAT snap (apply_qat / restore_qat).
    dec_b = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(384, 512))
    dec_b.load_state_dict(base_sd)
    o_b = losses.apply_qat(dec_b)
    blob_b = codec.encode_decoder(codec.quantize_state_dict(dec_b.state_dict()))
    losses.restore_qat(dec_b, o_b)

    assert blob_a == blob_b, (
        "sensitivity=None score-aware QAT must produce a brotli blob BIT-IDENTICAL to "
        "the vendored uniform-127 QAT (default-preserving guard at the codec layer)"
    )


# ===========================================================================
# R5 (2026-06-12) — determinism of the QAT grid on TIED sensitivities (lens A)
# + long-run sensitivity-EMA numerical stability over many steps (lens C).
# ===========================================================================
def test_rank_normalize_all_tied_collapses_to_uniform_every_call():
    """``_rank_normalize`` on an EXACTLY-tied vector returns all-0.5 (the uniform
    fallback) DETERMINISTICALLY — the one ``argsort``-on-ties path in the lever
    code must not produce a run-to-run-varying ordering."""
    from tac.torch_vehicle.score_aware_qat import _rank_normalize

    tied = torch.tensor([3.0, 3.0, 3.0, 3.0], dtype=torch.float64)
    r1 = _rank_normalize(tied)
    r2 = _rank_normalize(tied)
    assert torch.equal(r1, r2), "rank-normalize is non-deterministic on a tied vector"
    assert bool((r1 == 0.5).all().item()), "tied vector must collapse to uniform 0.5"


def test_per_tensor_levels_deterministic_under_partial_ties():
    """A sensitivity dict with TWO tied tensors produces an IDENTICAL per-tensor
    level map across repeated calls — the ``argsort`` tie-break must not silently
    change the QAT grid (hence the archive) run-to-run. A nondeterministic
    tie-break would make the deployed bytes differ on a tie."""
    names = ["blocks.0", "blocks.1", "blocks.2", "blocks.3"]
    sens = {"blocks.0": 1.0, "blocks.1": 2.0, "blocks.2": 2.0, "blocks.3": 9.0}
    maps = [per_tensor_levels_from_sensitivity(dict(sens), names) for _ in range(8)]
    assert all(m == maps[0] for m in maps), (
        f"QAT level map is non-deterministic under a partial tie: {maps}"
    )


def test_sensitivity_ema_bounded_and_finite_over_long_run():
    """The sensitivity EMA ``s_t = decay*prior + (1-decay)*||grad||`` stays FINITE
    and BOUNDED by max(grad-norm) over a LONG sequence of steps (including
    occasional gradient spikes) — the multi-thousand-step stability the 80-epoch
    R2 run could not reach. The EMA is a convex combination of bounded terms, so
    it cannot drift/overflow; this pins that property over 12000 updates."""
    torch.manual_seed(5)
    dec = nn.Sequential(nn.Conv2d(2, 3, 3), nn.Flatten(), nn.Linear(3 * 6 * 6, 4))
    name = next(n for n, m in dec.named_modules() if isinstance(m, nn.Conv2d))
    ema: dict[str, float] = {}
    max_seen = 0.0
    for step in range(12000):
        dec.zero_grad()
        x = torch.randn(1, 2, 8, 8)
        # inject a large spike every 500 steps to stress the EMA's boundedness.
        scale = 1e6 if (step % 500 == 0) else 1.0
        (dec(x).pow(2).mean() * scale).backward()
        accumulate_tensor_sensitivity(dec, ema, decay=0.99)
        cur = float(dict(dec.named_modules())[name].weight.grad.norm())
        max_seen = max(max_seen, cur)
        assert all(
            torch.isfinite(torch.tensor(v)) for v in ema.values()
        ), f"sensitivity EMA went non-finite at step {step}: {ema}"
    # An EMA of a non-negative bounded sequence stays within [0, max_seen] (+slack).
    assert all(0.0 <= v <= max_seen * (1 + 1e-6) for v in ema.values()), (
        f"sensitivity EMA exceeded the max grad norm {max_seen}: {ema}"
    )
