# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the frontier int5/int4 score-aware QAT module.

These verify the module ACTUALLY varies the quantization grid per the supplied
allocation, on real weights, with the STE gradient flowing — NOT that it returns
canonical markers. Every test would FAIL if the body were replaced by a constant.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from tac.frontier_int5_qat import (
    NLEVELS_FOR_NBITS,
    EMAState,
    FrontierInt5QATError,
    FrontierQATConfig,
    apply_frontier_qat,
    hard_quantize_state_dict_to_nbits,
    per_tensor_nbits_for_decoder,
    restore_frontier_qat,
)
from tac.post_hoc_weight_shrink import intn_qdq
from tac.torch_vehicle.score_aware_qat import _fake_quantize_n


def _tiny_decoder() -> nn.Module:
    """A tiny module with named Conv2d/Linear layers mimicking the frontier naming."""

    class _Dec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv2d(4, 8, 3, padding=1)
            self.blocks = nn.ModuleList([nn.Conv2d(8, 8, 3, padding=1) for _ in range(6)])
            self.rgb_1 = nn.Conv2d(8, 3, 1)

        def forward(self, x):  # pragma: no cover - not used in these tests
            return x

    torch.manual_seed(0)
    return _Dec()


# ── grid VARIES the weights (the mechanism, not a constant) ──────────────────


def test_int5_grid_actually_quantizes_to_31_distinct_values():
    """int5 (n_levels=15) snaps a tensor to <= 31 distinct values — the REAL grid."""
    torch.manual_seed(1)
    w = torch.randn(64, 64)
    q = _fake_quantize_n(w, NLEVELS_FOR_NBITS[5])
    n_distinct = int(torch.unique(q).numel())
    assert n_distinct <= 31, f"int5 must have <=31 distinct values, got {n_distinct}"
    # and it ACTUALLY changed the weights (not a no-op).
    assert not torch.equal(q, w)
    assert (q - w).abs().max() > 0


def test_int4_grid_is_coarser_than_int5_is_coarser_than_int8():
    """Fewer nbits ⇒ fewer distinct quantized values (the monotone grid)."""
    torch.manual_seed(2)
    w = torch.randn(128, 128)
    d8 = int(torch.unique(_fake_quantize_n(w, NLEVELS_FOR_NBITS[8])).numel())
    d5 = int(torch.unique(_fake_quantize_n(w, NLEVELS_FOR_NBITS[5])).numel())
    d4 = int(torch.unique(_fake_quantize_n(w, NLEVELS_FOR_NBITS[4])).numel())
    assert d4 < d5 < d8, f"expected d4<d5<d8, got {d4},{d5},{d8}"


def test_int5_quant_error_exceeds_int8_quant_error():
    """The coarser grid has STRICTLY larger quant error (it is doing real work)."""
    torch.manual_seed(3)
    w = torch.randn(256, 64)
    e8 = (_fake_quantize_n(w, NLEVELS_FOR_NBITS[8]) - w).abs().mean()
    e5 = (_fake_quantize_n(w, NLEVELS_FOR_NBITS[5]) - w).abs().mean()
    assert e5 > e8 > 0


# ── STE gradient flows (the finetune lever) ──────────────────────────────────


def test_ste_gradient_is_identity_through_int5_quant():
    """Forward = int5 quant, backward = identity (the STE the finetune needs)."""
    torch.manual_seed(4)
    w = torch.randn(32, 16, requires_grad=True)
    q = _fake_quantize_n(w, NLEVELS_FOR_NBITS[5])
    q.sum().backward()
    assert w.grad is not None
    assert torch.allclose(w.grad, torch.ones_like(w.grad))


def test_apply_frontier_qat_keeps_grad_flowing_to_protected_and_coarse():
    """After apply, a forward through the quantized weights yields a grad on EVERY
    quantized weight (STE) — both protected (int8) and coarse (int5)."""
    dec = _tiny_decoder()
    cfg = FrontierQATConfig(mode="score_aware", low_nbits=5, high_nbits=8)
    nb = per_tensor_nbits_for_decoder(dec, cfg)
    originals = apply_frontier_qat(dec, nb)
    x = torch.randn(2, 4, 8, 8)
    # a real forward through the (now-quantized) conv weights.
    y = dec.stem(x)
    for b in dec.blocks:
        y = b(y)
    y = dec.rgb_1(y)
    y.sum().backward()
    assert dec.stem.weight.grad is not None  # coarse (int5) tensor still gets a grad
    assert dec.blocks[0].weight.grad is not None  # protected (int8) tensor too
    restore_frontier_qat(dec, originals)


# ── default-preserving: n_levels=127 == vendored int8 ────────────────────────


def test_high_nbits_8_is_bit_identical_to_codec_int8_grid():
    """A high_nbits=8 tensor is BIT-IDENTICAL to the codec's int8 intn_qdq grid
    (so a protected tensor re-encodes losslessly through the codec)."""
    torch.manual_seed(5)
    w = torch.randn(48, 48)
    # _fake_quantize_n at 127 levels == the codec int8 grid (both symmetric, n_quant=127).
    ste = _fake_quantize_n(w, 127)
    codec = intn_qdq(w, 8)
    assert torch.allclose(ste, codec, atol=1e-6), "n_levels=127 must match codec int8"


# ── allocation: uniform vs score_aware ───────────────────────────────────────


def test_uniform_mode_puts_every_weight_on_low_grid():
    dec = _tiny_decoder()
    cfg = FrontierQATConfig(mode="uniform", low_nbits=5)
    nb = per_tensor_nbits_for_decoder(dec, cfg)
    assert nb  # non-empty
    assert all(v == 5 for v in nb.values()), nb


def test_score_aware_protects_heads_and_protect_list_at_high_nbits():
    dec = _tiny_decoder()
    cfg = FrontierQATConfig(
        mode="score_aware",
        low_nbits=5,
        high_nbits=8,
        head_prefixes=("rgb_1.", "rgb_0."),
        protect_tensors=("blocks.0.weight", "blocks.5.weight"),
    )
    nb = per_tensor_nbits_for_decoder(dec, cfg)
    assert nb["rgb_1"] == 8  # head prefix → protected (matched via rgb_1.weight)
    assert nb["blocks.0"] == 8  # in protect list → protected
    assert nb["blocks.5"] == 8  # in protect list → protected
    assert nb["stem"] == 5  # blind → coarse
    assert nb["blocks.1"] == 5  # not protected → coarse


def test_score_aware_differs_from_uniform_allocation():
    """The two modes produce DIFFERENT per-tensor grids (not the same constant)."""
    dec = _tiny_decoder()
    uni = per_tensor_nbits_for_decoder(dec, FrontierQATConfig(mode="uniform", low_nbits=5))
    sa = per_tensor_nbits_for_decoder(
        dec,
        FrontierQATConfig(
            mode="score_aware",
            low_nbits=5,
            high_nbits=8,
            head_prefixes=("rgb_1.",),
            protect_tensors=("blocks.0.weight",),
        ),
    )
    assert uni != sa


# ── hard-quantize export matches the trained grid ────────────────────────────


def test_hard_quantize_matches_intn_qdq_per_tensor():
    """The export hard-quantize snaps each weight to EXACTLY its per-tensor codec grid."""
    dec = _tiny_decoder()
    cfg = FrontierQATConfig(
        mode="score_aware",
        low_nbits=5,
        high_nbits=8,
        head_prefixes=("rgb_1.",),
        protect_tensors=("blocks.0.weight",),
    )
    nb = per_tensor_nbits_for_decoder(dec, cfg)
    hard = hard_quantize_state_dict_to_nbits(dec, nb)
    # blocks.0 protected at int8; stem coarse at int5 — check both snap to their grid.
    assert torch.allclose(hard["blocks.0.weight"], intn_qdq(dec.blocks[0].weight, 8))
    assert torch.allclose(hard["stem.weight"], intn_qdq(dec.stem.weight, 5))
    # a coarse tensor's export has <=31 distinct values (the int5 grid).
    assert int(torch.unique(hard["stem.weight"]).numel()) <= 31


# ── EMA tracks (the EMA non-negotiable) ──────────────────────────────────────


def test_ema_warmup_tracks_live_weights_early():
    """With warmup the EMA shadow MOVES toward the live weights on step 1 (not frozen)."""
    dec = _tiny_decoder()
    lat = torch.randn(4, 28)
    ema = EMAState(decay=0.999, warmup=True)
    ema.init_from(dec, lat)
    before = ema.shadow["stem.weight"].clone()
    # mutate the live weights, then update — shadow must move a non-trivial amount.
    with torch.no_grad():
        dec.stem.weight.add_(1.0)
    ema.update(dec, lat)
    after = ema.shadow["stem.weight"]
    moved = (after - before).abs().mean().item()
    assert moved > 0.05, f"warmup EMA should track early; moved only {moved}"


def test_ema_no_warmup_barely_moves_on_first_step():
    """Without warmup, decay=0.999 ⇒ the shadow barely moves on step 1 (the lag)."""
    dec = _tiny_decoder()
    lat = torch.randn(4, 28)
    ema = EMAState(decay=0.999, warmup=False)
    ema.init_from(dec, lat)
    before = ema.shadow["stem.weight"].clone()
    with torch.no_grad():
        dec.stem.weight.add_(1.0)
    ema.update(dec, lat)
    moved = (ema.shadow["stem.weight"] - before).abs().mean().item()
    assert moved < 0.01, f"no-warmup should lag; moved {moved}"


def test_config_rejects_bad_nbits_and_mode():
    import pytest

    with pytest.raises(FrontierInt5QATError):
        FrontierQATConfig(low_nbits=3)  # 3 not in grid
    with pytest.raises(FrontierInt5QATError):
        FrontierQATConfig(mode="bogus")
    with pytest.raises(FrontierInt5QATError):
        FrontierQATConfig(low_nbits=8, high_nbits=5)  # high < low
