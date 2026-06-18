# SPDX-License-Identifier: MIT
"""NO-FAKE tests for tac.post_hoc_weight_shrink — behavior, not constants.

Every test asserts the qdq round-trip ACTUALLY changes the weights to the claimed grid
(Catalog "Tests-verify-constants-not-behavior" forbidden class): if a function body were
replaced by ``return t.clone()`` these tests would FAIL.
"""
from __future__ import annotations

import pytest
import torch

from tac.post_hoc_weight_shrink import (
    E2M1_ABS,
    e2m1_qdq,
    intn_qdq,
    requantize_decoder_state_dict,
)


def _distinct_levels(t: torch.Tensor) -> int:
    return int(torch.unique(t).numel())


# ── intn_qdq ────────────────────────────────────────────────────────────────
def test_intn_qdq_int8_matches_shipped_codec_grid():
    """nbits=8 reproduces the shipped codec's symmetric int8 grid (n_quant=127)."""
    torch.manual_seed(0)
    t = torch.randn(64, 64) * 3.0
    q = intn_qdq(t, 8)
    s = t.abs().max() / 127.0
    expect = (t / s).round().clamp_(-127, 127) * s
    assert torch.allclose(q, expect, atol=1e-6)


def test_intn_qdq_fewer_bits_have_fewer_distinct_levels():
    """The grid sparsens monotonically as nbits drops — the core rate lever."""
    torch.manual_seed(1)
    t = torch.randn(128, 128) * 2.0
    levels = {nb: _distinct_levels(intn_qdq(t, nb)) for nb in (4, 5, 6, 8)}
    assert levels[4] < levels[5] < levels[6] <= levels[8]
    # int4 grid has at most 2*7+1 = 15 distinct dequantized values.
    assert levels[4] <= 15


def test_intn_qdq_actually_changes_weights_at_low_bits():
    """A low-bit qdq is NOT a passthrough (forbidden returns-input fake)."""
    torch.manual_seed(2)
    t = torch.randn(32, 32)
    q4 = intn_qdq(t, 4)
    assert not torch.allclose(q4, t)
    # error grows as bits drop.
    err4 = (q4 - t).abs().mean()
    err8 = (intn_qdq(t, 8) - t).abs().mean()
    assert err4 > err8


def test_intn_qdq_max_abs_bounded_by_original():
    """Symmetric grid never exceeds the original max magnitude (scale = max/qmax)."""
    torch.manual_seed(3)
    t = torch.randn(40, 40) * 5.0
    for nb in (4, 5, 6, 8):
        q = intn_qdq(t, nb)
        assert q.abs().max() <= t.abs().max() + 1e-5


def test_intn_qdq_zero_tensor_unchanged():
    t = torch.zeros(8, 8)
    assert torch.equal(intn_qdq(t, 4), t)


def test_intn_qdq_preserves_shape_dtype_device():
    t = torch.randn(3, 4, 5)
    q = intn_qdq(t, 5)
    assert q.shape == t.shape and q.dtype == t.dtype and q.device == t.device


def test_intn_qdq_rejects_nbits_below_2():
    with pytest.raises(ValueError):
        intn_qdq(torch.randn(4, 4), 1)


def test_intn_qdq_higher_bits_lower_error_monotone():
    torch.manual_seed(4)
    t = torch.randn(64, 64)
    errs = [(intn_qdq(t, nb) - t).abs().mean().item() for nb in (3, 4, 5, 6, 7, 8)]
    assert all(errs[i] >= errs[i + 1] - 1e-9 for i in range(len(errs) - 1))


# ── e2m1_qdq ────────────────────────────────────────────────────────────────
def test_e2m1_qdq_values_lie_on_the_e2m1_grid_per_channel():
    """Every dequantized value, divided by its channel scale, is an E2M1 magnitude."""
    torch.manual_seed(5)
    t = torch.randn(8, 16) * 2.0
    q = e2m1_qdq(t, per_channel=True)
    red = tuple(range(1, t.dim()))
    s = (t.abs().amax(dim=red, keepdim=True) / 6.0).clamp_min(1e-12)
    grid = torch.tensor(E2M1_ABS)
    normed = (q / s).abs()
    # each normed value must equal one of the 8 grid magnitudes.
    nearest = (normed.unsqueeze(-1) - grid).abs().min(dim=-1).values
    assert float(nearest.max()) < 1e-5


def test_e2m1_qdq_at_most_15_distinct_normalized_levels():
    torch.manual_seed(6)
    t = torch.randn(1, 4096)  # single channel so one scale
    q = e2m1_qdq(t, per_channel=True)
    s = (t.abs().max() / 6.0).clamp_min(1e-12)
    normed = torch.unique((q / s).round(decimals=4))
    assert normed.numel() <= 15  # 8 magnitudes * sign - shared zero


def test_e2m1_qdq_actually_changes_weights():
    torch.manual_seed(7)
    t = torch.randn(16, 16)
    assert not torch.allclose(e2m1_qdq(t), t)


def test_e2m1_qdq_per_tensor_vs_per_channel_differ():
    torch.manual_seed(8)
    # channels with very different scales — per-channel should fit each better.
    t = torch.cat([torch.randn(1, 32) * 0.1, torch.randn(1, 32) * 10.0], dim=0)
    q_pc = e2m1_qdq(t, per_channel=True)
    q_pt = e2m1_qdq(t, per_channel=False)
    assert not torch.allclose(q_pc, q_pt)
    # per-channel error <= per-tensor error (tighter scale per channel).
    assert (q_pc - t).abs().mean() <= (q_pt - t).abs().mean() + 1e-6


def test_e2m1_qdq_zero_tensor_unchanged():
    t = torch.zeros(4, 4)
    assert torch.allclose(e2m1_qdq(t), t)


# ── requantize_decoder_state_dict ───────────────────────────────────────────
def _toy_sd():
    torch.manual_seed(9)
    return {
        "blocks.0.weight": torch.randn(20, 28, 1, 1),
        "blocks.0.bias": torch.randn(20),  # 1-D: must stay fp32
        "rgb_1.weight": torch.randn(3, 20, 3, 3),  # head
        "skips.2.weight": torch.randn(20, 20, 1, 1),  # head
        "refine.weight": torch.randn(20, 20, 3, 3),  # head
    }


def test_requant_fp32_is_passthrough():
    sd = _toy_sd()
    out = requantize_decoder_state_dict(sd, "fp32")
    for k in sd:
        assert torch.allclose(out[k], sd[k])
    assert out is not sd  # new dict, not aliased


def test_requant_leaves_biases_untouched():
    """1-D tensors (biases) are never quantized regardless of mode."""
    sd = _toy_sd()
    for mode in ("int4", "int6", "fp4_all", "fp4_mixed"):
        out = requantize_decoder_state_dict(sd, mode)
        assert torch.allclose(out["blocks.0.bias"], sd["blocks.0.bias"])


def test_requant_intn_quantizes_all_weight_tensors():
    sd = _toy_sd()
    out = requantize_decoder_state_dict(sd, "int4")
    for k, v in sd.items():
        if v.dim() >= 2:
            assert not torch.allclose(out[k], v), f"{k} should be int4-quantized"
            assert _distinct_levels(out[k]) <= 15


def test_requant_fp4_mixed_keeps_heads_int8_interior_fp4():
    """fp4_mixed: heads on int8 grid, interior on E2M1 grid — structurally distinct."""
    sd = _toy_sd()
    out = requantize_decoder_state_dict(sd, "fp4_mixed")
    head = "rgb_1.weight"
    interior = "blocks.0.weight"
    # head should equal the int8 qdq of itself.
    assert torch.allclose(out[head], intn_qdq(sd[head], 8))
    # interior should equal the E2M1 qdq of itself, and NOT its int8 qdq.
    assert torch.allclose(out[interior], e2m1_qdq(sd[interior]))
    assert not torch.allclose(out[interior], intn_qdq(sd[interior], 8))


def test_requant_fp4_all_puts_every_weight_on_e2m1():
    sd = _toy_sd()
    out = requantize_decoder_state_dict(sd, "fp4_all")
    for k, v in sd.items():
        if v.dim() >= 2:
            assert torch.allclose(out[k], e2m1_qdq(v))


def test_requant_rejects_unknown_mode():
    with pytest.raises(ValueError):
        requantize_decoder_state_dict(_toy_sd(), "int9")


def test_requant_does_not_mutate_input():
    sd = _toy_sd()
    snap = {k: v.clone() for k, v in sd.items()}
    _ = requantize_decoder_state_dict(sd, "int4")
    for k in sd:
        assert torch.allclose(sd[k], snap[k])


def test_requant_int8_mode_roundtrips_through_codec_grid():
    """int8 mode reproduces exactly the shipped codec's per-tensor grid for weights."""
    sd = _toy_sd()
    out = requantize_decoder_state_dict(sd, "int8")
    for k, v in sd.items():
        if v.dim() >= 2:
            assert torch.allclose(out[k], intn_qdq(v, 8))
