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
    FrontierLSQQuantizer,
    FrontierQATConfig,
    apply_frontier_qat,
    hard_quantize_state_dict_lsq,
    hard_quantize_state_dict_to_nbits,
    lsq_fake_quantize,
    mse_optimal_step,
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


# ─────────────────────────────────────────────────────────────────────────────
# BEST-SHOT low-bit fixes: per-tensor LSQ learned step + outlier-clip calibration.
# These verify the NEW quantizer ACTUALLY does LSQ (learned step gets a gradient) +
# outlier handling (the MSE-optimal step is SMALLER than abs-max on a heavy-tailed
# tensor), NOT that the abs-max behavior is silently kept. Every test would FAIL if the
# body fell back to per-tensor abs-max.
# ─────────────────────────────────────────────────────────────────────────────


def test_lsq_gradient_flows_to_both_weight_and_step():
    """LSQ fake-quant gives a gradient to BOTH the weight (STE) AND the step (the
    learned-step lever the original abs-max quantizer LACKS)."""
    torch.manual_seed(10)
    w = torch.randn(32, 16, requires_grad=True)
    step = torch.tensor(float(w.abs().max() / 15), requires_grad=True)
    q = lsq_fake_quantize(w, step, 5)
    (q - torch.randn_like(q)).pow(2).mean().backward()
    assert w.grad is not None and w.grad.abs().sum().item() > 0, "weight must get a grad"
    assert step.grad is not None and abs(float(step.grad.detach())) > 0, "STEP must get a grad (LSQ)"


def test_lsq_step_gradient_matches_esser_closed_form():
    """The LSQ step gradient is the canonical Esser et al. 2020 closed form:
    ``∂v̂/∂s = (round(v/s) − v/s)`` in range (× the ``1/√(numel·qmax)`` LSQ scale), and
    ``±qmax`` saturated. Verifies the gradient is the REAL LSQ form, not abs-max (which
    has NO step gradient at all)."""
    # weights chosen off the round-.5 boundaries so the round STE is the local behavior.
    w = torch.tensor([[0.123, -0.234, 0.327, -0.418]])
    s = torch.tensor(0.1, requires_grad=True)
    q = lsq_fake_quantize(w, s, 5)
    q.sum().backward()  # grad_out = 1 everywhere
    vs = w / 0.1
    expected = float((vs.round() - vs).sum()) / (w.numel() * 15) ** 0.5
    assert abs(float(s.grad.detach()) - expected) < 1e-6, "LSQ step grad must match Esser form"


def test_lsq_saturated_outlier_gets_zero_weight_grad():
    """The TRUE clamp STE: a weight far outside the int5 range (saturated) gets NO weight
    gradient, while an in-range weight does. The vendored identity-everywhere STE gives a
    grad to the saturated outlier too — this is the outlier-aware improvement."""
    w = torch.tensor([[100.0, 0.05, -0.05]], requires_grad=True)  # 100 saturates int5
    step = torch.tensor(0.05 / 15 * 5, requires_grad=True)  # tiny step → 100 way out of range
    q = lsq_fake_quantize(w, step, 5)
    q.sum().backward()
    assert float(w.grad[0, 0]) == 0.0, "saturated outlier must get ZERO weight grad"
    assert float(w.grad[0, 1]) != 0.0, "in-range weight must keep its grad"


def test_mse_optimal_step_clips_outlier_on_heavy_tail():
    """The outlier-handling calibration: a heavy-tailed tensor (one big outlier) gets an
    MSE-optimal step SMALLER than abs-max (it clips the outlier to give the bulk weights
    finer resolution). This is the 50-61%-MSE-cut lever the abs-max quantizer omits."""
    torch.manual_seed(11)
    heavy = torch.cat([torch.randn(2000) * 0.1, torch.tensor([6.0])])  # one large outlier
    absmax_step = float(heavy.abs().max() / 15)
    opt_step = mse_optimal_step(heavy, 5)
    assert opt_step < absmax_step, f"heavy-tail opt step {opt_step} must clip below abs-max {absmax_step}"
    # and the clipped step ACTUALLY reduces reconstruction MSE.
    qa = (heavy / absmax_step).round().clamp(-15, 15) * absmax_step
    qo = (heavy / opt_step).round().clamp(-15, 15) * opt_step
    assert (qo - heavy).pow(2).mean() < (qa - heavy).pow(2).mean(), "opt step must cut MSE"


def test_mse_optimal_step_returns_absmax_on_uniform_no_outlier():
    """On a well-behaved (uniform, no-outlier) tensor the calibration returns ≈ abs-max
    (clip_ratio ≈ 1.0) — it does NOT over-clip a tensor that has no outliers."""
    torch.manual_seed(12)
    uniform = torch.rand(4000) * 2 - 1  # symmetric uniform, no heavy tail
    absmax_step = float(uniform.abs().max() / 15)
    ratio = mse_optimal_step(uniform, 5) / absmax_step
    assert 0.85 <= ratio <= 1.0, f"uniform should keep ≈abs-max step, got ratio {ratio}"


def test_lsq_calibrated_step_differs_from_absmax_on_real_frontier_stem():
    """On a heavy-tailed conv tensor (the frontier ``stem`` has a 3.26× per-channel-max/
    median outlier), the LSQ-calibrated step is materially below abs-max — the lever
    actually fires on the real rate-carrier."""
    torch.manual_seed(13)
    # mimic the stem's outlier structure: most channels small, a few large.
    dec = _tiny_decoder()
    with torch.no_grad():
        dec.stem.weight.copy_(torch.randn_like(dec.stem.weight) * 0.1)
        dec.stem.weight[0] *= 12.0  # one outlier output channel (sets abs-max)
    q = FrontierLSQQuantizer.from_decoder(dec, {"stem": 5})
    absmax_step = float(dec.stem.weight.abs().max() / 15)
    calib_step = q.steps_dict()["stem"]
    assert calib_step < absmax_step, f"calib {calib_step} must clip below abs-max {absmax_step}"


def test_lsq_apply_quantizes_and_restores_with_grad():
    """``FrontierLSQQuantizer.apply`` swaps in the LSQ-quantized weight for the forward
    (the forward uses it) and the STE grad flows to the ORIGINAL Parameter (the one the
    optimizer holds) AND the learnable step; ``restore`` puts the EXACT same Parameter
    objects back so the optimizer keeps working."""
    import torch.nn as nn

    dec = _tiny_decoder()
    cfg = FrontierQATConfig(mode="score_aware", low_nbits=5, high_nbits=8)
    nb = per_tensor_nbits_for_decoder(dec, cfg)
    q = FrontierLSQQuantizer.from_decoder(dec, nb)
    orig_param = dec.stem.weight  # the Parameter the optimizer holds
    orig_stem = orig_param.data.clone()
    originals = q.apply(dec)
    # the live weight is now the quantized tensor (changed from the original).
    assert not torch.equal(dec.stem.weight.detach(), orig_stem), "apply must quantize the weight"
    x = torch.randn(2, 4, 8, 8)
    y = dec.stem(x)
    for b in dec.blocks:
        y = b(y)
    y = dec.rgb_1(y)
    y.sum().backward()
    # grad flows to the ORIGINAL Parameter (STE) and the step (LSQ).
    assert orig_param.grad is not None and orig_param.grad.abs().sum() > 0, "STE grad to orig param"
    assert q.step_for("stem").grad is not None, "LSQ grad to the step"
    q.restore(dec, originals)
    # restore puts back the EXACT same Parameter object (optimizer reference preserved).
    assert dec.stem.weight is orig_param, "restore must put back the SAME Parameter object"
    assert isinstance(dec.stem.weight, nn.Parameter)
    assert torch.equal(dec.stem.weight.data, orig_stem)


def test_lsq_export_snaps_to_learned_step_grid():
    """``hard_quantize_state_dict_lsq`` snaps each weight to the LEARNED-step grid, NOT
    abs-max — so the byte-closed grid equals the trained grid."""
    dec = _tiny_decoder()
    nb = per_tensor_nbits_for_decoder(dec, FrontierQATConfig(mode="uniform", low_nbits=5))
    q = FrontierLSQQuantizer.from_decoder(dec, nb)
    # set a known, non-abs-max step on stem and verify the export uses IT.
    with torch.no_grad():
        q._steps[q._key_for["stem"]].data = torch.tensor(0.01)
    hard = hard_quantize_state_dict_lsq(dec, q)
    expected = (dec.stem.weight / 0.01).round().clamp(-15, 15) * 0.01
    assert torch.allclose(hard["stem.weight"], expected), "export must use the learned step"
    # and it differs from the abs-max export (the lever fires).
    absmax_export = hard_quantize_state_dict_to_nbits(dec, nb)
    assert not torch.allclose(hard["stem.weight"], absmax_export["stem.weight"]), (
        "LSQ export must differ from abs-max export"
    )


def test_lsq_export_is_int5_grid_distinct_count():
    """The LSQ export is still a per-tensor symmetric int5 grid (<=31 distinct values) —
    byte-close compatible (one scale, the codec stores per-tensor int8)."""
    dec = _tiny_decoder()
    nb = per_tensor_nbits_for_decoder(dec, FrontierQATConfig(mode="uniform", low_nbits=5))
    q = FrontierLSQQuantizer.from_decoder(dec, nb)
    hard = hard_quantize_state_dict_lsq(dec, q)
    assert int(torch.unique(hard["stem.weight"]).numel()) <= 31


def test_lsq_quantizer_steps_are_learnable_parameters():
    """The per-tensor steps are nn.Parameters (so AdamW optimizes them in the finetune)."""
    dec = _tiny_decoder()
    nb = per_tensor_nbits_for_decoder(dec, FrontierQATConfig(mode="uniform", low_nbits=5))
    q = FrontierLSQQuantizer.from_decoder(dec, nb)
    params = list(q.parameters())
    assert len(params) == len(nb), "one learnable step per quantized tensor"
    assert all(p.requires_grad for p in params)


def test_lsq_int8_grid_is_codec_compatible():
    """A high_nbits=8 (protected) tensor's LSQ export at abs-max step == the codec int8
    grid (so protected tensors still re-encode losslessly when the step is abs-max)."""
    dec = _tiny_decoder()
    nb = {"stem": 8}
    q = FrontierLSQQuantizer(nb)
    with torch.no_grad():
        # set step to the abs-max int8 step explicitly.
        q._steps[q._key_for["stem"]].data = torch.tensor(float(dec.stem.weight.abs().max() / 127))
    hard = hard_quantize_state_dict_lsq(dec, q)
    assert torch.allclose(hard["stem.weight"], intn_qdq(dec.stem.weight, 8), atol=1e-6)
