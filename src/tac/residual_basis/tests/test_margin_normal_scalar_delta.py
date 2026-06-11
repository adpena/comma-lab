# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the cheap margin-normal scalar flip-delta codec."""

from __future__ import annotations

import numpy as np

from tac.residual_basis.margin_normal_scalar_delta import (
    apply_margin_normal_scalar,
    build_margin_normal_scalar_plan,
    byte_close_margin_normal_scalar,
    inflate_margin_normal_scalar,
)


def _toy_correction(h=12, w=16, seed=0):
    rng = np.random.default_rng(seed)
    corr = np.zeros((h, w, 3), dtype=np.float64)
    # plant 10 support pixels with a clear dominant channel
    for _ in range(10):
        y, x = rng.integers(0, h), rng.integers(0, w)
        c = rng.integers(0, 3)
        corr[y, x, c] = rng.choice([-30.0, 30.0, 20.0])
    return corr


def test_plan_keeps_only_dominant_channel():
    corr = np.zeros((4, 4, 3))
    corr[1, 1] = [10.0, 40.0, -5.0]  # dominant = channel 1
    plan = build_margin_normal_scalar_plan(corr, pair_idx=0, witness_hw=(4, 4), l_inf_budget=64.0)
    assert plan.n_support_pixels == 1
    assert plan.chan[0] == 1
    assert abs(plan.mag[0] - 40.0) < 1e-9


def test_apply_changes_only_dominant_channel():
    corr = np.zeros((4, 4, 3))
    corr[2, 3] = [5.0, -50.0, 5.0]  # dominant = channel 1, sign negative
    plan = build_margin_normal_scalar_plan(corr, pair_idx=0, witness_hw=(4, 4), l_inf_budget=64.0)
    base = np.full((4, 4, 3), 128.0)
    out = apply_margin_normal_scalar(base, plan)
    # only (2,3,channel1) changed
    diff = out - base
    nz = np.abs(diff) > 1e-9
    assert nz.sum() == 1
    assert nz[2, 3, 1]
    assert abs(out[2, 3, 1] - 78.0) < 1.0  # 128 - 50 (mod int8 quant ~negligible here)


def test_apply_clips_to_0_255():
    corr = np.zeros((2, 2, 3))
    corr[0, 0] = [0.0, 0.0, 200.0]
    plan = build_margin_normal_scalar_plan(corr, pair_idx=0, witness_hw=(2, 2), l_inf_budget=255.0)
    base = np.full((2, 2, 3), 250.0)
    out = apply_margin_normal_scalar(base, plan)
    assert out[0, 0, 2] == 255.0  # clipped


def test_byteclose_roundtrip_parity():
    plans = [
        build_margin_normal_scalar_plan(_toy_correction(seed=s), pair_idx=s, witness_hw=(12, 16), l_inf_budget=64.0)
        for s in range(3)
    ]
    blob = byte_close_margin_normal_scalar(plans, l_inf_budget=64.0)
    dec = inflate_margin_normal_scalar(blob)
    assert len(dec) == 3
    for a, b in zip(plans, dec):
        assert np.array_equal(a.flat_px, b.flat_px)
        assert np.array_equal(a.chan, b.chan)
        # magnitudes match to int8 quant step (64/127)
        assert np.allclose(a.mag, b.mag, atol=64.0 / 127.0 + 1e-6)


def test_byteclose_apply_parity_on_witness():
    """The byte-closed-then-decoded plan reconstructs the same perturbed witness
    (within int8 quant) as the in-memory plan -- the delta bytes are REAL."""
    base = np.full((12, 16, 3), 100.0)
    corr = _toy_correction(seed=7)
    plan = build_margin_normal_scalar_plan(corr, pair_idx=0, witness_hw=(12, 16), l_inf_budget=64.0)
    blob = byte_close_margin_normal_scalar([plan], l_inf_budget=64.0)
    dec = inflate_margin_normal_scalar(blob)[0]
    out_mem = apply_margin_normal_scalar(base, plan)
    out_bc = apply_margin_normal_scalar(base, dec)
    assert np.max(np.abs(out_mem - out_bc)) <= 64.0 / 127.0 + 1e-6


def test_bytes_per_flip_far_under_3channel_rgb():
    """The cheap scalar codec must be FAR under the 3-channel int8 RGB cost (~3 B/flip
    of raw value bytes alone, ~187 B/flip measured with dilation+tube in v2).

    Even on WORST-CASE uniformly-scattered support (max index-gap entropy) the cheap
    codec stays under 8 B/flip; on REALISTIC clustered boundary support (small gaps)
    it drops far lower (see test_bytes_per_flip_clustered_is_lever_d)."""
    rng = np.random.default_rng(3)
    h, w = 192, 256
    corr = np.zeros((h, w, 3))
    idx = rng.choice(h * w, size=500, replace=False)
    for fi in idx:
        y, x = fi // w, fi % w
        corr[y, x, rng.integers(0, 3)] = rng.choice([-20.0, 20.0])
    plan = build_margin_normal_scalar_plan(corr, pair_idx=0, witness_hw=(h, w), l_inf_budget=32.0)
    blob = byte_close_margin_normal_scalar([plan], l_inf_budget=32.0)
    bpf = len(blob) / plan.n_support_pixels
    assert plan.n_support_pixels == 500
    assert bpf < 8.0, f"cheap codec worst-case should be <8 B/flip, got {bpf:.2f}"


def test_bytes_per_flip_clustered_amortized_is_lever_d():
    """REALISTIC case: flip pixels cluster at SegNet class boundaries (small index
    gaps) AND the JSON header (freq tables) amortizes across many pairs sharing one
    blob.  The cheap scalar codec then lands in lever-D territory (< 2 B/flip).

    The single-pair header overhead is real (~tens of B amortized over k); the
    contest packs all pairs into ONE blob, so the per-flip cost is the amortized one."""
    h, w = 192, 256
    plans = []
    total_k = 0
    for pi in range(60):  # many pairs share the header (as the real packing does)
        corr = np.zeros((h, w, 3))
        k = 0
        for row in range(40, 60):
            for col in range(20, 60):
                corr[row, col, (row + col) % 3] = 24.0
                k += 1
        plans.append(
            build_margin_normal_scalar_plan(corr, pair_idx=pi, witness_hw=(h, w), l_inf_budget=32.0)
        )
        total_k += k
    blob = byte_close_margin_normal_scalar(plans, l_inf_budget=32.0)
    bpf = len(blob) / total_k
    assert bpf < 2.0, f"clustered+amortized should be <2 B/flip, got {bpf:.2f}"


def test_empty_plan_roundtrips():
    plan = build_margin_normal_scalar_plan(np.zeros((4, 4, 3)), pair_idx=0, witness_hw=(4, 4), l_inf_budget=64.0)
    assert plan.n_support_pixels == 0
    blob = byte_close_margin_normal_scalar([plan], l_inf_budget=64.0)
    dec = inflate_margin_normal_scalar(blob)
    assert dec[0].n_support_pixels == 0
