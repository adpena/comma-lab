# SPDX-License-Identifier: MIT
"""NO-FAKE behavioral tests for the residual argmax-flip delta (Yousfi eureka).

These verify the delta is a REAL RGB perturbation that range-codes + numpy-inflates
bit-exactly, and that it touches ONLY the flip tube (not the whole frame, not a
label edit). Each test would FAIL if the function body were replaced by constants.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.residual_basis.residual_flip_delta import (
    FlipDeltaPlan,
    _dilate_bool,
    _quantize_values,
    _resize_mask_nearest,
    apply_flip_delta,
    build_appearance_flip_delta,
    byte_close_flip_delta,
    inflate_flip_delta,
)


def _toy(seed: int = 0):
    rng = np.random.default_rng(seed)
    hw = (96, 128)
    base = rng.uniform(0, 255, (*hw, 3))
    gt = np.clip(base + rng.uniform(-40, 40, (*hw, 3)), 0, 255)
    base_am = np.zeros((384, 512), np.int64)
    base_am[100:160, 200:320] = 1
    lstar = np.zeros((384, 512), np.int64)
    lstar[120:180, 220:340] = 1  # partial overlap -> a band of flips
    return base, gt, base_am, lstar, hw


def test_flip_delta_touches_only_flip_tube_not_whole_frame():
    base, gt, base_am, lstar, hw = _toy()
    plan = build_appearance_flip_delta(
        base, gt, base_am, lstar, pair_idx=0, l_inf_budget=64.0, dilate=0, strength=1.0
    )
    # delta exists (there ARE flips) but is sparse (< 40% of pixel-channels).
    assert plan.flat_idx.size > 0
    assert plan.flat_idx.size < hw[0] * hw[1] * 3 * 0.4
    applied = apply_flip_delta(base, plan)
    changed = np.abs(applied - base) > 0.5
    # exactly the kept positions changed; everything else is untouched.
    assert int(changed.sum()) == plan.flat_idx.size


def test_zero_flips_yields_empty_delta():
    base, gt, base_am, _lstar, _hw = _toy()
    plan = build_appearance_flip_delta(
        base, gt, base_am, base_am, pair_idx=0, l_inf_budget=64.0, dilate=0, strength=1.0
    )
    assert plan.flat_idx.size == 0
    assert plan.n_flips_before == 0
    applied = apply_flip_delta(base, plan)
    assert np.array_equal(applied, base)  # no-op when no flips


def test_delta_is_rgb_perturbation_not_a_label_edit():
    # The applied frame differs from base by a BOUNDED RGB amount (<= l_inf),
    # never by a class id. (A label-edit fake would leave RGB unchanged.)
    base, gt, base_am, lstar, _hw = _toy()
    li = 16.0
    plan = build_appearance_flip_delta(
        base, gt, base_am, lstar, pair_idx=0, l_inf_budget=li, dilate=0, strength=1.0
    )
    applied = apply_flip_delta(base, plan)
    diff = applied - base
    assert np.abs(diff).max() <= li + 1e-9  # RGB perturbation, L-inf bounded
    assert np.abs(diff).max() > 0.0  # it actually moved pixels (not a no-op)


def test_byte_close_inflate_is_bit_exact_parity():
    base, gt, base_am, lstar, _hw = _toy()
    li = 64.0
    plans = [
        build_appearance_flip_delta(base, gt, base_am, lstar, pair_idx=i,
                                    l_inf_budget=li, dilate=1, strength=1.0)
        for i in range(3)
    ]
    blob = byte_close_flip_delta(plans, l_inf_budget=li)
    assert blob[:4] == b"RFD1"
    plans2 = inflate_flip_delta(blob)
    assert len(plans2) == len(plans)
    for p, p2 in zip(plans, plans2, strict=True):
        # indices roundtrip EXACTLY
        assert np.array_equal(p.flat_idx, p2.flat_idx)
        # values roundtrip to the quantized grid bit-exactly
        qv, scale = _quantize_values(p.values, li)
        assert np.allclose(p2.values, qv.astype(np.float64) * scale, atol=1e-9)


def test_byte_close_charges_real_bytes_more_flips_costs_more():
    # A LARGER flip region (more disagreeing pixels) -> more kept entries ->
    # strictly more bytes (real cost, not free). We grow the flip count by
    # making L* disagree over a bigger area (same dilate, isolates count from
    # local density). A free/constant fake would not move the byte count.
    base, gt, base_am, _l0, _hw = _toy()
    small_lstar = np.zeros((384, 512), np.int64)
    small_lstar[120:140, 220:260] = 1  # small disagreement band
    big_lstar = np.zeros((384, 512), np.int64)
    big_lstar[120:300, 220:480] = 1  # much larger disagreement band
    small = build_appearance_flip_delta(base, gt, base_am, small_lstar, pair_idx=0,
                                        l_inf_budget=64.0, dilate=0, strength=1.0)
    big = build_appearance_flip_delta(base, gt, base_am, big_lstar, pair_idx=0,
                                      l_inf_budget=64.0, dilate=0, strength=1.0)
    assert big.flat_idx.size > small.flat_idx.size
    b_small = len(byte_close_flip_delta([small], l_inf_budget=64.0))
    b_big = len(byte_close_flip_delta([big], l_inf_budget=64.0))
    assert b_big > b_small


def test_empty_plan_byte_closes_and_inflates():
    empty = FlipDeltaPlan(
        pair_idx=0, witness_hw=(96, 128), flat_idx=np.zeros((0,), np.int64),
        values=np.zeros((0,), np.float64), n_flips_before=0, target_flip_pixels=0,
    )
    blob = byte_close_flip_delta([empty], l_inf_budget=64.0)
    plans2 = inflate_flip_delta(blob)
    assert plans2[0].flat_idx.size == 0


def test_dilate_bool_grows_mask():
    m = np.zeros((10, 10), bool)
    m[5, 5] = True
    d1 = _dilate_bool(m, 1)
    assert d1.sum() == 5  # plus-shaped 4-neighbourhood
    d2 = _dilate_bool(m, 2)
    assert d2.sum() > d1.sum()


def test_resize_mask_nearest_preserves_boolness_and_shape():
    m = np.zeros((384, 512), bool)
    m[100:200, 100:200] = True
    r = _resize_mask_nearest(m, 96, 128)
    assert r.shape == (96, 128)
    assert r.dtype == bool
    assert r.any()


def test_apply_clips_to_0_255():
    base = np.full((4, 4, 3), 250.0)
    plan = FlipDeltaPlan(
        pair_idx=0, witness_hw=(4, 4),
        flat_idx=np.array([0], np.int64), values=np.array([50.0]),
        n_flips_before=1, target_flip_pixels=1,
    )
    applied = apply_flip_delta(base, plan)
    assert applied.max() <= 255.0  # 250 + 50 clipped to 255


def test_apply_rejects_wrong_witness_shape():
    plan = FlipDeltaPlan(
        pair_idx=0, witness_hw=(96, 128), flat_idx=np.zeros((0,), np.int64),
        values=np.zeros((0,), np.float64), n_flips_before=0, target_flip_pixels=0,
    )
    with pytest.raises(ValueError):
        apply_flip_delta(np.zeros((10, 10, 3)), plan)
