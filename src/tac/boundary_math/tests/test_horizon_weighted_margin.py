# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the #169 horizon-weighted margin lever (reference twin + DSL leg + equation).

The defining contracts (asserted on REAL behavior, never constants):
  * BYTE-IDENTITY when OFF: weight==0 ⇒ the loss is EXACTLY 0.0 AND no mask is built (the trainer's
    default-OFF path, gated on hz_w>0, is byte-identical by construction — these pin the runtime twin).
  * REAL up-weighting when ON: ONLY pixels in (horizon rows) AND (GT margin ∈ [lo,hi)) with a margin
    DEFICIT below m_target contribute; a no-op / wrong-band would FAIL (the constants-not-behavior guard).
  * SATISFICING: m_wit >= m_target on the band ⇒ term 0 (relu saturates — no over-push into label-noise).
  * HALF-OPEN band: margin==lo IN, margin==hi OUT (matches the trainer's ``>= lo & < hi``).
  * EMPTY mask ⇒ 0.0 (the +1e-6 guard, never a /0). Row clamp degrades an over-wide band, never crashes.
  * Fail-closed: shape mismatches / bad args raise (never silently broadcast the wrong grid).
  * DSL leg: the HorizonWeightedMargin factory emits ONLY real trainer flags, defaults off, is composable,
    and threads start_epoch (the stage-boundary gate). Equation leg: registers + is queryable.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.horizon_weighted_margin import (
    horizon_margin_hinge_term,
    horizon_stratified_mask,
    horizon_weighted_margin_loss,
)


def _margin_grid(h=384, w=64, value=0.4):
    """A constant GT-margin field (H,W) — put every pixel in a chosen margin band by default."""
    return np.full((h, w), float(value), np.float32)


def _mwit(h=384, w=64, value=0.0):
    """A constant witness margin field (H,W); default 0 ⇒ full deficit below any positive target."""
    return np.full((h, w), float(value), np.float32)


# ---- stratified mask --------------------------------------------------------
def test_mask_is_rows_and_margin_band_intersection():
    """mask == 1 ONLY where row∈[rlo,rhi) AND margin∈[lo,hi); real intersection, not a constant."""
    h, w = 384, 8
    mg = np.zeros((h, w), np.float32)
    mg[100:110, :] = 0.4        # in-band margin, in horizon rows [96,288) -> ON
    mg[300:310, :] = 0.4        # in-band margin but rows outside [96,288) -> OFF (row gate)
    mg[120:130, :] = 0.9        # in horizon rows but margin > hi -> OFF (margin gate)
    m = horizon_stratified_mask(mg, row_lo=96, row_hi=288, margin_lo=0.3, margin_hi=0.5)
    assert m.dtype == np.float32
    assert m[100:110, :].sum() == 10 * w      # the intersection region is ON
    assert m[300:310, :].sum() == 0.0         # excluded by the row gate
    assert m[120:130, :].sum() == 0.0         # excluded by the margin gate


def test_band_is_half_open_lo_in_hi_out():
    """margin==lo is IN (reducible edge), margin==hi is OUT (matches trainer ``>= lo & < hi``)."""
    h, w = 384, 4
    mg = np.zeros((h, w), np.float32)
    mg[100, :] = 0.3            # == lo -> IN
    mg[101, :] = 0.5            # == hi -> OUT
    m = horizon_stratified_mask(mg, margin_lo=0.3, margin_hi=0.5)
    assert m[100, :].sum() == w        # lo included
    assert m[101, :].sum() == 0.0      # hi excluded


def test_row_clamp_degrades_overwide_band_no_crash():
    """An over-wide row band clamps to the real grid H (never crashes / empties by row alone)."""
    h, w = 200, 4
    mg = _margin_grid(h, w, 0.4)                     # every pixel in the margin band
    m = horizon_stratified_mask(mg, row_lo=0, row_hi=100_000, margin_lo=0.3, margin_hi=0.5)
    assert m.sum() == h * w                          # clamped to the full column, all in-band


def test_mask_fail_closed_on_bad_args():
    with pytest.raises(ValueError):
        horizon_stratified_mask(np.zeros((10, 10), np.float32), margin_lo=0.5, margin_hi=0.3)
    with pytest.raises(ValueError):
        horizon_stratified_mask(np.zeros((10, 10), np.float32), row_lo=200, row_hi=100)
    with pytest.raises(ValueError):
        horizon_stratified_mask(np.zeros((10,), np.float32))          # not 2-D


# ---- hinge term -------------------------------------------------------------
def test_hinge_is_mean_deficit_over_band():
    """term == mean over band px of relu(target - m_wit); computed, not asserted-constant."""
    h, w = 384, 4
    mask = np.zeros((h, w), np.float32)
    mask[100:110, :] = 1.0                           # 40 band pixels
    mw = _mwit(h, w, 0.1)                             # deficit = 0.5 - 0.1 = 0.4 everywhere in band
    term = horizon_margin_hinge_term(mw, mask, m_target=0.5)
    # sum = 40*0.4 = 16; denom = 40 + 1e-6 -> ~0.4
    assert term == pytest.approx(0.4, abs=1e-4)


def test_satisficing_zero_above_target():
    """m_wit >= m_target on the band ⇒ relu saturates ⇒ term 0 (no over-push into label-noise)."""
    h, w = 384, 4
    mask = np.zeros((h, w), np.float32)
    mask[100:110, :] = 1.0
    mw = _mwit(h, w, 0.9)                             # already above target 0.5
    assert horizon_margin_hinge_term(mw, mask, m_target=0.5) == 0.0


def test_empty_mask_is_zero_no_divzero():
    """An all-zero mask ⇒ sum(mask)==0 ⇒ the +1e-6 guard returns 0.0 (never a /0)."""
    h, w = 384, 4
    mask = np.zeros((h, w), np.float32)
    mw = _mwit(h, w, 0.0)
    assert horizon_margin_hinge_term(mw, mask, m_target=0.5) == 0.0


def test_hinge_fail_closed_on_shape_mismatch():
    with pytest.raises(ValueError):
        horizon_margin_hinge_term(np.zeros((10, 10)), np.zeros((10, 9)), m_target=0.5)


# ---- full loss --------------------------------------------------------------
def test_weight_zero_is_byte_identical_zero():
    """weight==0 (DEFAULT) ⇒ EXACTLY 0.0 — the trainer skips the branch (byte-identity)."""
    loss = horizon_weighted_margin_loss(_mwit(), _margin_grid(), weight=0.0)
    assert loss == 0.0
    assert isinstance(loss, float)


def test_weight_scales_the_hinge_linearly():
    """weight>0 ⇒ loss == weight * hinge_term (REAL up-weighting, linear in weight)."""
    h, w = 384, 4
    mg = np.zeros((h, w), np.float32)
    mg[100:110, :] = 0.4                              # in-band, horizon rows
    mw = _mwit(h, w, 0.1)                             # deficit 0.4 on the band
    base = horizon_weighted_margin_loss(mw, mg, weight=1.0)
    assert base == pytest.approx(0.4, abs=1e-4)
    assert horizon_weighted_margin_loss(mw, mg, weight=2.5) == pytest.approx(2.5 * base, rel=1e-6)


def test_only_reducible_horizon_band_contributes():
    """A deficit OUTSIDE the (rows AND margin) band contributes NOTHING (wrong-band would fail)."""
    h, w = 384, 4
    mg = np.zeros((h, w), np.float32)
    mg[300:310, :] = 0.4                             # in-margin-band but rows OUTSIDE horizon -> excluded
    mw = _mwit(h, w, 0.0)                            # full deficit everywhere
    assert horizon_weighted_margin_loss(mw, mg, weight=1.0) == 0.0
    # move the SAME deficit into the horizon rows -> now it contributes
    mg2 = np.zeros((h, w), np.float32)
    mg2[100:110, :] = 0.4
    assert horizon_weighted_margin_loss(mw, mg2, weight=1.0) > 0.0


def test_below_lo_label_noise_excluded():
    """A <lo margin (irreducible label-noise) is EXCLUDED even in the horizon rows (the core #169 point)."""
    h, w = 384, 4
    mg = np.full((h, w), 0.02, np.float32)          # <0.05 label-noise everywhere
    mg[100:200, :] = 0.02                            # even in horizon rows
    mw = _mwit(h, w, 0.0)
    assert horizon_weighted_margin_loss(mw, mg, weight=1.0) == 0.0


def test_loss_fail_closed_on_negative_weight():
    with pytest.raises(ValueError):
        horizon_weighted_margin_loss(_mwit(), _margin_grid(), weight=-1.0)


# ---- DSL leg ----------------------------------------------------------------
def test_dsl_factory_emits_only_real_flags_defaults_off():
    """HorizonWeightedMargin emits ONLY real trainer flags; default weight 0.0 (byte-identical)."""
    from pathlib import Path

    from tac.witness_dsl.curriculum_dsl import HorizonWeightedMargin, real_trainer_flags

    lev = HorizonWeightedMargin()
    assert lev.name == "horizon_weighted_margin"
    assert lev.overrides["--seg-horizon-margin-weight"] == 0.0    # default OFF
    real = real_trainer_flags(Path("experiments/train_levelset_witness_realized_through_R_mlx.py"))
    for flag in lev.overrides:
        assert flag in real, f"INVENTED FLAG (not in trainer argparse): {flag}"


def test_dsl_factory_threads_params_and_start_epoch():
    """start_epoch (the stage-boundary gate) + band params thread into the overrides."""
    from tac.witness_dsl.curriculum_dsl import HorizonWeightedMargin

    lev = HorizonWeightedMargin(weight=3.0, target=0.55, margin_lo=0.35, margin_hi=0.55,
                                row_lo=120, row_hi=260, start_epoch=800)
    assert lev.overrides["--seg-horizon-margin-weight"] == 3.0
    assert lev.overrides["--seg-horizon-margin-target"] == 0.55
    assert lev.overrides["--seg-horizon-margin-lo"] == 0.35
    assert lev.overrides["--seg-horizon-margin-hi"] == 0.55
    assert lev.overrides["--seg-horizon-row-lo"] == 120
    assert lev.overrides["--seg-horizon-row-hi"] == 260
    assert lev.overrides["--seg-horizon-margin-start-epoch"] == 800


def test_dsl_factory_fail_closed_on_bad_band():
    from tac.witness_dsl.curriculum_dsl import HorizonWeightedMargin

    with pytest.raises(ValueError):
        HorizonWeightedMargin(margin_lo=0.5, margin_hi=0.3)
    with pytest.raises(ValueError):
        HorizonWeightedMargin(row_lo=300, row_hi=100)
    with pytest.raises(ValueError):
        HorizonWeightedMargin(weight=-1.0)
    with pytest.raises(ValueError):
        HorizonWeightedMargin(target=0.0)


def test_dsl_lever_is_composable_by_name():
    """--dsl-lever HorizonWeightedMargin resolves to the horizon_weighted_margin Lever."""
    from tac.witness_dsl.lever_registry import name_composable_levers, resolve_composable_lever

    assert "HorizonWeightedMargin" in name_composable_levers()
    assert resolve_composable_lever("HorizonWeightedMargin").name == "horizon_weighted_margin"


def test_dsl_export_is_public():
    """HorizonWeightedMargin is exported from the witness_dsl package (not orphaned)."""
    import tac.witness_dsl as wd

    assert hasattr(wd, "HorizonWeightedMargin")
    assert "HorizonWeightedMargin" in wd.__all__


def test_registry_maps_all_horizon_flags():
    """After this build, NO --seg-horizon flag is UNMAPPED (the DSL holds them all)."""
    from tac.witness_dsl.lever_registry import completeness

    un = [f for f in completeness().unmapped if "horizon" in f]
    assert un == [], f"still-unmapped horizon flags: {un}"


# ---- equation leg -----------------------------------------------------------
def test_equation_builds_and_has_honest_tiers():
    """The #169 law builds with a VERIFIED degeneracy anchor + an ASSUMED (owed A/B) effect anchor."""
    from tac.canonical_equations.equation import (
        ASSUMED_AWAITING_VERIFICATION,
        VERIFIED_VIA_SOURCE_INSPECTION,
    )
    from tac.canonical_equations.horizon_weighted_margin_20260709 import (
        build_horizon_weighted_margin_v1,
    )

    eq = build_horizon_weighted_margin_v1()
    assert eq.equation_id == "horizon_weighted_margin_hinge_v1"
    statuses = {a.empirical_verification_status for a in eq.empirical_anchors}
    assert VERIFIED_VIA_SOURCE_INSPECTION in statuses
    assert ASSUMED_AWAITING_VERIFICATION in statuses
    # producer/consumer wired (no orphan equation)
    assert "tac.boundary_math.horizon_weighted_margin" in eq.canonical_producers
    assert "tac.witness_dsl.curriculum_dsl" in eq.canonical_consumers
