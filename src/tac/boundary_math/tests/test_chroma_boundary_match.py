# SPDX-License-Identifier: MIT
"""NO-FAKE tests for LEVER-4c annulus-directed chroma-boundary match (reference twin + DSL leg + equation).

The defining contracts (asserted on REAL behavior, never constants):
  * BYTE-IDENTITY when OFF: weight==0 ⇒ the loss is EXACTLY 0.0 AND no chroma/mask is built (the trainer's
    default-OFF path, gated on chroma_bnd_w>0 with providers None, is byte-identical by construction).
  * LUMA-INVARIANCE: chroma(rgb + c·[1,1,1]) == chroma(rgb) ⇒ ORTHOGONAL to every luma lever (the whole
    reason chroma is an INDEPENDENT d_seg DOF; a non-invariant impl would FAIL — the constants-not-behavior
    guard).
  * REAL up-weighting when ON: ONLY pixels inside the fragile annulus (GT margin < band) with a chroma
    MISMATCH contribute; a no-op / wrong-band / bulk term would FAIL.
  * PERFECT MATCH ⇒ 0 (the squared error is 0). EMPTY annulus ⇒ 0.0 (the +1e-6 guard, never a /0).
  * Fail-closed: shape mismatches / bad args raise (never silently broadcast the wrong grid / empty band).
  * DSL leg: the SegChromaBoundary factory emits ONLY real trainer flags, defaults off, is composable, is
    exported from tac.witness_dsl, and threads start_epoch (the stage-boundary gate). Equation leg:
    registers + is queryable with the honest verification tiers.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.chroma_boundary_match import (
    annulus_mask,
    bt601_chroma,
    chroma_boundary_loss,
    chroma_boundary_term,
)

_BT601 = (0.299, 0.587, 0.114)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------------- byte-identity when OFF ----
def test_weight_zero_is_byte_identical_zero():
    """weight==0 (default) ⇒ EXACTLY 0.0 (float), the trainer's byte-identical default-OFF path."""
    r = _rng(1)
    f1 = r.random((12, 16, 3)).astype(np.float32)
    gt = r.random((12, 16, 3)).astype(np.float32)
    m = r.random((12, 16)).astype(np.float32)  # random margins, some < band
    out = chroma_boundary_loss(f1, gt, m, weight=0.0, band=1.0)
    assert out == 0.0
    assert isinstance(out, float)
    # default weight is also 0.0
    assert chroma_boundary_loss(f1, gt, m) == 0.0


def test_weight_zero_short_circuits_before_any_build():
    """weight==0 must NOT even require a valid margin/band (it returns before building chroma/mask) — this
    is the byte-identity guarantee: the trainer never constructs the branch, so nothing downstream runs."""
    f1 = np.zeros((4, 4, 3), np.float32)
    gt = np.zeros((4, 4, 3), np.float32)
    bad_margin = np.zeros((7, 7), np.float32)  # WRONG grid — would raise if the mask were built
    assert chroma_boundary_loss(f1, gt, bad_margin, weight=0.0) == 0.0


# ----------------------------------------------------------------------------------- luma invariance -----
def test_chroma_is_luma_invariant():
    """chroma(rgb + c·[1,1,1]) == chroma(rgb): adding a CONSTANT luma to all 3 channels leaves chroma
    unchanged (the property that makes LEVER-4c orthogonal to every luma lever). This is REAL behavior — a
    non-invariant implementation (e.g. forgetting to subtract luma) would FAIL here."""
    r = _rng(2)
    rgb = r.random((10, 10, 3)).astype(np.float32)
    for c in (0.1, 0.5, -0.3, 1.7):
        shifted = (rgb + np.float32(c)).astype(np.float32)
        d = np.max(np.abs(bt601_chroma(rgb) - bt601_chroma(shifted)))
        assert d < 1e-6, f"chroma not luma-invariant for c={c}: maxdiff {d}"


def test_chroma_matches_bt601_definition():
    """chroma == rgb − (0.299R + 0.587G + 0.114B)·[1,1,1] — the EXACT BT.601 the trainer uses (not a proxy)."""
    r = _rng(3)
    rgb = r.random((5, 5, 3)).astype(np.float32)
    luma = _BT601[0] * rgb[..., 0] + _BT601[1] * rgb[..., 1] + _BT601[2] * rgb[..., 2]
    expect = rgb - luma[..., None]
    got = bt601_chroma(rgb)
    assert np.allclose(got, expect, atol=1e-6)


def test_chroma_luma_channel_is_zero():
    """The BT.601 luma of the chroma is ~0 by construction (chroma has no luma component)."""
    r = _rng(4)
    c = bt601_chroma(r.random((8, 8, 3)).astype(np.float32))
    luma_of_chroma = _BT601[0] * c[..., 0] + _BT601[1] * c[..., 1] + _BT601[2] * c[..., 2]
    assert np.max(np.abs(luma_of_chroma)) < 1e-6


# ------------------------------------------------------------------------- annulus mask (the fragile band) --
def test_annulus_mask_is_strict_less_than_band():
    """The annulus is 1[margin < band] (STRICT <, mirroring the trainer's (_mg < band)) — margin==band is
    OUT, margin just below is IN."""
    m = np.array([[0.5, 1.0], [1.5, 0.999]], np.float32)
    mk = annulus_mask(m, band=1.0)
    assert mk[0, 0] == 1.0   # 0.5 < 1.0 IN
    assert mk[0, 1] == 0.0   # 1.0 < 1.0 is FALSE — OUT
    assert mk[1, 0] == 0.0   # 1.5 OUT
    assert mk[1, 1] == 1.0   # 0.999 IN


def test_annulus_band_controls_fraction():
    """A SMALLER band selects FEWER pixels (the fragile band tightens) — REAL monotone behavior."""
    r = _rng(5)
    m = r.random((40, 40)).astype(np.float32)  # margins in [0,1)
    n_wide = float(annulus_mask(m, band=1.0).sum())
    n_tight = float(annulus_mask(m, band=0.25).sum())
    assert n_tight < n_wide
    assert n_wide == m.size  # all margins < 1.0


# --------------------------------------------------------------------------------- term degeneracies ------
def test_perfect_match_is_zero():
    """witness chroma == GT chroma on the annulus ⇒ squared error 0 ⇒ term 0.0 (not a constant — a
    mismatch below would be > 0, proven by the next test)."""
    r = _rng(6)
    f1 = r.random((10, 10, 3)).astype(np.float32)
    ann = np.ones((10, 10), np.float32)
    assert chroma_boundary_term(f1, bt601_chroma(f1), ann) == 0.0


def test_mismatch_is_positive_and_scales_with_error():
    """A chroma MISMATCH on the annulus gives a POSITIVE term that GROWS with the error magnitude (REAL
    behavior — the constants-not-behavior guard: a no-op would return 0 here)."""
    r = _rng(7)
    f1 = r.random((10, 10, 3)).astype(np.float32)
    ann = np.ones((10, 10), np.float32)
    gt_c = bt601_chroma(f1)
    small = chroma_boundary_term(f1 + 0.05, gt_c, ann)
    large = chroma_boundary_term(f1 + 0.20, gt_c, ann)
    assert small > 0.0
    assert large > small


def test_empty_annulus_is_zero_no_divzero():
    """No pixel with margin < band ⇒ sum(ann)==0 ⇒ the +1e-6 guard returns 0.0 (never a /0)."""
    r = _rng(8)
    f1 = r.random((6, 6, 3)).astype(np.float32)
    gt = r.random((6, 6, 3)).astype(np.float32)
    all_out = np.full((6, 6), 2.0, np.float32)  # every margin >= band
    out = chroma_boundary_loss(f1, gt, all_out, weight=1.0, band=1.0)
    assert out == 0.0


def test_only_fragile_annulus_contributes():
    """ONLY pixels inside the annulus (margin < band) contribute; a mismatch OUTSIDE the annulus is
    IGNORED. Two frames identical except an out-of-band pixel ⇒ same term; differ on an in-band pixel ⇒
    different term (the wrong-band guard)."""
    f1 = np.zeros((3, 3, 3), np.float32)
    gt = np.zeros((3, 3, 3), np.float32)
    margin = np.full((3, 3), 2.0, np.float32)  # all OUT
    margin[1, 1] = 0.5                          # ONLY the center is in-band
    # a mismatch on an OUT-of-band pixel changes nothing:
    f1_out = f1.copy()
    f1_out[0, 0, 1] = 0.9   # green bump on an out-of-band pixel
    base = chroma_boundary_loss(f1, gt, margin, weight=1.0, band=1.0)
    assert chroma_boundary_loss(f1_out, gt, margin, weight=1.0, band=1.0) == base == 0.0
    # a mismatch on the IN-band pixel DOES change it:
    f1_in = f1.copy()
    f1_in[1, 1, 1] = 0.9
    assert chroma_boundary_loss(f1_in, gt, margin, weight=1.0, band=1.0) > 0.0


def test_weight_scales_linearly():
    """The term is linear in weight (w·term), matching the trainer's L += chroma_bnd_w * chroma_bnd_term."""
    r = _rng(9)
    f1 = r.random((8, 8, 3)).astype(np.float32)
    gt = r.random((8, 8, 3)).astype(np.float32)
    m = np.zeros((8, 8), np.float32)  # all in-band
    a = chroma_boundary_loss(f1, gt, m, weight=1.0, band=1.0)
    b = chroma_boundary_loss(f1, gt, m, weight=3.0, band=1.0)
    assert a > 0.0
    assert abs(b - 3.0 * a) < 1e-6 * max(1.0, abs(b))


def test_loss_matches_manual_formula():
    """The full loss equals w · sum(||chroma(f1)-chroma(GT)||^2 · ann)/(sum(ann)+1e-6) — the exact trainer
    formula recomputed by hand (bit-faithfulness proof, not a self-referential tautology)."""
    r = _rng(10)
    f1 = r.random((7, 9, 3)).astype(np.float32)
    gt = r.random((7, 9, 3)).astype(np.float32)
    m = r.random((7, 9)).astype(np.float32)
    w, band = 2.5, 0.6
    cwit = bt601_chroma(f1).astype(np.float64)
    cgt = bt601_chroma(gt).astype(np.float64)
    ann = (m < band).astype(np.float64)
    cdiff2 = np.sum((cwit - cgt) ** 2, axis=-1)
    manual = w * float(np.sum(cdiff2 * ann) / (ann.sum() + 1e-6))
    got = chroma_boundary_loss(f1, gt, m, weight=w, band=band)
    assert abs(got - manual) < 1e-5 * max(1.0, abs(manual))


# --------------------------------------------------------------------------------------- fail-closed ------
def test_bt601_chroma_rejects_non_rgb():
    with pytest.raises(ValueError):
        bt601_chroma(np.zeros((4, 4, 2), np.float32))  # 2-channel last axis


def test_annulus_mask_rejects_non_2d_and_bad_band():
    with pytest.raises(ValueError):
        annulus_mask(np.zeros((4, 4, 1), np.float32), band=1.0)  # 3-D
    with pytest.raises(ValueError):
        annulus_mask(np.zeros((4, 4), np.float32), band=0.0)     # non-positive band = silent no-op


def test_term_rejects_grid_mismatch():
    f1 = np.zeros((5, 5, 3), np.float32)
    with pytest.raises(ValueError):
        chroma_boundary_term(f1, np.zeros((6, 6, 3), np.float32), np.ones((5, 5), np.float32))
    with pytest.raises(ValueError):
        chroma_boundary_term(f1, np.zeros((5, 5, 3), np.float32), np.ones((6, 6), np.float32))


def test_loss_rejects_negative_weight():
    f1 = np.zeros((4, 4, 3), np.float32)
    with pytest.raises(ValueError):
        chroma_boundary_loss(f1, f1, np.zeros((4, 4), np.float32), weight=-1.0)


# --------------------------------------------------------------------------------------- DSL leg ----------
def test_dsl_factory_default_off_and_flags_mapped():
    """SegChromaBoundary() defaults to weight 0.0 (byte-identical) and emits ONLY real trainer flags."""
    from tac.witness_dsl.curriculum_dsl import SegChromaBoundary
    lv = SegChromaBoundary()
    assert lv.overrides["--seg-chroma-boundary-weight"] == 0.0
    # every override key must be a REAL trainer flag (no invented flags)
    expect = {"--seg-chroma-boundary-weight", "--seg-chroma-boundary-margin-band",
              "--seg-chroma-boundary-start-epoch"}
    assert set(lv.overrides) == expect


def test_dsl_factory_threads_values_and_start_epoch():
    from tac.witness_dsl.curriculum_dsl import SegChromaBoundary
    lv = SegChromaBoundary(weight=0.4, margin_band=0.25, start_epoch=300)
    assert lv.overrides["--seg-chroma-boundary-weight"] == 0.4
    assert lv.overrides["--seg-chroma-boundary-margin-band"] == 0.25
    assert lv.overrides["--seg-chroma-boundary-start-epoch"] == 300  # the stage-boundary gate


def test_dsl_factory_fails_closed_on_bad_args():
    from tac.witness_dsl.curriculum_dsl import SegChromaBoundary
    with pytest.raises(ValueError):
        SegChromaBoundary(weight=-0.1)
    with pytest.raises(ValueError):
        SegChromaBoundary(margin_band=0.0)   # non-positive band = silent no-op
    with pytest.raises(ValueError):
        SegChromaBoundary(start_epoch=-5)


def test_dsl_factory_exported_from_witness_dsl():
    import tac.witness_dsl as wd
    assert hasattr(wd, "SegChromaBoundary")
    assert "SegChromaBoundary" in wd.__all__


def test_dsl_flags_are_real_argparse_flags():
    """The 3 override flags must exist in the trainer argparse (NO invented flags — grep-the-target rule)."""
    from pathlib import Path
    src = Path("experiments/train_levelset_witness_realized_through_R_mlx.py").read_text()
    for flag in ("--seg-chroma-boundary-weight", "--seg-chroma-boundary-margin-band",
                 "--seg-chroma-boundary-start-epoch"):
        assert f'"{flag}"' in src, f"{flag} not found in the trainer argparse"


# --------------------------------------------------------------------------------------- equation leg -----
def test_equation_builds_with_honest_tiers():
    from tac.canonical_equations.chroma_boundary_match_20260709 import (
        build_chroma_boundary_annulus_match_hinge_v1,
    )
    from tac.canonical_equations.equation import (
        ASSUMED_AWAITING_VERIFICATION,
        VERIFIED_VIA_SOURCE_INSPECTION,
    )
    eq = build_chroma_boundary_annulus_match_hinge_v1()
    assert eq.equation_id == "chroma_boundary_annulus_match_hinge_v1"
    statuses = {a.empirical_verification_status for a in eq.empirical_anchors}
    # exactly the two honest tiers: the degeneracy is source-verified, the effect is owed a converged A/B
    assert VERIFIED_VIA_SOURCE_INSPECTION in statuses
    assert ASSUMED_AWAITING_VERIFICATION in statuses
    # the callable path points at the reference twin (producer/consumer wiring is real)
    assert "chroma_boundary_match" in eq.python_callable_module_path


def test_equation_cites_the_dof_source_and_makes_no_score_claim():
    """The effect anchor must cite the chroma DOF equation as its SOURCE and be explicit that the ADD-BACK
    ΔS is UNMEASURED (means != ends; pointer UNMOVED)."""
    from tac.canonical_equations.chroma_boundary_match_20260709 import (
        build_chroma_boundary_annulus_match_hinge_v1,
    )
    eq = build_chroma_boundary_annulus_match_hinge_v1()
    effect = next(a for a in eq.empirical_anchors if "effect" in a.anchor_id)
    blob = str(effect.inputs) + str(effect.empirical_output)
    assert "chroma_decides_lane_and_movable_at_annulus_v1" in blob  # cites the DOF source
    assert "UNMEASURED" in blob
    assert "0.19110" in blob  # pointer UNMOVED disclaimer
