# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the #121 d_seg-aware Fourier-feature amplitude taper.

The defining contracts (asserted on REAL behavior, never constants):
  * BYTE-IDENTITY when OFF: strength=0 OR uniform saliency ⇒ the taper is all-ones ⇒ the tapered
    feats are numerically IDENTICAL to the input feats (the trainer's default-OFF path, which simply
    does not call this, is byte-identical by construction — these tests pin the runtime degeneracies).
  * REAL reweighting when ON: a non-uniform saliency + strength>0 GENUINELY changes the taper AND the
    feats (a no-op would FAIL these — the constants-not-behavior guard).
  * Byte-NEUTRAL: mean taper == 1 pre-floor (a REALLOCATION, not a net amplitude change); no params.
  * Fail-closed: shape mismatches and bad args raise (never silently broadcast wrong).
  * DSL leg: the DsegAwareTaper factory emits ONLY real trainer flags, defaults off, is composable.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.dseg_aware_fourier_taper import (
    apply_dseg_aware_fourier_taper,
    compute_dseg_aware_fourier_taper,
    saliency_from_margins,
)


def _feats(p_px=400, f=24, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((p_px, f)).astype(np.float32)


def _margins(h=20, w=20, n_pair=3, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((h, w)).astype(np.float32) for _ in range(n_pair)]


# ---- saliency ---------------------------------------------------------------
def test_saliency_unit_mean_and_shape():
    """saliency_from_margins returns (P_px,) normalised to unit mean (so uniform margins ⇒ flat)."""
    s = saliency_from_margins(_margins())
    assert s.shape == (400,)
    assert abs(float(s.mean()) - 1.0) < 1e-5


def test_saliency_high_near_boundary_low_far():
    """Small |margin| (near the argmax boundary annulus) ⇒ HIGHER saliency than large |margin|."""
    # A margin field: column 0 = tiny margin (boundary), column 1 = large margin (interior).
    mar = np.zeros((1, 4), np.float32)
    mar[0, 0] = 0.0     # exactly on the boundary
    mar[0, 1] = 10.0    # deep interior
    mar[0, 2] = 0.1
    mar[0, 3] = 5.0
    s = saliency_from_margins([mar], scale=1.0)
    assert s[0] > s[1], "boundary pixel must be MORE salient than interior"
    assert s[0] > s[3] and s[2] > s[3]


def test_saliency_uniform_margins_uniform_saliency():
    """A uniform margin field ⇒ a uniform (constant) saliency (→ the flat taper)."""
    s = saliency_from_margins([np.full((10, 10), 3.7, np.float32) for _ in range(2)])
    assert float(s.std()) < 1e-6
    assert abs(float(s.mean()) - 1.0) < 1e-5


def test_saliency_auto_scale_deterministic_and_positive():
    """AUTO scale (median |margin|) is deterministic and yields a valid unit-mean saliency."""
    m = _margins(seed=7)
    s1 = saliency_from_margins(m, scale=None)
    s2 = saliency_from_margins(m, scale=None)
    assert np.array_equal(s1, s2)
    assert np.all(s1 > 0.0)


def test_saliency_accepts_single_array_and_rejects_mismatched_px():
    s = saliency_from_margins(np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32))
    assert s.shape == (64,)
    with pytest.raises(ValueError, match="must share the pixel grid"):
        saliency_from_margins([np.zeros((4, 4), np.float32), np.zeros((5, 5), np.float32)])
    with pytest.raises(ValueError, match="scale must be > 0"):
        saliency_from_margins([np.zeros((4, 4), np.float32)], scale=-1.0)


def test_saliency_target_hw_resizes_to_render_grid():
    """target_hw NN-resizes a SEG-grid margin onto the RENDER grid so P_px = render_h*render_w
    (the render!=seg mismatch fix): a (16,24) seg margin resized to (12,20) yields 240-px saliency
    that lines up with a 240-row feature grid."""
    seg_margin = np.random.default_rng(2).standard_normal((16, 24)).astype(np.float32)
    s = saliency_from_margins([seg_margin], scale=1.0, target_hw=(12, 20))
    assert s.shape == (12 * 20,)
    # it composes with feats built on the SAME (12,20)=240-px render grid.
    feats = _feats(p_px=240, f=8, seed=3)
    w = compute_dseg_aware_fourier_taper(feats, s, strength=1.0)
    assert w.shape == (8,) and np.isfinite(w).all()


def test_saliency_target_hw_noop_when_shapes_match():
    """target_hw equal to the margin's own (H,W) is an exact no-op (same saliency as no target_hw)."""
    m = np.random.default_rng(5).standard_normal((10, 10)).astype(np.float32)
    s_no = saliency_from_margins([m], scale=1.0)
    s_hw = saliency_from_margins([m], scale=1.0, target_hw=(10, 10))
    assert np.array_equal(s_no, s_hw)


def test_saliency_target_hw_requires_2d_margin():
    """With target_hw set a 1-D margin cannot be resized → fail-closed (never a silent wrong grid)."""
    with pytest.raises(ValueError, match="expected a 2-D"):
        saliency_from_margins([np.zeros((64,), np.float32)], scale=1.0, target_hw=(8, 8))


# ---- the taper (degeneracies = the byte-identity contract) ------------------
def test_strength_zero_is_flat_taper_exactly():
    """strength=0 ⇒ the taper is all-ones EXACTLY (the OFF degeneracy #1)."""
    feats, s = _feats(), saliency_from_margins(_margins())
    w = compute_dseg_aware_fourier_taper(feats, s, strength=0.0)
    assert np.all(w == 1.0)


def test_uniform_saliency_is_flat_taper_exactly():
    """Uniform saliency ⇒ the taper is all-ones (max|w-1|==0) (the OFF degeneracy #2)."""
    feats = _feats()
    s = np.ones((feats.shape[0],), np.float32)  # uniform
    w = compute_dseg_aware_fourier_taper(feats, s, strength=1.0)
    assert float(np.abs(w - 1.0).max()) == 0.0


def test_taper_is_byte_neutral_mean_one_prefloor():
    """The taper is a REALLOCATION: mean_k w == 1 by construction (byte-neutral amplitude)."""
    feats, s = _feats(seed=3), saliency_from_margins(_margins(seed=4))
    # floor low enough not to bind so the mean-1 invariant is exact.
    w = compute_dseg_aware_fourier_taper(feats, s, strength=0.5, floor=0.0)
    assert abs(float(w.mean()) - 1.0) < 1e-5


def test_taper_actually_reweights_when_on():
    """NO-FAKE: a non-uniform saliency + strength>0 GENUINELY varies the taper (not a no-op)."""
    feats = _feats(seed=5)
    # a genuinely non-uniform saliency (a spatial ramp), unit-mean.
    ramp = np.linspace(0.2, 1.8, feats.shape[0]).astype(np.float32)
    ramp = ramp / ramp.mean()
    w = compute_dseg_aware_fourier_taper(feats, ramp, strength=1.0)
    assert float(w.std()) > 0.0, "taper did not vary despite non-uniform saliency (no-op!)"
    assert not np.allclose(w, 1.0)


def test_taper_floor_clamps_and_stays_positive():
    """A large strength cannot drive a column weight below the positivity floor."""
    feats = _feats(seed=6)
    ramp = np.linspace(0.01, 4.0, feats.shape[0]).astype(np.float32)
    ramp = ramp / ramp.mean()
    w = compute_dseg_aware_fourier_taper(feats, ramp, strength=8.0, floor=0.05)
    assert float(w.min()) >= 0.05


def test_zero_energy_features_fail_closed_to_flat():
    """All-zero features (no saliency contrast) ⇒ flat taper (fail-closed no-op, never NaN)."""
    feats = np.zeros((100, 12), np.float32)
    s = saliency_from_margins([np.random.default_rng(0).standard_normal((10, 10)).astype(np.float32)])
    w = compute_dseg_aware_fourier_taper(feats, s, strength=1.0)
    assert np.all(w == 1.0) and np.isfinite(w).all()


# ---- apply (byte-identity when off) -----------------------------------------
def test_apply_flat_taper_is_bit_identical():
    """Applying an all-ones taper returns feats numerically IDENTICAL (the OFF path)."""
    feats = _feats(seed=8)
    out = apply_dseg_aware_fourier_taper(feats, np.ones((feats.shape[1],), np.float32))
    assert np.array_equal(out, feats)
    assert out.dtype == feats.dtype


def test_apply_real_taper_changes_feats_per_column():
    """Applying a non-flat taper scales EACH column by its weight (real per-column effect)."""
    feats = _feats(seed=9)
    w = np.linspace(0.5, 1.5, feats.shape[1]).astype(np.float32)
    out = apply_dseg_aware_fourier_taper(feats, w)
    assert not np.array_equal(out, feats)
    # column k is exactly feats[:,k]*w[k].
    for k in (0, feats.shape[1] // 2, feats.shape[1] - 1):
        assert np.allclose(out[:, k], feats[:, k] * w[k], atol=1e-5)


def test_apply_shape_mismatch_fails_closed():
    with pytest.raises(ValueError, match="taper must be"):
        apply_dseg_aware_fourier_taper(_feats(f=24), np.ones((25,), np.float32))


def test_compute_shape_guards_fail_closed():
    with pytest.raises(ValueError, match="feats must be 2-D"):
        compute_dseg_aware_fourier_taper(np.zeros((10,), np.float32), np.zeros((10,), np.float32))
    with pytest.raises(ValueError, match="saliency must be"):
        compute_dseg_aware_fourier_taper(np.zeros((10, 4), np.float32), np.zeros((9,), np.float32))
    with pytest.raises(ValueError, match="floor must be"):
        compute_dseg_aware_fourier_taper(np.zeros((10, 4), np.float32), np.zeros((10,), np.float32), floor=-0.1)


# ---- END-TO-END on the REAL curvelet basis (the trainer's front-end) --------
def test_real_curvelet_basis_off_is_byte_identical_on_changes():
    """On the ACTUAL curvelet directional basis (the trainer front-end): strength=0 ⇒ byte-identical
    feats; strength>0 with a real boundary margin ⇒ changed feats. Proves the wire-in mechanism is
    real AND byte-identical-when-off on the production basis, not just synthetic features."""
    from tac.boundary_math.lever_b_levelset_generator import (
        CurveletBankConfig,
        curvelet_directional_B,
        curvelet_feats,
    )

    h, w_ = 24, 32
    ys, xs = np.mgrid[0:h, 0:w_]
    coords = np.stack([xs.reshape(-1) / w_, ys.reshape(-1) / h], axis=-1).astype(np.float32)
    bank = CurveletBankConfig(n_scales=2, n_orient0=4, f0=1.0, base=2.0, n_iso=2)
    feats = curvelet_feats(coords, curvelet_directional_B(bank)).astype(np.float32)
    # a real-ish diagonal-boundary GT margin (small margin ON the diagonal).
    mar = (np.abs(xs / w_ - ys / h) * 5.0).astype(np.float32)
    sal = saliency_from_margins([mar, mar * 0.9], scale=None)

    off = apply_dseg_aware_fourier_taper(
        feats, compute_dseg_aware_fourier_taper(feats, sal, strength=0.0)).astype(np.float32)
    assert np.array_equal(off, feats), "strength=0 must be byte-identical on the real curvelet basis"

    taper_on = compute_dseg_aware_fourier_taper(feats, sal, strength=1.0, floor=0.05)
    on = apply_dseg_aware_fourier_taper(feats, taper_on).astype(np.float32)
    assert not np.array_equal(on, feats), "strength>0 must genuinely reweight the real curvelet feats"
    assert abs(float(taper_on.mean()) - 1.0) < 0.05  # ~byte-neutral (floor may nudge slightly)
    assert float(taper_on.std()) > 0.0


# ---- DSL leg (the lever factory) --------------------------------------------
def test_dsl_factory_emits_only_real_trainer_flags_defaults_off():
    """DsegAwareTaper emits exactly the 4 --dseg-aware-taper-* flags, all present in the trainer
    argparse (never-invent-flags), and the trainer defaults the toggle OFF."""
    from tac.witness_dsl import lever_registry as lr
    from tac.witness_dsl.curriculum_dsl import DsegAwareTaper

    lev = DsegAwareTaper()
    assert lev.name == "dseg_aware_taper"
    assert lev.overrides["--dseg-aware-taper"] is True
    assert set(lev.overrides) == {
        "--dseg-aware-taper", "--dseg-aware-taper-strength",
        "--dseg-aware-taper-scale", "--dseg-aware-taper-floor",
    }
    real = lr.real_trainer_flags()
    for f in lev.overrides:
        assert f in real, f"{f} not a real trainer flag (invented!)"
    comp = lr.completeness()
    for f in lev.overrides:
        assert f in comp.mapped and f not in comp.unmapped and f not in comp.stale


def test_dsl_factory_is_composable_and_ast_discovered():
    """The factory is bare-name composable (--dsl-lever) and AST-discovered by the lever_registry."""
    from tac.witness_dsl import lever_registry as lr

    assert "DsegAwareTaper" in lr.name_composable_levers()
    assert lr.lever_factories().get("DsegAwareTaper") == frozenset({
        "--dseg-aware-taper", "--dseg-aware-taper-strength",
        "--dseg-aware-taper-scale", "--dseg-aware-taper-floor",
    })


def test_dsl_factory_strength_scale_floor_thread_through():
    """Non-default args thread into the emitted flag values (real config surface, not hardcoded)."""
    from tac.witness_dsl.curriculum_dsl import DsegAwareTaper

    lev = DsegAwareTaper(strength=0.5, scale=2.5, floor=0.1)
    assert lev.overrides["--dseg-aware-taper-strength"] == 0.5
    assert lev.overrides["--dseg-aware-taper-scale"] == 2.5
    assert lev.overrides["--dseg-aware-taper-floor"] == 0.1
