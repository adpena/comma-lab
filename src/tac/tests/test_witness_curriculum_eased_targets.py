"""Tests for tac.witness_curriculum eased-target operators (task #323).

Verify the load-bearing invariants on synthetic label maps (fast, no cache):
filtration monotonicity + continuity for the blob SDF-dilation; identity at param 0;
manifold-preservation (curve stays a curve) for the oriented widener; and the
birthability blob/curve classification.
"""
import numpy as np

from tac.witness_curriculum import birthability, oriented_width_eased, sdf_dilation_eased


def _blob(h=64, w=64, cx=32, cy=32, r=6):
    yy, xx = np.mgrid[0:h, 0:w]
    lab = np.zeros((h, w), dtype=np.int64)
    lab[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 3          # a movable blob
    return lab


def _dashed_line(h=64, w=64, col=32, dash=3, gap=4):
    lab = np.zeros((h, w), dtype=np.int64)
    y = 2
    while y < h - dash:
        lab[y:y + dash, col:col + 1] = 1                        # a lane dash
        y += dash + gap
    return lab


def test_sdf_dilation_identity_at_zero():
    lab = _blob()
    assert np.array_equal(sdf_dilation_eased(lab, 3, 0), lab)


def test_sdf_dilation_is_monotone_filtration():
    lab = _blob()
    areas = [(sdf_dilation_eased(lab, 3, r) == 3).mean() for r in range(0, 6)]
    assert all(areas[i] <= areas[i + 1] + 1e-12 for i in range(len(areas) - 1))
    # nested: every class-3 pixel at r is class-3 at r+1
    for r in range(0, 5):
        a = sdf_dilation_eased(lab, 3, r) == 3
        b = sdf_dilation_eased(lab, 3, r + 1) == 3
        assert np.all(b[a])


def test_sdf_dilation_continuity_bounded_step():
    lab = _blob()
    prev = sdf_dilation_eased(lab, 3, 0)
    for r in range(1, 6):
        cur = sdf_dilation_eased(lab, 3, r)
        step = (cur != prev).mean()
        assert step < 0.10                                     # bounded per-step change
        prev = cur


def test_oriented_width_identity_at_zero():
    lab = _dashed_line()
    assert np.array_equal(oriented_width_eased(lab, 1, 0), lab)


def test_oriented_width_preserves_curve_manifold():
    """A vertical dashed line widened along its tangent stays TALL and THIN
    (curve-like); it must not become a squat blob (which isotropic dilation would).
    VP placed directly above the line so the openpilot tangent is ~vertical (the real
    lstars have VP=(256,174) on-canvas; this synthetic 64×64 needs a consistent VP)."""
    lab = _dashed_line()
    eased = oriented_width_eased(lab, 1, 3, vanishing_point=(32.0, -500.0))
    ys, xs = np.where(eased == 1)
    height = ys.max() - ys.min()
    breadth = xs.max() - xs.min()
    assert height > breadth * 3                                # still an elongated curve


def test_oriented_width_vp_tangent_points_at_vanishing_point():
    """The openpilot default orients each segment toward the VANISHING POINT, not along
    the component's own shape-PCA (the #325 robustness fix for short/near-square dashes).
    A single near-square blob has an ill-defined shape-axis; with a VP off to the upper
    right, the VP-widened mask must extend toward that VP."""
    lab = np.zeros((80, 80), dtype=np.int64)
    lab[40:44, 20:24] = 1                                      # a ~square lane fragment
    vp = oriented_width_eased(lab, 1, 4, vanishing_point=(78.0, 2.0))
    ys, xs = np.where(vp == 1)
    # extends up-and-right toward (78,2): max x beyond the blob's right edge (23),
    # min y above the blob's top edge (40)
    assert xs.max() > 23 and ys.min() < 40


def test_oriented_width_pca_mode_is_still_available():
    """The legacy shape-PCA path (mode='pca') widens along the component's own axis
    regardless of the VP — kept for the clean A/B against the openpilot default."""
    lab = _dashed_line()
    eased = oriented_width_eased(lab, 1, 3, tangent_mode="pca")
    ys, xs = np.where(eased == 1)
    assert (ys.max() - ys.min()) > (xs.max() - xs.min()) * 3   # own vertical axis


def test_birthability_blob_vs_curve():
    # a big blob → birthable_blob
    big = np.zeros((64, 64), dtype=np.int64)
    big[10:40, 10:40] = 3
    b = birthability(big == 3)
    assert b.birthable_blob and b.largest_cc_frac > 0.9
    # a set of coherent line-segments spanning enough area → birthable_curve, not blob
    lines = np.zeros((64, 64), dtype=np.int64)
    for c in (10, 20, 30, 40):
        lines[2:60, c:c + 1] = 1
    bc = birthability(lines == 1)
    assert bc.coherent_seg_frac > 0.6
    assert bc.n_components >= 3


def test_empty_mask_is_not_birthable():
    b = birthability(np.zeros((16, 16), dtype=bool))
    assert not b.birthable and b.area_frac == 0.0
