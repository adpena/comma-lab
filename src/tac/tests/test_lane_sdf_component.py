"""Tests for the manifold-aware lane-SDF level-set component (FEED-dm).

NO-FAKE: these verify BEHAVIOR (the SDF actually signs by band membership; the
clustering actually finds lines; the band actually covers a planted lane; the
decomposition actually attributes flips), not constants.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.lane_sdf_component import (
    ContainmentDecomp,
    LaneLine,
    build_structured_lane_sdf,
    cluster_lane_lines,
    decompose_argmax_disagreement,
    ground_to_image_row,
    image_to_ground,
    inject_lane_sdf,
    lane_signed_distance,
    rasterize_lane_band,
)

_H, _W = 384, 512
_V_H = 174.0


def _planted_lane_labels(n_lines: int = 2, lateral_m=(-1.8, 1.8), dash=False) -> np.ndarray:
    """Build a (H,W) label map: road=0 below horizon, undrivable=3 above, with n_lines
    straight class-1 lane lines placed via the module's own ground->image map."""

    lab = np.zeros((_H, _W), np.int64)
    lab[: int(_V_H) + 1, :] = 3  # above horizon = undrivable (not road)
    rows = np.arange(int(_V_H) + 2, _H, dtype=np.float64)
    for k in range(n_lines):
        lat = float(lateral_m[k])
        ln = LaneLine(centerline_coeffs=np.array([lat]), halfwidth_coeffs=np.array([0.0, 3.0]),
                      dash_period_m=(8.0 if dash else 0.0), dash_phase_m=0.0, dash_duty=0.5,
                      forward_range=(4.0, 80.0))
        fwd, u_c = ground_to_image_row(rows, ln.lateral_of_forward, v_h=_V_H)
        hw = ln.halfwidth_of_v(rows)
        on = np.ones_like(rows, bool)
        if dash:
            on = (np.mod(fwd, 8.0) / 8.0) < 0.5
        for j, v in enumerate(rows):
            if not on[j]:
                continue
            lo = int(max(0, np.floor(u_c[j] - hw[j])))
            hi = int(min(_W, np.ceil(u_c[j] + hw[j]) + 1))
            if 0 <= u_c[j] < _W and hi > lo:
                lab[int(v), lo:hi] = 1
    return lab


# ---------------------------------------------------------------------------
# IPM
# ---------------------------------------------------------------------------
def test_ipm_round_trip_consistent():
    # pick image points below horizon; image->ground->image returns the same u
    v = np.array([250.0, 300.0, 360.0])
    u = np.array([200.0, 256.0, 320.0])
    fwd, lat = image_to_ground(u, v, v_h=_V_H)
    assert np.all(np.isfinite(fwd)) and np.all(fwd > 0)
    # reconstruct u via ground_to_image_row using a constant-lateral "centerline"
    for i in range(len(v)):
        _, u_c = ground_to_image_row(np.array([v[i]]), lambda f, L=lat[i]: np.full_like(f, L), v_h=_V_H)
        assert abs(float(u_c[0]) - u[i]) < 1e-6


def test_ipm_above_horizon_is_inf():
    fwd, lat = image_to_ground(np.array([256.0]), np.array([_V_H - 10.0]), v_h=_V_H)
    assert not np.isfinite(fwd[0])


def test_ipm_forward_increases_toward_horizon():
    fwd_near, _ = image_to_ground(np.array([256.0]), np.array([380.0]), v_h=_V_H)
    fwd_far, _ = image_to_ground(np.array([256.0]), np.array([200.0]), v_h=_V_H)
    assert fwd_far[0] > fwd_near[0]  # rows closer to horizon = farther forward


# ---------------------------------------------------------------------------
# signed distance
# ---------------------------------------------------------------------------
def test_lane_signed_distance_sign_convention():
    m = np.zeros((40, 40), bool)
    m[15:25, 15:25] = True
    phi = lane_signed_distance(m)
    assert phi[20, 20] > 0          # inside band -> positive
    assert phi[2, 2] < 0            # far outside -> negative
    # 1-Lipschitz-ish: interior center distance ~ half the box width
    assert phi[20, 20] >= 4.0


def test_lane_signed_distance_all_true_all_false():
    assert lane_signed_distance(np.ones((10, 10), bool))[0, 0] > 0
    assert lane_signed_distance(np.zeros((10, 10), bool))[0, 0] < 0


def test_lane_signed_distance_argmax_recovers_band_vs_negative():
    m = np.zeros((40, 40), bool)
    m[10:30, 18:22] = True
    phi1 = lane_signed_distance(m)
    phi0 = np.zeros_like(phi1)  # a flat competitor at 0
    pred_lane = (phi1 > phi0)
    # band interior must be predicted lane; far corner must not
    assert pred_lane[20, 20]
    assert not pred_lane[0, 0]


# ---------------------------------------------------------------------------
# clustering + fitting + rasterization (geometry behavior)
# ---------------------------------------------------------------------------
def test_cluster_finds_two_lines():
    lab = _planted_lane_labels(n_lines=2)
    clusters = cluster_lane_lines(lab, lane_cls=1, v_h=_V_H)
    assert len(clusters) == 2, f"expected 2 lane lines, got {len(clusters)}"
    for c in clusters:
        assert c.shape[1] == 2 and c.shape[0] > 25


def test_cluster_empty_when_no_lane():
    lab = np.zeros((_H, _W), np.int64)
    assert cluster_lane_lines(lab, lane_cls=1) == []


def test_build_structured_sdf_covers_planted_lane_low_fn():
    """The structured SDF band must cover a planted straight lane with small FN."""
    lab = _planted_lane_labels(n_lines=2)
    phi1, meta = build_structured_lane_sdf(lab, lane_cls=1, dash_gate=False)
    assert meta["n_lines"] == 2
    assert phi1.shape == (_H, _W)
    # FN among true lane pixels: phi1 should be >= 0 (in band) at most true-lane px
    is_lane = lab == 1
    covered = (phi1 >= 0) & is_lane
    fn_frac = 1.0 - covered.sum() / max(is_lane.sum(), 1)
    assert fn_frac < 0.15, f"planted-lane FN too high: {fn_frac}"


def test_build_structured_sdf_meta_floats_low_order():
    lab = _planted_lane_labels(n_lines=2)
    _, meta = build_structured_lane_sdf(lab, lane_cls=1, dash_gate=False)
    # ~ centerline(<=4) + halfwidth(2) per line; 2 lines -> low-order manifold
    assert meta["total_floats"] <= 2 * 6
    assert meta["total_floats"] >= 2 * 3


def test_rasterize_dash_gate_reduces_band_area():
    lab = _planted_lane_labels(n_lines=2, dash=True)
    clusters = cluster_lane_lines(lab, lane_cls=1, v_h=_V_H)
    from tac.boundary_math.lane_sdf_component import fit_lane_line
    lines = [fit_lane_line(c, fit_dash=True) for c in clusters]
    lines = [ln for ln in lines if ln is not None]
    band_cont = rasterize_lane_band(lines, dash_gate=False)
    band_dash = rasterize_lane_band(lines, dash_gate=True)
    # dash gate can only remove pixels (subset), and removes some if any line modeled a dash
    assert band_dash.sum() <= band_cont.sum()


# ---------------------------------------------------------------------------
# inject
# ---------------------------------------------------------------------------
def test_inject_replace():
    phi = np.zeros((8, 8, 5), np.float32)
    phi1 = np.full((8, 8), 3.0, np.float32)
    out = inject_lane_sdf(phi, phi1, lane_cls=1, mode="replace")
    assert np.allclose(out[..., 1], 3.0)
    assert np.allclose(out[..., 0], 0.0)  # other classes untouched


def test_inject_bias_adds():
    phi = np.ones((8, 8, 5), np.float32)
    phi1 = np.full((8, 8), 2.0, np.float32)
    out = inject_lane_sdf(phi, phi1, lane_cls=1, mode="bias", bias_scale=0.5)
    assert np.allclose(out[..., 1], 1.0 + 1.0)  # 1 + 0.5*2
    assert np.allclose(out[..., 0], 1.0)


def test_inject_bad_mode_raises():
    with pytest.raises(ValueError):
        inject_lane_sdf(np.zeros((4, 4, 5), np.float32), np.zeros((4, 4), np.float32), mode="nope")


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------
def test_decompose_perfect_is_zero():
    L = np.array([[0, 1], [1, 0]])
    d = decompose_argmax_disagreement(L.copy(), L, lane_cls=1, road_cls=0)
    assert isinstance(d, ContainmentDecomp)
    assert d.total_dseg == 0.0 and d.class0_dseg == 0.0 and d.lane_attributable == 0.0


def test_decompose_road_to_lane_is_containment_leak():
    L = np.array([[0, 0], [0, 0]])           # all road
    pred = np.array([[1, 0], [0, 0]])         # one road px flipped to lane
    d = decompose_argmax_disagreement(pred, L, lane_cls=1, road_cls=0)
    assert d.lane_fp_from_road == pytest.approx(0.25)
    assert d.class0_dseg == pytest.approx(0.25)
    assert d.lane_fn == 0.0


def test_decompose_lane_fn():
    L = np.array([[1, 1], [1, 1]])           # all lane
    pred = np.array([[0, 1], [1, 1]])         # one lane px missed
    d = decompose_argmax_disagreement(pred, L, lane_cls=1, road_cls=0)
    assert d.lane_fn == pytest.approx(0.25)
    assert d.lane_fp_from_road == 0.0


def test_decompose_ideal_sdf_baseline_zero():
    """Sanity that the ideal-SDF argmax reproduces L* exactly (containment baseline 0)."""
    from tac.boundary_math.lever_b_levelset_generator import signed_distance_fields
    lab = _planted_lane_labels(n_lines=2)
    phi = signed_distance_fields(lab, 5)
    pred = phi.argmax(-1)
    d = decompose_argmax_disagreement(pred, lab, lane_cls=1, road_cls=0)
    assert d.total_dseg == 0.0


def test_injected_continuous_band_contains_road():
    """Decisive behavior: inject continuous structured lane SDF into ideal fields ->
    road (class-0) stays mostly intact (containment), lane mostly recovered."""
    from tac.boundary_math.lever_b_levelset_generator import signed_distance_fields
    lab = _planted_lane_labels(n_lines=2)  # solid (non-dashed) -> no dash-gap FP
    phi_ideal = signed_distance_fields(lab, 5)
    phi1, _ = build_structured_lane_sdf(lab, lane_cls=1, dash_gate=False)
    pred = inject_lane_sdf(phi_ideal, phi1, lane_cls=1, mode="replace").argmax(-1)
    d = decompose_argmax_disagreement(pred, lab, lane_cls=1, road_cls=0)
    assert d.class0_dseg < 1e-3, f"containment failed: class0 {d.class0_dseg}"
    assert d.lane_fn < 1e-3, f"shape FN too high: {d.lane_fn}"
