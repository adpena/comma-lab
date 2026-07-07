# SPDX-License-Identifier: MIT
"""Tests for the #287 max-plus dash-comb corrector (``tac.boundary_math.dash_comb``).

BEHAVIOR-verifying (NO-FAKE class #2): each test would FAIL if the function body were
replaced by ``return constant`` — they assert the gate is actually periodic, actually
ego-transported (world invariance), the fit actually recovers a known synthetic
scale/phase, the combed raster actually gates dashed lines (and ONLY dashed lines,
ONLY in the near field), and the MLX twin matches the numpy fp32 authority.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.lane_sdf_component import LaneLine, _CAM_H, _FY, _V_HORIZON
from tac.boundary_math.dash_comb import (
    DashCombFit,
    build_combed_lane_band_priors,
    comb_gate_of_world,
    comb_row_gate,
    ego_cumulative_distance,
    fit_dash_comb,
    line_slot,
    rasterize_lane_coverage_combed,
)

try:
    import mlx.core as mx  # noqa: F401

    _HAS_MLX = True
except Exception:  # pragma: no cover
    _HAS_MLX = False

_H, _W = 384, 512


def _line(lat: float = 0.0, hw: float = 3.0, dash_period: float = 12.0,
          dash_phase: float = 2.0, forward_range=(5.0, 55.0)) -> LaneLine:
    return LaneLine(
        centerline_coeffs=np.array([0.0, float(lat)]),
        halfwidth_coeffs=np.array([0.0, float(hw)]),
        dash_period_m=float(dash_period),
        dash_phase_m=float(dash_phase),
        dash_duty=0.5,
        forward_range=tuple(forward_range),
        n_pixels=500,
    )


def _fit(scale: float = 0.0, period: float = 12.0, duty: float = 0.5,
         phase0: dict[int, float] | None = None) -> DashCombFit:
    p0 = dict(phase0 or {0: 0.0})
    return DashCombFit(
        scale=scale, period_m=period, duty=duty,
        phase0_by_slot=p0,
        period_by_slot={s: period for s in p0},
        duty_by_slot={s: duty for s in p0},
        transported_by_slot={s: True for s in p0},
        anchor_window=24,
        anchor_phase0_by_slot={},
        concentration_by_slot={0: 1.0},
        anchored_concentration_by_slot={0: 1.0},
        pairwise_concentration_by_slot={0: 1.0},
        mean_concentration=1.0, mean_pairwise_concentration=1.0,
        concentration_at_zero_scale=0.0, n_dashed_fits=100, n_pairs=10,
        slot_width_m=1.8, provenance={"global_phase0_m": 0.0},
    )


# ---------------------------------------------------------------- comb gate
def test_gate_hard_periodicity_and_duty():
    T, duty = 10.0, 0.5
    w = np.linspace(0.0, 40.0, 4001)
    g = comb_gate_of_world(w, period_m=T, duty=duty, phase0_m=0.0)
    assert g.dtype == np.float32
    assert set(np.unique(g)).issubset({0.0, 1.0})
    # ON fraction ~ duty
    assert abs(float(g.mean()) - duty) < 0.02
    # ON exactly where mod(w, T) < duty*T
    expect = (np.mod(w, T) < duty * T).astype(np.float32)
    assert np.array_equal(g, expect)


def test_gate_phase0_shifts_pattern():
    T = 10.0
    w = np.linspace(0.0, 40.0, 401)
    g0 = comb_gate_of_world(w, period_m=T, duty=0.5, phase0_m=0.0)
    g3 = comb_gate_of_world(w, period_m=T, duty=0.5, phase0_m=3.0)
    assert not np.array_equal(g0, g3)
    # shifting w by phase0 recovers the unshifted pattern
    gshift = comb_gate_of_world(w + 3.0, period_m=T, duty=0.5, phase0_m=3.0)
    assert np.array_equal(g0, gshift)


def test_gate_soft_ramp_bounded_and_saturates():
    T = 10.0
    w = np.linspace(0.0, 20.0, 2001)
    g = comb_gate_of_world(w, period_m=T, duty=0.5, phase0_m=0.0, softness_m=0.5)
    assert float(g.min()) == 0.0 and float(g.max()) == 1.0
    assert np.all((g >= 0.0) & (g <= 1.0))
    # strictly between 0 and 1 somewhere near the edges (a real ramp exists)
    assert np.any((g > 0.0) & (g < 1.0))
    # deep inside the ON cell it saturates to 1; deep in the gap to 0
    assert comb_gate_of_world(np.array([2.5]), period_m=T, duty=0.5, phase0_m=0.0,
                              softness_m=0.5)[0] == 1.0
    assert comb_gate_of_world(np.array([7.5]), period_m=T, duty=0.5, phase0_m=0.0,
                              softness_m=0.5)[0] == 0.0


def test_gate_world_invariance_ego_transport():
    """gate(f, E) == gate(f - d, E + d): a world-static dash pattern under ego motion."""
    T = 12.0
    f = np.linspace(5.0, 50.0, 901)
    d = 7.3
    g_a = comb_gate_of_world(f + 100.0, period_m=T, duty=0.4, phase0_m=1.0)
    g_b = comb_gate_of_world((f - d) + (100.0 + d), period_m=T, duty=0.4, phase0_m=1.0)
    assert np.array_equal(g_a, g_b)


def test_gate_rejects_bad_params():
    with pytest.raises(ValueError):
        comb_gate_of_world(np.zeros(3), period_m=0.0, duty=0.5, phase0_m=0.0)
    with pytest.raises(ValueError):
        comb_gate_of_world(np.zeros(3), period_m=10.0, duty=1.0, phase0_m=0.0)


@pytest.mark.skipif(not _HAS_MLX, reason="mlx unavailable")
def test_gate_mlx_parity_hard_and_soft():
    """MLX (fp32) vs numpy parity. The hard gate is a threshold function, so samples
    landing within fp32 rounding of a cell boundary may legitimately flip between the
    fp64 numpy path and fp32 MLX; parity is asserted away from that measure-zero set
    (numpy fp32 reference stays the bit-identical authority)."""
    import mlx.core as mx

    T, duty, ph = 11.0, 0.45, 2.2
    w = np.linspace(-25.0, 60.0, 3001).astype(np.float32)
    u = np.mod(w.astype(np.float64) - ph, T)
    edge_dist = np.minimum.reduce([u, np.abs(u - duty * T), T - u])
    interior = edge_dist > 1e-3
    assert interior.sum() > 2900  # the exclusion is measure-zero, not a loophole
    for soft in (0.0, 0.4):
        g_np = comb_gate_of_world(w, period_m=T, duty=duty, phase0_m=ph, softness_m=soft)
        g_mx = np.asarray(
            comb_gate_of_world_mlx_helper(mx.array(w), T, duty, ph, soft))
        assert g_np.shape == g_mx.shape
        assert float(np.max(np.abs(g_np[interior] - g_mx[interior]))) < 1e-5


def comb_gate_of_world_mlx_helper(w, T, duty, ph, soft):
    from tac.boundary_math.dash_comb import comb_gate_of_world_mlx

    return comb_gate_of_world_mlx(w, period_m=T, duty=duty, phase0_m=ph, softness_m=soft)


# ---------------------------------------------------------------- ego distance
def test_ego_cumulative_distance_scales_cumsum():
    raw = np.array([0.0, 1.0, 1.0, 2.0])
    E = ego_cumulative_distance(raw, 2.0)
    assert np.allclose(E, [0.0, 2.0, 4.0, 8.0])


# ---------------------------------------------------------------- slots
def test_line_slot_separates_left_right():
    left = _line(lat=-1.9)
    right = _line(lat=1.9)
    center = _line(lat=0.0)
    assert line_slot(left) != line_slot(right)
    assert line_slot(center) == 0


# ---------------------------------------------------------------- fit
def test_fit_recovers_synthetic_scale_and_phase():
    """Synthetic world-static dashes: phase_k = (w0 - E_k) mod T with a known ego scale.
    The fit must recover the scale (concentration ~1) and per-slot w0."""
    rng = np.random.default_rng(0)  # only to vary fwd_raw magnitudes; fit itself is deterministic
    P = 200
    T, duty, w0, a_true = 12.0, 0.5, 4.0, 0.05
    fwd_raw = 20.0 + 2.0 * rng.standard_normal(P)  # raw PoseNet-like channel
    E = a_true * np.cumsum(fwd_raw)
    per_pair = []
    for k in range(P):
        ph = float(np.mod(w0 - E[k], T))
        per_pair.append([_line(lat=0.0, dash_period=T, dash_phase=ph)])
    fit = fit_dash_comb(per_pair, fwd_raw, min_slot_count=20)
    assert fit.n_dashed_fits == P
    assert fit.mean_concentration > 0.98
    assert fit.mean_concentration > fit.concentration_at_zero_scale + 0.2
    assert abs(fit.period_m - T) < 1e-9
    assert abs(fit.duty - duty) < 1e-9
    slot = line_slot(per_pair[0][0])
    # w0 recovered modulo T
    dw = abs(fit.phase0_by_slot[slot] - w0)
    assert min(dw, T - dw) < 0.5
    # ego scale recovered (grid resolution tolerance)
    assert abs(fit.scale - a_true) / a_true < 0.05


def test_fit_two_slots_with_different_periods():
    """Per-slot cell parameters: two line families with DIFFERENT true periods must
    both be recovered (one global period would destroy the statistic — the n600 bug)."""
    rng = np.random.default_rng(2)
    P = 200
    a_true = 0.05
    fwd_raw = 20.0 + 2.0 * rng.standard_normal(P)
    E = a_true * np.cumsum(fwd_raw)
    Ta, Tb, w0a, w0b = 7.6, 12.2, 2.0, 5.0
    per_pair = []
    for k in range(P):
        per_pair.append([
            _line(lat=-1.9, dash_period=Ta, dash_phase=float(np.mod(w0a - E[k], Ta))),
            _line(lat=5.4, dash_period=Tb, dash_phase=float(np.mod(w0b - E[k], Tb))),
        ])
    fit = fit_dash_comb(per_pair, fwd_raw, min_slot_count=20)
    sa = line_slot(per_pair[0][0])
    sb = line_slot(per_pair[0][1])
    assert abs(fit.period_by_slot[sa] - Ta) < 1e-9
    assert abs(fit.period_by_slot[sb] - Tb) < 1e-9
    assert fit.mean_concentration > 0.95
    assert abs(fit.scale - a_true) / a_true < 0.05
    for s, w0, T in ((sa, w0a, Ta), (sb, w0b, Tb)):
        dw = abs(fit.phase0_by_slot[s] - w0)
        assert min(dw, T - dw) < 0.5
    # params_for routes each line to its own slot's cell parameters
    Tsa, _, _, tra = fit.params_for(per_pair[0][0])
    Tsb, _, _, trb = fit.params_for(per_pair[0][1])
    assert (Tsa, Tsb) == (Ta, Tb)
    assert tra and trb  # world-static synthetic dashes => both slots ego-transported


def test_fit_zero_transport_null_has_low_concentration():
    """Random per-pair phases (NOT ego-transported): concentration must be LOW —
    the fit reports the transport claim refuted instead of forcing a positive."""
    rng = np.random.default_rng(1)
    P = 200
    fwd_raw = np.full(P, 20.0)
    per_pair = [[_line(dash_period=12.0, dash_phase=float(rng.uniform(0, 12.0)))]
                for _ in range(P)]
    fit = fit_dash_comb(per_pair, fwd_raw, min_slot_count=20)
    assert fit.mean_concentration < 0.5


def test_fit_raises_on_too_few_dashed_lines():
    per_pair = [[_line(dash_period=0.0)] for _ in range(50)]  # all solid
    with pytest.raises(ValueError):
        fit_dash_comb(per_pair, np.full(50, 20.0))


# ---------------------------------------------------------------- row gate + raster
def test_comb_row_gate_far_field_ungated():
    rows = np.arange(_H, dtype=np.float64)
    below = rows > (_V_HORIZON + 1.0)
    vr = rows[below]
    g = comb_row_gate(vr, ego_dist_m=0.0, period_m=12.0, duty=0.5, phase0_m=0.0,
                      forward_max_m=55.0)
    forward = _CAM_H * _FY / np.maximum(vr - _V_HORIZON, 1e-3)
    far = forward >= 55.0
    assert np.all(g[far] == 1.0)
    near = forward < 55.0
    assert np.any(g[near] == 0.0) and np.any(g[near] == 1.0)


def test_combed_raster_gates_dashed_line_near_field_only():
    ln = _line(dash_period=12.0)
    fit = _fit(phase0={line_slot(ln): 0.0})
    cov = rasterize_lane_coverage_combed([ln], fit, 0.0, h=_H, w=_W)
    # compare against the ungated solid coverage of the same geometry
    from tac.boundary_math.analytic_lane_render_band import (
        rasterize_lane_coverage_range_dependent,
    )
    solid = rasterize_lane_coverage_range_dependent([ln], h=_H, w=_W, dash_gate=False)
    assert cov.shape == solid.shape
    assert np.all(cov <= solid + 1e-6)
    # some rows fully removed (gaps), some kept (dashes) in the near field
    rows = np.arange(_H, dtype=np.float64)
    forward = np.where(rows > _V_HORIZON + 1.0,
                       _CAM_H * _FY / np.maximum(rows - _V_HORIZON, 1e-3), np.inf)
    near = (forward < 55.0) & (solid.max(axis=1) > 0.5)
    assert near.any()
    kept = cov.max(axis=1)[near] > 0.5
    assert kept.any() and (~kept).any()
    # far field untouched
    far = (forward >= 55.0) & np.isfinite(forward) & (solid.max(axis=1) > 0.0)
    if far.any():
        assert np.allclose(cov[far], solid[far])


def test_combed_raster_leaves_solid_lines_untouched():
    ln = _line(dash_period=0.0)  # solid line: comb must NOT gate it
    fit = _fit()
    cov = rasterize_lane_coverage_combed([ln], fit, 123.0, h=_H, w=_W)
    from tac.boundary_math.analytic_lane_render_band import (
        rasterize_lane_coverage_range_dependent,
    )
    solid = rasterize_lane_coverage_range_dependent([ln], h=_H, w=_W, dash_gate=False)
    assert np.allclose(cov, solid)


def test_combed_raster_ego_distance_moves_the_gaps():
    ln = _line(dash_period=12.0)
    fit = _fit(phase0={line_slot(ln): 0.0})
    cov_a = rasterize_lane_coverage_combed([ln], fit, 0.0, h=_H, w=_W)
    cov_b = rasterize_lane_coverage_combed([ln], fit, 6.0, h=_H, w=_W)  # half period
    assert not np.allclose(cov_a, cov_b)


# ---------------------------------------------------------------- end-to-end builder
def _synthetic_dashed_lstar(ego_dist: float, *, T: float = 12.0, duty: float = 0.5,
                            w0: float = 0.0) -> np.ndarray:
    """A synthetic 384x512 lstar with a world-static dashed straight-ahead lane."""
    a = np.zeros((_H, _W), np.int64)
    rows = np.arange(_H, dtype=np.float64)
    forward = np.where(rows > _V_HORIZON + 1.0,
                       _CAM_H * _FY / np.maximum(rows - _V_HORIZON, 1e-3), np.inf)
    on = np.mod(ego_dist + forward - w0, T) < duty * T
    for v in range(_H):
        if np.isfinite(forward[v]) and 5.0 <= forward[v] <= 50.0 and on[v]:
            a[v, _W // 2 - 2:_W // 2 + 3] = 1
    return a


def test_build_combed_priors_end_to_end_synthetic():
    P = 24
    ds = 3.0  # meters per pair, constant speed
    lstars = np.stack([_synthetic_dashed_lstar(ds * k) for k in range(P)])
    gt_poses = np.zeros((P, 6), np.float64)
    gt_poses[:, 0] = 1.0  # raw forward channel; true scale = ds
    priors, fit = build_combed_lane_band_priors(
        lstars, gt_poses, comb_softness_m=0.0)
    assert len(priors) == P
    assert fit.n_pairs == P
    assert fit.n_dashed_fits >= P // 2  # most pairs fit as dashed
    # ego transport must beat the no-transport null on world-static synthetic dashes
    assert fit.mean_concentration > fit.concentration_at_zero_scale
    assert fit.mean_concentration > 0.7
    cov = priors[0].coverage
    assert cov.shape == (_H, _W)
    assert float(cov.max()) > 0.5
    assert np.isfinite(priors[0].band_recall)


def test_build_combed_priors_shape_mismatch_raises():
    with pytest.raises(ValueError):
        build_combed_lane_band_priors(np.zeros((3, _H, _W), np.int64),
                                      np.zeros((4, 6), np.float64))
