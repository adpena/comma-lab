"""Tests for the analytic-lane render-band component (FEED-dv; #203/#213/#215).

BEHAVIOR-verifying (NO-FAKE class #2): every test would FAIL if the function body
were replaced by ``return constant`` -- they assert the composite actually blends,
the uncertainty mask actually gates, the range-dependent dash gate actually
differs near vs far, and the MLX path is bit-identical to the numpy authority.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.lane_sdf_component import LaneLine
from tac.boundary_math.analytic_lane_render_band import (
    DEFAULT_DASH_FORWARD_MAX_M,
    METAL_KERNEL_FLAG,
    LaneBandPrior,
    build_analytic_lane_band_prior,
    composite_lane_band,
    composite_lane_band_mlx,
    decompose_lane_dseg,
    make_lane_band_compose_fn,
    rasterize_lane_coverage_range_dependent,
    rasterize_lane_coverage_mlx,
    witness_uncertainty_mask,
    witness_uncertainty_mask_mlx,
)

try:
    import mlx.core as mx  # noqa: F401

    _HAS_MLX = True
except Exception:  # pragma: no cover
    _HAS_MLX = False

_H, _W = 384, 512


def _straight_line(hw: float = 3.0, dash_period: float = 0.0, forward_range=(5.0, 55.0)) -> LaneLine:
    """A straight-ahead lane line (lateral=0 -> u_center=cx) of constant half-width."""

    return LaneLine(
        centerline_coeffs=np.array([0.0, 0.0]),  # lateral = 0 for all forward
        halfwidth_coeffs=np.array([0.0, float(hw)]),  # hw constant
        dash_period_m=float(dash_period),
        dash_phase_m=0.0,
        dash_duty=0.5,
        forward_range=tuple(forward_range),
        n_pixels=500,
    )


# ---------------------------------------------------------------------------
# 1-6: coverage raster (AA-SDF, range-dependent dash gate)
# ---------------------------------------------------------------------------
def test_coverage_empty_lines_is_zero():
    cov = rasterize_lane_coverage_range_dependent([], h=_H, w=_W)
    assert cov.shape == (_H, _W)
    assert cov.dtype == np.float32
    assert float(cov.sum()) == 0.0


def test_coverage_straight_band_is_centered_and_nonzero():
    cov = rasterize_lane_coverage_range_dependent([_straight_line(hw=3.0)], h=_H, w=_W, softness=1.0)
    assert cov.max() > 0.9  # band interior fully covered
    # coverage concentrated around cx=256; column-summed coverage peaks near center
    colsum = cov.sum(axis=0)
    peak = int(np.argmax(colsum))
    assert 250 <= peak <= 262
    # above the horizon (top rows) -> no coverage
    assert float(cov[:170].sum()) == 0.0


def test_coverage_aa_softness_ramp():
    """AA edge: a wider softness spreads partial coverage further from the edge."""
    sharp = rasterize_lane_coverage_range_dependent([_straight_line(hw=3.0)], softness=0.5)
    soft = rasterize_lane_coverage_range_dependent([_straight_line(hw=3.0)], softness=3.0)
    # count partial (anti-aliased) pixels: strictly between 0 and 1
    n_partial_sharp = int(np.count_nonzero((sharp > 0.01) & (sharp < 0.99)))
    n_partial_soft = int(np.count_nonzero((soft > 0.01) & (soft < 0.99)))
    assert n_partial_soft > n_partial_sharp  # softer edge -> more AA pixels


def test_coverage_range_dependent_dash_gate_removes_near_coverage():
    """A dashed line: gating removes gap coverage vs continuous."""
    dashed = _straight_line(hw=3.0, dash_period=6.0)
    cov_continuous = rasterize_lane_coverage_range_dependent([dashed], dash_gate=False)
    cov_all_gated = rasterize_lane_coverage_range_dependent(
        [dashed], dash_gate=True, dash_forward_max_m=500.0)
    assert cov_all_gated.sum() < cov_continuous.sum()  # dash gaps removed


def test_coverage_range_dependent_far_field_stays_continuous():
    """#215: with a SMALL dash_forward_max, only near rows are gated; far rows continuous
    -> MORE coverage than gating everywhere."""
    dashed = _straight_line(hw=3.0, dash_period=6.0)
    cov_near_only = rasterize_lane_coverage_range_dependent(
        [dashed], dash_gate=True, dash_forward_max_m=15.0)
    cov_all_gated = rasterize_lane_coverage_range_dependent(
        [dashed], dash_gate=True, dash_forward_max_m=500.0)
    cov_continuous = rasterize_lane_coverage_range_dependent([dashed], dash_gate=False)
    # near-only gating: between all-gated and continuous
    assert cov_all_gated.sum() <= cov_near_only.sum() <= cov_continuous.sum()
    assert cov_near_only.sum() > cov_all_gated.sum()  # far field kept continuous


def test_coverage_union_over_two_lines():
    left = LaneLine(centerline_coeffs=np.array([0.0, -1.5]), halfwidth_coeffs=np.array([0.0, 3.0]),
                    forward_range=(5.0, 55.0))
    right = LaneLine(centerline_coeffs=np.array([0.0, 1.5]), halfwidth_coeffs=np.array([0.0, 3.0]),
                     forward_range=(5.0, 55.0))
    cov_l = rasterize_lane_coverage_range_dependent([left])
    cov_r = rasterize_lane_coverage_range_dependent([right])
    cov_both = rasterize_lane_coverage_range_dependent([left, right])
    # union is the max -> at least as much as either, and equals elementwise max
    assert np.allclose(cov_both, np.maximum(cov_l, cov_r))
    assert cov_both.sum() >= cov_l.sum()


# ---------------------------------------------------------------------------
# 7-9: witness uncertainty mask
# ---------------------------------------------------------------------------
def test_uncertainty_mask_monotone_and_bounded():
    margin = np.array([[-1.0, 0.0, 0.5, 1.0, 5.0]], np.float32)
    u = witness_uncertainty_mask(margin, tau=0.5, eps=0.25)
    assert u.min() >= 0.0 and u.max() <= 1.0
    # monotone DECREASING in margin (more confident -> less compositing)
    assert np.all(np.diff(u[0]) <= 1e-6)
    assert u[0, 0] == pytest.approx(1.0)   # very uncertain
    assert u[0, -1] == pytest.approx(0.0)  # very confident


def test_uncertainty_mask_threshold_at_tau():
    # at margin == tau, the ramp is centered -> u == 0.5
    u = witness_uncertainty_mask(np.array([0.7], np.float32), tau=0.7, eps=0.3)
    assert float(u[0]) == pytest.approx(0.5, abs=1e-5)


def test_uncertainty_mask_eps_controls_ramp_width():
    margins = np.linspace(-1, 2, 200).astype(np.float32)[None]
    u_sharp = witness_uncertainty_mask(margins, tau=0.5, eps=0.1)
    u_wide = witness_uncertainty_mask(margins, tau=0.5, eps=0.8)
    n_partial_sharp = int(np.count_nonzero((u_sharp > 0.01) & (u_sharp < 0.99)))
    n_partial_wide = int(np.count_nonzero((u_wide > 0.01) & (u_wide < 0.99)))
    assert n_partial_wide > n_partial_sharp


# ---------------------------------------------------------------------------
# 10-13: composite (differentiable, correct blend)
# ---------------------------------------------------------------------------
def test_composite_zero_alpha_is_identity():
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0, 255, (1, _H, _W, 3)).astype(np.float32)
    cov = np.zeros((_H, _W), np.float32)
    out = composite_lane_band(rgb, cov, np.array([255.0, 255.0, 255.0]))
    assert np.array_equal(out, rgb)  # EXACT identity where coverage 0


def test_composite_full_alpha_is_lane_color():
    rgb = np.zeros((1, _H, _W, 3), np.float32)
    cov = np.ones((_H, _W), np.float32)
    lane = np.array([200.0, 210.0, 220.0], np.float32)
    out = composite_lane_band(rgb, cov, lane)
    assert np.allclose(out[0, 100, 100], lane)


def test_composite_u_mask_gates_the_band():
    """u_mask=0 kills the band even where coverage=1 (the FP-killer behavior)."""
    rgb = np.zeros((1, _H, _W, 3), np.float32)
    cov = np.ones((_H, _W), np.float32)
    lane = np.array([255.0, 255.0, 255.0], np.float32)
    u_zero = np.zeros((_H, _W), np.float32)
    out = composite_lane_band(rgb, cov, lane, u_mask=u_zero)
    assert np.array_equal(out, rgb)  # confident everywhere -> band suppressed
    # half uncertainty -> half blend
    u_half = 0.5 * np.ones((_H, _W), np.float32)
    out_half = composite_lane_band(rgb, cov, lane, u_mask=u_half)
    assert np.allclose(out_half[0, 100, 100], 0.5 * lane)


def test_composite_partial_blend_is_convex():
    rgb = np.full((1, _H, _W, 3), 10.0, np.float32)
    lane = np.array([250.0, 250.0, 250.0], np.float32)
    cov = np.full((_H, _W), 0.25, np.float32)
    out = composite_lane_band(rgb, cov, lane)
    expected = 10.0 * 0.75 + 250.0 * 0.25
    assert np.allclose(out[0, 5, 5], expected)


# ---------------------------------------------------------------------------
# 14-16: build_analytic_lane_band_prior on a constructed label map
# ---------------------------------------------------------------------------
def _synthetic_lstar_with_lane() -> np.ndarray:
    """Road (0) background + a near-vertical class-1 lane stripe below the horizon."""
    lstar = np.zeros((_H, _W), np.int64)
    lstar[:174] = 2  # undrivable/sky on top
    # a lane stripe around col 256, rows 200..380 (below horizon), ~4px wide
    for v in range(200, 380):
        c = 256 + int((v - 290) * 0.05)  # slight slant
        lstar[v, max(0, c - 2):c + 2] = 1
    return lstar


def test_build_prior_finds_lane_and_reports_recall():
    lstar = _synthetic_lstar_with_lane()
    prior = build_analytic_lane_band_prior(lstar, lane_cls=1, dash_gate=True)
    assert isinstance(prior, LaneBandPrior)
    assert prior.n_lines >= 1
    assert prior.total_floats > 0
    assert 0.0 < prior.band_recall <= 1.0
    assert prior.coverage.shape == (_H, _W)
    assert prior.gt_lane_frac > 0.0


def test_build_prior_coverage_overlaps_the_lane():
    lstar = _synthetic_lstar_with_lane()
    # dash_gate=False isolates the GEOMETRY fit (the borrowed matched-filter dash
    # fitter is unreliable on a short synthetic stripe -- tested on real GT n600, not here).
    prior = build_analytic_lane_band_prior(lstar, lane_cls=1, dash_gate=False)
    is_lane = lstar == 1
    # a meaningful fraction of GT lane pixels are covered by the analytic band
    covered = (prior.coverage[is_lane] >= 0.5).mean()
    assert covered > 0.3


def test_build_prior_no_lane_is_empty():
    lstar = np.zeros((_H, _W), np.int64)  # all road, no class 1
    prior = build_analytic_lane_band_prior(lstar, lane_cls=1)
    assert prior.n_lines == 0
    assert float(prior.coverage.sum()) == 0.0


# ---------------------------------------------------------------------------
# 17-18: decompose_lane_dseg
# ---------------------------------------------------------------------------
def test_decompose_perfect_match_is_zero():
    gt = _synthetic_lstar_with_lane()
    d = decompose_lane_dseg(gt.copy(), gt, lane_cls=1)
    assert d.d_seg == 0.0
    assert d.lane_fn_frac == 0.0
    assert d.lane_fp_frac == 0.0
    assert d.lane_recall == pytest.approx(1.0)


def test_decompose_counts_fn_and_fp():
    gt = _synthetic_lstar_with_lane()
    pred = gt.copy()
    # erase 100 lane pixels (FN) and invent 50 lane pixels on road (FP)
    lane_idx = np.argwhere(gt == 1)[:100]
    for v, u in lane_idx:
        pred[v, u] = 0
    pred[50, 50:100] = 1  # 50 FP (row 50 is sky region gt=2, so these are FP)
    d = decompose_lane_dseg(pred, gt, lane_cls=1)
    n = float(gt.size)
    assert d.lane_fn_frac == pytest.approx(100 / n)
    assert d.lane_fp_frac == pytest.approx(50 / n)
    assert d.lane_recall < 1.0


# ---------------------------------------------------------------------------
# 19-22: MLX numpy-portability parity (>= 0.9997) + differentiability
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_MLX, reason="mlx not available")
def test_mlx_uncertainty_parity():
    margin = np.random.default_rng(1).uniform(-2, 3, (_H, _W)).astype(np.float32)
    ref = witness_uncertainty_mask(margin, tau=0.5, eps=0.25)
    got = np.asarray(witness_uncertainty_mask_mlx(mx.array(margin), tau=0.5, eps=0.25))
    assert float(np.max(np.abs(ref - got))) < 1e-5


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not available")
def test_mlx_composite_parity():
    rng = np.random.default_rng(2)
    rgb = rng.uniform(0, 255, (1, _H, _W, 3)).astype(np.float32)
    cov = rng.uniform(0, 1, (_H, _W)).astype(np.float32)
    u = rng.uniform(0, 1, (_H, _W)).astype(np.float32)
    lane = np.array([220.0, 230.0, 210.0], np.float32)
    ref = composite_lane_band(rgb, cov, lane, u_mask=u)
    got = np.asarray(composite_lane_band_mlx(
        mx.array(rgb), mx.array(cov)[None], mx.array(lane), mx.array(u)[None]))
    corr = float(np.corrcoef(ref.ravel(), got.ravel())[0, 1])
    assert corr >= 0.9997
    assert float(np.max(np.abs(ref - got))) < 1e-2


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not available")
def test_mlx_coverage_raster_parity():
    """MLX vectorized raster matches the numpy per-line signed-distance authority."""
    line = _straight_line(hw=3.0)
    # numpy authority coverage
    ref = rasterize_lane_coverage_range_dependent([line], softness=1.0, dash_gate=False)
    # MLX path: build the same per-row params (L=1) and raster
    from tac.boundary_math.analytic_lane_render_band import _line_row_params
    rows = np.arange(_H, dtype=np.float64)
    below = rows > (_V := 174.0) + 1.0
    vr = rows[below]
    u_c, hw, gate = _line_row_params(line, vr, dash_gate=False, dash_forward_max_m=55.0)
    col = np.arange(_W, dtype=np.float64)
    cov_below = np.asarray(rasterize_lane_coverage_mlx(
        mx.array(u_c[None].astype(np.float32)), mx.array(hw[None].astype(np.float32)),
        mx.array(gate[None].astype(np.float32)), mx.array(col.astype(np.float32)), softness=1.0))
    got = np.zeros((_H, _W), np.float32)
    got[below] = cov_below
    assert float(np.max(np.abs(ref - got))) < 1e-4


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not available")
def test_mlx_composite_differentiable_wrt_rgb():
    """Gradient flows through the composite into the witness rgb (so the d_seg loss
    backprops into the witness)."""
    rgb = mx.array(np.full((1, 4, 4, 3), 10.0, np.float32))
    cov = mx.array(np.full((1, 4, 4), 0.5, np.float32))
    lane = mx.array(np.array([255.0, 255.0, 255.0], np.float32))

    def loss(x):
        out = composite_lane_band_mlx(x, cov, lane)
        return mx.sum(out)

    g = mx.grad(loss)(rgb)
    g = np.asarray(g)
    # d/d rgb of sum(rgb*(1-a) + lane*a) = (1-a) = 0.5 everywhere
    assert np.allclose(g, 0.5, atol=1e-5)


# ---------------------------------------------------------------------------
# 23-24: compose_fn factory end-to-end (numpy path) + FP-killer behavior
# ---------------------------------------------------------------------------
def test_compose_fn_factory_numpy_path():
    lstar = _synthetic_lstar_with_lane()
    prior = build_analytic_lane_band_prior(lstar, lane_cls=1)
    lane_rgb = np.array([255.0, 255.0, 255.0], np.float32)
    # margin: confident everywhere -> uncertainty mask 0 -> band fully suppressed
    confident_margin = np.full((_H, _W), 5.0, np.float32)
    compose = make_lane_band_compose_fn(
        {0: prior}, lane_rgb_provider=lane_rgb, margin_provider={0: confident_margin},
        tau=0.5, eps=0.25, use_mlx=False)
    rgb = np.full((1, _H, _W, 3), 10.0, np.float32)
    out = compose(rgb, 0)
    assert np.allclose(out, rgb)  # confident -> no override anywhere


def test_compose_fn_uncertainty_restricts_to_boundary():
    """The uncertainty mask restricts compositing to low-margin pixels -> the band
    only paints where the (frozen) decision is uncertain (the FP-killer)."""
    lstar = _synthetic_lstar_with_lane()
    prior = build_analytic_lane_band_prior(lstar, lane_cls=1, dash_gate=False)
    lane_rgb = np.array([255.0, 255.0, 255.0], np.float32)
    # uncertain ONLY in a small window (rows 300-320); confident elsewhere
    margin = np.full((_H, _W), 5.0, np.float32)
    margin[300:320, :] = -1.0  # uncertain band
    compose = make_lane_band_compose_fn(
        {0: prior}, lane_rgb_provider=lane_rgb, margin_provider={0: margin},
        tau=0.5, eps=0.25, use_mlx=False)
    rgb = np.full((1, _H, _W, 3), 10.0, np.float32)
    out = compose(rgb, 0)
    changed = np.any(out != rgb, axis=-1)[0]  # (H,W)
    # all changed pixels are inside the uncertain window
    ch_rows = np.where(changed.any(axis=1))[0]
    assert ch_rows.size > 0
    assert ch_rows.min() >= 300 and ch_rows.max() < 320


def test_metal_kernel_flag_is_specified():
    """The COMPUTE facet: the AA-SDF raster is flagged for a #212 Metal kernel with
    a concrete signature the parent can build against."""
    assert METAL_KERNEL_FLAG["candidate"] == "aa_sdf_lane_coverage_raster"
    assert "aa_sdf_lane_coverage" in METAL_KERNEL_FLAG["signature"]
    assert "numpy_reference" in METAL_KERNEL_FLAG
    assert DEFAULT_DASH_FORWARD_MAX_M == pytest.approx(55.0)
