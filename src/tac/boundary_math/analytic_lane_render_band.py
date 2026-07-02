"""Analytic-lane RENDER-BAND compositing component (FEED-dv; tasks #203/#213/#215).

The PRIMARY d_seg lane lever, in its NON-NAIVE form. Sister of
``tac.boundary_math.lane_sdf_component`` (which builds the phi_1 SDF for the
softmax-of-SDF level-set); THIS module composites an analytic lane band OVER a
RENDER (the witness bulk) BEFORE the contest R operator, so the frozen SegNet
reads the composited frame. It is designed to plug into the ``compose_fn`` hook
in ``experiments/train_witness_realized_through_R_mlx.py::render_through_R_mlx``
(``compose_fn(rgb_nhwc, code_idx) -> rgb_nhwc``) and its levelset sibling.

WHY THE NAIVE BAND HURT (MEASURED, ``tools/levelset_render_side_sizing_l7best_n600.py``
n600, l7-best levelset ckpt, [macOS-CPU advisory]):
  * witness-alone (c1): d_seg 0.00333, lane_recall 0.758, lane_fp 0.00060.
  * naive full-coverage band (c3): d_seg 0.00415 (+25%), lane_fp 0.00115 (~2x).
  * The naive band lays lane appearance over EVERY band pixel, so its
    dash-gap leakage + fit error (band-vs-GT-lane recall only 0.639) inject
    FALSE-POSITIVE lane pixels that OUTWEIGH the tiny recall gain (0.758->0.767).
  * FEED-dj: the band captures lane SHAPE (fn 0.00046) but its dash-gap
    FALSE-POSITIVES (0.00396 as full authority) are the entire residual.
So the FP is the whole enemy. THIS module's non-naive form kills the FP with
THREE composable levers, each measured for its own Delta d_seg:

  (a) AA-SDF coverage-integrated raster [``coverage_alpha_from_signed``] --
      anti-alias the band edge by ANALYTIC sub-pixel coverage of the REAL lane
      geometry (a 1-Lipschitz horizontal signed distance, ``clip(s/soft+0.5,0,1)``).
      AA belongs on REAL geometry: the render-side gate found supersample-AA HURTS
      the OVERSAMPLED synthetic witness (-49%) but coverage-integration of the
      REAL band edge is the correct anti-alias for placing the band.

  (b) RANGE-DEPENDENT dash gate [``rasterize_lane_coverage_range_dependent``] --
      #215: dash-gap FP is range-dependent (perspective x SegNet-Nyquist ~55m). A
      UNIFORM dash gate is WRONG: NEAR the car dashes are resolved -> gate the gaps
      (laying band in a gap is a big FP); BEYOND ~55m dashes fall below the SegNet
      Nyquist -> the net sees a smeared continuous line -> DON'T gate (gating there
      creates FN). The gate is applied ONLY where forward < ``dash_forward_max_m``.

  (c) WITNESS-UNCERTAINTY mask [``witness_uncertainty_mask``] -- composite the
      band ONLY where the decision is UNCERTAIN (low top1-top2 margin). Where the
      pixel is already confidently classified (high margin), the band does NOT
      override -> the dash-gap / fit-error FP (which lands in CONFIDENT regions) is
      killed; where uncertain (the boundary annulus = exactly where spectral-bias
      ERASES the lane, ``margin ~ 0``), the band supplies the structured prior ->
      recovers the erased island. This is the decisive FP-killer the naive band
      lacked. The margin RIDES the #141 unified margin-saliency lever
      (``tac.margin_saliency_map`` / ``seg_core.segnet_argmax_and_margin``, the SAME
      top1-top2 quantity) -- NOT a new heuristic. TWO canonical margin sources
      (``margin_provider``), measured head-to-head: (i) the FROZEN GT SegNet margin
      (fixed, precomputed = ``GTData.margins`` / the cache; marks the intrinsic
      boundary annulus, ZERO extra forward), (ii) the softmax-of-SDF WITNESS's OWN
      decision margin (top1-top2 of ``soft``; "don't override where the witness is
      confident", the most direct FP killer as the witness trains).

PER-CLASS DECOMPOSITION (GR-unified-action, memo
``project_gr_unified_action_full_witness_architecture`` +
``analytic_lane_band_primary_authority_decomposition``): the witness owns the
SMOOTH classes (Road/Undrivable/hood/MyCar -- sister components
``road_horizon_component`` for road/sky, ``hood_static_component`` for MyCar);
the LANE (class 1, the finest-scale ERASURE tail) is this analytic band as
RENDER-TIME authority. This module is the RENDER-COMPOSITE leg; the INIT-side
lane head-start (``lane_headstart``) and the phi_1 SDF injection
(``lane_sdf_component.inject_lane_sdf`` into ``lever_b_levelset_generator``) are
the SISTER legs of the same lane manifold -- composed, not duplicated.

Composite: ``comp = rgb*(1-a) + lane_rgb*a`` with ``a = coverage * u_mask``.

NO-FAKE: every function does the work its name claims on the REAL frozen CPU-torch
SegNet argmax label map (``lstar``) and the REAL witness margin. The band is a REAL
openpilot-IPM polynomial fit to the REAL class-1 pixels (reusing the proven
``lane_sdf_component`` cluster/fit primitives); the coverage is a REAL analytic
sub-pixel signed distance; the uncertainty mask is the REAL witness decision
margin. No stub, no synthetic fixture, no returns-markers-without-work. The
compositing is a REPRESENTATION-fidelity primitive; a numeric d_seg claim is
authoritative ONLY through R + the frozen CPU-torch SegNet argmax (never MPS).

COMPUTE (co-equal facet, operator 2026-07-01): MLX-first + numpy-portable
BIT-IDENTICAL reference (parity >= 0.9997). The differentiable RENDER-LOOP hot
path (``composite_lane_band`` + ``witness_uncertainty_mask``) is elementwise and
``mx.compile``-friendly (NO python pixel loops). The band coverage raster is a
per-code CPU precompute (done ONCE, cached) but is ALSO provided vectorized in
MLX (``rasterize_lane_coverage_mlx``) for the geometry-optimize-through-R lever
(#203) and is FLAGGED as a #212 custom-Metal-kernel candidate (see
``METAL_KERNEL_FLAG`` -- the AA-SDF coverage raster is the prime candidate).

Borrowed-substrate accounting:
  * BORROWED (cited): openpilot flat-ground IPM + comma10k class-1 (via
    ``lane_sdf_component``); the coverage/composite math is standard alpha
    compositing + sub-pixel coverage integration.
  * OURS-ORIGINAL: the THREE-lever FP-killing composite (AA-SDF coverage x
    range-dependent-dash-gate x witness-uncertainty-mask) as a differentiable
    render-band that gives the analytic lane STRUCTURALLY only where the witness
    erases it, so the witness capacity reallocates off the lane long-tail.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from tac.boundary_math.lane_sdf_component import (
    LaneLine,
    _CAM_H,
    _FX,
    _FY,
    _SEG_H,
    _SEG_W,
    _V_HORIZON,
    cluster_lane_lines,
    fit_lane_line,
)

# ---------------------------------------------------------------------------
# #212 Metal-kernel flag (COMPUTE facet). The hot AA-SDF coverage raster is a
# perfect elementwise Metal candidate: over a (H, W) grid with per-ROW broadcast
# line params. The parent can build + wire this per the #212 kernel suite.
# ---------------------------------------------------------------------------
METAL_KERNEL_FLAG: dict[str, Any] = {
    "candidate": "aa_sdf_lane_coverage_raster",
    "priority": "high (per-frame render-loop precompute; #203 makes it in-loop)",
    "signature": (
        "aa_sdf_lane_coverage(u_center: f32[L,H], hw: f32[L,H], gate: f32[L,H], "
        "col_grid: f32[W], softness: f32, out coverage: f32[H,W]) -- per (row v, col u): "
        "s = hw[l,v] - |col_grid[u] - u_center[l,v]|; cov_l = clip(s/softness + 0.5, 0, 1) * gate[l,v]; "
        "coverage[v,u] = max_l cov_l. Threadgroup over (H,W); reduce max over L lines (L~5)."
    ),
    "numpy_reference": "rasterize_lane_coverage_range_dependent (bit-identical authority)",
    "mlx_reference": "rasterize_lane_coverage_mlx",
    "compose_with": "TAC_MLX_CUSTOM_GROUPED_BACKWARD (~17x grouped-backward) + apply_contest_faithful_roundtrip_nhwc (fused-R)",
    "sister_suite": "tac.local_acceleration.metal_fused_r_operator / metal_grouped_conv_backward (#212)",
}

# SegNet-Nyquist forward horizon (#215): beyond ~55 m the perspective-compressed
# dash period falls below the SegNet argmax resolution -> the net reads a smeared
# continuous line -> do NOT dash-gate beyond this (gating there creates FN).
DEFAULT_DASH_FORWARD_MAX_M: float = 55.0


# ---------------------------------------------------------------------------
# Per-line render-frame geometry (VECTORIZED; no python pixel loops).
# forward(v) depends ONLY on the image row v (flat-ground IPM) -> dash on/off is
# EXACT per row; the lateral band is a per-row [u_center +/- hw] interval.
# ---------------------------------------------------------------------------
def _forward_of_rows(
    v_rows: np.ndarray, *, cam_h: float = _CAM_H, fy: float = _FY, v_h: float = _V_HORIZON,
) -> np.ndarray:
    """Ground forward distance (m) for image rows below the horizon (vectorized)."""

    v = np.asarray(v_rows, np.float64)
    return cam_h * fy / np.maximum(v - v_h, 1e-3)


def _line_row_params(
    line: LaneLine, v_rows: np.ndarray, *, dash_gate: bool, dash_forward_max_m: float,
    cam_h: float = _CAM_H, fx: float = _FX, fy: float = _FY, cx: float = _SEG_W / 2.0,
    v_h: float = _V_HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row (u_center, half_width, gate) for one lane line over ``v_rows`` (below
    horizon). ``gate`` in {0,1} folds range-dependent dash + forward-range validity.
    VECTORIZED over rows (no python loop)."""

    v = np.asarray(v_rows, np.float64)
    forward = _forward_of_rows(v, cam_h=cam_h, fy=fy, v_h=v_h)
    lateral = line.lateral_of_forward(forward)
    u_center = cx - lateral * fx / forward
    hw = line.halfwidth_of_v(v)

    f0, f1 = line.forward_range
    in_range = (forward >= f0 - 1.0) & (forward <= f1 + 5.0)

    on = np.ones_like(v, bool)
    if dash_gate and line.dash_period_m > 0.0:
        # RANGE-DEPENDENT: gate dashes only in the near field (forward resolvable);
        # beyond dash_forward_max_m the SegNet cannot resolve the gaps -> continuous.
        near = forward < float(dash_forward_max_m)
        phase = np.mod(forward - line.dash_phase_m, line.dash_period_m) / line.dash_period_m
        dash_on = phase < line.dash_duty
        on = np.where(near, dash_on, True)

    gate = (on & in_range).astype(np.float64)
    return u_center, hw, gate


def rasterize_lane_coverage_range_dependent(
    lines: list[LaneLine], *, h: int = _SEG_H, w: int = _SEG_W, softness: float = 1.0,
    dash_gate: bool = True, dash_forward_max_m: float = DEFAULT_DASH_FORWARD_MAX_M,
    v_h: float = _V_HORIZON, cx: float | None = None,
) -> np.ndarray:
    """AA-SDF coverage raster (H,W) in [0,1], VECTORIZED, range-dependent dash gate.

    Coverage of the union lane band via an ANALYTIC horizontal signed distance
    ``s(v,u) = hw(v) - |u - u_center(v)|`` (1-Lipschitz across the lateral edge),
    ``cov = clip(s/softness + 0.5, 0, 1) * gate(v)``, unioned (max) over lines.
    This is the FREE inflate-time rasterizer (generic algorithm, rule 118); only
    the per-line coeffs are counted. NO python pixel loop -- the only loop is over
    the ~5 lane lines (broadcast per row x col).
    """

    H, W = int(h), int(w)
    cxx = float(W / 2.0) if cx is None else float(cx)
    cov = np.zeros((H, W), np.float32)
    if not lines:
        return cov
    rows = np.arange(H, dtype=np.float64)
    below = rows > (v_h + 1.0)
    if not below.any():
        return cov
    vr = rows[below]  # (Hb,)
    col = np.arange(W, dtype=np.float64)[None, :]  # (1,W)
    soft = max(float(softness), 1e-6)
    acc = np.zeros((int(below.sum()), W), np.float64)  # (Hb,W)
    for ln in lines:
        u_c, hw, gate = _line_row_params(
            ln, vr, dash_gate=dash_gate, dash_forward_max_m=dash_forward_max_m, cx=cxx, v_h=v_h,
        )
        s = hw[:, None] - np.abs(col - u_c[:, None])          # (Hb,W) signed dist to lateral edge
        cov_l = np.clip(s / soft + 0.5, 0.0, 1.0) * gate[:, None]
        acc = np.maximum(acc, cov_l)
    cov[below] = acc.astype(np.float32)
    return cov


# ---------------------------------------------------------------------------
# LaneBandPrior: the per-code FIXED deterministic prior (from GT lstar). CPU
# numpy precompute; cached; not differentiated.
# ---------------------------------------------------------------------------
@dataclass
class LaneBandPrior:
    """Precomputed per-code analytic-lane render prior."""

    coverage: np.ndarray           # (H,W) float32 in [0,1] -- AA-SDF range-dep coverage
    lines: list[LaneLine]
    n_lines: int
    total_floats: int
    n_dash_modeled: int
    band_recall: float             # fraction of GT class-1 px with coverage>=0.5 (fit quality)
    gt_lane_frac: float            # class-1 fraction of the frame (observability)


def build_analytic_lane_band_prior(
    lstar: np.ndarray, *, lane_cls: int = 1, softness: float = 1.0,
    dash_gate: bool = True, dash_forward_max_m: float = DEFAULT_DASH_FORWARD_MAX_M,
    centerline_deg: int = 3, v_h: float = _V_HORIZON,
) -> LaneBandPrior:
    """Cluster -> fit lane lines -> AA-SDF range-dependent coverage. Returns a
    ``LaneBandPrior`` carrying the coverage map + manifold coords (byte accounting)
    + the band-vs-GT-lane recall (fit quality). Reuses the proven
    ``lane_sdf_component`` cluster/fit primitives (NO-FAKE: real polyfit to real px)."""

    a = np.asarray(lstar)
    h, w = a.shape
    clusters = cluster_lane_lines(a, lane_cls=lane_cls, v_h=v_h)
    lines: list[LaneLine] = []
    for c in clusters:
        ln = fit_lane_line(c, centerline_deg=centerline_deg, fit_dash=dash_gate, v_h=v_h)
        if ln is not None:
            lines.append(ln)
    cov = rasterize_lane_coverage_range_dependent(
        lines, h=h, w=w, softness=softness, dash_gate=dash_gate,
        dash_forward_max_m=dash_forward_max_m, v_h=v_h,
    )
    is_lane = a == int(lane_cls)
    nlane = int(is_lane.sum())
    band_recall = float((cov[is_lane] >= 0.5).mean()) if nlane else float("nan")
    return LaneBandPrior(
        coverage=cov,
        lines=lines,
        n_lines=len(lines),
        total_floats=int(sum(ln.n_floats() for ln in lines)),
        n_dash_modeled=int(sum(1 for ln in lines if ln.dash_period_m > 0.0)),
        band_recall=band_recall,
        gt_lane_frac=float(nlane) / float(a.size),
    )


# ---------------------------------------------------------------------------
# Witness-uncertainty mask (the decisive FP killer). numpy + MLX; elementwise.
# ---------------------------------------------------------------------------
def witness_uncertainty_mask(
    margin: np.ndarray, *, tau: float = 0.5, eps: float = 0.25,
) -> np.ndarray:
    """Soft indicator of witness UNCERTAINTY from the decision margin (top1-top2).

    ``u = clip((tau - margin)/eps + 0.5, 0, 1)`` -> 1 where margin << tau (uncertain,
    the boundary annulus where spectral bias ERASES the lane), 0 where margin >> tau
    (confident, where the naive band's FP lands). ``tau`` is the confidence threshold,
    ``eps`` the ramp width. numpy reference (bit-identical authority)."""

    m = np.asarray(margin, np.float32)
    e = max(float(eps), 1e-6)
    return np.clip((float(tau) - m) / e + 0.5, 0.0, 1.0).astype(np.float32)


def witness_uncertainty_mask_mlx(margin: Any, *, tau: float = 0.5, eps: float = 0.25) -> Any:
    """MLX twin of ``witness_uncertainty_mask`` (elementwise; mx.compile-friendly)."""

    import mlx.core as mx

    e = max(float(eps), 1e-6)
    return mx.clip((float(tau) - margin) / e + 0.5, 0.0, 1.0)


# ---------------------------------------------------------------------------
# The differentiable composite (render-loop HOT path). numpy + MLX; elementwise.
# ---------------------------------------------------------------------------
def composite_lane_band(
    rgb: np.ndarray, coverage: np.ndarray, lane_rgb: np.ndarray,
    u_mask: np.ndarray | None = None,
) -> np.ndarray:
    """``comp = rgb*(1-a) + lane_rgb*a`` with ``a = coverage * u_mask`` (u_mask None -> 1).

    ``rgb`` (..., H, W, 3): the witness render. ``coverage`` (H,W): AA-SDF band alpha.
    ``lane_rgb`` (..., H, W, 3) or (3,): the lane appearance. ``u_mask`` (H,W): witness
    uncertainty gate (FP killer). Differentiable w.r.t. ``rgb`` and ``lane_rgb``
    (coverage / u_mask are fixed constants). numpy reference (bit-identical authority)."""

    rgb = np.asarray(rgb, np.float32)
    a = np.asarray(coverage, np.float32)
    if u_mask is not None:
        a = a * np.asarray(u_mask, np.float32)
    a = a[..., None]  # (H,W,1) broadcast over channels
    lane = np.asarray(lane_rgb, np.float32)
    if lane.ndim == 1:
        lane = lane[None, None, :]
    return (rgb * (1.0 - a) + lane * a).astype(np.float32)


def composite_lane_band_mlx(
    rgb: Any, coverage: Any, lane_rgb: Any, u_mask: Any | None = None,
) -> Any:
    """MLX twin of ``composite_lane_band`` (elementwise; differentiable; mx.compile)."""

    import mlx.core as mx

    a = coverage
    if u_mask is not None:
        a = a * u_mask
    a = a[..., None]
    lane = lane_rgb
    if lane.ndim == 1:
        lane = lane[None, None, :]
    return rgb * (1.0 - a) + lane * a


def rasterize_lane_coverage_mlx(
    u_center: Any, hw: Any, gate: Any, col_grid: Any, *, softness: float = 1.0,
) -> Any:
    """MLX vectorized AA-SDF coverage raster (for the geometry-optimize-through-R
    lever #203 + the Metal-kernel parity reference).

    ``u_center``/``hw``/``gate`` are (L, Hb) per-line per-row params; ``col_grid`` is
    (W,). Returns coverage (Hb, W) = max over lines of clip(s/soft+0.5,0,1)*gate.
    NO python loop; the L axis is broadcast + reduced. mx.compile-friendly."""

    import mlx.core as mx

    soft = max(float(softness), 1e-6)
    s = hw[:, :, None] - mx.abs(col_grid[None, None, :] - u_center[:, :, None])  # (L,Hb,W)
    cov_l = mx.clip(s / soft + 0.5, 0.0, 1.0) * gate[:, :, None]
    return mx.max(cov_l, axis=0)  # (Hb,W)


# ---------------------------------------------------------------------------
# compose_fn factory for the trainer hook.
# ---------------------------------------------------------------------------
def make_lane_band_compose_fn(
    priors: dict[int, LaneBandPrior] | list[LaneBandPrior],
    *,
    lane_rgb_provider: Callable[[int], Any] | Any,
    margin_provider: Callable[[int], Any] | dict[int, Any] | None = None,
    tau: float = 0.5,
    eps: float = 0.25,
    weight: float = 1.0,
    use_mlx: bool = True,
) -> Callable[[Any, int], Any]:
    """Build ``compose_fn(rgb_nhwc, code_idx) -> rgb_nhwc`` for the trainer hook.

    Parameters
    ----------
    priors : per-code ``LaneBandPrior`` (dict keyed by code_idx, or list indexed by it).
    lane_rgb_provider : the lane APPEARANCE. Either a constant (3,) / (H,W,3) array, or
        a callable ``code_idx -> lane_rgb``. For the levelset witness the canonical
        choice is the witness's OWN per-pixel lane color ``sigmoid(palette[1]+tex)*255``
        (self-consistent, byte-free) -- pass a callable that renders it.
    margin_provider : the witness DECISION MARGIN (top1-top2) per code, for the
        uncertainty mask. Callable ``code_idx -> margin_hw`` or dict; None -> u_mask
        disabled (naive coverage-only). For the softmax-of-SDF witness the margin is
        ``top1(soft) - top2(soft)``; for an RGB witness pass a SegNet-margin proxy.
    tau, eps : uncertainty mask threshold + ramp.
    weight : global band strength in [0,1] (curriculum ramp knob).

    The coverage / u_mask are treated as FIXED constants (stop-gradient); the
    gradient flows through ``rgb`` (witness) and ``lane_rgb`` (also witness-derived).
    """

    def _get_prior(code_idx: int) -> LaneBandPrior:
        return priors[code_idx] if isinstance(priors, dict) else priors[int(code_idx)]

    def _get(provider: Any, code_idx: int) -> Any:
        if callable(provider):
            return provider(code_idx)
        if isinstance(provider, dict):
            return provider[code_idx]
        return provider

    def compose_fn(rgb_nhwc: Any, code_idx: int) -> Any:
        prior = _get_prior(code_idx)
        lane_rgb = _get(lane_rgb_provider, code_idx)
        cov = prior.coverage
        u_mask = None
        if margin_provider is not None:
            margin = _get(margin_provider, code_idx)
        if use_mlx:
            import mlx.core as mx

            cov_mx = mx.array(cov) if not isinstance(cov, mx.array) else cov
            cov_mx = cov_mx * float(weight)
            if margin_provider is not None:
                margin_mx = margin if isinstance(margin, mx.array) else mx.array(np.asarray(margin, np.float32))
                u_mask = mx.stop_gradient(witness_uncertainty_mask_mlx(margin_mx, tau=tau, eps=eps))
            cov_mx = mx.stop_gradient(cov_mx)
            lane_mx = lane_rgb if isinstance(lane_rgb, mx.array) else mx.array(np.asarray(lane_rgb, np.float32))
            # rgb is (1,H,W,3); coverage is (H,W) -> broadcast on the batch dim.
            return composite_lane_band_mlx(rgb_nhwc, cov_mx[None], lane_mx, u_mask[None] if u_mask is not None else None)
        # numpy path
        cov_w = cov * float(weight)
        if margin_provider is not None:
            u_mask = witness_uncertainty_mask(np.asarray(margin, np.float32), tau=tau, eps=eps)
        return composite_lane_band(np.asarray(rgb_nhwc), cov_w[None], lane_rgb, None if u_mask is None else u_mask[None])

    return compose_fn


# ---------------------------------------------------------------------------
# d_seg decomposition (lane FN / FP / recall) -- observability + the net-negative
# verdict. Reuses the sizing tool's decomposition semantics.
# ---------------------------------------------------------------------------
@dataclass
class LaneDsegDecomp:
    d_seg: float
    lane_recall: float
    lane_fn_frac: float
    lane_fp_frac: float


def decompose_lane_dseg(realized: np.ndarray, gt: np.ndarray, *, lane_cls: int = 1) -> LaneDsegDecomp:
    """d_seg + lane recall + lane FN/FP (fraction of ALL px) for one realized frame
    vs the frozen CPU-torch GT label map. ``realized`` and ``gt`` are (H,W) int maps."""

    r = np.asarray(realized)
    g = np.asarray(gt)
    n = float(g.size)
    is_lane = g == int(lane_cls)
    nlane = int(is_lane.sum())
    return LaneDsegDecomp(
        d_seg=float(np.count_nonzero(r != g) / n),
        lane_recall=(float(np.count_nonzero(r[is_lane] == int(lane_cls))) / nlane) if nlane else float("nan"),
        lane_fn_frac=float(np.sum(is_lane & (r != int(lane_cls))) / n),
        lane_fp_frac=float(np.sum((~is_lane) & (r == int(lane_cls))) / n),
    )


# ===========================================================================
# DECODE-CONSISTENT SERIALIZATION (#224 Wave E: the fork-B rate-118 boundary).
#
# THE PHANTOM THIS CLOSES (R5): the render-band was TRAIN-ONLY -- the training
# composite fits per-pair ``LaneLine``s from the GT class-1 argmax (NOT decode-
# available) and composites the band over the render, but the shipped inflate had
# NO band code, so a witness verdicted WITH the band scored WITHOUT it -> phantom.
#
# THE FIX (rule 118): the video-derived sufficient statistic is the per-pair list
# of ``LaneLine`` MANIFOLD COORDS (centerline_coeffs deg<=3 + halfwidth_coeffs deg1
# + dash (period,phase,duty) + forward_range) -- ~7-13 floats/line, NOT the H*W
# coverage field. Those coeffs are SERIALIZED (COUNTED in archive.zip); the
# ``rasterize_lane_coverage_range_dependent`` that EXPANDS them into the (H,W)
# coverage is a GENERIC deterministic rasterizer regenerated for FREE in inflate.py
# (0 bytes). The scalar render params (softness / dash-gate / weight / uncertainty
# tau,eps / lane_cls / geometry) ride the JSON header (lossless float64 repr).
#
# BIT-EXACT: coeffs are stored as raw float64 (the dtype the rasterizer casts to)
# so ``deserialize`` -> ``rasterize`` reproduces the compress-side coverage BIT-FOR-
# BIT. No GT mask, no scorer weights, no per-pixel table ships (NO-FAKE / rule 118).
# ===========================================================================

LANE_BAND_MAGIC = b"LBND1\x00"


@dataclass(frozen=True)
class LaneBandRenderConfig:
    """The scalar render-band params (decode-reproducible; ride the serialized header).

    ``weight`` scales the coverage (curriculum knob); ``u_mask_*`` control the
    WITNESS-uncertainty gate (the FP killer): the uncertainty is the witness's OWN
    softmax decision margin (top1-top2) -- DECODE-AVAILABLE (inflate computes it),
    so this form is decode-consistent. The GT-SegNet-margin variant (``c_full_gt``
    in the sizing tool) is NOT decode-consistent (needs GT) and is deliberately
    unsupported here."""

    softness: float = 1.0
    dash_gate: bool = True
    dash_forward_max_m: float = DEFAULT_DASH_FORWARD_MAX_M
    v_h: float = _V_HORIZON
    cx: float | None = None
    weight: float = 1.0
    lane_cls: int = 1
    u_mask_enabled: bool = False
    u_mask_tau: float = 0.85
    u_mask_eps: float = 0.35
    lane_rgb_mode: str = "witness_lane"  # v1: the witness's own per-pixel sigmoid(palette[lane]+tex)*255


def _line_to_floats(line: LaneLine) -> tuple[list[float], dict[str, Any]]:
    """(float64 payload, per-line layout meta) for ONE LaneLine. Order (fixed):
    centerline_coeffs | halfwidth_coeffs | [period,phase,duty iff dash] | forward_range[0], [1]."""

    cc = np.asarray(line.centerline_coeffs, np.float64).ravel()
    hc = np.asarray(line.halfwidth_coeffs, np.float64).ravel()
    has_dash = bool(line.dash_period_m > 0.0)
    payload: list[float] = [*cc.tolist(), *hc.tolist()]
    if has_dash:
        payload += [float(line.dash_period_m), float(line.dash_phase_m), float(line.dash_duty)]
    payload += [float(line.forward_range[0]), float(line.forward_range[1])]
    meta = {"nc": int(cc.size), "nh": int(hc.size), "has_dash": has_dash}
    return payload, meta


def _floats_to_line(vals: np.ndarray, meta: dict[str, Any]) -> LaneLine:
    """Inverse of ``_line_to_floats`` (bit-exact; ``vals`` is float64)."""

    off = 0
    nc, nh = int(meta["nc"]), int(meta["nh"])
    cc = np.asarray(vals[off:off + nc], np.float64); off += nc
    hc = np.asarray(vals[off:off + nh], np.float64); off += nh
    dp = dph = 0.0
    dd = 0.5
    if bool(meta["has_dash"]):
        dp = float(vals[off]); dph = float(vals[off + 1]); dd = float(vals[off + 2]); off += 3
    fr = (float(vals[off]), float(vals[off + 1])); off += 2
    return LaneLine(
        centerline_coeffs=cc, halfwidth_coeffs=hc,
        dash_period_m=dp, dash_phase_m=dph, dash_duty=dd, forward_range=fr,
    )


def serialize_lane_band(pairs_lines: list[list[LaneLine]], cfg: LaneBandRenderConfig) -> bytes:
    """Deterministic, bit-exact, brotli-friendly serialization of the per-pair lane
    manifold coords + the scalar render config. Layout:

        LANE_BAND_MAGIC | u32 header_len | header_json(utf8) | float64_payload

    ``header_json`` carries the structural ints + per-line layout + the (lossless
    float64 repr) scalar params; ``float64_payload`` is the concatenated coeff floats
    (pair-major, line-major). COUNTED bytes = ``len(serialize_lane_band(...))``; the
    caller brotli-compresses the result for the measured rate term."""

    header: dict[str, Any] = {
        "format": 1,
        "n_pairs": int(len(pairs_lines)),
        "softness": float(cfg.softness),
        "dash_gate": bool(cfg.dash_gate),
        "dash_forward_max_m": float(cfg.dash_forward_max_m),
        "v_h": float(cfg.v_h),
        "cx": (None if cfg.cx is None else float(cfg.cx)),
        "weight": float(cfg.weight),
        "lane_cls": int(cfg.lane_cls),
        "lane_rgb_mode": str(cfg.lane_rgb_mode),
        "u_mask": (
            {"source": "witness_margin", "tau": float(cfg.u_mask_tau), "eps": float(cfg.u_mask_eps)}
            if cfg.u_mask_enabled else None
        ),
        # geometry constants (generic same-rig IPM; provenance + decode reproducibility). rule-118 FREE.
        "geom": {"cam_h": _CAM_H, "fx": _FX, "fy": _FY, "seg_h": _SEG_H, "seg_w": _SEG_W},
        "pairs": [],
    }
    floats: list[float] = []
    for lines in pairs_lines:
        line_metas: list[dict[str, Any]] = []
        for ln in lines:
            payload, meta = _line_to_floats(ln)
            floats.extend(payload)
            line_metas.append(meta)
        header["pairs"].append(line_metas)
    mj = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    fb = np.asarray(floats, dtype=np.float64).tobytes()
    return LANE_BAND_MAGIC + struct.pack("<I", len(mj)) + mj + fb


def deserialize_lane_band(blob: bytes) -> tuple[list[list[LaneLine]], dict[str, Any]]:
    """Bit-exact inverse of ``serialize_lane_band``. Returns (per-pair lines, header dict)."""

    if blob[: len(LANE_BAND_MAGIC)] != LANE_BAND_MAGIC:
        raise ValueError("bad lane-band magic (NO-FAKE: refusing to guess).")
    off = len(LANE_BAND_MAGIC)
    (hlen,) = struct.unpack_from("<I", blob, off); off += 4
    header = json.loads(blob[off:off + hlen].decode("utf-8")); off += hlen
    vals = np.frombuffer(blob[off:], dtype=np.float64)
    pairs_lines: list[list[LaneLine]] = []
    vi = 0
    for line_metas in header["pairs"]:
        lines: list[LaneLine] = []
        for meta in line_metas:
            span = int(meta["nc"]) + int(meta["nh"]) + (3 if bool(meta["has_dash"]) else 0) + 2
            lines.append(_floats_to_line(vals[vi:vi + span], meta))
            vi += span
        pairs_lines.append(lines)
    return pairs_lines, header


def render_config_from_header(header: dict[str, Any]) -> LaneBandRenderConfig:
    """Reconstruct the scalar ``LaneBandRenderConfig`` from a deserialized header."""

    um = header.get("u_mask")
    return LaneBandRenderConfig(
        softness=float(header["softness"]), dash_gate=bool(header["dash_gate"]),
        dash_forward_max_m=float(header["dash_forward_max_m"]), v_h=float(header["v_h"]),
        cx=(None if header.get("cx") is None else float(header["cx"])),
        weight=float(header["weight"]), lane_cls=int(header["lane_cls"]),
        u_mask_enabled=bool(um is not None),
        u_mask_tau=float(um["tau"]) if um else 0.85,
        u_mask_eps=float(um["eps"]) if um else 0.35,
        lane_rgb_mode=str(header.get("lane_rgb_mode", "witness_lane")),
    )


def build_lane_band_pairs_from_lstars(
    lstars: list[np.ndarray] | np.ndarray, cfg: LaneBandRenderConfig,
    *, centerline_deg: int = 3,
) -> tuple[list[list[LaneLine]], dict[str, Any]]:
    """Fit the per-pair ``LaneLine`` manifold coords from the frozen GT SegNet argmax
    label maps (compress-time; the source video is fully available at compress time).
    Returns (per-pair lines, fit-quality stats). The lines ARE the counted video-
    derived statistic; ``build_analytic_lane_band_prior`` is reused so the fit +
    range-dependent dash exactly match the training composite (NO-FAKE)."""

    pairs_lines: list[list[LaneLine]] = []
    recalls: list[float] = []
    n_lines: list[int] = []
    for lst in lstars:
        prior = build_analytic_lane_band_prior(
            np.asarray(lst), lane_cls=cfg.lane_cls, softness=cfg.softness,
            dash_gate=cfg.dash_gate, dash_forward_max_m=cfg.dash_forward_max_m,
            centerline_deg=centerline_deg, v_h=cfg.v_h,
        )
        pairs_lines.append(prior.lines)
        if not np.isnan(prior.band_recall):
            recalls.append(float(prior.band_recall))
        n_lines.append(int(prior.n_lines))
    stats = {
        "n_pairs": int(len(pairs_lines)),
        "n_lines_mean": float(np.mean(n_lines)) if n_lines else 0.0,
        "band_recall_mean": float(np.mean(recalls)) if recalls else float("nan"),
        "total_lines": int(sum(n_lines)),
    }
    return pairs_lines, stats


def band_alpha(coverage: np.ndarray, u_mask: np.ndarray | None, weight: float) -> np.ndarray:
    """The composite alpha ``a = (coverage * weight) * u_mask`` (u_mask None -> 1).

    Matches ``make_lane_band_compose_fn`` (weight scales coverage BEFORE the
    uncertainty gate). float32."""

    a = np.asarray(coverage, np.float32) * np.float32(weight)
    if u_mask is not None:
        a = a * np.asarray(u_mask, np.float32)
    return a.astype(np.float32)


def composite_band_on_render(
    rgb: np.ndarray, lane_rgb: np.ndarray, coverage: np.ndarray,
    u_mask: np.ndarray | None, weight: float,
) -> np.ndarray:
    """``comp = rgb*(1-a) + lane_rgb*a`` with ``a = band_alpha(coverage,u_mask,weight)``.

    The canonical decode-consistent composite. ``rgb`` (H,W,3) witness bulk; ``lane_rgb``
    (H,W,3) lane appearance; both float. Returns float32. The inline inflate ``_lane_*``
    functions reproduce THIS op-for-op (bit-exact-gate proven)."""

    a = band_alpha(coverage, u_mask, weight)[..., None]
    return (np.asarray(rgb, np.float32) * (1.0 - a) + np.asarray(lane_rgb, np.float32) * a).astype(np.float32)
