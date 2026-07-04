"""Manifold-aware lane-SDF level-set component (FEED-dm).

UPGRADE of the crude lane-edge pixel-margin lever (LEVER-3 in
``experiments/train_levelset_witness_realized_through_R_mlx.py``) into a STRUCTURED
component of the K=5 signed-distance level set
(``tac.boundary_math.lever_b_levelset_generator``). Operator refinement 2026-06-27:
*"the lane edge stuff can probably be more precisely and optimally targeted and
engineered and contained and manifold across level set dimensions."*

THE MATH (grounded; FEED-di/dj + openpilot lane geometry).
  * The level-set witness represents the SegNet argmax partition as ``argmax_k phi_k``
    of K=5 SDF fields. Class 1 = lane markings (comma10k convention, CONFIRMED).
  * The lane region {phi_1 wins} is a ~7-float/line polynomial+dash MANIFOLD in the
    ground (road) frame: centerline ``lateral = poly_D(forward)`` (2-3 coeffs) +
    image half-width ``hw = poly_1(v)`` (2 coeffs) + dash ``(period, phase)`` in
    constant ground meters (2 params). FEED-dj measured the centerline+width band
    captures the lane SHAPE to false-negative d_seg 0.00046 (BELOW witness target
    0.00087); the dominant residual (0.00396) is dash-gap FALSE-POSITIVES.
  * So the OPTIMAL FORM of the lane is NOT a per-pixel margin loss over the whole
    lane region (diffuse, uncontained, 2D-pixel) but a SIGNED-DISTANCE FIELD
    ``phi_1 = SDF-to-(lane-polynomial-band-with-dash-gate)``:
      - PRECISE: an SDF is 1-Lipschitz (|grad phi|=1) -> the decision margin
        m = phi_top1 - phi_top2 is ~linear through zero -> sub-pixel boundary
        placement that R-survives the uint8 knife-edge (the SDF R-survival cure).
      - CONTAINED: an SDF is LOCALLY SUPPORTED (phi_1 > 0 only INSIDE the band,
        decays linearly negative outside) -> far from the lane manifold phi_1 << 0
        -> phi_0 (road) wins -> road is preserved STRUCTURALLY. Widening the lane
        margin geometrically (a local field) cannot globally inflate the lane head
        and trade the 50%-majority class-0 (the R4 risk the crude shared-argmax
        margin loss carries). The ONLY containment leak is dash gaps -> the
        ground-frame dash gate IS the containment knob.
      - MANIFOLD: DOF = the ~7 coeffs/line (the lane manifold coords) NOT H*W pixel
        weights. This is also the rate story (FEED-dj: ~35 floats/frame -> ~1-2 KB
        counted; the IPM rasterizer is FREE inflate-time generic algorithm, rule 118).

NO-FAKE: every function does the work its name claims on the REAL frozen CPU-torch
SegNet argmax label map. The SDF is a REAL scipy EDT on the REAL rasterized band of
a REAL polynomial fit to the REAL class-1 pixels; the disagreement decomposition is
the REAL argmax vs the REAL L*. No stub, no surrogate, no synthetic fixture. This is
a REPRESENTATION-fidelity primitive (numpy-fp32, CPU, $0) -> ``[macOS-CPU advisory]``
research-signal; it is NOT a trained-witness output and NOT a byte-closed row.

ADDITIVE / DEFAULT-OFF: this module is standalone geometry. It is NOT imported by the
trainer unless a future ``--lane-sdf-component`` flag is wired (integration spec in
the FEED-dm memo). Building it changes no existing behavior.

Borrowed-substrate accounting:
  * BORROWED (cited): openpilot road-frame IPM (commaai/openpilot
    common/transformations); scipy EDT (the same primitive
    ``lever_b_levelset_generator.signed_distance_fields`` uses); numpy polyfit;
    comma10k class-1 = lane markings.
  * OURS-ORIGINAL: parametrizing the SegNet argmax LANE class as a signed-distance
    field to a ground-frame polynomial+dash band, injected as phi_1 of the
    softmax-of-SDF level set so the lane is given STRUCTURALLY (contained by local
    support) and the witness capacity reallocates to the all-class boundary annulus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# IPM / camera constants (scorer resolution 512x384; src/tac/camera.py CONFIRMED).
# Small-angle flat-ground IPM (FEED-dj): forward = H*fy/(v - v_h); lateral = -(u-cx)*forward/fx.
_FX = 400.3
_FY = 399.5
_CX = 256.0
_CAM_H = 1.2          # camera height (m)
_V_HORIZON = 174.0    # VANISHING_POINT y (scorer rows); 188 is the IPM-optimal sweep value (FEED-dj)
_SEG_H = 384
_SEG_W = 512


# ---------------------------------------------------------------------------
# IPM (image <-> ground/road frame). Small-angle flat-ground; valid for v > v_h.
# ---------------------------------------------------------------------------
def image_to_ground(
    u: np.ndarray, v: np.ndarray, *, cam_h: float = _CAM_H, fx: float = _FX,
    fy: float = _FY, cx: float = _CX, v_h: float = _V_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Image (u,v) -> ground (forward, lateral) meters. Pixels at/above the horizon
    (v <= v_h) map to +inf forward (clamped); callers must mask them out."""

    u = np.asarray(u, np.float64)
    v = np.asarray(v, np.float64)
    dv = v - v_h
    safe = dv > 1e-3
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        forward = np.where(safe, cam_h * fy / np.where(safe, dv, 1.0), np.inf)
        fwd_safe = np.where(safe, forward, 0.0)  # avoid inf*finite -> nan warning
        lateral = np.where(safe, -(u - cx) * fwd_safe / fx, np.nan)
    return forward, lateral


def ground_to_image_row(
    v: np.ndarray, lateral_of_forward, *, cam_h: float = _CAM_H, fx: float = _FX,
    fy: float = _FY, cx: float = _CX, v_h: float = _V_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """For image rows ``v`` (below horizon) and a ground centerline
    ``lateral_of_forward(forward) -> lateral``, return (forward(v), u_center(v))."""

    v = np.asarray(v, np.float64)
    forward = cam_h * fy / np.maximum(v - v_h, 1e-3)
    lateral = np.asarray(lateral_of_forward(forward), np.float64)
    u_center = cx - lateral * fx / forward
    return forward, u_center


# ---------------------------------------------------------------------------
# Lane line model (the ~7-float/line manifold coords) + clustering + fitting.
# ---------------------------------------------------------------------------
@dataclass
class LaneLine:
    """One lane line on the ~7-float manifold (ground frame)."""

    centerline_coeffs: np.ndarray   # lateral = polyval(coeffs, forward) ; deg<=3 -> 2-4 floats
    halfwidth_coeffs: np.ndarray    # image half-width = polyval(coeffs, v) ; deg 1 -> 2 floats
    dash_period_m: float = 0.0      # ground forward period (m); 0 -> no dash gate (continuous)
    dash_phase_m: float = 0.0       # ground forward phase (m)
    dash_duty: float = 0.5          # fraction "on" per period
    forward_range: tuple[float, float] = (0.0, 0.0)
    n_pixels: int = 0

    def lateral_of_forward(self, forward: np.ndarray) -> np.ndarray:
        return np.polyval(self.centerline_coeffs, np.asarray(forward, np.float64))

    def halfwidth_of_v(self, v: np.ndarray) -> np.ndarray:
        return np.maximum(np.polyval(self.halfwidth_coeffs, np.asarray(v, np.float64)), 0.5)

    def n_floats(self) -> int:
        return int(len(self.centerline_coeffs) + len(self.halfwidth_coeffs)
                   + (3 if self.dash_period_m > 0 else 0))


def cluster_lane_lines(
    lstar: np.ndarray, *, lane_cls: int = 1, fwd_lo: float = 6.0, fwd_hi: float = 45.0,
    gap_m: float = 0.7, min_px: int = 25, v_h: float = _V_HORIZON,
) -> list[np.ndarray]:
    """Group class-``lane_cls`` pixels into lane LINES by ground lateral (the
    openpilot-aligned BEV grouping that merges a line's dashes across forward
    distance). Returns a list of (N_i, 2) int arrays of (v, u) pixel coords per line.

    Clusters are DEFINED on mid-field pixels (forward in [fwd_lo, fwd_hi], where the
    IPM lateral is stable), split at sorted-lateral gaps > ``gap_m``; then EVERY
    class pixel is assigned to its nearest cluster lateral.
    """

    a = np.asarray(lstar)
    vv, uu = np.where(a == int(lane_cls))
    if vv.size == 0:
        return []
    fwd, lat = image_to_ground(uu.astype(np.float64), vv.astype(np.float64), v_h=v_h)
    finite = np.isfinite(fwd) & np.isfinite(lat)
    if not finite.any():
        return []
    vv, uu, fwd, lat = vv[finite], uu[finite], fwd[finite], lat[finite]

    mid = (fwd >= fwd_lo) & (fwd <= fwd_hi)
    if mid.sum() < min_px:
        mid = np.ones_like(fwd, bool)  # fall back to all points
    mlat = np.sort(lat[mid])
    # split sorted mid-field laterals at gaps -> cluster centers
    splits = np.where(np.diff(mlat) > gap_m)[0]
    groups = np.split(mlat, splits + 1)
    centers = [float(np.median(g)) for g in groups if g.size >= 5]
    if not centers:
        centers = [float(np.median(mlat))]
    centers = np.asarray(centers, np.float64)

    # assign every class pixel to nearest center; keep clusters with >= min_px
    nearest = np.argmin(np.abs(lat[:, None] - centers[None, :]), axis=1)
    out: list[np.ndarray] = []
    for ci in range(centers.size):
        sel = nearest == ci
        if int(sel.sum()) >= min_px:
            out.append(np.stack([vv[sel], uu[sel]], axis=1).astype(np.int64))
    return out


def _fit_dash(
    on_off: np.ndarray, forward: np.ndarray, *, period_lo: float = 3.0,
    period_hi: float = 25.0, n_period: int = 64,
) -> tuple[float, float, float]:
    """Fit a periodic dash model (period, phase, duty) to an on/off signal sampled
    over ground ``forward`` by matched-filter search. Returns (0,0,0) if no clear
    periodicity (caller falls back to a continuous band)."""

    on_off = np.asarray(on_off, np.float64)
    forward = np.asarray(forward, np.float64)
    if on_off.size < 8 or on_off.std() < 1e-3:
        return 0.0, 0.0, 0.0
    duty = float(np.clip(on_off.mean(), 0.1, 0.9))
    best = (-1.0, 0.0, 0.0)  # (score, period, phase)
    periods = np.linspace(period_lo, period_hi, n_period)
    for per in periods:
        phases = np.linspace(0.0, per, 12, endpoint=False)
        for ph in phases:
            model = ((np.mod(forward - ph, per) / per) < duty).astype(np.float64)
            # correlation of zero-mean signals (matched filter)
            mm = model - model.mean()
            ss = on_off - on_off.mean()
            denom = (np.linalg.norm(mm) * np.linalg.norm(ss)) + 1e-9
            score = float(np.dot(mm, ss) / denom)
            if score > best[0]:
                best = (score, float(per), float(ph))
    # require a real match (correlation) AND that a dash gate is meaningful (duty<1)
    if best[0] < 0.25 or duty > 0.85:
        return 0.0, 0.0, 0.0
    return best[1], best[2], duty


def fit_lane_line(
    px_vu: np.ndarray, *, centerline_deg: int = 3, fit_dash: bool = True,
    v_h: float = _V_HORIZON,
) -> LaneLine | None:
    """Fit a ``LaneLine`` (centerline + image half-width + optional dash) to a
    cluster of (v,u) pixels. Returns None if too few/degenerate."""

    px = np.asarray(px_vu, np.int64)
    if px.shape[0] < 12:
        return None
    v = px[:, 0].astype(np.float64)
    u = px[:, 1].astype(np.float64)
    fwd, lat = image_to_ground(u, v, v_h=v_h)
    ok = np.isfinite(fwd) & np.isfinite(lat)
    if ok.sum() < 12:
        return None
    fwd, lat, v, u = fwd[ok], lat[ok], v[ok], u[ok]

    # centerline lateral = poly(forward); degree clamped by spread
    deg = int(min(centerline_deg, max(1, np.unique(np.round(fwd)).size - 1)))
    c_coeffs = np.polyfit(fwd, lat, deg)
    line = LaneLine(centerline_coeffs=c_coeffs, halfwidth_coeffs=np.array([0.0, 3.0]),
                    forward_range=(float(fwd.min()), float(fwd.max())), n_pixels=int(px.shape[0]))

    # image half-width = poly_1(v): per-row spread of |u - u_center|
    _, u_c = ground_to_image_row(v, line.lateral_of_forward, v_h=v_h)
    resid = np.abs(u - u_c)
    vr = np.round(v).astype(np.int64)
    rows = np.unique(vr)
    if rows.size >= 3:
        hw = np.array([np.percentile(resid[vr == r], 90) for r in rows], np.float64)
        hw = np.clip(hw, 0.5, 20.0)
        try:
            line.halfwidth_coeffs = np.polyfit(rows.astype(np.float64), hw, 1)
        except Exception:
            line.halfwidth_coeffs = np.array([0.0, float(np.median(hw))])
    else:
        line.halfwidth_coeffs = np.array([0.0, float(np.clip(np.median(resid) + 1.0, 1.0, 8.0))])

    if fit_dash:
        # build the per-row on/off along forward (reliable mid-field band only)
        rmask = (fwd >= 6.0) & (fwd <= 50.0)
        if rmask.sum() >= 16:
            grid = np.linspace(6.0, 50.0, 96)
            present = np.zeros_like(grid)
            for j, f in enumerate(grid):
                near = np.abs(fwd - f) <= 0.6
                present[j] = 1.0 if near.any() else 0.0
            per, ph, duty = _fit_dash(present, grid)
            line.dash_period_m, line.dash_phase_m, line.dash_duty = per, ph, duty
    return line


# ---------------------------------------------------------------------------
# Rasterize the lane band + build the signed-distance field phi_1.
# ---------------------------------------------------------------------------
def rasterize_lane_band(
    lines: list[LaneLine], *, h: int = _SEG_H, w: int = _SEG_W, dash_gate: bool = True,
    v_h: float = _V_HORIZON,
) -> np.ndarray:
    """Rasterize the union lane band (centerline +/- image half-width, gated by the
    ground-frame dash model) -> bool mask (H,W). Generic deterministic algorithm
    (the FREE inflate-time rasterizer; only the per-line coeffs are counted)."""

    band = np.zeros((int(h), int(w)), bool)
    rows = np.arange(int(h), dtype=np.float64)
    below = rows > (v_h + 1.0)
    vr = rows[below]
    for ln in lines:
        fwd, u_c = ground_to_image_row(vr, ln.lateral_of_forward, v_h=v_h)
        hw = ln.halfwidth_of_v(vr)
        on = np.ones_like(vr, bool)
        if dash_gate and ln.dash_period_m > 0.0:
            phase = (np.mod(fwd - ln.dash_phase_m, ln.dash_period_m) / ln.dash_period_m)
            on = phase < ln.dash_duty
        f0, f1 = ln.forward_range
        in_range = (fwd >= f0 - 1.0) & (fwd <= f1 + 5.0)
        for j, vv in enumerate(vr):
            if not (on[j] and in_range[j]):
                continue
            uc = u_c[j]
            half = hw[j]
            lo = int(max(0, np.floor(uc - half)))
            hi = int(min(w, np.ceil(uc + half) + 1))
            if hi > lo:
                band[int(vv), lo:hi] = True
    return band


def lane_signed_distance(band_mask: np.ndarray) -> np.ndarray:
    """phi_1 = signed distance to the lane band: +EDT inside, -EDT outside (1-Lipschitz,
    same construction as ``signed_distance_fields``). Returns (H,W) float32."""

    from scipy import ndimage

    m = np.asarray(band_mask, bool)
    if m.all():
        return np.full(m.shape, float(max(m.shape)), np.float32)
    if not m.any():
        return np.full(m.shape, -float(max(m.shape)), np.float32)
    d_in = ndimage.distance_transform_edt(m)
    d_out = ndimage.distance_transform_edt(~m)
    return (d_in - d_out).astype(np.float32)


def build_structured_lane_sdf(
    lstar: np.ndarray, *, lane_cls: int = 1, dash_gate: bool = True,
    centerline_deg: int = 3, v_h: float = _V_HORIZON,
) -> tuple[np.ndarray, dict]:
    """Top-level: cluster -> fit lane lines -> rasterize band -> signed distance.
    Returns (phi_1_lane (H,W) float32, meta). ``meta`` carries the manifold coords
    (lines, total floats) for the byte-close accounting + observability."""

    a = np.asarray(lstar)
    h, w = a.shape
    clusters = cluster_lane_lines(a, lane_cls=lane_cls, v_h=v_h)
    lines: list[LaneLine] = []
    for c in clusters:
        ln = fit_lane_line(c, centerline_deg=centerline_deg, fit_dash=dash_gate, v_h=v_h)
        if ln is not None:
            lines.append(ln)
    band = rasterize_lane_band(lines, h=h, w=w, dash_gate=dash_gate, v_h=v_h)
    phi = lane_signed_distance(band)
    meta = {
        "n_lines": len(lines),
        "total_floats": int(sum(ln.n_floats() for ln in lines)),
        "n_dash_modeled": int(sum(1 for ln in lines if ln.dash_period_m > 0.0)),
        "band_px": int(band.sum()),
    }
    return phi, meta


def inject_lane_sdf(phi_hwk: np.ndarray, phi_1_lane: np.ndarray, *, lane_cls: int = 1,
                    mode: str = "replace", bias_scale: float = 1.0) -> np.ndarray:
    """Inject the structured lane SDF into the K-field stack (H,W,K).

    mode="replace": phi[...,lane_cls] = phi_1_lane (the isolation/representation test).
    mode="bias":    phi[...,lane_cls] += bias_scale * phi_1_lane (the prior/bias option,
                    keeps the learned lane head and pulls it toward the manifold).
    mode="paint":   PAINT-THEN-SDF (facet-3, 2026-07-04, task #291). ``replace`` is a MEASURED
                    NO-OP for nucleation: the thin lane SDF (max ~+halfwidth ~3) LOSES the argmax
                    to the deep road/static-core SDF (~+20), so ``argmax`` never picks lane and
                    the seed part_frac[lane]==0 (the LANE NUCLEATION FAILURE — the tau/MCF stage
                    then erodes a zero-mass class, d_seg creeps UP). ``paint`` instead PAINTS the
                    lane LABEL into the argmax at the band pixels (phi_1_lane>0), then REBUILDS
                    all K per-class SDFs from the painted label map via ``signed_distance_fields``
                    (invariant argmax_k phi_k == labels EXACTLY), so the lane WINS by construction
                    and the seed is above the critical nucleus. Measured field-level: lane FN
                    0.0058 -> 0.0019 (3x) vs replace's 0.00583 -> 0.00538. bias_scale unused.
    """

    out = np.asarray(phi_hwk, np.float32).copy()
    if mode == "replace":
        out[..., int(lane_cls)] = np.asarray(phi_1_lane, np.float32)
    elif mode == "bias":
        out[..., int(lane_cls)] = out[..., int(lane_cls)] + float(bias_scale) * np.asarray(phi_1_lane, np.float32)
    elif mode == "paint":
        # Lazy import (avoids a module-load cycle: lever_b imports scipy/curvelet, not this file).
        from tac.boundary_math.lever_b_levelset_generator import signed_distance_fields
        n_k = int(out.shape[-1])
        labels = out.argmax(-1)                                   # current seed partition
        band = np.asarray(phi_1_lane, np.float32) > 0.0           # inside the lane band (+EDT)
        labels = np.where(band, int(lane_cls), labels)            # PAINT the lane label
        out = signed_distance_fields(labels, n_classes=n_k)       # REBUILD consistent K-field
    else:
        raise ValueError(f"mode must be 'replace', 'bias', or 'paint', got {mode!r}")
    return out


@dataclass
class ContainmentDecomp:
    """Disagreement decomposition for the lane-SDF injection vs the true L*."""

    total_dseg: float
    lane_fn: float            # true lane missed (shape FN; should ~= 0.00046)
    lane_fp_from_road: float  # road -> lane (dash-gap leakage = CONTAINMENT into class-0)
    lane_fp_from_other: float # other-class -> lane
    class0_dseg: float        # road pixels disturbed (CONTAINMENT metric; baseline 0)
    other_dseg: float         # non-road/non-lane pixels disturbed
    lane_attributable: float  # lane_fn + lane_fp_from_road + lane_fp_from_other


def decompose_argmax_disagreement(pred: np.ndarray, lstar: np.ndarray, *, lane_cls: int = 1,
                                  road_cls: int = 0) -> ContainmentDecomp:
    """Decompose ``pred != lstar`` into lane-attributable vs class-0 containment vs
    other. ``pred`` and ``lstar`` are (H,W) int label maps (frozen CPU-torch L*)."""

    p = np.asarray(pred)
    L = np.asarray(lstar)
    n = float(L.size)
    is_lane = L == lane_cls
    is_road = L == road_cls
    is_other = ~is_lane & ~is_road
    pred_lane = p == lane_cls
    return ContainmentDecomp(
        total_dseg=float(np.mean(p != L)),
        lane_fn=float(np.sum(is_lane & (p != lane_cls)) / n),
        lane_fp_from_road=float(np.sum(is_road & pred_lane) / n),
        lane_fp_from_other=float(np.sum(is_other & pred_lane) / n),
        class0_dseg=float(np.sum(is_road & (p != road_cls)) / n),
        other_dseg=float(np.sum(is_other & (p != L)) / n),
        lane_attributable=float(
            (np.sum(is_lane & (p != lane_cls)) + np.sum((~is_lane) & pred_lane)) / n
        ),
    )
