"""Anisotropy map of the SegNet argmax partition — a $0, memory-light geometric probe.

We measure, on the CACHED argmax labels + margin field (NO SegNet/PoseNet forward, NO
model inference), the per-edge and per-saddle anisotropy of the frozen-scorer partition:

  d_H = log(lambda_max / lambda_min)

of the local 2x2 structure tensor of the margin (Fisher-surrogate) field. This is the SAME
quantity whose byte-win the SPD-cone pose codec measured (there: log(cond) of the 6x6 pose
covariance; the geometry+factorization+SPD-cone treatment's leverage SCALES with d_H). Here
we read it off the argmax boundary to RANK where that treatment (proven on pose, tested on
lane) generalizes beyond the lane.

Taxonomy (per the level-set / Morse-Smale frame):
  * EDGES (separatrices, rank-1 directional): the structure tensor is elongated ALONG the
    boundary tangent -> high d_H. The margin structure tensor is SPD, so d_H >= 0 always.
  * SADDLES (Morse-Smale critical points, rank-2 hyperbolic): where >=3 classes meet (triple
    junctions). There the margin HESSIAN has MIXED-SIGN eigenvalues (hyperbolic), NOT a
    rank-1 edge. We locate them, verify the mixed-sign signature, and measure whether they
    concentrate the flip-prone (low-margin) mass.
  * TEMPORAL: the spatio-temporal structure tensor of the margin volume, and the ego-radial
    vs tangential motion split (why the ego-screw xi factorization compresses time).

Authority: ``[macOS-MLX advisory]`` — a geometric anisotropy map on cached argmax, NOT
through R + the frozen SegNet, NOT byte-closed. It ranks/routes future carriers; it moves no
score. Canonical class order is SELF-DETECTED from the spatial/static signature (never
luma-sorted): ``0=Road 1=Lane 2=Undrivable 3=Movable 4=MyCar``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

# Canonical comma10k class order (verified by signature, NEVER luma-sorted).
ROAD, LANE, UNDRIV, MOVABLE, MYCAR = 0, 1, 2, 3, 4
CLASS_NAMES = {ROAD: "Road", LANE: "Lane", UNDRIV: "Undrivable", MOVABLE: "Movable", MYCAR: "MyCar"}

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# class-order self-detection (spatial/static signature; fail-closed)
# --------------------------------------------------------------------------- #
def detect_class_order(lstars: np.ndarray) -> dict[str, Any]:
    """Verify the canonical class order from area + vertical-centroid + temporal-IoU.

    Returns a dict of per-class signatures. Raises if the cache's argmax does not match the
    canonical ``0=Road 1=Lane 2=Undriv 3=Movable 4=MyCar`` signature (guards against a
    luma-sorted or re-indexed cache silently mislabelling every edge)."""
    L = np.asarray(lstars)
    if L.ndim != 3:
        raise ValueError(f"detect_class_order: lstars must be (T,H,W); got {L.shape}")
    T, H, W = L.shape
    rows = np.arange(H, dtype=np.float64)[:, None]
    sig: dict[int, dict[str, float]] = {}
    for c in range(5):
        mask = L == c
        area = float(mask.mean())
        denom = max(int(mask.sum()), 1)
        vc = float((mask * rows[None]).sum() / denom)
        a, b = mask[0], mask[min(1, T - 1)]
        iou = float((a & b).sum() / max(int((a | b).sum()), 1))
        sig[c] = {"area": area, "vcentroid_row_frac": vc / H, "temporal_iou": iou}
    # Structural invariants that pin the canonical order:
    checks = {
        "Undriv_is_top": sig[UNDRIV]["vcentroid_row_frac"] < 0.35,
        "MyCar_is_bottom": sig[MYCAR]["vcentroid_row_frac"] > 0.80,
        "MyCar_static": sig[MYCAR]["temporal_iou"] > 0.95,
        "Lane_thin": sig[LANE]["area"] < 0.03,
        "Lane_unstable": sig[LANE]["temporal_iou"] < 0.60,
        "Undriv_largest": sig[UNDRIV]["area"] == max(sig[c]["area"] for c in range(5)),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise ValueError(
            f"detect_class_order: cache argmax fails canonical-order signature checks {failed}; "
            f"signatures={sig}. Refusing to mislabel edges (never luma-sort)."
        )
    return {"signatures": sig, "checks": checks, "order": "0=Road 1=Lane 2=Undriv 3=Movable 4=MyCar"}


# --------------------------------------------------------------------------- #
# per-pixel structure tensor of a scalar field (SPD; d_H >= 0)
# --------------------------------------------------------------------------- #
def structure_tensor_dH(
    field2d: np.ndarray, *, sigma: float = 2.0, floor: float = 1e-6
) -> dict[str, np.ndarray]:
    """Local 2x2 structure tensor J = G_sigma * (grad f)(grad f)^T of a 2-D field.

    Returns per-pixel eigenvalues (lam_max, lam_min), d_H = log(lam_max/lam_min), gradient
    energy (trace = lam_max+lam_min), and the dominant-eigenvector orientation (radians of
    the LARGE eigenvector; the structure-tensor large eigenvector is NORMAL to the edge, so
    the tangent is orientation + pi/2). ``floor`` clamps lam_min away from 0 for d_H."""
    f = np.asarray(field2d, dtype=np.float64)
    gy, gx = np.gradient(f)
    Jxx = gaussian_filter(gx * gx, sigma)
    Jyy = gaussian_filter(gy * gy, sigma)
    Jxy = gaussian_filter(gx * gy, sigma)
    tr = Jxx + Jyy
    diff = Jxx - Jyy
    disc = np.sqrt(np.maximum(diff * diff + 4.0 * Jxy * Jxy, 0.0))
    lam_max = 0.5 * (tr + disc)
    lam_min = 0.5 * (tr - disc)
    lam_min_c = np.maximum(lam_min, floor)
    lam_max_c = np.maximum(lam_max, floor)
    dH = np.log(lam_max_c / lam_min_c)
    # dominant (large) eigenvector orientation: angle of eigenvector for lam_max
    # for symmetric [[Jxx,Jxy],[Jxy,Jyy]], eigvec angle = 0.5*atan2(2Jxy, Jxx-Jyy)
    orient = 0.5 * np.arctan2(2.0 * Jxy, diff)
    return {
        "lam_max": lam_max,
        "lam_min": lam_min,
        "dH": dH,
        "energy": tr,
        "orient_normal": orient,
    }


# --------------------------------------------------------------------------- #
# Hessian of a scalar field (can be mixed-sign -> saddle)
# --------------------------------------------------------------------------- #
def hessian_eigs(field2d: np.ndarray, *, sigma: float = 2.0) -> dict[str, np.ndarray]:
    """Gaussian-smoothed Hessian eigenvalues of a 2-D field (mixed-sign = hyperbolic saddle).

    Returns lam1 >= lam2 (SIGNED) per pixel and the saddle indicator (lam1>0>lam2)."""
    f = np.asarray(field2d, dtype=np.float64)
    fs = gaussian_filter(f, sigma)
    fy, fx = np.gradient(fs)
    fyy, fyx = np.gradient(fy)
    fxy, fxx = np.gradient(fx)
    fxy = 0.5 * (fyx + fxy)
    tr = fxx + fyy
    diff = fxx - fyy
    disc = np.sqrt(np.maximum(diff * diff + 4.0 * fxy * fxy, 0.0))
    lam1 = 0.5 * (tr + disc)  # larger (signed)
    lam2 = 0.5 * (tr - disc)  # smaller (signed)
    is_saddle = (lam1 > 0.0) & (lam2 < 0.0)
    return {"lam1": lam1, "lam2": lam2, "is_saddle": is_saddle}


# --------------------------------------------------------------------------- #
# EDGE anisotropy: aggregate d_H per unordered class pair over the crack set
# --------------------------------------------------------------------------- #
@dataclass
class EdgeStat:
    pair: tuple[int, int]
    name: str
    n_cracks: int = 0
    dH_sum: float = 0.0
    dH_energy_sum: float = 0.0
    energy_sum: float = 0.0

    @property
    def dH_mean(self) -> float:
        return self.dH_sum / max(self.n_cracks, 1)

    @property
    def dH_energy_weighted(self) -> float:
        return self.dH_energy_sum / max(self.energy_sum, _EPS)


def _accumulate_edges(
    L: np.ndarray, dH: np.ndarray, energy: np.ndarray, acc: dict[tuple[int, int], EdgeStat]
) -> int:
    """Accumulate per-class-pair crack d_H for one frame. Returns total crack count."""
    H, W = L.shape
    total = 0
    # horizontal cracks: (i,j)-(i,j+1)
    for (dHa, dHb, ea, eb, la, lb) in (
        # vertical-neighbour cracks (down)
        (dH[:-1, :], dH[1:, :], energy[:-1, :], energy[1:, :], L[:-1, :], L[1:, :]),
        # horizontal-neighbour cracks (right)
        (dH[:, :-1], dH[:, 1:], energy[:, :-1], energy[:, 1:], L[:, :-1], L[:, 1:]),
    ):
        diff_mask = la != lb
        if not diff_mask.any():
            continue
        pa = la[diff_mask].ravel()
        pb = lb[diff_mask].ravel()
        crack_dH = 0.5 * (dHa[diff_mask] + dHb[diff_mask]).ravel()
        crack_e = 0.5 * (ea[diff_mask] + eb[diff_mask]).ravel()
        lo = np.minimum(pa, pb)
        hi = np.maximum(pa, pb)
        key = lo.astype(np.int64) * 5 + hi.astype(np.int64)
        for k in np.unique(key):
            m = key == k
            a = int(k // 5)
            b = int(k % 5)
            st = acc.setdefault(
                (a, b), EdgeStat((a, b), f"{CLASS_NAMES[a]}-{CLASS_NAMES[b]}")
            )
            n = int(m.sum())
            st.n_cracks += n
            st.dH_sum += float(crack_dH[m].sum())
            st.dH_energy_sum += float((crack_dH[m] * crack_e[m]).sum())
            st.energy_sum += float(crack_e[m].sum())
            total += n
    return total


# --------------------------------------------------------------------------- #
# SADDLE map: triple junctions + Hessian signature + hard-mass concentration
# --------------------------------------------------------------------------- #
def _triple_junctions(L: np.ndarray) -> np.ndarray:
    """Boolean map: pixel is a triple junction if its 2x2 block (with right/down/diag
    neighbours) contains >=3 distinct classes. Marks the top-left of the block."""
    H, W = L.shape
    a = L[:-1, :-1]
    b = L[:-1, 1:]
    c = L[1:, :-1]
    d = L[1:, 1:]
    # count distinct among the 4 via pairwise inequality is fiddly; use a small stack.
    stack = np.stack([a, b, c, d], axis=0)  # (4, H-1, W-1)
    # number of distinct classes present per block:
    present = np.zeros((5,) + a.shape, dtype=bool)
    for c_idx in range(5):
        present[c_idx] = (stack == c_idx).any(axis=0)
    n_distinct = present.sum(axis=0)
    out = np.zeros((H, W), dtype=bool)
    out[:-1, :-1] = n_distinct >= 3
    return out


# --------------------------------------------------------------------------- #
# vanishing-point / horizon geometry
# --------------------------------------------------------------------------- #
def fit_horizon_line(L: np.ndarray) -> dict[str, float]:
    """Fit the Road<->Undrivable boundary (the horizon) as v = m*u + b over columns.

    Per column u, take the median row of Road-Undriv cracks; robust line fit. Returns slope,
    intercept-at-center, residual (line-likeness), and coverage (fraction of columns with a
    boundary)."""
    H, W = L.shape
    down_a = L[:-1, :]
    down_b = L[1:, :]
    is_ru = ((down_a == ROAD) & (down_b == UNDRIV)) | ((down_a == UNDRIV) & (down_b == ROAD))
    rows_idx = np.arange(H - 1)[:, None]
    us: list[float] = []
    vs: list[float] = []
    for u in range(W):
        col = is_ru[:, u]
        if col.any():
            vs.append(float(np.median(rows_idx[col, 0])))
            us.append(float(u))
    if len(us) < 8:
        return {"n_cols": len(us), "coverage": len(us) / W, "fit_ok": 0.0}
    ua = np.asarray(us)
    va = np.asarray(vs)
    # robust: 1 iteration of least-squares then trim outliers by MAD then refit
    A = np.vstack([ua, np.ones_like(ua)]).T
    coef, *_ = np.linalg.lstsq(A, va, rcond=None)
    resid = va - A @ coef
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
    keep = np.abs(resid) < 4.0 * mad
    coef, *_ = np.linalg.lstsq(A[keep], va[keep], rcond=None)
    v_at_center = float(coef[0] * (W / 2.0) + coef[1])
    resid2 = va[keep] - A[keep] @ coef
    return {
        "n_cols": len(us),
        "coverage": len(us) / W,
        "slope": float(coef[0]),
        "v_at_center_row": v_at_center,
        "residual_rows_rms": float(np.sqrt(np.mean(resid2**2))),
        "near_horizontal_deg": float(np.degrees(np.arctan(abs(coef[0])))),
        "fit_ok": 1.0,
    }


# --------------------------------------------------------------------------- #
# TEMPORAL spatio-temporal structure tensor
# --------------------------------------------------------------------------- #
def temporal_structure_tensor(
    margins: np.ndarray,
    boundary_mask: np.ndarray,
    *,
    sigma_s: float = 2.0,
    vp_uv: tuple[float, float] | None = None,
    max_frames: int = 48,
) -> dict[str, Any]:
    """Global 3x3 spatio-temporal structure tensor of the margin volume over boundary pixels.

    Also splits the temporal-motion energy into ego-RADIAL (away from the vanishing point)
    vs TANGENTIAL components — a low radial/tangential ratio with high temporal coherence is
    the signature that ego expansion (the screw xi) is the dominant, factorable time axis.
    """
    m = np.asarray(margins, dtype=np.float64)
    T = min(m.shape[0], max_frames)
    m = m[:T]
    H, W = m.shape[1], m.shape[2]
    gt = np.gradient(m, axis=0)
    gy = np.empty_like(m)
    gx = np.empty_like(m)
    for t in range(T):
        gyt, gxt = np.gradient(gaussian_filter(m[t], sigma_s))
        gy[t] = gyt
        gx[t] = gxt
    # temporal smoothing of gt for coherence
    gt = gaussian_filter1d(gt, 1.0, axis=0)
    bm = boundary_mask[:T]
    sel = bm & (np.abs(gt) + np.abs(gx) + np.abs(gy) > 0)
    vx = gx[sel].ravel()
    vy = gy[sel].ravel()
    vt = gt[sel].ravel()
    J = np.zeros((3, 3))
    comps = [vx, vy, vt]
    for i in range(3):
        for j in range(3):
            J[i, j] = float(np.mean(comps[i] * comps[j]))
    w = np.linalg.eigvalsh(J)
    w = np.sort(w)[::-1]
    lam_max = max(w[0], _EPS)
    lam_min = max(w[-1], _EPS)
    dH3 = float(np.log(lam_max / lam_min))

    radial_tangential = None
    if vp_uv is not None:
        # radial direction from the vanishing point (in-plane spatial gradient projected onto
        # radial vs tangential). Temporal motion couples to the spatial gradient via brightness
        # constancy (normal-flow speed ~ -gt/|grad|); we split the spatial-gradient energy into
        # radial/tangential, temporally weighted by |gt|.
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        rx = xx - vp_uv[0]
        ry = yy - vp_uv[1]
        rn = np.sqrt(rx * rx + ry * ry) + 1e-6
        rx /= rn
        ry /= rn
        # tangential = perpendicular
        tx = -ry
        ty = rx
        radial_e = 0.0
        tang_e = 0.0
        for t in range(T):
            s = bm[t]
            if not s.any():
                continue
            gxr = gx[t]
            gyr = gy[t]
            wgt = np.abs(gt[t])
            rad = (gxr * rx + gyr * ry) ** 2 * wgt
            tan = (gxr * tx + gyr * ty) ** 2 * wgt
            radial_e += float(rad[s].sum())
            tang_e += float(tan[s].sum())
        radial_tangential = {
            "radial_energy": radial_e,
            "tangential_energy": tang_e,
            "radial_over_tangential": radial_e / max(tang_e, _EPS),
        }
    return {
        "dH_spatiotemporal": dH3,
        "eigs": [float(x) for x in w],
        "n_frames": T,
        "radial_tangential": radial_tangential,
    }


# --------------------------------------------------------------------------- #
# top-level driver
# --------------------------------------------------------------------------- #
@dataclass
class AnisotropyMap:
    class_order: dict[str, Any]
    edges: list[dict[str, Any]]
    saddles: dict[str, Any]
    horizon: dict[str, float]
    temporal: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "class_order": self.class_order,
            "edges": self.edges,
            "saddles": self.saddles,
            "horizon": self.horizon,
            "temporal": self.temporal,
            "meta": self.meta,
        }


def compute_anisotropy_map(
    lstars: np.ndarray,
    margins: np.ndarray,
    *,
    sigma_st: float = 2.0,
    sigma_hess: float = 2.0,
    hard_margin_pct: float = 5.0,
    saddle_radius: int = 3,
    vp_uv: tuple[float, float] | None = None,
    max_frames: int | None = None,
) -> AnisotropyMap:
    """Full anisotropy map over the cached argmax + margin volume.

    ``lstars`` (T,H,W) int argmax labels; ``margins`` (T,H,W) float Fisher-surrogate margin.
    """
    L = np.asarray(lstars)
    M = np.asarray(margins, dtype=np.float64)
    if L.shape != M.shape:
        raise ValueError(f"compute_anisotropy_map: lstars {L.shape} != margins {M.shape}")
    T = L.shape[0] if max_frames is None else min(L.shape[0], max_frames)
    H, W = L.shape[1], L.shape[2]
    order = detect_class_order(L)

    hard_thr = float(np.percentile(M[:T], hard_margin_pct))

    edge_acc: dict[tuple[int, int], EdgeStat] = {}
    total_cracks = 0
    boundary_mask = np.zeros((T, H, W), dtype=bool)

    # saddle aggregation
    saddle_total = 0
    saddle_mixed = 0
    saddle_hard_frac_num = 0.0
    saddle_hard_frac_den = 0
    hard_near_saddle = 0
    hard_total = 0
    tj_per_frame = []
    # FULL saddle eigenstructure (margin-Hessian AND margin structure-tensor AT triple
    # junctions) — the load-bearing rank-1-vs-rank-2 distinction. m = phi_top - phi_runnerup
    # IS the cached margin field, so the true Hessian sign-signature is DIRECT, not owed.
    saddle_hess_l1: list[float] = []
    saddle_hess_l2: list[float] = []
    saddle_st_dH: list[float] = []

    for t in range(T):
        Lt = L[t]
        Mt = M[t]
        st = structure_tensor_dH(Mt, sigma=sigma_st)
        dH = st["dH"]
        energy = st["energy"]
        total_cracks += _accumulate_edges(Lt, dH, energy, edge_acc)
        # boundary mask (any 4-neighbour differs)
        bm = np.zeros((H, W), dtype=bool)
        bm[:-1, :] |= Lt[:-1, :] != Lt[1:, :]
        bm[1:, :] |= Lt[:-1, :] != Lt[1:, :]
        bm[:, :-1] |= Lt[:, :-1] != Lt[:, 1:]
        bm[:, 1:] |= Lt[:, :-1] != Lt[:, 1:]
        boundary_mask[t] = bm

        # saddles: triple junctions ∩ Hessian mixed-sign nearby
        tj = _triple_junctions(Lt)
        tj_per_frame.append(int(tj.sum()))
        he = hessian_eigs(Mt, sigma=sigma_hess)
        is_saddle = he["is_saddle"]
        hl1 = he["lam1"]
        hl2 = he["lam2"]
        # dilate saddle-Hessian by saddle_radius around triple junctions
        ys, xs = np.nonzero(tj)
        saddle_total += len(ys)
        r = saddle_radius
        for y, x in zip(ys.tolist(), xs.tolist()):
            y0, y1 = max(0, y - r), min(H, y + r + 1)
            x0, x1 = max(0, x - r), min(W, x + r + 1)
            win_saddle = is_saddle[y0:y1, x0:x1]
            if win_saddle.any():
                saddle_mixed += 1
            # record the FULL eigenstructure at the junction pixel itself (the saddle locus):
            saddle_hess_l1.append(float(hl1[y, x]))
            saddle_hess_l2.append(float(hl2[y, x]))
            saddle_st_dH.append(float(dH[y, x]))

        # hard-mass concentration: fraction of hard (low-margin) pixels within radius of a TJ
        hard = Mt < hard_thr
        hard_total += int(hard.sum())
        # dilate triple junctions
        tj_dil = _dilate(tj, saddle_radius)
        hard_near_saddle += int((hard & tj_dil).sum())
        saddle_hard_frac_num += float((Mt[tj_dil].size and (Mt[tj_dil] < hard_thr).sum()) or 0)
        saddle_hard_frac_den += int(tj_dil.sum())

    # finalize edges
    total_boundary_cracks = max(total_cracks, 1)
    edges_out: list[dict[str, Any]] = []
    for (a, b), stat in sorted(edge_acc.items(), key=lambda kv: -kv[1].n_cracks):
        share = stat.n_cracks / total_boundary_cracks
        dH_mean = stat.dH_mean
        edges_out.append(
            {
                "pair": [a, b],
                "name": stat.name,
                "n_cracks": stat.n_cracks,
                "pixel_share": share,
                "dH_mean": dH_mean,
                "dH_energy_weighted": stat.dH_energy_weighted,
                "leverage_dH_x_share": dH_mean * share,
            }
        )

    # saddle stats
    area_frac_near_saddle = saddle_hard_frac_den / max(T * H * W, 1)
    hard_frac_overall = hard_total / max(T * H * W, 1)
    hard_frac_near_saddle = hard_near_saddle / max(saddle_hard_frac_den, 1)
    # --- FULL saddle eigenstructure: rank-1 (directionally codeable, #1 lever works) vs
    # rank-2 genuine 2D-hyperbolic (mixed-sign Hessian with comparable |eigs| -> lever FAILS,
    # needs saddle-aware coding). ---
    hl1 = np.asarray(saddle_hess_l1)
    hl2 = np.asarray(saddle_hess_l2)
    st_dH_saddle = np.asarray(saddle_st_dH)
    eig_summary: dict[str, Any] = {"n_junctions_sampled": int(hl1.size)}
    if hl1.size:
        mixed = (hl1 > 0.0) & (hl2 < 0.0)  # hyperbolic
        # Hessian rank-2-ness among mixed-sign: |lam_min|/|lam_max| in [0,1]; ->1 = genuine 2D
        amax = np.maximum(np.abs(hl1), np.abs(hl2)) + _EPS
        amin = np.minimum(np.abs(hl1), np.abs(hl2))
        hess_iso = amin / amax  # 1 = isotropic/2D, 0 = rank-1
        # classify each junction:
        #   directionally-codeable: structure tensor still elongated (st_dH high) OR Hessian
        #     nearly rank-1 (hess_iso low) -> the #1 directional lever can code it.
        #   genuine-2D-hyperbolic: mixed-sign AND Hessian eigs comparable (hess_iso high) AND
        #     structure tensor NOT strongly elongated -> lever fails, saddle-aware code needed.
        st_dH_med = float(np.median(st_dH_saddle))
        genuine_2d = mixed & (hess_iso > 0.33) & (st_dH_saddle < 3.0)
        directionally_codeable = ~genuine_2d
        eig_summary.update(
            {
                "hess_lam1_median": float(np.median(hl1)),
                "hess_lam2_median": float(np.median(hl2)),
                "mixed_sign_fraction": float(mixed.mean()),
                "hess_isotropy_ratio_median": float(np.median(hess_iso[mixed])) if mixed.any() else 0.0,
                "structure_tensor_dH_at_saddles_median": st_dH_med,
                "structure_tensor_dH_at_saddles_mean": float(np.mean(st_dH_saddle)),
                "frac_genuine_2d_hyperbolic": float(genuine_2d.mean()),
                "frac_directionally_codeable": float(directionally_codeable.mean()),
            }
        )
    saddles = {
        "triple_junctions_total": saddle_total,
        "triple_junctions_per_frame_mean": float(np.mean(tj_per_frame)) if tj_per_frame else 0.0,
        "hyperbolic_confirmed": saddle_mixed,
        "hyperbolic_fraction": saddle_mixed / max(saddle_total, 1),
        "eigenstructure": eig_summary,
        "saddle_neighborhood_area_frac": area_frac_near_saddle,
        "hard_frac_overall": hard_frac_overall,
        "hard_frac_in_saddle_neighborhoods": hard_frac_near_saddle,
        "hard_mass_concentration_ratio": hard_frac_near_saddle / max(hard_frac_overall, _EPS),
        "hard_margin_threshold": hard_thr,
        "saddle_radius": saddle_radius,
    }

    horizon = fit_horizon_line(L[0]) if T >= 1 else {"fit_ok": 0.0}
    # average horizon over frames for a stable estimate
    hor_center = []
    hor_slope = []
    hor_resid = []
    for t in range(min(T, 24)):
        h = fit_horizon_line(L[t])
        if h.get("fit_ok"):
            hor_center.append(h["v_at_center_row"])
            hor_slope.append(h["slope"])
            hor_resid.append(h["residual_rows_rms"])
    if hor_center:
        horizon = {
            **horizon,
            "v_at_center_row_mean": float(np.mean(hor_center)),
            "v_at_center_row_std": float(np.std(hor_center)),
            "slope_mean": float(np.mean(hor_slope)),
            "residual_rows_rms_mean": float(np.mean(hor_resid)),
            "n_frames_fit": len(hor_center),
        }

    temporal = temporal_structure_tensor(
        M, boundary_mask, sigma_s=sigma_st, vp_uv=vp_uv, max_frames=min(T, 48)
    )

    return AnisotropyMap(
        class_order=order,
        edges=edges_out,
        saddles=saddles,
        horizon=horizon,
        temporal=temporal,
        meta={
            "n_frames": T,
            "grid": [H, W],
            "sigma_structure_tensor": sigma_st,
            "sigma_hessian": sigma_hess,
            "total_boundary_cracks": total_cracks,
            "authority": "[macOS-MLX advisory] geometric anisotropy on cached argmax; NOT byte-closed",
        },
    )


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Square dilation by radius r (cheap, no scipy binary_dilation dependency)."""
    from scipy.ndimage import maximum_filter

    return maximum_filter(mask.astype(np.uint8), size=2 * r + 1) > 0
