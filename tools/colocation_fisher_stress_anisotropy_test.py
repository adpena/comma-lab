# SPDX-License-Identifier: MIT
"""DECISIVE $0 MEASUREMENT: matter-on-fixed-Fisher-background framing test.

Tests 3 falsifiable predictions of the "witness = matter on a fixed Fisher
background" framing, on the CACHED SegNet outputs (n96 GT frames). NO GPU.
Tag: [macOS advisory / research-signal] -- validates a DESIGN framing, NOT a
contest score. NO pointer claim.

The cache (``gt_n96.npz``) stores SegNet ARGMAX (``lstars``) + top1-top2 logit
MARGIN (``margins``) but NOT the full 5-class logits. Fisher curvature
``trace(diag(p)-ppᵀ) = 1 - Σ p_k²`` needs the full softmax, so we RECOMPUTE the
full logits by running the SAME frozen CPU-torch SegNet (the EXACT authority,
``measure_segnet_argmax`` lineage) on the cached ``gt_f1`` frames. We CROSS-CHECK
the recomputed argmax == cached ``lstars`` and recomputed margin == cached
``margins`` to PROVE the logit recompute is byte-faithful to the cache.

3 maps (per frame on the 512x384 grid; z = logits, p = softmax(z), T=1):
  (a) FISHER CURVATURE: curvature(x) = trace(F_x) = Σ_k p_k(1-p_k) = 1 - Σ p_k².
      Also report ‖F_x‖₂ (spectral norm; subsampled).
  (b) MARGIN STRESS: m(x) = z_top1 - z_top2; stress high where m small.
  (c) BOUNDARY TANGENT ANISOTROPY: structure tensor of margin field on the
      annulus Σ; ratio of along-boundary vs across-boundary variation.

3 tests + falsification thresholds (see memo).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n96.npz"
EPS = 1e-6
# Canonical comma10k order (CLAUDE.md NON-NEGOTIABLE -- do NOT luma-sort):
CLASS_NAMES = ["Road", "Lane", "Undrivable", "Movable", "MyCar"]


# ----------------------------------------------------------------------------- #
# Logit recompute (the EXACT frozen CPU-torch SegNet authority).
# ----------------------------------------------------------------------------- #
def segnet_logits(segnet, frame1_hwc_uint8: np.ndarray) -> np.ndarray:
    """Return full (5,384,512) logits of one frame1 under the real SegNet.

    Mirrors ``tac.optimization.frame1_seg_repair_atoms.measure_segnet_argmax``
    EXACTLY (degenerate pair, last-frame preprocess, one forward) but keeps the
    full logits instead of collapsing to argmax+margin.
    """
    import torch

    r = np.asarray(frame1_hwc_uint8, dtype=np.float64)
    if r.ndim != 3 or r.shape[-1] != 3:
        raise ValueError(f"frame1 must be (H,W,3); got {r.shape}")
    pair = torch.from_numpy(np.stack([r, r], axis=0)[None]).float()  # (1,2,H,W,3)
    xp = pair.permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        seg_in = segnet.preprocess_input(xp)  # last frame -> (1,3,384,512)
        logits = segnet(seg_in)  # (1,5,384,512)
    return logits[0].detach().cpu().numpy().astype(np.float64)  # (5,384,512)


# ----------------------------------------------------------------------------- #
# Map computations.
# ----------------------------------------------------------------------------- #
def softmax_chw(z: np.ndarray) -> np.ndarray:
    """z:(5,H,W) -> p:(5,H,W) softmax over axis 0 (T=1)."""
    zmax = z.max(axis=0, keepdims=True)
    e = np.exp(z - zmax)
    return e / e.sum(axis=0, keepdims=True)


def fisher_curvature_trace(p: np.ndarray) -> np.ndarray:
    """p:(5,H,W) -> trace(diag(p)-ppᵀ) = 1 - Σ p_k² : (H,W)."""
    return 1.0 - np.sum(p * p, axis=0)


def fisher_spectral_norm_subsample(p_flat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """p_flat:(K,5) at sampled pixels -> top eigenvalue of diag(p)-ppᵀ : (len(idx),)."""
    ps = p_flat[idx]  # (S,5)
    S = ps.shape[0]
    F = -ps[:, :, None] * ps[:, None, :]  # -ppᵀ
    di = np.arange(5)
    F[:, di, di] += ps  # + diag(p)
    w = np.linalg.eigvalsh(F)  # ascending (S,5)
    return w[:, -1]


def top2_margin(z: np.ndarray) -> np.ndarray:
    """z:(5,H,W) -> top1-top2 logit gap (H,W), clamped >=0."""
    part = np.partition(z, kth=[3, 4], axis=0)  # 4th,5th along axis0 = top2,top1
    return np.clip(part[4] - part[3], 0.0, None)


def argmax_chw(z: np.ndarray) -> np.ndarray:
    return z.argmax(axis=0).astype(np.int64)


def grad_xy(f: np.ndarray):
    """Central-difference gradient of f:(H,W) -> (gy,gx)."""
    gy = np.gradient(f, axis=0)
    gx = np.gradient(f, axis=1)
    return gy, gx


def gaussian_blur(f: np.ndarray, sigma: float) -> np.ndarray:
    """Separable gaussian blur (small, no scipy dependency)."""
    rad = max(1, int(round(3 * sigma)))
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    # convolve along rows then cols via np.apply / padding
    fp = np.pad(f, ((rad, rad), (0, 0)), mode="edge")
    out = np.zeros_like(f)
    for i, kv in enumerate(k):
        out += kv * fp[i:i + f.shape[0], :]
    fp2 = np.pad(out, ((0, 0), (rad, rad)), mode="edge")
    out2 = np.zeros_like(f)
    for i, kv in enumerate(k):
        out2 += kv * fp2[:, i:i + f.shape[1]]
    return out2


def argmax_boundary(lstar: np.ndarray) -> np.ndarray:
    """Boolean (H,W): pixel differs from any 4-neighbor in argmax."""
    b = np.zeros(lstar.shape, dtype=bool)
    b[:-1, :] |= lstar[:-1, :] != lstar[1:, :]
    b[1:, :] |= lstar[1:, :] != lstar[:-1, :]
    b[:, :-1] |= lstar[:, :-1] != lstar[:, 1:]
    b[:, 1:] |= lstar[:, 1:] != lstar[:, :-1]
    return b


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square dilation by `radius` (numpy roll OR)."""
    out = mask.copy()
    for _ in range(radius):
        nxt = out.copy()
        nxt[:-1, :] |= out[1:, :]
        nxt[1:, :] |= out[:-1, :]
        nxt[:, :-1] |= out[:, 1:]
        nxt[:, 1:] |= out[:, :-1]
        out = nxt
    return out


def structure_tensor_anisotropy(m: np.ndarray, sigma: float = 2.0):
    """Structure tensor of margin field m:(H,W).

    Returns (lam1, lam2, e1y, e1x) per pixel: lam1>=lam2 eigenvalues and the
    MAJOR eigenvector (direction of strongest gradient = across-boundary normal
    for a margin field). The MINOR eigenvector is the along-boundary tangent.
    """
    gy, gx = grad_xy(m)
    Jxx = gaussian_blur(gx * gx, sigma)
    Jyy = gaussian_blur(gy * gy, sigma)
    Jxy = gaussian_blur(gx * gy, sigma)
    tr = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    disc = np.sqrt(np.clip(tr * tr / 4.0 - det, 0.0, None))
    lam1 = tr / 2.0 + disc  # major
    lam2 = tr / 2.0 - disc  # minor
    # major eigenvector (e1): for [[Jxx,Jxy],[Jxy,Jyy]], eigvec for lam1
    e1x = Jxy
    e1y = lam1 - Jxx
    nrm = np.sqrt(e1x * e1x + e1y * e1y) + EPS
    e1x = e1x / nrm
    e1y = e1y / nrm
    return lam1, lam2, e1y, e1x, gy, gx


# ----------------------------------------------------------------------------- #
# Pearson via pooled sufficient statistics.
# ----------------------------------------------------------------------------- #
class PearsonAcc:
    def __init__(self):
        self.n = 0.0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def add(self, x: np.ndarray, y: np.ndarray):
        x = x.astype(np.float64).ravel()
        y = y.astype(np.float64).ravel()
        self.n += x.size
        self.sx += x.sum()
        self.sy += y.sum()
        self.sxx += (x * x).sum()
        self.syy += (y * y).sum()
        self.sxy += (x * y).sum()

    def r(self) -> float:
        n = self.n
        if n < 2:
            return float("nan")
        cov = self.sxy - self.sx * self.sy / n
        vx = self.sxx - self.sx * self.sx / n
        vy = self.syy - self.sy * self.sy / n
        d = np.sqrt(vx * vy)
        return float(cov / d) if d > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation on a (sub)sample."""
    from numpy import argsort
    def rank(a):
        order = argsort(a, kind="mergesort")
        r = np.empty(a.size, dtype=np.float64)
        r[order] = np.arange(a.size, dtype=np.float64)
        return r
    rx, ry = rank(x.ravel()), rank(y.ravel())
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


# ----------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--num-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--st-sigma", type=float, default=2.0, help="structure-tensor blur sigma")
    ap.add_argument("--band-radius", type=int, default=2, help="argmax-boundary dilation (px)")
    ap.add_argument("--specnorm-subsample", type=int, default=40000, help="px/frame for ‖F‖₂")
    ap.add_argument("--spearman-subsample", type=int, default=200000, help="px/frame for Spearman")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or (REPO / "experiments/results" / f"colocation_test_{utc}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.cache}", flush=True)
    z = np.load(args.cache)
    gt_f1 = z["gt_f1"]
    lstars_cached = z["lstars"]
    margins_cached = z["margins"]
    P = int(z["n_pairs"])
    N = P if args.num_frames in (0, None) else min(args.num_frames, P)
    print(f"[load] n_pairs={P} using N={N} frames", flush=True)

    from tac.boundary_math.seg_core import load_real_segnet
    t0 = time.time()
    segnet = load_real_segnet("cpu")
    print(f"[segnet] loaded in {time.time()-t0:.1f}s", flush=True)

    # Pooled margin quantiles for stress bands (from cached margins, all frames).
    mflat = margins_cached[:N].ravel()
    q_levels = [0.02, 0.05, 0.10]
    margin_q = {f"{int(q*100)}pct": float(np.quantile(mflat, q)) for q in q_levels}
    print(f"[margin quantiles] {margin_q}", flush=True)

    # Accumulators.
    pear_all_negm = PearsonAcc()       # curvature vs -m  (all pixels)
    pear_all_invm = PearsonAcc()       # curvature vs 1/(m+eps)
    pear_band_negm = PearsonAcc()      # curvature vs -m  (geometric boundary band)
    pear_band_invm = PearsonAcc()

    per_frame_pearson_all = []
    per_frame_pearson_band = []
    per_frame_spearman_all = []

    # spectral-norm: collect (trace, specnorm) sample corr + means
    specnorm_vals = []
    specnorm_trace_pair = PearsonAcc()

    # anisotropy accumulators (on annulus Sigma = small-margin ∪ boundary)
    aniso_sum_lam1 = 0.0
    aniso_sum_lam2 = 0.0
    aniso_grad_along = 0.0   # |∇m·tangent|
    aniso_grad_across = 0.0  # |∇m·normal|
    aniso_npix = 0
    per_frame_aniso = []      # median lam1/lam2 over Sigma per frame
    # per-class lane anisotropy (class 1 boundary)
    lane_sum_lam1 = 0.0
    lane_sum_lam2 = 0.0
    lane_npix = 0

    # flip-mass concentration: for each q-band, fraction of small-margin pixels in geometric band
    flipmass_in_band = {k: [0, 0] for k in margin_q}  # k -> [in_band, total]
    # per-class flip mass: total small-margin (2pct) pixels per GT class
    flipmass_per_class = np.zeros(5, dtype=np.int64)
    classmass_total = np.zeros(5, dtype=np.int64)
    # lane (class1) 2pct-band check
    lane_total = 0
    lane_in_2pct = 0
    band2_total = 0
    band2_lane = 0

    # cross-check faithfulness
    argmax_mismatch = 0
    argmax_total = 0
    margin_absdiff_max = 0.0
    margin_absdiff_mean_acc = 0.0

    t_loop = time.time()
    for i in range(N):
        zlog = segnet_logits(segnet, gt_f1[i])  # (5,384,512)
        p = softmax_chw(zlog)
        curv = fisher_curvature_trace(p)         # (H,W)
        m = top2_margin(zlog)                    # (H,W)
        am = argmax_chw(zlog)                    # (H,W)
        lstar = lstars_cached[i]

        # --- cross-check vs cache (faithfulness proof) ---
        argmax_mismatch += int(np.count_nonzero(am != lstar))
        argmax_total += am.size
        md = np.abs(m - margins_cached[i].astype(np.float64))
        margin_absdiff_max = max(margin_absdiff_max, float(md.max()))
        margin_absdiff_mean_acc += float(md.mean())

        negm = -m
        invm = 1.0 / (m + EPS)

        # --- Test 1: co-location ---
        pear_all_negm.add(curv, negm)
        pear_all_invm.add(curv, invm)
        per_frame_pearson_all.append(_pf(curv, negm))
        # spearman subsample
        ns = min(args.spearman_subsample, curv.size)
        sidx = rng.choice(curv.size, size=ns, replace=False)
        per_frame_spearman_all.append(spearman(curv.ravel()[sidx], negm.ravel()[sidx]))

        # geometric boundary band
        bnd = dilate(argmax_boundary(lstar), args.band_radius)
        if bnd.any():
            pear_band_negm.add(curv[bnd], negm[bnd])
            pear_band_invm.add(curv[bnd], invm[bnd])
            per_frame_pearson_band.append(_pf(curv[bnd], negm[bnd]))

        # --- spectral norm subsample ---
        pf_flat = p.reshape(5, -1).T  # (HW,5)
        nss = min(args.specnorm_subsample, pf_flat.shape[0])
        ssidx = rng.choice(pf_flat.shape[0], size=nss, replace=False)
        sn = fisher_spectral_norm_subsample(pf_flat, ssidx)
        specnorm_vals.append(sn.mean())
        specnorm_trace_pair.add(curv.ravel()[ssidx], sn)

        # --- Test 2: anisotropy ---
        lam1, lam2, e1y, e1x, gy, gx = structure_tensor_anisotropy(m, sigma=args.st_sigma)
        thr5 = margin_q["5pct"]
        sigma_mask = (m < thr5) | argmax_boundary(lstar)
        if sigma_mask.any():
            l1s = lam1[sigma_mask]; l2s = lam2[sigma_mask]
            aniso_sum_lam1 += float(l1s.sum())
            aniso_sum_lam2 += float(l2s.sum())
            aniso_npix += int(sigma_mask.sum())
            ratio = l1s / (l2s + EPS)
            per_frame_aniso.append(float(np.median(ratio)))
            # gradient projection: across = |∇m·e1| (major), along = |∇m·e2 (perp to e1)|
            gpe1 = np.abs(gx[sigma_mask] * e1x[sigma_mask] + gy[sigma_mask] * e1y[sigma_mask])
            # minor (tangent) = rotate e1 by 90deg: (e2x,e2y)=(-e1y,e1x)
            gpe2 = np.abs(gx[sigma_mask] * (-e1y[sigma_mask]) + gy[sigma_mask] * e1x[sigma_mask])
            aniso_grad_across += float(gpe1.sum())
            aniso_grad_along += float(gpe2.sum())
        # lane-class anisotropy (class 1 boundary tube)
        lane_bnd = dilate((lstar == 1), 1) & argmax_boundary(lstar)
        if lane_bnd.any():
            lane_sum_lam1 += float(lam1[lane_bnd].sum())
            lane_sum_lam2 += float(lam2[lane_bnd].sum())
            lane_npix += int(lane_bnd.sum())

        # --- Test 3: flip-mass concentration ---
        for k, thr in margin_q.items():
            sm = m < thr
            tot = int(sm.sum())
            inb = int((sm & bnd).sum())
            flipmass_in_band[k][0] += inb
            flipmass_in_band[k][1] += tot
        # per-class 2pct flip mass (use GT argmax = lstar for class assignment)
        sm2 = m < margin_q["2pct"]
        for c in range(5):
            flipmass_per_class[c] += int((sm2 & (lstar == c)).sum())
            classmass_total[c] += int((lstar == c).sum())
        # lane 60%-in-2pct cross-check
        lane_px = (lstar == 1)
        lane_total += int(lane_px.sum())
        lane_in_2pct += int((lane_px & sm2).sum())
        band2_total += int(sm2.sum())
        band2_lane += int((sm2 & lane_px).sum())

        if (i + 1) % 8 == 0 or i == N - 1:
            el = time.time() - t_loop
            print(f"[frame {i+1}/{N}] {el:.1f}s elapsed ({el/(i+1):.2f}s/frame)", flush=True)

    # ---- assemble results ----
    def _stats(a):
        a = np.asarray(a, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"mean": None, "std": None, "median": None, "n": 0}
        return {"mean": float(a.mean()), "std": float(a.std()), "median": float(np.median(a)), "n": int(a.size)}

    aniso_agg_ratio = aniso_sum_lam1 / (aniso_sum_lam2 + EPS)
    grad_ratio_across_along = aniso_grad_across / (aniso_grad_along + EPS)
    grad_ratio_along_across = aniso_grad_along / (aniso_grad_across + EPS)
    lane_agg_ratio = lane_sum_lam1 / (lane_sum_lam2 + EPS) if lane_npix else None

    test1_r_band = pear_band_negm.r()
    test1_r_all = pear_all_negm.r()

    def verdict_coloc(r):
        if r is None or not np.isfinite(r):
            return "INDETERMINATE"
        if r >= 0.5:
            return "CONFIRMED"
        if r >= 0.3:
            return "PARTIAL"
        return "REFUTED"

    def verdict_aniso(rr):
        if rr is None or not np.isfinite(rr):
            return "INDETERMINATE"
        if 4.0 <= rr <= 10.0:
            return "CONFIRMED"
        if (2.0 <= rr < 4.0) or (10.0 < rr <= 15.0):
            return "PARTIAL"
        return "REFUTED"

    flipmass_frac = {k: (v[0] / v[1] if v[1] else None) for k, v in flipmass_in_band.items()}
    per_class_flip_frac = {
        CLASS_NAMES[c]: {
            "frac_of_all_2pct_mass": float(flipmass_per_class[c] / max(1, flipmass_per_class.sum())),
            "within_class_2pct_rate": float(flipmass_per_class[c] / max(1, classmass_total[c])),
            "class_pixel_share": float(classmass_total[c] / max(1, classmass_total.sum())),
        }
        for c in range(5)
    }

    results = {
        "tag": "[macOS advisory / research-signal]",
        "score_claim": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "utc": utc,
        "n_frames": N,
        "n_pairs_cached": P,
        "grid": [384, 512],
        "data_source": {
            "cache": str(args.cache),
            "argmax_margin": "cached (lstars/margins)",
            "logits": "RECOMPUTED via frozen CPU-torch SegNet (load_real_segnet('cpu'))",
            "segnet_forward": "measure_segnet_argmax lineage (degenerate pair, last-frame preprocess)",
            "device": "cpu",
        },
        "faithfulness_crosscheck": {
            "argmax_mismatch_rate_vs_cached_lstars": float(argmax_mismatch / max(1, argmax_total)),
            "margin_absdiff_max_vs_cached": margin_absdiff_max,
            "margin_absdiff_mean_vs_cached": float(margin_absdiff_mean_acc / N),
            "note": "near-zero => recomputed logits are byte-faithful to the cache authority",
        },
        "margin_quantiles_logit_gap": margin_q,
        "test1_colocation_fisher_vs_stress": {
            "definition": "Pearson(curvature=1-Σp², stress=-margin); pooled across frames",
            "pearson_all_pixels_curv_vs_negm": test1_r_all,
            "pearson_all_pixels_curv_vs_invm": pear_all_invm.r(),
            "pearson_boundary_band_curv_vs_negm": test1_r_band,
            "pearson_boundary_band_curv_vs_invm": pear_band_invm.r(),
            "per_frame_pearson_all": _stats(per_frame_pearson_all),
            "per_frame_pearson_band": _stats(per_frame_pearson_band),
            "per_frame_spearman_all": _stats(per_frame_spearman_all),
            "band_radius_px": args.band_radius,
            "threshold": "CONFIRMED>=0.5 / PARTIAL 0.3-0.5 / REFUTED<0.3 (annulus band)",
            "VERDICT_band": verdict_coloc(test1_r_band),
            "VERDICT_all": verdict_coloc(test1_r_all),
        },
        "fisher_spectral_norm": {
            "mean_specnorm_subsampled": float(np.mean(specnorm_vals)),
            "corr_trace_vs_specnorm": specnorm_trace_pair.r(),
            "subsample_px_per_frame": args.specnorm_subsample,
        },
        "test2_boundary_anisotropy": {
            "definition": "structure tensor of margin field on Sigma=(m<5pct ∪ argmax-boundary); lam1>=lam2",
            "st_sigma": args.st_sigma,
            "aggregate_eigenvalue_ratio_lam1_over_lam2": aniso_agg_ratio,
            "per_frame_median_lam1_over_lam2": _stats(per_frame_aniso),
            "grad_proj_ratio_across_over_along": grad_ratio_across_along,
            "grad_proj_ratio_along_over_across": grad_ratio_along_across,
            "lane_class_boundary_eigenvalue_ratio": lane_agg_ratio,
            "n_sigma_pixels": aniso_npix,
            "n_lane_boundary_pixels": lane_npix,
            "threshold": "CONFIRMED 4-10 (~7) / PARTIAL 2-4 or 10-15 / REFUTED<2",
            "VERDICT_eigenratio": verdict_aniso(aniso_agg_ratio),
            "VERDICT_lane_eigenratio": verdict_aniso(lane_agg_ratio),
            "note": "lam1=across-boundary(normal), lam2=along-boundary(tangent); ratio = codim-1 orientation strength",
        },
        "test3_flipmass_concentration": {
            "definition": "fraction of small-margin (flip-prone) pixels inside geometric argmax-boundary band",
            "band_radius_px": args.band_radius,
            "frac_small_margin_in_boundary_band": flipmass_frac,
            "per_class_flip_mass_2pct": per_class_flip_frac,
            "lane_60pct_crosscheck": {
                "lane_pixels_total": lane_total,
                "frac_lane_pixels_in_2pct_margin_band": float(lane_in_2pct / max(1, lane_total)),
                "frac_2pct_band_that_is_lane": float(band2_lane / max(1, band2_total)),
                "note": "known prior: ~60% of LANE pixels in 2% margin band",
            },
        },
    }

    out_json = out_dir / "colocation_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[done] {out_json}", flush=True)
    print(json.dumps({
        "VERDICT_test1_coloc_band": results["test1_colocation_fisher_vs_stress"]["VERDICT_band"],
        "r_band": test1_r_band, "r_all": test1_r_all,
        "VERDICT_test2_aniso": results["test2_boundary_anisotropy"]["VERDICT_eigenratio"],
        "aniso_ratio": aniso_agg_ratio, "lane_aniso": lane_agg_ratio,
        "flipmass_2pct_in_band": flipmass_frac.get("2pct"),
        "argmax_mismatch": results["faithfulness_crosscheck"]["argmax_mismatch_rate_vs_cached_lstars"],
    }, indent=2), flush=True)
    return 0


def _pf(x, y):
    """Per-frame Pearson."""
    x = x.astype(np.float64).ravel(); y = y.astype(np.float64).ravel()
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
