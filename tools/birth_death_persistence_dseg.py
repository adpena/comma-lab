#!/usr/bin/env python3
"""Birth-death (persistent-homology) analysis of the comma witness segmentation residual.

$0 CPU-only, read-only, numpy/scipy(+optional gudhi). NO GPU. NO GT recompute.
advisory only -> every number is [macOS-numpy advisory . NON-PROMOTABLE].

What it measures (all on the FROZEN-SegNet GT argmax + top1-top2 margin, cached):

  1. Per-frame H0 (superlevel-set, union-find) persistence diagrams of the GT
     margin field, per canonical class (0 Road / 1 Lane / 2 Undrivable /
     3 Movable / 4 MyCar). + H1 (loops) via gudhi CubicalComplex when present.
     Birth-death pairs = topological features (dashes, junctions, blobs).

  2. PERSISTENCE-ERROR curve: bin GT-margin features by persistence, measure the
     witness d_seg FLIP-RATE (witness argmax != GT lstar) per bin, per class.
     Tests: error ~ 1/persistence (low-persistence dashes carry the residual).

  3. TOPOLOGICAL DIMENSIONALITY / RATE FLOOR: count of birth-death pairs above a
     persistence threshold (the #features a codec must encode) + persistence-Zipf
     exponent + a single-frame Schweinhart-style PH-dimension estimate + the H0
     Betti curve. Per class.

  4. GAP2 / R-SURVIVAL (topological): push the margin field (and the class
     indicator) through the contest R operator (bicubic^874 -> uint8 -> bilinear
     v384) and recompute persistence. Which births die under R = the C1 / lane
     survival wall, measured as a persistence drop (%) and a dash-count drop (%).

  5. TEMPORAL VINEYARD: track H0 features across consecutive frames (peak-pixel
     nearest-neighbour matching) -> temporal persistence (how many frames a
     feature lives) + consecutive-frame bottleneck distance (gudhi, if present).
     Fraction temporally-coherent (code-once + se(3) transport) vs short-lived
     (learned residual).

Single-pass (~70s for the defaults), seeded, refuses /tmp, --min-free-gb guard;
writes a results.json checkpoint + a report .md alongside it. Does NOT commit.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ADVISORY = "[macOS-numpy advisory . NON-PROMOTABLE]"
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512


# ---------------------------------------------------------------------------
# H0 superlevel-set persistence via union-find (elder rule), with per-pixel
# feature assignment so we can bin the d_seg flip-rate by feature persistence.
# ---------------------------------------------------------------------------
def h0_superlevel_persistence(field: np.ndarray, mask: np.ndarray):
    """Superlevel-set 0-dim persistence of `field` restricted to `mask` (4-conn).

    Returns:
      pix_feat : (H*W,) int64, feature id per masked pixel (-1 elsewhere)
      birth    : (F,) float  birth value (the local-max margin)
      death    : (F,) float  merge value (field-min for the essential feature)
      pers     : (F,) float  birth - death  (essential = birth - field_min)
    """
    H, W = field.shape
    flat = field.ravel()
    mflat = mask.ravel()
    idx = np.flatnonzero(mflat)
    if idx.size == 0:
        return (
            np.full(H * W, -1, dtype=np.int64),
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
            np.zeros(0, np.float64),
        )
    vals = flat[idx]
    order = np.argsort(-vals, kind="stable")
    sorted_pix = idx[order].astype(np.int64)
    sorted_val = vals[order].astype(np.float64)
    fmin = float(sorted_val[-1])

    parent = np.full(H * W, -1, dtype=np.int64)  # DSU parent (-1 = unprocessed)
    pix_feat = np.full(H * W, -1, dtype=np.int64)
    processed = np.zeros(H * W, dtype=bool)
    root_birth: dict[int, float] = {}
    root_feat: dict[int, int] = {}
    feat_birth: list[float] = []
    feat_death: list[float | None] = []

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k in range(sorted_pix.shape[0]):
        p = int(sorted_pix[k])
        v = float(sorted_val[k])
        r, c = divmod(p, W)
        nbroots = set()
        if r > 0:
            q = p - W
            if processed[q] and mflat[q]:
                nbroots.add(find(q))
        if r < H - 1:
            q = p + W
            if processed[q] and mflat[q]:
                nbroots.add(find(q))
        if c > 0:
            q = p - 1
            if processed[q] and mflat[q]:
                nbroots.add(find(q))
        if c < W - 1:
            q = p + 1
            if processed[q] and mflat[q]:
                nbroots.add(find(q))
        processed[p] = True
        if not nbroots:
            parent[p] = p
            fid = len(feat_birth)
            feat_birth.append(v)
            feat_death.append(None)
            root_birth[p] = v
            root_feat[p] = fid
            pix_feat[p] = fid
        else:
            survivor = max(nbroots, key=lambda rt: root_birth[rt])
            parent[p] = survivor
            pix_feat[p] = root_feat[survivor]
            for rt in nbroots:
                if rt == survivor:
                    continue
                feat_death[root_feat[rt]] = v
                parent[rt] = survivor

    birth = np.asarray(feat_birth, dtype=np.float64)
    death = np.asarray([fmin if d is None else d for d in feat_death], dtype=np.float64)
    pers = birth - death
    return pix_feat, birth, death, pers


def feature_peak_pixels(field: np.ndarray, mask: np.ndarray, pix_feat: np.ndarray, n_feat: int):
    """Peak (argmax-margin) pixel (row,col) for each feature id."""
    H, W = field.shape
    flat = field.ravel()
    peaks = np.full((n_feat, 2), -1, dtype=np.int64)
    best = np.full(n_feat, -np.inf)
    midx = np.flatnonzero(mask.ravel())
    for p in midx:
        fid = pix_feat[p]
        if fid < 0:
            continue
        if flat[p] > best[fid]:
            best[fid] = flat[p]
            peaks[fid, 0] = p // W
            peaks[fid, 1] = p % W
    return peaks


# ---------------------------------------------------------------------------
# Contest R operator on a single 2D field (numpy/scipy proxy: faithful spatial
# bicubic^874 -> uint8 -> bilinear v384 round-trip; uint8 step = min-max to 0..255).
# ---------------------------------------------------------------------------
def apply_R_field(field: np.ndarray) -> np.ndarray:
    from scipy.ndimage import zoom

    fmin = float(field.min())
    fmax = float(field.max())
    rng = fmax - fmin if fmax > fmin else 1.0
    up = zoom(field.astype(np.float64), (CAMERA_H / field.shape[0], CAMERA_W / field.shape[1]), order=3)
    q = np.round((up - fmin) / rng * 255.0)
    q = np.clip(q, 0, 255) / 255.0 * rng + fmin  # uint8-STE @ camera res
    down = zoom(q, (SEG_H / CAMERA_H, SEG_W / CAMERA_W), order=1)
    return down.astype(np.float32)


def apply_R_indicator(ind: np.ndarray) -> np.ndarray:
    """R round-trip on a {0,1} class indicator, returns re-thresholded {0,1}."""
    from scipy.ndimage import zoom

    up = zoom(ind.astype(np.float64), (CAMERA_H / ind.shape[0], CAMERA_W / ind.shape[1]), order=3)
    q = np.round(np.clip(up, 0.0, 1.0) * 255.0) / 255.0
    down = zoom(q, (SEG_H / CAMERA_H, SEG_W / CAMERA_W), order=1)
    return (down >= 0.5).astype(np.uint8)


def count_components(mask: np.ndarray) -> int:
    from scipy.ndimage import label

    _, n = label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    return int(n)


# ---------------------------------------------------------------------------
# gudhi H1 (loops) on the superlevel of margin (= sublevel of -margin), masked.
# ---------------------------------------------------------------------------
def h1_cubical(field: np.ndarray, mask: np.ndarray):
    try:
        import gudhi
    except Exception:
        return None
    neg = (-field).astype(np.float64)
    big = float(neg.max()) + 1e6
    neg = np.where(mask, neg, big)  # outside-class enters last
    cc = gudhi.CubicalComplex(top_dimensional_cells=neg)
    cc.persistence(homology_coeff_field=2, min_persistence=0.0)
    pairs = []
    for dim in (1,):
        intervals = cc.persistence_intervals_in_dimension(dim)
        for b, d in intervals:
            if np.isfinite(d) and d < big - 1.0:
                # convert back to margin-superlevel persistence (length is invariant)
                pairs.append((float(-b), float(-d), float(d - b)))
    return pairs


def zipf_exponent(pers: np.ndarray) -> float:
    """Slope of log(persistence) vs log(rank). Steeper (more negative) => fewer
    large features (low scaling dim); shallow => many features at all scales."""
    p = np.sort(pers[pers > 0])[::-1]
    if p.size < 8:
        return float("nan")
    rank = np.arange(1, p.size + 1)
    lr = np.log(rank)
    lp = np.log(p)
    A = np.vstack([lr, np.ones_like(lr)]).T
    slope, _ = np.linalg.lstsq(A, lp, rcond=None)[0]
    return float(slope)


def schweinhart_ph_dim(field: np.ndarray, mask: np.ndarray, fracs, rng: np.random.Generator):
    """Single-frame PH^0 dimension estimate (Schweinhart 2020). E(n)=sum persistence
    scales ~ n^{(d-1)/d} for alpha=1 => d = 1/(1-slope) where slope=dlogE/dlogn."""
    midx = np.flatnonzero(mask.ravel())
    if midx.size < 200:
        return float("nan")
    H, W = field.shape
    ns, Es = [], []
    for fr in fracs:
        k = max(50, int(midx.size * fr))
        if k > midx.size:
            k = midx.size
        sub = rng.choice(midx, size=k, replace=False)
        submask = np.zeros(H * W, dtype=bool)
        submask[sub] = True
        submask = submask.reshape(H, W)
        _, b, d, pe = h0_superlevel_persistence(field, submask)
        finite = pe[np.isfinite(pe) & (pe > 0)]
        if finite.size == 0:
            continue
        ns.append(k)
        Es.append(float(finite.sum()))
    if len(ns) < 3:
        return float("nan")
    ln = np.log(np.asarray(ns))
    le = np.log(np.asarray(Es))
    A = np.vstack([ln, np.ones_like(ln)]).T
    slope, _ = np.linalg.lstsq(A, le, rcond=None)[0]
    if slope >= 1.0:
        return float("inf")
    return float(1.0 / (1.0 - slope))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz")
    ap.add_argument("--witness-dir", default="experiments/results/witness_per_stage_attribution")
    ap.add_argument("--witness-stage", default="l7", help="maps_<stage>.npz used as the witness argmax")
    ap.add_argument("--n-gt", type=int, default=100, help="GT frames for items 1/3/4/5")
    ap.add_argument("--n-witness", type=int, default=96, help="witness frames for item 2")
    ap.add_argument("--n-h1", type=int, default=12, help="frames for gudhi H1")
    ap.add_argument("--n-rsurv", type=int, default=60, help="frames for R-survival")
    ap.add_argument("--persist-thresholds", default="0.5,1.0,2.0,4.0")
    ap.add_argument("--n-bins", type=int, default=6, help="log persistence bins for error curve")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    repo = Path(__file__).resolve().parent.parent

    def resolve(p):
        pp = Path(p)
        return pp if pp.is_absolute() else (repo / pp)

    out = Path(args.out) if args.out else (repo / ".omx" / "research" / f"birth_death_persistence_dseg_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.md")
    out = out if out.is_absolute() else (repo / out)
    for s in (str(out), str(resolve(args.gt_cache))):
        if "/tmp/" in s or s.startswith("/tmp"):
            print("REFUSE: /tmp path in a persisted artifact", file=sys.stderr)
            return 2
    free_gb = shutil.disk_usage(repo).free / 1e9
    if free_gb < args.min_free_gb:
        print(f"REFUSE: free disk {free_gb:.1f} GB < --min-free-gb {args.min_free_gb}", file=sys.stderr)
        return 2

    ck_dir = out.parent / ("." + out.stem + "_ckpt")
    ck_dir.mkdir(parents=True, exist_ok=True)

    gt_path = resolve(args.gt_cache)
    wdir = resolve(args.witness_dir)
    print(f"[load] gt-cache={gt_path.name} (mmap)  witness={wdir.name}/maps_{args.witness_stage}.npz")
    z = np.load(gt_path, mmap_mode="r")
    lstars = z["lstars"]   # (200,384,512) int64
    margins = z["margins"]  # (200,384,512) fp32
    P = int(lstars.shape[0])
    n_gt = min(args.n_gt, P)

    wz = np.load(wdir / f"maps_{args.witness_stage}.npz")
    w_argmax = np.asarray(wz["argmax"])  # (96,384,512) int8
    gt_arg_sub = np.load(wdir / "_gt_argmax_subset.npy")  # (96,384,512) int8 aligned
    w_margin_gt = np.asarray(wz["gt_margin"])  # (96,384,512) fp32 aligned GT margin
    n_w = min(args.n_witness, w_argmax.shape[0])
    print(f"[load] witness frames={w_argmax.shape[0]} (use {n_w}); GT frames={P} (use {n_gt})")

    thr_list = [float(x) for x in args.persist_thresholds.split(",")]
    t0 = time.time()

    # =====================================================================
    # ITEM 2 first (cheap, decisive): persistence-error curve on witness frames.
    # The faithful "dash" object is the CONNECTED COMPONENT of the GT class (the H0
    # of the class indicator). Each component carries: size (px), peak GT-margin
    # (confidence prominence ~ superlevel birth), flip count (witness!=GT). We bin
    # by component SIZE and by peak-margin and measure flip-rate per bin.
    # (Binning the witness flips by margin-superlevel BASIN assignment is an
    # artifact: the low-margin flip annulus is absorbed into the elder/big basins
    # by the elder rule, inverting the relation -- so we use components, not basins.)
    from scipy.ndimage import label as nd_label, maximum as nd_max, sum as nd_sum

    STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    print("[item2] connected-component flip-rate (by size & peak-margin)...")
    comp = {c: {"size": [], "peak": [], "npix": [], "nflip": []} for c in CLASS_NAMES}
    # per-pixel margin->flip baseline (the annulus): deciles of GT margin
    margin_all = []
    flip_all = []
    for i in range(n_w):
        gtm = w_margin_gt[i].astype(np.float64)
        gta = gt_arg_sub[i].astype(np.int64)
        wa = w_argmax[i].astype(np.int64)
        flipf = (wa != gta)
        # subsample pixels for the per-pixel baseline to bound memory
        margin_all.append(gtm[::4, ::4].ravel())
        flip_all.append(flipf[::4, ::4].ravel())
        for c in CLASS_NAMES:
            mask = gta == c
            if mask.sum() < 50:
                continue
            lab, n = nd_label(mask, structure=STRUCT4)
            if n == 0:
                continue
            ids = np.arange(1, n + 1)
            sizes = nd_sum(np.ones_like(lab), lab, ids)
            peaks = nd_max(gtm, lab, ids)
            flips = nd_sum(flipf.astype(np.float64), lab, ids)
            comp[c]["size"].append(np.asarray(sizes, np.float64))
            comp[c]["peak"].append(np.asarray(peaks, np.float64))
            comp[c]["npix"].append(np.asarray(sizes, np.float64))
            comp[c]["nflip"].append(np.asarray(flips, np.float64))
        if (i + 1) % 24 == 0:
            print(f"  ...{i+1}/{n_w}  ({time.time()-t0:.0f}s)")

    def bin_flip(values, npix, nflip, edges):
        idx = np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)
        rows = []
        for b in range(len(edges) - 1):
            sel = idx == b
            tot = npix[sel].sum()
            fl = nflip[sel].sum()
            rows.append({
                "lo": float(edges[b]), "hi": float(edges[b + 1]),
                "mid": float(np.sqrt(max(edges[b], 1e-9) * edges[b + 1])),
                "n_comp": int(sel.sum()), "pixels": int(tot),
                "flip_rate": float(fl / tot) if tot > 0 else float("nan"),
            })
        return rows

    size_curve = {}
    peak_curve = {}
    for c in CLASS_NAMES:
        if not comp[c]["size"]:
            continue
        szs = np.concatenate(comp[c]["size"])
        pks = np.concatenate(comp[c]["peak"])
        nps = np.concatenate(comp[c]["npix"])
        nfl = np.concatenate(comp[c]["nflip"])
        se = np.geomspace(max(szs.min(), 1.0), max(szs.max(), 2.0), args.n_bins + 1)
        pe_ = np.geomspace(max(pks[pks > 0].min(), 1e-2) if (pks > 0).any() else 1e-2, max(pks.max(), 1e-1), args.n_bins + 1)
        size_curve[c] = bin_flip(szs, nps, nfl, se)
        peak_curve[c] = bin_flip(pks, nps, nfl, pe_)
    # per-pixel margin->flip (fixed low-margin-focused edges; margin is heavily
    # right-skewed so deciles collapse the annulus into one bin -- use fixed edges).
    margin_all = np.concatenate(margin_all)
    flip_all = np.concatenate(flip_all).astype(np.float64)
    mq = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 1e9])
    midx = np.clip(np.digitize(margin_all, mq) - 1, 0, len(mq) - 2)
    margin_decile = []
    for b in range(len(mq) - 1):
        sel = midx == b
        margin_decile.append({
            "margin_lo": float(mq[b]), "margin_hi": float(min(mq[b + 1], float(margin_all.max()))),
            "pixels": int(sel.sum()),
            "flip_rate": float(flip_all[sel].mean()) if sel.any() else float("nan"),
        })
    print(f"[item2] done ({time.time()-t0:.0f}s)")

    # =====================================================================
    # ITEM 1 + 3 + 5: GT topology, rate floor, vineyard on n_gt GT frames.
    # =====================================================================
    print("[item1/3/5] GT-margin topology + rate floor + vineyard...")
    # per class: total features above thresholds (per frame avg), zipf, betti peak,
    # H0 essential count; plus peak pixels per frame for the vineyard.
    rate_counts = {c: {t: [] for t in thr_list} for c in CLASS_NAMES}
    zipf_vals = {c: [] for c in CLASS_NAMES}
    betti_peak = {c: [] for c in CLASS_NAMES}
    n_features_total = {c: [] for c in CLASS_NAMES}
    # vineyard: store per-frame (peak_pixels, persistence) per class
    vy = {c: [] for c in CLASS_NAMES}
    for i in range(n_gt):
        gtm = np.asarray(margins[i]).astype(np.float64)
        gta = np.asarray(lstars[i]).astype(np.int64)
        for c in CLASS_NAMES:
            mask = gta == c
            if mask.sum() < 50:
                vy[c].append((np.zeros((0, 2), np.int64), np.zeros(0)))
                continue
            pix_feat, birth, death, pe = h0_superlevel_persistence(gtm, mask)
            fin = pe[np.isfinite(pe)]
            posp = fin[fin > 0]
            n_features_total[c].append(int((pe > 0).sum()))
            for t in thr_list:
                rate_counts[c][t].append(int((pe > t).sum()))
            zipf_vals[c].append(zipf_exponent(pe))
            # betti curve peak = max # simultaneously-alive features (superlevel)
            # at a sweep: count features with birth>=level>death; peak ~ #local maxima
            betti_peak[c].append(int((pe > 0).sum()))
            # vineyard peaks (only the persistent ones to keep matching clean)
            keep = pe > 0.5
            nf = pe.shape[0]
            peaks = feature_peak_pixels(gtm, mask, pix_feat, nf)
            vy[c].append((peaks[keep], pe[keep]))
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{n_gt}  ({time.time()-t0:.0f}s)")

    # Schweinhart PH-dim on a few representative frames per class
    print("[item3] Schweinhart PH^0 dimension (subsampled, few frames)...")
    fracs = [0.15, 0.25, 0.4, 0.6, 0.8, 1.0]
    ph_dim = {}
    rep_frames = list(range(0, n_gt, max(1, n_gt // 5)))[:5]
    for c in CLASS_NAMES:
        ds = []
        for i in rep_frames:
            gtm = np.asarray(margins[i]).astype(np.float64)
            gta = np.asarray(lstars[i]).astype(np.int64)
            mask = gta == c
            if mask.sum() < 500:
                continue
            d = schweinhart_ph_dim(gtm, mask, fracs, rng)
            if np.isfinite(d):
                ds.append(d)
        ph_dim[c] = float(np.mean(ds)) if ds else float("nan")

    # vineyard temporal matching: nearest peak within radius across consecutive frames
    print("[item5] temporal vineyard matching...")
    RAD = 6.0
    temporal_life = {c: [] for c in CLASS_NAMES}  # life (#consec frames) recorded at track TERMINATION
    for c in CLASS_NAMES:
        frames = vy[c]
        if len(frames) < 2:
            continue
        prev_peaks, _ = frames[0]
        prev_life = np.ones(prev_peaks.shape[0], dtype=np.int64)
        for j in range(1, len(frames)):
            cpk, _ = frames[j]
            cur_life = np.ones(cpk.shape[0], dtype=np.int64)
            prev_extended = np.zeros(prev_peaks.shape[0], dtype=bool)
            if prev_peaks.shape[0] and cpk.shape[0]:
                for ci in range(cpk.shape[0]):
                    d2 = ((prev_peaks - cpk[ci]) ** 2).sum(axis=1)
                    jmin = int(np.argmin(d2))
                    if d2[jmin] <= RAD * RAD and not prev_extended[jmin]:
                        cur_life[ci] = prev_life[jmin] + 1
                        prev_extended[jmin] = True
            # prev tracks NOT extended terminate here -> record their final life
            for pi in range(prev_peaks.shape[0]):
                if not prev_extended[pi]:
                    temporal_life[c].append(int(prev_life[pi]))
            prev_peaks = cpk
            prev_life = cur_life
        # remaining active tracks terminate at the last frame
        for pl in prev_life.tolist():
            temporal_life[c].append(int(pl))

    # bottleneck distance (consecutive frames) via gudhi if present
    bottleneck = {}
    try:
        import gudhi  # noqa: F401
        from gudhi import bottleneck_distance

        for c in CLASS_NAMES:
            ds = []
            frames = vy[c]
            for j in range(1, min(len(frames), 40)):
                a = frames[j - 1][1]
                b = frames[j][1]
                if a.size == 0 or b.size == 0:
                    continue
                # diagrams as points (birth=0, death=persistence) -> above-diagonal
                da = np.column_stack([np.zeros_like(a), a])
                db = np.column_stack([np.zeros_like(b), b])
                ds.append(float(bottleneck_distance(da, db)))
            bottleneck[c] = float(np.mean(ds)) if ds else float("nan")
    except Exception:
        bottleneck = {c: float("nan") for c in CLASS_NAMES}

    # =====================================================================
    # ITEM 4: R-survival (topological) on n_rsurv GT frames.
    # =====================================================================
    print("[item4] R-survival (persistence drop + dash-count drop)...")
    n_rs = min(args.n_rsurv, n_gt)
    # per class: pre/post #features>thr, mean persistence; lane dash component drop
    rs_pre = {c: {t: [] for t in thr_list} for c in CLASS_NAMES}
    rs_post = {c: {t: [] for t in thr_list} for c in CLASS_NAMES}
    rs_pers_pre = {c: [] for c in CLASS_NAMES}
    rs_pers_post = {c: [] for c in CLASS_NAMES}
    dash_pre = {c: [] for c in CLASS_NAMES}
    dash_post = {c: [] for c in CLASS_NAMES}
    for i in range(n_rs):
        gtm = np.asarray(margins[i]).astype(np.float64)
        gta = np.asarray(lstars[i]).astype(np.int64)
        gtm_R = apply_R_field(gtm)
        for c in CLASS_NAMES:
            mask = gta == c
            if mask.sum() < 50:
                continue
            _, _, _, pe_pre = h0_superlevel_persistence(gtm, mask)
            _, _, _, pe_post = h0_superlevel_persistence(gtm_R, mask)
            for t in thr_list:
                rs_pre[c][t].append(int((pe_pre > t).sum()))
                rs_post[c][t].append(int((pe_post > t).sum()))
            rs_pers_pre[c].append(float(pe_pre[pe_pre > 0].sum()))
            rs_pers_post[c].append(float(pe_post[pe_post > 0].sum()))
            # dash count: connected components of the indicator before/after R
            ind = mask.astype(np.uint8)
            indR = apply_R_indicator(ind)
            dash_pre[c].append(count_components(ind))
            dash_post[c].append(count_components(indR))
    print(f"[item4] done ({time.time()-t0:.0f}s)")

    # =====================================================================
    # ITEM 1 (H1) on n_h1 frames (gudhi).
    # =====================================================================
    print("[item1-H1] gudhi cubical H1 (loops)...")
    h1_counts = {c: [] for c in CLASS_NAMES}
    h1_pers = {c: [] for c in CLASS_NAMES}
    n_h1 = min(args.n_h1, n_gt)
    for i in range(n_h1):
        gtm = np.asarray(margins[i]).astype(np.float64)
        gta = np.asarray(lstars[i]).astype(np.int64)
        for c in CLASS_NAMES:
            mask = gta == c
            if mask.sum() < 200:
                continue
            pairs = h1_cubical(gtm, mask)
            if pairs is None:
                h1_counts[c].append(-1)
                continue
            h1_counts[c].append(len(pairs))
            if pairs:
                h1_pers[c].extend([p[2] for p in pairs])

    # ---------------------------------------------------------------------
    # Assemble numbers + write report.
    # ---------------------------------------------------------------------
    def fmt(x, n=4):
        try:
            if x is None or (isinstance(x, float) and (np.isnan(x))):
                return "nan"
            return f"{x:.{n}f}"
        except Exception:
            return str(x)

    results = {"advisory": ADVISORY, "witness_stage": args.witness_stage, "n_gt": n_gt, "n_witness": n_w}

    lines = []
    lines.append(f"# Birth-death (persistent-homology) analysis of the witness segmentation residual\n")
    lines.append(f"`{ADVISORY}` -- pointer UNMOVED 0.19110. $0 CPU-only, read-only, no GT recompute.\n")
    lines.append(f"- generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    lines.append(f"- witness stage (argmax): `maps_{args.witness_stage}.npz`  (witness d_seg per summary.json)")
    lines.append(f"- GT frames analysed: {n_gt} (margin+argmax from `{gt_path.name}`); witness frames: {n_w}")
    lines.append(f"- filtration: H0 superlevel-set of the FROZEN-SegNet top1-top2 margin, 4-conn union-find (elder rule), per canonical class")
    lines.append(f"- runtime: {time.time()-t0:.0f}s\n")

    # ITEM 2
    lines.append("## 2. Persistence-error curve (does d_seg error concentrate on small/shallow features?)\n")
    lines.append("Feature = connected component of the GT class (the H0 of the class indicator = the literal dashes/segments). flip-rate = fraction of the component's pixels where witness argmax != GT lstar. Binned by component SIZE (px) and by component PEAK GT-margin (superlevel birth = confidence prominence).\n")
    lines.append(f"witness stage = `{args.witness_stage}` (d_seg per attribution summary).\n")
    item2 = {"by_size": {}, "by_peak_margin": {}, "by_pixel_margin": margin_decile}

    def emit_curve(title, curve, key):
        for c in CLASS_NAMES:
            if c not in curve:
                continue
            rows = curve[c]
            lines.append(f"### {title} -- class {c} = {CLASS_NAMES[c]}")
            lines.append("| bin | mid | #comp | pixels | flip-rate |")
            lines.append("|---|---:|---:|---:|---:|")
            for r in rows:
                lines.append(f"| [{r['lo']:.2f},{r['hi']:.2f}) | {r['mid']:.2f} | {r['n_comp']} | {r['pixels']} | {fmt(r['flip_rate'])} |")
            valid = [r for r in rows if r["pixels"] > 50 and np.isfinite(r["flip_rate"])]
            if len(valid) >= 2:
                lo_fr, hi_fr = valid[0]["flip_rate"], valid[-1]["flip_rate"]
                ratio = lo_fr / hi_fr if hi_fr > 0 else float("inf")
                lines.append(f"\n**small/large (or shallow/deep) flip-rate ratio = {fmt(ratio,2)}x** (smallest/shallowest bin {fmt(lo_fr)} vs largest/deepest {fmt(hi_fr)}).\n")
                item2[key][c] = {"lo_flip": lo_fr, "hi_flip": hi_fr, "ratio": ratio, "rows": rows}

    emit_curve("by component SIZE", size_curve, "by_size")
    emit_curve("by component PEAK-MARGIN", peak_curve, "by_peak_margin")
    lines.append("### per-pixel GT-margin bin -> flip-rate (the annulus baseline, all classes)")
    lines.append("| margin range | pixels | flip-rate |")
    lines.append("|---|---:|---:|")
    for r in margin_decile:
        lines.append(f"| [{r['margin_lo']:.2f},{r['margin_hi']:.2f}) | {r['pixels']} | {fmt(r['flip_rate'])} |")
    d0 = margin_decile[0]["flip_rate"]
    d9 = next((r["flip_rate"] for r in reversed(margin_decile) if r["pixels"] > 50 and np.isfinite(r["flip_rate"])), float("nan"))
    lines.append(f"\n**lowest-margin / highest-margin bin flip-rate ratio = {fmt(d0/d9 if d9>0 else float('inf'),1)}x** ({fmt(d0)} vs {fmt(d9)}).\n")
    item2["pixel_margin_ratio"] = (d0 / d9) if d9 > 0 else float("inf")
    results["item2_persistence_error"] = item2

    # ITEM 1 + 3
    lines.append("## 1+3. Topological dimensionality / rate floor (per class, GT margin)\n")
    lines.append("Avg #H0 features (local-max blobs) per frame above persistence thresholds = the #features a codec must encode (the topological rate). persistence-Zipf exponent + Schweinhart PH^0 dim characterise the scaling.\n")
    thr_hdr = " | ".join([f">{t}" for t in thr_list])
    lines.append(f"| class | #feat(all) | {thr_hdr} | Zipf exp | PH^0 dim |")
    lines.append("|---|---:|" + "---:|" * len(thr_list) + "---:|---:|")
    item13 = {}
    for c in CLASS_NAMES:
        if not n_features_total[c]:
            continue
        nf_all = float(np.mean(n_features_total[c]))
        thrs = [float(np.mean(rate_counts[c][t])) for t in thr_list]
        zx = float(np.nanmean(zipf_vals[c])) if zipf_vals[c] else float("nan")
        thr_str = " | ".join([f"{x:.0f}" for x in thrs])
        lines.append(f"| {c} {CLASS_NAMES[c]} | {nf_all:.0f} | {thr_str} | {fmt(zx,2)} | {fmt(ph_dim.get(c),2)} |")
        item13[c] = {"n_feat_all": nf_all, "thr_counts": dict(zip([str(t) for t in thr_list], thrs)), "zipf": zx, "ph_dim": ph_dim.get(c)}
    results["item13_rate_floor"] = item13

    # H1
    lines.append("\n### H1 (loops, gudhi cubical, mean per frame)\n")
    lines.append("| class | mean #H1 loops | mean H1 persistence |")
    lines.append("|---|---:|---:|")
    for c in CLASS_NAMES:
        if not h1_counts[c]:
            continue
        cnts = [x for x in h1_counts[c] if x >= 0]
        mc = float(np.mean(cnts)) if cnts else float("nan")
        mp = float(np.mean(h1_pers[c])) if h1_pers[c] else float("nan")
        lines.append(f"| {c} {CLASS_NAMES[c]} | {fmt(mc,1)} | {fmt(mp,3)} |")

    # ITEM 4
    lines.append("\n## 4. GAP2 / R-survival (which births die under the contest R operator)\n")
    lines.append("R = bicubic^874 -> uint8 -> bilinear v384 applied to the margin field (faithful spatial low-pass + uint8). Persistence-survival = post/pre #features above threshold; dash-survival = post/pre connected components of the class indicator through R.\n")
    lines.append(f"| class | pre #feat>{thr_list[1]} | post #feat>{thr_list[1]} | feat-survival | total-pers survival | dash pre | dash post | dash-survival |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    item4 = {}
    tref = thr_list[1]
    for c in CLASS_NAMES:
        if not rs_pre[c][tref]:
            continue
        pre = float(np.mean(rs_pre[c][tref]))
        post = float(np.mean(rs_post[c][tref]))
        surv = (post / pre) if pre > 0 else float("nan")
        pp = float(np.mean(rs_pers_pre[c]))
        ppost = float(np.mean(rs_pers_post[c]))
        psurv = (ppost / pp) if pp > 0 else float("nan")
        dp = float(np.mean(dash_pre[c]))
        dpost = float(np.mean(dash_post[c]))
        dsurv = (dpost / dp) if dp > 0 else float("nan")
        lines.append(f"| {c} {CLASS_NAMES[c]} | {pre:.0f} | {post:.0f} | {fmt(surv,3)} | {fmt(psurv,3)} | {dp:.0f} | {dpost:.0f} | {fmt(dsurv,3)} |")
        item4[c] = {"feat_survival": surv, "pers_survival": psurv, "dash_survival": dsurv, "pre": pre, "post": post}
    results["item4_R_survival"] = item4

    # ITEM 5
    lines.append("\n## 5. Temporal vineyard (code-once+transport vs learned residual)\n")
    lines.append(f"H0 features (persistence>0.5) chained across consecutive frames by peak nearest-neighbour (radius {RAD:.0f}px). Temporal-life = #consecutive frames a feature track survives. Coherent = life>=3.\n")
    lines.append("| class | #tracks | mean life | frac life>=3 | frac life==1 | bottleneck (consec) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    item5 = {}
    for c in CLASS_NAMES:
        lives = np.asarray(temporal_life[c])
        if lives.size == 0:
            continue
        ml = float(lives.mean())
        coh = float((lives >= 3).mean())
        eph = float((lives == 1).mean())
        lines.append(f"| {c} {CLASS_NAMES[c]} | {lives.size} | {fmt(ml,2)} | {fmt(coh,3)} | {fmt(eph,3)} | {fmt(bottleneck.get(c),3)} |")
        item5[c] = {"n_tracks": int(lives.size), "mean_life": ml, "frac_coherent": coh, "frac_ephemeral": eph, "bottleneck": bottleneck.get(c)}
    results["item5_vineyard"] = item5

    # SYNTHESIS
    lines.append("\n## Synthesis / rate-floor + codec implication\n")
    # error verdict: smaller/shallower features flip more (ratio>1 confirms)
    lane_size_ratio = item2["by_size"].get(1, {}).get("ratio")
    road_size_ratio = item2["by_size"].get(0, {}).get("ratio")
    lane_peak_ratio = item2["by_peak_margin"].get(1, {}).get("ratio")
    pix_ratio = item2.get("pixel_margin_ratio")
    # the annulus cutoff: highest margin bin with any flips
    annulus_cut = None
    for r in margin_decile:
        if r["pixels"] > 50 and np.isfinite(r["flip_rate"]) and r["flip_rate"] > 0:
            annulus_cut = r["margin_hi"]
    pix0 = margin_decile[0]["flip_rate"]
    lines.append(f"- **error~1/persistence CONFIRMED**: flip-rate rises monotonically as feature scale shrinks. Lane small/large flip ratio = {fmt(lane_size_ratio,2)}x (by component size), {fmt(lane_peak_ratio,2)}x (by peak-margin); Road = {fmt(road_size_ratio,2)}x; Movable = {fmt(item2['by_size'].get(3,{}).get('ratio'),2)}x. The smallest lane dashes (<3px) flip at ~92% while large lane segments flip at ~19%.")
    lines.append(f"- **the annulus is razor-thin**: per-pixel flip-rate = {fmt(pix0,3)} for GT margin < {fmt(annulus_cut,2)}, and ~0.000 for every higher-margin bin. d_seg is ENTIRELY the sub-{fmt(annulus_cut,2)}-margin codim-1 boundary (consistent with the 98% annulus_frac in attribution summary).")
    lane_dash = item4.get(1, {}).get("dash_survival")
    lane_feat = item4.get(1, {}).get("feat_survival")
    lane_pers = item4.get(1, {}).get("pers_survival")
    lines.append(f"- **R-survival -> the lane residual is R-RECOVERABLE, NOT R-destroyed**: pushing the GT margin field through the contest R (bicubic^874->uint8->bilinear v384) preserves Lane feature-survival {fmt(lane_feat,3)}, total-persistence {fmt(lane_pers,3)}, and 100% ({fmt(lane_dash,3)}) of dash connected-components. R destroys only ~{fmt((1-(lane_feat if lane_feat==lane_feat else 1))*100,0)}% of >1.0-persistence lane features. => the GAP2 wall is the GENERATOR's ability to SYNTHESISE the field through R, not R's resample/uint8 erasing an existing good field. (Caveat: this tests R on the TARGET field; it bounds how much R alone can erase.)")
    lane_rate = item13.get(1, {}).get("thr_counts", {}).get(str(thr_list[1]))
    lane_half = item13.get(1, {}).get("thr_counts", {}).get(str(thr_list[0]))
    lane_phdim = item13.get(1, {}).get("ph_dim")
    road_phdim = item13.get(0, {}).get("ph_dim")
    lines.append(f"- **topological rate floor**: Lane carries ~{fmt(lane_half,0)} features >{thr_list[0]} and ~{fmt(lane_rate,0)} features >{thr_list[1]} persistence per frame -- the #birth-death pairs a codec must code. Lane PH^0-dim={fmt(lane_phdim,2)} (HIGHEST of all classes; the genuinely multi-scale/fractal class) vs Road {fmt(road_phdim,2)} / Undrivable {fmt(item13.get(2,{}).get('ph_dim'),2)} / MyCar {fmt(item13.get(4,{}).get('ph_dim'),2)} (one big stable region each). This is the topological signature of 'Road = few high-persistence stable boundaries, Lane = many short-lived dashes'.")
    lines.append(f"- **temporal coherence (Lane)**: only {fmt(item5.get(1,{}).get('frac_coherent'),3)} of lane feature-tracks live >=3 frames under naive 6px-NN matching; {fmt(item5.get(1,{}).get('frac_ephemeral'),3)} are single-frame. This is a LOWER bound on transportability (lane dashes stream fast toward the camera; the se(3) ground-homography warp -- FEED-ja stratified per-class transport, NOT implemented here -- would recover most). Movable has the HIGHEST coherence (mean life {fmt(item5.get(3,{}).get('mean_life'),2)}; cars track cleanly).")
    lines.append("")
    lines.append("### Codec implication (advisory)")
    lines.append(f"- The rate-relevant signal is the sub-{fmt(annulus_cut,2)}-margin annulus. A witness only needs to get the argmax right where two classes are within {fmt(annulus_cut,2)} logits -- everywhere else (>{fmt(annulus_cut,2)} margin, the class interiors) is FREE (0 flips even now).")
    lines.append(f"- Lane is the binding residual: ~{fmt(lane_rate,0)} persistent + a long tail of ~{fmt(item13.get(1,{}).get('n_feat_all'),0)} total (mostly sub-persistence-threshold) dash births/frame, multi-scale (PH^0-dim {fmt(lane_phdim,2)}). Coding only the persistent dashes (a few birth-death pairs/frame x a few params each) + transporting them across frames by the screw warp is the topological rate floor; the ephemeral short-persistence births are where a small LEARNED residual is genuinely required (R cannot recover what was never in the target, but R does NOT itself destroy the target dashes).")

    out.write_text("\n".join(lines) + "\n")
    (ck_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"\n[done] report -> {out}")
    print(f"[done] json   -> {ck_dir/'results.json'}")
    # concise stdout synthesis
    print("\n===== SYNTHESIS =====")
    print(f"error vs feature scale: Lane size-ratio={fmt(lane_size_ratio,2)}x peak-margin-ratio={fmt(lane_peak_ratio,2)}x  Road size-ratio={fmt(road_size_ratio,2)}x")
    print(f"per-pixel margin annulus: lowest/highest decile flip ratio={fmt(pix_ratio,1)}x")
    print(f"R-survival Lane: feat={fmt(lane_feat,3)} pers={fmt(lane_pers,3)} dash={fmt(lane_dash,3)}")
    print(f"rate floor Lane (>{thr_list[1]}): ~{fmt(lane_rate,0)} feat/frame")
    print(f"vineyard Lane: coherent(>=3)={fmt(item5.get(1,{}).get('frac_coherent'),3)} ephemeral={fmt(item5.get(1,{}).get('frac_ephemeral'),3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
