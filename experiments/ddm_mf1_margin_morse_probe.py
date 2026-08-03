# SPDX-License-Identifier: MIT
"""ddm_mf1 -- is the margin field a Morse function, and does its Morse-Smale complex
coincide with the SegNet argmax partition?

$0, scorer-free. Consumes ONLY the cached frozen-CPU-torch SegNet outputs in
``experiments/results/mlx_fleet_gt_cache/gt_n600.npz`` (``lstars`` int64 argmax +
``margins`` float32 top1-minus-runner-up), which are the EXACT outputs of the same
frozen authority ``upstream/evaluate.py`` scores through.  No score claim; every row is
``[macOS-CPU advisory]`` structural geometry.

Implements the falsifiers pre-registered in
``.omx/research/ddm_mf1_margin_morse_licence_20260803.md`` §2 BEFORE measurement:

  F0   Morse-Smale 2-cells are open disks -> do argmax components have holes?
  F1a  is the critical set of ``m`` 0-dimensional (isolated) or 1-dimensional (curves)?
  F1b  non-degeneracy: neighbour ties (plateaus) + duplicated critical values.
  F2   bijection  R = #local maxima(m) / #argmax connected components.
  F3/F5 coincidence: steepest-ascent (Forman discrete-gradient) descending-manifold
        partition of ``m`` vs the argmax partition, at h=0 and at the best-case
        persistence level h*.
  F4   directional asymmetry of the transverse margin depth profile per ORDERED
        class pair (the owed ``msal_uni`` per-side probe's precondition).

Also emits the per-component area/perimeter census the displacement-carrier exchange
rate in §5 of the memo is computed from.

No external deps beyond numpy+scipy (skimage is absent and the venv is SHARED with live
sister arms -- it is not mutated).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"

# comma10k canonical order -- MEASURED, never re-derived by luma-sort (CLAUDE.md).
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")

C4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
C8 = np.ones((3, 3), dtype=bool)
NBR8 = np.ones((3, 3), dtype=bool)
NBR8[1, 1] = False

# F4 is measured on the ordered pairs whose undirected interface carries >=1e5 boundary
# pixels over n600 (ddm_sx1 edge_len census); smaller pairs are reported but not fitted.
F4_PAIRS = ((0, 1), (0, 2), (0, 3), (0, 4), (2, 3))

_OFFSETS8 = tuple(
    (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)
)


def _shift(a: np.ndarray, dy: int, dx: int, fill: float) -> np.ndarray:
    """Shift ``a`` by (dy,dx) with constant ``fill`` at the border."""
    out = np.full_like(a, fill)
    h, w = a.shape
    ys, ye = max(0, dy), min(h, h + dy)
    xs, xe = max(0, dx), min(w, w + dx)
    out[ys:ye, xs:xe] = a[ys - dy : ye - dy, xs - dx : xe - dx]
    return out


def _boundary_mask(lab: np.ndarray) -> np.ndarray:
    """4-connected partition boundary: pixel differs from at least one 4-neighbour."""
    b = np.zeros(lab.shape, dtype=bool)
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[1:, :] |= lab[:-1, :] != lab[1:, :]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    return b


def _crack_length(lab: np.ndarray) -> int:
    """Number of 4-adjacent unlike pairs (crack edges) -- the true interface length."""
    return int((lab[:-1, :] != lab[1:, :]).sum() + (lab[:, :-1] != lab[:, 1:]).sum())


def _components(labels: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    """4-connected connected components of the 5-class argmax partition.

    Returns ``(comp_image, n_components, class_of_component)``.  The per-component class
    is needed because the displacement residual is localised to the ``Movable`` edges
    (ddm_sx1 §2.5), so a displacement carrier must be priced on Movable components, not
    on all components.
    """
    out = np.zeros(labels.shape, dtype=np.int32)
    total = 0
    cls: list[int] = []
    for k in range(5):
        mask = labels == k
        if not mask.any():
            continue
        comp, n = ndimage.label(mask, structure=C4)
        out[mask] = comp[mask] + total
        total += n
        cls.extend([k] * n)
    return out, total, np.asarray(cls, dtype=np.int8)


def _hole_census(labels: np.ndarray, comp: np.ndarray) -> dict:
    """F0 -- do argmax components have holes (i.e. are they NOT open disks)?

    Digital topology: components are 4-connected FOREGROUND, so holes must be counted
    with 8-connected BACKGROUND (the Jordan-curve-consistent pairing).  Passing a
    4-connected structure here would let a background region that escapes to the border
    only diagonally be miscounted as a hole -- an OVER-count.  ``binary_fill_holes``'s
    ``structure`` is the background propagation element, hence C8.
    """
    n_holes = 0
    holed = set()
    for k in range(5):
        mask = labels == k
        if not mask.any():
            continue
        filled = ndimage.binary_fill_holes(mask, structure=C8)
        holes = filled & ~mask
        if not holes.any():
            continue
        hl, nh = ndimage.label(holes, structure=C8)
        n_holes += int(nh)
        # attribute each hole to the surrounding component: dilate into the mask.
        touch = ndimage.binary_dilation(holes, structure=C8) & mask
        holed.update(np.unique(comp[touch]).tolist())
    holed.discard(0)
    px_in_holed = int(np.isin(comp, list(holed)).sum()) if holed else 0
    return {"n_holes": n_holes, "n_holed_components": len(holed), "px_in_holed": px_in_holed}


def _critical_census(m: np.ndarray, bnd: np.ndarray) -> dict:
    """F1a/F1b -- critical-set dimension, plateaus, duplicated critical values."""
    nmax = ndimage.maximum_filter(m, footprint=NBR8, mode="nearest")
    nmin = ndimage.minimum_filter(m, footprint=NBR8, mode="nearest")

    strict_max = m > nmax
    strict_min = m < nmin
    weak_max = m >= nmax
    weak_min = m <= nmin
    tie = (m == nmax) | (m == nmin)

    # F1a: is the (weak) minimum set 0-dimensional?
    wl, wn = ndimage.label(weak_min, structure=C8)
    px_in_big = 0
    if wn:
        sizes = np.bincount(wl.ravel())
        sizes[0] = 0
        px_in_big = int(sizes[sizes >= 4].sum())
    n_weak_min = int(weak_min.sum())

    # F1b: duplicated critical VALUES among strict maxima (MS complex needs distinct).
    vals = m[strict_max]
    n_dup = 0
    if vals.size:
        u, c = np.unique(vals, return_counts=True)
        n_dup = int(c[c > 1].sum())

    crit = strict_max | strict_min
    n_crit = int(crit.sum())
    return {
        "n_strict_max": int(strict_max.sum()),
        "n_strict_min": int(strict_min.sum()),
        "n_weak_min": n_weak_min,
        "n_weak_max": int(weak_max.sum()),
        "weakmin_px_in_comp_ge4": px_in_big,
        "weakmin_components": int(wn),
        "n_crit": n_crit,
        "n_crit_with_tie": int((crit & tie).sum()),
        "n_strict_max_dupval": n_dup,
        "n_weak_min_on_bnd": int((weak_min & bnd).sum()),
        "n_bnd": int(bnd.sum()),
    }


def _steepest_ascent_roots(m: np.ndarray) -> np.ndarray:
    """Forman discrete-gradient descending manifolds via steepest-ascent V-paths.

    Each pixel points at its strictly-greatest 8-neighbour (deterministic first-wins tie
    break); pixels that are their own maximum are fixed points.  Pointer-doubling resolves
    every path to its root in O(log depth) vectorised gathers.
    """
    h, w = m.shape
    flat_idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    best_val = m.copy()
    best_idx = flat_idx.copy()
    for dy, dx in _OFFSETS8:
        sv = _shift(m, dy, dx, -np.inf)
        si = _shift(flat_idx, dy, dx, -1)
        take = sv > best_val
        best_val = np.where(take, sv, best_val)
        best_idx = np.where(take, si, best_idx)
    parent = best_idx.ravel().copy()
    # pointer doubling to the fixed point
    for _ in range(24):
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            break
        parent = nxt
    return parent.reshape(h, w)


def _partition_agreement(cells: np.ndarray, bnd: np.ndarray, comp: np.ndarray) -> dict:
    """F3/F5 -- boundary precision/recall of a candidate cell partition vs argmax."""
    cbnd = _boundary_mask(cells)
    inter = int((cbnd & bnd).sum())
    n_c, n_b = int(cbnd.sum()), int(bnd.sum())
    # asymmetric refinement purity: is each cell inside ONE argmax component?
    # (fully vectorised: joint (cell, comp) histogram, then max-per-cell via reduceat)
    ncomp1 = int(comp.max()) + 1
    key = cells.ravel().astype(np.int64) * ncomp1 + comp.ravel().astype(np.int64)
    u, c = np.unique(key, return_counts=True)
    cell_of_u = u // ncomp1
    gstart = np.flatnonzero(np.r_[True, cell_of_u[1:] != cell_of_u[:-1]])
    pure = int(np.maximum.reduceat(c, gstart).sum())
    n_cells = int(gstart.size)
    return {
        "cell_bnd_px": n_c,
        "argmax_bnd_px": n_b,
        "bnd_inter": inter,
        "n_cells": n_cells,
        "refine_pure_px": pure,
    }


def _persistence_curve(m: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """beta_0 of the superlevel filtration {m >= t}: number of alive 0-classes at t."""
    out = np.empty(levels.size, dtype=np.int64)
    for i, t in enumerate(levels):
        _, n = ndimage.label(m >= t, structure=C8)
        out[i] = n
    return out


def _asymmetry_profiles(labels: np.ndarray, m: np.ndarray, depth: int, acc: dict) -> None:
    """F4 -- transverse margin depth profile per ORDERED class pair (a -> into a's side)."""
    for a, b in F4_PAIRS:
        ma, mb = labels == a, labels == b
        if not ma.any() or not mb.any():
            continue
        for src, dst, key in ((ma, mb, (a, b)), (mb, ma, (b, a))):
            seed = src & ndimage.binary_dilation(dst, structure=C4)
            if not seed.any():
                continue
            cur = seed
            seen = seed.copy()
            row = acc.setdefault(key, [np.zeros(depth), np.zeros(depth, dtype=np.int64)])
            for d in range(depth):
                n = int(cur.sum())
                if n == 0:
                    break
                row[0][d] += float(m[cur].sum())
                row[1][d] += n
                nxt = ndimage.binary_dilation(cur, structure=C4) & src & ~seen
                seen |= nxt
                cur = nxt


def _component_census(comp: np.ndarray, n_comp: int, bnd: np.ndarray) -> tuple:
    """Per-component area + boundary-pixel perimeter (displacement-carrier input)."""
    if n_comp == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    area = np.bincount(comp.ravel(), minlength=n_comp + 1)[1:]
    peri = np.bincount(comp[bnd].ravel(), minlength=n_comp + 1)[1:]
    return area, peri


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-cache", type=Path, default=GT_CACHE)
    ap.add_argument("--frames", type=int, default=0, help="0 = ALL (n600, the only evidence tier)")
    ap.add_argument("--depth", type=int, default=8, help="F4 transverse profile depth in px")
    ap.add_argument("--out", type=Path, default=REPO / ".omx/research/ddm_mf1_morse_probe_n600.json")
    args = ap.parse_args(argv)

    if not args.gt_cache.exists():
        print(f"FATAL: gt cache absent: {args.gt_cache}", file=sys.stderr)
        return 2

    t0 = time.time()
    z = np.load(args.gt_cache)
    labels_all = z["lstars"].astype(np.int8)
    margins_all = z["margins"]
    P = labels_all.shape[0] if args.frames <= 0 else min(args.frames, labels_all.shape[0])
    H, W = labels_all.shape[1], labels_all.shape[2]
    print(f"[mf1] n_pairs={P} grid={W}x{H} load={time.time()-t0:.1f}s", flush=True)

    levels = np.concatenate([[0.0], np.geomspace(0.02, 40.0, 39)]).astype(np.float32)

    tot = {
        "PX": 0, "bnd_px": 0, "crack_len": 0, "n_components": 0,
        "n_holes": 0, "n_holed_components": 0, "px_in_holed": 0,
        "n_strict_max": 0, "n_strict_min": 0, "n_weak_min": 0, "n_weak_max": 0,
        "weakmin_px_in_comp_ge4": 0, "weakmin_components": 0,
        "n_crit": 0, "n_crit_with_tie": 0, "n_strict_max_dupval": 0,
        "n_weak_min_on_bnd": 0,
        "h0_cell_bnd_px": 0, "h0_bnd_inter": 0, "h0_n_cells": 0, "h0_refine_pure_px": 0,
        "hs_cell_bnd_px": 0, "hs_bnd_inter": 0, "hs_n_cells": 0, "hs_refine_pure_px": 0,
        "hs_unassigned_px": 0,
        "triple_pts": 0, "quad_pts": 0,
    }
    beta0 = np.zeros(levels.size, dtype=np.float64)
    hstar = []
    ceil_min: list[float] = []
    ceil_prec: list[float] = []
    ceil_rec: list[float] = []
    ceil_t: list[float] = []
    f4: dict = {}
    areas: list[np.ndarray] = []
    peris: list[np.ndarray] = []
    clsz: list[np.ndarray] = []
    m_all_q = []

    for i in range(P):
        L = labels_all[i]
        M = margins_all[i].astype(np.float32)
        bnd = _boundary_mask(L)
        comp, n_comp, ccls = _components(L)

        tot["PX"] += H * W
        tot["bnd_px"] += int(bnd.sum())
        tot["crack_len"] += _crack_length(L)
        tot["n_components"] += n_comp

        hc = _hole_census(L, comp)
        for k in ("n_holes", "n_holed_components", "px_in_holed"):
            tot[k] += hc[k]

        cc = _critical_census(M, bnd)
        for k in ("n_strict_max", "n_strict_min", "n_weak_min", "n_weak_max",
                  "weakmin_px_in_comp_ge4", "weakmin_components", "n_crit",
                  "n_crit_with_tie", "n_strict_max_dupval", "n_weak_min_on_bnd"):
            tot[k] += cc[k]

        # junction degree: 2x2 blocks holding 3 vs 4 distinct labels
        blk = np.stack([L[:-1, :-1], L[:-1, 1:], L[1:, :-1], L[1:, 1:]], axis=0)
        ndist = np.zeros(blk.shape[1:], dtype=np.int8)
        for k in range(5):
            ndist += (blk == k).any(axis=0)
        tot["triple_pts"] += int((ndist == 3).sum())
        tot["quad_pts"] += int((ndist >= 4).sum())

        root = _steepest_ascent_roots(M)
        ag0 = _partition_agreement(root, bnd, comp)
        tot["h0_cell_bnd_px"] += ag0["cell_bnd_px"]
        tot["h0_bnd_inter"] += ag0["bnd_inter"]
        tot["h0_n_cells"] += ag0["n_cells"]
        tot["h0_refine_pure_px"] += ag0["refine_pure_px"]

        b0 = _persistence_curve(M, levels)
        beta0 += b0
        # h*: beta_0(t) is unimodal in t (born at maxima as t falls, then merge to 1 at
        # t=0).  The SIMPLIFIED branch is the low-t one: raising t off zero severs the
        # near-boundary valley and splits the frame into ~the argmax regions.  h* = the
        # smallest t>0 whose alive-class count first reaches n_comp.  This is the MOST
        # FAVOURABLE construction for the coincidence hypothesis and is chosen as such.
        pos = np.flatnonzero((levels > 0) & (b0 >= n_comp))
        j = int(pos[0]) if pos.size else int(np.argmax(b0))
        ts = float(levels[j])
        hstar.append(ts)
        seeds, _ = ndimage.label(ts <= M, structure=C8)
        simp = seeds.ravel()[root.ravel()].reshape(H, W)
        # pixels whose steepest-ascent root peaks BELOW h* fall outside every seed and
        # are pooled into cell 0 -- reported so the agreement number cannot hide them.
        tot["hs_unassigned_px"] += int((simp == 0).sum())
        ags = _partition_agreement(simp, bnd, comp)
        tot["hs_cell_bnd_px"] += ags["cell_bnd_px"]
        tot["hs_bnd_inter"] += ags["bnd_inter"]
        tot["hs_n_cells"] += ags["n_cells"]
        tot["hs_refine_pure_px"] += ags["refine_pure_px"]

        # STEELMAN: instead of trusting one h*, sweep the WHOLE ladder and keep the
        # per-frame BEST min(precision, recall).  Choosing t per frame with knowledge of
        # the answer is unfairly favourable to the coincidence hypothesis on purpose --
        # it is a CEILING, so a ceiling below the bar refutes the hypothesis outright.
        nb_f = int(bnd.sum())
        best = (-1.0, 0.0, 0.0, 0.0)
        for tl in levels[levels > 0].tolist():
            sd, _ = ndimage.label(tl <= M, structure=C8)
            sp = sd.ravel()[root.ravel()].reshape(H, W)
            cb = _boundary_mask(sp)
            ncb = int(cb.sum())
            it = int((cb & bnd).sum())
            pp, rr = it / max(ncb, 1), it / max(nb_f, 1)
            if min(pp, rr) > best[0]:
                best = (min(pp, rr), pp, rr, tl)
        ceil_min.append(best[0])
        ceil_prec.append(best[1])
        ceil_rec.append(best[2])
        ceil_t.append(best[3])

        _asymmetry_profiles(L, M, args.depth, f4)
        a, p = _component_census(comp, n_comp, bnd)
        areas.append(a)
        peris.append(p)
        clsz.append(ccls)
        m_all_q.append(np.quantile(M, [0.05, 0.5, 0.95]))

        if (i + 1) % 50 == 0:
            print(f"[mf1] {i+1}/{P} elapsed={time.time()-t0:.0f}s", flush=True)

    area = np.concatenate(areas)
    peri = np.concatenate(peris)
    ccl = np.concatenate(clsz)
    f4_out = {}
    for (a, b), (s, c) in sorted(f4.items()):
        prof = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        ok = c > 0
        xs = np.arange(args.depth)
        far = float(np.polyfit(xs[ok], prof[ok], 1)[0]) if ok.sum() >= 3 else float("nan")
        # NEAR-FIELD slope: a thin class (Lane is 0.59% of area) TRUNCATES the profile,
        # so an 8-bin fit confounds wall shape with class WIDTH.  Restrict to the first
        # bins that still carry >=20% of the seed-bin support -- truncation-free.
        sup = c / max(int(c[0]), 1)
        okn = (xs < 4) & (sup >= 0.2) & ok
        near = float(np.polyfit(xs[okn], prof[okn], 1)[0]) if okn.sum() >= 3 else float("nan")
        f4_out[f"{CLASS_NAMES[a]}->{CLASS_NAMES[b]}"] = {
            "profile_mean_margin": [None if not o else float(v) for v, o in zip(prof, ok, strict=True)],
            "counts": c.tolist(),
            "support_frac": sup.tolist(),
            "slope_far_8bin_per_px": far,
            "slope_near_per_px": near,
            "near_bins_used": int(okn.sum()),
            "step01": float(prof[1] - prof[0]) if ok[1] else None,
            "step12": float(prof[2] - prof[1]) if ok[2] else None,
        }

    res = {
        "arm": "ddm_mf1",
        "authority": "macOS-CPU advisory (frozen CPU-torch SegNet cached fields); score_claim=false",
        "source": str(args.gt_cache.relative_to(REPO)),
        "n_pairs": P,
        "grid": [W, H],
        "totals": tot,
        "beta0_levels": levels.tolist(),
        "beta0_mean_per_frame": (beta0 / P).tolist(),
        "hstar_mean": float(np.mean(hstar)),
        "hstar_median": float(np.median(hstar)),
        "ceiling_per_frame_optimal_threshold": {
            "mean_min_prec_rec": float(np.mean(ceil_min)),
            "max_min_prec_rec": float(np.max(ceil_min)),
            "mean_precision": float(np.mean(ceil_prec)),
            "mean_recall": float(np.mean(ceil_rec)),
            "mean_t": float(np.mean(ceil_t)),
        },
        "margin_quantiles_mean": np.mean(np.stack(m_all_q), axis=0).tolist(),
        "component_census": {
            "n_total": int(area.size),
            "per_frame": float(area.size / P),
            "area_quantiles": np.quantile(area, [0.5, 0.9, 0.99]).tolist(),
            "n_area_ge_64": int((area >= 64).sum()),
            "n_area_ge_256": int((area >= 256).sum()),
            "perimeter_sum_area_ge_64": int(peri[area >= 64].sum()),
            "perimeter_sum_area_ge_256": int(peri[area >= 256].sum()),
            "perimeter_mean_area_ge_64": float(peri[area >= 64].mean()) if (area >= 64).any() else 0.0,
        "per_class": {
            CLASS_NAMES[k]: {
                "n_total": int((ccl == k).sum()),
                "n_area_ge_64": int(((ccl == k) & (area >= 64)).sum()),
                "perimeter_px_sum_area_ge_64": int(peri[(ccl == k) & (area >= 64)].sum()),
                "area_px_sum": int(area[ccl == k].sum()),
            }
            for k in range(5)
        },
        },
        "f4_asymmetry": f4_out,
        "elapsed_s": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"[mf1] wrote {args.out} in {res['elapsed_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
