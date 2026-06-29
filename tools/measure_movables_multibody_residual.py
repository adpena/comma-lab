#!/usr/bin/env python3
"""GAP1 — the MOVABLES (class-3) multi-body residual: MEASURE + ECONOMICS.

DAG FEED-iw/ja follow-up. The ego-pose stratified warp explains the rigid scene
(Road / hood / sky / lane move with ego-motion via a depth-stratified homography);
MOVABLES (class 3, cars) move INDEPENDENTLY of ego-motion, so they are the *second*
residual term of the v2 witness. This tool bounds that term at $0 on cached GT and
asks: is it per-object-low-rank, and is multi-body per-object-pose modelling worth
its bytes, or do we ACCEPT the residual?

Authority / NO-FAKE: this is an ADVISORY structural measurement on the FROZEN
CPU-torch SegNet argmax (the same authority that built the GT cache; f1-seg recompute
is bit-exact vs cache). It is NOT a contest score. The frontier pointer is UNMOVED
(0.19110). It bounds a residual; it does not move it.

What it measures (within-pair, the TRUE contest cadence f0->f1, 1 frame apart):
  - object inventory: class-3 connected components per frame (count, area).
  - within-pair class-3 prediction residual under four witness variants, as the
    movables-attributable d_seg (symmetric-difference / scorer-pixels):
        V0  no movable model        (predict empty)           -> = class-3 area
        V1  temporal persistence    (copy the f0 mask)
        V2  per-object translation  (oracle dy,dx per object)
        V3  per-object translate+scale (oracle dy,dx,scale)   -> per-object RIGID floor
  - births/deaths: f1 movable px with no f0 correspondence (un-modelable by warp).
  - margin at class-3 boundary vs interior (flip-proneness / irreducibility).
  - [--consecutive] cross-frame tracking -> trajectory low-rank (polynomial RMS),
    persistence (track lengths). Only valid when the cache holds CONSECUTIVE pairs.
  - ECONOMICS: the multi-body prize (V1->V3), the irreducible floor (V3), and the
    rate/d_seg break-even in archive bytes.

Run (≈35s for n96 incl. f0-seg recompute; cached thereafter):
  PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/measure_movables_multibody_residual.py \
      --cache experiments/results/mlx_fleet_gt_cache/gt_n96.npz --consecutive
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

MOVABLE = 3  # comma10k canonical order [Road,Lane,Undrivable,Movable,MyCar] (CLAUDE.md, MEASURED)
SEG_H, SEG_W = 384, 512
SCORER_PX = SEG_H * SEG_W  # 196_608

# Contest score S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/RATE_DENOM.
RATE_DENOM = 37_545_489
D_SEG_WEIGHT = 100.0
BYTES_PER_S_POINT = RATE_DENOM / 25.0  # 1,501,819.56 bytes == 1 S-point of rate
S_PER_BYTE = 25.0 / RATE_DENOM


# --------------------------------------------------------------------------- #
# f0-seg via the canonical frozen CPU-torch authority (same as the GT cache).
# --------------------------------------------------------------------------- #
def compute_f0_segs(gt_f0: np.ndarray, sidecar: Path | None) -> np.ndarray:
    if sidecar is not None and sidecar.exists():
        cached = np.load(sidecar)["lstars_f0"]
        if cached.shape[0] == gt_f0.shape[0]:
            return cached
    from tac.boundary_math.seg_core import (  # noqa: E402  (heavy torch import, lazy)
        load_real_segnet,
        segnet_argmax_and_margin,
    )

    seg = load_real_segnet("cpu")
    P = gt_f0.shape[0]
    out = np.zeros((P, SEG_H, SEG_W), np.int64)
    for i in range(P):
        lstar, _margin = segnet_argmax_and_margin(seg, gt_f0[i])
        out[i] = np.asarray(lstar, np.int64)
    if sidecar is not None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        np.savez(sidecar, lstars_f0=out)
    return out


# --------------------------------------------------------------------------- #
# connected-component helpers
# --------------------------------------------------------------------------- #
def verify_movable_class(f1seg: np.ndarray) -> dict:
    """Self-detect / validate that class index ``MOVABLE`` matches the Movable signature.

    CLAUDE.md (MEASURED 2026-06-27): canonical comma10k order is
    [Road, Lane, Undrivable, Movable, MyCar]; class 3 = Movable, ~1.56% area, mid-band
    vertical centroid (rows ~174-215 of 384). Per the class-index NON-NEGOTIABLE, never
    blind-hardcode -- assert the signature so a wrong-order cache fails closed.
    """
    m = f1seg == MOVABLE
    frac = float(m.mean())
    row_counts = m.sum(axis=(0, 2)).astype(float)  # (SEG_H,) movable px per row, summed over frames
    cy = float((row_counts * np.arange(SEG_H)).sum() / max(1.0, row_counts.sum()))  # area-weighted row
    ok = (0.003 <= frac <= 0.05) and (120 <= cy <= 260)
    if not ok:
        raise SystemExit(
            f"[movables] class index {MOVABLE} fails Movable signature "
            f"(area_frac={frac:.4f} expect 0.005-0.03; mean_row={cy:.0f} expect ~140-240). "
            "Cache may use a non-canonical class order -- refusing to mislabel (CLAUDE.md class-index rule)."
        )
    return {"class_index": MOVABLE, "area_frac": frac, "mean_row": cy, "signature_ok": ok}


def components(mask: np.ndarray, min_area: int) -> list[dict]:
    lab, n = ndimage.label(mask)
    out: list[dict] = []
    if not n:
        return out
    areas = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    cents = ndimage.center_of_mass(mask, lab, range(1, n + 1))
    for k in range(n):
        if areas[k] < min_area:
            continue
        out.append(
            {"lab": k + 1, "area": int(areas[k]), "cy": float(cents[k][0]),
             "cx": float(cents[k][1]), "labimg": lab}
        )
    return out


def _shift(mask: np.ndarray, dy: float, dx: float) -> np.ndarray:
    return ndimage.shift(mask.astype(np.float32), (dy, dx), order=0, mode="constant", cval=0) > 0.5


def _scale_translate(mask: np.ndarray, cy: float, cx: float, s: float, dy: float, dx: float) -> np.ndarray:
    # output = scale*(input-c)+c+shift  =>  input = (output - c - shift)/scale + c  (inverse map)
    M = np.array([[1.0 / s, 0.0], [0.0, 1.0 / s]])
    off = np.array([cy - (cy + dy) / s, cx - (cx + dx) / s])
    return ndimage.affine_transform(mask.astype(np.float32), M, offset=off, order=0, mode="constant", cval=0) > 0.5


def _match(c0: list[dict], c1: list[dict], max_dist: float = 40.0, area_ratio: float = 3.0):
    """Greedy nearest-centroid + area-gated matching c0 -> c1. Returns list of (a, b)."""
    used: set[int] = set()
    pairs = []
    for a in c0:
        best, bd = None, 1e9
        for j, b in enumerate(c1):
            if j in used:
                continue
            dd = ((a["cy"] - b["cy"]) ** 2 + (a["cx"] - b["cx"]) ** 2) ** 0.5
            ar = max(a["area"], b["area"]) / max(1, min(a["area"], b["area"]))
            if dd < bd and dd < max_dist and ar < area_ratio:
                bd, best = dd, j
        if best is not None:
            used.add(best)
            pairs.append((a, c1[best]))
    matched_c1 = {id(b) for _, b in pairs}
    return pairs, matched_c1


# --------------------------------------------------------------------------- #
# within-pair (contest cadence) movables residual decomposition
# --------------------------------------------------------------------------- #
def within_pair_residual(f0seg: np.ndarray, f1seg: np.ndarray, min_area: int) -> dict:
    P = f1seg.shape[0]
    V = {"V0": [], "V1": [], "V2": [], "V3": []}
    born = []
    nobj = []
    for i in range(P):
        m0 = f0seg[i] == MOVABLE
        m1 = f1seg[i] == MOVABLE
        a1 = int(m1.sum())
        nobj.append(len(components(m1, min_area)))
        V["V0"].append(a1 / SCORER_PX)
        V["V1"].append(int((m0 ^ m1).sum()) / SCORER_PX)
        c0 = components(m0, min_area)
        c1 = components(m1, min_area)
        pairs, matched_c1 = _match(c0, c1)
        p2 = np.zeros_like(m1)
        p3 = np.zeros_like(m1)
        for a, b in pairs:
            objm = a["labimg"] == a["lab"]
            dy, dx = b["cy"] - a["cy"], b["cx"] - a["cx"]
            p2 |= _shift(objm, dy, dx)
            s = (b["area"] / max(1, a["area"])) ** 0.5
            p3 |= _scale_translate(objm, a["cy"], a["cx"], s, dy, dx)
        V["V2"].append(int((p2 ^ m1).sum()) / SCORER_PX)
        V["V3"].append(int((p3 ^ m1).sum()) / SCORER_PX)
        born.append(sum(b["area"] for b in c1 if id(b) not in matched_c1) / SCORER_PX)

    def st(a):
        a = np.asarray(a, float)
        return {"mean": float(a.mean()), "median": float(np.median(a)), "max": float(a.max()), "min": float(a.min())}

    return {
        "pred_residual_dseg": {k: st(v) for k, v in V.items()},
        "born_unmatched_f1_frac": st(born),
        "objs_per_frame": st(nobj),
        "_means": {k: float(np.mean(v)) for k, v in V.items()},
    }


def margin_boundary(f1seg: np.ndarray, margins: np.ndarray) -> dict:
    bm, im = [], []
    for i in range(f1seg.shape[0]):
        m = f1seg[i] == MOVABLE
        er = ndimage.binary_erosion(m, iterations=1)
        bm.append(margins[i][m & ~er])
        im.append(margins[i][er])
    bm = np.concatenate(bm) if bm else np.array([0.0])
    im = np.concatenate(im) if im else np.array([0.0])

    def s(x):
        return {"n": int(x.size), "median": float(np.median(x)),
                "frac_lt_0p5": float((x < 0.5).mean()), "frac_lt_1": float((x < 1.0).mean())}

    return {"boundary": s(bm), "interior": s(im)}


def trajectory_lowrank(f1seg: np.ndarray, min_area: int) -> dict:
    """Cross-frame tracking on CONSECUTIVE f1 segs -> persistence + polynomial RMS."""
    per_frame = [components(f1seg[i] == MOVABLE, min_area) for i in range(f1seg.shape[0])]
    tracks: list[list[tuple[int, dict]]] = []
    active: list[tuple[int, int, dict]] = []
    for i, comps in enumerate(per_frame):
        used: set[int] = set()
        new_active = []
        for ti, lf, lc in active:
            if i - lf > 2:
                continue
            best, bd = None, 1e9
            for j, c in enumerate(comps):
                if j in used:
                    continue
                dd = ((c["cy"] - lc["cy"]) ** 2 + (c["cx"] - lc["cx"]) ** 2) ** 0.5
                ar = max(c["area"], lc["area"]) / max(1, min(c["area"], lc["area"]))
                if dd < bd and dd < 40 and ar < 3.0:
                    bd, best = dd, j
            if best is not None:
                used.add(best)
                tracks[ti].append((i, comps[best]))
                new_active.append((ti, i, comps[best]))
            else:
                new_active.append((ti, lf, lc))
        for j, c in enumerate(comps):
            if j in used:
                continue
            tracks.append([(i, c)])
            new_active.append((len(tracks) - 1, i, c))
        active = new_active

    lens = [len(t) for t in tracks]
    d1, d2 = [], []
    for t in tracks:
        if len(t) < 4:
            continue
        tt = np.array([f for f, _ in t], float)
        cy = np.array([c["cy"] for _, c in t])
        cx = np.array([c["cx"] for _, c in t])
        for deg, dst in ((1, d1), (2, d2)):
            py, px = np.polyfit(tt, cy, deg), np.polyfit(tt, cx, deg)
            ry, rx = cy - np.polyval(py, tt), cx - np.polyval(px, tt)
            dst.append(float(np.sqrt((ry ** 2 + rx ** 2).mean())))

    def st(a):
        a = np.asarray(a, float)
        return None if a.size == 0 else {"mean": float(a.mean()), "median": float(np.median(a)), "max": float(a.max())}

    return {
        "n_tracks": len(tracks),
        "track_len": {"mean": float(np.mean(lens)), "median": float(np.median(lens)),
                      "max": int(np.max(lens)), "frac_singletons": float(np.mean(np.array(lens) == 1))},
        "centroid_poly_rms_px": {"deg1": st(d1), "deg2": st(d2)},
    }


# --------------------------------------------------------------------------- #
# economics
# --------------------------------------------------------------------------- #
def economics(means: dict) -> dict:
    floor_v3 = means["V3"]
    prize = means["V1"] - means["V3"]  # what per-object RIGID buys over naive copy
    prize_s = D_SEG_WEIGHT * prize
    breakeven_bytes = prize_s * BYTES_PER_S_POINT
    # cheap multi-body sidecar cost model: K objects over drive, low-rank trajectory.
    # per object: deg-2 trajectory * (dy,dx,scale) = 9 coeffs * ~2 bytes ~= 18 bytes.
    est = {f"K={K}": {"bytes": K * 18, "net_S_if_full_prize": prize_s - K * 18 * S_PER_BYTE}
           for K in (50, 150, 600)}
    return {
        "rate_constants": {"bytes_per_S_point": round(BYTES_PER_S_POINT, 1), "S_per_byte": S_PER_BYTE,
                           "d_seg_weight": D_SEG_WEIGHT},
        "irreducible_floor_V3_dseg": floor_v3,
        "irreducible_floor_V3_S": D_SEG_WEIGHT * floor_v3,
        "multibody_prize_V1_minus_V3_dseg": prize,
        "multibody_prize_S_max": prize_s,
        "breakeven_byte_budget_for_full_prize": round(breakeven_bytes, 0),
        "cheap_sidecar_cost_model": est,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    ap.add_argument("--min-area", type=int, default=5, help="drop CC smaller than this (noise)")
    ap.add_argument("--consecutive", action="store_true",
                    help="cache holds consecutive pairs -> compute trajectory low-rank")
    ap.add_argument("--f0-seg-cache", default="", help="sidecar npz for recomputed f0 segs")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    t0 = time.time()
    d = np.load(args.cache)
    f1seg = d["lstars"]
    margins = d["margins"]
    gt_f0 = d["gt_f0"]
    if args.max_pairs:
        f1seg, margins, gt_f0 = f1seg[: args.max_pairs], margins[: args.max_pairs], gt_f0[: args.max_pairs]

    # f0-seg recompute is deterministic + rebuildable -> cache to ephemeral .omx/tmp
    # (CLAUDE.md disk-hygiene: ".omx/tmp/ for explicitly ephemeral local scratch"; never
    # litter experiments/results with rebuildable bulk).
    sidecar = Path(args.f0_seg_cache) if args.f0_seg_cache else (
        Path(".omx/tmp/movables_f0seg") / (Path(args.cache).stem + "_f0seg.npz"))
    f0seg = compute_f0_segs(gt_f0, sidecar)

    class_check = verify_movable_class(f1seg)
    wp = within_pair_residual(f0seg, f1seg, args.min_area)
    out = {
        "tool": "measure_movables_multibody_residual",
        "authority": "frozen-CPU-torch SegNet argmax (advisory; NOT a contest score; pointer UNMOVED 0.19110)",
        "cache": args.cache,
        "P": int(f1seg.shape[0]),
        "cadence": "within-pair 1-frame (contest f0->f1)",
        "secs": round(time.time() - t0, 1),
        "class3_area_frac_mean": float((f1seg == MOVABLE).mean()),
        "movable_class_signature": class_check,
        "within_pair": {k: v for k, v in wp.items() if k != "_means"},
        "margin": margin_boundary(f1seg, margins),
        "economics": economics(wp["_means"]),
    }
    if args.consecutive:
        out["trajectory_lowrank"] = trajectory_lowrank(f1seg, args.min_area)

    txt = json.dumps(out, indent=2)
    print(txt)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt)


if __name__ == "__main__":
    main()
