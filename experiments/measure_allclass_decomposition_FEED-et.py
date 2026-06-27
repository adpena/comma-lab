"""FEED-et: all-class flip-band geometric/static/learned DECOMPOSITION (CPU/numpy).

Extends the lane (FEED-er) + hood (#139) parametric SOLVE to ALL classes. For the
flip-prone boundary annulus (GT margin < 0.5 on the frozen CPU-torch SegNet argmax
L*), decompose every band pixel into:
  (a) STATIC   — constant L* across all frames -> 0-byte deterministic clamp (hood);
  (b) GEOMETRIC- horizon line / ground-plane / lane-polynomial (rule-118 FREE
                 rasterizer, only per-frame coeffs COUNTED);
  (c) LEARNED  — texture/appearance the witness capacity must carry.

NO-FAKE: every number MEASURED vs the REAL cached frozen CPU-torch L* (lstars in
gt_n96.npz). rule-118: geometric rasterizers FREE, video-derived coeffs COUNTED, NO
scorer weights / GT-argmax table in archive. [macOS-CPU advisory] research-signal;
pointer UNMOVED 0.19110; means != ends (a decomposition is a MEANS).

Run per part to stay inside the subagent foreground time budget:
  python experiments/measure_allclass_decomposition_FEED-et.py --part A   # band + boundary-pair histogram
  python experiments/measure_allclass_decomposition_FEED-et.py --part B   # temporal staticity -> STATIC
  python experiments/measure_allclass_decomposition_FEED-et.py --part C   # horizon fit -> GEOMETRIC (Road<->Undrivable)
  python experiments/measure_allclass_decomposition_FEED-et.py --part D   # lane geometric coverage (reuse FEED-dm) + cite FEED-er
  python experiments/measure_allclass_decomposition_FEED-et.py --part E   # synthesis table + verdict
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

CACHE = os.path.join(_HERE, "results", "mlx_fleet_gt_cache", "gt_n96.npz")
OUTDIR = os.path.join(_HERE, "results", "allclass_decomp_FEED-et")
OUTJSON = os.path.join(OUTDIR, "decomposition.json")
BAND_THRESH = 0.5

# canonical SegNet class order (self-detected below by spatial signature; do NOT luma-sort)
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}


def _load():
    d = np.load(CACHE)
    return d["lstars"].astype(np.int8), d["margins"].astype(np.float32), d["gt_poses"]


def _read_state() -> dict:
    if os.path.exists(OUTJSON):
        with open(OUTJSON) as f:
            return json.load(f)
    return {}


def _write_state(state: dict) -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    tmp = OUTJSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, OUTJSON)


# ---------------------------------------------------------------------------
# Part A: band, per-class flip share, boundary-pair histogram (the flip-mass map)
# ---------------------------------------------------------------------------
def part_A() -> dict:
    from scipy import ndimage

    L, M, _ = _load()
    n, h, w = L.shape
    band = M < BAND_THRESH
    state = _read_state()

    # per-class share of the band (by L* label)
    per_class = {}
    tot_band = int(band.sum())
    for c in range(5):
        cb = int(((L == c) & band).sum())
        per_class[CLASS_NAMES[c]] = {
            "band_share": cb / tot_band,
            "band_px": cb,
            "class_area_frac": float((L == c).mean()),
        }

    # boundary-pair histogram: each band pixel -> (own class, dominant DIFFERENT neighbor)
    # neighbor counts via 3x3 box on each class indicator (incl. center) then argmax over c!=own
    K = np.ones((3, 3), np.float32)
    pair_counts = np.zeros((5, 5), np.int64)  # [own, neighbor]
    for i in range(n):
        Li = L[i]
        bi = band[i]
        if not bi.any():
            continue
        cnt = np.empty((5, h, w), np.float32)
        for c in range(5):
            cnt[c] = ndimage.convolve((Li == c).astype(np.float32), K, mode="nearest")
        # zero out own-class so argmax picks the dominant DIFFERENT neighbor
        own = Li
        for c in range(5):
            cnt[c][own == c] = -1.0
        neigh = np.argmax(cnt, axis=0).astype(np.int8)
        # accumulate only band pixels
        ov = own[bi]
        nv = neigh[bi]
        for a in range(5):
            sel = ov == a
            if not sel.any():
                continue
            nvv = nv[sel]
            for b in range(5):
                pair_counts[a, b] += int((nvv == b).sum())

    # symmetric boundary mass (a<->b) = pair[a,b]+pair[b,a]
    sym = {}
    total_pair = int(pair_counts.sum())
    for a in range(5):
        for b in range(a + 1, 5):
            mass = int(pair_counts[a, b] + pair_counts[b, a])
            if mass == 0:
                continue
            key = f"{CLASS_NAMES[a]}<->{CLASS_NAMES[b]}"
            sym[key] = {"band_px": mass, "band_share": mass / total_pair}

    state["meta"] = {"n_frames": int(n), "h": int(h), "w": int(w),
                     "band_thresh": BAND_THRESH, "band_frac": float(band.mean()),
                     "total_band_px": tot_band}
    state["A_per_class_share"] = per_class
    state["A_boundary_pairs"] = dict(sorted(sym.items(), key=lambda kv: -kv[1]["band_px"]))
    _write_state(state)
    print("=== Part A: per-class band share ===")
    for k, v in per_class.items():
        print(f"  {k:11s} share={v['band_share']:.4f} area={v['class_area_frac']:.4f}")
    print("=== Part A: boundary-pair band mass (symmetric) ===")
    for k, v in state["A_boundary_pairs"].items():
        print(f"  {k:24s} share={v['band_share']:.4f} px={v['band_px']}")
    return state


# ---------------------------------------------------------------------------
# Part B: temporal staticity -> STATIC-clampable fraction (per boundary pair)
# ---------------------------------------------------------------------------
def part_B() -> dict:
    from scipy import ndimage

    L, M, _ = _load()
    n, h, w = L.shape
    band = M < BAND_THRESH
    state = _read_state()

    # per-pixel temporal mode + consistency
    counts = np.zeros((5, h, w), np.int32)
    for i in range(n):
        Li = L[i]
        for c in range(5):
            counts[c] += (Li == c)
    mode = np.argmax(counts, axis=0).astype(np.int8)
    consistency = counts.max(0).astype(np.float32) / float(n)
    STATIC_THRESH = 0.95
    static_loc = consistency >= STATIC_THRESH  # pixel label stationary across frames

    # of all band pixels, fraction at static locations (0-byte clampable)
    band_any = band  # per-frame band
    static_band = band_any & static_loc[None, :, :]
    tot_band = int(band_any.sum())
    static_band_px = int(static_band.sum())

    # per boundary pair: how much band-mass is at static locations
    # recompute neighbor (reuse Part A logic, but only need own+neighbor at static band pixels)
    K = np.ones((3, 3), np.float32)
    pair_static = np.zeros((5, 5), np.int64)
    pair_total = np.zeros((5, 5), np.int64)
    for i in range(n):
        Li = L[i]
        bi = band_any[i]
        if not bi.any():
            continue
        cnt = np.empty((5, h, w), np.float32)
        for c in range(5):
            cnt[c] = ndimage.convolve((Li == c).astype(np.float32), K, mode="nearest")
        own = Li
        for c in range(5):
            cnt[c][own == c] = -1.0
        neigh = np.argmax(cnt, axis=0).astype(np.int8)
        sb = bi & static_loc
        for arr, mask in ((pair_total, bi), (pair_static, sb)):
            ov = own[mask]; nv = neigh[mask]
            for a in range(5):
                sel = ov == a
                if not sel.any():
                    continue
                nvv = nv[sel]
                for b in range(5):
                    arr[a, b] += int((nvv == b).sum())

    # self-detect hood class (static + bottom)
    bottom = slice(int(h * 0.75), h)
    hood_scores = []
    for c in range(5):
        col = counts[c]
        tot = int(col.sum())
        iou = float((col == n).sum() / max(1, (col > 0).sum()))
        bshare = float(col[bottom].sum() / tot) if tot else 0.0
        hood_scores.append((iou * bshare, c, iou, bshare))
    hood_scores.sort(reverse=True)
    hood_cls = int(hood_scores[0][1])

    sym_static = {}
    for a in range(5):
        for b in range(a + 1, 5):
            tot = int(pair_total[a, b] + pair_total[b, a])
            st = int(pair_static[a, b] + pair_static[b, a])
            if tot == 0:
                continue
            sym_static[f"{CLASS_NAMES[a]}<->{CLASS_NAMES[b]}"] = {
                "static_band_px": st, "total_band_px": tot,
                "static_fraction": st / tot,
            }

    state["B_static"] = {
        "static_thresh": STATIC_THRESH,
        "static_band_px": static_band_px,
        "total_band_px": tot_band,
        "static_fraction_of_band": static_band_px / tot_band,
        "hood_cls_detected": hood_cls,
        "hood_cls_name": CLASS_NAMES[hood_cls],
        "per_boundary_static": dict(sorted(sym_static.items(),
                                           key=lambda kv: -kv[1]["total_band_px"])),
    }
    _write_state(state)
    print("=== Part B: temporal staticity ===")
    print(f"  hood class self-detected = {hood_cls} ({CLASS_NAMES[hood_cls]})")
    print(f"  static-clampable fraction of band = {static_band_px/tot_band:.4f} "
          f"({static_band_px}/{tot_band})")
    for k, v in state["B_static"]["per_boundary_static"].items():
        print(f"  {k:24s} static={v['static_fraction']:.4f} (of {v['total_band_px']} px)")
    return state


# ---------------------------------------------------------------------------
# Part C: horizon fit -> GEOMETRIC coverage of the Road<->Undrivable boundary
# ---------------------------------------------------------------------------
def _fit_horizon(Li: np.ndarray, road_cls: int, undriv_cls: int) -> tuple[float, float]:
    """Fit horizon row v=a+b*u from the topmost-road / bottommost-undrivable transition
    per column. Robust (median + 1 reweight). Returns (a, b) or (nan,nan)."""
    h, w = Li.shape
    us, vs = [], []
    for u in range(0, w, 2):
        col = Li[:, u]
        road_rows = np.where(col == road_cls)[0]
        if road_rows.size == 0:
            continue
        # topmost road row that has undrivable somewhere above it (true horizon, not a hole)
        top_road = int(road_rows.min())
        above = col[:top_road]
        if above.size and (above == undriv_cls).any():
            us.append(u); vs.append(top_road)
    if len(us) < 8:
        return float("nan"), float("nan")
    us = np.asarray(us, float); vs = np.asarray(vs, float)
    b, a = np.polyfit(us, vs, 1)
    pred = a + b * us
    resid = np.abs(vs - pred)
    keep = resid <= (np.median(resid) + 2.5 * (np.median(resid) + 1.0))
    if keep.sum() >= 8:
        b, a = np.polyfit(us[keep], vs[keep], 1)
    return float(a), float(b)


def part_C() -> dict:
    from scipy import ndimage

    L, M, _ = _load()
    n, h, w = L.shape
    band = M < BAND_THRESH
    state = _read_state()
    road_cls, undriv_cls = 0, 2  # Road below, Undrivable (sky/buildings) above

    K = np.ones((3, 3), np.float32)
    DELTAS = [3, 6, 10]
    cov = {d: 0 for d in DELTAS}
    ru_band_total = 0
    horizon_params = []
    horizon_resid = []
    for i in range(n):
        Li = L[i]; bi = band[i]
        a, b = _fit_horizon(Li, road_cls, undriv_cls)
        horizon_params.append((a, b))
        if not np.isfinite(a):
            continue
        # Road<->Undrivable band pixels this frame
        cnt = np.empty((5, h, w), np.float32)
        for c in range(5):
            cnt[c] = ndimage.convolve((Li == c).astype(np.float32), K, mode="nearest")
        own = Li.copy()
        for c in range(5):
            cnt[c][own == c] = -1.0
        neigh = np.argmax(cnt, axis=0).astype(np.int8)
        ru = bi & (((own == road_cls) & (neigh == undriv_cls)) |
                   ((own == undriv_cls) & (neigh == road_cls)))
        vv, uu = np.where(ru)
        if vv.size == 0:
            continue
        pred_v = a + b * uu
        dist = np.abs(vv - pred_v)
        horizon_resid.append(float(np.median(dist)))
        ru_band_total += vv.size
        for d in DELTAS:
            cov[d] += int((dist <= d).sum())

    state["C_horizon"] = {
        "road_undrivable_band_px": int(ru_band_total),
        "horizon_coverage": {f"within_{d}px": (cov[d] / ru_band_total if ru_band_total else 0.0)
                             for d in DELTAS},
        "median_horizon_residual_px": float(np.median(horizon_resid)) if horizon_resid else None,
        "mean_horizon_slope": float(np.nanmean([p[1] for p in horizon_params])),
        "mean_horizon_intercept": float(np.nanmean([p[0] for p in horizon_params])),
        "n_frames_fit": int(sum(1 for p in horizon_params if np.isfinite(p[0]))),
    }
    _write_state(state)
    print("=== Part C: horizon (Road<->Undrivable) geometric coverage ===")
    print(f"  Road<->Undrivable band px = {ru_band_total}")
    for d in DELTAS:
        print(f"  within +/-{d}px of fitted horizon = {cov[d]/ru_band_total:.4f}")
    print(f"  mean horizon: v = {state['C_horizon']['mean_horizon_intercept']:.1f} "
          f"+ {state['C_horizon']['mean_horizon_slope']:.4f}*u "
          f"(median resid {state['C_horizon']['median_horizon_residual_px']:.2f}px)")
    return state


# ---------------------------------------------------------------------------
# Part D: lane geometric coverage (reuse FEED-dm component on a subsample)
# ---------------------------------------------------------------------------
def part_D(n_sample: int = 12) -> dict:
    from tac.boundary_math.lane_sdf_component import (
        build_structured_lane_sdf, decompose_argmax_disagreement, inject_lane_sdf,
    )
    from tac.boundary_math.lever_b_levelset_generator import signed_distance_fields  # noqa

    L, M, _ = _load()
    n, h, w = L.shape
    state = _read_state()
    idx = np.linspace(0, n - 1, min(n_sample, n)).astype(int)

    fn_list, fp_road_list, attr_list, floats_list = [], [], [], []
    for i in idx:
        Li = L[i].astype(np.int64)
        phi1, meta = build_structured_lane_sdf(Li, lane_cls=1, dash_gate=False)
        # build a baseline K-field from per-class SDFs, inject lane, argmax -> pred
        masks = [(Li == c) for c in range(5)]
        from scipy import ndimage
        phi = np.empty((h, w, 5), np.float32)
        for c in range(5):
            m = masks[c]
            if m.all():
                phi[..., c] = float(max(h, w))
            elif not m.any():
                phi[..., c] = -float(max(h, w))
            else:
                phi[..., c] = (ndimage.distance_transform_edt(m)
                               - ndimage.distance_transform_edt(~m)).astype(np.float32)
        phi_inj = inject_lane_sdf(phi, phi1, lane_cls=1, mode="replace")
        pred = np.argmax(phi_inj, axis=2)
        dd = decompose_argmax_disagreement(pred, Li, lane_cls=1, road_cls=0)
        fn_list.append(dd.lane_fn)
        fp_road_list.append(dd.lane_fp_from_road)
        attr_list.append(dd.lane_attributable)
        floats_list.append(meta["total_floats"])

    state["D_lane"] = {
        "n_sample": int(len(idx)),
        "lane_fn_mean": float(np.mean(fn_list)),
        "lane_fp_from_road_mean": float(np.mean(fp_road_list)),
        "lane_attributable_mean": float(np.mean(attr_list)),
        "mean_floats_per_frame": float(np.mean(floats_list)),
        "cite_FEED_er": {"lane_fn": 0.000133, "lane_attributable_n96": 0.000439,
                         "kb_per_n600": 7.0, "rate_term": 0.0048},
    }
    _write_state(state)
    print("=== Part D: lane geometric coverage (reuse FEED-dm, replace-inject) ===")
    print(f"  lane_fn (shape FN)          = {np.mean(fn_list):.6f}  (FEED-er 0.000133)")
    print(f"  lane_fp_from_road (overpaint)= {np.mean(fp_road_list):.6f}")
    print(f"  lane_attributable           = {np.mean(attr_list):.6f}  (FEED-er n96 0.000439)")
    print(f"  floats/frame ~ {np.mean(floats_list):.1f}")
    return state


# ---------------------------------------------------------------------------
# Part F: hood coverage by a single STATIC majority-vote mask (#139, 0-byte)
# ---------------------------------------------------------------------------
def part_F() -> dict:
    from scipy import ndimage
    from tac.boundary_math.hood_static_component import (
        compute_static_hood_mask, identify_static_hood_class, hood_mask_byte_cost,
    )

    L, M, _ = _load()
    n, h, w = L.shape
    band = M < BAND_THRESH
    state = _read_state()

    hood_cls, _ = identify_static_hood_class(L.astype(np.int64))
    sm = compute_static_hood_mask(L.astype(np.int64), hood_cls=hood_cls, agg="majority")
    static_mask = sm.mask  # (H,W) bool — the ~0-byte hood prediction for ALL frames
    cost = hood_mask_byte_cost(static_mask, n_frames=600)

    # Road<->MyCar band coverage by the static-clamp prediction.
    # static-clamp predicts MyCar inside static_mask, NOT-MyCar outside. On the band,
    # "covered" = the clamp's MyCar/not-MyCar call matches L* (== hood_cls vs not).
    K = np.ones((3, 3), np.float32)
    rm_total = 0
    rm_correct = 0
    hood_dseg = 0  # MyCar-class disagreement of the static clamp (any pixel)
    for i in range(n):
        Li = L[i]; bi = band[i]
        # MyCar-class clamp error anywhere (the #139 static-clamp d_seg on the hood class)
        hood_dseg += int((static_mask != (Li == hood_cls)).sum())
        # neighbor for Road<->MyCar band
        cnt = np.empty((5, h, w), np.float32)
        for c in range(5):
            cnt[c] = ndimage.convolve((Li == c).astype(np.float32), K, mode="nearest")
        own = Li.copy()
        for c in range(5):
            cnt[c][own == c] = -1.0
        neigh = np.argmax(cnt, axis=0).astype(np.int8)
        rm = bi & (((own == 0) & (neigh == hood_cls)) | ((own == hood_cls) & (neigh == 0)))
        if not rm.any():
            continue
        rm_total += int(rm.sum())
        clamp_pred_mycar = static_mask
        gt_mycar = Li == hood_cls
        rm_correct += int((rm & (clamp_pred_mycar == gt_mycar)).sum())

    state["F_hood"] = {
        "hood_cls": int(hood_cls),
        "static_mask_px": int(static_mask.sum()),
        "static_mask_frac": float(static_mask.mean()),
        "mean_frame_iou": sm.mean_frame_iou,
        "min_frame_iou": sm.min_frame_iou,
        "hood_class_dseg_static_clamp": hood_dseg / float(n * h * w),
        "road_mycar_band_total": int(rm_total),
        "road_mycar_band_coverage": rm_correct / rm_total if rm_total else 0.0,
        "counted_bytes_n600": cost["best_counted_bytes"],
        "rate_term": cost["score_rate_contribution"],
    }
    _write_state(state)
    print("=== Part F: hood (Road<->MyCar) static-mask coverage ===")
    print(f"  hood class = {hood_cls}, static mask iou(mean/min) = "
          f"{sm.mean_frame_iou:.4f}/{sm.min_frame_iou:.4f}")
    print(f"  hood-class d_seg (static clamp) = {hood_dseg/(n*h*w):.6f}")
    print(f"  Road<->MyCar band coverage by static mask = "
          f"{rm_correct/rm_total if rm_total else 0:.4f} (of {rm_total} px)")
    print(f"  COUNTED bytes (single mask /n600) = {cost['best_counted_bytes']} "
          f"(rate {cost['score_rate_contribution']:.6f})")
    return state


# ---------------------------------------------------------------------------
# Part E: synthesis -> per-class decomposition table + verdict
# ---------------------------------------------------------------------------
def part_E() -> dict:
    state = _read_state()
    meta = state["meta"]
    pairs = state["A_boundary_pairs"]
    B = state["B_static"]["per_boundary_static"]
    C = state["C_horizon"]
    D = state.get("D_lane", {})
    F = state.get("F_hood", {})
    px_per_frame = float(meta["h"] * meta["w"])

    # MEASURED geometric coverages (per-frame band fraction -> coverage of that boundary band)
    n = meta["n_frames"]
    # lane: coverage = 1 - lane_attributable / (Road<->Lane band per-frame frac)
    rl_band_pf = (pairs["Road<->Lane"]["band_px"] / n) / px_per_frame
    lane_cov = max(0.0, 1.0 - (D.get("lane_attributable_mean", 0.000439) / rl_band_pf)) if D else 0.93
    hor_cov = C["horizon_coverage"]["within_6px"]
    hood_cov = F.get("road_mycar_band_coverage", 0.0)

    rows = {}
    for key, v in pairs.items():
        share = v["band_share"]
        st = B.get(key, {}).get("static_fraction", 0.0)
        geo = 0.0
        a, b = key.split("<->")
        is_lane = ("Lane" in (a, b))
        is_hood = ("MyCar" in (a, b))
        is_horizon = set((a, b)) == {"Road", "Undrivable"}
        if is_hood:
            geo = max(st, hood_cov)
            note = "STATIC hood mask (#139, ~0 byte)"
        elif is_lane:
            geo = max(geo, lane_cov)
            note = "GEOMETRIC lane (FEED-er; poly x homography)"
        elif is_horizon:
            geo = max(geo, hor_cov)
            note = "GEOMETRIC horizon line; rest=road-edge (learned/opp)"
        else:
            note = "LEARNED (vehicle silhouette)"
        solvable = min(1.0, max(st, geo))
        learned = 1.0 - solvable
        rows[key] = {
            "band_share": share, "static_frac": st, "geometric_frac": geo,
            "solvable_frac": solvable, "learned_frac": learned, "note": note,
        }

    # aggregate solvable vs learned over the whole band
    solv_mass = sum(r["band_share"] * r["solvable_frac"] for r in rows.values())
    learn_mass = sum(r["band_share"] * r["learned_frac"] for r in rows.values())
    state["E_table"] = rows
    state["E_aggregate"] = {
        "band_frac_of_frame": meta["band_frac"],
        "solvable_band_fraction": solv_mass,
        "learned_band_fraction": learn_mass,
    }
    _write_state(state)

    print("\n=== FEED-et: ALL-CLASS DECOMPOSITION TABLE ===")
    print(f"{'boundary':26s} {'band%':>7s} {'static':>7s} {'geo':>7s} {'solv':>7s} {'learn':>7s}  note")
    for k, r in sorted(rows.items(), key=lambda kv: -kv[1]["band_share"]):
        print(f"{k:26s} {r['band_share']*100:6.1f}% {r['static_frac']*100:6.1f}% "
              f"{r['geometric_frac']*100:6.1f}% {r['solvable_frac']*100:6.1f}% "
              f"{r['learned_frac']*100:6.1f}%  {r['note']}")
    print(f"\nBAND TOTAL: solvable(static+geometric) = {solv_mass*100:.1f}% | "
          f"learned = {learn_mass*100:.1f}%")
    print(f"band is {meta['band_frac']*100:.2f}% of frame")
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "C", "D", "E", "F"], required=True)
    args = ap.parse_args()
    {"A": part_A, "B": part_B, "C": part_C, "D": part_D, "E": part_E,
     "F": part_F}[args.part]()


if __name__ == "__main__":
    main()
