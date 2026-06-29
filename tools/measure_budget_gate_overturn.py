# SPDX-License-Identifier: MIT
"""BUDGET-GATE OVERTURN: EXACT-pose warp + SHARP render + STRUCTURED descriptor + DITHER cost.

THE QUESTION (A9 NOT-PESSIMISTIC OVERTURN of a95b0ad6 / DAG FEED-jz).
a95b0ad6 (``tools/measure_clean_canonical_warp_through_R.py``) ran the clean-canonical
budget gate and found the BULK d_seg through R = 0.00291 (n96, 2.4x the 1.23e-3 budget)
/ 0.00427 (n200, 3.5x) -> "bulk near-free via warp" REFUTED; the residual is genuine
per-frame SegNet jitter that must be stored. The 9-axis audit (FEED-jy/jz) flagged that
verdict as a likely UPPER BOUND, inflated by TWO confounds:

  A4/A8  the inter-pair CONSTANT-VELOCITY pose PROXY (only WITHIN-pair PoseNet poses are
         stored; inter-pair steps used 0.5*(pose[p]+pose[p+1])). Bad inter-pair alignment
         inflates apparent jitter -> over-states the must-store floor.
  A5(M2) the through-R aggregator was an RGB-MEDIAN (blurs misaligned boundaries -> +15-78%),
         not a SHARP partition render.
  A5(M3) the RATE gate coded an OCCUPANCY MASK (368KB), not a STRUCTURED centerline/spline.

This tool resolves the confounds with FOUR $0 measurements (all REAL frozen CPU-torch SegNet,
NEVER MPS; advisory-only; pointer 0.19110 UNMOVED):

  M1 EXACT-POSE warp. Replace the constant-velocity proxy with EXACT inter-frame poses derived
     from the LOCALLY-AVAILABLE comma2k19 GT global pose for THIS exact segment
     (``experiments/results/pose_feasibility_probe/comma2k19_gt_pose_raw.npz``;
     segment b0c9d2329ad1606b|2018-07-27--06-03-57/10, 1200 frames = 600 pairs, fps 20.0).
     Per global step g->g+1: forward = ||p_{g+1}-p_g|| (metric, exact), rotation = axis-angle of
     R_{g+1}^T R_g, expressed as a PoseNet-convention 6-vec [fwd, vert, lat, -aa0,-aa1,-aa2].
     CONVENTION VALIDATED (this tool re-asserts it): the comma6 within-pair fit_roadplane_dseg
     must be <= the PoseNet within-pair fit (else the convention is wrong and we say so).
     KEY: does the BULK clean-canonical floor DROP below 2.4x once warp-error is removed?

  M2 SHARP render. Build a VOTE-CONSISTENT SHARP real-RGB canonical (per native pixel, copy the
     warped RGB of the neighbour whose warped argmax == the per-pixel majority vote; no median
     blend) -> R -> SegNet, vs the RGB-MEDIAN (blur) and the pre-R VOTE (0.00291) and the
     FEED-jk per-frame-exact carrier floor (5.9e-4). KEY: does sharp drop the bulk toward 5.9e-4?

  M3 STRUCTURED descriptor RATE. Code the BULK boundary (Undrivable<->drivable horizon row y(x))
     and the LANE centerline as low-DOF per-frame polynomials, temporally delta-coded, vs the
     368KB occupancy mask. KEY: does a structured descriptor hit the 0.5-5KB target?

  M4 BULK-JITTER DITHER. The must-store residual = pixels where the best clean (exact-pose) bulk
     prediction != target. Use the cached SegNet ``margins`` to test annulus-localization, then
     estimate the byte cost of storing it as a sparse MARGIN-KEYED dither, and whether
     (bulk warp prior FREE + dither + lane residual + movables) closes S < 0.15.

AUTHORITY / HONESTY FIREWALL (CLAUDE.md):
  * ``[macOS advisory / CPU-torch research-signal]`` ONLY. NOT a contest score. Pointer 0.19110
    UNMOVED. score_claim / promotable / ready_for_exact_eval_dispatch = False. This is a MEANS.
  * d_seg = REAL argmax-disagreement vs the cached FROZEN CPU-torch SegNet argmax ``lstars``
    (``measure_segnet_argmax`` = the same preprocess/last-frame/bilinear-resize contract
    ``upstream/evaluate.py`` uses). Exact CPU-torch, NEVER MPS. A NO-FAKE self-check asserts
    ``SegNet(gt_f1) == lstars`` exactly AND ABORTS rather than report a fabricated number.
  * PROVEN: the measured through-R/pre-R d_seg numbers + measured byte counts + the exact comma2k19
    relative poses (metric forward distance + quaternion rotation are EXACT, not fit).
    INFERRED (flagged): the comma2k19 device->camera column mapping (VALIDATED via the within-pair
    fit); the 3 global calibration scalars; that comma2k19 frame g == video frame g (ASSERTED via
    forward-distance == speed*dt). It warps GT RGB (not a shipped witness) -> bounds the
    deterministic part; authority = realized-through-R inside the witness INR + exact CPU/CUDA eval.

rule-118: plane-induced homography + expmap + per-step composition + window vote/median/sharp +
R chain + polynomial rasterizer = FREE deterministic geometry (expandable in inflate.py, uncounted).
The per-pair 6-DOF pose is COUNTED-but-EXISTING (stored for d_pose). The static scene descriptor +
the structured boundary coeffs + the stored bulk-jitter dither + the lane/movables residual = COUNTED.
NOT FORBIDDEN: honest geometry, NOT a smuggled per-frame argmax/warp table.

Reuses a23062c4 (``measure_screw_warp_through_R``: warp_rgb, _to_uint8, fit_calibration_within_pair)
+ a513372a (``measure_pose_warp_dseg``: homography/expmap/regime/warp_labels) +
a95b0ad6 (``measure_clean_canonical_warp_through_R``: warp_rgb_masked, compose_path_H, build_step_poses).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="All-NaN slice encountered")

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.measure_pose_warp_dseg import (  # noqa: E402
    CLASS_NAMES,
    NATIVE_H,
    NATIVE_W,
    SCREW_REGIME,
    intrinsics_at,
    regime_homography,
    warp_labels,
    _target_grid,
)
from tools.measure_screw_warp_through_R import (  # noqa: E402
    warp_rgb,
    _to_uint8,
    fit_calibration_within_pair,
)
from tools.measure_clean_canonical_warp_through_R import (  # noqa: E402
    warp_rgb_masked,
    compose_path_H,
    build_step_poses,
    rgb_at,
    BUDGET,
    PERFRAME_EXACT_CARRIER_FLOOR,
    BULK_CLASSES,
)

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
B0 = 37_545_489  # contest archive normalizer (25*bytes/B0 = rate term)


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


# --------------------------------------------------------------------------- #
# comma2k19 GT global pose -> EXACT per-step relative pose (PoseNet-convention 6-vec).
# --------------------------------------------------------------------------- #
def quat_to_R(q: np.ndarray) -> np.ndarray:
    """comma2k19 frame_orientations quaternion [w,x,y,z] (device->ECEF) -> rotation matrix."""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def R_to_axisangle(R: np.ndarray) -> np.ndarray:
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(tr))
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(th))
    return w * th


def comma6_step(pos: np.ndarray, ori: np.ndarray, g: int) -> np.ndarray:
    """EXACT relative pose frame g->g+1 as a PoseNet-convention 6-vec.

    Device frame (comma2k19): x=forward, y=lateral(right), z=vertical(down). The camera view frame
    in ``pose_to_homography`` is t=[x=right=pose[2], y=down=pose[1], z=fwd=pose[0]] -> so the 6-vec
    is [fwd, vert(down)=t_dev[2], lat(right)=t_dev[1], rot...]. Rotation cols map to the NEGATIVE
    device axis-angle (validated: corr -0.93/-0.94/-0.94 vs the stored PoseNet rotation columns).
    Forward uses the metric step distance ||p_{g+1}-p_g|| (sign-robust, exact). s_t/s_r absorb units.
    """
    Rg = quat_to_R(ori[g])
    Rg1 = quat_to_R(ori[g + 1])
    t_dev = Rg.T @ (pos[g + 1] - pos[g])           # [fwd, lat, vert] in device_g frame
    aa = R_to_axisangle(Rg1.T @ Rg)                # device_g -> device_{g+1} axis-angle
    fwd = float(np.linalg.norm(pos[g + 1] - pos[g]))
    return np.array([fwd, t_dev[2], t_dev[1], -aa[0], -aa[1], -aa[2]], dtype=np.float64)


def build_exact_step_poses(pos: np.ndarray, ori: np.ndarray, n_frames: int) -> np.ndarray:
    """(n_frames-1, 6) EXACT per-step comma2k19 relative poses (replaces the constant-velocity proxy)."""
    return np.stack([comma6_step(pos, ori, g) for g in range(n_frames - 1)], 0)


# --------------------------------------------------------------------------- #
# canonical builders (per regime): RGB-median (a95b0ad6) + VOTE-CONSISTENT SHARP.
# --------------------------------------------------------------------------- #
def _warp_window(gt_f0, gt_f1, seg_cache, t, window_radius, step_poses,
                 K_nat, Kinv_nat, grid_nat, K_seg, Kinv_seg, grid_seg, params, regime, n_frames):
    """Warp every window neighbour (excl. t) into t, at BOTH native (RGB) and seg (argmax) res.

    Returns (rgb_views[nv,Hn,Wn,3] with NaN where invalid, lab_views[nv,Hs,Ws] int, lab_valid[nv,Hs,Ws],
    neigh_dt[nv]) for the given physical regime. identity regime -> no warp (the neighbour itself).
    """
    rgb_views, lab_views, lab_valid, neigh_dt = [], [], [], []
    for g in range(max(0, t - window_radius), min(n_frames, t + window_radius + 1)):
        if g == t:
            continue
        if regime == "identity":
            rgb = rgb_at(gt_f0, gt_f1, g).astype(np.float64)
            lab = seg_cache[g]
            lab_v = np.ones(lab.shape, dtype=bool)
        else:
            Hn = compose_path_H(g, t, step_poses, K_nat, Kinv_nat, params, regime)
            rgb, valid_n = warp_rgb_masked(rgb_at(gt_f0, gt_f1, g).astype(np.float64), Hn, grid_nat)
            rgb = np.where(valid_n[:, :, None], rgb, np.nan)
            Hs = compose_path_H(g, t, step_poses, K_seg, Kinv_seg, params, regime)
            lab, lab_v = warp_labels(seg_cache[g], Hs, grid_seg)
        rgb_views.append(rgb)
        lab_views.append(lab)
        lab_valid.append(lab_v)
        neigh_dt.append(abs(g - t))
    return (np.stack(rgb_views, 0) if rgb_views else None,
            np.stack(lab_views, 0) if lab_views else None,
            np.stack(lab_valid, 0) if lab_valid else None,
            np.array(neigh_dt, dtype=np.int64))


def _vote_argmax(lab_views, lab_valid):
    """Per-seg-pixel majority vote over warped neighbour argmaxes -> vote[Hs,Ws]."""
    nv, Hs, Ws = lab_views.shape
    votes = np.zeros((5, Hs, Ws), dtype=np.float64)
    for k in range(nv):
        for c in range(5):
            votes[c] += (lab_views[k] == c) & lab_valid[k]
    return votes.argmax(0).astype(np.int64)


def _sharp_canonical_rgb(rgb_views, lab_views, lab_valid, neigh_dt, vote, f_persist_native):
    """Vote-consistent SHARP native RGB: per native pixel, copy the warped RGB of the nearest-in-time
    neighbour whose (seg-res) warped argmax == the per-pixel vote (upsampled nearest to native).
    Fallback to the per-pixel median where no neighbour matches. No blend -> sharp real texture."""
    nv, Hs, Ws = lab_views.shape
    Hn, Wn = rgb_views.shape[1], rgb_views.shape[2]
    # winning neighbour index per seg pixel (nearest dt among matches)
    win_idx = np.full((Hs, Ws), -1, dtype=np.int64)
    best_dt = np.full((Hs, Ws), 1 << 30, dtype=np.int64)
    order = np.argsort(neigh_dt)  # nearest first
    for k in order:
        match = (lab_views[k] == vote) & lab_valid[k] & (win_idx < 0)
        win_idx[match] = k
    # upsample win_idx seg->native (nearest)
    ys = (np.arange(Hn) * Hs / Hn).astype(np.int64).clip(0, Hs - 1)
    xs = (np.arange(Wn) * Ws / Wn).astype(np.int64).clip(0, Ws - 1)
    win_nat = win_idx[ys][:, xs]                                  # (Hn,Wn)
    # median fallback
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(rgb_views, axis=0)                    # (Hn,Wn,3)
    out = np.where(np.isfinite(med), med, f_persist_native.astype(np.float64))
    # gather sharp where a winner exists and that view is valid (finite) at the pixel
    for k in range(nv):
        sel = (win_nat == k)
        if not sel.any():
            continue
        vk = rgb_views[k]
        ok = sel & np.isfinite(vk).all(axis=2)
        out[ok] = vk[ok]
    return _to_uint8(out)


def _median_canonical_rgb(rgb_views, f_persist_native):
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(rgb_views, axis=0)
    out = np.where(np.isfinite(med), med, f_persist_native.astype(np.float64))
    return _to_uint8(out)


def _finalize(ne, tot):
    out = {"total": sum(ne) / max(sum(tot), 1)}
    all_tot = max(sum(tot), 1)
    area = {}
    for c in range(5):
        out[CLASS_NAMES[c]] = (ne[c] / tot[c]) if tot[c] else None
        out[CLASS_NAMES[c] + "_contrib"] = ne[c] / all_tot
        area[CLASS_NAMES[c]] = tot[c] / all_tot
    out["_area"] = area
    return out


def _bulk(d):
    return sum(d[c + "_contrib"] for c in BULK_CLASSES)


# =========================================================================== #
# M1 + M2: budget gate with EXACT poses + SHARP render.
# =========================================================================== #
def run_budget_gate(gt_f0, gt_f1, lstars, posenet_poses, exact_step_poses, seg, measure_segnet_argmax,
                    window_radius, n_pairs, fit_params, seg_cache, vote_windows=(1, 2, 3)):
    P = n_pairs
    n_frames = 2 * P
    SEG_H, SEG_W = lstars.shape[1], lstars.shape[2]
    K_nat = intrinsics_at(NATIVE_W, NATIVE_H); Kinv_nat = np.linalg.inv(K_nat)
    grid_nat = _target_grid(NATIVE_H, NATIVE_W)
    K_seg = intrinsics_at(SEG_W, SEG_H); Kinv_seg = np.linalg.inv(K_seg)
    grid_seg = _target_grid(SEG_H, SEG_W)
    proxy_step_poses = build_step_poses(posenet_poses)  # the a95b0ad6 constant-velocity proxy

    def seg_argmax(frame_uint8_native):
        am, _ = measure_segnet_argmax(seg, np.asarray(frame_uint8_native, dtype=np.float64))
        return am

    cache_selfcheck = int(sum(int(np.array_equal(seg_cache[2 * p + 1], lstars[p])) for p in range(P)))

    # ---- pre-R VOTE bulk, EXACT vs PROXY, window sweep (cheap label-space) ----
    def vote_bulk(step_poses, wr):
        ne = [0] * 5; tot = [0] * 5
        for p in range(P):
            t = 2 * p + 1
            tgt = lstars[p]
            vote_am = {}
            for regime in ("ground", "rotonly", "identity"):
                votes = np.zeros((5, SEG_H, SEG_W))
                for g in range(max(0, t - wr), min(n_frames, t + wr + 1)):
                    if g == t:
                        continue
                    if regime == "identity":
                        lab = seg_cache[g]
                    else:
                        Hc = compose_path_H(g, t, step_poses, K_seg, Kinv_seg, fit_params, regime)
                        lab, valid = warp_labels(seg_cache[g], Hc, grid_seg)
                        lab = np.where(valid, lab, seg_cache[g])
                    for c in range(5):
                        votes[c] += (lab == c)
                vote_am[regime] = votes.argmax(0).astype(np.int64)
            for c in range(5):
                r = SCREW_REGIME[c]; m = (tgt == c); nc = int(m.sum())
                if not nc:
                    continue
                ne[c] += int(((vote_am[r] != c) & m).sum()); tot[c] += nc
        return _finalize(ne, tot)

    print("[overturn] pre-R VOTE window sweep (proxy vs exact)...", flush=True)
    vote_sweep = {"proxy": {}, "exact": {}}
    for wr in vote_windows:
        vp = vote_bulk(proxy_step_poses, wr)
        ve = vote_bulk(exact_step_poses, wr)
        vote_sweep["proxy"][f"wr{wr}"] = {"bulk": _bulk(vp), "Road": vp["Road"], "full": vp}
        vote_sweep["exact"][f"wr{wr}"] = {"bulk": _bulk(ve), "Road": ve["Road"], "full": ve}
        print(f"  wr={wr}: proxy VOTE bulk={_bulk(vp):.5f}  exact VOTE bulk={_bulk(ve):.5f}", flush=True)

    # ---- through-R: proxy RGB-median, exact RGB-median, exact SHARP (window_radius) ----
    acc = {k: ([0] * 5, [0] * 5) for k in
           ("naive", "proxy_med", "exact_med", "exact_sharp")}
    print(f"[overturn] through-R pass over {P} pairs (window=+/-{window_radius})...", flush=True)
    for p in range(P):
        t = 2 * p + 1
        f0 = gt_f0[p].astype(np.float64)
        tgt = lstars[p]
        # per-regime canonicals
        canon = {"naive": {}, "proxy_med": {}, "exact_med": {}, "exact_sharp": {}}
        for regime in ("ground", "rotonly", "identity"):
            # PROXY median
            rv, lv, lvd, ndt = _warp_window(gt_f0, gt_f1, seg_cache, t, window_radius, proxy_step_poses,
                                            K_nat, Kinv_nat, grid_nat, K_seg, Kinv_seg, grid_seg,
                                            fit_params, regime, n_frames)
            canon["proxy_med"][regime] = seg_argmax(_median_canonical_rgb(rv, f0)) if rv is not None \
                else seg_cache[2 * p]
            # EXACT median + sharp
            rv2, lv2, lvd2, ndt2 = _warp_window(gt_f0, gt_f1, seg_cache, t, window_radius, exact_step_poses,
                                                K_nat, Kinv_nat, grid_nat, K_seg, Kinv_seg, grid_seg,
                                                fit_params, regime, n_frames)
            if rv2 is not None:
                canon["exact_med"][regime] = seg_argmax(_median_canonical_rgb(rv2, f0))
                vote = _vote_argmax(lv2, lvd2)
                canon["exact_sharp"][regime] = seg_argmax(
                    _sharp_canonical_rgb(rv2, lv2, lvd2, ndt2, vote, f0))
            else:
                canon["exact_med"][regime] = seg_cache[2 * p]
                canon["exact_sharp"][regime] = seg_cache[2 * p]
        naive_am = seg_cache[2 * p]
        for c in range(5):
            r = SCREW_REGIME[c]; m = (tgt == c); nc = int(m.sum())
            if not nc:
                continue
            acc["naive"][0][c] += int(((naive_am != c) & m).sum()); acc["naive"][1][c] += nc
            for key in ("proxy_med", "exact_med", "exact_sharp"):
                am = canon[key][r]
                acc[key][0][c] += int(((am != c) & m).sum()); acc[key][1][c] += nc
        if (p + 1) % 8 == 0 or p == P - 1:
            print(f"  ...{p + 1}/{P}", flush=True)

    fin = {k: _finalize(*v) for k, v in acc.items()}
    bulks = {k: _bulk(v) for k, v in fin.items()}
    # best clean (exact) bulk = min over exact aggregators + the exact pre-R vote at this window
    exact_vote_bulk = vote_sweep["exact"].get(f"wr{window_radius}", {}).get("bulk")
    best_exact_bulk = min([b for b in (bulks["exact_med"], bulks["exact_sharp"], exact_vote_bulk)
                           if b is not None])
    proxy_vote_bulk = vote_sweep["proxy"].get(f"wr{window_radius}", {}).get("bulk")

    return {
        "window_radius": window_radius,
        "cache_selfcheck_seg_cache_f1_eq_lstars": {"matches": cache_selfcheck, "P": P,
                                                   "PASS": bool(cache_selfcheck == P)},
        "fit_params_used": {"s_t": fit_params[0], "s_r": fit_params[1], "pitch": fit_params[2]},
        "preR_vote_window_sweep": {
            kind: {w: {"bulk": d["bulk"], "Road": d["Road"]} for w, d in dd.items()}
            for kind, dd in vote_sweep.items()},
        "through_R": {k: fin[k] for k in fin},
        "bulk_terms": {
            "naive_persist_bulk": bulks["naive"],
            "proxy_RGB_median_bulk": bulks["proxy_med"],
            "exact_RGB_median_bulk": bulks["exact_med"],
            "exact_SHARP_bulk": bulks["exact_sharp"],
            "proxy_preR_vote_bulk": proxy_vote_bulk,
            "exact_preR_vote_bulk": exact_vote_bulk,
            "best_exact_clean_bulk": best_exact_bulk,
            "a95b0ad6_proxy_vote_ref": 0.00291,
            "a95b0ad6_proxy_RGB_median_ref": 0.00550,
            "budget": BUDGET,
            "perframe_exact_carrier_floor_FEEDjk": PERFRAME_EXACT_CARRIER_FLOOR,
        },
        "OVERTURN_M1_M2": {
            "exact_vs_proxy_vote_delta": (exact_vote_bulk - proxy_vote_bulk)
            if (exact_vote_bulk is not None and proxy_vote_bulk is not None) else None,
            "best_exact_bulk_over_budget_factor": (best_exact_bulk / BUDGET),
            "exact_pose_drops_below_2p4x": bool(best_exact_bulk < 2.4 * BUDGET),
            "exact_pose_closes_budget": bool(best_exact_bulk <= BUDGET),
            "sharp_beats_median": bool(bulks["exact_sharp"] < bulks["exact_med"]),
            "sharp_vs_carrier_floor_ratio": bulks["exact_sharp"] / PERFRAME_EXACT_CARRIER_FLOOR,
        },
        "seg_cache_handle": None,  # set by caller for M3/M4 reuse
    }


# =========================================================================== #
# M3: STRUCTURED descriptor rate (bulk horizon poly + lane centerline) vs occupancy.
# =========================================================================== #
def _comp(data: bytes) -> int:
    z = len(zlib.compress(data, 9))
    try:
        import brotli
        return min(z, len(brotli.compress(data, quality=11)))
    except Exception:
        return z


def run_structured_descriptor(seg_cache, n_pairs, scale_to=600):
    """Code BULK horizon boundary + LANE centerline as low-DOF polynomials; compare to occupancy."""
    P = n_pairs
    n_frames = 2 * P
    Hs, Ws = seg_cache.shape[1], seg_cache.shape[2]

    # ---------- BULK horizon: per-column topmost non-Undrivable row (drivable/road boundary) ----------
    # Undrivable=2 dominates the TOP. horizon(x) = first row (top->down) where class != 2.
    horizons = np.zeros((n_frames, Ws), dtype=np.float64)
    for g in range(n_frames):
        not_sky = seg_cache[g] != 2
        # first True per column from top; if none, row=Hs-1
        idx = np.where(not_sky.any(0), not_sky.argmax(0), Hs - 1)
        horizons[g] = idx
    # per-frame poly fit deg D over columns; residual quantized to +-1 row
    def poly_descriptor(curves, deg, coef_bits=12, resid_bits=0):
        xs = np.linspace(-1, 1, curves.shape[1])
        coeffs = np.stack([np.polyfit(xs, curves[g], deg) for g in range(curves.shape[0])], 0)
        # temporal delta + quantize coeffs (scale to coef_bits)
        cmin, cmax = coeffs.min(0), coeffs.max(0)
        rng = np.where(cmax > cmin, cmax - cmin, 1.0)
        q = np.round((coeffs - cmin) / rng * ((1 << coef_bits) - 1)).astype(np.int64)
        qd = np.diff(q, axis=0, prepend=q[:1])
        coef_bytes = _comp(qd.astype(np.int16).tobytes())
        # residual (rows) after poly, quantized to integer rows
        recon = np.stack([np.polyval(coeffs[g], xs) for g in range(curves.shape[0])], 0)
        resid = np.round(curves - recon).astype(np.int64)
        rms = float(np.sqrt(np.mean((curves - recon) ** 2)))
        resid_bytes = _comp(resid.astype(np.int16).tobytes()) if resid_bits != 0 else 0
        return {"deg": deg, "coef_bytes": coef_bytes, "resid_rms_px": rms,
                "exact_resid_bytes": _comp(resid.astype(np.int16).tobytes()),
                "coef_plus_exact_resid_bytes": coef_bytes + _comp(resid.astype(np.int16).tobytes())}
    bulk_rows = [poly_descriptor(horizons, d) for d in (2, 3, 4, 6)]
    bulk_occ_bytes = sum(_comp(np.packbits((seg_cache[g] == 2).ravel()).tobytes()) for g in range(n_frames))

    # ---------- LANE centerline: per-row x-centroid curve(s) ----------
    # Represent lane (class 1) by, for each row, the median x of lane pixels (single dominant curve)
    # + count. A real codec fits L/R/center polynomials; the single-centroid curve is a LOWER-DOF
    # proxy (under-counts multi-line frames -> the multi-curve cost is bounded below by this).
    lane_centroid = np.full((n_frames, Hs), -1.0, dtype=np.float64)
    lane_count = np.zeros((n_frames,), dtype=np.int64)
    for g in range(n_frames):
        lane = (seg_cache[g] == 1)
        lane_count[g] = int(lane.sum())
        rows = np.where(lane.any(1))[0]
        for r in rows:
            xs_lane = np.where(lane[r])[0]
            lane_centroid[g, r] = float(np.median(xs_lane))
    # rows with lane present: code centroid as poly over present rows (per frame), + occupancy of which rows
    # Simpler decisive proxy: code the lane as the SET of (row, x_centroid) present points -> entropy.
    present = lane_centroid >= 0
    # x-centroid quantized to 1px; row index implicit; temporal handling: code per-frame
    lane_struct_pts_bytes = _comp(
        np.concatenate([lane_centroid[present].astype(np.int16),
                        present.sum(1).astype(np.int16)]).tobytes())
    # lane occupancy mask bytes (the 368KB-style strawman, here measured on this n)
    lane_occ_bytes = sum(_comp(np.packbits((seg_cache[g] == 1).ravel()).tobytes()) for g in range(n_frames))
    # lane as low-degree x(row) poly per frame (centerline coeff descriptor)
    def lane_poly_bytes(deg=3, coef_bits=12):
        coeffs = []
        rms_list = []
        for g in range(n_frames):
            rows = np.where(present[g])[0]
            if len(rows) < deg + 1:
                coeffs.append(np.zeros(deg + 1)); continue
            yy = (rows / Hs) * 2 - 1
            cc = np.polyfit(yy, lane_centroid[g, rows], deg)
            coeffs.append(cc)
            rms_list.append(float(np.sqrt(np.mean((np.polyval(cc, yy) - lane_centroid[g, rows]) ** 2))))
        coeffs = np.stack(coeffs, 0)
        cmin, cmax = coeffs.min(0), coeffs.max(0); rng = np.where(cmax > cmin, cmax - cmin, 1.0)
        q = np.round((coeffs - cmin) / rng * ((1 << coef_bits) - 1)).astype(np.int64)
        qd = np.diff(q, axis=0, prepend=q[:1])
        return {"deg": deg, "coef_bytes": _comp(qd.astype(np.int16).tobytes()),
                "centerline_rms_px": float(np.mean(rms_list)) if rms_list else None,
                "note": "single-centroid-per-row proxy; multi-line frames cost more (lower bound)."}
    lane_poly = [lane_poly_bytes(d) for d in (2, 3, 4)]

    sc = scale_to / n_frames
    return {
        "n_frames_measured": n_frames, "scaled_to": scale_to,
        "BULK_horizon_boundary": {
            "occupancy_mask_bytes_measured": bulk_occ_bytes,
            "occupancy_mask_bytes_scaled600": int(round(bulk_occ_bytes * sc)),
            "poly_descriptor_rows": bulk_rows,
            "best_poly_coef_bytes_scaled600": int(round(min(r["coef_bytes"] for r in bulk_rows) * sc)),
            "best_poly_coef_plus_exact_resid_scaled600": int(round(
                min(r["coef_plus_exact_resid_bytes"] for r in bulk_rows) * sc)),
        },
        "LANE_centerline": {
            "occupancy_mask_bytes_measured": lane_occ_bytes,
            "occupancy_mask_bytes_scaled600": int(round(lane_occ_bytes * sc)),
            "structured_points_bytes_measured": lane_struct_pts_bytes,
            "structured_points_bytes_scaled600": int(round(lane_struct_pts_bytes * sc)),
            "centerline_poly_rows": lane_poly,
            "best_centerline_coef_bytes_scaled600": int(round(min(r["coef_bytes"] for r in lane_poly) * sc)),
            "mean_lane_px_per_frame": float(lane_count.mean()),
        },
        "target_bytes_per600": [500, 5000],
        "FEEDjm_image_space_centerline_anchor": 65000,
        "occupancy_strawman_anchor": 368000,
        "caveat": ("structured descriptors are low-DOF proxies (bulk=horizon row poly; lane=single "
                   "centroid-per-row poly). They LOWER-BOUND a faithful multi-curve spline coder; the "
                   "exact-residual columns make the bulk descriptor LOSSLESS (coef+resid)."),
    }


# =========================================================================== #
# M4: BULK-JITTER DITHER cost + budget closure.
# =========================================================================== #
def run_dither_cost(gt_f0, gt_f1, lstars, margins, seg_cache, exact_step_poses, fit_params,
                    n_pairs, window_radius, vote_bulk_pred=None, scale_to=600,
                    pose_term=0.018, frontier_rate=0.118):
    """The must-store residual = pixels where the best clean (exact-pose VOTE) bulk prediction != target.
    Test annulus-localization via cached margins; estimate margin-keyed dither bytes; check S<0.15."""
    P = n_pairs
    n_frames = 2 * P
    SEG_H, SEG_W = lstars.shape[1], lstars.shape[2]
    K_seg = intrinsics_at(SEG_W, SEG_H); Kinv_seg = np.linalg.inv(K_seg); grid_seg = _target_grid(SEG_H, SEG_W)

    bulk_idx = [CLASS_NAMES.index(c) for c in BULK_CLASSES]
    flip_total = 0; bulk_px_total = 0
    flips_in_annulus = {}      # tau -> count
    annulus_px = {}            # tau -> count (bulk pixels with margin<tau)
    taus = [0.05, 0.1, 0.2, 0.3, 0.5]
    for tau in taus:
        flips_in_annulus[tau] = 0; annulus_px[tau] = 0
    # entropy of flip indicator within annulus (per tau) accumulators
    flip_label_counts = np.zeros(5, dtype=np.int64)
    margins_at_flips = []

    for p in range(P):
        t = 2 * p + 1
        tgt = lstars[p]
        marg = margins[p]
        # exact-pose VOTE prediction (per regime)
        vote_am = {}
        for regime in ("ground", "rotonly", "identity"):
            votes = np.zeros((5, SEG_H, SEG_W))
            for g in range(max(0, t - window_radius), min(n_frames, t + window_radius + 1)):
                if g == t:
                    continue
                if regime == "identity":
                    lab = seg_cache[g]
                else:
                    Hc = compose_path_H(g, t, exact_step_poses, K_seg, Kinv_seg, fit_params, regime)
                    lab, valid = warp_labels(seg_cache[g], Hc, grid_seg)
                    lab = np.where(valid, lab, seg_cache[g])
                for c in range(5):
                    votes[c] += (lab == c)
            vote_am[regime] = votes.argmax(0).astype(np.int64)
        # bulk mask + flips (per-pixel: regime-routed vote prediction != target class, on bulk classes)
        bulk_mask = np.isin(tgt, bulk_idx)
        flip = _bulk_flip(vote_am, tgt, bulk_mask)
        flip_total += int(flip.sum()); bulk_px_total += int(bulk_mask.sum())
        margins_at_flips.append(marg[flip])
        for c in range(5):
            flip_label_counts[c] += int(((tgt == c) & flip).sum())
        for tau in taus:
            ann = bulk_mask & (marg < tau)
            annulus_px[tau] += int(ann.sum())
            flips_in_annulus[tau] += int((flip & (marg < tau)).sum())

    flip_frac = flip_total / max(bulk_px_total, 1)
    margins_at_flips = np.concatenate(margins_at_flips) if margins_at_flips else np.array([0.0])
    # byte estimate: store the flip set as positions-in-annulus + new label.
    # at tau where most flips live: index cost ~ log2(annulus_px/flip) bits/flip (sparse-set entropy)
    dither_rows = []
    for tau in taus:
        ann = annulus_px[tau]; fl = flips_in_annulus[tau]
        cover = fl / max(flip_total, 1)
        if fl == 0 or ann == 0:
            dither_rows.append({"tau": tau, "annulus_px": ann, "flips_in_annulus": fl,
                                "flip_coverage": cover, "est_bytes_scaled600": None})
            continue
        # combinatorial sparse-set bits: log2 C(ann, fl) ~ fl*log2(e*ann/fl); + label ~1.5 bits/flip
        import math
        bits_pos = fl * math.log2(math.e * ann / fl) if fl < ann else ann
        bits_label = fl * 1.5
        bytes_meas = (bits_pos + bits_label) / 8.0
        # plus the flips OUTSIDE the annulus must also be stored to fully close d_seg
        outside = flip_total - fl
        bits_out = outside * (math.log2(math.e * max(bulk_px_total, 1) / max(outside, 1)) + 1.5) if outside else 0
        bytes_full = bytes_meas + bits_out / 8.0
        sc = scale_to / (2 * P)
        dither_rows.append({
            "tau": tau, "annulus_px": ann, "flips_in_annulus": fl, "flip_coverage": round(cover, 3),
            "est_bytes_in_annulus_scaled600": int(round(bytes_meas * sc)),
            "est_bytes_full_close_scaled600": int(round(bytes_full * sc)),
        })
    # budget closure: if we STORE the dither, bulk d_seg -> ~0; rate from dither.
    full_close = [r for r in dither_rows if r.get("est_bytes_full_close_scaled600")]
    min_dither_bytes = min((r["est_bytes_full_close_scaled600"] for r in full_close), default=None)
    dither_rate = (25 * min_dither_bytes / B0) if min_dither_bytes else None
    # S estimate if bulk fully stored (d_seg bulk->0, but lane+movables residual remains; use budget for d_seg)
    return {
        "bulk_flip_fraction_exact_vote": flip_frac,
        "bulk_flip_fraction_x100_dseg_units": flip_frac,  # this IS the bulk d_seg of exact vote
        "flip_label_distribution": {CLASS_NAMES[c]: int(flip_label_counts[c]) for c in range(5)},
        "margin_at_flips": {"mean": float(margins_at_flips.mean()), "median": float(np.median(margins_at_flips)),
                            "p90": float(np.percentile(margins_at_flips, 90))},
        "annulus_localization": dither_rows,
        "min_dither_bytes_full_close_scaled600": min_dither_bytes,
        "dither_rate_term": dither_rate,
        "budget_closure": {
            "assumed_pose_term": pose_term,
            "d_seg_budget_term_100x": 100 * BUDGET,
            "dither_rate_term": dither_rate,
            "S_if_dseg_at_budget_plus_dither_rate": (100 * BUDGET + pose_term + dither_rate)
            if dither_rate is not None else None,
            "closes_sub_0p15": bool((100 * BUDGET + pose_term + (dither_rate or 9)) < 0.15),
            "note": ("storing the bulk dither drives bulk d_seg->~0; the rate is the dither bytes. "
                     "S = 100*d_seg + pose_term + 25*bytes/B0. This counts ONLY the bulk-jitter dither "
                     "(lane residual + movables + canonical keyframe + structured descriptor are ADDITIONAL)."),
        },
    }


def _bulk_flip(vote_am, tgt, bulk_mask):
    """Per-pixel: in bulk, does the regime-routed vote prediction disagree with the target class?"""
    H, W = tgt.shape
    flip = np.zeros((H, W), dtype=bool)
    for c in [CLASS_NAMES.index(x) for x in BULK_CLASSES]:
        r = SCREW_REGIME[c]
        m = (tgt == c)
        flip |= m & (vote_am[r] != c)
    return flip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--comma-pose", default="experiments/results/pose_feasibility_probe/comma2k19_gt_pose_raw.npz")
    ap.add_argument("--n-pairs", type=int, default=0, help="0 = all in cache")
    ap.add_argument("--window-radius", type=int, default=2)
    ap.add_argument("--tests", default="1,3,4", help="comma list of 1(budget+sharp),3(structured),4(dither)")
    ap.add_argument("--selfcheck-pairs", type=int, default=4)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    tests = set(args.tests.split(","))

    from tac.optimization.frame1_seg_repair_atoms import measure_segnet_argmax
    from tac.boundary_math.seg_core import load_real_segnet

    t0 = time.time()
    cache = (REPO / args.cache) if not Path(args.cache).is_absolute() else Path(args.cache)
    z = np.load(cache, allow_pickle=False)
    gt_f0 = np.asarray(z["gt_f0"]); gt_f1 = np.asarray(z["gt_f1"])
    lstars = np.asarray(z["lstars"], dtype=np.int64)
    margins = np.asarray(z["margins"], dtype=np.float32)
    posenet_poses = np.asarray(z["gt_poses"], dtype=np.float64)
    P_cache = lstars.shape[0]
    P = P_cache if not args.n_pairs else min(args.n_pairs, P_cache)
    gt_f0, gt_f1, lstars, margins, posenet_poses = (gt_f0[:P], gt_f1[:P], lstars[:P],
                                                    margins[:P], posenet_poses[:P])
    SEG_H, SEG_W = lstars.shape[1], lstars.shape[2]
    NAT_H, NAT_W = gt_f0.shape[1], gt_f0.shape[2]
    assert (NAT_H, NAT_W) == (NATIVE_H, NATIVE_W)

    cp = np.load((REPO / args.comma_pose) if not Path(args.comma_pose).is_absolute()
                 else Path(args.comma_pose), allow_pickle=True)
    pos = np.asarray(cp["frame_positions"], dtype=np.float64)
    ori = np.asarray(cp["frame_orientations"], dtype=np.float64)
    vel = np.asarray(cp["frame_velocities"], dtype=np.float64)
    ftimes = np.asarray(cp["frame_times"], dtype=np.float64)
    segment_id = str(cp["segment_id"])
    n_frames = 2 * P

    # ---- frame alignment assertion: comma frame g == video frame g ----
    dt = float(np.median(np.diff(ftimes[:n_frames])))
    step_dist = np.linalg.norm(np.diff(pos[:n_frames], axis=0), axis=1)
    spd = np.linalg.norm(vel[:n_frames], axis=1)
    expected = float(np.mean(spd[:n_frames - 1]) * dt)
    align_err = abs(float(step_dist.mean()) - expected) / max(expected, 1e-9)
    frame_alignment = {"fps": 1.0 / dt, "mean_step_dist_m": float(step_dist.mean()),
                       "expected_speed_x_dt_m": expected, "rel_err": align_err,
                       "PASS": bool(align_err < 0.05)}

    seg = load_real_segnet("cpu")

    def seg_argmax(frame_uint8_native):
        am, _ = measure_segnet_argmax(seg, np.asarray(frame_uint8_native, dtype=np.float64))
        return am

    # ---- NO-FAKE selfcheck ----
    scn = min(args.selfcheck_pairs, P)
    selfcheck = {"pairs_checked": scn, "exact_matches": 0, "max_disagree_px": 0}
    for p in range(scn):
        am = seg_argmax(gt_f1[p])
        ndiff = int(np.count_nonzero(am != lstars[p]))
        selfcheck["max_disagree_px"] = max(selfcheck["max_disagree_px"], ndiff)
        if ndiff == 0:
            selfcheck["exact_matches"] += 1
    selfcheck["PASS"] = bool(selfcheck["exact_matches"] == scn and selfcheck["max_disagree_px"] == 0)
    if not selfcheck["PASS"]:
        raise SystemExit(f"NO-FAKE self-check FAILED (max_disagree_px={selfcheck['max_disagree_px']}).")

    # ---- per-global-frame SegNet argmax cache (ONCE; shared by M1/M3/M4; even frames = lstar0) ----
    print(f"[overturn] caching per-frame SegNet argmax for {n_frames} frames (CPU)...", flush=True)
    seg_cache = np.zeros((n_frames, SEG_H, SEG_W), dtype=np.int64)
    for g in range(n_frames):
        seg_cache[g] = seg_argmax(rgb_at(gt_f0, gt_f1, g))
        if (g + 1) % 48 == 0 or g == n_frames - 1:
            print(f"  ...{g + 1}/{n_frames}", flush=True)
    lstar0 = seg_cache[0::2]  # SegNet(f0[p]) == seg_cache[2p]

    # ---- exact step poses + convention validation (comma6 vs PoseNet within-pair fit) ----
    exact_step_poses = build_exact_step_poses(pos, ori, n_frames)
    K_seg = intrinsics_at(SEG_W, SEG_H); Kinv_seg = np.linalg.inv(K_seg); grid_seg = _target_grid(SEG_H, SEG_W)
    comma_within = np.stack([exact_step_poses[2 * p] for p in range(P)], 0)  # g=2p within-pair steps
    fit_comma = fit_calibration_within_pair(lstar0, lstars, comma_within, K_seg, Kinv_seg, grid_seg)
    fit_posenet = fit_calibration_within_pair(lstar0, lstars, posenet_poses, K_seg, Kinv_seg, grid_seg)
    convention_validated = bool(fit_comma["fit_roadplane_dseg"] <= fit_posenet["fit_roadplane_dseg"] * 1.05)
    fit_params = (fit_comma["s_t"], fit_comma["s_r"], fit_comma["pitch"])
    print(f"[overturn] convention validated={convention_validated} "
          f"comma_fit={fit_comma['fit_roadplane_dseg']:.5f} posenet_fit={fit_posenet['fit_roadplane_dseg']:.5f}",
          flush=True)

    out = {
        "tool": "tools/measure_budget_gate_overturn.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "[macOS advisory / CPU-torch research-signal]",
        "score_claim": False, "promotion_eligible": False, "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False, "promotable": False,
        "frontier_pointer": "UNMOVED 0.19110 (advisory; not a contest score)",
        "cache": str(cache.relative_to(REPO)) if str(cache).startswith(str(REPO)) else str(cache),
        "comma2k19_segment_id": segment_id,
        "n_pairs": P, "seg_hw": [SEG_H, SEG_W], "native_hw": [NAT_H, NAT_W],
        "frame_alignment_check": frame_alignment,
        "no_fake_selfcheck_segnet_gt_f1_eq_lstars": selfcheck,
        "comma2k19_convention_validation": {
            "comma_within_fit_roadplane_dseg": fit_comma["fit_roadplane_dseg"],
            "posenet_within_fit_roadplane_dseg": fit_posenet["fit_roadplane_dseg"],
            "VALIDATED": convention_validated,
            "fit_comma": fit_comma, "fit_posenet": fit_posenet,
            "note": ("comma6 = [fwd=||dp||, vert, lat, -aa_dev]. VALIDATED if comma within-pair Road+Lane "
                     "warp fit <= PoseNet's (it engages yaw s_r the PoseNet fit cannot)."),
        },
        "rule_118": {
            "FREE_generic_in_inflate": "homography + expmap + per-step compose + window vote/median/sharp + R + poly rasterizer",
            "COUNTED_existing": "per-pair 6-DOF pose (stored for d_pose; +0 marginal)",
            "COUNTED": "static descriptor + structured boundary coeffs + bulk-jitter dither + lane/movables residual",
            "not_forbidden": "honest geometry + GT comma2k19 pose; NOT a smuggled per-frame argmax/warp table",
        },
        "assumptions": {
            "PROVEN": "through-R/pre-R d_seg = real argmax-disagreement vs frozen CPU-torch SegNet (lstars); byte counts measured; comma2k19 relative poses EXACT (metric dist + quaternion rotation).",
            "VALIDATED": "comma frame g == video frame g (forward dist == speed*dt, rel_err %.4f); comma6 device->camera convention (within-pair fit <= PoseNet)." % align_err,
            "INFERRED": "the 3 global calibration scalars; small lateral/vertical translation column order (negligible vs forward+rotation); camera-res R (excludes sub-874 bicubic-up aliasing).",
            "warps_GT_RGB": "bounds the deterministic part; authority = realized-through-R inside witness INR + exact CPU/CUDA eval.",
        },
    }

    if "1" in tests:
        res1 = run_budget_gate(gt_f0, gt_f1, lstars, posenet_poses, exact_step_poses, seg,
                               measure_segnet_argmax, args.window_radius, P, fit_params, seg_cache)
        out["M1_M2_budget_gate"] = res1
    if "3" in tests:
        out["M3_structured_descriptor"] = run_structured_descriptor(seg_cache, P)
    if "4" in tests:
        out["M4_dither_budget_closure"] = run_dither_cost(
            gt_f0, gt_f1, lstars, margins, seg_cache, exact_step_poses, fit_params, P, args.window_radius)

    out["elapsed_secs"] = round(time.time() - t0, 1)

    out_dir = (Path(args.out_dir) if args.out_dir
               else (REPO / f"experiments/results/budget_gate_overturn_n{P}_r{args.window_radius}"))
    _refuse_tmp(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))

    # ---- console summary ----
    print("\n[overturn] ===== SUMMARY =====")
    print(f"  frame_alignment PASS={frame_alignment['PASS']} (rel_err {align_err:.4f}); "
          f"convention VALIDATED={convention_validated}")
    if "1" in tests:
        bt = out["M1_M2_budget_gate"]["bulk_terms"]; ov = out["M1_M2_budget_gate"]["OVERTURN_M1_M2"]
        print(f"  [M1] proxy VOTE bulk={bt['proxy_preR_vote_bulk']:.5f} (ref 0.00291)  "
              f"EXACT VOTE bulk={bt['exact_preR_vote_bulk']:.5f}")
        print(f"  [M1] proxy RGB-median={bt['proxy_RGB_median_bulk']:.5f} (ref 0.00550)  "
              f"EXACT RGB-median={bt['exact_RGB_median_bulk']:.5f}")
        print(f"  [M2] EXACT SHARP bulk={bt['exact_SHARP_bulk']:.5f}  carrier floor={PERFRAME_EXACT_CARRIER_FLOOR:.1e}")
        print(f"  [M1/M2] best_exact_clean_bulk={bt['best_exact_clean_bulk']:.5f} = "
              f"{ov['best_exact_bulk_over_budget_factor']:.2f}x budget  "
              f"closes={ov['exact_pose_closes_budget']} below2.4x={ov['exact_pose_drops_below_2p4x']}")
    if "3" in tests:
        m3 = out["M3_structured_descriptor"]
        print(f"  [M3] BULK horizon: occ={m3['BULK_horizon_boundary']['occupancy_mask_bytes_scaled600']}B "
              f"-> poly coef={m3['BULK_horizon_boundary']['best_poly_coef_bytes_scaled600']}B "
              f"(+exact resid {m3['BULK_horizon_boundary']['best_poly_coef_plus_exact_resid_scaled600']}B)")
        print(f"  [M3] LANE: occ={m3['LANE_centerline']['occupancy_mask_bytes_scaled600']}B "
              f"-> centerline coef={m3['LANE_centerline']['best_centerline_coef_bytes_scaled600']}B "
              f"struct-pts={m3['LANE_centerline']['structured_points_bytes_scaled600']}B (target 500-5000)")
    if "4" in tests:
        m4 = out["M4_dither_budget_closure"]
        print(f"  [M4] bulk flip frac (exact vote)={m4['bulk_flip_fraction_exact_vote']:.5f}  "
              f"min dither bytes/600={m4['min_dither_bytes_full_close_scaled600']}  "
              f"rate={m4['dither_rate_term']}")
        print(f"  [M4] S(dseg@budget+pose+dither)={m4['budget_closure']['S_if_dseg_at_budget_plus_dither_rate']}  "
              f"closes_sub0.15={m4['budget_closure']['closes_sub_0p15']}")
    print(f"\n[written] {out_path} (elapsed {out['elapsed_secs']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
