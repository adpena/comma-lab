# SPDX-License-Identifier: MIT
"""Measure whether the stored ego-pose carries the d_seg TRAJECTORY for free.

GROK / DAG FEED-iv central claim: ``d_seg`` and ``d_pose`` are two readouts of
the SAME sufficient statistic (the ego-pose), because each frame's SegNet
argmax partition = ``homography(ego-pose) · canonical_scene``. If true, the
pose sidecar we already store for ``d_pose`` (6 numbers/frame) IS the d_seg
modulation for free, leaving only a survival + movables residual.

This tool runs the LOCAL consequence of that claim, which is robust to
cumulative-pose drift: if ``frame[p] = H_p · C`` for a shared canonical ``C``,
then ``frame[p+1] = (H_{p+1} H_p^{-1}) · frame[p] = H_rel · frame[p]`` where
``H_rel`` is the plane-induced homography of the RELATIVE ego-pose. So we test:

    predict lstars[p+1]  :=  warp( lstars[p] , H_rel(pose) )

and compare its d_seg (argmax disagreement vs the actual frozen-SegNet argmax)
against the no-motion null (persist: ``predict := lstars[p]``). We decompose the
residual PER CLASS: Road/Lane are road-plane (homography) classes that SHOULD be
pose-explained; MyCar (ego hood) is static-in-image (needs identity, not a
ground warp); Undrivable is sky-dominated (plane at infinity); Movables move
independently (irreducible).

AUTHORITY / HONESTY FIREWALL (CLAUDE.md):
  * ``[macOS advisory / research-signal]`` ONLY. NOT a contest score. The
    canonical frontier pointer is UNMOVED. This is a frozen-instance,
    direct-partition warp test = a PROXY for warp-inside-the-witness-INR,
    measured PRE-R (no bicubic/uint8 round-trip) -> necessary, not sufficient.
  * d_seg here is the REAL argmax-disagreement against ``lstars`` = the FROZEN
    CPU-torch SegNet argmax cached by ``tools/build_shared_gt_cache_for_mlx_fleet``
    (the same authority the witness trainer uses inline). No surrogate.
  * PROVEN: the measured warp/persist d_seg numbers. INFERRED (flagged in the
    JSON ``assumptions`` block): the physical interpretation of the raw learned
    PoseNet 6-vector columns, the adjacent-pose proxy for the relative motion,
    and the calibration units (we FIT 3 global scalars).

Reuses: openpilot/comma2k19 EON intrinsics (fx=fy=910, cx=582, cy=437 native
1164x874; ``tac.calibrated_geometry`` pins the same), camera height 1.22 m
(openpilot ``HEIGHT_INIT``), plane-induced homography ``H = K (R - t n^T / d) K^{-1}``
(Hartley & Zisserman). See ``.omx/research/grok_pose_warp_dseg_test_*.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
CLASS_NAMES = ["Road", "Lane", "Undriv", "Movable", "MyCar"]  # comma10k canonical order
ROAD_PLANE_CLASSES = (0, 1)  # Road + Lane: true ground-plane (homography) classes

# openpilot / comma2k19 EON road camera intrinsics (NATIVE 1164x874).
# Source (subagent-verified, 2 independent repos agree): comma2k19 utils/camera.py
# (eon_focal_length=910.0, FULL_FRAME_SIZE=(1164,874), pp=(W/2,H/2)) and openpilot
# common/transformations/camera.py (_neo_config). Camera height = 1.22 m
# (openpilot selfdrive/locationd/calibrationd.py HEIGHT_INIT). Pitch calibrated
# online (init 0, bounds [-0.091, 0.17] rad) -> we FIT it.
NATIVE_W, NATIVE_H = 1164, 874
NATIVE_FX = NATIVE_FY = 910.0
NATIVE_CX, NATIVE_CY = 582.0, 437.0
CAMERA_HEIGHT_M = 1.22


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def intrinsics_at(seg_w: int, seg_h: int) -> np.ndarray:
    """EON K scaled from native (1164x874) to the SegNet working resolution."""
    sx, sy = seg_w / NATIVE_W, seg_h / NATIVE_H
    return np.array(
        [[NATIVE_FX * sx, 0.0, NATIVE_CX * sx],
         [0.0, NATIVE_FY * sy, NATIVE_CY * sy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _expmap_so3(omega: np.ndarray) -> np.ndarray:
    """Rodrigues axis-angle -> rotation matrix (numpy, deterministic)."""
    theta = float(np.linalg.norm(omega))
    K = np.array([[0.0, -omega[2], omega[1]],
                  [omega[2], 0.0, -omega[0]],
                  [-omega[1], omega[0], 0.0]], dtype=np.float64)
    if theta < 1e-12:
        return np.eye(3) + K  # 1st-order (theta~0)
    return (np.eye(3)
            + (np.sin(theta) / theta) * K
            + ((1.0 - np.cos(theta)) / (theta * theta)) * (K @ K))


def pose_to_homography(pose6: np.ndarray, K: np.ndarray, Kinv: np.ndarray,
                       s_t: float, s_r: float, pitch: float) -> np.ndarray:
    """Plane-induced homography H = K (R - t n^T / d) K^{-1} from a learned pose.

    ASSUMPTION (flagged): raw PoseNet 6-vector = [fwd, c1, c2, r0, r1, r2]; col0
    (~33, dominant) is forward translation -> view-frame z. View frame is
    (x->right, y->down, z->forward); road normal points up (=-y) tilted by pitch.
    Only t/d matters and d, s_t are degenerate -> we fix d=1.22 and FIT s_t (which
    also absorbs the learned-units->metric scale and the adjacent-pose factor).
    """
    t = s_t * np.array([pose6[2], pose6[1], pose6[0]], dtype=np.float64)  # (x,y,z=fwd)
    R = _expmap_so3(s_r * np.array([pose6[3], pose6[4], pose6[5]], dtype=np.float64))
    n = np.array([0.0, -np.cos(pitch), -np.sin(pitch)], dtype=np.float64)  # road up, tilted
    M = R - np.outer(t, n) / CAMERA_HEIGHT_M
    return K @ M @ Kinv


def warp_labels(src: np.ndarray, H: np.ndarray, tgt_grid: np.ndarray):
    """Nearest-neighbour inverse-warp an int label map. Returns (pred, valid)."""
    Hh, Ww = src.shape
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Hinv = np.linalg.inv(H)
        src_h = Hinv @ tgt_grid  # (3, Hh*Ww)
        z = src_h[2]
        su = src_h[0] / z
        sv = src_h[1] / z
    valid = np.isfinite(su) & np.isfinite(sv) & (z > 0)
    valid &= (su >= 0) & (su <= Ww - 1) & (sv >= 0) & (sv <= Hh - 1)
    sui = np.clip(np.round(su), 0, Ww - 1).astype(np.int64)
    svi = np.clip(np.round(sv), 0, Hh - 1).astype(np.int64)
    pred = src[svi, sui].reshape(Hh, Ww)
    return pred, valid.reshape(Hh, Ww)


def _target_grid(Hh: int, Ww: int) -> np.ndarray:
    us, vs = np.meshgrid(np.arange(Ww), np.arange(Hh))
    return np.stack([us.ravel(), vs.ravel(), np.ones(Hh * Ww)], 0).astype(np.float64)


def per_class_dseg(pred: np.ndarray, tgt: np.ndarray, valid: np.ndarray) -> dict:
    """d_seg total + per-class (keyed by TARGET class), over valid pixels."""
    ne = (pred != tgt) & valid
    out = {"total": float(ne.sum() / max(valid.sum(), 1)),
           "coverage": float(valid.mean())}
    for c in range(5):
        m = (tgt == c) & valid
        out[CLASS_NAMES[c]] = float(ne[m].mean()) if m.sum() else None
        out[CLASS_NAMES[c] + "_area"] = float(((tgt == c) & valid).mean())
    return out


def eval_predictor(L, poses, K, Kinv, tgt_grid, mode, params):
    """Aggregate d_seg over all p->p+1 transitions for persist + warp predictors.

    Returns dict with 'persist', 'warp', 'static_mode' aggregate d_seg + per-class.
    """
    P = L.shape[0]
    s_t, s_r, pitch = params
    # accumulate confusion-free counts: ne_count[class], tot_count[class], valid_tot
    def fresh():
        return {"ne": 0, "tot": 0, "ne_c": [0] * 5, "tot_c": [0] * 5}
    persist, warp = fresh(), fresh()
    for p in range(P - 1):
        src, tgt = L[p], L[p + 1]
        # persist (identity / no warp): valid everywhere
        ne_p = (src != tgt)
        persist["ne"] += int(ne_p.sum()); persist["tot"] += tgt.size
        # warp by relative pose (adjacent-pose proxy = pose of pair ending at target)
        H = pose_to_homography(poses[p + 1], K, Kinv, s_t, s_r, pitch)
        pred, valid = warp_labels(src, H, tgt_grid)
        ne_w = (pred != tgt) & valid
        warp["ne"] += int(ne_w.sum()); warp["tot"] += int(valid.sum())
        for c in range(5):
            mp = (tgt == c)
            persist["ne_c"][c] += int(ne_p[mp].sum()); persist["tot_c"][c] += int(mp.sum())
            mw = mp & valid
            warp["ne_c"][c] += int(ne_w[mw].sum()); warp["tot_c"][c] += int(mw.sum())

    def fin(d):
        o = {"total": d["ne"] / max(d["tot"], 1)}
        for c in range(5):
            o[CLASS_NAMES[c]] = (d["ne_c"][c] / d["tot_c"][c]) if d["tot_c"][c] else None
            o[CLASS_NAMES[c] + "_area"] = d["tot_c"][c] / max(d["tot"], 1)
        return o
    # static-mode (no warp, no pose): each frame vs global per-pixel mode
    ne_m = (L != mode[None])
    static = {"total": float(ne_m.mean())}
    for c in range(5):
        m = (L == c)
        static[CLASS_NAMES[c]] = float(ne_m[m].mean()) if m.sum() else None
    return {"persist": fin(persist), "warp": fin(warp), "static_mode": static,
            "warp_coverage": warp["tot"] / max((L.shape[0] - 1) * L[0].size, 1)}


def fit_calibration(L, poses, K, Kinv, tgt_grid, fit_classes=ROAD_PLANE_CLASSES):
    """Coordinate-descent fit of (s_t, s_r, pitch) minimizing road-plane d_seg.

    LOW capacity (3 global scalars shared across ALL transitions) -> cannot
    overfit; the per-frame variation is 100% from the stored pose. We allow s_t
    to take either sign so the warp direction is data-determined (no convention
    guess). Objective = aggregate d_seg over Road+Lane target pixels.
    """
    P = L.shape[0]

    def objective(s_t, s_r, pitch):
        ne = 0
        tot = 0
        for p in range(P - 1):
            src, tgt = L[p], L[p + 1]
            H = pose_to_homography(poses[p + 1], K, Kinv, s_t, s_r, pitch)
            pred, valid = warp_labels(src, H, tgt_grid)
            sel = np.isin(tgt, fit_classes) & valid
            ne += int(((pred != tgt) & sel).sum())
            tot += int(sel.sum())
        return ne / max(tot, 1)

    # stage 1: s_t over a signed log grid (s_r=0, pitch=0)
    mags = np.concatenate([-np.logspace(-1, -4, 14), [0.0], np.logspace(-4, -1, 14)])
    best = (0.0, 0.0, 0.0)
    best_obj = objective(0.0, 0.0, 0.0)  # == persist on road-plane classes
    persist_obj = best_obj
    for s in mags:
        o = objective(float(s), 0.0, 0.0)
        if o < best_obj:
            best_obj, best = o, (float(s), 0.0, 0.0)
    # stage 2: pitch over openpilot bounds at best s_t
    for pit in np.linspace(-0.09, 0.17, 14):
        o = objective(best[0], 0.0, float(pit))
        if o < best_obj:
            best_obj, best = o, (best[0], 0.0, float(pit))
    # stage 3: s_r (rotation scale) at best s_t, pitch
    for sr in np.concatenate([[0.0], np.logspace(-2, 1.0, 10)]):
        o = objective(best[0], float(sr), best[2])
        if o < best_obj:
            best_obj, best = o, (best[0], float(sr), best[2])
    # stage 4: refine s_t around the winner
    if best[0] != 0.0:
        for s in best[0] * np.linspace(0.4, 1.8, 12):
            o = objective(float(s), best[1], best[2])
            if o < best_obj:
                best_obj, best = o, (float(s), best[1], best[2])
    return {"s_t": best[0], "s_r": best[1], "pitch": best[2],
            "fit_roadplane_dseg": best_obj, "persist_roadplane_dseg": persist_obj,
            "fit_classes": [CLASS_NAMES[c] for c in fit_classes]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--n-pairs", type=int, default=0, help="0 = all in cache")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    t0 = time.time()
    cache = (REPO / args.cache) if not Path(args.cache).is_absolute() else Path(args.cache)
    z = np.load(cache, allow_pickle=False)
    L = np.asarray(z["lstars"], dtype=np.int64)
    poses = np.asarray(z["gt_poses"], dtype=np.float64)
    if args.n_pairs and args.n_pairs < L.shape[0]:
        L, poses = L[: args.n_pairs], poses[: args.n_pairs]
    P, Hh, Ww = L.shape
    K = intrinsics_at(Ww, Hh)
    Kinv = np.linalg.inv(K)
    tgt_grid = _target_grid(Hh, Ww)
    counts = np.stack([(L == c).sum(0) for c in range(5)], 0)
    mode = counts.argmax(0)

    fit = fit_calibration(L, poses, K, Kinv, tgt_grid)
    res = eval_predictor(L, poses, K, Kinv, tgt_grid, mode, (fit["s_t"], fit["s_r"], fit["pitch"]))

    # ---- decomposition + verdict (per-class warp vs persist) ----
    persist, warp = res["persist"], res["warp"]
    decomp = {}
    for c in range(5):
        nm = CLASS_NAMES[c]
        pv, wv = persist[nm], warp[nm]
        if pv is None or wv is None:
            decomp[nm] = None
            continue
        decomp[nm] = {
            "persist_dseg": pv, "warp_dseg": wv,
            "abs_improvement": pv - wv,
            "rel_improvement": (pv - wv) / pv if pv > 0 else 0.0,
            "target_area": persist[nm + "_area"],
        }
    road = decomp["Road"]; lane = decomp["Lane"]
    roadplane_improves = bool(road and road["abs_improvement"] > 0)
    # fraction of total persist d_seg that the pose-warp explains on road-plane classes
    pp_total = persist["total"]
    explained_roadplane = sum(
        (decomp[CLASS_NAMES[c]]["abs_improvement"] * decomp[CLASS_NAMES[c]]["target_area"])
        for c in ROAD_PLANE_CLASSES if decomp[CLASS_NAMES[c]]
    )
    if roadplane_improves and road["rel_improvement"] >= 0.10:
        verdict = "CONFIRMED_PARTIAL"
        verdict_note = ("pose-homography compresses the Road d_seg trajectory "
                        f"({road['rel_improvement']*100:.0f}% on Road); the residual concentrates in "
                        "Lane-survival + Movables + static classes (which need identity, not a ground "
                        "warp). The stored pose carries the road-plane d_seg modulation.")
    elif roadplane_improves:
        verdict = "WEAK_SIGNAL"
        verdict_note = ("pose-homography reduces Road d_seg but by <10%; pose carries less of the "
                        "trajectory than hoped at this (proxy, pre-R, raw-learned-pose) operating point.")
    else:
        verdict = "REFUTED"
        verdict_note = ("the fitted pose-homography does NOT reduce Road d_seg below persist; the raw "
                        "learned pose does not linearly drive a ground-plane warp that explains the "
                        "argmax trajectory (calibration did not close / non-planarity / pose!=ego-motion).")

    out = {
        "tool": "tools/measure_pose_warp_dseg.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "[macOS advisory / research-signal]",
        "score_claim": False, "promotion_eligible": False, "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False, "promotable": False,
        "frontier_pointer": "UNMOVED (advisory measurement; not a contest score)",
        "cache": str(cache.relative_to(REPO)) if str(cache).startswith(str(REPO)) else str(cache),
        "n_pairs": P, "n_transitions": P - 1, "seg_hw": [Hh, Ww],
        "intrinsics_K_seg_res": K.tolist(),
        "camera_height_m": CAMERA_HEIGHT_M,
        "fit_calibration": fit,
        "baselines": {
            "B_static_mode_total": res["static_mode"]["total"],
            "B_persist_total": persist["total"],
            "W_pose_warp_total": warp["total"],
            "warp_coverage": res["warp_coverage"],
        },
        "persist_per_class": persist,
        "warp_per_class": warp,
        "static_mode_per_class": res["static_mode"],
        "decomposition": decomp,
        "roadplane_explained_dseg_contribution": explained_roadplane,
        "persist_total_dseg": pp_total,
        "verdict": verdict,
        "verdict_note": verdict_note,
        "byte_implication": ("pose = 6 floats/frame, ALREADY stored for d_pose; any d_seg trajectory "
                             "it explains is FREE bytes (only the canonical scene + survival/movables "
                             "residual must be paid)."),
        "assumptions": {
            "PROVEN": "warp/persist/static d_seg = real argmax-disagreement vs frozen CPU-torch SegNet "
                      "argmax (lstars); no surrogate.",
            "INFERRED_pose_columns": "raw learned PoseNet 6-vector interpreted as [fwd, lat, vert, r0, r1, r2]; "
                                     "col0 (~33) is forward. Physical axis mapping is a flagged assumption.",
            "INFERRED_relative_pose": "relative motion frame(2p+1)->frame(2p+3) proxied by pose[p+1] "
                                      "(pairs are non-overlapping seq_len=2 -> lstars are 2 frames apart); "
                                      "the per-frame factor + learned-units->metric scale are absorbed into fitted s_t.",
            "PROXY_pre_R": "direct-partition warp PRE round-trip (no bicubic->uint8->resize R operator); "
                           "necessary-not-sufficient for warp-inside-the-witness-INR. Realized-through-R + "
                           "exact CPU/CUDA eval is the authority.",
            "calibration": f"3 global scalars fit (s_t, s_r, pitch) on Road+Lane; K from EON intrinsics scaled to {Ww}x{Hh}.",
        },
        "elapsed_secs": round(time.time() - t0, 1),
    }

    out_dir = Path(args.out_dir) if args.out_dir else (REPO / f"experiments/results/grok_pose_warp_dseg_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    _refuse_tmp(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("verdict", "verdict_note", "baselines", "fit_calibration")}, indent=2))
    print(f"\n[written] {out_path}")
    print(f"[per-class warp vs persist]")
    for c in range(5):
        d = decomp[CLASS_NAMES[c]]
        if d:
            print(f"  {CLASS_NAMES[c]:8s} area={d['target_area']:.3f}  persist={d['persist_dseg']:.4f}  "
                  f"warp={d['warp_dseg']:.4f}  rel_impr={d['rel_improvement']*100:+.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
