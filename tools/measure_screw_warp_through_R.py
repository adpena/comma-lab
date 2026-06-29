# SPDX-License-Identifier: MIT
"""SCREW/TWIST warp d_seg THROUGH the contest render operator R (GAP-2 / the binding wall).

a513372a (``tools/measure_pose_warp_dseg.py`` ``screw_analysis``) measured the
single-ego-twist per-class stratified warp PRE-R, in LABEL space, pure numpy:
a ~0-byte WIN (Road reproduced exactly, hood->identity, sky->rotation-only; the
per-class-homography oracle's edge was 85% non-physical clip overfit). BUT that
probe had **no R operator**, so the binding Lane-survival residual (the ~0.58
lower bound) and the real bulk-warp number through the contest scorer were NOT
measurable. This tool converts the PRE-R lower bound into the REAL through-R
number: it warps the GT RGB by the SAME screw-derived per-class homographies,
pushes the warped RGB through the contest R chain, runs the FROZEN CPU-torch
SegNet, and measures the realized argmax-disagreement vs L* (the cached GT
SegNet argmax).

GEOMETRY (cleaner than a513372a's across-pair label probe): we warp WITHIN a
pair -- ``gt_f0[p]`` (frame 2p) forward to predict ``gt_f1[p]`` (frame 2p+1).
``L*[p] = lstars[p] = SegNet(gt_f1[p])`` is the d_seg reference. The pose
``gt_poses[p]`` is the raw PoseNet 6-vector estimated FROM that exact pair, so it
IS the f0->f1 relative ego-motion -- NO adjacent-pose proxy (a513372a had to use
``poses[p+1]``; we do not).

R OPERATOR (what acts here): ``gt_f0`` is ALREADY at camera resolution
(874x1164), so a deterministic camera-res warp witness produces a camera-res
frame. The contest R for such a frame is ``uint8 @ 874`` (the stored video
knife-edge) -> the scorer's ``preprocess_input`` bilinear-down to 384x512. The
contest's bicubic-UP-to-874 stage of R is IDENTITY for a camera-res witness
(there is no sub-camera render to up-sample), so it does not act here. This
models a CAMERA-RES deterministic-warp witness; it does NOT include the extra
sub-camera bicubic-up aliasing a low-capacity sub-874 INR would add (flagged).

ACCOUNTING (faithful through-R analog of a513372a's per-target-class screw):
per-class d_seg is read under the class's PHYSICAL REGIME warp:
  * Road / Lane / Movable -> ground plane-induced homography H = K(R - t n^T/d)K^-1
  * Undrivable (sky, Z->inf) -> rotation-only H = K R K^-1 (t-term dropped)
  * MyCar (ego hood, rigid) -> identity (== naive copy)
Each distinct regime is ONE frozen-SegNet forward on the warped RGB through R;
the screw total is the area-weighted aggregate (each target-class-c pixel scored
under regime(c)). This mirrors the label probe's ``dseg_for_target_class`` but
through R + the real SegNet. It is a per-target-class-stratified DIAGNOSTIC (the
right "does regime(c) predict class c" question), NOT a single blind-routed
composite render (a real witness routes the static classes via a stored hood-mask
/ sky-region descriptor -- the screw byte accounting -- which is a separate step).

AUTHORITY / HONESTY FIREWALL (CLAUDE.md):
  * ``[macOS advisory / CPU-torch research-signal]`` ONLY. NOT a contest score.
    Canonical frontier pointer 0.19110 UNMOVED. score_claim/promotable=False.
  * d_seg = REAL argmax-disagreement vs the cached frozen CPU-torch SegNet argmax
    ``lstars`` (``measure_segnet_argmax`` = the SAME preprocess_input/last-frame/
    bilinear-resize contract ``upstream/evaluate.py`` uses). Exact CPU-torch,
    NEVER MPS. A NO-FAKE self-check asserts ``SegNet(gt_f1) == lstars`` exactly.
  * It warps GT RGB (not a witness RGB we would ship) -> it bounds the deterministic
    bulk-warp part (upper-bound-ish: the true previous frame is a richer source
    than a single stored canonical) and gives a realistic Lane-through-R signal.
  * PROVEN: the measured through-R d_seg numbers. INFERRED (flagged): the physical
    interpretation of the raw PoseNet 6-vector columns; the 3 fitted global
    calibration scalars (s_t, s_r, pitch), fit in label space then APPLIED to RGB.

rule-118: the plane-induced-homography + expmap + per-class-regime + warp + R is a
FREE deterministic geometric algorithm (expandable inside inflate.py, uncounted).
The per-pair 6-DOF pose is COUNTED-but-EXISTING (already stored for d_pose). The
static scene descriptor (n, d, hood-mask, calibration) is COUNTED-but-TINY. NOT
FORBIDDEN: honest geometry, not a smuggled per-frame argmax/warp table.

Reuses a513372a's screw-homography fit machinery (pose_to_homography / expmap /
regime selection / warp_labels) from ``tools/measure_pose_warp_dseg.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Reuse a513372a's screw-homography machinery (the SAME fit / regime / warp).
from tools.measure_pose_warp_dseg import (  # noqa: E402
    CLASS_NAMES,
    NATIVE_H,
    NATIVE_W,
    ROAD_PLANE_CLASSES,
    SCREW_REGIME,
    intrinsics_at,
    pose_to_homography,
    regime_homography,
    warp_labels,
    _target_grid,
)

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


# --------------------------------------------------------------------------- #
# RGB inverse warp (bilinear) at native camera resolution, persist-fallback.
# --------------------------------------------------------------------------- #
def warp_rgb(src_hwc: np.ndarray, H: np.ndarray, tgt_grid: np.ndarray) -> np.ndarray:
    """Inverse-warp an (H,W,3) RGB frame by homography ``H`` (source->target).

    For each TARGET pixel q, sample the source at ``H^{-1} q`` with BILINEAR
    interpolation (RGB is a continuous field, unlike labels). Where the warp maps
    off-frame / behind the camera (invalid), FALL BACK to the source pixel at the
    same target location (persist) -- the SAME non-gameable accounting a513372a
    uses for the label warp. Returns float64 (H,W,3) in [0,255].
    """
    Hh, Ww, C = src_hwc.shape
    srcf = src_hwc.astype(np.float64)
    flat = srcf.reshape(-1, C)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Hinv = np.linalg.inv(H)
        src_h = Hinv @ tgt_grid  # (3, Hh*Ww)
        z = src_h[2]
        su = src_h[0] / z
        sv = src_h[1] / z
    valid = (
        np.isfinite(su) & np.isfinite(sv) & (z > 0)
        & (su >= 0) & (su <= Ww - 1) & (sv >= 0) & (sv <= Hh - 1)
    )
    su_c = np.clip(su, 0.0, Ww - 1)
    sv_c = np.clip(sv, 0.0, Hh - 1)
    x0 = np.floor(su_c).astype(np.int64)
    y0 = np.floor(sv_c).astype(np.int64)
    x1 = np.minimum(x0 + 1, Ww - 1)
    y1 = np.minimum(y0 + 1, Hh - 1)
    wx = (su_c - x0)[:, None]
    wy = (sv_c - y0)[:, None]
    Ia = flat[y0 * Ww + x0]
    Ib = flat[y0 * Ww + x1]
    Ic = flat[y1 * Ww + x0]
    Id = flat[y1 * Ww + x1]
    top = Ia * (1.0 - wx) + Ib * wx
    bot = Ic * (1.0 - wx) + Id * wx
    sampled = top * (1.0 - wy) + bot * wy
    # persist fallback where invalid: source pixel at the same target location.
    out = np.where(valid[:, None], sampled, flat)
    return out.reshape(Hh, Ww, C)


def _to_uint8(frame_f: np.ndarray) -> np.ndarray:
    """The stored-video uint8 knife-edge at camera resolution (part of R)."""
    return np.clip(np.round(frame_f), 0.0, 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Within-pair label-space calibration fit (cheap, pure numpy) on lstar0->lstars.
# --------------------------------------------------------------------------- #
def _label_warp_dseg_roadplane(L0, L1, poses, K, Kinv, tgt_grid, s_t, s_r, pitch,
                               fit_classes=ROAD_PLANE_CLASSES) -> float:
    """Aggregate full-coverage (persist-fallback) road-plane d_seg of warp(L0)->L1."""
    ne = 0
    tot = 0
    P = L0.shape[0]
    for p in range(P):
        H = pose_to_homography(poses[p], K, Kinv, s_t, s_r, pitch)
        pred, valid = warp_labels(L0[p], H, tgt_grid)
        pred = np.where(valid, pred, L0[p])  # persist fallback
        tsel = np.isin(L1[p], fit_classes)
        ne += int(((pred != L1[p]) & tsel).sum())
        tot += int(tsel.sum())
    return ne / max(tot, 1)


def fit_calibration_within_pair(L0, L1, poses, K, Kinv, tgt_grid) -> dict:
    """Coordinate-descent fit of (s_t, s_r, pitch) on Road+Lane, within-pair warp.

    LOW capacity (3 global scalars shared across ALL pairs) -> cannot overfit; the
    per-frame variation is 100% from the stored pose. Same staged search as
    a513372a's ``fit_calibration`` but on the within-pair lstar0->lstars warp using
    the exact pair pose ``gt_poses[p]`` (no adjacent-pose proxy).
    """
    def obj(s_t, s_r, pitch):
        return _label_warp_dseg_roadplane(L0, L1, poses, K, Kinv, tgt_grid, s_t, s_r, pitch)

    persist_obj = obj(0.0, 0.0, 0.0)
    best = (0.0, 0.0, 0.0)
    best_obj = persist_obj
    mags = np.concatenate([-np.logspace(-1, -4, 14), [0.0], np.logspace(-4, -1, 14)])
    for s in mags:
        o = obj(float(s), 0.0, 0.0)
        if o < best_obj:
            best_obj, best = o, (float(s), 0.0, 0.0)
    for pit in np.linspace(-0.09, 0.17, 14):
        o = obj(best[0], 0.0, float(pit))
        if o < best_obj:
            best_obj, best = o, (best[0], 0.0, float(pit))
    for sr in np.concatenate([[0.0], np.logspace(-2, 1.0, 10)]):
        o = obj(best[0], float(sr), best[2])
        if o < best_obj:
            best_obj, best = o, (best[0], float(sr), best[2])
    if best[0] != 0.0:
        for s in best[0] * np.linspace(0.4, 1.8, 12):
            o = obj(float(s), best[1], best[2])
            if o < best_obj:
                best_obj, best = o, (float(s), best[1], best[2])
    return {"s_t": best[0], "s_r": best[1], "pitch": best[2],
            "fit_roadplane_dseg": best_obj, "persist_roadplane_dseg": persist_obj}


def _label_screw_per_class(L0, L1, poses, K, Kinv, tgt_grid, params) -> dict:
    """PRE-R label-space screw per-class d_seg (cross-check vs a513372a)."""
    s_t, s_r, pitch = params
    ne_c = [0] * 5
    tot_c = [0] * 5
    P = L0.shape[0]
    # cache the 3 distinct regime warps per pair
    for p in range(P):
        preds = {}
        for regime in ("ground", "rotonly", "identity"):
            if regime == "identity":
                preds[regime] = L0[p]
            else:
                H = regime_homography(poses[p], K, Kinv, params, regime)
                pr, valid = warp_labels(L0[p], H, tgt_grid)
                preds[regime] = np.where(valid, pr, L0[p])
        for c in range(5):
            pred = preds[SCREW_REGIME[c]]
            m = (L1[p] == c)
            ne_c[c] += int(((pred != c) & m).sum())
            tot_c[c] += int(m.sum())
    out = {"total": sum(ne_c) / max(sum(tot_c), 1)}
    for c in range(5):
        out[CLASS_NAMES[c]] = (ne_c[c] / tot_c[c]) if tot_c[c] else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--n-pairs", type=int, default=0, help="0 = all in cache")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--selfcheck-pairs", type=int, default=4,
                    help="# pairs to assert SegNet(gt_f1)==lstars exactly (NO-FAKE).")
    args = ap.parse_args(argv)

    from tac.optimization.frame1_seg_repair_atoms import measure_segnet_argmax
    from tac.boundary_math.seg_core import load_real_segnet

    t0 = time.time()
    cache = (REPO / args.cache) if not Path(args.cache).is_absolute() else Path(args.cache)
    z = np.load(cache, allow_pickle=False)
    gt_f0 = np.asarray(z["gt_f0"])      # (P,874,1164,3) uint8
    gt_f1 = np.asarray(z["gt_f1"])      # (P,874,1164,3) uint8
    lstars = np.asarray(z["lstars"], dtype=np.int64)  # (P,384,512)
    poses = np.asarray(z["gt_poses"], dtype=np.float64)  # (P,6)
    P_cache = lstars.shape[0]
    P = P_cache if not args.n_pairs else min(args.n_pairs, P_cache)
    gt_f0, gt_f1, lstars, poses = gt_f0[:P], gt_f1[:P], lstars[:P], poses[:P]

    SEG_H, SEG_W = lstars.shape[1], lstars.shape[2]
    NAT_H, NAT_W = gt_f0.shape[1], gt_f0.shape[2]
    assert (NAT_H, NAT_W) == (NATIVE_H, NATIVE_W), f"native res {NAT_H}x{NAT_W} != {NATIVE_H}x{NATIVE_W}"

    seg = load_real_segnet("cpu")

    def seg_argmax(frame_uint8_native):
        am, _ = measure_segnet_argmax(seg, np.asarray(frame_uint8_native, dtype=np.float64))
        return am  # (SEG_H, SEG_W) int64

    # ---- NO-FAKE self-check: SegNet(gt_f1) MUST reproduce lstars exactly ----
    sc = min(args.selfcheck_pairs, P)
    selfcheck = {"pairs_checked": sc, "exact_matches": 0, "max_disagree_px": 0}
    for p in range(sc):
        am = seg_argmax(gt_f1[p])
        ndiff = int(np.count_nonzero(am != lstars[p]))
        selfcheck["max_disagree_px"] = max(selfcheck["max_disagree_px"], ndiff)
        if ndiff == 0:
            selfcheck["exact_matches"] += 1
    selfcheck["PASS"] = bool(selfcheck["exact_matches"] == sc and selfcheck["max_disagree_px"] == 0)
    if not selfcheck["PASS"]:
        raise SystemExit(
            f"NO-FAKE self-check FAILED: SegNet(gt_f1) != lstars "
            f"(max_disagree_px={selfcheck['max_disagree_px']}). SegNet pipeline is not faithful; "
            "aborting rather than reporting a fabricated number."
        )

    # ---- L*0 (source partition) for the within-pair calibration fit ----
    print(f"[screw-thru-R] computing lstar0 (SegNet of gt_f0) for {P} pairs...", flush=True)
    lstar0 = np.zeros_like(lstars)
    for p in range(P):
        lstar0[p] = seg_argmax(gt_f0[p])

    K_seg = intrinsics_at(SEG_W, SEG_H)
    Kinv_seg = np.linalg.inv(K_seg)
    grid_seg = _target_grid(SEG_H, SEG_W)
    fit = fit_calibration_within_pair(lstar0, lstars, poses, K_seg, Kinv_seg, grid_seg)
    params = (fit["s_t"], fit["s_r"], fit["pitch"])
    print(f"[screw-thru-R] calibration fit: {fit}", flush=True)

    # PRE-R label-space screw (cross-check vs a513372a ~0.01074 n96).
    pre_r_label = _label_screw_per_class(lstar0, lstars, poses, K_seg, Kinv_seg, grid_seg, params)
    print(f"[screw-thru-R] PRE-R label screw total={pre_r_label['total']:.5f} (a513372a n96=0.01074)", flush=True)

    # ---- THROUGH-R: warp RGB at native res, R, frozen SegNet, per-class d_seg ----
    K_nat = intrinsics_at(NAT_W, NAT_H)
    Kinv_nat = np.linalg.inv(K_nat)
    grid_nat = _target_grid(NAT_H, NAT_W)

    naive_ne = [0] * 5
    naive_tot = [0] * 5
    screw_ne = [0] * 5
    screw_tot = [0] * 5
    print(f"[screw-thru-R] through-R pass over {P} pairs (3 SegNet forwards/pair)...", flush=True)
    for p in range(P):
        # identity regime == naive copy through R (gt_f0 already uint8 @ camera res)
        am_identity = seg_argmax(gt_f0[p])
        # ground regime warp
        H_g = regime_homography(poses[p], K_nat, Kinv_nat, params, "ground")
        am_ground = seg_argmax(_to_uint8(warp_rgb(gt_f0[p], H_g, grid_nat)))
        # rotonly regime warp
        H_r = regime_homography(poses[p], K_nat, Kinv_nat, params, "rotonly")
        am_rotonly = seg_argmax(_to_uint8(warp_rgb(gt_f0[p], H_r, grid_nat)))
        regime_argmax = {"ground": am_ground, "rotonly": am_rotonly, "identity": am_identity}

        tgt = lstars[p]
        for c in range(5):
            m = (tgt == c)
            nc = int(m.sum())
            if not nc:
                continue
            # naive: identity (no warp) everywhere
            naive_ne[c] += int(((am_identity != c) & m).sum())
            naive_tot[c] += nc
            # screw: route class c to its physical regime
            sc_pred = regime_argmax[SCREW_REGIME[c]]
            screw_ne[c] += int(((sc_pred != c) & m).sum())
            screw_tot[c] += nc
        if (p + 1) % 16 == 0 or p == P - 1:
            print(f"  ...{p + 1}/{P}", flush=True)

    def _finalize(ne, tot):
        out = {"total": sum(ne) / max(sum(tot), 1)}
        per_area = {}
        all_tot = max(sum(tot), 1)
        for c in range(5):
            out[CLASS_NAMES[c]] = (ne[c] / tot[c]) if tot[c] else None
            per_area[CLASS_NAMES[c]] = tot[c] / all_tot
            # area-weighted contribution to the total d_seg
            out[CLASS_NAMES[c] + "_contrib"] = (ne[c] / all_tot)
        out["_area"] = per_area
        return out

    naive = _finalize(naive_ne, naive_tot)
    screw = _finalize(screw_ne, screw_tot)

    # ---- bulk vs lane vs movables decomposition (contributions to screw total) ----
    bulk_classes = ["Road", "Undriv", "MyCar"]   # deterministic-screw bulk (ground+sky+hood)
    bulk_contrib = sum(screw[c + "_contrib"] for c in bulk_classes)
    lane_contrib = screw["Lane_contrib"]
    movable_contrib = screw["Movable_contrib"]
    screw_total = screw["total"]
    BUDGET = 1.23e-3  # the v2 d_seg total budget the task names (100*d_seg ~ 0.123)
    decomposition = {
        "screw_through_R_total": screw_total,
        "bulk_warp_contrib": bulk_contrib,
        "lane_contrib": lane_contrib,
        "movable_contrib": movable_contrib,
        "bulk_classes": bulk_classes,
        "budget_d_seg_total": BUDGET,
        "bulk_fits_under_budget": bool(bulk_contrib <= BUDGET),
        "total_fits_under_budget": bool(screw_total <= BUDGET),
        "lane_share_of_total": (lane_contrib / screw_total) if screw_total else None,
        "bulk_share_of_total": (bulk_contrib / screw_total) if screw_total else None,
    }

    # screw vs naive: does the warp HELP through R+SegNet?
    warp_helps_total = bool(screw_total < naive["total"])
    per_class_warp_help = {
        CLASS_NAMES[c]: {
            "naive": naive[CLASS_NAMES[c]],
            "screw": screw[CLASS_NAMES[c]],
            "delta_screw_minus_naive": (
                (screw[CLASS_NAMES[c]] - naive[CLASS_NAMES[c]])
                if (screw[CLASS_NAMES[c]] is not None and naive[CLASS_NAMES[c]] is not None)
                else None
            ),
        }
        for c in range(5)
    }

    out = {
        "tool": "tools/measure_screw_warp_through_R.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "[macOS advisory / CPU-torch research-signal]",
        "score_claim": False, "promotion_eligible": False, "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False, "promotable": False,
        "frontier_pointer": "UNMOVED 0.19110 (advisory measurement; not a contest score)",
        "cache": str(cache.relative_to(REPO)) if str(cache).startswith(str(REPO)) else str(cache),
        "n_pairs": P, "seg_hw": [SEG_H, SEG_W], "native_hw": [NAT_H, NAT_W],
        "no_fake_selfcheck_segnet_gt_f1_eq_lstars": selfcheck,
        "calibration_fit_within_pair": fit,
        "pre_R_label_screw": pre_r_label,
        "pre_R_label_screw_a513372a_n96_total": 0.01074,
        "through_R_naive_copy": naive,
        "through_R_screw": screw,
        "warp_helps_through_R_total": warp_helps_total,
        "per_class_warp_help": per_class_warp_help,
        "decomposition_bulk_vs_lane_vs_movables": decomposition,
        "R_operator": ("warp@874 (bilinear, persist-fallback) -> uint8 @ 874 (knife-edge) -> "
                       "scorer preprocess_input bilinear-down 384x512 -> frozen CPU-torch SegNet -> argmax. "
                       "bicubic-UP-to-874 is identity for a camera-res warp witness (no sub-camera render)."),
        "accounting_note": ("per-target-class-stratified: each target-class-c pixel scored under regime(c) "
                            "[Road/Lane/Movable=ground, Undriv=rotonly, MyCar=identity]. Faithful through-R "
                            "analog of a513372a's label-space screw; NOT a single blind-routed composite render."),
        "rule_118": {
            "FREE_generic_in_inflate": "plane-induced homography + expmap + per-class regime + warp + R",
            "COUNTED_existing": "per-pair 6-DOF pose (already stored for d_pose; +0 marginal)",
            "COUNTED_tiny": "static scene descriptor (n, d, hood-mask, calibration s_t/s_r/pitch)",
            "not_forbidden": "honest geometry, not a smuggled per-frame argmax/warp table",
        },
        "assumptions": {
            "PROVEN": "through-R d_seg = real argmax-disagreement vs frozen CPU-torch SegNet argmax (lstars); "
                      "self-checked SegNet(gt_f1)==lstars exactly.",
            "INFERRED_pose_columns": "raw PoseNet 6-vector [fwd,lat,vert,r0,r1,r2]; col0 dominant forward.",
            "calibration": "3 global scalars (s_t,s_r,pitch) fit in label space (lstar0->lstars), applied to RGB.",
            "warps_GT_RGB": "warps the true previous frame gt_f0 (not a shipped witness RGB): bounds the "
                            "deterministic bulk-warp part (richer source than one stored canonical) and gives a "
                            "realistic Lane-through-R signal. Realized-through-R inside the witness INR + exact "
                            "CPU/CUDA eval on byte-closed bytes is the authority.",
            "camera_res_R": "models a camera-res warp witness; excludes sub-874 bicubic-up aliasing of a "
                            "low-capacity INR.",
        },
        "elapsed_secs": round(time.time() - t0, 1),
    }

    out_dir = (Path(args.out_dir) if args.out_dir
               else (REPO / f"experiments/results/screw_warp_through_R_n{P}"))
    _refuse_tmp(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(out, indent=2))

    # ---- console summary ----
    print("\n[screw-thru-R] === THROUGH-R per-class d_seg (n=%d) ===" % P, flush=True)
    print(f"  {'class':8s} {'regime':9s} {'area':>7s} {'naive':>9s} {'screw':>9s} {'contrib':>9s}", flush=True)
    for c in range(5):
        nm = CLASS_NAMES[c]
        nv = naive[nm]
        sv = screw[nm]
        print(f"  {nm:8s} {SCREW_REGIME[c]:9s} {screw['_area'][nm]:>7.3f} "
              f"{(nv if nv is not None else float('nan')):>9.4f} "
              f"{(sv if sv is not None else float('nan')):>9.4f} "
              f"{screw[nm + '_contrib']:>9.5f}", flush=True)
    print(f"  {'TOTAL':8s} {'':9s} {'':>7s} {naive['total']:>9.5f} {screw['total']:>9.5f}", flush=True)
    print(f"\n  PRE-R label screw total : {pre_r_label['total']:.5f}  (a513372a n96=0.01074)", flush=True)
    print(f"  through-R screw total   : {screw['total']:.5f}", flush=True)
    print(f"  through-R naive total   : {naive['total']:.5f}   warp_helps={warp_helps_total}", flush=True)
    print(f"\n  bulk-warp (Road+sky+hood) contrib : {bulk_contrib:.5f}  "
          f"(fits under {BUDGET:.2e}? {decomposition['bulk_fits_under_budget']})", flush=True)
    print(f"  lane contrib                      : {lane_contrib:.5f}  "
          f"({decomposition['lane_share_of_total']*100:.0f}% of total)", flush=True)
    print(f"  movable contrib                   : {movable_contrib:.5f}", flush=True)
    print(f"\n[written] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
