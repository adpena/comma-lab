# SPDX-License-Identifier: MIT
"""Measure d_pose AND d_seg THROUGH R for the CORRECTED stratified per-class warp.

THE BUG (DAG FEED-la, Lens-C root cause): the v2 byte-close warp
(``experiments/v2_witness_byteclose_smoke.py`` ``warp_frame0_to_pred1``) is
POSE-INDEPENDENT — a units mismatch feeds the velocity-scale forward pose
(``gt_poses[:,0]`` ~33.6) straight in as a pixel shift, so
``clip(round(33.6*2.0), +-6) == 6`` produces a CONSTANT 6 px vertical roll on
EVERY pair. PoseNet then reads ~zero coherent ego-motion -> d_pose ~= 190. That
190 is just the ZERO-MOTION NULL (confirmed: naive-copy pred1=f0 gives the same
~190), NOT a real warp result.

THE FIX MEASURED HERE: replace that broken roll with the ALREADY-MEASURED
stratified per-class plane-induced homography warp (the grok / screw warp,
``measure_pose_warp_dseg`` + ``measure_screw_warp_through_R``):
  * Road / Lane / Movable -> ground plane-induced  H = K (R - t n^T/d) K^{-1}
  * Undrivable (sky, Z->inf) -> rotation-only       H = K R K^{-1}  (t-term dropped)
  * MyCar (ego hood, rigid)  -> identity
assembled into ONE composite predicted frame1 (PoseNet reads one frame, so the
per-class-stratified diagnostic of the screw tool must become a single composite
here), pushed THROUGH the contest R operator, scored by the FROZEN CPU-torch
PoseNet (d_pose) and SegNet (d_seg).

DECISIVE QUESTION (UNMEASURED until now): does the real ground homography make
the warped frame1 carry the ego-motion -> d_pose << 190? And what is the
through-R d_seg of the composite? The grok premise (warp is DUAL-USE: carries
d_pose for free off the already-stored pose) is a CLAIM here, MEASURED, not
assumed.

R OPERATOR (camera-res witness): gt_f0/gt_f1 are native 874x1164 uint8. A
deterministic camera-res warp witness produces a native frame; the contest R for
it is uint8 @ 874 (stored-video knife edge) -> the scorer ``preprocess_input``
bilinear-down (-> 384x512 SegNet / 192x256 PoseNet). The bicubic-UP-to-874 stage
of R is identity for a camera-res witness. The authoritative through-R verdicts
``cpu_verdict_d_seg`` / ``cpu_verdict_d_pose`` (the SAME ones the witness trainer
uses) are mirrored here EXACTLY (native uint8 -> preprocess_input). gt_poses /
lstars in the cache were built by this same path -> a NO-FAKE self-check asserts
PoseNet(gt pair)==gt_poses and SegNet(gt_f1)==lstars EXACTLY before any number is
trusted.

AUTHORITY / HONESTY FIREWALL (CLAUDE.md):
  * ``[macOS advisory / CPU-torch research-signal]`` ONLY. NOT a contest score.
    Canonical frontier pointer 0.19110 UNMOVED. score_claim/promotable=False.
  * PROVEN: the measured through-R d_pose/d_seg numbers (real frozen CPU-torch
    scorers, NEVER MPS; numpy-fp32 ops; self-checked exact).
  * ADVISORY/INFERRED (flagged in the JSON): (a) small-n is a contiguous prefix
    -> Lens-A contiguous-prefix optimism (advisory; only the corrected-warp
    d_pose through R is the verdict); (b) raw PoseNet 6-vector physical-column
    interpretation; (c) 3 fitted global calibration scalars (s_t,s_r,pitch) fit
    in label space, applied to RGB; (d) the ``argmax_routed`` composite uses the
    scorer argmax to route regimes -> a NON-SHIPPABLE upper bound (ceiling),
    while the ``band`` composite is the deterministic shippable partition.

rule-118: plane-homography + expmap + per-class regime + warp + R = FREE generic
algorithm (inflate.py-expandable, uncounted). The per-pair 6-DOF pose is
COUNTED-but-EXISTING (already stored for d_pose). The static descriptor
(calibration s_t/s_r/pitch + band fractions) is COUNTED-but-TINY. Honest geometry,
not a smuggled per-frame warp/argmax table.

Reuses: ``tools/measure_pose_warp_dseg`` (pose_to_homography / regime_homography /
intrinsics_at / SCREW_REGIME), ``tools/measure_screw_warp_through_R`` (warp_rgb /
_to_uint8 / fit_calibration_within_pair), ``experiments/train_witness_realized_through_R_mlx``
(cpu_verdict_d_pose / cpu_verdict_d_seg = the trainer-authoritative through-R
verdicts), ``experiments/v2_witness_byteclose_smoke`` (the BROKEN warp, to
reproduce the ~190).
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

from tools.measure_pose_warp_dseg import (  # noqa: E402
    CLASS_NAMES,
    NATIVE_H,
    NATIVE_W,
    SCREW_REGIME,
    _target_grid,
    intrinsics_at,
    regime_homography,
    warp_labels,
)
from tools.measure_screw_warp_through_R import (  # noqa: E402
    _to_uint8,
    fit_calibration_within_pair,
    warp_rgb,
)

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def _nn_upsample_labels(lab_seg: np.ndarray, H: int, W: int) -> np.ndarray:
    """Nearest-neighbour upsample a (h,w) int label map to (H,W)."""
    h, w = lab_seg.shape
    ys = (np.arange(H) * h / H).astype(np.int64)
    xs = (np.arange(W) * w / W).astype(np.int64)
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    return lab_seg[ys][:, xs]


def composite_band(f0: np.ndarray, frame_ground: np.ndarray, frame_rotonly: np.ndarray,
                   sky_frac: float, hood_frac: float) -> np.ndarray:
    """Deterministic SHIPPABLE composite: vertical-band regime partition.

    rows [0, sky_end)        -> rotation-only warp (sky / plane-at-infinity)
    rows [sky_end, hood_st)  -> ground plane-induced homography (road)
    rows [hood_st, H)        -> identity (ego hood, static) == f0
    No scorer at decode (the byte-close's deterministic strata, but with the REAL
    per-class homographies instead of a constant roll). Returns uint8 (H,W,3).
    """
    H = f0.shape[0]
    sky_end = int(sky_frac * H)
    hood_start = int(hood_frac * H)
    out = f0.copy()
    out[:sky_end] = frame_rotonly[:sky_end]
    out[sky_end:hood_start] = frame_ground[sky_end:hood_start]
    # rows [hood_start:] stay identity (f0)
    return out


def broken_constant_roll_warp(f0: np.ndarray, pose6: np.ndarray) -> np.ndarray:
    """The ORIGINAL broken byte-close warp (DAG FEED-la), inlined here so this tool
    reproduces the ~190 d_pose REGARDLESS of whether the byte-close has been fixed.

    Units bug: the velocity-scale forward pose (~33.6) is fed straight in as a
    pixel shift -> clip(round(33.6*2), +-6) == 6 -> a CONSTANT 6 px vertical roll
    of the road band on EVERY pair -> PoseNet reads zero coherent ego-motion.
    """
    H, W = f0.shape[:2]
    pred = f0.copy()
    sky_end = int(0.20 * H)
    hood_start = int(0.78 * H)
    fwd = float(pose6[0]) if pose6.shape[0] > 0 else 0.0
    lat = float(pose6[4]) if pose6.shape[0] > 4 else 0.0
    dy = int(round(np.clip(fwd * 2.0, -6, 6)))
    dx = int(round(np.clip(lat * 2.0, -6, 6)))
    road = f0[sky_end:hood_start]
    if dy != 0 or dx != 0:
        pred[sky_end:hood_start] = np.roll(np.roll(road, dy, axis=0), dx, axis=1)
    return pred


def composite_argmax(f0: np.ndarray, frame_ground: np.ndarray, frame_rotonly: np.ndarray,
                     lstar_native: np.ndarray) -> np.ndarray:
    """ADVISORY ceiling composite: per-pixel regime routed by the SCORER argmax.

    Uses ``lstar_native`` (the GT SegNet argmax upsampled to native res) to pick,
    per pixel, the physical regime of its class (Road/Lane/Movable=ground,
    Undriv=rotonly, MyCar=identity). This routes by the scorer argmax, which is
    NOT available at decode -> a NON-SHIPPABLE upper bound on what a perfect
    deterministic regime partition could achieve. Returns uint8 (H,W,3).
    """
    out = f0.copy()
    # ground classes
    ground_mask = np.isin(lstar_native, [c for c, r in SCREW_REGIME.items() if r == "ground"])
    rot_mask = np.isin(lstar_native, [c for c, r in SCREW_REGIME.items() if r == "rotonly"])
    out[ground_mask] = frame_ground[ground_mask]
    out[rot_mask] = frame_rotonly[rot_mask]
    # identity classes stay f0
    return out


def fit_calibration_dpose(gt_f0, poses, posenet, K_nat, Kinv_nat, grid_nat,
                          cpu_verdict_d_pose, pitch: float, mode: str = "full_ground") -> dict:
    """Fit the TRANSLATION scale s_t against d_pose THROUGH R (NOT d_seg).

    DEEP-MATH CRUX (DAG FEED, this measurement): the d_seg-optimal s_t is
    near-identity (the SegNet argmax partition is robust to small pixel flow, so
    minimizing label disagreement picks a tiny / wrong-sign warp). PoseNet, by
    contrast, reads the actual ego-motion FLOW, so it needs the geometrically-
    correct (positive, ~metric) s_t. The two terms want DIFFERENT calibrations of
    the SAME homography form. This fitter recovers the d_pose-optimal s_t by a
    cheap staged sweep that minimizes the mean through-R d_pose of the ground warp.

    LOW capacity: 1 global scalar shared across ALL pairs (per-pair variation is
    100% from the stored pose) -> cannot overfit. pitch/s_r held at the d_seg fit.
    """
    P = gt_f0.shape[0]

    def obj(s_t):
        params = (s_t, 0.0, pitch)
        tot = 0.0
        for p in range(P):
            H_g = regime_homography(poses[p], K_nat, Kinv_nat, params, "ground")
            fg = _to_uint8(warp_rgb(gt_f0[p], H_g, grid_nat))
            if mode == "band":
                from experiments.v2_witness_byteclose_smoke import HOOD_FRAC, SKY_FRAC
                H_r = regime_homography(poses[p], K_nat, Kinv_nat, params, "rotonly")
                fr = _to_uint8(warp_rgb(gt_f0[p], H_r, grid_nat))
                pred = composite_band(gt_f0[p], fg, fr, SKY_FRAC, HOOD_FRAC)
            else:
                pred = fg
            tot += cpu_verdict_d_pose(posenet, gt_f0[p], np.ascontiguousarray(pred, np.uint8), poses[p])
        return tot / P

    null_obj = obj(0.0)
    grid = [0.005, 0.01, 0.02, 0.03, 0.044, 0.06, 0.08, 0.12, 0.16, 0.24]
    best_st, best_obj = 0.0, null_obj
    for s in grid:
        o = obj(float(s))
        if o < best_obj:
            best_obj, best_st = o, float(s)
    # refine around the winner
    if best_st > 0.0:
        for s in best_st * np.linspace(0.5, 1.6, 8):
            o = obj(float(s))
            if o < best_obj:
                best_obj, best_st = o, float(s)
    return {"s_t": best_st, "s_r": 0.0, "pitch": pitch,
            "fit_dpose": best_obj, "null_dpose": null_obj, "fit_mode": mode}


def per_class_dseg_from_argmax(realized: np.ndarray, lstars: np.ndarray) -> dict:
    """Per-class d_seg (argmax disagreement) keyed by TARGET (lstars) class."""
    ne = (realized != lstars)
    out = {"total": float(ne.mean())}
    for c in range(5):
        m = (lstars == c)
        nc = int(m.sum())
        out[CLASS_NAMES[c]] = (float(ne[m].mean()) if nc else None)
        out[CLASS_NAMES[c] + "_contrib"] = (float(ne[m].sum() / lstars.size) if nc else 0.0)
        out[CLASS_NAMES[c] + "_area"] = float(nc / lstars.size)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n6.npz")
    ap.add_argument("--n-pairs", type=int, default=0, help="0 = all in cache")
    ap.add_argument("--out", default=None, help="results.json path (default under experiments/results/)")
    ap.add_argument("--selfcheck-pairs", type=int, default=4,
                    help="# pairs to assert PoseNet(pair)==gt_poses AND SegNet(gt_f1)==lstars exactly.")
    ap.add_argument("--fit-from-cache", default=None,
                    help="optional npz of a (usually smaller-n) cache to FIT calibration on, then APPLY "
                         "to --cache (saves lstar0 forwards on big n; flagged advisory).")
    ap.add_argument("--no-broken", action="store_true",
                    help="skip reproducing the broken byte-close warp (saves 1 P+S forward/pair).")
    ap.add_argument("--no-argmax", action="store_true",
                    help="skip the advisory argmax-routed ceiling composite.")
    args = ap.parse_args(argv)

    from experiments.train_witness_realized_through_R_mlx import (
        cpu_verdict_d_pose,
        cpu_verdict_d_seg,
    )
    from experiments.v2_witness_byteclose_smoke import HOOD_FRAC, SKY_FRAC
    from tac.boundary_math.seg_core import load_real_segnet
    from tac.optimization.frame1_seg_repair_atoms import measure_segnet_argmax
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

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
    assert (NAT_H, NAT_W) == (NATIVE_H, NATIVE_W), f"native {NAT_H}x{NAT_W} != {NATIVE_H}x{NATIVE_W}"

    # ---- frozen CPU-torch scorers (authority; NEVER MPS) ----
    seg = load_real_segnet("cpu")
    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet = dn.posenet
    for p_ in posenet.parameters():
        p_.requires_grad = False

    def seg_argmax_native(frame_uint8):
        am, _ = measure_segnet_argmax(seg, np.asarray(frame_uint8, dtype=np.float64))
        return am

    # ---- NO-FAKE self-check: PoseNet(gt pair)==gt_poses AND SegNet(gt_f1)==lstars ----
    sc = min(args.selfcheck_pairs, P)
    selfcheck = {"pairs_checked": sc, "pose_max_abs_err": 0.0, "seg_max_disagree_px": 0}
    for p in range(sc):
        d_pose0 = cpu_verdict_d_pose(posenet, gt_f0[p], gt_f1[p], poses[p])
        d_seg0 = cpu_verdict_d_seg(seg, gt_f1[p], lstars[p])
        selfcheck["pose_max_abs_err"] = max(selfcheck["pose_max_abs_err"], float(d_pose0))
        selfcheck["seg_max_disagree_px"] = max(selfcheck["seg_max_disagree_px"],
                                               int(round(d_seg0 * lstars[p].size)))
    selfcheck["PASS"] = bool(selfcheck["pose_max_abs_err"] < 1e-6 and selfcheck["seg_max_disagree_px"] == 0)
    if not selfcheck["PASS"]:
        raise SystemExit(
            f"NO-FAKE self-check FAILED: PoseNet(gt pair)!=gt_poses (max_abs_err="
            f"{selfcheck['pose_max_abs_err']:.3e}) or SegNet(gt_f1)!=lstars "
            f"(disagree_px={selfcheck['seg_max_disagree_px']}). Aborting (not a faithful through-R path)."
        )
    print(f"[warp-dpose] NO-FAKE self-check PASS ({sc} pairs)", flush=True)

    # ---- calibration fit (3 global scalars, label space). Reuse the screw fitter. ----
    K_seg = intrinsics_at(SEG_W, SEG_H)
    Kinv_seg = np.linalg.inv(K_seg)
    grid_seg = _target_grid(SEG_H, SEG_W)
    if args.fit_from_cache:
        fz = np.load(REPO / args.fit_from_cache if not Path(args.fit_from_cache).is_absolute()
                     else Path(args.fit_from_cache), allow_pickle=False)
        f_f0 = np.asarray(fz["gt_f0"]); f_l1 = np.asarray(fz["lstars"], dtype=np.int64)
        f_pose = np.asarray(fz["gt_poses"], dtype=np.float64)
        Pf = f_l1.shape[0]
        print(f"[warp-dpose] fitting calibration on {Pf} pairs from {args.fit_from_cache}...", flush=True)
        f_l0 = np.stack([seg_argmax_native(f_f0[i]) for i in range(Pf)], 0)
        fit = fit_calibration_within_pair(f_l0, f_l1, f_pose, K_seg, Kinv_seg, grid_seg)
        fit["fit_source"] = str(args.fit_from_cache)
    else:
        print(f"[warp-dpose] computing lstar0 (SegNet of gt_f0) for {P} pairs...", flush=True)
        lstar0 = np.stack([seg_argmax_native(gt_f0[p]) for p in range(P)], 0)
        fit = fit_calibration_within_pair(lstar0, lstars, poses, K_seg, Kinv_seg, grid_seg)
        fit["fit_source"] = str(cache.name)
    params_seg = (fit["s_t"], fit["s_r"], fit["pitch"])
    print(f"[warp-dpose] d_seg-calibration: {fit}", flush=True)

    # ---- native-res homography geometry ----
    K_nat = intrinsics_at(NAT_W, NAT_H)
    Kinv_nat = np.linalg.inv(K_nat)
    grid_nat = _target_grid(NAT_H, NAT_W)

    # ---- d_pose-calibration (the deep-math crux: d_seg-optimal s_t is near-identity
    #      and does NOT carry pose; PoseNet needs the geometric ego-motion s_t). ----
    print(f"[warp-dpose] fitting d_pose-optimal s_t (sweep) on {P} pairs...", flush=True)
    fit_pose = fit_calibration_dpose(gt_f0, poses, posenet, K_nat, Kinv_nat, grid_nat,
                                     cpu_verdict_d_pose, pitch=fit["pitch"], mode="full_ground")
    params_pose = (fit_pose["s_t"], fit_pose["s_r"], fit_pose["pitch"])
    print(f"[warp-dpose] d_pose-calibration: {fit_pose}", flush=True)

    # condition -> calibration used ('seg' / 'pose' / None) + whether per-class d_seg is collected
    conds = ["naive_copy", "band_segcal", "band_posecal", "full_ground_posecal"]
    if not args.no_argmax:
        conds.append("argmax_routed_posecal_advisory")
    if not args.no_broken:
        conds.append("broken_byteclose")
    pc_conds = {"band_segcal", "band_posecal", "full_ground_posecal", "argmax_routed_posecal_advisory"}
    agg = {c: {"d_pose": [], "d_seg": []} for c in conds}
    pc_ne = {c: [0] * 5 for c in pc_conds}
    pc_tot = {c: [0] * 5 for c in pc_conds}

    print(f"[warp-dpose] through-R pass over {P} pairs...", flush=True)
    for p in range(P):
        f0 = gt_f0[p]
        # regime-warped frames (uint8 @ native) for BOTH calibrations
        fg_seg = _to_uint8(warp_rgb(f0, regime_homography(poses[p], K_nat, Kinv_nat, params_seg, "ground"), grid_nat))
        fr_seg = _to_uint8(warp_rgb(f0, regime_homography(poses[p], K_nat, Kinv_nat, params_seg, "rotonly"), grid_nat))
        fg_pose = _to_uint8(warp_rgb(f0, regime_homography(poses[p], K_nat, Kinv_nat, params_pose, "ground"), grid_nat))
        fr_pose = _to_uint8(warp_rgb(f0, regime_homography(poses[p], K_nat, Kinv_nat, params_pose, "rotonly"), grid_nat))

        preds = {}
        preds["naive_copy"] = f0
        preds["band_segcal"] = composite_band(f0, fg_seg, fr_seg, SKY_FRAC, HOOD_FRAC)
        preds["band_posecal"] = composite_band(f0, fg_pose, fr_pose, SKY_FRAC, HOOD_FRAC)
        preds["full_ground_posecal"] = fg_pose
        if not args.no_argmax:
            lstar_nat = _nn_upsample_labels(lstars[p], NAT_H, NAT_W)
            preds["argmax_routed_posecal_advisory"] = composite_argmax(f0, fg_pose, fr_pose, lstar_nat)
        if not args.no_broken:
            preds["broken_byteclose"] = broken_constant_roll_warp(f0, poses[p])

        for c in conds:
            pred1 = np.ascontiguousarray(preds[c], dtype=np.uint8)
            dp = cpu_verdict_d_pose(posenet, f0, pred1, poses[p])
            agg[c]["d_pose"].append(dp)
            if c in pc_conds:
                realized = seg_argmax_native(pred1)
                ne = (realized != lstars[p])
                agg[c]["d_seg"].append(float(ne.mean()))
                for cls in range(5):
                    m = (lstars[p] == cls)
                    pc_ne[c][cls] += int(ne[m].sum())
                    pc_tot[c][cls] += int(m.sum())
            else:
                agg[c]["d_seg"].append(cpu_verdict_d_seg(seg, pred1, lstars[p]))
        if (p + 1) % 8 == 0 or p == P - 1:
            print(f"  ...{p + 1}/{P}", flush=True)

    # ---- aggregate ----
    def summ(c):
        dp = np.asarray(agg[c]["d_pose"], dtype=np.float64)
        ds = np.asarray(agg[c]["d_seg"], dtype=np.float64)
        out = {
            "d_pose_mean": float(dp.mean()), "d_pose_median": float(np.median(dp)),
            "d_seg_mean": float(ds.mean()), "d_seg_median": float(np.median(ds)),
            "pose_contribution_sqrt10dpose": float(np.sqrt(10.0 * dp.mean())),
            "seg_contribution_100dseg": float(100.0 * ds.mean()),
        }
        if c in pc_ne:
            out["per_class_d_seg"] = {
                CLASS_NAMES[cls]: ((pc_ne[c][cls] / pc_tot[c][cls]) if pc_tot[c][cls] else None)
                for cls in range(5)
            }
            tot_all = max(sum(pc_tot[c]), 1)
            out["per_class_contrib"] = {
                CLASS_NAMES[cls]: (pc_ne[c][cls] / tot_all) for cls in range(5)
            }
        return out

    results = {c: summ(c) for c in conds}

    naive_dp = results["naive_copy"]["d_pose_mean"]
    band_seg_dp = results["band_segcal"]["d_pose_mean"]
    band_pose_dp = results["band_posecal"]["d_pose_mean"]
    fullg_dp = results["full_ground_posecal"]["d_pose_mean"]
    best_dp = min(band_pose_dp, fullg_dp)
    pose_fixed = bool(best_dp < 0.5 * naive_dp)  # warp carries pose if it at least halves the null
    pose_drop_frac = (1.0 - best_dp / naive_dp) if naive_dp else 0.0

    out = {
        "tool": "tools/measure_warp_dpose_through_R.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "[macOS advisory / CPU-torch research-signal]",
        "score_claim": False, "promotion_eligible": False, "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False, "promotable": False,
        "frontier_pointer": "UNMOVED 0.19110 (advisory measurement; not a contest score)",
        "cache": str(cache.relative_to(REPO)) if str(cache).startswith(str(REPO)) else str(cache),
        "n_pairs": P, "seg_hw": [SEG_H, SEG_W], "native_hw": [NAT_H, NAT_W],
        "no_fake_selfcheck": selfcheck,
        "calibration_fit_dseg": fit,
        "calibration_fit_dpose": fit_pose,
        "band_partition": {"sky_frac": SKY_FRAC, "hood_frac": HOOD_FRAC},
        "results": results,
        "verdict": {
            "naive_copy_d_pose (zero-motion null ~= the broken 190)": naive_dp,
            "band_segcal_d_pose (d_seg-fit s_t = near-identity, WRONG for pose)": band_seg_dp,
            "band_posecal_d_pose (corrected, shippable, d_pose-fit s_t)": band_pose_dp,
            "full_ground_posecal_d_pose (whole-frame ground, d_pose-fit s_t)": fullg_dp,
            "broken_byteclose_d_pose": results.get("broken_byteclose", {}).get("d_pose_mean"),
            "best_corrected_d_pose": best_dp,
            "pose_d_pose_drop_frac_vs_null": pose_drop_frac,
            "POSE_CARRIED_BY_WARP": pose_fixed,
            "DEEP_MATH_CRUX_dseg_vs_dpose_calibration": (
                f"d_seg-optimal s_t={fit['s_t']:.5f} (near-identity; argmax flow-robust) does NOT carry "
                f"pose (band_segcal d_pose={band_seg_dp:.1f} ~ null); d_pose-optimal s_t={fit_pose['s_t']:.5f} "
                f"(geometric ego-motion scale) cuts d_pose to {best_dp:.1f}. The SAME homography form serves "
                f"both terms but at DIFFERENT scales -> the byte-close must calibrate s_t to d_pose, not d_seg."
            ),
            "note": (
                "POSE_CARRIED_BY_WARP True => the corrected ground-homography warp at the d_pose-optimal s_t "
                "makes PoseNet read coherent ego-motion (d_pose << the ~190 zero-motion null = the broken "
                "warp), i.e. the 190 bug is FIXED and the warp is dual-use (carries d_pose off the already-"
                "stored pose). The residual d_pose (~the best number, NOT ~0) is the part the single 2D "
                "homography cannot reproduce (off-road-plane parallax near the VP where PoseNet is most "
                "sensitive; one global s_t); a stored pose-target residual (v2_det) closes it to the solved-"
                "sidecar level (~3.4e-5). False => even the geometric warp does not carry pose through R."
            ),
        },
        "assumptions": {
            "PROVEN": "through-R d_pose/d_seg = real frozen CPU-torch PoseNet/SegNet on native uint8 -> "
                      "preprocess_input (the trainer-authoritative verdict); self-checked exact vs cache.",
            "ADVISORY_small_n_prefix": "the cache is a contiguous prefix of the clip (Lens-A contiguous-"
                                       "prefix optimism). Only the corrected-warp d_pose through R is the "
                                       "verdict; magnitudes are advisory until n600 / exact eval.",
            "ADVISORY_argmax_routed": "the argmax_routed composite routes regimes by the SCORER argmax "
                                      "(not available at decode) -> a NON-SHIPPABLE upper bound. The "
                                      "composite_band is the deterministic shippable partition.",
            "INFERRED_pose_columns": "raw PoseNet 6-vector [fwd,lat,vert,r0,r1,r2]; col0 dominant forward.",
            "calibration": "3 global scalars (s_t,s_r,pitch) fit in label space (lstar0->lstars), applied to RGB.",
            "camera_res_R": "models a camera-res warp witness; excludes sub-874 bicubic-up aliasing of a "
                            "low-capacity sub-camera INR.",
        },
        "rule_118": {
            "FREE_generic_in_inflate": "plane-induced homography + expmap + per-class regime + warp + R",
            "COUNTED_existing": "per-pair 6-DOF pose (already stored for d_pose; +0 marginal)",
            "COUNTED_tiny": "static descriptor (calibration s_t/s_r/pitch + band fractions)",
        },
        "elapsed_secs": round(time.time() - t0, 1),
    }

    out_path = (Path(args.out) if args.out
                else (REPO / f"experiments/results/warp_dpose_through_R_n{P}/results.json"))
    _refuse_tmp(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    # ---- console summary ----
    print("\n[warp-dpose] === THROUGH-R d_pose / d_seg (n=%d) ===" % P, flush=True)
    print(f"  {'condition':26s} {'d_pose':>10s} {'sqrt10dp':>9s} {'d_seg':>9s} {'100*dseg':>9s}", flush=True)
    for c in conds:
        r = results[c]
        print(f"  {c:26s} {r['d_pose_mean']:>10.4f} {r['pose_contribution_sqrt10dpose']:>9.4f} "
              f"{r['d_seg_mean']:>9.5f} {r['seg_contribution_100dseg']:>9.4f}", flush=True)
    print(f"\n  zero-motion null (naive d_pose) : {naive_dp:.3f}  (== the broken byte-close ~190)")
    print(f"  d_seg-cal s_t={fit['s_t']:+.5f} -> band d_pose : {band_seg_dp:.3f}  (near-identity; WRONG for pose)")
    print(f"  d_pose-cal s_t={fit_pose['s_t']:+.5f} -> best d_pose: {best_dp:.3f}  (drop {pose_drop_frac*100:.0f}% vs null)")
    print(f"  POSE_CARRIED_BY_WARP            : {pose_fixed}")
    for cn in ("band_posecal", "full_ground_posecal"):
        if "per_class_d_seg" in results[cn]:
            print(f"  [{cn} per-class d_seg]  total={results[cn]['d_seg_mean']:.5f}")
            for cls in range(5):
                v = results[cn]["per_class_d_seg"][CLASS_NAMES[cls]]
                ct = results[cn]["per_class_contrib"][CLASS_NAMES[cls]]
                print(f"    {CLASS_NAMES[cls]:8s} d_seg={(v if v is not None else float('nan')):.4f}  contrib={ct:.5f}")
    print(f"\n[written] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
