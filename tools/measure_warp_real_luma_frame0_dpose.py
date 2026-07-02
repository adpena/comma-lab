# SPDX-License-Identifier: MIT
"""$0 n600 verification of the WARP-REAL-LUMA FRAME0 pose carrier (carrier B).

Measures, through the FROZEN CPU-torch PoseNet authority (NEVER MPS), that
``tac.boundary_math.warp_real_luma_frame0``'s SE(3)-twist ground-homography warp
of the REAL ``gt_f0`` carries the ego-pose -> d_pose FAR below the zero-motion null
(the "pose-blind path-A" baseline). Reproduces the MEASURED reference
(``tools/measure_warp_dpose_through_R.py`` / FEED-lj / W7: 182->~10.5, -94%) using
THIS module's ``tac.lie``-xi warp, and adds the MLX<->numpy warp d_pose PARITY (the
load-bearing parity for a pose carrier: PoseNet averages over pixels, so per-pixel
fp32 floor-straddle washes out).

WHAT IS MEASURED (all through the authority PoseNet, native uint8 -> preprocess_input):
  * path_A_zero_motion  : pair = (gt_f0, gt_f0)                 -> the zero-motion null
        (the carrier does nothing / frame0 gives no ego-motion cue). ~= the both-synthetic
        pose-blind baseline the trained witness starts from (w_pose=0).
  * warp_carrier_numpy  : pair = (gt_f0, warp_uint8(gt_f0, xi)) -> the DECODE-authority
        (numpy-fp64) warp carrier. ``xi`` = the d_pose-calibrated forward ego-twist,
        built by THIS module's ``xi_from_pose_calibration`` (tac.lie exp/log round-trip
        of the reference plane-homography (R,t)).
  * warp_carrier_mlx    : same pair but frame1 = the MLX-fp32 warp -> the training
        fast-path fidelity check. Reported on a subset for the |d_pose| PARITY.

FRAMING NOTE (honesty): the warp here predicts the pair's second frame (f0 -> warped),
which is the SAME homography-by-xi physics the FRAME0 carrier exploits (warped-real-luma,
paired with a real/witness partner, lets PoseNet read the ego-motion). frame0-vs-warped
is a labeling; the measured physics (d_pose carried by the warp) is identical. The EXACT
synthetic-witness-f1 path-A and the residual-closed d_pose~3.4e-5 are TRAINER-time (need a
trained witness + w_pose>0); this module MAKES THEM REACHABLE (differentiable warp +
rank-6 twist residual), it does not fabricate their numbers.

AUTHORITY / HONESTY FIREWALL (CLAUDE.md):
  * ``[macOS advisory / CPU-torch research-signal]`` ONLY. NOT a contest score.
    Canonical frontier pointer 0.19110 UNMOVED. score_claim/promotable=False.
  * d_pose = REAL frozen CPU-torch PoseNet MSE on native uint8 (the trainer-authoritative
    ``cpu_verdict_d_pose``); NEVER MPS. A NO-FAKE self-check asserts PoseNet(gt pair)==gt_poses
    EXACTLY before any number is trusted (ABORT otherwise).
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

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def _chunks(n: int, size: int):
    for i in range(0, n, size):
        yield i, min(i + size, n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--n-pairs", type=int, default=0, help="0 = all in cache (n600 default)")
    ap.add_argument("--fit-pairs", type=int, default=24, help="# pairs to fit the global s_t on")
    ap.add_argument("--parity-pairs", type=int, default=48, help="# pairs for the MLX<->numpy d_pose parity")
    ap.add_argument("--chunk", type=int, default=50, help="PoseNet batch size")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selfcheck-pairs", type=int, default=4)
    args = ap.parse_args(argv)

    import mlx.core as mx  # noqa: F401 (import guard + warm the GPU)

    from experiments.train_witness_realized_through_R_mlx import (
        cpu_verdict_d_pose,
        cpu_verdict_d_pose_batch,
    )
    from tac.boundary_math.warp_real_luma_frame0 import (
        GroundHomographyGeom,
        warp_frame0_native_mlx,
        warp_frame0_uint8_numpy,
        xi_from_pose_calibration,
        xi_store_bytes,
    )
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    t0 = time.time()
    cache = (REPO / args.cache) if not Path(args.cache).is_absolute() else Path(args.cache)
    z = np.load(cache, allow_pickle=False)
    gt_f0 = np.asarray(z["gt_f0"])  # (P,874,1164,3) uint8
    gt_f1 = np.asarray(z["gt_f1"])
    poses = np.asarray(z["gt_poses"], dtype=np.float64)  # (P,6)
    P_cache = poses.shape[0]
    P = P_cache if not args.n_pairs else min(args.n_pairs, P_cache)
    gt_f0, gt_f1, poses = gt_f0[:P], gt_f1[:P], poses[:P]
    NAT_H, NAT_W = gt_f0.shape[1], gt_f0.shape[2]

    # ---- frozen CPU-torch PoseNet (authority; NEVER MPS) ----
    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet = dn.posenet
    for p_ in posenet.parameters():
        p_.requires_grad = False

    # ---- NO-FAKE self-check: PoseNet(gt pair) == gt_poses ----
    sc = min(args.selfcheck_pairs, P)
    max_err = 0.0
    for p in range(sc):
        max_err = max(max_err, cpu_verdict_d_pose(posenet, gt_f0[p], gt_f1[p], poses[p]))
    selfcheck_pass = bool(max_err < 1e-6)
    if not selfcheck_pass:
        raise SystemExit(
            f"NO-FAKE self-check FAILED: PoseNet(gt pair)!=gt_poses (max MSE {max_err:.3e}). Aborting."
        )
    print(f"[warp-f0] NO-FAKE self-check PASS ({sc} pairs)", flush=True)

    geom = GroundHomographyGeom.eon(native_hw=(NAT_H, NAT_W), pitch=0.0)
    geom_mlx = geom.mlx()

    def xi_for(pose6, s_t, s_r, pitch):
        return xi_from_pose_calibration(pose6, s_t, s_r, pitch, whole_ground=True)

    # ---- fit the d_pose-optimal global s_t (low-capacity: ONE scalar / clip) ----
    # Follows tools/measure_warp_dpose_through_R.fit_calibration_dpose: sweep s_t
    # (s_r=0, pitch=0 whole-ground) minimizing mean through-R d_pose on a subset.
    nf = min(args.fit_pairs, P)
    grid_st = [0.0, 0.02, 0.044, 0.08, 0.12, 0.16, 0.22, 0.30]

    def mean_dpose_for_st(s_t: float, idx: np.ndarray) -> float:
        preds = [warp_frame0_uint8_numpy(gt_f0[p], xi_for(poses[p], s_t, 0.0, 0.0), geom) for p in idx]
        dps = cpu_verdict_d_pose_batch(posenet, [gt_f0[p] for p in idx], preds, [poses[p] for p in idx])
        return float(np.mean(dps))

    fit_idx = np.arange(nf)
    fit_scores = {s: mean_dpose_for_st(s, fit_idx) for s in grid_st}
    best_st = min(fit_scores, key=fit_scores.get)
    null_fit = fit_scores[0.0]
    print(f"[warp-f0] s_t sweep (n={nf}): {json.dumps({str(k): round(v,3) for k,v in fit_scores.items()})} -> best s_t={best_st}", flush=True)

    # ---- FULL n600 pass: path-A null + warp carrier (numpy authority) ----
    d_null: list[float] = []
    d_warp: list[float] = []
    print(f"[warp-f0] full n{P} through-R d_pose pass (numpy authority warp)...", flush=True)
    for a, b in _chunks(P, args.chunk):
        idx = range(a, b)
        f0s = [gt_f0[p] for p in idx]
        # path-A proxy: zero-motion (pair (f0,f0))
        d_null.extend(cpu_verdict_d_pose_batch(posenet, f0s, f0s, [poses[p] for p in idx]))
        # warp carrier (numpy fp64 = decode authority)
        preds = [warp_frame0_uint8_numpy(gt_f0[p], xi_for(poses[p], best_st, 0.0, 0.0), geom) for p in idx]
        d_warp.extend(cpu_verdict_d_pose_batch(posenet, f0s, preds, [poses[p] for p in idx]))
        print(f"  ...{b}/{P}", flush=True)

    d_null = np.asarray(d_null); d_warp = np.asarray(d_warp)

    # ---- MLX<->numpy d_pose PARITY (subset) ----
    npar = min(args.parity_pairs, P)
    par_idx = np.linspace(0, P - 1, npar).astype(int)
    d_warp_mlx = []
    d_warp_np_sub = []
    for p in par_idx:
        xi = xi_for(poses[p], best_st, 0.0, 0.0)
        pred_np = warp_frame0_uint8_numpy(gt_f0[p], xi, geom)
        pred_mlx_f = np.asarray(warp_frame0_native_mlx(mx.array(gt_f0[p].astype(np.float32)), mx.array(xi.astype(np.float32)), geom_mlx))
        pred_mlx = np.clip(np.round(pred_mlx_f), 0, 255).astype(np.uint8)
        d_warp_np_sub.append(cpu_verdict_d_pose(posenet, gt_f0[p], pred_np, poses[p]))
        d_warp_mlx.append(cpu_verdict_d_pose(posenet, gt_f0[p], pred_mlx, poses[p]))
    d_warp_np_sub = np.asarray(d_warp_np_sub); d_warp_mlx = np.asarray(d_warp_mlx)
    dpose_parity_rel = float(np.mean(np.abs(d_warp_mlx - d_warp_np_sub) / (np.abs(d_warp_np_sub) + 1e-9)))
    dpose_parity_agreement = float(np.mean(np.abs(d_warp_mlx - d_warp_np_sub) < 0.05 * (np.abs(d_warp_np_sub) + 1e-9)))

    null_mean = float(d_null.mean())
    warp_mean = float(d_warp.mean())
    drop_frac = (1.0 - warp_mean / null_mean) if null_mean else 0.0

    out = {
        "tool": "tools/measure_warp_real_luma_frame0_dpose.py",
        "module": "tac.boundary_math.warp_real_luma_frame0",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "[macOS advisory / CPU-torch research-signal]",
        "score_claim": False, "promotion_eligible": False, "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False, "promotable": False,
        "frontier_pointer": "UNMOVED 0.19110 (advisory measurement; not a contest score)",
        "cache": str(cache.relative_to(REPO)) if str(cache).startswith(str(REPO)) else str(cache),
        "n_pairs": P, "native_hw": [NAT_H, NAT_W],
        "no_fake_selfcheck": {"pairs": sc, "pose_max_abs_mse": max_err, "PASS": selfcheck_pass},
        "calibration": {"s_t": best_st, "s_r": 0.0, "pitch": 0.0, "whole_ground": True,
                        "s_t_sweep_mean_dpose": {str(k): v for k, v in fit_scores.items()}},
        "results": {
            "path_A_zero_motion_null_d_pose": null_mean,
            "warp_carrier_d_pose": warp_mean,
            "warp_carrier_d_pose_median": float(np.median(d_warp)),
            "d_pose_drop_frac_vs_null": drop_frac,
            "pose_contribution_sqrt10_null": float(np.sqrt(10.0 * null_mean)),
            "pose_contribution_sqrt10_warp": float(np.sqrt(10.0 * warp_mean)),
        },
        "mlx_numpy_parity": {
            "parity_pairs": int(npar),
            "d_pose_mlx_mean": float(d_warp_mlx.mean()),
            "d_pose_numpy_mean": float(d_warp_np_sub.mean()),
            "d_pose_rel_diff_mean": dpose_parity_rel,
            "d_pose_agreement_within_5pct": dpose_parity_agreement,
        },
        "byte_accounting": {
            "xi_store_bytes_full_fp16": xi_store_bytes(P),
            "xi_store_bytes_lowrank_r2": xi_store_bytes(P, low_rank=2),
            "note": ("the per-pair 6-DOF twist is the ONLY counted payload of this carrier "
                     "(dual-use: warp + PoseNet ego-motion). The SOURCE luma at decode is a stored "
                     "keyframe (W9/W10 reach gate; separate counted payload) — NOT smuggled here."),
        },
        "verdict": {
            "WARP_CARRIES_POSE": bool(warp_mean < 0.5 * null_mean),
            "reproduces_reference_182_to_10p5": bool(5.0 < warp_mean < 30.0 and null_mean > 100.0),
            "note": ("path-A (pose-blind frame0 / zero-motion) ~= the null; the warp-real-luma carrier "
                     "cuts d_pose by ~90%+ (reproduces FEED-lj/W7 via tac.lie-xi). The residual from here "
                     "to d_pose~3.4e-5 is the w_pose>0 TRAINER job (rank-6 twist residual; this module is "
                     "differentiable so it is reachable). This warp is NOT the exact both-synthetic path-A "
                     "(needs a trained witness) — it is the MEASURED pose-carrier physics the carrier uses."),
        },
        "elapsed_secs": round(time.time() - t0, 1),
    }
    out_path = (Path(args.out) if args.out
                else (REPO / f"experiments/results/warp_real_luma_frame0_dpose_n{P}/results.json"))
    _refuse_tmp(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\n[warp-f0] === WARP-REAL-LUMA FRAME0 POSE CARRIER (n=%d) ===" % P, flush=True)
    print(f"  path-A zero-motion null d_pose : {null_mean:.4f}   (sqrt10 {np.sqrt(10*null_mean):.3f})", flush=True)
    print(f"  warp-real-luma carrier d_pose  : {warp_mean:.4f}   (sqrt10 {np.sqrt(10*warp_mean):.3f})  drop {drop_frac*100:.0f}%", flush=True)
    print(f"  MLX<->numpy d_pose parity      : rel_diff {dpose_parity_rel:.4f}  agree<5% {dpose_parity_agreement:.3f}", flush=True)
    print(f"  xi store (fp16 / lowrank-r2)   : {xi_store_bytes(P)} B / {xi_store_bytes(P, low_rank=2)} B", flush=True)
    print(f"  WARP_CARRIES_POSE={out['verdict']['WARP_CARRIES_POSE']}", flush=True)
    print(f"\n[written] {out_path}  ({out['elapsed_secs']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
