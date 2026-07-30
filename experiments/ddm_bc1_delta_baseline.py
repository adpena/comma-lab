# SPDX-License-Identifier: MIT
"""ddm_bc1 §3.5 delta precompute — the GT-ideal baseline + Knee-A pose-mass tail (self-contained).

Produces the fixed reference table the DEGRADED directional-delta verdict reads (MAIN Option A):
per pair {baseline_dpose = d_pose(GT_f0, GT_f1, target) ~1e-11 (the un-dropped ideal), the
noise floor} + the pose-mass TAIL subset (top-K by KNEE-A SENSITIVITY = how much freezing the
grid-dropped sky+hood rows of frame_1 hurts d_pose — exactly the pairs the coarse grid's
far-field freeze most affects). Fully self-contained: GT cache (gt_f0/gt_f1) + p3v2 targets +
frozen CPU PoseNet; NO v4c/v4d archive coupling (v4d's pose_warp format diverges and is not
needed — d_pose(GT_f0, .) with the IDEAL frame_0 isolates frame_1's pose cost directly).

Axis: [macOS-CPU advisory]. score_claim=false. Pointer 0.1910828242 [contest-CPU] UNMOVED.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path("/Users/adpena/Projects/pact")
_GT = str(_REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
CAMERA_H = 874
SEG_H = 384
DS = 16  # token grid downsample


def _skyhood_frozen(f1: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Simulate the coarse-grid far-field freeze on frame_1: the grid rows the mask DROPS
    (sky top + hood bottom) render as a constant (the renderer cannot update those cells).
    Constant-extrapolate the dropped camera-row bands from their kept boundary row."""
    kept_grid_rows = np.flatnonzero(keep_mask.any(axis=1))
    r_lo, r_hi = int(kept_grid_rows.min()), int(kept_grid_rows.max())
    # grid row -> SegNet row (x DS) -> camera row (x CAMERA_H/SEG_H)
    cam_lo = round(r_lo * DS * CAMERA_H / SEG_H)          # top of the kept band (camera)
    cam_hi = round((r_hi + 1) * DS * CAMERA_H / SEG_H)    # bottom of the kept band (camera)
    out = f1.copy()
    if cam_lo > 0:
        out[:cam_lo] = f1[cam_lo][None]                        # sky band -> boundary row
    if cam_hi < CAMERA_H:
        out[cam_hi:] = f1[cam_hi - 1][None]                    # hood band -> boundary row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--tail-k", type=int, default=17)  # QA66: top-17 = 74.3% pose mass
    ap.add_argument("--gt-cache", default=_GT)
    ap.add_argument("--keep-mask", default="/Volumes/VertigoDataTier/pact/ddm_sg1_20260731/"
                                           "qa24_grid_keep_mask_50.npy")
    ap.add_argument("--out", default="/Volumes/VertigoDataTier/pact/ddm_bc1_20260731/"
                                     "qa24_delta_reference.npz")
    args = ap.parse_args()

    for p in (str(_REPO / "src"), str(_REPO / "experiments"), str(_REPO / "upstream")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from ddm_composed_s_verdict import ComposedSVerdict

    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap

    verdict = ComposedSVerdict(args.num_pairs)
    if not verdict.available:
        print(f"REFUSE: ComposedSVerdict unavailable: {verdict.reason}", file=sys.stderr)
        return 4
    gt_f0 = open_stored_npy_memmap(args.gt_cache, "gt_f0")
    gt_f1 = open_stored_npy_memmap(args.gt_cache, "gt_f1")
    keep_mask = np.load(args.keep_mask)
    n = min(int(args.num_pairs), gt_f0.shape[0])

    baseline = np.zeros(n, np.float64)
    knee = np.zeros(n, np.float64)  # pose cost of freezing the grid-dropped sky+hood in f1
    for i in range(n):
        f0 = np.asarray(gt_f0[i], np.uint8)
        f1 = np.asarray(gt_f1[i], np.uint8)
        baseline[i] = verdict.d_pose_ideal_f0(i, f1, f0)
        knee[i] = verdict.d_pose_ideal_f0(i, _skyhood_frozen(f1, keep_mask), f0) - baseline[i]
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n} baseline_mean={baseline[: i + 1].mean():.3e} "
                  f"knee_mean={knee[: i + 1].mean():.5f}")

    tail_ids = np.argsort(-knee)[: int(args.tail_k)].astype(np.int64)
    tail_mass = float(knee[tail_ids].sum() / max(knee.sum(), 1e-12))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, baseline_dpose=baseline, knee_sensitivity=knee, tail_ids=tail_ids)
    print(f"SAVED {out}")
    print(f"baseline d_pose(GT_f0,GT_f1): mean={baseline.mean():.3e} max={baseline.max():.3e} "
          f"(the NOISE FLOOR — unchanged ideal frame_1; must be ~0)")
    print(f"knee-A sensitivity (freeze sky+hood in f1): mean={knee.mean():.5f} "
          f"max={knee.max():.5f}")
    print(f"tail-{args.tail_k}: {tail_ids.tolist()} = {tail_mass:.1%} of knee-A pose mass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
