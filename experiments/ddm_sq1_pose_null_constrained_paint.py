#!/usr/bin/env python
"""ddm_sq1 Job 1c -- the POSE cure for the in-band realizer, measured not asserted.

Job 1b measured: the S4 cure (margin-optimal solved paint) pays the seg debt, but it moves
d_pose by ~1.5 orders of magnitude.  Pose is then the binding debt.  The named cure is gc16's
P7 rider: constrain the correction to the frame_1 yuv6-NULL subspace (Q3 / ph4).

DERIVED HERE, verified numerically (`upstream/frame_utils.py:51` rgb_to_yuv6):
per 2x2 block of SCORER pixels there are 12 RGB DOF and exactly 6 pose constraints --
  4 x  per-pixel  Y_p = .299R + .587G + .114B
  1 x  block-mean U  (with dY=0, dU = dB/1.772  =>  mean(dB) = 0)
  1 x  block-mean V  (with dY=0, dV = dR/1.402  =>  mean(dR) = 0)
so the pose-null subspace is REAL rank 6 of 12.  Independently reproduces ph5o's rank-6
generic basis.

THE CAVEAT THIS UNIT REFUSES TO HIDE.  Q3's "d_pose EXACTLY 0" is a REAL-valued statement.
Our actuator is INTEGER at the scorer lattice (all four camera pixels of a scorer pixel carry
one uint8 value, and D reproduces it exactly), and dY = 0 with coefficients .299/.587/.114 has
no nontrivial integer solution.  So exact nullity is NOT reachable by this actuator; the
achievable object is a minimum-|d yuv6| integer lattice point and its residual d_pose is an
OPEN MEASUREMENT.  This script measures that residual instead of assuming it away.

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE.  score_claim=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from ddm_sq1_eta_seg_realization import (  # noqa: E402
    CAM_H,
    CAM_W,
    N_PAIRS_TOTAL,
    Scorer,
    decode_gt_frames,
    label_boundary_band,
    seq_len,
)
from ddm_sq1_stage_decomposition_and_solved_paint import (  # noqa: E402
    confusion,
    realize_scorer_paint_to_camera,
    resize_to_scorer,
)

KY = (0.299, 0.587, 0.114)


def pose_null_projector() -> torch.Tensor:
    """(12,12) projector onto the frame_1 yuv6-null subspace of one 2x2 scorer block."""
    A = np.zeros((6, 12))
    for p in range(4):
        A[p, 3 * p: 3 * p + 3] = KY
    for p in range(4):
        A[4, 3 * p + 0] = 0.25      # block-mean dR = 0  (kills dV)
        A[5, 3 * p + 2] = 0.25      # block-mean dB = 0  (kills dU)
    # pinv, not solve(A@A.T): the normal-equations form trips a spurious
    # divide-by-zero warning in the macOS Accelerate BLAS path.  Same projector.
    P = np.eye(12) - np.linalg.pinv(A) @ A
    assert np.allclose(P @ P, P) and np.abs(A @ P).max() < 1e-10
    assert np.linalg.matrix_rank(P) == 6
    return torch.from_numpy(P).float()


def project_null(delta: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Project a (1,3,H,W) scorer-lattice delta blockwise onto the pose-null subspace."""
    b, c, h, w = delta.shape
    x = delta.reshape(b, c, h // 2, 2, w // 2, 2)          # (1,3,H/2,2,W/2,2)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(b, h // 2, w // 2, 12)
    x = x @ P.T
    x = x.reshape(b, h // 2, w // 2, 2, 2, c).permute(0, 5, 1, 3, 2, 4)
    return x.reshape(b, c, h, w)


def snap_band_to_blocks(band: np.ndarray) -> np.ndarray:
    """Grow the band to whole 2x2 scorer blocks.

    REQUIRED for exact nullity, not cosmetic.  Two of the six constraints are BLOCK-mean
    conditions (mean dR = 0, mean dB = 0).  Masking a projected delta at pixel granularity
    would zero part of a block and destroy exactly those two constraints -- the projection and
    a sub-block mask do not commute.  At block granularity they do: an included block is
    untouched by the mask, an excluded block is identically zero, and both are null.
    The cost is a slightly larger edited set, which is reported (`band_frac_snapped`).
    """
    b = band.reshape(band.shape[0] // 2, 2, band.shape[1] // 2, 2).any(axis=(1, 3))
    return np.repeat(np.repeat(b, 2, axis=0), 2, axis=1)


def solve_null_constrained(segnet, dec_f1, band, lgt, P, *, steps, lr, eval_every):
    """Same S4 cure as Job 1b, but every iterate lives in the pose-null subspace."""
    base = resize_to_scorer(dec_f1)
    tgt = torch.from_numpy(lgt.astype(np.int64))[None]
    m = torch.from_numpy(snap_band_to_blocks(band))[None, None].float()
    best = None
    with torch.enable_grad():
        raw = torch.zeros_like(base, requires_grad=True)
        opt = torch.optim.Adam([raw], lr=lr)
        for it in range(steps + 1):
            cur = torch.clamp(base + project_null(raw, P) * m, 0.0, 255.0)
            if it % eval_every == 0 or it == steps:
                q = torch.round(cur).detach()
                with torch.no_grad():
                    lam = segnet(q).argmax(dim=1)[0].numpy().astype(np.uint8)
                n_bad = int((lam != lgt).sum())
                if best is None or n_bad < best[0]:
                    best = (n_bad, q[0].permute(1, 2, 0).numpy().astype(np.uint8), f"null@{it}")
            if it == steps:
                break
            loss = torch.nn.functional.cross_entropy(segnet(cur), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return best


def yuv6_shift(a_u8: np.ndarray, b_u8: np.ndarray) -> dict:
    """How far did frame_1's yuv6 actually move? (the integer-lattice residual)."""
    def y(t):
        R, G, B = t[..., 0], t[..., 1], t[..., 2]
        Y = np.clip(R * KY[0] + G * KY[1] + B * KY[2], 0, 255)
        U = np.clip((B - Y) / 1.772 + 128.0, 0, 255)
        V = np.clip((R - Y) / 1.402 + 128.0, 0, 255)
        us = (U[0::2, 0::2] + U[1::2, 0::2] + U[0::2, 1::2] + U[1::2, 1::2]) * 0.25
        vs = (V[0::2, 0::2] + V[1::2, 0::2] + V[0::2, 1::2] + V[1::2, 1::2]) * 0.25
        return Y, us, vs
    Ya, ua, va = y(a_u8.astype(np.float64))
    Yb, ub, vb = y(b_u8.astype(np.float64))
    return {"max_abs_dY": float(np.abs(Ya - Yb).max()),
            "mean_abs_dY": float(np.abs(Ya - Yb).mean()),
            "max_abs_dU": float(np.abs(ua - ub).max()),
            "max_abs_dV": float(np.abs(va - vb).max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", type=Path, required=True)
    ap.add_argument("--gt-mkv", type=Path, required=True)
    ap.add_argument("--pairs-npy", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--lr", type=float, default=6.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    P = pose_null_projector()
    pairs = np.load(args.pairs_npy).tolist()
    if args.limit:
        pairs = pairs[: args.limit]
    raw = np.memmap(args.sub_dir / "inflated" / "0.raw", dtype=np.uint8, mode="r",
                    shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3))
    wanted = set()
    for p in pairs:
        wanted.update({seq_len * p, seq_len * p + 1})
    gt_frames = decode_gt_frames(args.gt_mkv, wanted)
    sc = Scorer(args.threads)
    segnet = sc.net.segnet

    rows = []
    if args.resume and args.out.exists():
        rows = json.loads(args.out.read_text()).get("rows", [])
        done = {int(r["pair"]) for r in rows}
        pairs = [p for p in pairs if p not in done]
    print(f"[sq1c] ready t={time.time()-t0:.1f}s, {len(pairs)} pairs, null rank 6/12", flush=True)

    for n, p in enumerate(pairs):
        tp = time.time()
        dec = np.stack([raw[seq_len * p], raw[seq_len * p + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * p], gt_frames[seq_len * p + 1]])
        lstar = sc.seg_argmax(dec)
        lgt = sc.seg_argmax(gt)
        band = label_boundary_band(lstar, 1)
        flips0 = lstar != lgt
        pose_gt = sc.pose_out(gt)

        rec = {"pair": int(p), "flips_before": int(flips0.sum()),
               "described_in_band": int((flips0 & band).sum()),
               "C_before": confusion(lgt, lstar).tolist(),
               "band_frac": float(band.mean()),
               "band_frac_snapped": float(snap_band_to_blocks(band).mean()),
               "d_pose_before": sc.d_pose(pose_gt, sc.pose_out(dec))}

        nbad, paint, tag = solve_null_constrained(
            segnet, dec[1], band, lgt, P,
            steps=args.steps, lr=args.lr, eval_every=args.eval_every)
        band_snapped = snap_band_to_blocks(band)
        cam = realize_scorer_paint_to_camera(dec[1], band_snapped, paint)
        pair_e = np.stack([dec[0], cam])
        lam = sc.seg_argmax(pair_e)
        fa = lam != lgt
        base_sc = np.round(resize_to_scorer(dec[1])[0].permute(1, 2, 0).numpy()).astype(np.uint8)
        rec.update({
            "tag": tag,
            "flips_after": int(fa.sum()),
            "fixed": int((flips0 & ~fa).sum()),
            "introduced": int(((~flips0) & fa).sum()),
            "eta_net": (int(flips0.sum()) - int(fa.sum())) / max(rec["described_in_band"], 1),
            "C_after": confusion(lgt, lam).tolist(),
            "d_pose_after": sc.d_pose(pose_gt, sc.pose_out(pair_e)),
            "yuv6_residual": yuv6_shift(base_sc, paint),
        })
        rows.append(rec)
        print(f"[sq1c] pair {p:3d} ({n+1}/{len(pairs)}) desc {rec['described_in_band']:5d} "
              f"eta {rec['eta_net']:+.4f} d_pose {rec['d_pose_before']:.6f}->"
              f"{rec['d_pose_after']:.6f} maxdY {rec['yuv6_residual']['max_abs_dY']:.3f} "
              f"[{time.time()-tp:.1f}s]", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"schema": "ddm_sq1_pose_null.v1", "score_claim": False,
                       "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
                       "pointer": "0.1910828242 [contest-CPU] UNMOVED",
                       "null_rank": 6, "null_dof_per_block": 12,
                       "solver": {"steps": args.steps, "lr": args.lr},
                       "rows": rows}, f, indent=1)
    print(f"[sq1c] DONE t={time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
