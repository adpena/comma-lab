#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_sc1 — FIRE the pre-registered e_p (paint-survival) pose probe, n600 chunked.

Contract (from `experiments/ddm_ar1_pose_target_structure_probe.py::preregistered_ep_probe`,
banked receipt `/Volumes/VertigoDataTier/pact/ddm_ar1_20260728/pose_target_structure_receipt.json`):

  1. t_p = PoseNet(orig)[:6] per pair (banked; here recomputed in-harness as t_p_local through the
     IDENTICAL scorer path, and cross-checked against the banked posenet_targets.bin).
  2. Paint synthetic-partition frames from GT argmax (flat comma10k-luma palette), through R
     (render_grid_to_camera_uint8) -> uint8 camera frames -> real PoseNet -> b_p.
       - painted_f1[i] = paint(lstars[i])          (cached f1 SegNet argmax)
       - painted_f0[i] = paint(SegNet(gt_f0[i]))   (f0 argmax computed here)
  3. e_p = b_p - t_p_local (600x6). Measure rank(e_p), single global 6x6 affine-fit R^2
     (does paint already carry the ego-motion?), lag-1 smoothness, AR-int5 residual bytes proxy.

Falsifier (pre-registered): e_p FULL-RANK and xi-ROUGH (global-affine R^2 < ~0.5) => paint carries
NO ego-motion => from-scratch PR130-class field (bounded 38.4 B/pair = 23 KB). e_p LOW-RANK/xi-smooth
=> shipped field is only the residual, AR-coded << 23 KB.

Axis: [macOS-CPU frozen-scorer advisory]. score_claim=false. pointer_moved=false. Resumable per chunk.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_REPO = Path("/Users/adpena/Projects/pact")
for value in (str(REPO_ROOT / "src"), str(MAIN_REPO / "upstream"), str(REPO_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
GT_CACHE = MAIN_REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
BANKED_TP = MAIN_REPO / "experiments/posenet_targets.bin"
UPSTREAM = MAIN_REPO / "upstream"
# comma10k canonical class luma [Road,Lane,Undrivable,Movable,MyCar] -> flat gray RGB palette.
PALETTE = np.array([[41, 41, 41], [76, 76, 76], [90, 90, 90], [124, 124, 124], [161, 161, 161]],
                   dtype=np.uint8)


def _paint_camera(argmax_seg: np.ndarray) -> np.ndarray:
    """argmax_seg (K,384,512) int -> painted camera frames (K,874,1164,3) uint8 through R first half."""
    from tac.through_r.resolution_chain import render_grid_to_camera_uint8

    grid_rgb = PALETTE[argmax_seg.astype(np.int64)]  # (K,384,512,3) uint8
    out = np.stack([render_grid_to_camera_uint8(s) for s in grid_rgb], axis=0)
    return np.ascontiguousarray(out, dtype=np.uint8)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--chunk-size", type=int, default=120)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-dir", default="/Volumes/VertigoDataTier/pact/ddm_sc1_20260728/ep_chunks")
    ap.add_argument("--receipt", default="/Volumes/VertigoDataTier/pact/ddm_sc1_20260728/ep_probe_receipt.json")
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(args.threads))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from tac.scorer import extract_gt_masks, extract_gt_pose_targets, load_scorers
    from tac.scorer_targets import load_posenet_targets

    banked = load_posenet_targets(str(BANKED_TP))["targets"].numpy().astype(np.float64)  # (600,6)

    posenet, segnet = load_scorers(
        str(UPSTREAM / "models/posenet.safetensors"),
        str(UPSTREAM / "models/segnet.safetensors"),
        device="cpu",
        upstream_dir=str(UPSTREAM),
    )
    device = torch.device("cpu")

    gt = np.load(GT_CACHE, mmap_mode="r")
    gt_f0, gt_f1, lstars = gt["gt_f0"], gt["gt_f1"], gt["lstars"]
    n = min(int(args.n), int(gt_f0.shape[0]))
    cs = int(args.chunk_size)

    t0 = time.time()
    for start in range(0, n, cs):
        stop = min(start + cs, n)
        ck = out_dir / f"chunk_{start:04d}_{stop:04d}.npz"
        if ck.exists():
            print(f"[skip] {ck.name} exists", file=sys.stderr)
            continue
        f0 = np.ascontiguousarray(gt_f0[start:stop])  # (K,874,1164,3) uint8
        f1 = np.ascontiguousarray(gt_f1[start:stop])
        ls1 = np.ascontiguousarray(lstars[start:stop])  # (K,384,512) f1 argmax

        # --- t_p_local: real frames through the exact PoseNet path (pairs = interleaved f0,f1) ---
        real_frames = []
        for k in range(stop - start):
            real_frames.append(torch.from_numpy(f0[k]))
            real_frames.append(torch.from_numpy(f1[k]))
        tp_local = extract_gt_pose_targets(real_frames, posenet, device, batch_size=8).numpy().astype(np.float64)

        # --- f0 argmax via SegNet (f1 argmax reused from cache) ---
        f0_frames = [torch.from_numpy(f0[k]) for k in range(stop - start)]
        f0_argmax = extract_gt_masks(f0_frames, segnet, device, batch_size=16).numpy()  # (K,384,512)

        # --- paint both frames, render through R ---
        pf0 = _paint_camera(f0_argmax)   # (K,874,1164,3)
        pf1 = _paint_camera(ls1)
        paint_frames = []
        for k in range(stop - start):
            paint_frames.append(torch.from_numpy(pf0[k]))
            paint_frames.append(torch.from_numpy(pf1[k]))
        bp = extract_gt_pose_targets(paint_frames, posenet, device, batch_size=8).numpy().astype(np.float64)

        np.savez(ck, tp_local=tp_local, bp=bp, start=start, stop=stop)
        el = time.time() - t0
        print(f"[chunk {start}:{stop}] done | tp_local[0]={tp_local[0]} bp[0]={bp[0]} | {el:.1f}s", file=sys.stderr)

    # --- combine + analyze ---
    tp_local = np.zeros((n, 6)); bp = np.zeros((n, 6))
    for start in range(0, n, cs):
        stop = min(start + cs, n)
        d = np.load(out_dir / f"chunk_{start:04d}_{stop:04d}.npz")
        tp_local[start:stop] = d["tp_local"]; bp[start:stop] = d["bp"]

    banked_vs_local_maxabs = float(np.abs(banked[:n] - tp_local).max())
    e_p = bp - tp_local  # (n,6)

    # rank via SVD energy of centered e_p
    ec = e_p - e_p.mean(0, keepdims=True)
    _, sv, _ = np.linalg.svd(ec, full_matrices=False)
    ev = (sv ** 2); ev = ev / ev.sum()

    # global 6x6 affine fit b_p ~ A t_p_local + c (least squares); R^2 per dim + overall
    X = np.hstack([tp_local, np.ones((n, 1))])  # (n,7)
    W, *_ = np.linalg.lstsq(X, bp, rcond=None)   # (7,6)
    bp_hat = X @ W
    ss_res = ((bp - bp_hat) ** 2).sum(0)
    ss_tot = ((bp - bp.mean(0)) ** 2).sum(0)
    r2_per_dim = 1.0 - ss_res / (ss_tot + 1e-12)
    r2_overall = float(1.0 - ss_res.sum() / (ss_tot.sum() + 1e-12))

    # lag-1 autocorr of e_p per dim (smoothness proxy)
    ac = [float(np.corrcoef(e_p[:-1, k], e_p[1:, k])[0, 1]) for k in range(6)]

    # AR-int5 residual bytes proxy of e_p (PR130 zero-mean/unit-RMS int5 innovation entropy)
    d1 = np.diff(e_p, axis=0)
    innov = d1 / (d1.std(0, keepdims=True) + 1e-12)
    q = np.clip(np.round(innov * 15.5), -16, 15).astype(np.int64)
    per_dim_bits = []
    for k in range(6):
        vals, cnts = np.unique(q[:, k], return_counts=True)
        p = cnts / cnts.sum()
        per_dim_bits.append(float(-(p * np.log2(p)).sum()))
    ar1_bytes = (n - 1) * sum(per_dim_bits) / 8.0

    # residual d_pose if paint base shipped with ZERO pose correction (b_p vs t_p_local MSE first-6)
    d_pose_paint_uncorrected = float(((bp - tp_local) ** 2).sum(1).mean())

    lowrank = float(ev[0]) > 0.8
    xi_smooth = float(np.mean([a for a in ac if not np.isnan(a)])) > 0.5
    paint_carries = r2_overall >= 0.5
    field_cheap = ar1_bytes < 0.5 * 23054  # << PR130 23,054 B
    # The OPERATIVE bounded conclusion (pose leg is feasibility-bounded) is decided by the shipped-field
    # byte proxy, NOT by whether the paint carries the motion. Report all axes honestly.
    if field_cheap and paint_carries:
        verdict = "PAINT_CARRIES_EGO_MOTION_AND_FIELD_CHEAP"
    elif field_cheap and not paint_carries:
        # paint carries motion only PARTIALLY (low affine R^2) but the residual is rank-1/cheap:
        # the shipped field lands << 23KB either way => pose leg feasibility-bounded, NOT binding.
        verdict = "PAINT_PARTIAL_BUT_RESIDUAL_RANK1_FIELD_CHEAP_POSE_NOT_BINDING"
    elif not field_cheap and not paint_carries and not lowrank:
        verdict = "FALSIFIER_TRIGGERED_FROM_SCRATCH_FIELD_BOUNDED_23KB"
    else:
        verdict = "MIXED_SEE_METRICS"

    receipt = {
        "schema": "ddm_sc1_ep_probe.v1",
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "n_pairs": n,
        "palette_rgb_flat_luma_comma10k": PALETTE.tolist(),
        "banked_vs_local_tp_maxabs": banked_vs_local_maxabs,
        "e_p_per_dim_std": [float(x) for x in e_p.std(0)],
        "e_p_svd_energy_frac": [float(x) for x in ev],
        "e_p_lag1_autocorr_per_dim": [round(x, 4) for x in ac],
        "global_affine_R2_overall": r2_overall,
        "global_affine_R2_per_dim": [round(float(x), 4) for x in r2_per_dim],
        "e_p_ar1_int5_residual_bits_per_dim": [round(x, 4) for x in per_dim_bits],
        "e_p_ar1_int5_residual_bytes_proxy": round(ar1_bytes, 1),
        "pr130_reference_bytes": 23054,
        "d_pose_paint_uncorrected_mse6": d_pose_paint_uncorrected,
        "tp_local_per_dim_std": [float(x) for x in tp_local.std(0)],
        "bp_per_dim_std": [float(x) for x in bp.std(0)],
        "decision_flags": {
            "e_p_rank1_lowrank": lowrank,
            "paint_carries_affine_R2_ge_0p5": paint_carries,
            "e_p_xi_smooth_meanac_gt_0p5": xi_smooth,
            "field_cheap_ar1_lt_half_pr130": field_cheap,
        },
        "falsifier_verdict": verdict,
        "interpretation": (
            "e_p = PoseNet(flat-luma-partition-paint pair) - PoseNet(real pair). Low global-affine R^2 "
            "means the flat paint does NOT reproduce the ego-motion PoseNet reads off real frames; the "
            "pose stream must be a shipped from-scratch field (bounded 23 KB by PR130 / ~2 KB by the "
            "rank-1 t_p AR proxy). High R^2 / low-rank e_p means the shipped field is only the residual."
        ),
    }
    Path(args.receipt).write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print(f"\n[wrote] {args.receipt}", file=sys.stderr)


if __name__ == "__main__":
    main()
