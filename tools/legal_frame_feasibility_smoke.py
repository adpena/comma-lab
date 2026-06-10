#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""$0 macOS-CPU smoke for the legal-frame FEASIBILITY solve (task #73).

THE QUESTION (pre-registration `legal_frame_feasibility_dykstra_20260610T175421Z.md`): does Dykstra
alternating projection onto (SegNet argmax CELL) ∩ (PoseNet TUBE) ∩ (cheap subspace) hold BOTH
d_seg≈0 AND pose-in-tube SIMULTANEOUSLY at LOW byte — where the lever-B GENERATORS could not
(palette pose-blind 12.14; INR pose-ceiling 0.0036)?

THE SETUP (two bases per pair, both honest):
  * BASE-GT : base frame1 = the GT (resized) frame1. delta=0 is IN-cell + IN-tube at ZERO byte.
    The load-bearing move is projecting delta onto C (cheap) — does the cheap projection of a
    perturbation stay feasible? (Here delta starts 0, so this measures whether the cheap projector
    KEEPS the GT feasible — the geometric reference.)
  * BASE-COMP : base frame1 = the FRONTIER receiver comp frame1 (ALREADY cheaply encoded by the
    0.191 archive). The comp frame is OFF the GT cell (the frontier residual d_seg≈0.067). The
    feasibility question: can a CHEAP delta on top of the comp frame pull it INTO the GT cell ∩
    tube? If yes -> a cheap feasible carrier exists beyond the frontier. If the cheap projection
    cannot close the residual -> cell∩tube∩cheap is EMPTY at low byte (the sharp geometric finding).

The scorers consume the RESIZED (384,512) input directly; we run the solve on that exact grid
(SegNet uses frame1 only; PoseNet uses the (frame0, frame1) pair). GT decode via
frame_utils.yuv420_to_rgb. EXACT frozen CPU-torch SegNet/PoseNet. NEVER MPS. [macOS-CPU advisory].
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _setup_paths(root: Path) -> None:
    harness = root / "experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis"
    for p in (root, root / "src", root / "upstream", harness):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=str, default="0,1,2,3", help="comma pair indices")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=5e-4)
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--sparse-keep-frac", type=float, default=0.10)
    ap.add_argument("--quant-step", type=float, default=1.0)
    ap.add_argument("--max-outer", type=int, default=12)
    ap.add_argument("--base", choices=["gt", "comp", "both"], default="both")
    ap.add_argument("--output-json", type=str, default="")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    _setup_paths(root)

    import render_and_score_lib as L  # type: ignore
    import torch
    import torch.nn.functional as F
    from modules import PoseNet, SegNet, posenet_sd_path, segnet_sd_path  # type: ignore
    from safetensors.torch import load_file

    from tac.boundary_math.dykstra_legal_frame import (
        SEG_H,
        SEG_W,
        FeasibilityConfig,
        FrozenScorerProjector,
        solve_legal_frame_feasibility,
    )
    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )

    torch.manual_seed(1234)
    np.random.seed(1234)

    pair_indices = [int(x) for x in args.pairs.split(",") if x.strip() != ""]

    # ---- load frozen scorers (CPU) ----
    seg = SegNet().eval()
    seg.load_state_dict(load_file(segnet_sd_path, device="cpu"))
    pose = PoseNet().eval()
    pose.load_state_dict(load_file(posenet_sd_path, device="cpu"))
    for net in (seg, pose):
        for p in net.parameters():
            p.requires_grad_(False)

    # ---- GT + frontier comp frames (camera-res) -> resized scorer grid ----
    gt = L.decode_gt_pairs(pair_indices)
    renderer = L.FrontierRenderer()
    comp = renderer.render_baseline_pairs(pair_indices)  # dict pi -> (2,3,CAMERA_H,CAMERA_W)

    def resize_chw(cam_chw: torch.Tensor) -> torch.Tensor:
        return F.interpolate(cam_chw.unsqueeze(0), size=(SEG_H, SEG_W), mode="bilinear")[0]

    cfg = FeasibilityConfig(
        gamma=args.gamma,
        tau=args.tau,
        rank=args.rank,
        sparse_keep_frac=args.sparse_keep_frac,
        quant_step=args.quant_step,
        max_outer=args.max_outer,
        project_cheap=True,
    )

    rows: list[dict] = []
    bases = ["gt", "comp"] if args.base == "both" else [args.base]

    tok = patch_upstream_yuv6_globally()
    try:
        for pi in pair_indices:
            g0_cam = gt[pi][0].float().permute(2, 0, 1)  # (3,H,W) uint8->float
            g1_cam = gt[pi][1].float().permute(2, 0, 1)
            g0 = resize_chw(g0_cam)
            g1_gt = resize_chw(g1_cam)
            # TARGET = the GT scorer outputs (the cell + tube are defined by the GT).
            tgt_proj = FrozenScorerProjector(seg, pose, g1_gt, g0)
            zero = np.zeros((3, SEG_H, SEG_W), dtype=np.float64)
            target_argmax = tgt_proj.seg_argmax(zero)
            target_pose6 = tgt_proj.pose6(zero)

            for base in bases:
                # APPLES-TO-APPLES PAIRING (corrected): the contest d_pose pairs comp-frame0 with
                # comp-frame1 (NOT GT-frame0 with comp-frame1 — that manufactures phantom ~10 pose
                # per the GT-decode bug class). base-gt pairs (g0, g1_gt); base-comp pairs the REAL
                # frontier comp pair (c0, c1). The TARGET is always the GT pair's scorer outputs.
                if base == "gt":
                    base_f0 = g0
                    base_f1 = g1_gt
                else:
                    base_f0 = resize_chw(comp[pi][0].float())  # frontier comp frame0 (the real pair)
                    base_f1 = resize_chw(comp[pi][1].float())  # frontier comp frame1 (cheap)
                proj = FrozenScorerProjector(seg, pose, base_f1, base_f0)
                # Baseline (delta=0) operating point of THIS base vs the GT target.
                base_d_seg = proj.d_seg(zero, target_argmax)
                base_d_pose = proj.d_pose(zero, target_pose6)

                t0 = time.time()
                res = solve_legal_frame_feasibility(
                    proj, target_argmax, target_pose6, cfg, d_seg_hold=0.01, d_pose_hold=args.tau
                )
                dt = time.time() - t0
                row = {
                    "pair": pi,
                    "base": base,
                    "base_d_seg": base_d_seg,
                    "base_d_pose": base_d_pose,
                    "final_d_seg": res.d_seg,
                    "final_d_pose": res.d_pose,
                    "delta_bytes": res.delta_bytes,
                    "held_both_at_low_byte": res.held_both_at_low_byte,
                    "converged": res.converged,
                    "outer_iters": res.outer_iters,
                    "d_seg_trace": res.d_seg_trace,
                    "d_pose_trace": res.d_pose_trace,
                    "seconds": round(dt, 2),
                    "evidence_grade": "[macOS-CPU advisory]",
                }
                rows.append(row)
                print(
                    f"pair {pi:>3} base={base:<4} | base(d_seg={base_d_seg:.5f} "
                    f"d_pose={base_d_pose:.3e}) -> final(d_seg={res.d_seg:.5f} "
                    f"d_pose={res.d_pose:.3e}) | delta_bytes={res.delta_bytes} "
                    f"held_both={res.held_both_at_low_byte} iters={res.outer_iters} {dt:.1f}s",
                    flush=True,
                )
    finally:
        unpatch_upstream_yuv6(tok)

    # ---- aggregate verdict ----
    comp_rows = [r for r in rows if r["base"] == "comp"]
    held_comp = [r for r in comp_rows if r["held_both_at_low_byte"]]
    summary = {
        "n_pairs": len(pair_indices),
        "config": {
            "gamma": args.gamma,
            "tau": args.tau,
            "rank": args.rank,
            "sparse_keep_frac": args.sparse_keep_frac,
            "quant_step": args.quant_step,
            "max_outer": args.max_outer,
        },
        "comp_base_held_both_count": len(held_comp),
        "comp_base_total": len(comp_rows),
        "rows": rows,
        "evidence_grade": "[macOS-CPU advisory]",
        "promotable": False,
        "authority_tier": "exact_cpu_advisory",
    }
    if comp_rows:
        mean_base_seg = float(np.mean([r["base_d_seg"] for r in comp_rows]))
        mean_final_seg = float(np.mean([r["final_d_seg"] for r in comp_rows]))
        mean_final_pose = float(np.mean([r["final_d_pose"] for r in comp_rows]))
        mean_bytes = float(np.mean([r["delta_bytes"] for r in comp_rows]))
        summary["comp_mean_base_d_seg"] = mean_base_seg
        summary["comp_mean_final_d_seg"] = mean_final_seg
        summary["comp_mean_final_d_pose"] = mean_final_pose
        summary["comp_mean_delta_bytes"] = mean_bytes
        print(
            f"\nVERDICT comp-base: mean d_seg {mean_base_seg:.5f} -> {mean_final_seg:.5f} | "
            f"mean final d_pose {mean_final_pose:.3e} | mean delta_bytes {mean_bytes:.0f} | "
            f"held_both {len(held_comp)}/{len(comp_rows)}",
            flush=True,
        )

    if args.output_json:
        outp = Path(args.output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, indent=2))
        print(f"wrote {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
