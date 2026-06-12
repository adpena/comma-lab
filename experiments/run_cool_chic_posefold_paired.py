# SPDX-License-Identifier: MIT
"""PAIRED pose-fold A/B — isolate the stored-pose FiLM effect on d_pose.

Lane ``lane_cool_chic_score_aware_basis_20260611``. The full B1-CLOSE configs
(30-45 ep, 96x128 render, 192x256 scorer) exceed a single synchronous window, so
this runs a MATCHED-epoch paired A/B (same seed, same config, same byte-close)
that isolates the ONE variable under test: the Quantizr stored-pose FiLM.

  A = FiLM OFF (the B1-CLOSE carrier): d_pose reconstructed from the coarse grid.
  B = FiLM ON  (pose-fold): the GT 6-pose is STORED + range-coded (~1 KB) and
      FiLM-conditions frame1; d_pose should COLLAPSE.

Both are byte-closed (REAL range-coded archive) and RE-SCORED through the frozen
torch-CPU DistortionNet on the inflated render. Reports d_pose / d_seg / bytes /
advisory-S for both arms and the delta. ``[macOS-CPU advisory]`` / non-promotable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

# repo root on path so the sibling runner (the canonical re-score) imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tac.residual_basis.cool_chic_byte_close import (
    byte_close,
    carrier_to_state,
    inflate_numpy,
    render_pair_numpy,  # noqa: F401  (used via score path import)
)
from tac.residual_basis.cool_chic_carrier import CoolChicGridSpec, CoolChicPairCarrier
from tac.score_aware_loop.targets import load_frozen_distortion_net
from tac.score_aware_loop.trainer import ScoreAwareLoopConfig, ScoreAwareTrainer

# Reuse the EXACT byte-closed re-score (the decisive measure) from the runner.
from experiments.run_cool_chic_byte_closed_advisory_s import score_byte_closed

CACHE = "experiments/results/capstone_gt_targets_cache/gt_targets_n48.pt"
ARCHIVE_DENOM = 37_545_489.0


def fit_and_score(net, seg, pose, *, pose_film, cfg_tuple, scorer_hw, epochs,
                  grid_step, delta_step, pose_step, seed):
    bh, bw, ng, cpg, sh, oh, ow = cfg_tuple
    torch.manual_seed(seed)
    n_pairs = int(seg.shape[0])
    spec = CoolChicGridSpec(base_h=bh, base_w=bw, n_grids=ng, channels_per_grid=cpg)
    carrier = CoolChicPairCarrier(
        n_pairs=n_pairs, spec=spec, synth_hidden=sh, out_hw=(oh, ow),
        pose_film_enabled=pose_film,
    )
    if pose_film:
        carrier.set_stored_pose(pose.float())
    cfg = ScoreAwareLoopConfig(
        epochs=epochs, batch_size=n_pairs, scorer_hw=scorer_hw,
        pose_enabled=True, eval_every=max(epochs, 1),
        seg_loss_form="ce_seg_loss", decoder_lr=3e-3, latent_lr_mult=10.0,
        ema_decay=0.99, seed=seed,
    )
    tr = ScoreAwareTrainer(carrier, net, seg, pose, cfg)
    t0 = time.time()
    res = tr.train()
    wall = time.time() - t0
    tr._ema_for_eval.apply_to(tr.model)  # byte-close the EMA shadow (CLAUDE.md "EMA")
    state = carrier_to_state(carrier)
    blob = byte_close(state, grid_step=grid_step, delta_step=delta_step, pose_step=pose_step)
    real_bytes = len(blob)
    state2 = inflate_numpy(blob)
    d_seg_bc, d_pose_bc = score_byte_closed(state2, net, seg, pose, scorer_hw=scorer_hw)
    pose_term = math.sqrt(10.0 * d_pose_bc) if d_pose_bc is not None else 0.0
    rate_term = 25.0 * real_bytes / ARCHIVE_DENOM
    s = 100.0 * d_seg_bc + pose_term + rate_term
    return {
        "pose_film": pose_film,
        "real_bytes": real_bytes,
        "charged_breakdown": carrier.charged_bytes(),
        "d_seg_byteclosed": d_seg_bc,
        "d_pose_byteclosed": d_pose_bc,
        "advisory_s": s,
        "terms": {"seg": 100.0 * d_seg_bc, "pose": pose_term, "rate": rate_term},
        "d_seg_best_ema_train": res["d_seg_best_ema"],
        "train_wall_seconds": round(wall, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--scorer-h", type=int, default=192)
    ap.add_argument("--scorer-w", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--grid-step", type=float, default=0.05)
    ap.add_argument("--delta-step", type=float, default=0.05)
    ap.add_argument("--pose-step", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=0)
    # one arm per invocation so each fits a synchronous window; appends to --out.
    ap.add_argument("--arm", choices=["off", "on"], required=True)
    ap.add_argument("--config", type=int, default=1)
    ap.add_argument(
        "--out",
        default="experiments/results/lane_cool_chic_score_aware_basis_20260611/posefold_paired.json",
    )
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    configs = [
        (24, 32, 3, 2, 16, 48, 64),
        (24, 32, 4, 2, 16, 64, 96),
        (48, 64, 4, 3, 24, 96, 128),
    ]
    cfg_tuple = configs[args.config]
    d = torch.load(CACHE, map_location="cpu", weights_only=False)
    seg = d["seg"][: args.n_pairs]
    pose = d["pose"][: args.n_pairs].float()
    net = load_frozen_distortion_net(device="cpu")

    row = fit_and_score(
        net, seg, pose, pose_film=(args.arm == "on"), cfg_tuple=cfg_tuple,
        scorer_hw=(args.scorer_h, args.scorer_w), epochs=args.epochs,
        grid_step=args.grid_step, delta_step=args.delta_step,
        pose_step=args.pose_step, seed=args.seed,
    )
    row["config"] = {
        "tuple": list(cfg_tuple), "epochs": args.epochs, "seed": args.seed,
        "scorer_hw": [args.scorer_h, args.scorer_w], "n_pairs": args.n_pairs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if Path(args.out).exists():
        try:
            rows = json.load(open(args.out)).get("rows", [])
        except Exception:
            rows = []
    rows.append(row)
    with open(args.out, "w") as f:
        json.dump({
            "lane": "lane_cool_chic_score_aware_basis_20260611",
            "axis_tag": "[macOS-CPU advisory]", "promotable": False,
            "experiment": "posefold_paired_AB", "rows": rows,
        }, f, indent=2)
    print(
        f"arm={args.arm} cfg{args.config} ep{args.epochs}: "
        f"REAL {row['real_bytes']}B (pose_bytes "
        f"{row['charged_breakdown'].get('stored_pose_bytes', 0):.0f}) | "
        f"d_seg={row['d_seg_byteclosed']:.4f} d_pose={row['d_pose_byteclosed']:.4e} "
        f"S={row['advisory_s']:.4f} [seg {row['terms']['seg']:.3f} + "
        f"pose {row['terms']['pose']:.3f} + rate {row['terms']['rate']:.2e}] "
        f"{row['train_wall_seconds']}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
