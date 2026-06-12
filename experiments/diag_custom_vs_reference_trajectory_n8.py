#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""FAST decisive A/B: custom vs reference backward TRAJECTORY at aggressive LR.

The n600 run diverged (d_seg 0.028 -> 0.49, d_pose 0.84 -> 36) under the PR95
curriculum's high effective LR + muon. The n8/40ep validation (muon_lr 0.03) did
NOT diverge; the n100 A/B (muon_lr 0.03) does NOT diverge. So the divergence is
LR-driven. The decisive question for the KERNEL: at an aggressive LR that DOES
force divergence, do the CUSTOM and REFERENCE mlx_gpu backends diverge
IDENTICALLY (kernel exonerated — both are the mlx_gpu forward + a correct
backward) or does ONLY the custom diverge (kernel convicted)?

This runs both arms from the SAME seed/init/permutation in one process at n8 (so
the torch-CPU exact d_seg/d_pose eval is cheap) sweeping muon_lr up until
divergence appears, then reports both arms' trajectories side by side. Exact
d_seg/d_pose on the torch-CPU AUTHORITY for both arms. $0, NO MPS.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def build(max_pairs, base_channels, seed, muon_lr, grad_clip, batch_size, custom_backward, targets_cache):
    os.environ["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1" if custom_backward else "0"
    os.environ.setdefault("MLX_METAL_GPU_ARCH", "applegpu_g15")
    from tac.capstone_vq_nerv.capstone_trainer import CapstoneTrainConfig, CapstoneTrainer
    from tac.capstone_vq_nerv.vq_nerv_bundle import CapstoneVqNervBundle, CapstoneVqNervConfig
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge
    from tac.score_aware_loop.targets import build_gt_targets, load_frozen_distortion_net

    net = load_frozen_distortion_net(device="cpu")
    cache = Path(targets_cache) / f"gt_targets_n{max_pairs}.pt"
    if cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        seg_t, pose_t = blob["seg"][:max_pairs], blob["pose"][:max_pairs]
    else:
        seg_t, pose_t, _ = build_gt_targets(net, max_pairs=max_pairs, device="cpu")
        seg_t, pose_t = seg_t[:max_pairs], pose_t[:max_pairs]
    bundle = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=max_pairs, base_channels=base_channels,
                             carrier="stored_latent", seed=seed)
    )
    bridge = TorchScorerBridge(net, seg_t, pose_t, seg_loss_form="ce_seg_loss",
                               seg_weight=100.0, pose_weight=1.0, eval_roundtrip=True)
    cfg = CapstoneTrainConfig(epochs=1, batch_size=batch_size, seed=seed,
                              scorer_backend="mlx_gpu", authority_recheck_every=0,
                              muon_lr=muon_lr, grad_clip=grad_clip, grad_clip_muon=grad_clip)
    return CapstoneTrainer(bundle, bridge, pose_t.float().cpu().numpy(), cfg)


def run_arm(trainer, epochs, eval_every, max_pairs, batch_size, seed, label):
    rng = np.random.RandomState(seed)
    traj = []
    d = trainer.exact_d_seg(use_ema=False)
    p = trainer.mean_d_pose(use_ema=False)
    traj.append({"epoch": 0, "d_seg": d, "d_pose": p})
    print(f"[{label}] init d_seg={d:.5f} d_pose={p:.4f}", flush=True)
    diverged = False
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(max_pairs)
        for start in range(0, max_pairs, batch_size):
            trainer.step(perm[start:start + batch_size], lr_scale=1.0)
        if epoch % eval_every == 0 or epoch == epochs:
            d = trainer.exact_d_seg(use_ema=False)
            p = trainer.mean_d_pose(use_ema=False)
            traj.append({"epoch": epoch, "d_seg": d, "d_pose": p})
            print(f"[{label}] ep{epoch}: d_seg={d:.5f} d_pose={p:.4f}", flush=True)
            if d > 0.40 and p > 5.0 and epoch >= eval_every * 2:
                print(f"[{label}] DIVERGED ep{epoch}", flush=True)
                diverged = True
                break
    return traj, diverged


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--base-channels", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--muon-lr", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=50.0)
    ap.add_argument("--targets-cache", default="experiments/results/capstone_gt_targets_cache")
    ap.add_argument("--out-json", default="experiments/results/custom_vs_reference_trajectory_n8.json")
    args = ap.parse_args()
    torch.set_num_threads(min(6, torch.get_num_threads()))

    print(f"=== CUSTOM backward (muon_lr={args.muon_lr}) ===", flush=True)
    tc = build(args.max_pairs, args.base_channels, args.seed, args.muon_lr,
               args.grad_clip, args.batch_size, True, args.targets_cache)
    traj_c, div_c = run_arm(tc, args.epochs, args.eval_every, args.max_pairs,
                            args.batch_size, args.seed, "custom")
    del tc

    print(f"\n=== REFERENCE backward (muon_lr={args.muon_lr}) ===", flush=True)
    tr = build(args.max_pairs, args.base_channels, args.seed, args.muon_lr,
               args.grad_clip, args.batch_size, False, args.targets_cache)
    traj_r, div_r = run_arm(tr, args.epochs, args.eval_every, args.max_pairs,
                            args.batch_size, args.seed, "reference")
    del tr

    print("\n=== SIDE-BY-SIDE (exact torch-CPU authority) ===")
    print(f"{'epoch':>6} | {'custom_dseg':>12} {'custom_dpose':>12} | {'ref_dseg':>12} {'ref_dpose':>12}")
    by_c = {r["epoch"]: r for r in traj_c}
    by_r = {r["epoch"]: r for r in traj_r}
    for ep in sorted(set(by_c) | set(by_r)):
        c = by_c.get(ep)
        r = by_r.get(ep)
        cs = f"{c['d_seg']:>12.5f} {c['d_pose']:>12.4f}" if c else f"{'-':>12} {'-':>12}"
        rs = f"{r['d_seg']:>12.5f} {r['d_pose']:>12.4f}" if r else f"{'-':>12} {'-':>12}"
        print(f"{ep:>6} | {cs} | {rs}")

    if div_c and div_r:
        verdict = "KERNEL_EXONERATED_both_custom_and_reference_diverge_at_this_LR"
    elif div_c and not div_r:
        verdict = "KERNEL_CONVICTED_custom_only_diverges"
    elif not div_c and not div_r:
        verdict = "NO_DIVERGENCE_increase_muon_lr"
    else:
        verdict = "REFERENCE_ONLY_DIVERGES_unexpected"
    print(f"\nVERDICT: {verdict}")
    print(f"diverged: custom={div_c} reference={div_r}")
    out = {"config": vars(args), "custom": traj_c, "reference": traj_r,
           "diverged": {"custom": div_c, "reference": div_r}, "verdict": verdict}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
