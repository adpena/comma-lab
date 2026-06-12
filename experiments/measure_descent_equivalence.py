#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Descent-equivalence A/B: does MLX-GPU-gradient training reach the SAME
exact-(torch-CPU)-d_seg trajectory as torch-CPU-gradient training?

This is the DECISIVE LOCAL-MLX-DREAM measurement for task 2. The wire-in memo
proved the MLX-GPU per-step pixel GRADIENT matches torch-CPU's to cosine 0.99986
on a single batch. But a training SIGNAL needs a good DIRECTION over many steps,
not bit-exact argmax — the open question is whether accumulating that ~0.9999
gradient over a descent actually reaches the same basin as the exact gradient,
or whether the small per-step drift compounds and diverges.

Method (apples-to-apples, NO-FAKE):
  * Build TWO trainers from the SAME seed (identical init).
  * Trainer A uses scorer_backend=torch_cpu_bridge (the authority gradient).
  * Trainer B uses scorer_backend=mlx_gpu (the fast gradient).
  * Run the SAME number of steps with the SAME pair permutations on both.
  * At each checkpoint epoch, measure the EXACT d_seg + d_pose on BOTH trainers
    using the SAME torch-CPU authority bridge (trainer.exact_d_seg uses the
    torch-CPU bridge regardless of gradient backend — so the metric is the
    authority for both arms; only the GRADIENT that drove the steps differs).
  * Report the per-epoch exact-d_seg of each arm + the absolute gap.

If the two exact-d_seg trajectories track (gap << the descent), the MLX-GPU
gradient is descent-equivalent -> the backward bottleneck can be moved to the
GPU. If they diverge, the exact scorer must stay in the gradient loop.

Authority: the exact d_seg/d_pose are torch-CPU (the only authority). The
mlx_gpu arm's GRADIENT is research-signal; its REPORTED d_seg is torch-CPU.
$0, local, no MPS.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def _build(max_pairs, base_channels, backend, carrier, targets_cache, seed,
           muon_lr, grad_clip, batch_size):
    if backend == "mlx_gpu":
        import os
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
                             carrier=carrier, seed=seed)
    )
    bridge = TorchScorerBridge(net, seg_t, pose_t, seg_loss_form="ce_seg_loss",
                               seg_weight=100.0, pose_weight=1.0, eval_roundtrip=True)
    pose_store = pose_t.float().cpu().numpy()
    cfg = CapstoneTrainConfig(
        epochs=1, batch_size=batch_size, seed=seed, scorer_backend=backend,
        authority_recheck_every=0, muon_lr=muon_lr, grad_clip=grad_clip,
        grad_clip_muon=grad_clip,
    )
    return CapstoneTrainer(bundle, bridge, pose_store, cfg)


def _run_arm(trainer, epochs, eval_every, max_pairs, batch_size, seed, label):
    """Run `epochs` of muon-throughout descent; eval exact d_seg every eval_every."""
    rng = np.random.RandomState(seed)
    traj = []
    d0 = trainer.exact_d_seg(use_ema=False)
    p0 = trainer.mean_d_pose(use_ema=False)
    print(f"[{label}] init exact_d_seg={d0:.6f} d_pose={p0:.6f}", flush=True)
    traj.append({"epoch": 0, "exact_d_seg": d0, "mean_d_pose": p0})
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(max_pairs)
        for start in range(0, max_pairs, batch_size):
            idx = perm[start : start + batch_size]
            # step() updates VQ-EMA + weight-EMA internally; do NOT double-update.
            trainer.step(idx, lr_scale=1.0)
        if epoch % eval_every == 0 or epoch == epochs:
            d = trainer.exact_d_seg(use_ema=False)
            p = trainer.mean_d_pose(use_ema=False)
            traj.append({"epoch": epoch, "exact_d_seg": d, "mean_d_pose": p})
            print(f"[{label}] epoch {epoch}: exact_d_seg={d:.6f} d_pose={p:.6f} "
                  f"({(time.time()-t0)/epoch:.1f}s/ep)", flush=True)
    return traj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument("--base-channels", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--carrier", default="stored_latent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--muon-lr", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=50.0)
    ap.add_argument("--targets-cache", default="experiments/results/capstone_gt_targets_cache")
    ap.add_argument("--out-json", default="experiments/results/descent_equivalence_ab.json")
    args = ap.parse_args()
    torch.set_num_threads(min(6, torch.get_num_threads()))

    print("=== ARM A: torch_cpu_bridge (authority gradient) ===", flush=True)
    a = _build(args.max_pairs, args.base_channels, "torch_cpu_bridge", args.carrier,
               args.targets_cache, args.seed, args.muon_lr, args.grad_clip, args.batch_size)
    traj_a = _run_arm(a, args.epochs, args.eval_every, args.max_pairs,
                      args.batch_size, args.seed, "torch_cpu")
    del a

    print("\n=== ARM B: mlx_gpu (fast gradient) ===", flush=True)
    b = _build(args.max_pairs, args.base_channels, "mlx_gpu", args.carrier,
               args.targets_cache, args.seed, args.muon_lr, args.grad_clip, args.batch_size)
    traj_b = _run_arm(b, args.epochs, args.eval_every, args.max_pairs,
                      args.batch_size, args.seed, "mlx_gpu")

    # Adjudicate through the CANONICAL BOTH-TERMS acceptance gate. This is the
    # structural fix for the n600 lesson: the old comparison block scored d_seg
    # ONLY and never gated d_pose (the axis that diverged). The gate forces BOTH
    # terms and REFUSES a d_seg-only pass. Arm A (torch-CPU authority gradient) is
    # the baseline; arm B (the fast/candidate gradient) is the candidate.
    from tac.mlx_pr95_port.speedup_acceptance_gate import evaluate_descent_equivalence

    verdict = evaluate_descent_equivalence(
        traj_a, traj_b, n_pairs=args.max_pairs
    )
    print("\n=== BOTH-TERMS ACCEPTANCE GATE (exact torch-CPU d_seg AND d_pose) ===")
    print(f"{'epoch':>6} {'base_seg':>10} {'cand_seg':>10} {'base_pose':>11} {'cand_pose':>11}")
    by_ep_a = {r["epoch"]: r for r in traj_a}
    by_ep_b = {r["epoch"]: r for r in traj_b}
    for ep in sorted(set(by_ep_a) & set(by_ep_b)):
        a, b = by_ep_a[ep], by_ep_b[ep]
        print(f"{ep:>6} {a['exact_d_seg']:>10.6f} {b['exact_d_seg']:>10.6f} "
              f"{a['mean_d_pose']:>11.6f} {b['mean_d_pose']:>11.6f}")
    print(f"\nseg:  {verdict.seg.reason}")
    print(f"pose: {verdict.pose.reason}")
    print(f"\nVERDICT: {'PASS' if verdict.passed else 'REJECT'} "
          f"(n={verdict.n_pairs}, epochs_compared={verdict.epochs_compared})")
    for r in verdict.reasons:
        print(f"  - {r}")

    out = {"config": vars(args), "arm_torch_cpu": traj_a, "arm_mlx_gpu": traj_b,
           "gate_passed": verdict.passed,
           "gate_generalization_warning": verdict.generalization_warning,
           "gate_reasons": list(verdict.reasons),
           "seg_verdict": {
               "tracks": verdict.seg.tracks_within_tol,
               "diverged": verdict.seg.diverged,
               "final_abs_gap": verdict.seg.final_abs_gap,
               "reason": verdict.seg.reason,
           },
           "pose_verdict": {
               "tracks": verdict.pose.tracks_within_tol,
               "diverged": verdict.pose.diverged,
               "diverged_at_epoch": verdict.pose.diverged_at_epoch,
               "final_abs_gap": verdict.pose.final_abs_gap,
               "reason": verdict.pose.reason,
           }}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out_json}")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
