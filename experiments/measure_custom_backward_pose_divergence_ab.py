#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""3-arm A/B: is the n600 POSE divergence the custom Metal backward, or the
mlx_gpu FORWARD fidelity (independent of the backward kernel)?

The custom grouped/depthwise-conv backward (TAC_MLX_CUSTOM_GROUPED_BACKWARD=1)
was validated at n8 (d_seg AND d_pose both descended) but DIVERGED at n600 on
the POSE axis (d_pose 0.835 -> 6.94 -> 36.46). The per-layer + end-to-end
gradient parity of the custom backward vs the trusted Python-loop reference is
cosine 1.000000 (proven). So the question this A/B answers decisively:

  * Arm A:  scorer_backend=torch_cpu_bridge          (the AUTHORITY gradient)
  * Arm B:  scorer_backend=mlx_gpu, CUSTOM backward   (TAC_MLX_CUSTOM_GROUPED_BACKWARD=1)
  * Arm B': scorer_backend=mlx_gpu, REFERENCE backward (TAC_MLX_CUSTOM_GROUPED_BACKWARD=0)

All three from the SAME seed (identical init), SAME permutations, SAME recipe.
Exact d_seg + d_pose measured on the torch-CPU AUTHORITY for ALL arms.

Decisive logic (NO-FAKE):
  * If B and B' DIVERGE IDENTICALLY on d_pose (and A does not), the divergence
    is the mlx_gpu FORWARD fidelity (documented ~2.76e-4 pose drift in
    MLXGpuScorerBridge), NOT the custom backward -> the kernel is EXONERATED.
  * If ONLY B (custom) diverges while B' (reference) stays bounded, the custom
    backward has a real n-scale bug -> CONVICTED, debug the kernel.

Authority: exact d_seg/d_pose are torch-CPU (the only authority). $0, local, NO
MPS. The mlx_gpu GRADIENT is research-signal; the REPORTED d_seg/d_pose is the
torch-CPU authority for every arm.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def _build(max_pairs, base_channels, backend, carrier, targets_cache, seed,
           muon_lr, grad_clip, batch_size, custom_backward):
    # Toggle the custom backward BEFORE the bridge builds the MLX adapter so the
    # adapter routes strided-grouped convs to the custom kernel (or the
    # reference path) per this arm.
    os.environ["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1" if custom_backward else "0"
    if backend == "mlx_gpu":
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
    rng = np.random.RandomState(seed)
    traj = []
    d0 = trainer.exact_d_seg(use_ema=False)
    p0 = trainer.mean_d_pose(use_ema=False)
    print(f"[{label}] init exact_d_seg={d0:.6f} d_pose={p0:.6f}", flush=True)
    traj.append({"epoch": 0, "exact_d_seg": d0, "mean_d_pose": p0})
    t0 = time.time()
    diverged = False
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(max_pairs)
        for start in range(0, max_pairs, batch_size):
            idx = perm[start: start + batch_size]
            trainer.step(idx, lr_scale=1.0)
        if epoch % eval_every == 0 or epoch == epochs:
            d = trainer.exact_d_seg(use_ema=False)
            p = trainer.mean_d_pose(use_ema=False)
            traj.append({"epoch": epoch, "exact_d_seg": d, "mean_d_pose": p})
            print(f"[{label}] epoch {epoch}: exact_d_seg={d:.6f} d_pose={p:.6f} "
                  f"({(time.time()-t0)/epoch:.1f}s/ep)", flush=True)
            # Early-stop a clearly-diverged arm (d_seg back near random AND
            # d_pose exploding) to save wall-clock — record the divergence.
            if d > 0.40 and p > 5.0 and epoch >= eval_every * 2:
                print(f"[{label}] DIVERGED at epoch {epoch} (d_seg={d:.3f} d_pose={p:.3f}) — stopping arm", flush=True)
                diverged = True
                break
    return traj, diverged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, default=96)
    ap.add_argument("--base-channels", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--carrier", default="stored_latent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--muon-lr", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=50.0)
    ap.add_argument("--targets-cache", default="experiments/results/capstone_gt_targets_cache")
    ap.add_argument("--out-json", default="experiments/results/custom_backward_pose_divergence_ab.json")
    ap.add_argument("--skip-torch", action="store_true", help="skip the slow torch-CPU authority arm")
    args = ap.parse_args()
    torch.set_num_threads(min(6, torch.get_num_threads()))

    arms = {}
    diverged = {}

    if not args.skip_torch:
        print("=== ARM A: torch_cpu_bridge (authority gradient) ===", flush=True)
        a = _build(args.max_pairs, args.base_channels, "torch_cpu_bridge", args.carrier,
                   args.targets_cache, args.seed, args.muon_lr, args.grad_clip,
                   args.batch_size, custom_backward=False)
        arms["torch_cpu"], diverged["torch_cpu"] = _run_arm(
            a, args.epochs, args.eval_every, args.max_pairs, args.batch_size, args.seed, "torch_cpu")
        del a

    print("\n=== ARM B: mlx_gpu + CUSTOM backward (TAC_MLX_CUSTOM_GROUPED_BACKWARD=1) ===", flush=True)
    b = _build(args.max_pairs, args.base_channels, "mlx_gpu", args.carrier,
               args.targets_cache, args.seed, args.muon_lr, args.grad_clip,
               args.batch_size, custom_backward=True)
    arms["mlx_custom"], diverged["mlx_custom"] = _run_arm(
        b, args.epochs, args.eval_every, args.max_pairs, args.batch_size, args.seed, "mlx_custom")
    del b

    print("\n=== ARM B': mlx_gpu + REFERENCE backward (TAC_MLX_CUSTOM_GROUPED_BACKWARD=0) ===", flush=True)
    c = _build(args.max_pairs, args.base_channels, "mlx_gpu", args.carrier,
               args.targets_cache, args.seed, args.muon_lr, args.grad_clip,
               args.batch_size, custom_backward=False)
    arms["mlx_reference"], diverged["mlx_reference"] = _run_arm(
        c, args.epochs, args.eval_every, args.max_pairs, args.batch_size, args.seed, "mlx_reference")
    del c

    print("\n=== D_POSE TRAJECTORIES (exact torch-CPU authority on all arms) ===")
    all_eps = sorted({r["epoch"] for arm in arms.values() for r in arm})
    by = {k: {r["epoch"]: r for r in arm} for k, arm in arms.items()}
    hdr = f"{'epoch':>6}"
    for k in arms:
        hdr += f" | {k+'_dseg':>16} {k+'_dpose':>16}"
    print(hdr)
    for ep in all_eps:
        line = f"{ep:>6}"
        for k in arms:
            r = by[k].get(ep)
            if r:
                line += f" | {r['exact_d_seg']:>16.6f} {r['mean_d_pose']:>16.6f}"
            else:
                line += f" | {'-':>16} {'-':>16}"
        print(line)

    verdict = "INDETERMINATE"
    if "mlx_custom" in diverged and "mlx_reference" in diverged:
        if diverged["mlx_custom"] and diverged["mlx_reference"]:
            verdict = "KERNEL_EXONERATED_mlx_forward_fidelity_divergence_both_arms"
        elif diverged["mlx_custom"] and not diverged["mlx_reference"]:
            verdict = "KERNEL_CONVICTED_custom_only_diverges"
        elif not diverged["mlx_custom"] and not diverged["mlx_reference"]:
            verdict = "NO_DIVERGENCE_AT_THIS_N_increase_n_or_epochs"
        else:
            verdict = "REFERENCE_DIVERGES_custom_does_not_unexpected"
    print(f"\nVERDICT: {verdict}")
    print(f"diverged flags: {diverged}")

    out = {"config": vars(args), "arms": arms, "diverged": diverged, "verdict": verdict}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
