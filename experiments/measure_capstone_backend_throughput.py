#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Measure per-step training throughput (fwd+bwd) for the capstone trainer on
both scorer backends (torch_cpu_bridge vs mlx_gpu) at a chosen pair count.

This is the DECISIVE throughput micro-benchmark for the LOCAL-MLX-DREAM
feasibility question: it times PURE training steps (the optimizer step granular
fwd+bwd through the chosen scorer backend) with NO eval, so the number is the
clean steps/sec the n600 wall-clock projects from. It reuses the real trainer +
real frozen torch-CPU scorer + real GT targets + real bundle (NO synthetic
fixtures — the slow_conv2d amortization degenerates on toy inputs, per the
2026-06-11 throughput profile).

Authority: [macOS-CPU advisory] / [macOS-MLX research-signal]. $0, local, no MPS.
NOT a pointer move. The torch-CPU scorer is the trusted authority per CLAUDE.md
"local CPU + MLX GPU good".

Usage:
    .venv/bin/python experiments/measure_capstone_backend_throughput.py \
        --max-pairs 48 --base-channels 20 --batch-size 8 --warmup 2 --steps 6 \
        --backends torch_cpu_bridge mlx_gpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def _build_trainer(max_pairs: int, base_channels: int, batch_size: int, backend: str,
                   carrier: str, targets_cache: str, seed: int):
    """Build a CapstoneTrainer with the real frozen scorer + real GT targets.

    Mirrors the canonical construction in ``run_capstone_campaign.py:main`` so
    the timing is on the EXACT trainer the n600 launch uses.
    """
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
        seg_t, pose_t, n = blob["seg"], blob["pose"], int(blob["n"])
    else:
        seg_t, pose_t, n = build_gt_targets(net, max_pairs=max_pairs, device="cpu")
    assert n >= max_pairs, f"cache has n={n} < max_pairs={max_pairs}"
    seg_t, pose_t, n = seg_t[:max_pairs], pose_t[:max_pairs], max_pairs

    bundle = CapstoneVqNervBundle(
        CapstoneVqNervConfig(
            num_pairs=n, base_channels=base_channels, carrier=carrier, seed=seed,
        )
    )
    bridge = TorchScorerBridge(
        net, seg_t, pose_t, seg_loss_form="ce_seg_loss",
        seg_weight=100.0, pose_weight=1.0, eval_roundtrip=True,
    )
    pose_store = pose_t.float().cpu().numpy()
    cfg = CapstoneTrainConfig(
        epochs=1, batch_size=batch_size, seed=seed,
        scorer_backend=backend, authority_recheck_every=0,
        muon_lr=0.03, grad_clip=50.0, grad_clip_muon=50.0,
    )
    trainer = CapstoneTrainer(bundle, bridge, pose_store, cfg)
    return trainer


def _time_steps(trainer, max_pairs: int, batch_size: int, warmup: int, steps: int):
    """Time `steps` real training steps (fwd+bwd), after `warmup` untimed steps."""
    rng = np.random.RandomState(0)
    # warmup (graph build, first-touch allocation, optimizer state init)
    for _ in range(warmup):
        perm = rng.permutation(max_pairs)
        for start in range(0, max_pairs, batch_size):
            idx = perm[start : start + batch_size]
            trainer.step(idx, lr_scale=1.0)

    t0 = time.time()
    n_steps = 0
    for _ in range(steps):
        perm = rng.permutation(max_pairs)
        for start in range(0, max_pairs, batch_size):
            idx = perm[start : start + batch_size]
            trainer.step(idx, lr_scale=1.0)
            n_steps += 1
    dt = time.time() - t0
    steps_per_pass = (max_pairs + batch_size - 1) // batch_size
    return {
        "wall_s": dt,
        "n_steps": n_steps,
        "n_epochs_equiv": steps,  # `steps` full passes over the set
        "s_per_step": dt / max(n_steps, 1),
        "s_per_epoch": dt / max(steps, 1),  # one epoch = one full pass over max_pairs
        "steps_per_pass": steps_per_pass,
        "pairs_per_s": (n_steps * batch_size) / dt if dt > 0 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, default=48)
    ap.add_argument("--base-channels", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4, help="timed full passes over the set")
    ap.add_argument("--carrier", default="stored_latent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets-cache", default="experiments/results/capstone_gt_targets_cache")
    ap.add_argument("--backends", nargs="+", default=["torch_cpu_bridge", "mlx_gpu"])
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    torch.set_num_threads(min(6, torch.get_num_threads()))
    results = {}
    for backend in args.backends:
        print(f"\n=== backend={backend} n={args.max_pairs} bs={args.batch_size} ===", flush=True)
        trainer = _build_trainer(
            args.max_pairs, args.base_channels, args.batch_size, backend,
            args.carrier, args.targets_cache, args.seed,
        )
        m = _time_steps(trainer, args.max_pairs, args.batch_size, args.warmup, args.steps)
        results[backend] = m
        print(
            f"  s/step={m['s_per_step']:.3f}  s/epoch={m['s_per_epoch']:.2f}  "
            f"pairs/s={m['pairs_per_s']:.3f}  (steps_per_pass={m['steps_per_pass']})",
            flush=True,
        )

    # Project n600 epoch wall-clock from the s/epoch measured here, scaled by the
    # steps_per_pass ratio (n600 has more steps/epoch than this n).
    if "torch_cpu_bridge" in results and "mlx_gpu" in results:
        t = results["torch_cpu_bridge"]["s_per_step"]
        g = results["mlx_gpu"]["s_per_step"]
        speedup = t / g if g > 0 else 0.0
        n600_steps_per_epoch = (600 + args.batch_size - 1) // args.batch_size
        print(f"\n=== n600 projection (bs={args.batch_size}, {n600_steps_per_epoch} steps/epoch) ===")
        print(f"  torch_cpu_bridge: {t * n600_steps_per_epoch / 60:.1f} min/epoch")
        print(f"  mlx_gpu:          {g * n600_steps_per_epoch / 60:.1f} min/epoch  (speedup {speedup:.2f}x)")
        results["_n600_projection"] = {
            "steps_per_epoch": n600_steps_per_epoch,
            "torch_min_per_epoch": t * n600_steps_per_epoch / 60,
            "mlx_min_per_epoch": g * n600_steps_per_epoch / 60,
            "speedup": speedup,
        }
    results["_config"] = vars(args)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
