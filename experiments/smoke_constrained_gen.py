#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Smoke test for constrained generation — runs locally on MPS/CPU.

Verifies the full pipeline:
  1. Load scorers (SegNet + PoseNet)
  2. Extract masks + pose targets from GT video
  3. Run constrained_generate for 10 steps
  4. Verify output shape and basic sanity

Usage:
    PYTHONPATH=src:upstream .venv/bin/python experiments/smoke_constrained_gen.py
    PYTHONPATH=src:upstream .venv/bin/python experiments/smoke_constrained_gen.py --device mps --steps 50
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = REPO_ROOT / "upstream"


def main() -> int:
    parser = argparse.ArgumentParser(description="Constrained gen smoke test")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--n-frames", type=int, default=20, help="Use first N frames (default 20 for speed)")
    parser.add_argument("--coupled", action="store_true", help="Use coupled trajectory optimization (joint pairs)")
    parser.add_argument("--video", type=Path, default=UPSTREAM / "videos" / "0.mkv")
    args = parser.parse_args()

    device = args.device
    print("=== Constrained Gen Smoke Test ===")
    print(f"  Device: {device}")
    print(f"  Steps: {args.steps}")
    print(f"  Frames: {args.n_frames}")

    # 1. Load scorers via tac.scorer (handles PoseNet vs DistortionNet correctly)
    print("\n[1/4] Loading scorers...")
    from tac.scorer import load_differentiable_scorers
    posenet, segnet = load_differentiable_scorers(UPSTREAM, device=device)
    print(f"  Scorers loaded + patched differentiable on {device}")

    # 2. Extract masks + pose targets
    print("\n[2/4] Extracting masks + pose targets...")
    import av
    import numpy as np

    container = av.open(str(args.video))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= args.n_frames:
            break
    container.close()
    gt_np = np.stack(frames)  # (N, H, W, 3)
    gt_t = torch.from_numpy(gt_np).float()
    print(f"  GT shape: {gt_t.shape}")

    # Extract masks
    import torch.nn.functional as F
    gt_chw = gt_t.permute(0, 3, 1, 2)  # (N, 3, H, W)
    seg_input = F.interpolate(gt_chw, size=(384, 512), mode="bilinear", align_corners=False)
    masks_list = []
    with torch.no_grad():
        for i in range(0, seg_input.shape[0], 4):
            batch = seg_input[i:i+4].to(device).unsqueeze(1)
            inp = segnet.preprocess_input(batch)
            logits = segnet(inp)
            masks_list.append(logits.argmax(dim=1).cpu())
    masks = torch.cat(masks_list, dim=0)  # (N, 384, 512)
    print(f"  Masks shape: {masks.shape}, classes: {masks.unique().tolist()}")

    # Extract pose targets for consecutive pairs
    n_pairs = (args.n_frames // 2)
    pose_targets_list = []
    with torch.no_grad():
        for i in range(0, args.n_frames - 1, 2):
            pair = gt_chw[i:i+2].unsqueeze(0).to(device)
            inp = posenet.preprocess_input(pair)
            out = posenet(inp)
            pose = out["pose"] if isinstance(out, dict) else out
            pose_targets_list.append(pose[..., :6].cpu())
    pose_targets = torch.cat(pose_targets_list, dim=0)  # (n_pairs, 6)
    print(f"  Pose targets shape: {pose_targets.shape}")

    # 3. Run constrained generation
    print(f"\n[3/4] Running constrained_generate ({args.steps} steps)...")
    from tac.constrained_gen import constrained_generate, coupled_trajectory_optimize

    t0 = time.time()
    if args.coupled:
        print("  Mode: COUPLED trajectory (joint pair optimization)")
        generated = coupled_trajectory_optimize(
            masks=masks.to(device),
            expected_pose=pose_targets.to(device),
            posenet=posenet,
            segnet=segnet,
            noise_seed=42,
            num_steps=args.steps,
            lr=0.01,
            seg_weight=100.0,
            pose_weight=10.0,
            compress_weight=1.0,
            device=str(device),
            log_every=max(1, args.steps // 5),
        )
    else:
        print("  Mode: INDEPENDENT frame optimization")
        generated = constrained_generate(
            masks=masks.to(device),
            expected_pose=pose_targets.to(device),
            posenet=posenet,
            segnet=segnet,
            noise_seed=42,
            num_steps=args.steps,
            lr=0.1,
            seg_weight=50.0,
            pose_weight=50.0,
            device=device,
            log_every=max(1, args.steps // 5),
            early_stop_patience=0,
            segnet_batch_size=4,
            posenet_batch_size=4,
        )
    gen_time = time.time() - t0
    print(f"  Generated: {generated.shape} in {gen_time:.1f}s ({args.steps / gen_time:.1f} steps/s)")

    # 4. Sanity checks
    print("\n[4/4] Sanity checks...")
    assert generated.shape[0] == args.n_frames, f"Expected {args.n_frames} frames, got {generated.shape[0]}"
    assert generated.shape[1] == 384, f"Expected H=384, got {generated.shape[1]}"
    assert generated.shape[2] == 512, f"Expected W=512, got {generated.shape[2]}"
    assert generated.shape[3] == 3, f"Expected C=3, got {generated.shape[3]}"
    vmin, vmax = generated.min().item(), generated.max().item()
    print(f"  Shape: {generated.shape} -- OK")
    print(f"  Value range: [{vmin:.1f}, {vmax:.1f}]")
    assert vmax <= 256 and vmin >= -1, f"Values out of expected range: [{vmin}, {vmax}]"

    # Quick proxy: verify SegNet agreement on generated frames
    gen_chw = generated.permute(0, 3, 1, 2).float()
    gen_masks = []
    with torch.no_grad():
        for i in range(0, gen_chw.shape[0], 4):
            batch = gen_chw[i:i+4].to(device).unsqueeze(1)
            inp = segnet.preprocess_input(batch)
            logits = segnet(inp)
            gen_masks.append(logits.argmax(dim=1).cpu())
    gen_masks = torch.cat(gen_masks, dim=0)
    agreement = (gen_masks == masks).float().mean().item()
    print(f"  SegNet agreement: {agreement:.4f} ({agreement*100:.1f}%)")

    print("\n=== SMOKE TEST PASSED ===")
    print("  Constrained gen is functional.")
    print(f"  {args.steps} steps on {device} took {gen_time:.1f}s")
    if args.steps >= 50:
        print(f"  Estimated 1000 steps: {gen_time / args.steps * 1000:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
