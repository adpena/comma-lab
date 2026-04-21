#!/usr/bin/env python3
"""Test FP4 export quality against scorers.

Run when MPS is available (not during pose TTO):
    PYTHONPATH=src:upstream .venv/bin/python experiments/test_fp4_quality.py

Measures:
    1. Export renderer to FP8 and FP4
    2. Render 10 pairs with each
    3. Score both through PoseNet/SegNet
    4. Report proxy delta and size savings
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "upstream"))

from tac.renderer import AsymmetricPairGenerator
from tac.renderer_export import (
    export_asymmetric_checkpoint,
    export_asymmetric_checkpoint_fp4,
    load_asymmetric_checkpoint,
    load_asymmetric_checkpoint_fp4,
)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load checkpoint
    ckpt_path = Path("/tmp/distill_v2_latest_best.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("experiments/results/v5_lagrangian_renderer/renderer_best.pt")
    if not ckpt_path.exists():
        print("ERROR: No checkpoint found")
        sys.exit(1)

    model = AsymmetricPairGenerator(
        num_classes=5, embed_dim=6, base_ch=36, mid_ch=60,
        motion_hidden=32, depth=1, pose_dim=6,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    # Export both formats
    fp8_path = Path("/tmp/quality_test_fp8.bin")
    fp4_path = Path("/tmp/quality_test_fp4.bin")
    fp8_size = export_asymmetric_checkpoint(model, fp8_path, default_bits=8)
    fp4_size = export_asymmetric_checkpoint_fp4(model, fp4_path, block_size=32)

    model_fp8 = load_asymmetric_checkpoint(fp8_path, device=device)
    model_fp4 = load_asymmetric_checkpoint_fp4(fp4_path, device=device)

    # Load scorers
    from tac.scorer import load_differentiable_scorers
    posenet, segnet = load_differentiable_scorers(device=device)

    # Load masks
    upstream_root = Path("upstream")
    from tac.camera import extract_masks_from_video
    masks = extract_masks_from_video(
        upstream_root / "videos" / "0.mkv",
        segnet_model_path=upstream_root / "models" / "segnet.onnx",
        n_frames=20,
        device=device,
    )

    # Score 10 pairs with each model
    n_pairs = 10
    results = {"fp8": {"seg": [], "pose": []}, "fp4": {"seg": [], "pose": []}}

    for pair_idx in range(n_pairs):
        mask_t0 = masks[pair_idx * 2].unsqueeze(0).to(device)
        mask_t1 = masks[pair_idx * 2 + 1].unsqueeze(0).to(device)
        pose = torch.zeros(1, 6, device=device)

        with torch.no_grad():
            frames_fp8 = model_fp8(mask_t0, mask_t1, pose=pose)
            frames_fp4 = model_fp4(mask_t0, mask_t1, pose=pose)

        # Score through PoseNet and SegNet (simplified proxy)
        for label, frames in [("fp8", frames_fp8), ("fp4", frames_fp4)]:
            # frames is (frame_t0, frame_t1), each (1, 3, H, W) in [0, 255]
            pair = torch.cat([frames[0], frames[1]], dim=0)  # (2, 3, H, W)
            # Resize to scorer input
            pair_resized = F.interpolate(pair, size=(874, 1164), mode="bilinear", align_corners=False)
            pair_hwc = pair_resized.permute(0, 2, 3, 1)  # (2, H, W, 3)

            # PoseNet
            from tac.scorer import posenet_forward_pair
            pose_dist = posenet_forward_pair(posenet, pair_hwc)
            results[label]["pose"].append(pose_dist.item())

            # SegNet (compare to GT masks)
            gt_mask_pair = masks[pair_idx * 2:pair_idx * 2 + 2].to(device)
            from tac.scorer import segnet_disagreement
            seg_dist = segnet_disagreement(segnet, pair_hwc, gt_mask_pair)
            results[label]["seg"].append(seg_dist.item())

    # Report
    import statistics
    print(f"\n{'='*60}")
    print(f"FP4 Quality Test Results ({n_pairs} pairs)")
    print(f"{'='*60}")
    print(f"Archive size: FP8={fp8_size:,} bytes, FP4={fp4_size:,} bytes ({fp4_size/fp8_size*100:.1f}%)")
    print(f"Size savings: {fp8_size - fp4_size:,} bytes")
    print()

    for label in ["fp8", "fp4"]:
        avg_pose = statistics.mean(results[label]["pose"])
        avg_seg = statistics.mean(results[label]["seg"])
        import math
        score = 100 * avg_seg + math.sqrt(10 * avg_pose) + 25 * (fp8_size if label == "fp8" else fp4_size) / 37545489
        print(f"{label.upper()}: seg={avg_seg:.6f}, pose={avg_pose:.6f}, score={score:.4f}")

    # Verdict
    delta_pose = statistics.mean(results["fp4"]["pose"]) - statistics.mean(results["fp8"]["pose"])
    delta_seg = statistics.mean(results["fp4"]["seg"]) - statistics.mean(results["fp8"]["seg"])
    rate_savings = 25 * (fp8_size - fp4_size) / 37545489

    print(f"\nDelta: pose={delta_pose:+.6f}, seg={delta_seg:+.6f}, rate_savings={rate_savings:.6f}")
    net = 100 * delta_seg + (delta_pose * 0.5) - rate_savings  # approximate
    print(f"Net score impact (approx): {net:+.4f}")
    if net < 0:
        print("VERDICT: FP4 is NET POSITIVE (improves score)")
    else:
        print("VERDICT: FP4 is NET NEGATIVE (needs QAT)")


if __name__ == "__main__":
    main()
