#!/usr/bin/env python3
"""Hybrid inflate: renderer for easy pairs + constrained gen for hard pairs.

The top 20% hardest pairs (by PoseNet difficulty) get optimized via
gradient descent through frozen scorers at inflate time. The remaining
80% use the fast renderer forward pass.

This stays within the 30-min T4 budget:
  - 480 easy pairs via renderer: ~30s
  - 120 hard pairs via constrained gen (100 steps): ~14 min on T4
  - Total inflate: ~15 min

The difficulty map is pre-computed at compress time and stored in the
archive (2KB). At inflate time, pairs above the threshold get the
expensive constrained gen path.

Usage:
    python hybrid_inflate.py archive_dir inflated_dir video_names.txt \
        --hard-fraction 0.2 --cg-steps 100
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description="Hybrid inflate: renderer + constrained gen")
    parser.add_argument("archive_dir", type=str)
    parser.add_argument("inflated_dir", type=str)
    parser.add_argument("video_names_file", type=str)
    parser.add_argument("--hard-fraction", type=float, default=0.2,
                        help="Fraction of pairs to optimize with constrained gen")
    parser.add_argument("--cg-steps", type=int, default=100,
                        help="Constrained gen optimization steps for hard pairs")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    inflated_dir = Path(args.inflated_dir)
    inflated_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    t_start = time.monotonic()

    # ── Load renderer ────────────────────────────────────────────────
    renderer_path = archive_dir / "renderer.bin"
    raw = renderer_path.read_bytes()
    if raw[:4] == b"ASYM":
        sys.path.insert(0, "src")
        from tac.renderer_export import load_asymmetric_checkpoint
        renderer = load_asymmetric_checkpoint(raw, device=device)
    elif raw[:4] == b"FP4A":
        sys.path.insert(0, "src")
        from tac.renderer_export import load_asymmetric_checkpoint_fp4
        renderer = load_asymmetric_checkpoint_fp4(raw, device=device)
    else:
        raise ValueError(f"Unknown renderer format: {raw[:4]}")
    renderer.eval()
    for p in renderer.parameters():
        p.requires_grad_(False)
    print(f"Renderer: {sum(p.numel() for p in renderer.parameters()):,} params", file=sys.stderr)

    # ── Load masks ───────────────────────────────────────────────────
    mask_path = archive_dir / "masks.mkv"
    cmd = ["ffmpeg", "-v", "quiet", "-i", str(mask_path),
           "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(mask_path)],
        capture_output=True, text=True, check=True)
    w, h = map(int, probe.stdout.strip().split(","))
    pixels = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, h, w)
    scale = 255 // 4
    masks = torch.from_numpy(
        np.clip(np.round(pixels.astype(np.float32) / scale).astype(np.int64), 0, 4)
    ).long()
    if masks.shape[1] < 384 or masks.shape[2] < 512:
        masks = F.interpolate(masks.float().unsqueeze(1), size=(384, 512),
                              mode="nearest").squeeze(1).long()
    N = masks.shape[0]
    n_pairs = N // 2
    print(f"Masks: {masks.shape}", file=sys.stderr)

    # ── Load poses ───────────────────────────────────────────────────
    poses = None
    for poses_name in ["optimized_poses.pt", "poses.pt"]:
        poses_path = archive_dir / poses_name
        if poses_path.exists():
            poses = torch.load(str(poses_path), map_location="cpu", weights_only=True).float()
            print(f"Poses: {poses.shape} from {poses_name}", file=sys.stderr)
            break

    # ── Load difficulty map (if exists) ──────────────────────────────
    difficulty_path = archive_dir / "difficulty.pt"
    if difficulty_path.exists():
        difficulty = torch.load(str(difficulty_path), map_location="cpu", weights_only=True)
        print(f"Difficulty map: {difficulty.shape}", file=sys.stderr)
    else:
        # No difficulty map — use uniform (all pairs get renderer)
        difficulty = torch.zeros(n_pairs)
        print("No difficulty map — all pairs use renderer", file=sys.stderr)

    # ── Classify pairs as easy/hard ──────────────────────────────────
    n_hard = int(n_pairs * args.hard_fraction)
    if n_hard > 0 and difficulty.max() > 0:
        _, hard_indices = difficulty.topk(n_hard)
        hard_set = set(hard_indices.tolist())
    else:
        hard_set = set()
    n_easy = n_pairs - len(hard_set)
    print(f"Easy: {n_easy}, Hard (CG): {len(hard_set)}", file=sys.stderr)

    # ── Generate frames ──────────────────────────────────────────────
    OUT_H, OUT_W = 874, 1164
    all_frames = [None] * N  # frame buffer

    # Easy pairs: renderer forward pass
    t_easy = time.monotonic()
    with torch.inference_mode():
        for i in range(n_pairs):
            if i in hard_set:
                continue
            m_t = masks[2 * i:2 * i + 1].to(device=device)
            m_t1 = masks[2 * i + 1:2 * i + 2].to(device=device)
            pose = poses[i:i + 1].to(device) if poses is not None else None
            kwargs = {"pose": pose} if pose is not None else {}
            pair = renderer(m_t, m_t1, **kwargs)  # (1, 2, H, W, 3)
            for j in range(2):
                frame = pair[0, j].permute(2, 0, 1).unsqueeze(0).float()
                frame = F.interpolate(frame, size=(OUT_H, OUT_W),
                                      mode="bilinear", align_corners=False)
                all_frames[2 * i + j] = frame.round().clamp(0, 255).to(torch.uint8).cpu()
    print(f"Easy pairs: {time.monotonic() - t_easy:.1f}s", file=sys.stderr)

    # Hard pairs: constrained gen
    if hard_set:
        t_hard = time.monotonic()
        # Load scorers for constrained gen
        sys.path.insert(0, "upstream")
        from tac.scorer import load_differentiable_scorers
        posenet, segnet = load_differentiable_scorers("upstream", device=device)
        from tac.constrained_gen import coupled_trajectory_optimize

        hard_list = sorted(hard_set)
        hard_masks_idx = []
        hard_poses_list = []
        for i in hard_list:
            hard_masks_idx.extend([2 * i, 2 * i + 1])
            if poses is not None:
                hard_poses_list.append(poses[i])

        hard_masks = masks[hard_masks_idx].to(device)
        hard_poses_t = torch.stack(hard_poses_list).to(device) if hard_poses_list else None

        result = coupled_trajectory_optimize(
            masks=hard_masks,
            expected_pose=hard_poses_t,
            posenet=posenet, segnet=segnet,
            num_steps=args.cg_steps,
            lr=1.0,
            seg_weight=100.0, pose_weight=10.0,
            device=device, segnet_loss_mode="hinge",
            eval_roundtrip=True,
        )
        # result is (2*n_hard, H, W, 3)
        for k, i in enumerate(hard_list):
            for j in range(2):
                frame = result[2 * k + j].permute(2, 0, 1).unsqueeze(0).float()
                frame = F.interpolate(frame, size=(OUT_H, OUT_W),
                                      mode="bilinear", align_corners=False)
                all_frames[2 * i + j] = frame.round().clamp(0, 255).to(torch.uint8).cpu()
        print(f"Hard pairs ({len(hard_set)}): {time.monotonic() - t_hard:.1f}s", file=sys.stderr)

    # ── Write .raw output ────────────────────────────────────────────
    video_names = Path(args.video_names_file).read_text().splitlines()
    video_names = [v.strip() for v in video_names if v.strip()]
    stem = video_names[0].rsplit(".", 1)[0]

    raw_path = inflated_dir / f"{stem}.raw"
    with open(raw_path, "wb") as f:
        for frame in all_frames:
            if frame is None:
                raise RuntimeError("Missing frame in output buffer")
            # (1, 3, H, W) uint8 → (H, W, 3) → bytes
            f.write(frame.squeeze(0).permute(1, 2, 0).numpy().tobytes())

    total = time.monotonic() - t_start
    print(f"Total inflate: {total:.1f}s ({N} frames)", file=sys.stderr)

    # Write video_names.txt
    vn_path = Path(args.video_names_file)
    vn_path.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
