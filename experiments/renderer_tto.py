#!/usr/bin/env python3
"""Renderer + TTO: test-time optimization on renderer output frames.

The pivotal experiment: take our auth=0.87 renderer output and refine it
via gradient descent against PoseNet+SegNet scorers. If PoseNet drops from
0.031 to <0.005, score goes from 0.87 to ~0.54.

The key insight: coupled_trajectory_optimize already supports warm-starting
from init_frames. We generate frames with the renderer, then TTO refines
them in batches of pairs (PoseNet evaluates non-overlapping pairs, so each
batch is independent).

Usage:
    # Local MPS (smoke test):
    PYTHONPATH=src:upstream python experiments/renderer_tto.py \
        --checkpoint /path/to/renderer_best.pt --device mps --smoke

    # Local MPS (40 frames):
    PYTHONPATH=src:upstream python experiments/renderer_tto.py \
        --checkpoint /path/to/renderer_best.pt --device mps --n-frames 40

    # Modal T4 (full run):
    PYTHONPATH=src:upstream python experiments/renderer_tto.py \
        --checkpoint /results/asym_v5_lagrangian_fixed/renderer_best.pt \
        --device cuda --n-frames 1200

    # bat00 2070S:
    PYTHONPATH=src:upstream python experiments/renderer_tto.py \
        --checkpoint ~/pact/checkpoints/renderer_best.pt --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Renderer + TTO: test-time optimization on renderer output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Path to renderer .pt checkpoint")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--n-frames", type=int, default=1200, help="Number of frames to process")
    p.add_argument("--tto-steps", type=int, default=500, help="TTO optimization steps per batch")
    p.add_argument("--tto-lr", type=float, default=0.005, help="TTO learning rate (lower than from-noise)")
    p.add_argument("--batch-pairs", type=int, default=50, help="Pairs per TTO batch (50 = 100 frames)")
    p.add_argument("--seg-weight", type=float, default=100.0, help="SegNet loss weight")
    p.add_argument("--pose-weight", type=float, default=10.0, help="PoseNet loss weight")
    p.add_argument("--compress-weight", type=float, default=0.5, help="Compressibility weight")
    p.add_argument("--upstream", type=str, default="upstream/", help="Path to upstream repo")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory (default: timestamped)")
    p.add_argument("--video", type=str, default=None, help="Path to GT video (default: upstream/videos/0.mkv)")
    p.add_argument("--smoke", action="store_true", help="Smoke test: 20 frames, 50 steps")
    p.add_argument("--simulate-resize", action="store_true",
                   help="Simulate official scorer's resolution round-trip (384→874→384) in proxy scoring. "
                        "Makes proxy score more faithful to auth eval at the cost of slight pessimism.")
    return p.parse_args()


def load_renderer(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load AsymmetricPairGenerator from checkpoint."""
    from tac.renderer import AsymmetricPairGenerator

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Detect model config from checkpoint if available
    model_cfg = ckpt.get("model_config", {})
    model = AsymmetricPairGenerator(
        num_classes=model_cfg.get("num_classes", 5),
        embed_dim=model_cfg.get("embed_dim", 6),
        base_ch=model_cfg.get("base_ch", 36),
        mid_ch=model_cfg.get("mid_ch", 60),
        motion_hidden=model_cfg.get("motion_hidden", 32),
        depth=model_cfg.get("depth", 1),
        max_flow_px=model_cfg.get("max_flow_px", 20.0),
        max_residual=model_cfg.get("max_residual", 20.0),
        flow_only=model_cfg.get("flow_only", False),
    )

    # Load weights — handle both direct state_dict and nested checkpoint formats
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        # Try loading the whole checkpoint as a state dict
        model.load_state_dict(ckpt)

    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[renderer] Loaded {n_params:,} params from {checkpoint_path}")
    return model


def extract_gt_masks(
    gt_frames: list[torch.Tensor],
    segnet: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Extract SegNet argmax masks from GT frames.

    Args:
        gt_frames: list of (H, W, 3) uint8 tensors at camera resolution.
        segnet: frozen SegNet model.
        device: computation device.
        batch_size: frames per forward pass.

    Returns:
        (N, seg_H, seg_W) long tensor of class indices at SegNet resolution.
    """
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W

    masks = []
    for i in range(0, len(gt_frames), batch_size):
        batch = gt_frames[i : i + batch_size]
        # Stack to (B, H, W, 3), convert to (B, C, H, W) float
        frames_t = torch.stack(batch).float().to(device)  # (B, H, W, 3)
        frames_chw = frames_t.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # Resize to SegNet input resolution if needed
        _, _, H, W = frames_chw.shape
        if H != SEGNET_INPUT_H or W != SEGNET_INPUT_W:
            frames_chw = F.interpolate(
                frames_chw, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear", align_corners=False,
            )

        # SegNet.preprocess_input expects (B, T, C, H, W) and selects x[:, -1, ...]
        # Use T=1 so it selects the only frame. Output is (B, classes, H, W).
        seg_in_btchw = frames_chw.unsqueeze(1)  # (B, 1, C, H, W)
        seg_in = segnet.preprocess_input(seg_in_btchw)
        with torch.no_grad():
            seg_out = segnet(seg_in)
        mask = seg_out.argmax(dim=1)  # (B, H, W) — all frames, no skipping
        masks.append(mask.cpu())

    return torch.cat(masks, dim=0).long()


def extract_gt_pose_targets(
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Extract PoseNet targets from GT frames using non-overlapping pairs.

    For N frames, produces N//2 pose targets: pair(0,1), pair(2,3), ...

    Args:
        gt_frames: list of (H, W, 3) uint8 tensors.
        posenet: frozen PoseNet model.
        device: computation device.
        batch_size: pairs per forward pass.

    Returns:
        (P, 6) float tensor of pose targets, P = len(gt_frames) // 2.
    """
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W

    N = len(gt_frames)
    P = N // 2
    targets = []

    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)
        batch_pairs = []
        for k in range(start, end):
            f0 = gt_frames[2 * k].float()      # (H, W, 3)
            f1 = gt_frames[2 * k + 1].float()  # (H, W, 3)
            pair = torch.stack([f0, f1], dim=0)  # (2, H, W, 3)
            batch_pairs.append(pair)

        pairs = torch.stack(batch_pairs).to(device)  # (B, 2, H, W, 3)
        pairs_chw = pairs.permute(0, 1, 4, 2, 3).contiguous()  # (B, 2, C, H, W)

        # Resize to SegNet/PoseNet input resolution if needed
        B, T, C, H, W = pairs_chw.shape
        if H != SEGNET_INPUT_H or W != SEGNET_INPUT_W:
            pairs_flat = pairs_chw.reshape(B * T, C, H, W)
            pairs_flat = F.interpolate(
                pairs_flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear", align_corners=False,
            )
            pairs_chw = pairs_flat.reshape(B, T, C, SEGNET_INPUT_H, SEGNET_INPUT_W)

        posenet_in = posenet.preprocess_input(pairs_chw)
        with torch.no_grad():
            posenet_out = posenet(posenet_in)
        pose = posenet_out["pose"][..., :6]  # (B, 6)
        targets.append(pose.cpu())

    return torch.cat(targets, dim=0).float()


def generate_renderer_frames(
    renderer: torch.nn.Module,
    masks: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Generate frames using the renderer on mask pairs.

    The renderer processes non-overlapping pairs (mask_2k, mask_2k+1) and
    produces (B, 2, H, W, 3) output. We collect all frames into a single tensor.

    Args:
        renderer: AsymmetricPairGenerator model.
        masks: (N, H, W) long tensor of segmentation masks.
        device: computation device.
        batch_size: pairs per forward pass.

    Returns:
        (N, H, W, 3) float tensor of rendered frames in [0, 255].
    """
    N = masks.shape[0]
    P = N // 2
    all_frames = []

    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)
        mask_t = masks[2 * start:2 * end:2].to(device)       # (B, H, W) even
        mask_t1 = masks[2 * start + 1:2 * end + 1:2].to(device)  # (B, H, W) odd

        with torch.no_grad():
            pair = renderer(mask_t, mask_t1)  # (B, 2, H, W, 3) HWC [0, 255]

        # Unpack pairs to individual frames: (2k) = pair[:,0], (2k+1) = pair[:,1]
        f0 = pair[:, 0]  # (B, H, W, 3)
        f1 = pair[:, 1]  # (B, H, W, 3)

        # Interleave: frame_2k, frame_2k+1
        B = f0.shape[0]
        interleaved = torch.stack([f0, f1], dim=1).reshape(2 * B, *f0.shape[1:])
        all_frames.append(interleaved.cpu())

    return torch.cat(all_frames, dim=0).float()


def compute_proxy_score(
    frames: torch.Tensor,
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    rate: float = 0.0,
    batch_size: int = 16,
    simulate_resize: bool = False,
) -> dict:
    """Compute proxy score matching the official scorer formula.

    Evaluates SegNet hard disagreement and PoseNet MSE on non-overlapping pairs.

    Args:
        frames: (N, H, W, 3) float tensor of candidate frames.
        gt_frames: list of (H, W, 3) uint8 GT frames.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        device: computation device.
        rate: rate term (archive_size / uncompressed_size).
        batch_size: pairs per forward pass.

    Returns:
        dict with seg, pose, rate, score, and contributions.
    """
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W

    N = frames.shape[0]
    P = N // 2
    total_pose, total_seg, n_pairs = 0.0, 0.0, 0

    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)

        # Build candidate pairs (B, 2, H, W, 3)
        cand_pairs = []
        gt_pairs = []
        for k in range(start, end):
            cf0 = frames[2 * k]
            cf1 = frames[2 * k + 1]
            cand_pairs.append(torch.stack([cf0, cf1], dim=0))

            gf0 = gt_frames[2 * k].float()
            gf1 = gt_frames[2 * k + 1].float()
            gt_pairs.append(torch.stack([gf0, gf1], dim=0))

        cand_t = torch.stack(cand_pairs).to(device)  # (B, 2, H, W, 3)
        gt_t = torch.stack(gt_pairs).to(device)       # (B, 2, H, W, 3)

        # Convert to (B, 2, C, H, W) and resize if needed
        cand_chw = cand_t.permute(0, 1, 4, 2, 3).contiguous()
        gt_chw = gt_t.permute(0, 1, 4, 2, 3).contiguous()

        B, T, C, H, W = cand_chw.shape
        if H != SEGNET_INPUT_H or W != SEGNET_INPUT_W:
            cand_flat = cand_chw.reshape(B * T, C, H, W)
            gt_flat = gt_chw.reshape(B * T, C, H, W)
            cand_flat = F.interpolate(cand_flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W), mode="bilinear", align_corners=False)
            gt_flat = F.interpolate(gt_flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W), mode="bilinear", align_corners=False)
            cand_chw = cand_flat.reshape(B, T, C, SEGNET_INPUT_H, SEGNET_INPUT_W)
            gt_chw = gt_flat.reshape(B, T, C, SEGNET_INPUT_H, SEGNET_INPUT_W)

        # uint8 round-trip to match official scorer (GT is already integer-valued from uint8 decode)
        cand_chw = cand_chw.round().clamp(0, 255)

        # Simulate official scorer's resolution round-trip: up to camera res then back down.
        # The official scorer loads (874, 1164) frames and resizes to (384, 512) internally.
        # TTO-optimized (384, 512) frames go through up→down which introduces interpolation loss.
        if simulate_resize:
            CAMERA_H, CAMERA_W = 874, 1164
            flat = cand_chw.reshape(-1, *cand_chw.shape[2:])
            flat = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
            flat = flat.round().clamp(0, 255)  # uint8 at camera res
            flat = F.interpolate(flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W), mode="bilinear", align_corners=False)
            cand_chw = flat.reshape(B, T, *flat.shape[1:])

        with torch.no_grad():
            # PoseNet
            fp_in = posenet.preprocess_input(cand_chw)
            gp_in = posenet.preprocess_input(gt_chw)
            fp_out = posenet(fp_in)
            gp_out = posenet(gp_in)
            pose_mse = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean(dim=-1)
            total_pose += pose_mse.sum().item()

            # SegNet
            fs_in = segnet.preprocess_input(cand_chw)
            gs_in = segnet.preprocess_input(gt_chw)
            fs_out = segnet(fs_in)
            gs_out = segnet(gs_in)
            diff = (fs_out.argmax(dim=1) != gs_out.argmax(dim=1)).float()
            seg_disagree = diff.mean(dim=tuple(range(1, diff.ndim)))
            total_seg += seg_disagree.sum().item()

            n_pairs += B

    avg_pose = total_pose / max(n_pairs, 1)
    avg_seg = total_seg / max(n_pairs, 1)
    score = 100.0 * avg_seg + math.sqrt(10.0 * avg_pose) + 25.0 * rate

    return {
        "score": score,
        "pose": avg_pose,
        "seg": avg_seg,
        "rate": rate,
        "pose_contribution": math.sqrt(10.0 * avg_pose),
        "seg_contribution": 100.0 * avg_seg,
        "rate_contribution": 25.0 * rate,
        "n_pairs": n_pairs,
    }


def estimate_vram_mb(batch_pairs: int) -> float:
    """Estimate peak VRAM usage for a TTO batch.

    Formula (empirically calibrated):
    - Per frame: ~2.4MB (384*512*3*4 bytes float32 + grad)
    - Adam buffers: 2x per frame (exp_avg + exp_avg_sq)
    - Snapshot: 1x per frame
    - Scorers: ~200MB fixed (SegNet ~100MB, PoseNet ~100MB)
    - Autograd graph: ~3MB per frame per scorer forward
    - Simplified: ~28MB per frame * batch_pairs * 2 + 200MB fixed
    """
    n_frames = batch_pairs * 2
    per_frame_mb = 28.0  # float32 + grad + adam + snapshot + autograd
    fixed_mb = 200.0     # scorers
    return n_frames * per_frame_mb + fixed_mb


def run_batched_tto(
    renderer_frames: torch.Tensor,
    masks: torch.Tensor,
    pose_targets: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    batch_pairs: int = 50,
    tto_steps: int = 500,
    tto_lr: float = 0.005,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    compress_weight: float = 0.5,
    output_dir: Path | None = None,
) -> torch.Tensor:
    """Run TTO in batches of pairs to avoid OOM.

    PoseNet evaluates non-overlapping pairs (2k, 2k+1). Each pair is independent
    for scoring purposes, so we can batch-optimize groups of pairs independently.

    Supports checkpoint resume: if output_dir is provided and contains
    tto_batch_NNN.pt files, completed batches are loaded instead of re-run.

    Args:
        renderer_frames: (N, H, W, 3) float tensor from renderer.
        masks: (N, H, W) long tensor of segmentation masks.
        pose_targets: (P, 6) float tensor, P = N//2.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        device: computation device.
        batch_pairs: number of pairs per batch (50 = 100 frames).
        tto_steps: optimization steps per batch.
        tto_lr: Adam learning rate.
        seg_weight: SegNet constraint weight.
        pose_weight: PoseNet constraint weight.
        compress_weight: compressibility weight.
        output_dir: directory for batch checkpoints (enables resume).

    Returns:
        (N, H, W, 3) float tensor of TTO-refined frames in [0, 255].
    """
    from tac.constrained_gen import coupled_trajectory_optimize

    N = renderer_frames.shape[0]
    P = N // 2
    if batch_pairs <= 0:
        raise ValueError(f"batch_pairs must be > 0, got {batch_pairs}")
    n_batches = math.ceil(P / batch_pairs)
    refined_frames = torch.zeros_like(renderer_frames)

    print(f"\n[tto] Starting batched TTO: {P} pairs in {n_batches} batches "
          f"({batch_pairs} pairs/batch, {tto_steps} steps, lr={tto_lr})")

    for batch_idx in range(n_batches):
        pair_start = batch_idx * batch_pairs
        pair_end = min(pair_start + batch_pairs, P)
        frame_start = 2 * pair_start
        frame_end = 2 * pair_end
        n_pairs_this = pair_end - pair_start
        n_frames_this = frame_end - frame_start

        # ── Checkpoint resume ────────────────────────────────────────────
        if output_dir is not None:
            ckpt_path = output_dir / f"tto_batch_{batch_idx:03d}.pt"
            if ckpt_path.exists():
                batch_result = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                expected_shape = refined_frames[frame_start:frame_end].shape
                if batch_result.shape != expected_shape:
                    print(f"[tto] Batch {batch_idx + 1}/{n_batches}: checkpoint shape "
                          f"{batch_result.shape} != expected {expected_shape}, re-running")
                else:
                    refined_frames[frame_start:frame_end] = batch_result
                    print(f"[tto] Batch {batch_idx + 1}/{n_batches}: RESUMED from checkpoint")
                    continue

        print(f"\n[tto] Batch {batch_idx + 1}/{n_batches}: "
              f"pairs [{pair_start}:{pair_end}] = {n_pairs_this} pairs, "
              f"{n_frames_this} frames")

        batch_masks = masks[frame_start:frame_end]
        batch_pose = pose_targets[pair_start:pair_end]
        batch_init = renderer_frames[frame_start:frame_end]

        t0 = time.monotonic()
        batch_result = coupled_trajectory_optimize(
            masks=batch_masks,
            expected_pose=batch_pose,
            posenet=posenet,
            segnet=segnet,
            num_steps=tto_steps,
            lr=tto_lr,
            seg_weight=seg_weight,
            pose_weight=pose_weight,
            compress_weight=compress_weight,
            noise_seed=42 + batch_idx,  # different seed per batch (only used if no init)
            device=str(device),
            log_every=max(tto_steps // 5, 1),
            init_frames=batch_init,
        )
        dt = time.monotonic() - t0

        refined_frames[frame_start:frame_end] = batch_result.cpu()
        print(f"[tto] Batch {batch_idx + 1}/{n_batches} done in {dt:.1f}s "
              f"({dt / n_frames_this:.2f}s/frame)")

        # ── Save batch checkpoint ────────────────────────────────────────
        if output_dir is not None:
            ckpt_path = output_dir / f"tto_batch_{batch_idx:03d}.pt"
            torch.save(batch_result.cpu(), ckpt_path)
            print(f"[tto] Checkpoint saved: {ckpt_path}")

        # Free GPU memory between batches
        del batch_result
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    return refined_frames


def main():
    args = parse_args()

    # Smoke test overrides
    if args.smoke:
        args.n_frames = 20
        args.tto_steps = 50
        args.batch_pairs = 10
        print("[smoke] Smoke test mode: 20 frames, 50 steps, 10 pairs/batch")
        print("[smoke] NOTE: 50 steps is insufficient for convergence. Use --tto-steps 500+ for meaningful results.")

    # Ensure even frame count (pairs)
    args.n_frames = args.n_frames - (args.n_frames % 2)

    device = torch.device(args.device)
    upstream = Path(args.upstream)
    if args.output_dir is None:
        ts = time.strftime("%Y%m%dT%H%M%S")
        args.output_dir = f"experiments/results/renderer_tto_{ts}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.video or str(upstream / "videos" / "0.mkv")

    print(f"[config] device={device}, n_frames={args.n_frames}, "
          f"tto_steps={args.tto_steps}, tto_lr={args.tto_lr}, "
          f"batch_pairs={args.batch_pairs}")
    print(f"[config] seg_weight={args.seg_weight}, pose_weight={args.pose_weight}, "
          f"compress_weight={args.compress_weight}")
    print(f"[config] checkpoint={args.checkpoint}")
    print(f"[config] video={video_path}")
    print(f"[config] output_dir={output_dir}")

    t_total_start = time.monotonic()

    # ── Step 1: Load scorers ─────────────────────────────────────────────
    print("\n[1/8] Loading scorers...")
    t0 = time.monotonic()
    from tac.scorer import load_scorers
    posenet, segnet = load_scorers(
        posenet_path=upstream / "models" / "posenet.safetensors",
        segnet_path=upstream / "models" / "segnet.safetensors",
        device=str(device),
        upstream_dir=str(upstream),
    )
    t_scorers = time.monotonic() - t0
    print(f"[1/8] Scorers loaded in {t_scorers:.1f}s")

    # ── Step 2: Load renderer ────────────────────────────────────────────
    print("\n[2/8] Loading renderer...")
    t0 = time.monotonic()
    renderer = load_renderer(args.checkpoint, device)
    t_renderer = time.monotonic() - t0
    print(f"[2/8] Renderer loaded in {t_renderer:.1f}s")

    # ── Step 3: Decode GT video ──────────────────────────────────────────
    print(f"\n[3/8] Decoding GT video ({args.n_frames} frames)...")
    t0 = time.monotonic()
    from tac.data import decode_video
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W

    # Decode at SegNet resolution directly — renderer operates at this resolution
    gt_frames_full = decode_video(video_path, target_h=SEGNET_INPUT_H, target_w=SEGNET_INPUT_W)
    gt_frames = gt_frames_full[: args.n_frames]
    # Update n_frames to actual count (video may be shorter than requested)
    args.n_frames = len(gt_frames) - (len(gt_frames) % 2)  # ensure even
    gt_frames = gt_frames[: args.n_frames]
    assert args.n_frames >= 2, f"Need at least 2 frames, got {len(gt_frames)}"
    t_decode = time.monotonic() - t0
    print(f"[3/8] Decoded {args.n_frames} frames ({gt_frames[0].shape}) in {t_decode:.1f}s")

    # ── Step 4: Extract masks via SegNet ─────────────────────────────────
    print("\n[4/8] Extracting SegNet masks from GT frames...")
    t0 = time.monotonic()
    masks = extract_gt_masks(gt_frames, segnet, device)
    t_masks = time.monotonic() - t0
    print(f"[4/8] Extracted {masks.shape[0]} masks ({masks.shape}) in {t_masks:.1f}s")

    # ── Step 5: Generate renderer frames ─────────────────────────────────
    print("\n[5/8] Generating renderer frames...")
    t0 = time.monotonic()
    renderer_frames = generate_renderer_frames(renderer, masks, device)
    t_render = time.monotonic() - t0
    print(f"[5/8] Generated {renderer_frames.shape[0]} frames ({renderer_frames.shape}) in {t_render:.1f}s")

    # Free renderer from GPU — not needed after this
    del renderer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Step 6: Extract GT pose targets ──────────────────────────────────
    print("\n[6/8] Extracting GT pose targets...")
    t0 = time.monotonic()
    pose_targets = extract_gt_pose_targets(gt_frames, posenet, device)
    t_pose = time.monotonic() - t0
    print(f"[6/8] Extracted {pose_targets.shape[0]} pose targets ({pose_targets.shape}) in {t_pose:.1f}s")

    # ── Baseline proxy score (renderer only, no TTO) ─────────────────────
    print("\n[...] Computing baseline proxy score (renderer only)...")
    baseline = compute_proxy_score(renderer_frames, gt_frames, posenet, segnet, device,
                                   simulate_resize=args.simulate_resize)
    print(f"[baseline] score={baseline['score']:.4f} | "
          f"seg={baseline['seg']:.6f} ({baseline['seg_contribution']:.4f}) | "
          f"pose={baseline['pose']:.6f} ({baseline['pose_contribution']:.4f}) | "
          f"rate={baseline['rate']:.4f} ({baseline['rate_contribution']:.4f})")

    # ── VRAM estimate ─────────────────────────────────────────────────────
    vram_est = estimate_vram_mb(args.batch_pairs)
    print(f"\n[vram] Estimated peak VRAM: {vram_est:.0f} MB "
          f"({args.batch_pairs} pairs/batch = {args.batch_pairs * 2} frames)")
    if vram_est > 22000:
        print(f"[vram] FATAL: estimated {vram_est:.0f} MB exceeds A10G capacity (22 GB). "
              "Reduce --batch-pairs or use a larger GPU.")
        sys.exit(1)
    elif vram_est > 14000:
        print(f"[vram] WARNING: estimated {vram_est:.0f} MB exceeds T4 safe limit (14 GB). "
              "May OOM on T4. Consider reducing --batch-pairs.")

    # ── Step 7: Run batched TTO ──────────────────────────────────────────
    print("\n[7/8] Running batched TTO...")
    t0 = time.monotonic()
    tto_frames = run_batched_tto(
        renderer_frames=renderer_frames,
        masks=masks,
        pose_targets=pose_targets,
        posenet=posenet,
        segnet=segnet,
        device=device,
        batch_pairs=args.batch_pairs,
        tto_steps=args.tto_steps,
        tto_lr=args.tto_lr,
        seg_weight=args.seg_weight,
        pose_weight=args.pose_weight,
        compress_weight=args.compress_weight,
        output_dir=output_dir,
    )
    t_tto = time.monotonic() - t0
    print(f"\n[7/8] TTO completed in {t_tto:.1f}s ({t_tto / args.n_frames:.2f}s/frame)")

    # ── Step 8: Compute proxy score on TTO-refined frames ────────────────
    print("\n[8/8] Computing TTO proxy score...")
    tto_result = compute_proxy_score(tto_frames, gt_frames, posenet, segnet, device,
                                     simulate_resize=args.simulate_resize)

    t_total = time.monotonic() - t_total_start

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RENDERER + TTO RESULTS")
    print("=" * 72)
    print(f"  Baseline (renderer only):")
    print(f"    score = {baseline['score']:.4f}")
    print(f"    seg   = {baseline['seg']:.6f}  (contribution: {baseline['seg_contribution']:.4f})")
    print(f"    pose  = {baseline['pose']:.6f}  (contribution: {baseline['pose_contribution']:.4f})")
    print(f"  After TTO:")
    print(f"    score = {tto_result['score']:.4f}")
    print(f"    seg   = {tto_result['seg']:.6f}  (contribution: {tto_result['seg_contribution']:.4f})")
    print(f"    pose  = {tto_result['pose']:.6f}  (contribution: {tto_result['pose_contribution']:.4f})")
    print(f"  Improvement:")
    delta_score = baseline['score'] - tto_result['score']
    delta_pose = baseline['pose'] - tto_result['pose']
    delta_seg = baseline['seg'] - tto_result['seg']
    print(f"    score: {delta_score:+.4f} ({'better' if delta_score > 0 else 'worse'})")
    print(f"    pose:  {delta_pose:+.6f}")
    print(f"    seg:   {delta_seg:+.6f}")
    print(f"  Timing:")
    print(f"    total = {t_total:.1f}s | TTO = {t_tto:.1f}s | "
          f"scorers = {t_scorers:.1f}s | decode = {t_decode:.1f}s")
    print("=" * 72)

    # ── Save results ─────────────────────────────────────────────────────
    torch.save(tto_frames.to(torch.uint8), output_dir / "tto_frames.pt")
    print(f"\n[save] TTO frames saved to {output_dir / 'tto_frames.pt'} "
          f"({tto_frames.shape[0] * tto_frames.shape[1] * tto_frames.shape[2] * 3 / 1e6:.0f}MB uint8)")

    results = {
        "baseline": baseline,
        "tto": tto_result,
        "improvement": {
            "score": delta_score,
            "pose": delta_pose,
            "seg": delta_seg,
        },
        "config": {
            "checkpoint": args.checkpoint,
            "device": args.device,
            "n_frames": args.n_frames,
            "tto_steps": args.tto_steps,
            "tto_lr": args.tto_lr,
            "batch_pairs": args.batch_pairs,
            "seg_weight": args.seg_weight,
            "pose_weight": args.pose_weight,
            "compress_weight": args.compress_weight,
            "video": video_path,
            "smoke": args.smoke,
        },
        "timing": {
            "total_s": round(t_total, 2),
            "tto_s": round(t_tto, 2),
            "scorers_s": round(t_scorers, 2),
            "renderer_s": round(t_renderer, 2),
            "decode_s": round(t_decode, 2),
            "masks_s": round(t_masks, 2),
            "render_s": round(t_render, 2),
            "pose_targets_s": round(t_pose, 2),
        },
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
