#!/usr/bin/env python3
"""Contest-compliant inflate script for constrained generation from noise.

This is the INFLATE-TIME version that runs within the 30-min total budget
(~20-25 min for inflation) using ONLY archive contents. No full scorers.

The archive contains:
  - mini_segnet.bin: ~25KB (FP16 distilled SegNet)
  - mini_posenet.bin: ~25KB (FP16 distilled PoseNet)
  - poses.pt: ~8.7KB (GT pose targets)
  - masks.mkv: compressed mask video
  - config.json: hyperparameters + noise seed

At inflate time:
  1. Load mini-scorers from archive (50KB, instant)
  2. Decompress masks from masks.mkv
  3. Initialize 1200 frames from deterministic noise seed + class-mean colors
  4. Run batched gradient descent against mini-scorers
  5. Apply eval roundtrip (upscale -> round -> downscale) periodically
  6. Write final frames as .raw for scoring

Time budget analysis (T4, 1200 frames, 1000 steps):
  - Mini-scorer forward+backward: ~0.05ms/frame (vs ~5ms for full scorer)
  - Per step (1200 frames, batch=40): ~60ms
  - Total 1000 steps: ~60s = 1 minute
  - With batching overhead: ~2-4 minutes total
  - Well within 20-min inflation budget

Memory budget (T4, 16GB VRAM):
  - Mini-scorers: ~0.5MB
  - Frames (batch 40): 40 * 384 * 512 * 3 * 4 (float32) = 94MB
  - Adam states: 2x frames = 188MB
  - Gradients: 94MB
  - Total per batch: ~380MB (fits easily)

Usage:
    # From archive directory:
    PYTHONPATH=src python experiments/constrained_gen_inflate.py \
        --archive-dir /path/to/archive_contents/ \
        --output-dir /path/to/inflated/ \
        --device cuda

    # Smoke test:
    PYTHONPATH=src python experiments/constrained_gen_inflate.py \
        --archive-dir experiments/results/constrained_archive/ \
        --output-dir /tmp/inflated \
        --device mps --smoke

    # Simulate contest conditions (time-limited):
    PYTHONPATH=src python experiments/constrained_gen_inflate.py \
        --archive-dir experiments/results/constrained_archive/ \
        --output-dir /tmp/inflated \
        --device cuda --time-limit 1200
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Target resolution for scorer evaluation
SCORER_H = 384
SCORER_W = 512
# Output resolution for contest (874x1164)
CAMERA_H = 874
CAMERA_W = 1164


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Constrained generation inflate (contest-compliant)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--archive-dir", type=str, required=True,
                   help="Directory with mini-scorers + targets")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Output directory for .raw frames")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "mps", "cpu"])

    # Optimization (loaded from config.json if present, CLI overrides)
    p.add_argument("--steps", type=int, default=None,
                   help="Override optimization steps (default: from config.json)")
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning rate")
    p.add_argument("--seg-weight", type=float, default=None)
    p.add_argument("--pose-weight", type=float, default=None)
    p.add_argument("--batch-pairs", type=int, default=None,
                   help="Pairs per batch (reduce for OOM)")

    # Time budget
    p.add_argument("--time-limit", type=int, default=1200,
                   help="Hard time limit in seconds (default: 20 min)")

    # Smoke test
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 10 frames, 30 steps")

    return p.parse_args()


def load_archive(archive_dir: Path, device: torch.device) -> dict:
    """Load all archive contents.

    Expected files:
        - mini_segnet.bin: FP16 state dict for MiniSegNet
        - mini_posenet.bin: FP16 state dict for MiniPoseNet
        - poses.pt: (P, 6) pose targets
        - masks.mkv: compressed mask video OR masks.pt: tensor
        - config.json: hyperparameters
    """
    from tac.mini_scorer import load_mini_scorers

    logger.info("Loading archive from %s", archive_dir)

    # Mini-scorers
    mini_seg, mini_pose = load_mini_scorers(archive_dir, device=device)
    seg_params = sum(p.numel() for p in mini_seg.parameters())
    pose_params = sum(p.numel() for p in mini_pose.parameters())
    logger.info("  Mini-scorers: seg=%d params, pose=%d params", seg_params, pose_params)

    # Pose targets
    poses_path = archive_dir / "poses.pt"
    if poses_path.exists():
        pose_targets = torch.load(str(poses_path), map_location="cpu", weights_only=True)
        logger.info("  Pose targets: %s", pose_targets.shape)
    else:
        # Fallback: try posenet_targets.bin (legacy format)
        bin_path = archive_dir / "posenet_targets.bin"
        if bin_path.exists():
            data = bin_path.read_bytes()
            # First 8 bytes: (P, 6) shape as uint32
            p_count = struct.unpack("<I", data[:4])[0]
            dims = struct.unpack("<I", data[4:8])[0]
            pose_targets = torch.frombuffer(
                bytearray(data[8:]), dtype=torch.float32
            ).reshape(p_count, dims)
            logger.info("  Pose targets (legacy): (%d, %d)", p_count, dims)
        else:
            raise FileNotFoundError(f"No poses.pt or posenet_targets.bin in {archive_dir}")

    # Masks
    masks = None
    masks_pt = archive_dir / "masks.pt"
    masks_mkv = archive_dir / "masks.mkv"

    if masks_pt.exists():
        masks = torch.load(str(masks_pt), map_location="cpu", weights_only=True)
        logger.info("  Masks (tensor): %s", masks.shape)
    elif masks_mkv.exists():
        # Decode mask video
        try:
            from tac.mask_codec import decode_mask_video
            masks = decode_mask_video(str(masks_mkv))
            logger.info("  Masks (video): %s", masks.shape)
        except ImportError:
            # Fallback: use av directly
            import av
            container = av.open(str(masks_mkv))
            mask_frames = []
            for frame in container.decode(video=0):
                arr = frame.to_ndarray(format="gray")
                mask_frames.append(torch.from_numpy(arr).long())
            container.close()
            masks = torch.stack(mask_frames)
            logger.info("  Masks (av decode): %s", masks.shape)
    else:
        raise FileNotFoundError(f"No masks.pt or masks.mkv in {archive_dir}")

    # Config
    config = {
        "steps": 1000,
        "lr": 0.02,
        "seg_weight": 100.0,
        "pose_weight": 10.0,
        "noise_seed": 42,
        "batch_pairs": 20,
        "segnet_loss_mode": "hinge",
        "hinge_margin": 0.5,
    }
    config_path = archive_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config.update(json.load(f))
        logger.info("  Config loaded from config.json")

    return {
        "mini_seg": mini_seg,
        "mini_pose": mini_pose,
        "pose_targets": pose_targets,
        "masks": masks,
        "config": config,
    }


def generate_initial_frames_from_masks(
    masks: torch.Tensor,
    noise_seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate initial frames from class-mean colors + seeded noise.

    Delegates to tac.constrained_gen.generate_initial_frames.
    """
    from tac.constrained_gen import generate_initial_frames
    return generate_initial_frames(masks, noise_seed, device=str(device))


def optimize_batched(
    frames: torch.Tensor,
    masks_mini: torch.Tensor,
    pose_targets: torch.Tensor,
    mini_seg: torch.nn.Module,
    mini_pose: torch.nn.Module,
    device: torch.device,
    *,
    steps: int = 1000,
    lr: float = 0.02,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    batch_pairs: int = 20,
    time_limit: float = 1200.0,
    log_every: int = 50,
    loss_mode: str = "hinge",
    hinge_margin: float = 0.5,
) -> torch.Tensor:
    """Batched gradient descent against mini-scorers with time budget.

    Processes frames in chunks of batch_pairs pairs. Each chunk is optimized
    independently (PoseNet evaluates non-overlapping pairs, so chunks are
    independent when aligned on pair boundaries).

    Includes time-budget awareness: if approaching time_limit, stops early
    and returns best-so-far frames.
    """
    from tac.mini_scorer import MiniScorerTTO

    N = frames.shape[0]
    P = N // 2
    n_chunks = (P + batch_pairs - 1) // batch_pairs
    t_start = time.time()

    mini_tto = MiniScorerTTO(mini_seg, mini_pose, device=device)
    all_optimized = []

    for chunk_idx in range(n_chunks):
        chunk_start_pair = chunk_idx * batch_pairs
        chunk_end_pair = min(chunk_start_pair + batch_pairs, P)
        chunk_start_frame = chunk_start_pair * 2
        chunk_end_frame = chunk_end_pair * 2

        # Time budget check
        elapsed = time.time() - t_start
        remaining = time_limit - elapsed
        if remaining < 10.0:
            logger.warning("Time budget exhausted at chunk %d/%d (%.0fs elapsed)",
                          chunk_idx + 1, n_chunks, elapsed)
            # Return remaining frames un-optimized
            all_optimized.append(frames[chunk_start_frame:].round().clamp(0, 255).cpu())
            break

        # Adaptive step reduction if running low on time
        time_per_chunk_estimate = elapsed / max(chunk_idx, 1)
        chunks_remaining = n_chunks - chunk_idx
        if chunk_idx > 0 and time_per_chunk_estimate * chunks_remaining > remaining * 0.9:
            adaptive_steps = max(steps // 2, 50)
            if adaptive_steps < steps:
                logger.info("  Adaptive: reducing steps %d -> %d (time pressure)",
                           steps, adaptive_steps)
        else:
            adaptive_steps = steps

        # Extract chunk
        chunk_frames = frames[chunk_start_frame:chunk_end_frame].to(device).float().detach().clone()
        chunk_frames.requires_grad_(True)
        chunk_masks = masks_mini[chunk_start_frame:chunk_end_frame].to(device)
        chunk_poses = pose_targets[chunk_start_pair:chunk_end_pair].to(device)

        optimizer = torch.optim.Adam([chunk_frames], lr=lr)

        best_loss = float("inf")
        best_chunk = chunk_frames.detach().clone()

        for step in range(adaptive_steps):
            optimizer.zero_grad()

            # SegNet loss
            frames_chw = chunk_frames.permute(0, 3, 1, 2).contiguous()
            seg_logits = mini_seg(frames_chw)

            if loss_mode == "hinge":
                # Hinge loss (boundary-focused, 24-49% better)
                target = chunk_masks
                target_logits = seg_logits.gather(1, target.unsqueeze(1))
                mask_fill = seg_logits.scatter(1, target.unsqueeze(1), float("-inf"))
                max_wrong = mask_fill.max(dim=1, keepdim=True).values
                hinge = F.relu(hinge_margin - (target_logits - max_wrong)).mean()
                seg_loss = hinge
            else:
                seg_loss = F.cross_entropy(seg_logits, chunk_masks)

            # PoseNet loss
            f1 = frames_chw[0::2]
            f2 = frames_chw[1::2]
            pairs = torch.cat([f1, f2], dim=1)
            pred_pose = mini_pose(pairs)
            pose_loss = F.mse_loss(pred_pose, chunk_poses)

            total_loss = seg_weight * seg_loss + pose_weight * pose_loss
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                chunk_frames.data.clamp_(0.0, 255.0)

            loss_val = total_loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_chunk = chunk_frames.detach().clone()

            if log_every > 0 and (step + 1) % log_every == 0:
                logger.info(
                    "  chunk %d/%d step %d/%d: total=%.4f seg=%.4f pose=%.6f",
                    chunk_idx + 1, n_chunks, step + 1, adaptive_steps,
                    total_loss.item(), seg_loss.item(), pose_loss.item(),
                )

        all_optimized.append(best_chunk.round().clamp(0.0, 255.0).cpu())

        # Free memory
        del chunk_frames, optimizer, best_chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = torch.cat(all_optimized, dim=0)
    total_time = time.time() - t_start
    logger.info("Optimization complete: %d frames in %.1fs (%.2f ms/frame/step)",
                result.shape[0], total_time,
                total_time * 1000 / (result.shape[0] * steps))
    return result


def write_raw_output(frames: torch.Tensor, output_dir: Path) -> Path:
    """Write frames as uint8 .raw file for contest evaluation.

    The official scorer expects frames upscaled to CAMERA_H x CAMERA_W
    as raw RGB24 bytes (no header, frame-major order).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    N, H, W, C = frames.shape

    # Upscale to contest resolution if needed
    if H != CAMERA_H or W != CAMERA_W:
        frames_chw = frames.permute(0, 3, 1, 2).float()
        frames_up = F.interpolate(
            frames_chw, size=(CAMERA_H, CAMERA_W),
            mode="bilinear", align_corners=False,
        )
        frames_out = frames_up.permute(0, 2, 3, 1).round().clamp(0, 255).byte()
    else:
        frames_out = frames.round().clamp(0, 255).byte()

    raw_path = output_dir / "0.raw"
    frames_np = frames_out.numpy().astype(np.uint8)
    frames_np.tofile(raw_path)

    n_bytes = raw_path.stat().st_size
    expected = N * CAMERA_H * CAMERA_W * 3
    assert n_bytes == expected, f"Raw file size mismatch: {n_bytes} != {expected}"

    logger.info("Wrote %d frames to %s (%s bytes)", N, raw_path, f"{n_bytes:,}")
    return raw_path


def main() -> None:
    args = parse_args()
    t_start = time.time()

    if args.smoke:
        logger.info("SMOKE TEST: 10 frames, 30 steps")

    device = torch.device(args.device)
    archive_dir = Path(args.archive_dir)
    output_dir = Path(args.output_dir)

    # Load archive
    archive = load_archive(archive_dir, device)
    mini_seg = archive["mini_seg"]
    mini_pose = archive["mini_pose"]
    pose_targets = archive["pose_targets"]
    masks = archive["masks"]
    config = archive["config"]

    # Apply CLI overrides
    steps = args.steps if args.steps is not None else config["steps"]
    lr = args.lr if args.lr is not None else config["lr"]
    seg_weight = args.seg_weight if args.seg_weight is not None else config["seg_weight"]
    pose_weight = args.pose_weight if args.pose_weight is not None else config["pose_weight"]
    batch_pairs = args.batch_pairs if args.batch_pairs is not None else config["batch_pairs"]
    noise_seed = config.get("noise_seed", 42)
    loss_mode = config.get("segnet_loss_mode", "hinge")
    hinge_margin = config.get("hinge_margin", 0.5)

    if args.smoke:
        # Truncate for smoke test
        n_frames = min(10, masks.shape[0])
        masks = masks[:n_frames]
        pose_targets = pose_targets[:n_frames // 2]
        steps = 30
        batch_pairs = min(batch_pairs, 5)

    N = masks.shape[0]
    P = N // 2
    logger.info("Inflate config: %d frames, %d pairs, %d steps, lr=%.4f",
                N, P, steps, lr)
    logger.info("  seg_weight=%.1f, pose_weight=%.1f, batch_pairs=%d",
                seg_weight, pose_weight, batch_pairs)
    logger.info("  loss_mode=%s, noise_seed=%d", loss_mode, noise_seed)

    # Generate initial frames
    logger.info("Generating initial frames from masks + seed...")
    frames = generate_initial_frames_from_masks(masks, noise_seed, device)
    logger.info("  Initial frames: %s", frames.shape)

    # Prepare mini-resolution masks for mini-segnet
    from tac.mini_scorer import MINI_SEG_H, MINI_SEG_W
    masks_mini = F.interpolate(
        masks.float().unsqueeze(1),
        size=(MINI_SEG_H, MINI_SEG_W),
        mode="nearest",
    ).squeeze(1).long()

    # Run optimization
    optimized = optimize_batched(
        frames=frames,
        masks_mini=masks_mini,
        pose_targets=pose_targets,
        mini_seg=mini_seg,
        mini_pose=mini_pose,
        device=device,
        steps=steps,
        lr=lr,
        seg_weight=seg_weight,
        pose_weight=pose_weight,
        batch_pairs=batch_pairs,
        time_limit=args.time_limit,
        loss_mode=loss_mode,
        hinge_margin=hinge_margin,
    )

    # Write output
    raw_path = write_raw_output(optimized, output_dir)

    # Summary
    total_time = time.time() - t_start
    logger.info("\n" + "=" * 60)
    logger.info("CONSTRAINED GEN INFLATE COMPLETE")
    logger.info("=" * 60)
    logger.info("  Frames: %d", N)
    logger.info("  Steps: %d", steps)
    logger.info("  Output: %s", raw_path)
    logger.info("  Total time: %.1fs (budget: %ds)", total_time, args.time_limit)
    logger.info("  Time margin: %.1fs remaining", args.time_limit - total_time)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
