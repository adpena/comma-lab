#!/usr/bin/env python3
"""Constrained generation compress: extract masks + pose targets from GT video.

Runs the frozen SegNet on GT frames to extract segmentation masks, and the
frozen PoseNet on GT frame pairs to extract expected pose targets. These
become the constraints for the inflate-time gradient descent.

The archive is extremely small (~8KB):
    - masks.bin: LZMA-compressed uint8 masks (~239 bytes for 1200 frames)
    - pose_targets.bin: float16 pose targets (~7KB for 600 pairs)
    - seed.bin: noise seed (8 bytes)
    - meta.json: metadata for reproducibility

Usage:
    python compress.py --upstream-root <path> --device <cuda|mps|cpu>
"""
from __future__ import annotations

import argparse
import json
import logging
import lzma
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Constrained gen compress")
    p.add_argument("--upstream-root", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--video-names-file", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--n-frames", type=int, default=1200)
    return p.parse_args()


def decode_gt_video(video_path: Path, n_frames: int, target_h: int, target_w: int):
    """Decode GT video and resize to scorer resolution.

    Returns list of (H, W, 3) uint8 tensors.
    """
    import av

    frames = []
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    for frame in container.decode(video=0):
        if len(frames) >= n_frames:
            break
        img = frame.to_ndarray(format="rgb24")  # (H_orig, W_orig, 3)
        t = torch.from_numpy(img).float()
        # Resize to scorer resolution
        t_chw = t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        t_resized = F.interpolate(
            t_chw, size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        )
        t_hwc = t_resized.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
        frames.append(t_hwc)

    container.close()
    return frames


def extract_masks(frames: list[torch.Tensor], segnet, device: torch.device, batch_size: int = 32):
    """Run SegNet on frames to extract argmax masks.

    Args:
        frames: list of (H, W, 3) uint8 tensors.
        segnet: frozen SegNet model.
        device: computation device.
        batch_size: frames per batch.

    Returns:
        (N, H, W) long tensor of class indices.
    """
    N = len(frames)
    all_masks = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = torch.stack(frames[start:end]).to(device).float()  # (B, H, W, 3)
        batch_chw = batch.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)

        with torch.no_grad():
            logits = segnet(batch_chw)  # (B, C, H', W')
            masks = logits.argmax(dim=1)  # (B, H', W')

        all_masks.append(masks.cpu())

    return torch.cat(all_masks, dim=0)  # (N, H', W')


def extract_pose_targets(
    frames: list[torch.Tensor],
    posenet,
    device: torch.device,
    batch_size: int = 16,
):
    """Run PoseNet on consecutive frame pairs to extract pose targets.

    Args:
        frames: list of (H, W, 3) uint8 tensors, length N.
        posenet: frozen PoseNet model.
        device: computation device.
        batch_size: pairs per batch.

    Returns:
        (P, 6) float tensor of pose targets, P = N // 2.
    """
    N = len(frames)
    P = N // 2
    all_poses = []

    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)
        B = end - start

        pair_list = []
        for k in range(start, end):
            f0 = frames[2 * k].float()       # (H, W, 3)
            f1 = frames[2 * k + 1].float()   # (H, W, 3)
            # Stack as (2, H, W, 3) then convert to (2, 3, H, W)
            pair_chw = torch.stack([f0, f1]).permute(0, 3, 1, 2)  # (2, 3, H, W)
            pair_list.append(pair_chw)

        pairs = torch.stack(pair_list).to(device)  # (B, 2, 3, H, W)

        with torch.no_grad():
            posenet_in = posenet.preprocess_input(pairs)
            posenet_out = posenet(posenet_in)
            pose = posenet_out["pose"][..., :6]  # (B, 6)

        all_poses.append(pose.cpu())

    return torch.cat(all_poses, dim=0)  # (P, 6)


def build_archive(
    masks: torch.Tensor,
    expected_pose: torch.Tensor,
    noise_seed: int,
    output_dir: Path,
) -> dict:
    """Write the constrained-gen archive files.

    Returns metadata dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Masks: uint8 + LZMA
    masks_np = masks.cpu().numpy().astype(np.uint8)
    masks_bytes = masks_np.tobytes()
    shape_header = struct.pack("<III", *masks_np.shape)
    compressed = lzma.compress(masks_bytes, preset=9)
    (output_dir / "masks.bin").write_bytes(shape_header + compressed)

    # 2. Pose targets: float16
    pose_np = expected_pose.cpu().to(torch.float16).numpy()
    pose_header = struct.pack("<II", *pose_np.shape)
    (output_dir / "pose_targets.bin").write_bytes(pose_header + pose_np.tobytes())

    # 3. Seed
    (output_dir / "seed.bin").write_bytes(struct.pack("<Q", noise_seed))

    # Metadata
    masks_size = len(shape_header + compressed)
    pose_size = len(pose_header + pose_np.tobytes())
    total = masks_size + pose_size + 8

    meta = {
        "num_frames": int(masks.shape[0]),
        "mask_shape": list(masks.shape),
        "pose_shape": list(expected_pose.shape),
        "noise_seed": noise_seed,
        "compressed_masks_bytes": masks_size,
        "pose_bytes": pose_size,
        "total_bytes": total,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    logger.info("Archive written to %s", output_dir)
    logger.info("  masks:  %s bytes", f"{masks_size:,}")
    logger.info("  poses:  %s bytes", f"{pose_size:,}")
    logger.info("  seed:   8 bytes")
    logger.info("  TOTAL:  %s bytes (%.1f KB)", f"{total:,}", total / 1024)

    return meta


def main():
    args = parse_args()

    upstream = Path(args.upstream_root)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    logger.info("Constrained Generation Compress")
    logger.info("  upstream: %s", upstream)
    logger.info("  device: %s", device)
    logger.info("  seed: %d", args.seed)

    # Load scorers
    # No differentiable patch needed -- compress only uses inference
    # (torch.no_grad), not gradient optimization.
    logger.info("Loading scorers...")
    t0 = time.monotonic()
    try:
        from tac.scorer import load_differentiable_scorers
        posenet, segnet = load_differentiable_scorers(upstream, device=str(device))
    except ImportError:
        sys.path.insert(0, str(upstream))
        from modules import PoseNet, SegNet
        from safetensors.torch import load_file

        posenet = PoseNet().eval()
        posenet.load_state_dict(
            load_file(str(upstream / "models" / "posenet.safetensors"), device="cpu")
        )
        segnet = SegNet().eval()
        segnet.load_state_dict(
            load_file(str(upstream / "models" / "segnet.safetensors"), device="cpu")
        )
        posenet, segnet = posenet.to(device), segnet.to(device)
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
    logger.info("  Scorers loaded in %.1fs", time.monotonic() - t0)

    # Decode GT video
    video_names_file = Path(args.video_names_file)
    video_names = [v.strip() for v in video_names_file.read_text().strip().splitlines() if v.strip()]

    for rel in video_names:
        video_path = upstream / "videos" / rel
        if not video_path.exists():
            logger.error("Video not found: %s", video_path)
            sys.exit(1)

        logger.info("Decoding %s (%d frames)...", video_path, args.n_frames)
        t0 = time.monotonic()
        # Use scorer resolution (384x512) since that's what SegNet/PoseNet operate on
        frames = decode_gt_video(video_path, args.n_frames, target_h=384, target_w=512)
        # Ensure even count
        if len(frames) % 2:
            frames = frames[:-1]
        logger.info("  Decoded %d frames in %.1fs", len(frames), time.monotonic() - t0)

        # Extract masks
        logger.info("Extracting SegNet masks...")
        t0 = time.monotonic()
        masks = extract_masks(frames, segnet, device)
        logger.info("  Masks extracted in %.1fs: %s", time.monotonic() - t0, list(masks.shape))

        # Extract pose targets
        logger.info("Extracting PoseNet targets...")
        t0 = time.monotonic()
        pose_targets = extract_pose_targets(frames, posenet, device)
        logger.info("  Pose targets extracted in %.1fs: %s",
                     time.monotonic() - t0, list(pose_targets.shape))

        # Build archive
        logger.info("Building archive...")
        meta = build_archive(masks, pose_targets, args.seed, output_dir)

        # Verify: rate calculation
        # Official scorer: rate = archive.zip size / sum(uncompressed file sizes)
        # The uncompressed dir contains raw video files (MKV), not raw pixels.
        # Here we report both the official denominator (MKV file size) and the
        # raw pixel denominator for reference.
        mkv_size = video_path.stat().st_size
        rate_official = meta["total_bytes"] / mkv_size
        raw_bytes = 3 * 1164 * 874 * len(frames)
        rate_raw = meta["total_bytes"] / raw_bytes
        logger.info("  Rate (official, vs MKV): %.6f (archive %s / MKV %s)",
                     rate_official, f"{meta['total_bytes']:,}", f"{mkv_size:,}")
        logger.info("  Rate (vs raw pixels): %.6f (archive %s / raw %s)",
                     rate_raw, f"{meta['total_bytes']:,}", f"{raw_bytes:,}")
        logger.info("  Rate contribution to score: 25 * %.6f = %.4f",
                     rate_official, 25 * rate_official)

    logger.info("Compress complete.")


if __name__ == "__main__":
    main()
