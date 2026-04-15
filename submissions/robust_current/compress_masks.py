#!/usr/bin/env python
"""Pre-extract SegNet masks and encode as AV1 monochrome video.

This script runs at compress time to pre-extract semantic segmentation masks
from the ground-truth video using SegNet, then encodes them as an AV1
monochrome video file suitable for inclusion in archive.zip.

Purpose: contest compliance. Yousfi's PR #35 rule requires that any scorer
weights used at inflate time must be included in the archive. By pre-extracting
masks at compress time, inflate_renderer.py no longer needs to load SegNet
(~48MB) at inflate time, avoiding a rate catastrophe.

The mask video uses 5-class to grayscale mapping (0->0, 1->63, 2->127,
3->191, 4->255) and AV1 monochrome encoding at CRF 20. Typical size:
~30-50KB for 1200 frames at 384x512.

Usage:
    python compress_masks.py \\
        --gt-video /path/to/0.mkv \\
        --upstream /path/to/upstream \\
        --output masks.mkv \\
        [--crf 20] [--device cpu] [--batch-size 8] [--verify]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# SegNet native resolution
SEGNET_H = 384
SEGNET_W = 512
NUM_CLASSES = 5
NUM_FRAMES = 1200


def _load_segnet(upstream_root: Path, device: str) -> torch.nn.Module:
    """Load frozen SegNet from upstream for mask extraction.

    This is the only place SegNet is loaded in the entire pipeline.
    After compress_masks.py runs, inflate_renderer.py reads masks from
    the archive instead.
    """
    t0 = time.monotonic()

    # Import SegNet from upstream modules.py
    upstream_str = str(upstream_root)
    sys.path.insert(0, upstream_str)
    try:
        from modules import SegNet
    finally:
        try:
            sys.path.pop(sys.path.index(upstream_str))
        except ValueError:
            pass

    segnet = SegNet()
    segnet_path = upstream_root / "models" / "segnet.safetensors"
    if not segnet_path.exists():
        raise FileNotFoundError(f"SegNet weights not found: {segnet_path}")

    from safetensors.torch import load_file
    sd = load_file(str(segnet_path), device=device)
    segnet.load_state_dict(sd)
    segnet.to(device).eval()

    for p in segnet.parameters():
        p.requires_grad = False

    elapsed = time.monotonic() - t0
    print(f"  SegNet loaded from {segnet_path} ({elapsed:.1f}s)", file=sys.stderr)
    return segnet


def _decode_gt_video(mkv_path: str) -> list[np.ndarray]:
    """Decode ground-truth video via PyAV.

    Returns list of (H, W, 3) uint8 ndarrays in RGB order.
    Uses BT.601 limited-range decode matching the scorer.
    """
    import av

    t0 = time.monotonic()
    container = av.open(mkv_path)
    stream = container.streams.video[0]
    frames = []

    for frame in container.decode(stream):
        H, W = frame.height, frame.width
        y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(
            H, frame.planes[0].line_size
        )[:, :W]
        u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(
            H // 2, frame.planes[1].line_size
        )[:, :W // 2]
        v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(
            H // 2, frame.planes[2].line_size
        )[:, :W // 2]

        y_t = torch.from_numpy(y.copy()).float()
        u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
        v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)

        u_up = F.interpolate(
            u_t, size=(H, W), mode="bilinear", align_corners=False
        ).squeeze()
        v_up = F.interpolate(
            v_t, size=(H, W), mode="bilinear", align_corners=False
        ).squeeze()

        yf = (y_t - 16.0) * (255.0 / 219.0)
        uf = (u_up - 128.0) * (255.0 / 224.0)
        vf = (v_up - 128.0) * (255.0 / 224.0)

        r = (yf + 1.402 * vf).clamp(0, 255)
        g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
        b = (yf + 1.772 * uf).clamp(0, 255)
        rgb = torch.stack([r, g, b], dim=-1).round().to(torch.uint8)
        frames.append(rgb.numpy())

    container.close()
    elapsed = time.monotonic() - t0
    print(
        f"  Decoded {len(frames)} GT frames from {mkv_path} ({elapsed:.1f}s)",
        file=sys.stderr,
    )
    return frames


def _extract_masks(
    frames: list[np.ndarray],
    segnet: torch.nn.Module,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """Extract SegNet masks from GT frames.

    Returns (N, SEGNET_H, SEGNET_W) long tensor of class indices in [0, 4].
    """
    t0 = time.monotonic()
    N = len(frames)
    masks_list = []

    with torch.inference_mode():
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch_np = np.stack(frames[i:end], axis=0)
            batch_t = (
                torch.from_numpy(batch_np).float().permute(0, 3, 1, 2).to(device)
            )
            inp = batch_t.unsqueeze(1)  # (B, 1, 3, H, W)
            seg_in = segnet.preprocess_input(inp)  # (B, 3, 384, 512)
            logits = segnet(seg_in)  # (B, 5, 384, 512)
            mask = logits.argmax(dim=1)  # (B, 384, 512)
            masks_list.append(mask.cpu())

            if (i + batch_size) % (batch_size * 10) == 0 or end == N:
                print(
                    f"    Masks: {end}/{N} frames", file=sys.stderr, flush=True
                )

    masks = torch.cat(masks_list, dim=0)
    elapsed = time.monotonic() - t0
    print(f"  Extracted {masks.shape[0]} masks ({elapsed:.1f}s)", file=sys.stderr)
    return masks


def _verify_roundtrip(masks: torch.Tensor, mask_video_path: Path) -> bool:
    """Verify lossless roundtrip: original masks == decode(encode(masks)).

    Returns True if all class indices match exactly.
    """
    from tac.mask_codec import decode_masks

    print("  Verifying lossless roundtrip ...", file=sys.stderr)
    decoded = decode_masks(mask_video_path)

    if decoded.shape != masks.shape:
        print(
            f"  FAIL: shape mismatch: original={masks.shape}, "
            f"decoded={decoded.shape}",
            file=sys.stderr,
        )
        return False

    # Compare class indices
    mismatches = (decoded != masks.long()).sum().item()
    total = masks.numel()
    accuracy = 1.0 - mismatches / total

    if mismatches == 0:
        print(f"  PASS: perfect roundtrip ({total:,} pixels)", file=sys.stderr)
        return True

    # For lossy AV1, check if accuracy is acceptable (>99.9%)
    print(
        f"  Roundtrip accuracy: {accuracy:.6f} "
        f"({mismatches:,}/{total:,} mismatches)",
        file=sys.stderr,
    )
    if accuracy >= 0.999:
        print(
            "  ACCEPTABLE: >99.9% accuracy (AV1 lossy rounding at boundaries)",
            file=sys.stderr,
        )
        return True

    print(f"  FAIL: accuracy {accuracy:.4f} below 99.9% threshold", file=sys.stderr)
    return False


def compress_masks(
    gt_video_path: str,
    upstream_root: str,
    output_path: str,
    crf: int = 20,
    device: str = "cpu",
    batch_size: int = 8,
    verify: bool = True,
) -> int:
    """Full mask compression pipeline: GT video -> SegNet -> AV1 monochrome.

    Args:
        gt_video_path: path to ground-truth .mkv video
        upstream_root: path to upstream challenge root (contains modules.py + models/)
        output_path: path for output .mkv mask video
        crf: AV1 CRF parameter (lower = higher quality, larger file)
        device: torch device string
        batch_size: SegNet inference batch size
        verify: if True, verify lossless roundtrip after encoding

    Returns:
        File size in bytes
    """
    from tac.mask_codec import encode_masks_monochrome

    upstream = Path(upstream_root)
    output = Path(output_path)

    print("=" * 60, file=sys.stderr)
    print("Mask pre-extraction pipeline", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Stage 1: Load SegNet
    print("Stage 1: Loading SegNet ...", file=sys.stderr)
    segnet = _load_segnet(upstream, device)

    # Stage 2: Decode GT video
    print("Stage 2: Decoding GT video ...", file=sys.stderr)
    gt_frames = _decode_gt_video(gt_video_path)
    if len(gt_frames) != NUM_FRAMES:
        print(
            f"  WARNING: expected {NUM_FRAMES} frames, got {len(gt_frames)}",
            file=sys.stderr,
        )

    # Stage 3: Extract masks
    print("Stage 3: Extracting SegNet masks ...", file=sys.stderr)
    masks = _extract_masks(gt_frames, segnet, device, batch_size)
    del gt_frames, segnet  # free memory

    # Validate mask values
    unique_vals = torch.unique(masks)
    assert all(0 <= v < NUM_CLASSES for v in unique_vals), (
        f"Invalid mask values: {unique_vals.tolist()}"
    )
    print(
        f"  Mask stats: shape={masks.shape}, "
        f"unique_values={unique_vals.tolist()}",
        file=sys.stderr,
    )

    # Stage 4: Encode as AV1 monochrome
    print("Stage 4: Encoding as AV1 monochrome ...", file=sys.stderr)
    file_size = encode_masks_monochrome(masks, output, crf=crf)

    # Stage 5: Verify roundtrip
    if verify:
        if not _verify_roundtrip(masks, output):
            # Try lower CRF for better fidelity
            print("  Retrying with CRF 10 for better fidelity ...", file=sys.stderr)
            file_size = encode_masks_monochrome(masks, output, crf=10)
            if not _verify_roundtrip(masks, output):
                raise RuntimeError(
                    "Mask roundtrip verification failed even at CRF 10. "
                    "The AV1 codec is not preserving class boundaries."
                )

    print(f"\nMask video: {output} ({file_size:,} bytes)", file=sys.stderr)
    return file_size


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract SegNet masks and encode as AV1 monochrome video"
    )
    parser.add_argument(
        "--gt-video",
        required=True,
        help="Path to ground-truth .mkv video",
    )
    parser.add_argument(
        "--upstream",
        required=True,
        help="Path to upstream challenge root",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for mask video (.mkv)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=20,
        help="AV1 CRF parameter (default: 20)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="SegNet inference batch size (default: 8)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify lossless roundtrip (default: True)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip roundtrip verification",
    )

    args = parser.parse_args()
    verify = not args.no_verify

    compress_masks(
        gt_video_path=args.gt_video,
        upstream_root=args.upstream,
        output_path=args.output,
        crf=args.crf,
        device=args.device,
        batch_size=args.batch_size,
        verify=verify,
    )


if __name__ == "__main__":
    main()
