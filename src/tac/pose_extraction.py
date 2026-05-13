"""GT pose extraction utility for FiLM-conditioned renderer.

Extracts PoseNet outputs from GT frame pairs and saves them for use as
FiLM conditioning signals at compress/inflate time. The pose vectors
encode ego-motion between consecutive frames.

Storage cost: 600 pairs x 6 values x 2 bytes (fp16) = 7.2KB.

Usage::

    python -m tac.pose_extraction \
        --upstream upstream/ \
        --output experiments/results/gt_poses.pt \
        --device mps
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def extract_gt_poses(
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    device: str | torch.device = "cpu",
    batch_size: int = 16,
) -> torch.Tensor:
    """Extract GT PoseNet outputs for all consecutive frame pairs.

    These pose vectors serve as FiLM conditioning signals for the renderer.
    Precomputed at compress time, stored in archive (600 x 6 x 2 bytes = 7.2KB fp16).

    Uses non-overlapping pairs: (0,1), (2,3), ..., (1198,1199) = 600 pairs.
    This matches the official scorer's evaluation protocol (seq_len=2).

    Args:
        gt_frames: list of 1200 (H, W, 3) uint8 tensors
        posenet: frozen PoseNet model
        device: computation device
        batch_size: pairs per forward pass

    Returns:
        (600, 6) float tensor of GT pose outputs
    """
    from tac.scorer import extract_gt_pose_targets

    device = torch.device(device) if isinstance(device, str) else device
    return extract_gt_pose_targets(gt_frames, posenet, device, batch_size)


def main() -> None:
    """CLI entry point for GT pose extraction."""
    parser = argparse.ArgumentParser(
        description="Extract GT PoseNet poses for FiLM conditioning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--upstream", type=str, default="upstream/",
        help="Path to upstream challenge repo (contains models/ and videos/)",
    )
    parser.add_argument(
        "--output", type=str, default="experiments/results/gt_poses.pt",
        help="Output path for poses tensor (.pt file, saved as fp16)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Computation device",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Pairs per forward pass",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to GT video (default: upstream/videos/0.mkv)",
    )
    args = parser.parse_args()

    upstream = Path(args.upstream)
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))

    video_path = Path(args.video) if args.video else upstream / "videos" / "0.mkv"
    posenet_path = upstream / "models" / "posenet.safetensors"

    if not video_path.exists():
        print(f"ERROR: GT video not found at {video_path}", file=sys.stderr)
        sys.exit(1)
    if not posenet_path.exists():
        print(f"ERROR: PoseNet model not found at {posenet_path}", file=sys.stderr)
        sys.exit(1)

    # Load PoseNet
    from tac.scorer import load_scorers

    device = torch.device(args.device)
    segnet_path = upstream / "models" / "segnet.safetensors"
    posenet, _ = load_scorers(posenet_path, segnet_path, device=device, upstream_dir=upstream)

    # Load GT frames
    print(f"Loading GT video from {video_path}...")
    import av

    gt_frames: list[torch.Tensor] = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgb24")
            gt_frames.append(torch.from_numpy(arr))

    print(f"Loaded {len(gt_frames)} frames ({gt_frames[0].shape})")

    # Extract poses
    t0 = time.monotonic()
    poses = extract_gt_poses(gt_frames, posenet, device=device, batch_size=args.batch_size)
    elapsed = time.monotonic() - t0

    print(f"Extracted {poses.shape[0]} pose vectors in {elapsed:.1f}s")
    print(f"  Shape: {poses.shape}")
    print(f"  Range: [{poses.min():.4f}, {poses.max():.4f}]")
    print(f"  Mean abs: {poses.abs().mean():.6f}")

    # Save as fp16 (7.2KB)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(poses.half(), output_path)

    size_bytes = output_path.stat().st_size
    print(f"\nSaved to {output_path} ({size_bytes} bytes, fp16)")


if __name__ == "__main__":
    main()
