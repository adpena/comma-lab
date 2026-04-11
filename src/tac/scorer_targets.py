"""PoseNet target extraction for supervised test-time optimization.

Pre-computes PoseNet outputs on ground truth frame pairs and stores them
for use at inflate time. This enables SUPERVISED TTO: instead of
self-supervised losses (temporal consistency, etc.), we minimize MSE
directly against the KNOWN PoseNet targets that the scorer will compare.

The scorer computes PoseNet distortion as:
    MSE(PoseNet(generated_pair)[:6], PoseNet(original_pair)[:6])

If we store PoseNet(original_pair)[:6] for all 600 pairs, then at inflate
time we can optimize the postfilter to match these exact targets.

Storage: 600 pairs x 6 floats x 2 bytes (float16) = 7,200 bytes raw.
With zlib compression: typically < 5KB.

Usage::

    # At compress time: extract and save targets
    from tac.scorer_targets import extract_posenet_targets, save_posenet_targets
    targets = extract_posenet_targets(gt_frames, posenet, device='cpu')
    save_posenet_targets(targets, 'posenet_targets.bin')

    # At inflate time: load targets for supervised TTO
    from tac.scorer_targets import load_posenet_targets
    targets = load_posenet_targets('posenet_targets.bin')
"""

from __future__ import annotations

import io
import struct
import sys
import zlib
from pathlib import Path

import torch

# Magic bytes for file format identification
_MAGIC = b"PNTG"  # PoseNet TarGets
_VERSION = 1


def extract_posenet_targets(
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    device: str | torch.device = "cpu",
    batch_size: int = 8,
    verbose: bool = True,
) -> dict:
    """Pre-compute PoseNet outputs for all consecutive frame pairs.

    Processes ground truth frames through the frozen PoseNet exactly as
    the scorer does: preprocess_input -> forward -> extract first 6 outputs.

    Args:
        gt_frames: list of (H, W, 3) uint8 tensors (ground truth frames)
        posenet: frozen PoseNet model (with preprocess_input method)
        device: computation device
        batch_size: pairs per forward pass
        verbose: print progress

    Returns:
        dict with:
            'targets': (N_pairs, 6) float32 tensor -- exact PoseNet outputs
            'n_pairs': int -- number of pairs
            'n_frames': int -- total frame count
    """
    device = torch.device(device) if isinstance(device, str) else device
    n_frames = len(gt_frames)
    # Pairs are consecutive non-overlapping: (0,1), (2,3), (4,5), ...
    # This matches the scorer's seq_len=2 pairing
    n_pairs = n_frames // 2

    if verbose:
        print(f"[scorer_targets] Extracting PoseNet targets: {n_frames} frames -> {n_pairs} pairs", file=sys.stderr)

    all_targets = []

    posenet.eval()
    with torch.inference_mode():
        for batch_start in range(0, n_pairs, batch_size):
            batch_end = min(batch_start + batch_size, n_pairs)
            batch_pairs = []

            for pair_idx in range(batch_start, batch_end):
                frame_idx = pair_idx * 2
                f0 = gt_frames[frame_idx]  # (H, W, 3) uint8
                f1 = gt_frames[frame_idx + 1]
                # Stack as (1, 2, H, W, 3) to match scorer input format
                pair = torch.stack([f0, f1]).unsqueeze(0)
                batch_pairs.append(pair)

            # (B, 2, H, W, 3) -> need (B, 2, C, H, W) for preprocess_input
            batch = torch.cat(batch_pairs, dim=0).to(device)
            # Convert to (B, T, C, H, W) float as the scorer does
            batch_float = batch.float().permute(0, 1, 4, 2, 3).contiguous()

            # Run through PoseNet preprocessing + forward (matches evaluate.py)
            preprocessed = posenet.preprocess_input(batch_float)
            output = posenet(preprocessed)

            # Extract first 6 pose outputs (the distortion-relevant ones)
            pose_targets = output["pose"][..., :6].cpu()  # (B, 6)
            all_targets.append(pose_targets)

            if verbose and (batch_end % (batch_size * 10) == 0 or batch_end == n_pairs):
                print(f"  [{batch_end}/{n_pairs} pairs]", file=sys.stderr)

    targets = torch.cat(all_targets, dim=0)  # (n_pairs, 6)
    assert targets.shape == (n_pairs, 6), f"Expected ({n_pairs}, 6), got {targets.shape}"

    if verbose:
        print(
            f"[scorer_targets] Extracted {n_pairs} targets, range [{targets.min():.4f}, {targets.max():.4f}]",
            file=sys.stderr,
        )

    return {
        "targets": targets,
        "n_pairs": n_pairs,
        "n_frames": n_frames,
    }


def save_posenet_targets(targets_dict: dict, path: str | Path) -> int:
    """Save PoseNet targets to a compressed binary file.

    Format:
        4 bytes: magic "PNTG"
        2 bytes: version (uint16)
        4 bytes: n_pairs (uint32)
        4 bytes: n_frames (uint32)
        4 bytes: compressed_size (uint32)
        compressed_size bytes: zlib-compressed float16 tensor data

    Args:
        targets_dict: output from extract_posenet_targets
        path: output file path

    Returns:
        File size in bytes.
    """
    targets = targets_dict["targets"]  # (n_pairs, 6) float32
    n_pairs = targets_dict["n_pairs"]
    n_frames = targets_dict["n_frames"]

    # Convert to float16 for storage (sufficient precision for MSE targets)
    targets_f16 = targets.half()
    raw_bytes = targets_f16.numpy().tobytes()

    # Compress with zlib level 9
    compressed = zlib.compress(raw_bytes, level=9)

    path = Path(path)
    buf = io.BytesIO()
    buf.write(_MAGIC)
    buf.write(struct.pack("<H", _VERSION))
    buf.write(struct.pack("<I", n_pairs))
    buf.write(struct.pack("<I", n_frames))
    buf.write(struct.pack("<I", len(compressed)))
    buf.write(compressed)

    data = buf.getvalue()
    path.write_bytes(data)

    print(
        f"[scorer_targets] Saved {n_pairs} targets to {path}: "
        f"{len(data)} bytes (raw {len(raw_bytes)}, "
        f"compressed {len(compressed)})",
        file=sys.stderr,
    )

    return len(data)


def load_posenet_targets(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> dict | None:
    """Load PoseNet targets from a compressed binary file.

    Args:
        path: path to posenet_targets.bin
        device: target device for the tensor

    Returns:
        dict with 'targets' (n_pairs, 6) float32 tensor, or None if file
        doesn't exist (graceful fallback).
    """
    path = Path(path)
    if not path.exists():
        return None

    data = path.read_bytes()
    buf = io.BytesIO(data)

    magic = buf.read(4)
    if magic != _MAGIC:
        print(f"[scorer_targets] WARNING: invalid magic in {path}, skipping", file=sys.stderr)
        return None

    version = struct.unpack("<H", buf.read(2))[0]
    if version != _VERSION:
        print(f"[scorer_targets] WARNING: unsupported version {version} in {path}", file=sys.stderr)
        return None

    n_pairs = struct.unpack("<I", buf.read(4))[0]
    n_frames = struct.unpack("<I", buf.read(4))[0]
    compressed_size = struct.unpack("<I", buf.read(4))[0]
    compressed = buf.read(compressed_size)

    raw_bytes = zlib.decompress(compressed)
    import numpy as np

    targets_f16 = np.frombuffer(raw_bytes, dtype=np.float16).reshape(n_pairs, 6)
    targets = torch.from_numpy(targets_f16.copy()).float().to(device)

    print(f"[scorer_targets] Loaded {n_pairs} PoseNet targets from {path} ({len(data)} bytes)", file=sys.stderr)

    return {
        "targets": targets,
        "n_pairs": n_pairs,
        "n_frames": n_frames,
    }


def extract_and_save(
    gt_video_path: str | Path,
    posenet_path: str | Path,
    output_path: str | Path,
    upstream_dir: str | Path | None = None,
    device: str = "cpu",
) -> int:
    """Convenience: decode GT video, extract targets, save to file.

    This is the function compress.sh should call.

    Args:
        gt_video_path: path to ground truth video (e.g., videos/0.mkv)
        posenet_path: path to posenet.safetensors
        output_path: where to save posenet_targets.bin
        upstream_dir: upstream repo root (for importing modules.py)
        device: computation device

    Returns:
        File size in bytes.
    """
    from .data import decode_video
    from .scorer import load_scorers

    # Decode GT video
    print(f"[scorer_targets] Decoding GT video: {gt_video_path}", file=sys.stderr)
    gt_frames = decode_video(str(gt_video_path))
    print(f"[scorer_targets] Decoded {len(gt_frames)} GT frames", file=sys.stderr)

    # Load PoseNet (we don't need SegNet)
    segnet_path = Path(posenet_path).parent / "segnet.safetensors"
    posenet, _ = load_scorers(
        posenet_path,
        segnet_path,
        device=device,
        upstream_dir=upstream_dir,
    )

    # Extract and save
    targets = extract_posenet_targets(gt_frames, posenet, device=device)
    return save_posenet_targets(targets, output_path)


if __name__ == "__main__":
    """CLI: python -m tac.scorer_targets --gt-video videos/0.mkv --posenet models/posenet.safetensors --output posenet_targets.bin"""
    import argparse

    parser = argparse.ArgumentParser(description="Extract PoseNet targets for supervised TTO")
    parser.add_argument("--gt-video", required=True, help="Path to ground truth video")
    parser.add_argument("--posenet", required=True, help="Path to posenet.safetensors")
    parser.add_argument("--output", required=True, help="Output path for targets file")
    parser.add_argument("--upstream", default=None, help="Upstream repo dir (for modules.py)")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    args = parser.parse_args()

    if args.upstream is None:
        args.upstream = str(Path(args.posenet).parent.parent)

    size = extract_and_save(
        gt_video_path=args.gt_video,
        posenet_path=args.posenet,
        output_path=args.output,
        upstream_dir=args.upstream,
        device=args.device,
    )
    print(f"Done. Output: {args.output} ({size} bytes)")
