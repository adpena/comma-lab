#!/usr/bin/env python3
"""Ego-Motion Pre-Computation — Research Roadmap Item #1.

Validates and exercises the full ego-motion precomputation pipeline:
  1. Extract PoseNet 6-DOF targets from ground truth video (compress time)
  2. Save/load round-trip verification
  3. Demonstrate supervised TTO with pre-computed targets (inflate time)
  4. Council analysis: conditioning vs loss target

The infrastructure already exists in tac. This script:
  - Validates it works end-to-end on real data
  - Measures target statistics for council review
  - Documents the correct invocation for production use

Usage (with GT video + scorer models available):
    python experiments/precompute_ego_motion.py \
        --gt-video upstream/videos/0.mkv \
        --posenet upstream/models/posenet.safetensors \
        --output experiments/posenet_targets.bin \
        --device cpu

Usage (statistics-only on existing targets file):
    python experiments/precompute_ego_motion.py \
        --analyze experiments/posenet_targets.bin

Production integration (compress.sh):
    POSENET_TARGETS_ENABLE=1 bash submissions/robust_current/compress.sh

Production integration (inflate):
    python submissions/robust_current/inflate_postfilter.py \\
        --supervised-tto-steps 10 \\
        --supervised-tto-lr 1e-4 \\
        ...
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def analyze_targets(path: str) -> None:
    """Load and analyze pre-computed PoseNet targets."""
    from tac.scorer_targets import load_posenet_targets

    targets_dict = load_posenet_targets(path)
    if targets_dict is None:
        print(f"ERROR: Could not load targets from {path}", file=sys.stderr)
        sys.exit(1)

    targets = targets_dict["targets"]  # (n_pairs, 6) float32
    n_pairs = targets_dict["n_pairs"]
    n_frames = targets_dict["n_frames"]

    file_bytes = Path(path).stat().st_size

    print(f"\n{'='*60}")
    print(f"Ego-Motion Target Analysis: {path}")
    print(f"{'='*60}")
    print(f"  Pairs:          {n_pairs}")
    print(f"  Frames:         {n_frames}")
    print(f"  Shape:          {targets.shape}")
    print(f"  File size:      {file_bytes:,} bytes ({file_bytes/1024:.1f} KB)")
    print(f"  Rate impact:    ~{file_bytes / (n_frames * 874 * 1164 * 3) * 8:.6f} bpp")
    print()

    # Per-component statistics
    # The 6 components are (tx, ty, tz, rx, ry, rz) — translation + rotation
    labels = ["tx", "ty", "tz", "rx", "ry", "rz"]
    print(f"  {'Component':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'|Range|':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for i, label in enumerate(labels):
        col = targets[:, i]
        print(
            f"  {label:<10} {col.mean().item():>10.6f} {col.std().item():>10.6f} "
            f"{col.min().item():>10.6f} {col.max().item():>10.6f} "
            f"{(col.max() - col.min()).item():>10.6f}"
        )

    print()

    # Inter-pair variance (how much does ego-motion change frame-to-frame?)
    if n_pairs > 1:
        diffs = targets[1:] - targets[:-1]
        print(f"  Inter-pair differences (ego-motion change rate):")
        print(f"  {'Component':<10} {'Mean |d|':>10} {'Max |d|':>10}")
        print(f"  {'-'*10} {'-'*10} {'-'*10}")
        for i, label in enumerate(labels):
            col = diffs[:, i].abs()
            print(f"  {label:<10} {col.mean().item():>10.6f} {col.max().item():>10.6f}")

    # Overall distortion budget
    print(f"\n  Overall target norm (L2): {targets.norm(dim=1).mean().item():.6f}")
    print(f"  If inflate reproduces these exactly: PoseNet distortion = 0.0")
    print(f"  Current best PoseNet distortion: ~0.15 (from proxy eval)")

    # Verify float16 round-trip precision
    targets_f16 = targets.half().float()
    max_err = (targets - targets_f16).abs().max().item()
    mean_err = (targets - targets_f16).abs().mean().item()
    print(f"\n  Float16 quantization error:")
    print(f"    Max: {max_err:.8f}")
    print(f"    Mean: {mean_err:.8f}")
    print(f"    Acceptable: {'YES' if max_err < 0.01 else 'NO — consider float32 storage'}")

    print(f"\n{'='*60}")
    print(f"Council Question: Conditioning vs Loss Target?")
    print(f"{'='*60}")
    print(f"""
  OPTION A — Loss Target (RECOMMENDED, already implemented):
    Store PoseNet(GT)[:6] as regression targets.
    At inflate time, supervised TTO minimizes:
      MSE(PoseNet(postfilter(compressed))[:6], stored_target[:6])
    Pros: No architecture changes. Direct scorer optimization.
    Cons: Requires PoseNet forward pass at inflate time.
    Status: FULLY IMPLEMENTED in tac.tto.supervised_tto()

  OPTION B — MotionPredictor Conditioning:
    Feed pre-computed ego-motion as extra input to the MotionPredictor.
    The renderer receives KNOWN camera motion instead of predicting it.
    Pros: Could improve motion prediction quality during TRAINING.
    Cons: Requires architecture changes. MotionPredictor doesn't exist yet.
    Status: NOT IMPLEMENTED — requires renderer architecture work.

  VERDICT: Option A is the immediate win. The infrastructure exists.
  Option B is a follow-on if we build a learned renderer (Roadmap #15).
  For now, the ego-motion targets serve as supervised TTO targets.

  Production activation:
    compress.sh:  POSENET_TARGETS_ENABLE=1
    inflate:      --supervised-tto-steps 10 --supervised-tto-lr 1e-4
""")


def extract_targets(
    gt_video: str,
    posenet_path: str,
    output: str,
    upstream_dir: str | None = None,
    device: str = "cpu",
) -> str:
    """Extract PoseNet targets from GT video and save to binary."""
    from tac.scorer_targets import extract_and_save

    if upstream_dir is None:
        upstream_dir = str(Path(posenet_path).parent.parent)

    print(f"\nExtracting PoseNet ego-motion targets ...")
    print(f"  GT video:  {gt_video}")
    print(f"  PoseNet:   {posenet_path}")
    print(f"  Upstream:  {upstream_dir}")
    print(f"  Device:    {device}")
    print(f"  Output:    {output}")

    t0 = time.monotonic()
    size = extract_and_save(
        gt_video_path=gt_video,
        posenet_path=posenet_path,
        output_path=output,
        upstream_dir=upstream_dir,
        device=device,
    )
    elapsed = time.monotonic() - t0

    print(f"\nExtraction complete: {size:,} bytes in {elapsed:.1f}s")
    return output


def verify_roundtrip(path: str) -> bool:
    """Verify save/load round-trip integrity."""
    from tac.scorer_targets import load_posenet_targets, save_posenet_targets

    import tempfile

    print(f"\nVerifying round-trip integrity ...")
    original = load_posenet_targets(path)
    if original is None:
        print("ERROR: Could not load original", file=sys.stderr)
        return False

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as f:
        save_posenet_targets(original, f.name)
        reloaded = load_posenet_targets(f.name)

    if reloaded is None:
        print("ERROR: Could not reload", file=sys.stderr)
        return False

    # Compare within float16 tolerance
    max_err = (original["targets"] - reloaded["targets"]).abs().max().item()
    print(f"  Round-trip max error: {max_err:.8f}")
    print(f"  n_pairs match: {original['n_pairs'] == reloaded['n_pairs']}")
    print(f"  n_frames match: {original['n_frames'] == reloaded['n_frames']}")

    ok = max_err < 0.01 and original["n_pairs"] == reloaded["n_pairs"]
    print(f"  Verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Ego-motion pre-computation (Research Roadmap #1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--analyze", type=str, default=None,
        help="Path to existing posenet_targets.bin to analyze (skip extraction)",
    )
    parser.add_argument(
        "--gt-video", type=str, default=None,
        help="Path to ground truth video (e.g., upstream/videos/0.mkv)",
    )
    parser.add_argument(
        "--posenet", type=str, default=None,
        help="Path to posenet.safetensors",
    )
    parser.add_argument(
        "--output", type=str, default="experiments/posenet_targets.bin",
        help="Output path for targets file",
    )
    parser.add_argument(
        "--upstream", type=str, default=None,
        help="Upstream repo directory (for modules.py import)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Computation device (cpu/cuda/mps)",
    )
    parser.add_argument(
        "--verify", action="store_true", default=True,
        help="Run round-trip verification after extraction",
    )

    args = parser.parse_args()

    if args.analyze:
        # Analysis-only mode
        analyze_targets(args.analyze)
        if args.verify:
            verify_roundtrip(args.analyze)
        return

    if not args.gt_video or not args.posenet:
        # Try standard locations
        default_gt = PROJECT_ROOT / "upstream" / "videos" / "0.mkv"
        default_pn = PROJECT_ROOT / "upstream" / "models" / "posenet.safetensors"

        if default_gt.exists() and default_pn.exists():
            args.gt_video = str(default_gt)
            args.posenet = str(default_pn)
            if args.upstream is None:
                args.upstream = str(PROJECT_ROOT / "upstream")
            print(f"Using default paths:")
            print(f"  GT video: {args.gt_video}")
            print(f"  PoseNet:  {args.posenet}")
        else:
            print("ERROR: --gt-video and --posenet are required", file=sys.stderr)
            print("  Or place files at upstream/videos/0.mkv and upstream/models/posenet.safetensors")
            sys.exit(1)

    # Extract
    output = extract_targets(
        gt_video=args.gt_video,
        posenet_path=args.posenet,
        output=args.output,
        upstream_dir=args.upstream,
        device=args.device,
    )

    # Verify
    if args.verify:
        verify_roundtrip(output)

    # Analyze
    analyze_targets(output)


if __name__ == "__main__":
    main()
