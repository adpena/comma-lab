#!/usr/bin/env python
"""Extract PoseNet targets from ground truth video for supervised TTO.

Run this once during the compress phase to pre-compute targets.
The output file (posenet_targets.bin) is bundled into archive.zip.

Usage:
    uv run python experiments/extract_posenet_targets.py

    # Custom paths:
    uv run python experiments/extract_posenet_targets.py \
        --upstream workspace/upstream/comma_video_compression_challenge \
        --output submissions/robust_current/posenet_targets.bin \
        --device mps
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Extract PoseNet targets for supervised TTO"
    )
    parser.add_argument(
        "--upstream",
        default="workspace/upstream/comma_video_compression_challenge",
        help="Path to upstream challenge root",
    )
    parser.add_argument(
        "--output",
        default="submissions/robust_current/posenet_targets.bin",
        help="Output path for targets file",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device (cpu/cuda/mps)",
    )
    args = parser.parse_args()

    upstream = Path(args.upstream)
    gt_video = upstream / "videos" / "0.mkv"
    posenet_path = upstream / "models" / "posenet.safetensors"

    if not gt_video.exists():
        print(f"ERROR: GT video not found: {gt_video}", file=sys.stderr)
        sys.exit(1)
    if not posenet_path.exists():
        print(f"ERROR: PoseNet model not found: {posenet_path}", file=sys.stderr)
        sys.exit(1)

    from tac.scorer_targets import extract_and_save

    size = extract_and_save(
        gt_video_path=gt_video,
        posenet_path=posenet_path,
        output_path=args.output,
        upstream_dir=str(upstream),
        device=args.device,
    )

    print(f"\nDone. Output: {args.output} ({size} bytes)")
    print(f"Rate impact: {size}/37545489 * 25 = {size/37545489 * 25:.6f} score")
    print(f"\nTo use: set POSENET_TARGETS_ENABLE=1 in config.env "
          f"or copy to submissions/robust_current/")


if __name__ == "__main__":
    main()
