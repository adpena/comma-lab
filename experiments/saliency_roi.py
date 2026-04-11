#!/usr/bin/env python3
"""Generate PoseNet + SegNet saliency maps — thin wrapper around tac.saliency."""
import sys
sys.path.insert(0, "workspace/upstream/comma_video_compression_challenge")

from tac.saliency import compute_saliency  # noqa: E402

if __name__ == "__main__":
    video_path = "workspace/upstream/comma_video_compression_challenge/videos/0.mkv"
    models_dir = "workspace/upstream/comma_video_compression_challenge/models"
    output_path = "experiments/masks/posenet_saliency.npy"

    print("Computing PoseNet saliency map ...", file=sys.stderr)
    saliency = compute_saliency(video_path, models_dir, output_path, sample_step=20)
    print(f"Done. Shape: {saliency.shape}", file=sys.stderr)
