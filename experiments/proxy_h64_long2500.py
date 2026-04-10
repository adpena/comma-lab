#!/usr/bin/env python3
"""Proxy-score checkpoints using hardened metrics matching official scorer.

Uses hard argmax SegNet disagreement + PoseNet MSE + rate term.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tac.architectures import build_postfilter
from tac.quantization import load_int8
from tac.data import decode_video, decode_archive
from tac.scorer import load_scorers, detect_device
from tac.evaluate import proxy_score

REPO = Path(__file__).parent.parent
UPSTREAM = REPO / "workspace" / "upstream" / "comma_video_compression_challenge"
ARCHIVE_ZIP = REPO / "submissions" / "robust_current" / "archive.zip"
GT_VIDEO = UPSTREAM / "videos" / "0.mkv"
MODELS_DIR = UPSTREAM / "models"

# Compute rate from archive size
GT_DIR = UPSTREAM / "videos"
archive_size = ARCHIVE_ZIP.stat().st_size
uncompressed_size = sum(f.stat().st_size for f in GT_DIR.rglob('*') if f.is_file())
rate = archive_size / uncompressed_size

# Default: evaluate the saliency-fixed h64 long2500 checkpoint
int8_path = sys.argv[1] if len(sys.argv) > 1 else str(
    REPO / "experiments" / "postfilter_weights" / "postfilter_standard_h64_long2500_best_int8.pt"
)

device = detect_device()
print(f"Device: {device}")
print(f"Rate: {archive_size:,} / {uncompressed_size:,} = {rate:.8f}")

# Build model and load int8
model = build_postfilter("standard", hidden=64, kernel=3)
load_int8(int8_path, model, device=str(device))
print(f"Loaded int8 from {int8_path}")

# Decode data — use the SUBMISSION archive (matches what the scorer sees)
print("Decoding submission archive...")
comp_frames = decode_archive(str(ARCHIVE_ZIP))
print(f"{len(comp_frames)} compressed frames")

print("Decoding GT video...")
gt_frames = decode_video(str(GT_VIDEO))
print(f"{len(gt_frames)} GT frames")

# Load scorers
print("Loading scorers...")
posenet, segnet = load_scorers(
    MODELS_DIR / "posenet.safetensors",
    MODELS_DIR / "segnet.safetensors",
    device=device,
    upstream_dir=str(UPSTREAM),
)

# Score with hardened metrics
print("Running hardened proxy evaluation (argmax SegNet + MSE PoseNet + rate)...")
result = proxy_score(model, comp_frames, gt_frames, posenet, segnet, device=device, rate=rate)
print(f"\n{'='*60}")
print(f"  HARDENED PROXY SCORE: {result['score']:.4f}")
print(f"  PoseNet distortion:  {result['pose']:.8f}")
print(f"  SegNet distortion:   {result['seg']:.8f}")
print(f"  Rate:                {result['rate']:.8f}")
print(f"  Rate contribution:   {25 * result['rate']:.4f}")
print(f"  N samples:           {result['n_pairs']}")
print(f"{'='*60}")
print(f"\nCompare to promoted floor: 1.727")
print(f"Delta: {result['score'] - 1.727:+.4f}")
