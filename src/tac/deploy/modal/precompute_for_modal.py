#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Precompute all heavy data locally, save to disk for Modal upload.

Modal should spend 100% of GPU time on training, 0% on data loading.
This script prepares:
1. Decoded compressed frames (tensor, ready to use)
2. Decoded GT frames (tensor, ready to use)
3. Saliency weights (ready to use)
4. Pre-loaded scorer state dicts (avoids LFS pull on Modal)

Usage:
    .venv/bin/python deploy/modal/precompute_for_modal.py
    .venv/bin/modal volume put comma-lab-weights precomputed/ precomputed/ --force
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tac.data import decode_archive, decode_video, load_raw_saliency

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "workspace" / "upstream" / "comma_video_compression_challenge"
OUT = REPO / "experiments" / "precomputed"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Compressed frames
    print("Decoding compressed archive...")
    comp_frames = decode_archive(str(REPO / "submissions" / "robust_current" / "archive.zip"))
    comp_tensor = torch.stack(comp_frames)  # (1200, H, W, 3) uint8
    torch.save(comp_tensor, OUT / "comp_frames.pt")
    print(f"  Saved comp_frames.pt: {comp_tensor.shape} ({(OUT / 'comp_frames.pt').stat().st_size / 1e6:.1f} MB)")

    # 2. GT frames
    print("Decoding GT video...")
    gt_frames = decode_video(str(UPSTREAM / "videos" / "0.mkv"))
    gt_tensor = torch.stack(gt_frames)  # (1200, H, W, 3) uint8
    torch.save(gt_tensor, OUT / "gt_frames.pt")
    print(f"  Saved gt_frames.pt: {gt_tensor.shape} ({(OUT / 'gt_frames.pt').stat().st_size / 1e6:.1f} MB)")

    # 3. Saliency
    saliency_path = REPO / "experiments" / "masks" / "posenet_saliency.npy"
    if saliency_path.exists():
        sal = load_raw_saliency(str(saliency_path))
        torch.save(sal, OUT / "saliency.pt")
        print(f"  Saved saliency.pt: {sal.shape}")

    # 4. Scorer state dicts (avoid LFS pull on Modal)
    from safetensors.torch import load_file
    posenet_sd = load_file(str(UPSTREAM / "models" / "posenet.safetensors"))
    segnet_sd = load_file(str(UPSTREAM / "models" / "segnet.safetensors"))
    torch.save({"posenet": posenet_sd, "segnet": segnet_sd}, OUT / "scorer_weights.pt")
    print(f"  Saved scorer_weights.pt ({(OUT / 'scorer_weights.pt').stat().st_size / 1e6:.1f} MB)")

    print(f"\nAll precomputed data in {OUT}/")
    print(f"Total: {sum(f.stat().st_size for f in OUT.glob('*.pt')) / 1e6:.1f} MB")
    print("\nUpload to Modal:")
    print(f"  .venv/bin/modal volume put comma-lab-weights {OUT}/ precomputed/ --force")


if __name__ == "__main__":
    main()
