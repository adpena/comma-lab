#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""DALI vs PyAV mask validation (P0 blocker #4).

Decodes GT video frames via both PyAV (CPU) and DALI (GPU), runs SegNet
on each, and compares argmax mask disagreement rate pixel-by-pixel.

PASS: disagreement < 1%
FAIL: disagreement >= 1%

Usage (on Lightning T4):
    PYTHONPATH=src:upstream python experiments/validate_dali_masks.py \
        --video-dir /home/zeus/content/upstream/videos \
        --video-names /home/zeus/content/upstream/public_test_video_names.txt \
        --weights-dir /home/zeus/content/upstream/models \
        --max-frames 100 \
        --device cuda

Usage (local, CPU-only — skips DALI, just validates PyAV path):
    PYTHONPATH=src:upstream python experiments/validate_dali_masks.py \
        --video-dir upstream/videos \
        --video-names upstream/public_test_video_names.txt \
        --weights-dir upstream/models \
        --max-frames 20 \
        --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
def _find_upstream() -> Path | None:
    candidates = [
        Path("/home/zeus/content/upstream"),
        Path(__file__).resolve().parent.parent / "upstream",
    ]
    for p in candidates:
        if (p / "modules.py").exists():
            return p
    return None


def _ensure_upstream_on_path():
    up = _find_upstream()
    if up is not None and str(up) not in sys.path:
        sys.path.insert(0, str(up))
    return up


# ---------------------------------------------------------------------------
# PyAV decoder: returns list of (H, W, 3) uint8 tensors
# ---------------------------------------------------------------------------
def decode_pyav(video_path: str, max_frames: int) -> list[torch.Tensor]:
    """Decode frames via PyAV, matching upstream yuv420_to_rgb."""
    import av
    from frame_utils import yuv420_to_rgb

    fmt = "hevc" if video_path.endswith(".hevc") else None
    container = av.open(video_path, format=fmt)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        rgb = yuv420_to_rgb(frame)  # (H, W, 3) uint8
        frames.append(rgb)
        if len(frames) >= max_frames:
            break
    container.close()
    return frames


# ---------------------------------------------------------------------------
# DALI decoder: returns list of (H, W, 3) uint8 tensors
# ---------------------------------------------------------------------------
def decode_dali(video_path: str, max_frames: int, device_id: int = 0) -> list[torch.Tensor]:
    """Decode frames via NVIDIA DALI GPU pipeline."""
    import nvidia.dali.fn as fn
    from nvidia.dali import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
    from frame_utils import hevc_buffer_mmap, frame_count

    seq_len = 1  # one frame at a time for pixel-exact comparison

    @pipeline_def
    def pipe():
        vid = fn.experimental.inputs.video(
            name="inbuf",
            sequence_length=seq_len,
            device="mixed",
            no_copy=True,
            blocking=False,
            last_sequence_policy="pad",
        )
        return vid

    mv, (mm, f) = hevc_buffer_mmap(video_path)
    n_frames = min(frame_count(video_path), max_frames)
    batch_size = 1

    p = pipe(batch_size=batch_size, num_threads=2, device_id=device_id, prefetch_queue_depth=2)
    p.build()
    p.feed_input("inbuf", [mv])

    it = DALIGenericIterator(
        [p],
        output_map=["video"],
        auto_reset=False,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        last_batch_padded=False,
        prepare_first_batch=False,
    )

    frames = []
    for data in it:
        vid = data[0]["video"]  # (B, seq_len, H, W, C) on GPU
        for b in range(vid.shape[0]):
            for s in range(vid.shape[1]):
                frame = vid[b, s].cpu()  # (H, W, 3) uint8
                frames.append(frame)
                if len(frames) >= n_frames:
                    break
            if len(frames) >= n_frames:
                break
        if len(frames) >= n_frames:
            break

    torch.cuda.synchronize()
    it.reset()
    del it, p
    mv.release()
    mm.close()
    f.close()

    return frames[:n_frames]


# ---------------------------------------------------------------------------
# SegNet mask extraction
# ---------------------------------------------------------------------------
def extract_masks(frames: list[torch.Tensor], segnet: torch.nn.Module, device: str) -> torch.Tensor:
    """Run SegNet on frames, return argmax masks (N, H_seg, W_seg) int64.

    Uses SegNet.preprocess_input to match the scorer preprocessing exactly:
    preprocess_input expects (B, T, C, H, W) with T=1, resizes internally
    to (384, 512). Output: (B, 5, 384, 512) logits -> argmax -> (B, 384, 512).
    """
    masks = []
    batch_size = 16
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i : i + batch_size]
        # Stack and convert: (B, H, W, 3) -> (B, 3, H, W) float32
        batch = torch.stack(batch_frames).float().permute(0, 3, 1, 2).to(device)
        # Use preprocess_input for scorer-consistent preprocessing
        inp = batch.unsqueeze(1)  # (B, 1, C, H, W) — T=1 temporal dim
        seg_in = segnet.preprocess_input(inp)
        with torch.no_grad():
            logits = segnet(seg_in)  # (B, 5, 384, 512)
        mask = logits.argmax(dim=1).cpu()  # (B, 384, 512)
        masks.append(mask)

    return torch.cat(masks, dim=0)  # (N, 384, 512)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DALI vs PyAV mask disagreement validation")
    parser.add_argument("--video-dir", type=str, required=True, help="Directory with GT .hevc videos")
    parser.add_argument("--video-names", type=str, required=True, help="Text file listing video filenames")
    parser.add_argument("--weights-dir", type=str, required=True, help="Directory with posenet.safetensors, segnet.safetensors")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames per video to compare")
    parser.add_argument("--max-videos", type=int, default=3, help="Max videos to test (0=all)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--threshold", type=float, default=0.01, help="Disagreement threshold (default 1%%)")
    args = parser.parse_args()

    upstream = _ensure_upstream_on_path()
    if upstream is None:
        print("FAIL: Cannot find upstream directory (need modules.py)")
        sys.exit(1)

    # Load video names
    video_names_path = Path(args.video_names)
    video_names = [line.strip() for line in video_names_path.read_text().splitlines() if line.strip()]
    if args.max_videos > 0:
        video_names = video_names[: args.max_videos]

    print(f"Videos to validate: {len(video_names)}")
    print(f"Max frames per video: {args.max_frames}")
    print(f"Device: {args.device}")
    print(f"Threshold: {args.threshold * 100:.1f}%")
    print()

    # Load SegNet
    weights_dir = Path(args.weights_dir)
    from modules import SegNet
    from safetensors.torch import load_file

    segnet = SegNet()
    segnet.load_state_dict(load_file(str(weights_dir / "segnet.safetensors")))
    segnet.to(args.device)
    segnet.eval()
    print(f"SegNet loaded from {weights_dir / 'segnet.safetensors'}")

    # CPU-only mode: skip DALI, just validate PyAV path works
    use_dali = args.device == "cuda"
    if not use_dali:
        print("WARNING: CPU mode -- DALI unavailable, validating PyAV path only")
        print()

    video_dir = Path(args.video_dir)
    total_pixels = 0
    total_disagree = 0
    all_rates = []

    for vid_name in video_names:
        vid_path = str(video_dir / vid_name)
        print(f"--- {vid_name} ---")

        # Decode via PyAV
        print("  PyAV decoding...", end=" ", flush=True)
        pyav_frames = decode_pyav(vid_path, args.max_frames)
        print(f"{len(pyav_frames)} frames")

        if not use_dali:
            # CPU-only: just extract masks to verify pipeline works
            print("  Extracting PyAV masks...", end=" ", flush=True)
            pyav_masks = extract_masks(pyav_frames, segnet, args.device)
            print(f"shape {tuple(pyav_masks.shape)}")
            print(f"  Classes present: {sorted(pyav_masks.unique().tolist())}")
            print("  [CPU-only mode: skipping DALI comparison]")
            print()
            continue

        # Decode via DALI
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
        print("  DALI decoding...", end=" ", flush=True)
        dali_frames = decode_dali(vid_path, args.max_frames, device_id=device_id)
        print(f"{len(dali_frames)} frames")

        # Align frame counts
        n = min(len(pyav_frames), len(dali_frames))
        if n == 0:
            print("  WARNING: 0 frames decoded, skipping")
            continue
        pyav_frames = pyav_frames[:n]
        dali_frames = dali_frames[:n]

        # Extract masks
        print("  Extracting PyAV masks...", end=" ", flush=True)
        pyav_masks = extract_masks(pyav_frames, segnet, args.device)
        print(f"shape {tuple(pyav_masks.shape)}")

        print("  Extracting DALI masks...", end=" ", flush=True)
        dali_masks = extract_masks(dali_frames, segnet, args.device)
        print(f"shape {tuple(dali_masks.shape)}")

        # Compare argmax disagreement
        disagree = (pyav_masks != dali_masks).sum().item()
        n_pixels = pyav_masks.numel()
        rate = disagree / n_pixels if n_pixels > 0 else 0.0

        total_pixels += n_pixels
        total_disagree += disagree
        all_rates.append(rate)

        print(f"  Pixels: {n_pixels:,}")
        print(f"  Disagreeing: {disagree:,}")
        print(f"  Rate: {rate * 100:.4f}%")
        print()

    # Final verdict
    print("=" * 60)
    if not use_dali:
        print("RESULT: PyAV pipeline validated (CPU-only mode)")
        print("Re-run on GPU with --device cuda to validate DALI agreement")
        print("=" * 60)
        return

    overall_rate = total_disagree / total_pixels if total_pixels > 0 else 0.0
    print(f"OVERALL DISAGREEMENT: {total_disagree:,} / {total_pixels:,} = {overall_rate * 100:.4f}%")
    print(f"Threshold: {args.threshold * 100:.1f}%")
    print()

    if overall_rate < args.threshold:
        print("PASS: DALI and PyAV masks agree within threshold")
        print("Asymmetric warp training can proceed with DALI decoding.")
    else:
        print("FAIL: DALI and PyAV mask disagreement exceeds threshold")
        print("DO NOT proceed with DALI for training -- decode mismatch will")
        print("contaminate scorer-space loss and invalidate training signal.")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
