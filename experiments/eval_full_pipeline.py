#!/usr/bin/env python3
"""Full pipeline evaluation: archive → inflate with post-filter → official scorer.

Matches the exact scoring pipeline to validate proxy improvements.
"""
import os, sys, math, shutil, tempfile, zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import av
import numpy as np

REPO = Path(__file__).parent.parent
UPSTREAM = REPO / "workspace" / "upstream" / "comma_video_compression_challenge"
ARCHIVE_ZIP = REPO / "submissions" / "robust_current" / "archive.zip"
GT_DIR = UPSTREAM / "videos"
MODELS_DIR = UPSTREAM / "models"
VIDEO_NAMES = UPSTREAM / "public_test_video_names.txt"

# Add upstream to path for official modules
sys.path.insert(0, str(UPSTREAM))


# ── Post-filter (matches inflate_postfilter.py exactly) ──
class PostFilter(nn.Module):
    def __init__(self, hidden=16, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)


def load_postfilter_int8(path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    meta = state.get("__meta__", {"hidden": 64, "kernel": 3})
    if isinstance(meta, dict):
        hidden = meta.get("hidden", 64)
        kernel = meta.get("kernel", 3)
    else:
        hidden, kernel = 64, 3

    model = PostFilter(hidden=hidden, kernel=kernel)
    float_state = {}
    seen = set()
    for raw_key in state.keys():
        if raw_key == "__meta__":
            continue
        if raw_key.endswith(".q") or raw_key.endswith(".s"):
            base = raw_key[:-2]
            if base in seen:
                continue
            seen.add(base)
            q = state[base + ".q"].float()
            s = state[base + ".s"]
            if s.ndim == 0:
                float_state[base] = q * s
            else:
                float_state[base] = q * s.view(-1, *([1] * (q.ndim - 1)))
        else:
            float_state[raw_key] = state[raw_key].float()

    model.load_state_dict(float_state)
    return model.eval().to(device)


def yuv420_to_rgb(frame):
    """Convert PyAV YUV420p frame to RGB uint8 tensor."""
    arr = frame.to_ndarray(format='rgb24')
    return torch.from_numpy(arr)


def inflate_video(video_path, dst_path, model, target_w=1164, target_h=874, device="cpu"):
    """Inflate one video with post-filter."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    n = 0
    with open(dst_path, 'wb') as f:
        for frame in container.decode(stream):
            t = yuv420_to_rgb(frame)
            H, W, _ = t.shape
            x = t.permute(2, 0, 1).unsqueeze(0).float()
            if H != target_h or W != target_w:
                x = F.interpolate(x, size=(target_h, target_w), mode='bicubic', align_corners=False)
                x = x.clamp(0, 255)
            with torch.no_grad():
                x = model(x.to(device))
            out = x.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).cpu()
            f.write(out.contiguous().numpy().tobytes())
            n += 1
    container.close()
    return n


def main():
    postfilter_path = sys.argv[1] if len(sys.argv) > 1 else str(
        REPO / "experiments" / "postfilter_weights" / "postfilter_standard_h64_long2500_best_int8.pt"
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Post-filter: {postfilter_path}")

    # Load post-filter
    model = load_postfilter_int8(postfilter_path, device=device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count} params")

    # Create temp dir for inflated output
    tmpdir = tempfile.mkdtemp(prefix="pact_eval_")
    inflated_dir = Path(tmpdir) / "inflated"
    inflated_dir.mkdir()
    archive_dir = Path(tmpdir) / "archive"
    archive_dir.mkdir()

    try:
        # Extract archive
        print(f"Extracting {ARCHIVE_ZIP}...")
        with zipfile.ZipFile(ARCHIVE_ZIP) as z:
            z.extractall(archive_dir)

        # Read video names
        video_names = [l.strip() for l in VIDEO_NAMES.read_text().splitlines() if l.strip()]
        print(f"Videos to inflate: {len(video_names)}")

        # Inflate each video
        for vname in video_names:
            stem = vname.rsplit(".", 1)[0]
            mkv_path = archive_dir / f"{stem}.mkv"
            out_path = inflated_dir / f"{stem}.raw"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Inflating {stem}...", end=" ", flush=True)
            n = inflate_video(mkv_path, out_path, model, device=device)
            print(f"{n} frames")

        # Compute rate
        compressed_size = ARCHIVE_ZIP.stat().st_size
        uncompressed_size = sum(f.stat().st_size for f in GT_DIR.rglob('*') if f.is_file())
        rate = compressed_size / uncompressed_size
        print(f"\nRate: {compressed_size:,} / {uncompressed_size:,} = {rate:.8f}")

        # Run official scorer
        print("\nRunning official scorer...")
        from modules import DistortionNet, segnet_sd_path, posenet_sd_path
        from frame_utils import TensorVideoDataset, AVVideoDataset, camera_size, seq_len

        distortion_net = DistortionNet().eval().to(device=device)
        distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

        ds_gt = AVVideoDataset(video_names, data_dir=GT_DIR, batch_size=16, device=device, num_threads=2, seed=1234, prefetch_queue_depth=4)
        ds_gt.prepare_data()
        dl_gt = torch.utils.data.DataLoader(ds_gt, batch_size=None, num_workers=0)

        ds_comp = TensorVideoDataset(video_names, data_dir=inflated_dir, batch_size=16, device=device, num_threads=2, seed=1234, prefetch_queue_depth=4)
        ds_comp.prepare_data()
        dl_comp = torch.utils.data.DataLoader(ds_comp, batch_size=None, num_workers=0)

        posenet_dists = torch.zeros([], device=device)
        segnet_dists = torch.zeros([], device=device)
        batch_sizes = torch.zeros([], device=device)

        with torch.inference_mode():
            for (_, _, batch_gt), (_, _, batch_comp) in zip(dl_gt, dl_comp):
                batch_gt = batch_gt.to(device)
                batch_comp = batch_comp.to(device)
                posenet_dist, segnet_dist = distortion_net.compute_distortion(batch_gt, batch_comp)
                posenet_dists += posenet_dist.sum()
                segnet_dists += segnet_dist.sum()
                batch_sizes += batch_gt.shape[0]

        posenet_dist = (posenet_dists / batch_sizes).item()
        segnet_dist = (segnet_dists / batch_sizes).item()
        score = 100 * segnet_dist + math.sqrt(posenet_dist * 10) + 25 * rate

        print(f"\n{'='*60}")
        print(f"  PoseNet Distortion: {posenet_dist:.8f}")
        print(f"  SegNet Distortion:  {segnet_dist:.8f}")
        print(f"  Rate:               {rate:.8f}")
        print(f"  FINAL SCORE:        {score:.4f}")
        print(f"{'='*60}")
        print(f"\nCompare to promoted: 1.727")
        print(f"Delta: {score - 1.727:+.4f}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
