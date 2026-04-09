#!/usr/bin/env python
"""Canonical inflate path matching baseline_fast/inflate.py exactly.

Uses PyAV → yuv420_to_rgb (BT.601 limited range) → torch bicubic upscale.
No USM, no ffmpeg subprocess, no PIL. This matches the evaluator's expectations.
"""
import sys
from pathlib import Path

# frame_utils is at the upstream repo root, added to sys.path at runtime
import av, torch
import torch.nn.functional as F


def yuv420_to_rgb(frame) -> torch.Tensor:
    """BT.601 limited range YUV420 to RGB, matching frame_utils.py exactly."""
    import numpy as np
    H, W = frame.height, frame.width
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(H, frame.planes[0].line_size)[:, :W]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(H // 2, frame.planes[1].line_size)[:, :W // 2]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(H // 2, frame.planes[2].line_size)[:, :W // 2]

    y_t = torch.from_numpy(y.copy()).float()
    u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)

    u_up = F.interpolate(u_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()
    v_up = F.interpolate(v_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()

    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)

    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)


def inflate_video(video_path: str, dst: str, target_w: int = 1164, target_h: int = 874) -> int:
    """Decode MKV, apply BT.601 YUV→RGB, torch bicubic upscale, write raw RGB."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    n = 0
    with open(dst, 'wb') as f:
        for frame in container.decode(stream):
            t = yuv420_to_rgb(frame)  # (H, W, 3)
            H, W, _ = t.shape
            if H != target_h or W != target_w:
                x = t.permute(2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)
                x = F.interpolate(x, size=(target_h, target_w), mode='bicubic', align_corners=False)
                t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
            f.write(t.contiguous().numpy().tobytes())
            n += 1
    container.close()
    return n


if __name__ == "__main__":
    archive_dir = sys.argv[1]
    inflated_dir = sys.argv[2]
    video_names_file = sys.argv[3]

    archive_dir = Path(archive_dir)
    inflated_dir = Path(inflated_dir)
    inflated_dir.mkdir(parents=True, exist_ok=True)

    for line in Path(video_names_file).read_text().splitlines():
        rel = line.strip()
        if not rel:
            continue
        stem = rel.rsplit(".", 1)[0]
        mkv_path = archive_dir / f"{stem}.mkv"
        out_path = inflated_dir / f"{stem}.raw"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Inflating {mkv_path} -> {out_path}", file=sys.stderr)
        n = inflate_video(str(mkv_path), str(out_path))
        print(f"  {n} frames", file=sys.stderr)
