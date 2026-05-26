#!/usr/bin/env python
"""Inflate path for the Lane FC FiLM-Canvas SegMap variant.

Archive layout matches the standard SegMap arm:
    archive/
      segmap_weights.tar.xz   -- block-FP-quantized SegMap+FiLM weights packed
                                 by tac.block_fp_codec.pack_payload_tar_xz.
      grayscale.mkv           -- 1-channel grayscale mask video.
      optimized_poses.pt      -- optional, anchor poses (kept for parity).

The only delta from inflate_segmap.py is the model class:
``tac.segmap_film_canvas_renderer.SegMapFilmCanvas`` instead of
``tac.segmap_renderer.SegMap``. The SegMapFilmCanvas auto-detects the
``film_table.weight`` key in the state_dict; if absent, this loader
gracefully falls back to vanilla SegMap (so a Lane SA archive accidentally
routed here still inflates — degraded by the missing FiLM modulation but
not catastrophic).

STRICT-SCORER-RULE COMPLIANCE: this script does NOT load PoseNet/SegNet.
The renderer is the only neural component loaded at inflate time.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


OUT_W, OUT_H = 1164, 874
SEG_W, SEG_H = 512, 384
NUM_FRAMES = 1200
NUM_CLASSES = 5


def _raw_output_path(inflated_dir: Path, video_name: str) -> Path:
    """Return the raw path expected by contest auth eval for ``video_name``."""

    rel = Path(video_name).with_suffix(".raw")
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe video name: {video_name!r}")
    return inflated_dir / rel


def _ensure_repo_on_path() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent
    src_dir = repo_root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _decode_grayscale_mkv(mkv_path: Path) -> torch.Tensor:
    import av

    frames = []
    with av.open(str(mkv_path)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            for frame in packet.decode():
                frames.append(frame.to_ndarray(format="gray"))
    if not frames:
        raise RuntimeError(f"grayscale.mkv yielded 0 frames at {mkv_path}")
    return torch.from_numpy(np.stack(frames, axis=0)).contiguous()


def _grayscale_to_classes(gray: torch.Tensor) -> torch.Tensor:
    from tac.mask_grayscale_lut import decode_grayscale_to_classes

    return decode_grayscale_to_classes(gray)


def _classes_to_one_hot(classes: torch.Tensor) -> torch.Tensor:
    return F.one_hot(classes.long(), num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()


def _grayscale_to_mask_features(
    gray: torch.Tensor, *, device: torch.device, mode: str
) -> torch.Tensor:
    if mode == "soft_lut":
        from tac.mask_grayscale_lut import grayscale_to_probability_map

        return grayscale_to_probability_map(
            gray.to(device), sigma=15.0, channel_first=True
        ).float()
    if mode == "hard_onehot":
        classes = _grayscale_to_classes(gray)
        return _classes_to_one_hot(classes).to(device)
    raise RuntimeError(
        f"unknown SEGMAP_GRAYSCALE_MODE={mode!r}; expected soft_lut or hard_onehot"
    )


def _build_model(state_dict: dict, hidden: int, block_hidden: int,
                 num_blocks: int, max_frame_index: int):
    """Detect Lane FC FiLM table; route to SegMapFilmCanvas or vanilla SegMap."""
    from tac.segmap_film_canvas_renderer import (
        FILM_TABLE_KEY,
        SegMapFilmCanvas,
        has_film_table,
    )

    if has_film_table(state_dict):
        print(
            f"[inflate-segmap-fc] state_dict carries {FILM_TABLE_KEY!r} -> "
            f"loading SegMapFilmCanvas (FiLM modulation active).",
            file=sys.stderr,
        )
        model = SegMapFilmCanvas(
            hidden=hidden,
            block_hidden=block_hidden,
            num_blocks=num_blocks,
            max_frame_index=max_frame_index,
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model
    # Fallback: vanilla SegMap (no FiLM). Useful if a Lane SA archive accidentally
    # routes here.
    from tac.segmap_renderer import SegMap

    print(
        "[inflate-segmap-fc] state_dict has no FiLM table; falling back to "
        "vanilla SegMap (degraded — FiLM modulation absent).",
        file=sys.stderr,
    )
    model = SegMap(
        hidden=hidden,
        block_hidden=block_hidden,
        num_blocks=num_blocks,
        max_frame_index=max_frame_index,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _load_state(payload_path: Path) -> dict:
    from tac.block_fp_codec import unpack_payload_tar_xz

    return unpack_payload_tar_xz(payload_path)


def inflate(archive_dir: Path, inflated_dir: Path, video_names_file: Path,
            payload_filename: str = "segmap_weights.tar.xz",
            mask_filename: str = "grayscale.mkv",
            hidden: int = 24, block_hidden: int = 24, num_blocks: int = 8,
            max_frame_index: int = NUM_FRAMES,
            target_w: int = OUT_W, target_h: int = OUT_H) -> None:
    _ensure_repo_on_path()
    archive_dir = Path(archive_dir)
    inflated_dir = Path(inflated_dir)
    inflated_dir.mkdir(parents=True, exist_ok=True)

    payload_path = archive_dir / payload_filename
    mask_path = archive_dir / mask_filename
    for f, label in [(payload_path, payload_filename), (mask_path, mask_filename)]:
        if not f.exists():
            raise FileNotFoundError(f"missing {label}: {f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inflate-segmap-fc] device={device}", file=sys.stderr)

    t0 = time.monotonic()
    state_dict = _load_state(payload_path)
    model = _build_model(state_dict, hidden=hidden, block_hidden=block_hidden,
                         num_blocks=num_blocks,
                         max_frame_index=max_frame_index).to(device)
    print(f"[inflate-segmap-fc] loaded model in {time.monotonic() - t0:.2f}s",
          file=sys.stderr)

    t0 = time.monotonic()
    gray = _decode_grayscale_mkv(mask_path)
    if gray.shape[-2:] != (SEG_H, SEG_W):
        raise RuntimeError(
            f"grayscale.mkv resolution {gray.shape[-2:]} != ({SEG_H}, {SEG_W}). "
            f"Half-resolution masks are FORBIDDEN (Check 76)."
        )
    grayscale_mode = os.environ.get("SEGMAP_GRAYSCALE_MODE", "soft_lut").strip().lower()
    if grayscale_mode in {"hard", "one_hot", "onehot"}:
        grayscale_mode = "hard_onehot"
    if grayscale_mode not in {"soft_lut", "hard_onehot"}:
        raise RuntimeError(
            f"unknown SEGMAP_GRAYSCALE_MODE={grayscale_mode!r}; "
            "expected soft_lut or hard_onehot"
        )
    print(
        f"[inflate-segmap-fc] decoded grayscale in {time.monotonic() - t0:.2f}s "
        f"(mode={grayscale_mode})",
        file=sys.stderr,
    )

    video_names = [
        ln.strip() for ln in Path(video_names_file).read_text().splitlines() if ln.strip()
    ]
    if not video_names:
        raise RuntimeError(f"video_names_file empty: {video_names_file}")
    out_name = video_names[0]
    out_path = _raw_output_path(inflated_dir, out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = gray.shape[0]
    if n_total != NUM_FRAMES:
        raise RuntimeError(f"expected {NUM_FRAMES} frames, got {n_total}")

    batch = 16 if device.type == "cuda" else 4
    t0 = time.monotonic()
    n_written = 0
    with out_path.open("wb") as f, torch.no_grad():
        for start in range(0, n_total, batch):
            end = min(start + batch, n_total)
            chunk_oh = _grayscale_to_mask_features(
                gray[start:end], device=device, mode=grayscale_mode
            )
            frame_idx = torch.arange(start, end, device=device, dtype=torch.long)
            rgb = model(chunk_oh, frame_idx)
            rgb_native = F.interpolate(
                rgb, size=(target_h, target_w), mode="bicubic", align_corners=False
            )
            rgb_u8 = rgb_native.clamp(0.0, 255.0).round().to(torch.uint8)
            rgb_hwc = rgb_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            f.write(rgb_hwc.tobytes())
            n_written += rgb_hwc.shape[0]
    elapsed = time.monotonic() - t0
    print(
        f"[inflate-segmap-fc] wrote {n_written} frames to {out_path} in {elapsed:.1f}s",
        file=sys.stderr,
    )

    actual = out_path.stat().st_size
    expected = target_w * target_h * 3 * n_written
    if actual != expected:
        raise RuntimeError(
            f"output size mismatch {actual} != {expected} (target_w={target_w}, "
            f"target_h={target_h}, n={n_written})"
        )


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Inflate a Lane FC SegMap+FiLM archive.")
    parser.add_argument("archive_dir")
    parser.add_argument("inflated_dir")
    parser.add_argument("video_names_file")
    parser.add_argument("--payload-filename", default="segmap_weights.tar.xz")
    parser.add_argument("--mask-filename", default="grayscale.mkv")
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--block-hidden", type=int, default=24)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--max-frame-index", type=int, default=NUM_FRAMES)
    parser.add_argument("--target-w", type=int, default=OUT_W)
    parser.add_argument("--target-h", type=int, default=OUT_H)
    args = parser.parse_args()

    inflate(
        archive_dir=Path(args.archive_dir),
        inflated_dir=Path(args.inflated_dir),
        video_names_file=Path(args.video_names_file),
        payload_filename=args.payload_filename,
        mask_filename=args.mask_filename,
        hidden=args.hidden,
        block_hidden=args.block_hidden,
        num_blocks=args.num_blocks,
        max_frame_index=args.max_frame_index,
        target_w=args.target_w,
        target_h=args.target_h,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
