#!/usr/bin/env python
"""Inflate path for the SegMap-paradigm archive (Lane SA / SC++ / SO).

Archive layout:
    archive/
      segmap_weights.tar.xz   -- block-FP-quantized SegMap weights packed
                                 by tac.block_fp_codec.pack_payload_tar_xz.
      grayscale.mkv           -- 1-channel grayscale mask video; decoded
                                 back to 5-class via the Gaussian softmax
                                 LUT in tac.mask_grayscale_lut.
      optimized_poses.pt      -- optional, per-pair affine embeddings
                                 indexed by [2*idx, 2*idx+1].

Pipeline:
    grayscale.mkv  -> Gaussian-LUT      -> 5-class one-hot (1200, 5, 384, 512)
    one-hot        -> SegMap forward    -> RGB frames      (1200, 3, 384, 512)
    frames         -> bilinear upscale  -> raw RGB         (1200, 3, 874, 1164)

STRICT-SCORER-RULE COMPLIANCE: this script does NOT load the 73MB SegNet
or PoseNet weights. The renderer (SegMap) is the only neural component
loaded at inflate time.
"""
from __future__ import annotations

import argparse
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


def _build_segmap(state_dict: dict, hidden: int, block_hidden: int,
                   num_blocks: int, max_frame_index: int):
    from tac.segmap_renderer import SegMap

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
            poses_filename: str = "optimized_poses.pt",
            hidden: int = 24, block_hidden: int = 24, num_blocks: int = 8,
            max_frame_index: int = NUM_FRAMES,
            target_w: int = OUT_W, target_h: int = OUT_H) -> None:
    _ensure_repo_on_path()

    archive_dir = Path(archive_dir)
    inflated_dir = Path(inflated_dir)
    inflated_dir.mkdir(parents=True, exist_ok=True)

    payload_path = archive_dir / payload_filename
    mask_path = archive_dir / mask_filename

    for f, label in [
        (payload_path, "segmap_weights.tar.xz"),
        (mask_path, "grayscale.mkv"),
    ]:
        if not f.exists():
            raise FileNotFoundError(f"missing {label}: {f}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[inflate-segmap] device={device}", file=sys.stderr)

    t0 = time.monotonic()
    state_dict = _load_state(payload_path)
    model = _build_segmap(state_dict, hidden=hidden, block_hidden=block_hidden,
                           num_blocks=num_blocks,
                           max_frame_index=max_frame_index).to(device)
    print(f"[inflate-segmap] loaded SegMap in {time.monotonic() - t0:.2f}s",
          file=sys.stderr)

    t0 = time.monotonic()
    gray = _decode_grayscale_mkv(mask_path)
    if gray.shape[-2:] != (SEG_H, SEG_W):
        raise RuntimeError(
            f"grayscale.mkv resolution {gray.shape[-2:]} != ({SEG_H}, {SEG_W}). "
            f"Half-resolution masks are FORBIDDEN (Check 76)."
        )
    classes = _grayscale_to_classes(gray)
    print(f"[inflate-segmap] grayscale->classes in {time.monotonic() - t0:.2f}s",
          file=sys.stderr)

    video_names = [
        ln.strip() for ln in Path(video_names_file).read_text().splitlines() if ln.strip()
    ]
    if not video_names:
        raise RuntimeError(f"video_names_file empty: {video_names_file}")
    out_name = video_names[0]
    out_path = inflated_dir / f"{out_name}.raw"

    n_total = classes.shape[0]
    if n_total != NUM_FRAMES:
        raise RuntimeError(f"expected {NUM_FRAMES} frames, got {n_total}")

    batch = 16 if device.type == "cuda" else 4
    t0 = time.monotonic()
    n_written = 0
    with out_path.open("wb") as f, torch.no_grad():
        for start in range(0, n_total, batch):
            end = min(start + batch, n_total)
            chunk_cls = classes[start:end]
            chunk_oh = _classes_to_one_hot(chunk_cls).to(device)
            frame_idx = torch.arange(start, end, device=device, dtype=torch.long)
            rgb = model(chunk_oh, frame_idx)
            rgb_native = F.interpolate(
                rgb, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            rgb_u8 = (rgb_native.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
            rgb_hwc = rgb_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            f.write(rgb_hwc.tobytes())
            n_written += rgb_hwc.shape[0]
    elapsed = time.monotonic() - t0
    print(
        f"[inflate-segmap] wrote {n_written} frames to {out_path} in {elapsed:.1f}s",
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
    parser = argparse.ArgumentParser(description="Inflate a SegMap archive.")
    parser.add_argument("archive_dir")
    parser.add_argument("inflated_dir")
    parser.add_argument("video_names_file")
    parser.add_argument("--payload-filename", default="segmap_weights.tar.xz")
    parser.add_argument("--mask-filename", default="grayscale.mkv")
    parser.add_argument("--poses-filename", default="optimized_poses.pt")
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
        poses_filename=args.poses_filename,
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
