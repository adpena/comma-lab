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
    grayscale.mkv  -> Gaussian-LUT      -> 5-class soft map (1200, 5, 384, 512)
    soft map       -> SegMap forward    -> RGB frames      (1200, 3, 384, 512)
    frames         -> bicubic upscale   -> raw RGB         (1200, 3, 874, 1164)

STRICT-SCORER-RULE COMPLIANCE: this script does NOT load the 73MB SegNet
or PoseNet weights. The renderer (SegMap) is the only neural component
loaded at inflate time.
"""
from __future__ import annotations

import argparse
import hashlib
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
    gray: torch.Tensor,
    *,
    device: torch.device,
    mode: str,
    class_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode == "soft_lut":
        from tac.mask_grayscale_lut import grayscale_to_probability_map

        targets = None
        if class_targets is not None:
            targets = class_targets.to(device=device, dtype=torch.float32)
        return grayscale_to_probability_map(
            gray.to(device), sigma=15.0, targets=targets, channel_first=True
        ).float()
    if mode == "hard_onehot":
        if class_targets is not None:
            raise RuntimeError(
                "custom class targets require SEGMAP_GRAYSCALE_MODE=soft_lut; "
                "hard_onehot uses the fixed nearest-class decoder"
            )
        classes = _grayscale_to_classes(gray)
        return _classes_to_one_hot(classes).to(device)
    raise RuntimeError(
        f"unknown SEGMAP_GRAYSCALE_MODE={mode!r}; expected soft_lut or hard_onehot"
    )


def _build_segmap(state_dict: dict, hidden: int, block_hidden: int,
                   num_blocks: int, max_frame_index: int):
    """Instantiate the SegMap variant declared by SEGMAP_ARCH env var.

    Default = canonical SegMap (6-DOF affine, frame_affine_embedding shape (N, 6)).
    SEGMAP_ARCH=segmap_homography → SegMapHomography (8-DOF perspective,
        frame_affine_embedding shape (N, 8)). Lane HM-S sets this via
        config.env so a single inflate dispatcher can cover both archs.

    The state_dict's frame_affine_embedding shape is the source of truth —
    a mismatch raises strict-load error which surfaces the operator bug
    rather than silently returning zeros.
    """
    import os
    from tac.segmap_renderer import SegMap, SegMapHomography

    arch = os.environ.get("SEGMAP_ARCH", "segmap").strip().lower() or "segmap"
    if arch == "segmap_homography":
        cls = SegMapHomography
    elif arch == "segmap":
        cls = SegMap
    else:
        raise RuntimeError(
            f"unknown SEGMAP_ARCH={arch!r}; expected 'segmap' or 'segmap_homography'"
        )
    print(f"[inflate-segmap] arch={arch} ({cls.__name__})", file=sys.stderr)

    model = cls(
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


def _resolve_archive_member(archive_dir: Path, filename: str, label: str) -> Path:
    rel = Path(filename)
    if not filename or rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError(
            f"{label} must be a nonempty relative archive member path without '..', "
            f"got {filename!r}"
        )
    return archive_dir / rel


def _load_class_targets_payload(path: Path) -> torch.Tensor:
    from tac.learnable_class_targets import LearnableClassTargets

    data = path.read_bytes()
    targets = LearnableClassTargets.deserialize_from_bytes(data)()
    sha256 = hashlib.sha256(data).hexdigest()
    print(
        f"[inflate-segmap] loaded class targets payload {path.name} "
        f"bytes={len(data)} sha256={sha256} targets={targets.tolist()}",
        file=sys.stderr,
    )
    return targets.detach().to(torch.float32)


def inflate(archive_dir: Path, inflated_dir: Path, video_names_file: Path,
            payload_filename: str = "segmap_weights.tar.xz",
            mask_filename: str = "grayscale.mkv",
            poses_filename: str = "optimized_poses.pt",
            class_targets_filename: str | None = None,
            hidden: int = 24, block_hidden: int = 24, num_blocks: int = 8,
            max_frame_index: int = NUM_FRAMES,
            target_w: int = OUT_W, target_h: int = OUT_H) -> None:
    _ensure_repo_on_path()

    archive_dir = Path(archive_dir)
    inflated_dir = Path(inflated_dir)
    inflated_dir.mkdir(parents=True, exist_ok=True)

    payload_path = archive_dir / payload_filename
    mask_path = archive_dir / mask_filename
    class_targets_path = None
    if class_targets_filename:
        class_targets_path = _resolve_archive_member(
            archive_dir, class_targets_filename, "class_targets_filename"
        )

    for f, label in [
        (payload_path, "segmap_weights.tar.xz"),
        (mask_path, "grayscale.mkv"),
    ]:
        if not f.exists():
            raise FileNotFoundError(f"missing {label}: {f}")
    if class_targets_path is not None and not class_targets_path.exists():
        raise FileNotFoundError(f"missing class targets payload: {class_targets_path}")

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
    grayscale_mode = os.environ.get("SEGMAP_GRAYSCALE_MODE", "soft_lut").strip().lower()
    if grayscale_mode in {"hard", "one_hot", "onehot"}:
        grayscale_mode = "hard_onehot"
    if grayscale_mode not in {"soft_lut", "hard_onehot"}:
        raise RuntimeError(
            f"unknown SEGMAP_GRAYSCALE_MODE={grayscale_mode!r}; "
            "expected soft_lut or hard_onehot"
        )
    class_targets = (
        _load_class_targets_payload(class_targets_path)
        if class_targets_path is not None
        else None
    )
    if class_targets is not None and grayscale_mode != "soft_lut":
        raise RuntimeError(
            "custom class targets are only valid with SEGMAP_GRAYSCALE_MODE=soft_lut"
        )
    print(
        f"[inflate-segmap] decoded grayscale in {time.monotonic() - t0:.2f}s "
        f"(mode={grayscale_mode}, class_targets={'custom' if class_targets is not None else 'fixed'})",
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
                gray[start:end],
                device=device,
                mode=grayscale_mode,
                class_targets=class_targets,
            )
            frame_idx = torch.arange(start, end, device=device, dtype=torch.long)
            rgb = model(chunk_oh, frame_idx)
            # Round 2 review CRITICAL: training _eval_roundtrip_chain uses
            # bicubic for 384 -> 874 -> uint8 -> 384. Using bilinear here
            # breaks train/inference parity and re-introduces proxy-auth
            # drift. Match the canonical bicubic mode.
            rgb_native = F.interpolate(
                rgb, size=(target_h, target_w), mode="bicubic", align_corners=False
            )
            # SegMap.forward returns sigmoid(...) * 255.0 already in [0, 255] range.
            # Round 1 review CRITICAL: previous clamp(0, 1) * 255 zeroed all output.
            rgb_u8 = rgb_native.clamp(0.0, 255.0).round().to(torch.uint8)
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
    import os as _os

    parser = argparse.ArgumentParser(description="Inflate a SegMap archive.")
    parser.add_argument("archive_dir")
    parser.add_argument("inflated_dir")
    parser.add_argument("video_names_file")
    parser.add_argument("--payload-filename", default="segmap_weights.tar.xz")
    parser.add_argument("--mask-filename", default="grayscale.mkv")
    parser.add_argument("--poses-filename", default="optimized_poses.pt")
    parser.add_argument(
        "--class-targets-filename",
        default=_os.environ.get("SEGMAP_CLASS_TARGETS_FILENAME", ""),
        help="Optional 10-byte fp16 Lane LCT payload archive member. Empty "
             "default preserves the fixed Selfcomp class targets.",
    )
    # CLI defaults are the canonical Lane SC++/SA arch (hidden=24, block_hidden=24,
    # num_blocks=8). Lane DARTS-S sweeps over alternative archs and overrides via
    # SEGMAP_HIDDEN / SEGMAP_BLOCK_HIDDEN / SEGMAP_NUM_BLOCKS env vars (sourced
    # from config.env per lane). Env wins over default; CLI wins over env.
    parser.add_argument("--hidden", type=int,
                        default=int(_os.environ.get("SEGMAP_HIDDEN", "24")))
    parser.add_argument("--block-hidden", type=int,
                        default=int(_os.environ.get("SEGMAP_BLOCK_HIDDEN", "24")))
    parser.add_argument("--num-blocks", type=int,
                        default=int(_os.environ.get("SEGMAP_NUM_BLOCKS", "8")))
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
        class_targets_filename=args.class_targets_filename or None,
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
