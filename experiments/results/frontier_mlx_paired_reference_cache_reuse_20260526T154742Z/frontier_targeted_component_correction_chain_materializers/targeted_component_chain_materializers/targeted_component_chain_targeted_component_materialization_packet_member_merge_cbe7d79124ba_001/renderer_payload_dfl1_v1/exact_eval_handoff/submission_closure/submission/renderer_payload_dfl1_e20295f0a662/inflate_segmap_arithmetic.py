#!/usr/bin/env python
"""Lane SH inflate path — arithmetic-coded SegMap weights (Shannon eureka).

Archive layout:
    archive/
      payload.bin             -- Lane SH binary container (SHv1) holding
                                 arithmetic-coded qint streams +
                                 passthrough bias / embedding tensors
                                 (replaces segmap_weights.tar.xz).
      grayscale.mkv           -- 1-channel grayscale mask video.
      optimized_poses.pt      -- optional anchor poses.

The decoder uses ``tac.arithmetic_qint_codec.unpack_arithmetic_payload`` to
materialise the float SegMap state_dict, then routes through the standard
SegMap renderer. The only delta vs inflate_segmap.py is the payload loader.

STRICT-SCORER-RULE COMPLIANCE: only the SegMap renderer is loaded at inflate.
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


def _normalize_grayscale_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized in {"hard", "one_hot", "onehot"}:
        return "hard_onehot"
    if normalized in {"soft_lut", "hard_onehot"}:
        return normalized
    raise RuntimeError(
        f"unknown SEGMAP_GRAYSCALE_MODE={normalized!r}; expected soft_lut or hard_onehot"
    )


def _grayscale_to_mask_features(
    gray: torch.Tensor,
    *,
    device: torch.device,
    mode: str,
    class_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    normalized = _normalize_grayscale_mode(mode)
    if normalized == "soft_lut":
        from tac.mask_grayscale_lut import grayscale_to_probability_map

        targets = None
        if class_targets is not None:
            targets = class_targets.to(device=device, dtype=torch.float32)
        return grayscale_to_probability_map(
            gray.to(device), sigma=15.0, targets=targets, channel_first=True
        ).float()
    if class_targets is not None:
        raise RuntimeError(
            "custom class targets require SEGMAP_GRAYSCALE_MODE=soft_lut; "
            "hard_onehot uses the fixed nearest-class decoder"
        )
    classes = _grayscale_to_classes(gray)
    return _classes_to_one_hot(classes).to(device)


def _build_segmap(state_dict: dict, hidden: int, block_hidden: int,
                   num_blocks: int, max_frame_index: int):
    """Detect FiLM table; route to SegMapFilmCanvas or vanilla SegMap.

    Lane SH is paradigm-agnostic — it just changes the codec, not the model
    class — so we honour either checkpoint type that ships through it.
    """
    from tac.segmap_film_canvas_renderer import has_film_table

    if has_film_table(state_dict):
        from tac.segmap_film_canvas_renderer import SegMapFilmCanvas

        model = SegMapFilmCanvas(
            hidden=hidden,
            block_hidden=block_hidden,
            num_blocks=num_blocks,
            max_frame_index=max_frame_index,
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

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
    print(f"[inflate-segmap-arith] arch={arch} ({cls.__name__})", file=sys.stderr)

    model = cls(
        hidden=hidden,
        block_hidden=block_hidden,
        num_blocks=num_blocks,
        max_frame_index=max_frame_index,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


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
        f"[inflate-segmap-arith] loaded class targets payload {path.name} "
        f"bytes={len(data)} sha256={sha256} targets={targets.tolist()}",
        file=sys.stderr,
    )
    return targets.detach().to(torch.float32)


def inflate(archive_dir: Path, inflated_dir: Path, video_names_file: Path,
            payload_filename: str = "payload.bin",
            mask_filename: str = "grayscale.mkv",
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
    for f, label in [(payload_path, payload_filename), (mask_path, mask_filename)]:
        if not f.exists():
            raise FileNotFoundError(f"missing {label}: {f}")
    if class_targets_path is not None and not class_targets_path.exists():
        raise FileNotFoundError(f"missing class targets payload: {class_targets_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inflate-segmap-arith] device={device}", file=sys.stderr)

    from tac.arithmetic_qint_codec import unpack_arithmetic_payload

    t0 = time.monotonic()
    state_dict = unpack_arithmetic_payload(str(payload_path))
    model = _build_segmap(
        state_dict,
        hidden=hidden,
        block_hidden=block_hidden,
        num_blocks=num_blocks,
        max_frame_index=max_frame_index,
    ).to(device)
    print(
        f"[inflate-segmap-arith] loaded SegMap in {time.monotonic() - t0:.2f}s",
        file=sys.stderr,
    )

    t0 = time.monotonic()
    gray = _decode_grayscale_mkv(mask_path)
    if gray.shape[-2:] != (SEG_H, SEG_W):
        raise RuntimeError(
            f"grayscale.mkv resolution {gray.shape[-2:]} != ({SEG_H}, {SEG_W})."
        )
    grayscale_mode = _normalize_grayscale_mode(
        os.environ.get("SEGMAP_GRAYSCALE_MODE", "soft_lut")
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
        f"[inflate-segmap-arith] decoded grayscale in {time.monotonic() - t0:.2f}s "
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
            rgb_native = F.interpolate(
                rgb, size=(target_h, target_w), mode="bicubic", align_corners=False
            )
            rgb_u8 = rgb_native.clamp(0.0, 255.0).round().to(torch.uint8)
            rgb_hwc = rgb_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            f.write(rgb_hwc.tobytes())
            n_written += rgb_hwc.shape[0]
    elapsed = time.monotonic() - t0
    print(
        f"[inflate-segmap-arith] wrote {n_written} frames to {out_path} in {elapsed:.1f}s",
        file=sys.stderr,
    )

    actual = out_path.stat().st_size
    expected = target_w * target_h * 3 * n_written
    if actual != expected:
        raise RuntimeError(
            f"output size mismatch {actual} != {expected}"
        )


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Inflate a Lane SH arithmetic-coded SegMap archive."
    )
    parser.add_argument("archive_dir")
    parser.add_argument("inflated_dir")
    parser.add_argument("video_names_file")
    parser.add_argument("--payload-filename", default="payload.bin")
    parser.add_argument("--mask-filename", default="grayscale.mkv")
    parser.add_argument(
        "--class-targets-filename",
        default=os.environ.get("SEGMAP_CLASS_TARGETS_FILENAME", ""),
        help="Optional 10-byte fp16 Lane LCT payload archive member. Empty "
             "default preserves the fixed Selfcomp class targets.",
    )
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
