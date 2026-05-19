# SPDX-License-Identifier: MIT
"""Vendored contest inflate helpers for pr106_latent_sidecar_r2.

This file mirrors the shared helper surface used by repository-side substrate
runtimes. It is vendored inside ``src/`` so ``inflate.sh`` remains self-contained
when run with an empty ``PYTHONPATH``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO

import torch
import torch.nn.functional as F

CAMERA_HW: tuple[int, int] = (874, 1164)


def select_inflate_device(requested: str | None = None) -> str:
    """Return ``"cuda"`` or ``"cpu"`` for inflate-time rendering."""

    value = (requested or os.environ.get("PACT_INFLATE_DEVICE") or "auto").strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PACT_INFLATE_DEVICE=cuda but torch.cuda is not available")
        return "cuda"
    if value == "cpu":
        return "cpu"
    raise RuntimeError(f"unsupported PACT_INFLATE_DEVICE={value!r}; expected auto/cpu/cuda")


def raw_output_path(output_dir: Path, video_name: str) -> Path:
    """Return the contest raw path for one safe relative video name."""

    raw = str(video_name).replace("\\", "/").strip()
    rel = Path(raw)
    if (
        not raw
        or "//" in raw
        or rel.is_absolute()
        or any(part in {"", ".."} for part in rel.parts)
    ):
        raise ValueError(f"unsafe file_list video name for raw output: {video_name!r}")
    root = output_dir.resolve(strict=False)
    target = (output_dir / rel.with_suffix(".raw")).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"file_list video name escapes output directory: {video_name!r}"
        ) from exc
    return target


def write_rgb_pair_to_raw(
    fh: BinaryIO,
    rgb_0: torch.Tensor,
    rgb_1: torch.Tensor,
    *,
    input_range: str = "unit",
    resize_mode: str = "bicubic",
) -> int:
    """Append one rendered frame-pair to an open contest ``.raw`` file."""

    if rgb_0.shape != rgb_1.shape or rgb_0.dim() != 4 or rgb_0.shape[0] != 1 or rgb_0.shape[1] != 3:
        raise ValueError(
            "write_rgb_pair_to_raw expects two tensors shaped (1, 3, H, W); "
            f"got {tuple(rgb_0.shape)} and {tuple(rgb_1.shape)}"
        )
    frames = torch.cat([rgb_0, rgb_1], dim=0)
    if input_range == "unit":
        frames = frames * 255.0
    elif input_range != "byte":
        raise ValueError(f"input_range must be 'unit' or 'byte', got {input_range!r}")
    if tuple(frames.shape[-2:]) != CAMERA_HW:
        frames = F.interpolate(frames, size=CAMERA_HW, mode=resize_mode, align_corners=False)
    frames_u8 = (
        frames.clamp(0.0, 255.0)
        .permute(0, 2, 3, 1)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    fh.write(frames_u8.tobytes(order="C"))
    return int(frames_u8.shape[0])


__all__ = [
    "CAMERA_HW",
    "raw_output_path",
    "select_inflate_device",
    "write_rgb_pair_to_raw",
]
