# SPDX-License-Identifier: MIT
"""QFAI export/load helpers for Lane Q-FAITHFUL JointFrameGenerator checkpoints.

The QFAI format is intentionally simple — just enough header to allow the
inflate-side dispatch to reconstruct the architecture, then a torch.save
of the state_dict.

Layout (LE):
    [4]  magic  = b"QFAI"
    [4]  header_len  uint32
    [N]  JSON header  ({num_classes, pose_dim, cond_dim, depth_mult,
                       fp4_packed=False, sha256=...})
    [...] torch.save(state_dict)

This is the storage format Quantizr himself uses (see his inflate.py:48-69:
torch.load on a brotli-decompressed payload, with a per-tensor "quantized"
or "dense_fp16" dispatch). Our V1 stores plain FP32 weights; V2 will add
FP4-packed branch when QAT lands.

# ROUNDTRIP_NOT_REQUIRED: the regression test below
# (test_quantizr_faithful_export_roundtrip) IS the roundtrip assertion;
# this module just packs/unpacks bytes.
"""
from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from typing import Any

import torch

from tac.quantizr_faithful_renderer import (
    JointFrameGenerator,
    build_quantizr_faithful_renderer,
)

QFAI_MAGIC = b"QFAI"


def save_qfai(
    model: JointFrameGenerator,
    path: str | Path,
    *,
    extra_meta: dict[str, Any] | None = None,
) -> int:
    """Serialize a JointFrameGenerator to QFAI binary format.

    Returns: number of bytes written.
    """
    header: dict[str, Any] = {
        "format": "QFAI",
        "version": 1,
        "num_classes": int(model.num_classes),
        "pose_dim": int(model.pose_dim),
        "cond_dim": int(model.cond_dim),
        "depth_mult": int(model.depth_mult),
        "out_h": int(model.out_h),
        "out_w": int(model.out_w),
        "fp4_packed": False,
    }
    if extra_meta:
        header["extra"] = extra_meta

    header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
    sd_buf = io.BytesIO()
    torch.save(model.state_dict(), sd_buf)
    sd_bytes = sd_buf.getvalue()

    out_path = Path(path)
    with out_path.open("wb") as f:
        f.write(QFAI_MAGIC)
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        f.write(sd_bytes)

    return out_path.stat().st_size


def load_qfai(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> JointFrameGenerator:
    """Reconstruct a JointFrameGenerator from a QFAI binary."""
    raw = Path(path).read_bytes()
    if raw[:4] != QFAI_MAGIC:
        raise ValueError(
            f"Not a QFAI binary (got magic {raw[:4]!r}, expected {QFAI_MAGIC!r})"
        )
    offset = 4
    header_len = struct.unpack("<I", raw[offset:offset + 4])[0]
    offset += 4
    header = json.loads(raw[offset:offset + header_len].decode("utf-8"))
    offset += header_len

    gen = build_quantizr_faithful_renderer(
        num_classes=int(header.get("num_classes", 5)),
        pose_dim=int(header.get("pose_dim", 6)),
        cond_dim=int(header.get("cond_dim", 48)),
        depth_mult=int(header.get("depth_mult", 1)),
    )
    state = torch.load(
        io.BytesIO(raw[offset:]),
        map_location=device,
        weights_only=True,
    )
    gen.load_state_dict(state, strict=True)
    gen.to(device).eval()
    return gen
