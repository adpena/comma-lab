#!/usr/bin/env python
"""NSCS02 standalone inflate (substrate-engineering, ~95 LOC).

Renders 1200 frames at (192, 256), upsamples to camera-native (1164, 874)
via canonical bicubic, writes raw uint8 RGB to ``dst.raw``.

Per HNeRV parity discipline:
- L3 monolithic single-file 0.bin (NSCS02 wire format).
- L4 inflate.py <= 100 LOC reviewable in 30 seconds.
- L9 runtime closure: only ``torch`` + ``torch.nn.functional`` +
  ``brotli`` + ``numpy``.

Per CLAUDE.md "Strict scorer rule": no contest scorer loaded at inflate.
Per A1 council Round 1 finding F1/F11 + Catalog #205: device pinned
via ``PACT_INFLATE_DEVICE`` env var; MPS forbidden.
"""
import os
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from codec import (  # noqa: E402
    LATENT_DIM,
    BASE_CHANNELS,
    N_PAIRS,
    NSCS02_ARCHIVE_MAGIC,
    parse_nscs02_archive_bytes,
)
from model import NSCS02Decoder  # noqa: E402

CAMERA_H, CAMERA_W = 874, 1164
RENDER_H, RENDER_W = 192, 256


def select_inflate_device() -> torch.device:
    """Honor ``PACT_INFLATE_DEVICE`` (auto/cpu/cuda); MPS forbidden.

    Mirrors the canonical
    ``tac.substrates._shared.inflate_runtime.select_inflate_device``
    body byte-for-byte (modulo the ``torch.device`` return-type wrap).
    Per CLAUDE.md "Contest runtime closure" non-negotiable the inflate
    tree must be self-contained so we cannot import the canonical
    helper at inflate time. Parity is enforced by
    ``src/tac/tests/test_inflate_select_device_parity.py``.
    """
    value = (os.environ.get("PACT_INFLATE_DEVICE") or "auto").strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PACT_INFLATE_DEVICE=cuda but torch.cuda is not available")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    raise RuntimeError(f"unsupported PACT_INFLATE_DEVICE={value!r}; expected auto/cpu/cuda")


def inflate(src_bin: str, dst_raw: str) -> int:
    with open(src_bin, "rb") as f:
        archive_bytes = f.read()
    decoder_sd, latents = parse_nscs02_archive_bytes(archive_bytes)

    device = select_inflate_device()
    decoder = NSCS02Decoder(
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        render_hw=(RENDER_H, RENDER_W),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, N_PAIRS, 16):
            j = min(i + 16, N_PAIRS)
            batch = j - i
            decoded = decoder(latents[i:j])  # (batch, 2, 3, 192, 256)
            flat = decoded.reshape(batch * 2, 3, RENDER_H, RENDER_W)
            up = F.interpolate(
                flat, size=(CAMERA_H, CAMERA_W),
                mode="bicubic", align_corners=False,
            )
            frames = (
                up.clamp(0, 255).permute(0, 2, 3, 1).round()
                .to(torch.uint8).cpu().numpy()
            )
            fout.write(frames.tobytes())
            n += batch * 2
    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
