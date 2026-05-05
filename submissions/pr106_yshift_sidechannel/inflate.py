#!/usr/bin/env python
"""Inflate pr106_yshift_sidechannel: PR106 HNeRV decoder + per-pair Y-shift sidechannel.

Wire format (single 0.bin in archive.zip):

    Offset  Bytes  Field                                    Notes
    ────────────────────────────────────────────────────────────────────────
    0       1      magic 0xFC                               yshift dispatch byte
    1       3      pr106_len (uint24 LE)                    bytes of inner PR106 archive
    4       N      pr106 packed archive (raw bytes)         starts with 0xFF magic
    4+N     1      sidechannel magic 0x1                    1 = SC01-YSHIFT v1
    5+N     2      sidechannel_len (uint16 LE)              brotli'd SC01 payload size
    7+N     M      brotli(SC01 sidechannel payload)         decompresses to header + raw

Inner SC01 payload (post-brotli-decompress):

    Offset  Bytes  Field                                    Notes
    ────────────────────────────────────────────────────────────────────────
    0       4      magic b"SC01"
    4       1      mode_id (= 7 = MODE_Y_SHIFT)
    5       1      n_channels (= 3 = [Y_offset, dy, dx])
    6       4      n_pairs (uint32 LE; equals n_pair * 2 frames)
    10      4      step (float32 LE; Y_offset scale factor)
    14      M-14   raw int8 stream length n_pairs * 3       per-frame [y_off, dy, dx]

At inflate time:
    1. Parse PR106 packed archive → decoder weights + latents.
    2. Forward decoder pair-by-pair → 384x512 RGB pair.
    3. Bicubic upsample → 874x1164.
    4. Round to uint8.
    5. Apply YSHIFT correction per frame: x += y_off * step then shift_rgb(dy, dx).
    6. Write contiguous (N, H, W, 3) bytes.

CUDA-required (CLAUDE.md MPS-auth-eval-is-NOISE non-negotiable).

Sister of submissions/apogee_intN/inflate.py; uses the SAME PR106 codec source
(submissions/apogee_intN/src/codec.py + model.py) — pr106_yshift_sidechannel
is purely an additive sidechannel on top of the PR106 packed archive.
"""
from __future__ import annotations

import io
import struct
import sys
from pathlib import Path

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
# Reuse PR106 codec + model from sister submission to avoid duplication.
APOGEE_SRC = HERE.parent / "apogee_intN" / "src"
sys.path.insert(0, str(APOGEE_SRC))

from model import HNeRVDecoder  # type: ignore[import-not-found]
from codec import parse_packed_archive  # type: ignore[import-not-found]


CAMERA_H, CAMERA_W = 874, 1164

YSHIFT_MAGIC_BYTE = 0xFC  # outer dispatch byte for pr106_yshift_sidechannel
SIDECHANNEL_VERSION = 1   # post-PR106-len, before SC01 brotli'd payload

SC01_MAGIC = b"SC01"
SC01_HEADER = struct.Struct("<4sBBIf")  # magic + mode_id + n_channels + n_frames + step
SIDECHANNEL_MODE_Y_SHIFT = 7  # mirrors codex_metric_yshift


def parse_yshift_archive(archive_bytes: bytes) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict, dict | None]:
    """Parse pr106_yshift_sidechannel 0.bin layout.

    Returns (state_dict, latents, meta, sidechannel_or_None).
    """
    if not archive_bytes:
        raise ValueError("empty archive")
    magic = archive_bytes[0]
    if magic != YSHIFT_MAGIC_BYTE:
        raise ValueError(
            f"pr106_yshift magic mismatch: got 0x{magic:02X}, expected 0x{YSHIFT_MAGIC_BYTE:02X}"
        )
    pr106_len = int.from_bytes(archive_bytes[1:4], "little")
    pr106_end = 4 + pr106_len
    if pr106_end > len(archive_bytes):
        raise ValueError(f"pr106_len {pr106_len} exceeds archive size {len(archive_bytes)}")
    pr106_bytes = archive_bytes[4:pr106_end]
    decoder_sd, latents, meta = parse_packed_archive(pr106_bytes)

    sidechannel = None
    if pr106_end < len(archive_bytes):
        sc_version = archive_bytes[pr106_end]
        if sc_version != SIDECHANNEL_VERSION:
            raise ValueError(
                f"sidechannel version mismatch: got {sc_version}, expected {SIDECHANNEL_VERSION}"
            )
        sc_len = struct.unpack_from("<H", archive_bytes, pr106_end + 1)[0]
        sc_payload_start = pr106_end + 3
        sc_payload_end = sc_payload_start + sc_len
        if sc_payload_end != len(archive_bytes):
            raise ValueError(
                f"sidechannel trailing bytes: expected_end={sc_payload_end}, total={len(archive_bytes)}"
            )
        sidechannel = decode_sidechannel_blob(archive_bytes[sc_payload_start:sc_payload_end])
    return decoder_sd, latents, meta, sidechannel


def decode_sidechannel_blob(blob: bytes) -> dict:
    """Brotli-decompress + parse SC01-YSHIFT payload. Returns {mode_id, step, raw[N,3]}."""
    raw = brotli.decompress(blob)
    if len(raw) < SC01_HEADER.size:
        raise ValueError(f"sidechannel blob too small: {len(raw)} < {SC01_HEADER.size}")
    magic, mode_id, channels, frame_count, step = SC01_HEADER.unpack_from(raw)
    if magic != SC01_MAGIC:
        raise ValueError(f"bad SC01 magic: {magic!r}")
    if mode_id != SIDECHANNEL_MODE_Y_SHIFT:
        raise ValueError(f"unsupported sidechannel mode_id={mode_id}, expected {SIDECHANNEL_MODE_Y_SHIFT}")
    if channels != 3:
        raise ValueError(f"YSHIFT expects 3 channels, got {channels}")
    expected = SC01_HEADER.size + frame_count * channels
    if len(raw) != expected:
        raise ValueError(f"bad SC01 length: expected {expected}, got {len(raw)}")
    raw_int = np.frombuffer(raw[SC01_HEADER.size:], dtype=np.int8).copy()  # signed
    raw_arr = raw_int.reshape(int(frame_count), channels)
    return {
        "mode_id": int(mode_id),
        "channels": int(channels),
        "step": float(step),
        "raw": raw_arr,                # (n_frames, 3) int8 — [y_off, dy, dx]
        "n_frames": int(frame_count),
    }


def shift_rgb_uint8(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Integer pixel translation; mirrors codex_metric_yshift shift_rgb but on uint8 numpy.

    frame: (H, W, 3) uint8.
    Returns same shape, with a (dy, dx) shift applied — pixels falling off the
    edge are filled by the un-shifted source (codex pattern: fallback = self).
    """
    if dy == 0 and dx == 0:
        return frame
    h, w, _ = frame.shape
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    out = frame.copy()
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
    return out


def apply_yshift(frame_u8: np.ndarray, sc_row: np.ndarray, step: float) -> np.ndarray:
    """Apply per-frame [y_off, dy, dx] correction. frame_u8: (H, W, 3) uint8.

    y_off is added to the Y (luma) channel via additive RGB shift (codex pattern
    line 592: `x = x + raw[0] * step` is applied uniformly across channels —
    matching the codex_metric_yshift mode-7 wire). For PR106's RGB output, this
    is a uniform DC shift on all channels, which is the simplest portable form.
    """
    y_off = float(sc_row[0]) * step
    dy = int(sc_row[1])
    dx = int(sc_row[2])
    if y_off != 0.0:
        out = frame_u8.astype(np.float32) + y_off
        out = np.clip(out, 0, 255).round().astype(np.uint8)
    else:
        out = frame_u8
    return shift_rgb_uint8(out, dy, dx)


def inflate(src_bin: str, dst_raw: str) -> int:
    archive_bytes = Path(src_bin).read_bytes()
    decoder_sd, latents, meta, sidechannel = parse_yshift_archive(archive_bytes)

    if not torch.cuda.is_available():
        sys.exit("pr106_yshift_sidechannel inflate requires GPU (per CLAUDE.md MPS-auth-eval-is-NOISE)")
    device = torch.device("cuda")
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n_pairs = meta["n_pairs"]
    eval_h, eval_w = meta["eval_size"]
    sc_step = float(sidechannel["step"]) if sidechannel else 0.0
    sc_raw = sidechannel["raw"] if sidechannel else None
    expected_frames = n_pairs * 2
    if sidechannel and sidechannel["n_frames"] != expected_frames:
        raise ValueError(
            f"sidechannel frame count mismatch: SC01 has {sidechannel['n_frames']}, "
            f"decoder produces {expected_frames}"
        )
    print(
        f"pr106_yshift_sidechannel: decoder loaded; sidechannel={'present' if sidechannel else 'ABSENT'}, "
        f"step={sc_step}, n_pairs={n_pairs}",
        file=sys.stderr,
    )

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            B = j - i
            decoded = decoder(latents[i:j])  # (B, 2, 3, eval_h, eval_w)
            flat = decoded.reshape(B * 2, 3, eval_h, eval_w)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            frames = (
                up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
            )  # (B*2, H, W, 3) uint8
            if sc_raw is not None:
                for k in range(B * 2):
                    frame_idx = i * 2 + k
                    frames[k] = apply_yshift(frames[k], sc_raw[frame_idx], sc_step)
            fout.write(frames.tobytes())
            n += B * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m submissions.pr106_yshift_sidechannel.inflate <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
