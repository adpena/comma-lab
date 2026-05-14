"""D4 residual codec — int8-quantize + brotli-pack per-pair photometric residual.

The photometric residual ``frame_0_gt - warp(frame_1)`` is small (video
temporal correlation is high; residual energy is typically < 5% per pixel at
20 Hz). We quantize each residual at coarse spatial resolution then brotli
the byte stream.

Default budget:
- Spatial: 96×128 (1/16 of scorer 384×512) — captures ego-motion drift +
  photometric variation without modeling high-frequency texture.
- Channels: 3 (RGB).
- Quantization: int8 with per-pair (min, max) scale stored separately.
- Codec: brotli (deterministic q9, matching sister substrates).

Per-pair raw cost: 96 × 128 × 3 = 36,864 B raw int8; brotli typically closes
high-correlation residuals to 1-3 KB/pair. At 600 pairs target ~600 KB - 2 MB
which is too large for a sub-200 KB archive on its own; we therefore include
a `coarse_h, coarse_w` parameter that can be tuned (default 48×64 for
~512 KB unpacked; brotli closes to ~50-150 KB).

The budget-allocation strategy (operator decision): higher-motion pairs get
more bits via a larger residual grid OR finer quantization. v1 uses a
uniform per-pair budget; per-pair Fisher-weighted allocation is M10 in the
deep-math memo (sister mechanism, deferred to subsequent landing).
"""

from __future__ import annotations

import struct

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

_BROTLI_QUALITY: int = 9
"""Deterministic brotli quality; matches sister substrates."""

# Magic prefix for residual blob; allows future format evolution.
_RES_MAGIC: bytes = b"RES1"

# Quantization range: per-pair (min, max) stored as float16 (2 floats per channel).
_QUANT_DTYPE = np.int8


def encode_residual_blob(
    residual: torch.Tensor,
    *,
    coarse_hw: tuple[int, int] = (48, 64),
) -> bytes:
    """Quantize + brotli-pack the per-pair photometric residual.

    Args:
        residual: ``(num_pairs, 3, H, W)`` float tensor (typically in
            [-1, 1] range since it's a unit-range frame difference).
        coarse_hw: target spatial resolution for the residual (default
            48×64 = 1/8 scorer height/width). Lower resolution = smaller
            archive, but loses high-frequency reconstruction fidelity.

    Returns:
        brotli-compressed bytes encoding ``RES1 + header + scales + int8_data``.

    Header layout::

        MAGIC(4) "RES1"
        NUM_PAIRS(4) u32
        COARSE_H(2) u16
        COARSE_W(2) u16
        CHANNELS(1) u8 (== 3)
        FLAGS(1) u8 (reserved, == 0)

    After header:
        SCALES(num_pairs * 2 * 2) — per-pair (min, max) as float16 each
        DATA(num_pairs * 3 * coarse_h * coarse_w) — int8 values
    """
    if residual.dim() != 4 or residual.shape[1] != 3:
        raise ValueError(
            f"residual must be (N, 3, H, W); got {tuple(residual.shape)}"
        )
    num_pairs = int(residual.shape[0])
    target_h, target_w = coarse_hw
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"coarse_hw must be positive; got {coarse_hw}")

    # Bilinear-resize to coarse grid.
    coarse = torch.nn.functional.interpolate(
        residual,
        size=coarse_hw,
        mode="bilinear",
        align_corners=False,
    ).detach().cpu().to(torch.float32).numpy()  # (N, 3, h, w)

    # Per-pair quantization (range computed across all channels + pixels).
    flat = coarse.reshape(num_pairs, -1)
    pair_min = flat.min(axis=1).astype(np.float16)  # (N,)
    pair_max = flat.max(axis=1).astype(np.float16)  # (N,)
    # Avoid degenerate ranges
    span = np.maximum(pair_max.astype(np.float32) - pair_min.astype(np.float32), 1e-6)
    # int8 range [-127, +127]
    normalized = (coarse - pair_min.astype(np.float32).reshape(-1, 1, 1, 1)) / span.reshape(
        -1, 1, 1, 1
    )  # [0, 1]
    quantized = np.clip(np.round(normalized * 254.0 - 127.0), -127, 127).astype(_QUANT_DTYPE)

    header = struct.pack(
        "<4sIHHBB",
        _RES_MAGIC,
        num_pairs,
        int(target_h),
        int(target_w),
        3,
        0,
    )
    # Pack (min, max) per pair: 2 float16 each
    scales_bytes = bytearray()
    for i in range(num_pairs):
        scales_bytes.extend(struct.pack("<ee", float(pair_min[i]), float(pair_max[i])))
    data_bytes = quantized.tobytes(order="C")
    raw = header + bytes(scales_bytes) + data_bytes
    return bytes(brotli.compress(raw, quality=_BROTLI_QUALITY))


def decode_residual_blob(blob: bytes, *, expected_num_pairs: int) -> torch.Tensor:
    """Inverse of ``encode_residual_blob``.

    Args:
        blob: brotli-compressed bytes from ``encode_residual_blob``.
        expected_num_pairs: validation cross-check (must match header).

    Returns:
        ``(num_pairs, 3, coarse_h, coarse_w)`` float32 residual tensor.

    Raises:
        ValueError: on header mismatch or truncation.
    """
    raw = brotli.decompress(blob)
    if len(raw) < 14:
        raise ValueError(f"residual blob too short ({len(raw)} bytes)")
    magic, num_pairs, coarse_h, coarse_w, channels, flags = struct.unpack(
        "<4sIHHBB", raw[:14]
    )
    if magic != _RES_MAGIC:
        raise ValueError(f"bad residual magic: {magic!r} (expected {_RES_MAGIC!r})")
    if num_pairs != expected_num_pairs:
        raise ValueError(
            f"residual num_pairs {num_pairs} != expected {expected_num_pairs}"
        )
    if channels != 3:
        raise ValueError(f"residual channels {channels} != 3")
    if flags != 0:
        raise ValueError(f"residual flags must be 0; got {flags}")

    pos = 14
    scales_byte_len = num_pairs * 4  # 2 float16 per pair
    if pos + scales_byte_len > len(raw):
        raise ValueError("residual blob truncated in scales section")
    scales = np.frombuffer(
        raw[pos : pos + scales_byte_len], dtype=np.float16
    ).reshape(num_pairs, 2).astype(np.float32)
    pos += scales_byte_len
    pair_min = scales[:, 0]
    pair_max = scales[:, 1]

    data_byte_len = num_pairs * channels * coarse_h * coarse_w
    if pos + data_byte_len != len(raw):
        raise ValueError(
            f"residual blob size mismatch: have {len(raw) - pos} bytes for "
            f"data, expected {data_byte_len}"
        )
    quantized = np.frombuffer(raw[pos:], dtype=_QUANT_DTYPE).reshape(
        num_pairs, channels, coarse_h, coarse_w
    ).astype(np.float32)

    # Inverse quantize: normalized = (q + 127) / 254
    normalized = (quantized + 127.0) / 254.0  # [0, 1]
    span = np.maximum(pair_max - pair_min, 1e-6).reshape(-1, 1, 1, 1)
    residual = normalized * span + pair_min.reshape(-1, 1, 1, 1)
    return torch.from_numpy(residual.copy())


__all__ = ["decode_residual_blob", "encode_residual_blob"]
