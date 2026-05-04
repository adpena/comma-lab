"""Flat-FP4 codec: encode/decode model weights for inflate.py.

Both sides MUST agree on:
  - SCHEMA: ordered list of (key, kind, shape)
  - block_size = 32
  - FP4 levels = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
  - Nibble layout: bit3=sign, bits0-2=level idx
  - Pack: high nibble first, two per byte

Imported by both build_archive.py (encode) and inflate.py (decode).
"""
import io
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BLOCK_SIZE = 32
FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


# ── Encoding ─────────────────────────────────────────────────────────
def quant_to_nibbles(weight, block_size=BLOCK_SIZE):
    flat = weight.detach().float().reshape(-1).cpu()
    n = flat.numel()
    pad = (block_size - n % block_size) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])
    blocks = flat.view(-1, block_size)
    ma = blocks.abs().amax(1, keepdim=True)
    scales = torch.where(ma > 0, ma / 6.0, torch.ones_like(ma))
    norm = blocks / scales
    signs = (norm < 0).to(torch.uint8)
    levels = FP4_LEVELS.view(1, 1, -1).float()
    mag_idx = (norm.abs().unsqueeze(-1) - levels).abs().argmin(-1).to(torch.uint8)
    # Keep ALL padded nibbles (block-aligned) so decoder can reconstruct full blocks.
    nibbles = ((signs.view(-1, block_size) << 3) | mag_idx).view(-1)
    return nibbles.to(torch.uint8), scales.view(-1).to(torch.float16)


def pack_nibbles(nibbles_u8):
    n = nibbles_u8.numel()
    pad = n % 2
    if pad:
        nibbles_u8 = torch.cat([nibbles_u8, torch.zeros(1, dtype=torch.uint8)])
    pairs = nibbles_u8.view(-1, 2)
    return ((pairs[:, 0] << 4) | (pairs[:, 1] & 0x0F)).numpy().tobytes()


def encode(sd, schema):
    parts = []
    for key, kind, shape in schema:
        t = sd[key]
        if kind == 'fp4_w':
            nibbles, scales = quant_to_nibbles(t)
            parts.append(pack_nibbles(nibbles))
            parts.append(scales.numpy().tobytes())
        elif kind in ('fp16_w', 'fp16_b'):
            parts.append(t.detach().to(torch.float16).cpu().numpy().tobytes())
        else:
            raise ValueError(kind)
    return b"".join(parts)


# ── Decoding ─────────────────────────────────────────────────────────
def unpack_nibbles(packed_bytes, count):
    arr = np.frombuffer(packed_bytes, dtype=np.uint8)
    out = np.empty(arr.size * 2, dtype=np.uint8)
    out[0::2] = (arr >> 4) & 0x0F
    out[1::2] = arr & 0x0F
    return out[:count]


def dequantize_nibbles(nibbles_u8, scales_fp16, shape):
    """Reverse of quant_to_nibbles + pack_nibbles."""
    n = int(np.prod(shape))
    pad = (BLOCK_SIZE - n % BLOCK_SIZE) % BLOCK_SIZE
    nib_padded_count = n + pad
    nibbles = torch.from_numpy(nibbles_u8.astype(np.int64))
    if nibbles.numel() < nib_padded_count:
        # nibbles came from packed bytes — possibly an extra nibble at the end
        nibbles = nibbles[:nib_padded_count]
    elif nibbles.numel() > nib_padded_count:
        nibbles = nibbles[:nib_padded_count]
    blocks = nibbles.view(-1, BLOCK_SIZE)
    signs = (blocks >> 3) & 1
    mag_idx = blocks & 0x7
    levels = FP4_LEVELS.to(torch.float32)
    mags = levels[mag_idx]
    vals = torch.where(signs.bool(), -mags, mags)
    scales = torch.from_numpy(scales_fp16.astype(np.float32)).view(-1, 1)
    out = (vals * scales).view(-1)[:n]
    return out.view(*shape).to(torch.float32)


def decode(payload_bytes, schema):
    """Returns dict[name] = tensor (float32 / on cpu)."""
    sd = {}
    o = 0
    for key, kind, shape in schema:
        n = int(np.prod(shape))
        if kind == 'fp4_w':
            n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
            n_nibbles_padded = n_blocks * BLOCK_SIZE
            packed_count = (n_nibbles_padded + 1) // 2  # 2 nibbles per byte
            scale_count = n_blocks
            packed_bytes = payload_bytes[o : o + packed_count]
            o += packed_count
            scales_bytes = payload_bytes[o : o + scale_count * 2]
            o += scale_count * 2
            nibbles = unpack_nibbles(packed_bytes, n_nibbles_padded)
            scales_fp16 = np.frombuffer(scales_bytes, dtype=np.float16)
            sd[key] = dequantize_nibbles(nibbles, scales_fp16, shape)
        elif kind in ('fp16_w', 'fp16_b'):
            byte_count = n * 2
            buf = payload_bytes[o : o + byte_count]
            o += byte_count
            arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32).reshape(*shape)
            sd[key] = torch.from_numpy(arr.copy())
        else:
            raise ValueError(kind)
    if o != len(payload_bytes):
        raise RuntimeError(f"flat_fp4.decode: trailing bytes {o} vs {len(payload_bytes)}")
    return sd


# ── Schema builder (used at encode time only — decoder ships static SCHEMA) ──
def build_schema(gen):
    schema = []
    for name, m in gen.named_modules():
        if isinstance(m, nn.Conv2d):
            quantize = getattr(m, 'quantize_weight', True)
            schema.append((f"{name}.weight", 'fp4_w' if quantize else 'fp16_w', tuple(m.weight.shape)))
            if m.bias is not None:
                schema.append((f"{name}.bias", 'fp16_b', tuple(m.bias.shape)))
        elif isinstance(m, nn.Embedding):
            quantize = getattr(m, 'quantize_weight', True)
            schema.append((f"{name}.weight", 'fp4_w' if quantize else 'fp16_w', tuple(m.weight.shape)))
        elif isinstance(m, nn.Linear):
            schema.append((f"{name}.weight", 'fp16_w', tuple(m.weight.shape)))
            if m.bias is not None:
                schema.append((f"{name}.bias", 'fp16_b', tuple(m.bias.shape)))
        elif isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                schema.append((f"{name}.weight", 'fp16_w', tuple(m.weight.shape)))
            if m.bias is not None:
                schema.append((f"{name}.bias", 'fp16_b', tuple(m.bias.shape)))
    return schema
