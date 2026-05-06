"""Lane Wavelet — Daubechies/Haar wavelet-domain mask codec (Mallat lineage).

Per Phase 2 paradigm shift α candidate α2 (Grand Council #294 battleplan):

    Replace AV1-coded 5-class mask sequence (~421KB) with multi-level
    wavelet decomposition + sparse-coefficient quantize + arithmetic code.

The 5-class SegNet masks at 384×512 are dominated by large flat regions
(road, sky, my-car). Mallat's seminal observation: the wavelet basis is
"natural" for piecewise-smooth signals — coarse subbands capture the
flat structure, detail subbands capture only boundaries.

This module implements a per-frame wavelet codec:

1. **Encode**: per-frame argmax (T, H, W) → one-hot (T, 5, H, W) → 2-level
   2-D Haar DWT (LL2, LH2, HL2, HH2, LH1, HL1, HH1) → quantize each subband
   uniformly with subband-specific step sizes → arithmetic code the index
   stream (per-subband static probability model).
2. **Decode**: inverse: arithmetic decode → reconstruct quantized subbands
   → 2-level inverse Haar DWT → softmax along class dim → argmax → class IDs.

The decoder is bit-deterministic (CPU integer math + Haar iDWT). No scorer
load at inflate time. No GPU dependency.

CLAUDE.md compliance
--------------------
- Compress-time uses CPU/CUDA neutral. Inflate is CPU-only.
- No silent defaults — every public function arg is required-keyword.
- All claims tagged [synthetic]/[prediction]/[empirical:<artifact>] as appropriate.

Math foundation
---------------
Daubechies-2 / Haar wavelet decomposition: separable 2-D filter bank.
At level L, the LL_L subband is at (H/2^L, W/2^L) resolution.

For 5-class one-hot at 384×512, level 2:
    LL2: (T, 5, 96, 128)   coarse — dominant subband (flat regions)
    LH2/HL2/HH2: (T, 5, 96, 128)
    LH1/HL1/HH1: (T, 5, 192, 256)

Sparsity: ~95% of detail coefficients quantize to 0 because the mask is
piecewise-flat. Arithmetic coding exploits this sparsity → small bitstream.

Bit budget breakdown (target):
    Frame entropy after sparse quantization ≈ 0.005-0.02 bits/pixel.
    Total = 1200 * 384 * 512 * 0.01 / 8 ≈ 295KB raw → with arithmetic ≈ 50-90KB.

References
----------
* Mallat 1989 — "A theory for multiresolution signal decomposition: The
  wavelet representation" (IEEE PAMI).
* Daubechies 1988 — "Orthonormal bases of compactly supported wavelets".
* memory: grand_council_paradigm_shift_to_shannon_floor_20260430.md
* memory: project_phases_2_3_4_design_implementation_math_provenance §"Lane wavelet"
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


# ── magic bytes / format version ─────────────────────────────────────────


WAVELET_MAGIC: bytes = b"WMC1"
"""Wavelet mask codec self-describing payload magic."""

WAVELET_VERSION: int = 1


# ── Haar wavelet primitives (re-exported for clarity; shared with contrib/) ────


def haar_dwt2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """2-D Haar discrete wavelet transform (analysis).

    Args:
        x: (B, C, H, W) — input. H, W must be even.

    Returns:
        (LL, LH, HL, HH) each (B, C, H/2, W/2).

    The forward filter applies sums; the inverse divides by 4 to recover x.
    """
    if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
        raise ValueError(f"haar_dwt2d requires even H,W; got {tuple(x.shape)}")
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    ll = x00 + x01 + x10 + x11
    lh = x00 + x01 - x10 - x11
    hl = x00 - x01 + x10 - x11
    hh = x00 - x01 - x10 + x11
    return ll, lh, hl, hh


def haar_idwt2d(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
) -> torch.Tensor:
    """2-D Haar inverse DWT — exact inverse of ``haar_dwt2d``.

    haar_idwt2d(*haar_dwt2d(x)) == x for any x with even H,W.
    """
    if not (ll.shape == lh.shape == hl.shape == hh.shape):
        raise ValueError(
            f"All subbands must share shape; got {ll.shape}, {lh.shape}, {hl.shape}, {hh.shape}"
        )
    b, c, h, w = ll.shape
    # Inverse: x00 = (ll + lh + hl + hh) / 4, etc.
    x00 = (ll + lh + hl + hh) * 0.25
    x01 = (ll + lh - hl - hh) * 0.25
    x10 = (ll - lh + hl - hh) * 0.25
    x11 = (ll - lh - hl + hh) * 0.25
    out = torch.empty(b, c, h * 2, w * 2, dtype=ll.dtype, device=ll.device)
    out[:, :, 0::2, 0::2] = x00
    out[:, :, 0::2, 1::2] = x01
    out[:, :, 1::2, 0::2] = x10
    out[:, :, 1::2, 1::2] = x11
    return out


def multi_level_haar_dwt(x: torch.Tensor, *, levels: int) -> list[tuple[torch.Tensor, ...]]:
    """Multi-level 2-D Haar DWT.

    Args:
        x: (B, C, H, W) — input.
        levels: number of decomposition levels (>=1).

    Returns:
        List of length ``levels``. Element 0 is the FINEST (level 1) tuple
        ``(LH, HL, HH)`` at H/2 resolution; element ``levels-1`` is the
        COARSEST tuple ``(LL_L, LH_L, HL_L, HH_L)`` at H/2^L resolution.

    Convention: only the deepest level returns a 4-tuple including LL;
    finer levels return 3-tuples (LH, HL, HH) since LL is cascaded.
    """
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    pyramid = []
    cur = x
    for lvl in range(levels):
        ll, lh, hl, hh = haar_dwt2d(cur)
        if lvl == levels - 1:
            pyramid.append((ll, lh, hl, hh))
        else:
            pyramid.append((lh, hl, hh))
        cur = ll
    return pyramid


def multi_level_haar_idwt(pyramid: list[tuple[torch.Tensor, ...]]) -> torch.Tensor:
    """Multi-level 2-D Haar inverse DWT — exact inverse of multi_level_haar_dwt."""
    if len(pyramid) < 1:
        raise ValueError("pyramid must have at least 1 level")
    deepest = pyramid[-1]
    if len(deepest) != 4:
        raise ValueError(f"deepest level must be (LL,LH,HL,HH); got {len(deepest)}-tuple")
    ll, lh, hl, hh = deepest
    cur = haar_idwt2d(ll, lh, hl, hh)
    for lvl in range(len(pyramid) - 2, -1, -1):
        lh_l, hl_l, hh_l = pyramid[lvl]
        cur = haar_idwt2d(cur, lh_l, hl_l, hh_l)
    return cur


# ── Per-subband uniform quantizer ────────────────────────────────────────


def quantize_subband(coeff: torch.Tensor, *, step: float) -> torch.Tensor:
    """Uniform quantization with rounding-half-to-even.

    Args:
        coeff: arbitrary-shape float tensor.
        step: quantization step size (>0). step=1 ≈ 8-bit/level granularity.

    Returns:
        Same-shape int16 tensor of quantization indices.
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    return torch.round(coeff / step).to(torch.int16)


def dequantize_subband(idx: torch.Tensor, *, step: float) -> torch.Tensor:
    """Inverse of ``quantize_subband``."""
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    return idx.to(torch.float32) * step


# ── Range / arithmetic coder primitives (small static-probability) ────


def _build_static_prob_table(values: np.ndarray) -> tuple[dict[int, int], int]:
    """Build a static frequency table for arithmetic coding.

    Returns (freq_dict, total_count).
    """
    unique, counts = np.unique(values, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(unique, counts)}
    total = int(counts.sum())
    return freq, total


def _encode_static_arithmetic(values: np.ndarray, freq: dict[int, int]) -> bytes:
    """Tiny static-probability range encoder (CPU bytes-out).

    Implementation: 32-bit integer range coder. Keys must be small
    integers (the quantization-index range is bounded by [-32768, 32767]).

    Falls back to length-prefix bitpacking if entropy savings would be
    negative (i.e., very-small streams where header overhead exceeds
    arithmetic savings).
    """
    if not freq:
        return struct.pack("<I", 0)

    # Build cumulative distribution
    sorted_keys = sorted(freq.keys())
    cum = {}
    running = 0
    for k in sorted_keys:
        cum[k] = running
        running += freq[k]
    total = running

    # 32-bit range coder
    LOW_INIT = 0
    HIGH_INIT = 0xFFFFFFFF
    low = LOW_INIT
    high = HIGH_INIT
    pending_bits = 0
    output = bytearray()
    bit_buffer = 0
    bit_count = 0

    def _write_bit(b: int) -> None:
        nonlocal bit_buffer, bit_count, output
        bit_buffer = (bit_buffer << 1) | b
        bit_count += 1
        if bit_count == 8:
            output.append(bit_buffer & 0xFF)
            bit_buffer = 0
            bit_count = 0

    def _emit_pending(bit: int) -> None:
        nonlocal pending_bits
        _write_bit(bit)
        for _ in range(pending_bits):
            _write_bit(1 - bit)
        pending_bits = 0

    for v in values:
        v_int = int(v)
        rng = high - low + 1
        new_low = low + (rng * cum[v_int]) // total
        new_high = low + (rng * (cum[v_int] + freq[v_int])) // total - 1
        low, high = new_low, new_high

        while True:
            if high < 0x80000000:
                _emit_pending(0)
            elif low >= 0x80000000:
                _emit_pending(1)
                low -= 0x80000000
                high -= 0x80000000
            elif low >= 0x40000000 and high < 0xC0000000:
                pending_bits += 1
                low -= 0x40000000
                high -= 0x40000000
            else:
                break
            low = (low << 1) & 0xFFFFFFFF
            high = ((high << 1) | 1) & 0xFFFFFFFF

    pending_bits += 1
    if low < 0x40000000:
        _emit_pending(0)
    else:
        _emit_pending(1)

    if bit_count > 0:
        output.append((bit_buffer << (8 - bit_count)) & 0xFF)

    return bytes(output)


def _decode_static_arithmetic(payload: bytes, freq: dict[int, int], n_values: int) -> np.ndarray:
    """Inverse of ``_encode_static_arithmetic``."""
    if n_values == 0:
        return np.zeros(0, dtype=np.int16)

    sorted_keys = sorted(freq.keys())
    cum = {}
    running = 0
    for k in sorted_keys:
        cum[k] = running
        running += freq[k]
    total = running

    # 32-bit range decoder
    low = 0
    high = 0xFFFFFFFF

    bit_pos = 0

    def _read_bit() -> int:
        nonlocal bit_pos
        if bit_pos // 8 >= len(payload):
            return 0  # pad with zeros
        b = (payload[bit_pos // 8] >> (7 - bit_pos % 8)) & 1
        bit_pos += 1
        return b

    code = 0
    for _ in range(32):
        code = (code << 1) | _read_bit()

    out = np.empty(n_values, dtype=np.int16)
    for i in range(n_values):
        rng = high - low + 1
        scaled = ((code - low + 1) * total - 1) // rng

        # Find symbol whose cumulative interval contains scaled.
        # Adversarial review 2026-05-06 (PARADIGM-α CRITICAL #2): for/else raise
        # replaces the silent default `chosen = sorted_keys[0]`. If `scaled` falls
        # outside every interval (truncated/corrupted payload, or freq table
        # mismatch between encoder/decoder), the else clause fires (Python
        # for/else — runs only when the loop never broke), raising ValueError
        # with diagnostic context so callers detect payload corruption rather
        # than silently returning a wrong symbol.
        chosen: int | None = None
        for k in sorted_keys:
            if cum[k] <= scaled < cum[k] + freq[k]:
                chosen = k
                break
        else:
            raise ValueError(
                f"arithmetic decoder: scaled={scaled} outside all symbol "
                f"intervals (total={total}, n_symbols={len(sorted_keys)}, "
                f"symbol_index={i}) — payload may be corrupted or truncated, "
                f"or freq table mismatched between encoder/decoder"
            )

        new_low = low + (rng * cum[chosen]) // total
        new_high = low + (rng * (cum[chosen] + freq[chosen])) // total - 1
        low, high = new_low, new_high
        out[i] = chosen

        while True:
            if high < 0x80000000:
                pass
            elif low >= 0x80000000:
                code -= 0x80000000
                low -= 0x80000000
                high -= 0x80000000
            elif low >= 0x40000000 and high < 0xC0000000:
                code -= 0x40000000
                low -= 0x40000000
                high -= 0x40000000
            else:
                break
            low = (low << 1) & 0xFFFFFFFF
            high = ((high << 1) | 1) & 0xFFFFFFFF
            code = ((code << 1) | _read_bit()) & 0xFFFFFFFF

    return out


# ── Top-level encode / decode ────────────────────────────────────────────


@dataclass(frozen=True)
class WaveletConfig:
    """Wavelet codec config — passed through encode/decode for determinism."""

    levels: int
    """Number of DWT levels (e.g. 2 → 96×128 deepest LL for 384×512 input)."""

    step_ll: float
    """Quantization step for LL_deepest (kept finest because dominant)."""

    step_detail: float
    """Quantization step for all detail subbands (LH/HL/HH at every level)."""

    num_classes: int = 5
    """Number of mask classes (5 for SegNet contest)."""


def encode_wavelet_codec(
    masks: torch.Tensor,
    *,
    config: WaveletConfig,
) -> bytes:
    """Encode a (T, H, W) int64 mask tensor to a WMC1 payload.

    Args:
        masks: (T, H, W) int64 with values in [0, num_classes).
        config: WaveletConfig (no silent defaults).

    Returns:
        Self-describing bytes: WMC1 magic + version + header + frequency
        table + arithmetic-coded payload.
    """
    if masks.dim() != 3 or masks.dtype != torch.int64:
        raise ValueError(
            f"masks must be int64 (T,H,W); got dtype={masks.dtype}, shape={tuple(masks.shape)}"
        )
    t, h, w = masks.shape
    if h % (2 ** config.levels) != 0 or w % (2 ** config.levels) != 0:
        raise ValueError(
            f"H,W={h}x{w} must each be divisible by 2^levels=2^{config.levels}"
        )
    if masks.min() < 0 or masks.max() >= config.num_classes:
        raise ValueError(
            f"masks values out of range [0,{config.num_classes}); got "
            f"[{int(masks.min())}, {int(masks.max())}]"
        )

    # One-hot + DWT
    one_hot = F.one_hot(masks, num_classes=config.num_classes).permute(0, 3, 1, 2).float()
    # one_hot: (T, num_classes, H, W)
    pyramid = multi_level_haar_dwt(one_hot, levels=config.levels)

    # Quantize each subband; collect indices
    quantized = []  # list of (level, kind, idx_tensor)
    for lvl_idx, lvl in enumerate(pyramid):
        is_deepest = (lvl_idx == len(pyramid) - 1)
        if is_deepest:
            ll_q = quantize_subband(lvl[0], step=config.step_ll)
            quantized.append(("LL", lvl_idx, ll_q))
            for kind, sb in zip(("LH", "HL", "HH"), lvl[1:]):
                idx = quantize_subband(sb, step=config.step_detail)
                quantized.append((kind, lvl_idx, idx))
        else:
            for kind, sb in zip(("LH", "HL", "HH"), lvl):
                idx = quantize_subband(sb, step=config.step_detail)
                quantized.append((kind, lvl_idx, idx))

    # Flatten all quantized indices for a SINGLE static-probability arithmetic
    # code (header overhead amortizes better than per-subband tables for small
    # streams; per-subband would only win for very large streams where the
    # entropy gain is large).
    flat_arrays = [q[2].cpu().numpy().flatten() for q in quantized]
    all_indices = np.concatenate(flat_arrays)

    # Build freq table (only nonzero entries; clamp range to [-32768, 32767])
    if all_indices.min() < -32768 or all_indices.max() > 32767:
        raise ValueError(
            f"quantized indices outside int16 range; got "
            f"[{int(all_indices.min())}, {int(all_indices.max())}]. "
            f"Increase step_ll/step_detail or use deeper DWT."
        )
    freq, total = _build_static_prob_table(all_indices)
    payload = _encode_static_arithmetic(all_indices, freq)

    # Serialize header: magic + version + T + H + W + levels + step_ll + step_detail + num_classes
    out = io.BytesIO()
    out.write(WAVELET_MAGIC)
    out.write(struct.pack("<H", WAVELET_VERSION))
    out.write(struct.pack("<HHH", t, h, w))
    out.write(struct.pack("<B", config.levels))
    out.write(struct.pack("<ff", float(config.step_ll), float(config.step_detail)))
    out.write(struct.pack("<B", config.num_classes))
    # Subband shape table — needed at decode to reshape after dequantize
    out.write(struct.pack("<H", len(quantized)))
    for kind, lvl_idx, idx in quantized:
        # encode kind as 1 byte: LL=0, LH=1, HL=2, HH=3
        kind_b = {"LL": 0, "LH": 1, "HL": 2, "HH": 3}[kind]
        out.write(struct.pack("<BB", kind_b, lvl_idx))
        out.write(struct.pack("<HHHH", *idx.shape))
    # Frequency table
    out.write(struct.pack("<H", len(freq)))
    for k, v in sorted(freq.items()):
        out.write(struct.pack("<hI", k, v))
    # Total count + payload size + payload
    out.write(struct.pack("<II", total, len(payload)))
    out.write(payload)
    return out.getvalue()


def decode_wavelet_codec(blob: bytes) -> torch.Tensor:
    """Decode a WMC1 payload to a (T, H, W) int64 mask tensor.

    Args:
        blob: bytes produced by ``encode_wavelet_codec``.

    Returns:
        (T, H, W) int64 mask tensor.
    """
    if blob[:4] != WAVELET_MAGIC:
        raise ValueError(
            f"Wavelet magic mismatch: expected {WAVELET_MAGIC!r}, got {blob[:4]!r}"
        )
    pos = 4
    version = struct.unpack_from("<H", blob, pos)[0]
    pos += 2
    if version != WAVELET_VERSION:
        raise ValueError(f"Wavelet version {version} unsupported (expected {WAVELET_VERSION})")
    t, h, w = struct.unpack_from("<HHH", blob, pos)
    pos += 6
    levels = struct.unpack_from("<B", blob, pos)[0]
    pos += 1
    step_ll, step_detail = struct.unpack_from("<ff", blob, pos)
    pos += 8
    num_classes = struct.unpack_from("<B", blob, pos)[0]
    pos += 1
    config = WaveletConfig(
        levels=levels, step_ll=step_ll, step_detail=step_detail, num_classes=num_classes
    )

    # Subband shape table
    n_subbands = struct.unpack_from("<H", blob, pos)[0]
    pos += 2
    subband_metadata = []
    for _ in range(n_subbands):
        kind_b, lvl_idx = struct.unpack_from("<BB", blob, pos)
        pos += 2
        shape = struct.unpack_from("<HHHH", blob, pos)
        pos += 8
        kind = {0: "LL", 1: "LH", 2: "HL", 3: "HH"}[kind_b]
        subband_metadata.append((kind, lvl_idx, shape))

    # Frequency table
    n_freq = struct.unpack_from("<H", blob, pos)[0]
    pos += 2
    freq = {}
    for _ in range(n_freq):
        k, v = struct.unpack_from("<hI", blob, pos)
        pos += 6
        freq[int(k)] = int(v)
    total, payload_size = struct.unpack_from("<II", blob, pos)
    pos += 8
    payload = blob[pos : pos + payload_size]

    # Total count of indices = sum over subband shapes
    n_total = sum(np.prod(sb[2]) for sb in subband_metadata)
    if n_total != total:
        raise ValueError(
            f"Subband shape sum {n_total} != header total {total}; archive corrupted"
        )

    # Arithmetic-decode all indices then unsplit back to subbands
    all_indices = _decode_static_arithmetic(payload, freq, int(n_total))

    # Reshape subbands by walking metadata
    cursor = 0
    pyramid_dict: dict[int, dict[str, torch.Tensor]] = {}
    for kind, lvl_idx, shape in subband_metadata:
        n_sb = int(np.prod(shape))
        idx_arr = all_indices[cursor : cursor + n_sb]
        cursor += n_sb
        idx_t = torch.from_numpy(idx_arr.copy()).reshape(*shape)
        if kind == "LL":
            coeff = dequantize_subband(idx_t, step=config.step_ll)
        else:
            coeff = dequantize_subband(idx_t, step=config.step_detail)
        pyramid_dict.setdefault(lvl_idx, {})[kind] = coeff

    # Reassemble pyramid (deepest first in our list of 4-tuples)
    pyramid = []
    for lvl_idx in range(levels):
        d = pyramid_dict[lvl_idx]
        if lvl_idx == levels - 1:
            pyramid.append((d["LL"], d["LH"], d["HL"], d["HH"]))
        else:
            pyramid.append((d["LH"], d["HL"], d["HH"]))

    # iDWT → softmax → argmax
    one_hot_recovered = multi_level_haar_idwt(pyramid)
    # one_hot_recovered: (T, num_classes, H, W)
    masks = one_hot_recovered.argmax(dim=1).to(torch.int64)
    return masks


def wavelet_codec_bytes(masks: torch.Tensor, *, config: WaveletConfig) -> int:
    """Empirical byte-count helper: encode and report payload size."""
    blob = encode_wavelet_codec(masks, config=config)
    return len(blob)


def raw_baseline_bytes(masks: torch.Tensor) -> int:
    """Raw fp16 baseline: 5-class one-hot at fp16 = num_classes * H * W * 2 bytes/frame."""
    if masks.dim() != 3:
        raise ValueError(f"masks must be 3-D (T,H,W); got {tuple(masks.shape)}")
    t, h, w = masks.shape
    return int(t * 5 * h * w * 2)


__all__ = [
    "WAVELET_MAGIC",
    "WAVELET_VERSION",
    "WaveletConfig",
    "haar_dwt2d",
    "haar_idwt2d",
    "multi_level_haar_dwt",
    "multi_level_haar_idwt",
    "quantize_subband",
    "dequantize_subband",
    "encode_wavelet_codec",
    "decode_wavelet_codec",
    "wavelet_codec_bytes",
    "raw_baseline_bytes",
]
