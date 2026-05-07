"""PR #101 (hnerv_ft_microcodec) split-Brotli + per-tensor byte-map decoder codec.

This is a 1:1 port of PR101's ``submissions/hnerv_ft_microcodec/src/codec.py``
``decode_decoder_compact`` path, plus the encoder side that PR101 only ships
inside their compress-time pipeline. The encoder produces a ``decoder_blob``
that PR101's decoder reads back losslessly.

Source-of-truth: ``experiments/results/public_pr_intake_full/
public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/src/
codec.py`` (480 LOC), pre-read in this session.

What PR101's codec achieves vs PR106's monolithic-brotli decoder section:
    PR106 ``decoder_packed_brotli``: 170,127 bytes (single brotli over a fixed
                                     INT8-zigzag schema)
    PR101 ``decoder_blob``:          162,164 bytes (split into 7 brotli streams
                                     + per-tensor byte permutations)

That gap (-7,963 bytes / -4.7%) maps to score Δ ≈ -0.0053 rate component if
applied to the PR106 substrate (math: ``25 × 7963 / 37545489``).

The encoder primitives we expose:

- :func:`encode_decoder_compact` — torch ``state_dict`` → bytes (the
  ``decoder_blob``).
- :func:`decode_decoder_compact` — bytes → torch ``state_dict``. Verbatim port
  of PR101's function (sans the ``HNeRVDecoder`` import — we use the
  in-repo schema instead).
- :func:`validate_byte_map_savings` — Contrarian gate: per-tensor measurement
  of the brotli-stream output bytes WITH vs WITHOUT the byte-map permutation.
  If a byte_map REGRESSES brotli output for a particular tensor on the
  caller's state_dict, log a WARNING — operator decides whether to skip the
  byte_map for that tensor.

Strict-scorer-rule: this module loads NO scorer weights and has no MPS/CUDA
dependency. CPU-only, deterministic, byte-faithful round-trip.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import brotli
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants ported from PR101 src/codec.py:22-87
# ---------------------------------------------------------------------------

DECODER_BLOB_LEN = 162_164
"""PR101's authored decoder_blob length on their trained checkpoint. This is
their hardcoded length-prefix-eliding constant; on synthetic / different
weights the actual length will differ."""

LATENT_BLOB_LEN = 15_387
N_PAIRS = 600
LATENT_DIM = 28
BASE_CHANNELS = 36
EVAL_SIZE = (384, 512)

# Storage permutation of tensor indices used for split-brotli ordering. This
# is the order that minimizes split-stream output bytes on PR101's trained
# weights (a permutation of range(28)).
DECODER_STORAGE_ORDER = (
    14, 22, 7, 6, 19, 10, 25, 4, 20, 9, 12, 15, 5, 11,
    18, 1, 21, 3, 27, 13, 2, 26, 24, 17, 16, 23, 8, 0,
)

# Stream-end indices into DECODER_STORAGE_ORDER. Means: tensors 0..1 are one
# brotli stream, 1..2 another, 2..22 a third (largest), etc. 7 streams total.
DECODER_STREAM_ENDS = (1, 2, 22, 23, 26, 27, 28)

# Per-tensor 4D storage permutation (axis order on disk). Inverse permutation
# is applied at decode time. Picked at compress time per tensor for max
# brotli redundancy.
CONV4_STORAGE_PERMS: dict[int, tuple[int, int, int, int]] = {
    2: (3, 0, 2, 1),
    4: (3, 0, 2, 1),
    6: (0, 1, 2, 3),
    8: (3, 0, 1, 2),
    10: (3, 0, 2, 1),
    12: (3, 0, 1, 2),
    14: (1, 0, 2, 3),
    16: (3, 0, 2, 1),
    18: (1, 0, 2, 3),
    20: (0, 3, 2, 1),
    22: (0, 3, 2, 1),
    24: (0, 2, 3, 1),
    26: (0, 1, 3, 2),
}
CONV4_INVERSE_PERMS: dict[int, tuple[int, int, int, int]] = {
    idx: tuple(int(value) for value in np.argsort(perm))
    for idx, perm in CONV4_STORAGE_PERMS.items()
}

# Per-tensor zigzag variant. PR101 picked these by sweeping at compress time
# for each tensor. Values: "zig" (default), "negzig", "twos", "off".
DECODER_BYTE_MAPS: dict[int, str] = {
    9: "negzig",
    14: "negzig",
    20: "twos",
    27: "off",
}

# Quantization range. PR101 uses signed-7-bit (clamp to [-127, 127]) which
# matches PR106's N_QUANT and keeps zigzag output in u8.
N_QUANT = 127


# ---------------------------------------------------------------------------
# State-schema definition (matches PR106's FIXED_STATE_SCHEMA + PR101 model)
# ---------------------------------------------------------------------------

# Same architecture as PR106 (see model.py / ``HNeRVDecoder`` in the PR101
# intake), so the (name, shape) tuples are identical. We hardcode them here
# to keep this module zero-dep on tac.hnerv_decoder_recode (which would pull
# in the entire PR106 recode pipeline).

FIXED_STATE_SCHEMA: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("stem.weight", (1728, 28)),
    ("stem.bias", (1728,)),
    ("blocks.0.weight", (144, 36, 3, 3)),
    ("blocks.0.bias", (144,)),
    ("blocks.1.weight", (144, 36, 3, 3)),
    ("blocks.1.bias", (144,)),
    ("blocks.2.weight", (108, 36, 3, 3)),
    ("blocks.2.bias", (108,)),
    ("blocks.3.weight", (80, 27, 3, 3)),
    ("blocks.3.bias", (80,)),
    ("blocks.4.weight", (72, 20, 3, 3)),
    ("blocks.4.bias", (72,)),
    ("blocks.5.weight", (72, 18, 3, 3)),
    ("blocks.5.bias", (72,)),
    ("skips.2.weight", (27, 36, 1, 1)),
    ("skips.2.bias", (27,)),
    ("skips.3.weight", (20, 27, 1, 1)),
    ("skips.3.bias", (20,)),
    ("skips.4.weight", (18, 20, 1, 1)),
    ("skips.4.bias", (18,)),
    ("refine.0.weight", (9, 18, 3, 3)),
    ("refine.0.bias", (9,)),
    ("refine.1.weight", (18, 9, 3, 3)),
    ("refine.1.bias", (18,)),
    ("rgb_0.weight", (3, 18, 3, 3)),
    ("rgb_0.bias", (3,)),
    ("rgb_1.weight", (3, 18, 3, 3)),
    ("rgb_1.bias", (3,)),
)


class Pr101SplitBrotliCodecError(ValueError):
    """Raised when a PR101 split-Brotli payload is invalid or input mismatched."""


# ---------------------------------------------------------------------------
# Zigzag / byte-map helpers (ported from PR101 codec.py:225-239)
# ---------------------------------------------------------------------------

def _zigzag_encode_i8(arr_i8: np.ndarray) -> np.ndarray:
    """Encode signed int8 → unsigned u8 using zigzag (same as PR106 encoder)."""
    arr = arr_i8.astype(np.int32)
    return np.where(arr >= 0, 2 * arr, -2 * arr - 1).astype(np.uint8)


def _zigzag_decode_u8(arr_u8: np.ndarray) -> np.ndarray:
    arr = arr_u8.astype(np.int32)
    return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)


def _encode_mapped_u8(q_i8: np.ndarray, byte_map: str) -> np.ndarray:
    """Encode signed int8 quantized values to uint8 bytes via ``byte_map``."""
    if byte_map == "zig":
        return _zigzag_encode_i8(q_i8)
    if byte_map == "negzig":
        return _zigzag_encode_i8((-q_i8.astype(np.int16)).astype(np.int8))
    if byte_map == "off":
        # Map int8 → uint8 by adding 128 (signed offset binary).
        return (q_i8.astype(np.int16) + 128).astype(np.uint8)
    if byte_map == "twos":
        # Reinterpret int8 bytes as uint8 (two's complement).
        return q_i8.view(np.uint8) if q_i8.dtype == np.int8 else q_i8.astype(np.int8).view(np.uint8)
    raise Pr101SplitBrotliCodecError(f"unknown decoder byte map: {byte_map}")


def decode_mapped_u8(arr_u8: np.ndarray, byte_map: str) -> np.ndarray:
    """Inverse of :func:`_encode_mapped_u8`. Verbatim port of PR101's helper."""
    if byte_map == "zig":
        return _zigzag_decode_u8(arr_u8)
    if byte_map == "negzig":
        return (-_zigzag_decode_u8(arr_u8).astype(np.int16)).astype(np.int8)
    if byte_map == "off":
        return (arr_u8.astype(np.int16) - 128).astype(np.int8)
    if byte_map == "twos":
        return arr_u8.view(np.int8)
    raise Pr101SplitBrotliCodecError(f"unknown decoder byte map: {byte_map}")


# ---------------------------------------------------------------------------
# 4D conv permutation helpers
# ---------------------------------------------------------------------------

def apply_conv4_perm(values: np.ndarray, idx: int, *, inverse: bool = False) -> np.ndarray:
    """Apply (or invert) the per-tensor 4D storage permutation."""
    if values.ndim != 4:
        raise Pr101SplitBrotliCodecError(
            f"apply_conv4_perm requires 4D array, got ndim={values.ndim}"
        )
    if idx not in CONV4_STORAGE_PERMS:
        return values
    perm = CONV4_INVERSE_PERMS[idx] if inverse else CONV4_STORAGE_PERMS[idx]
    return np.transpose(values, perm).copy()


def apply_byte_map_inverse(arr_u8: np.ndarray, byte_map: str) -> np.ndarray:
    """Public alias to :func:`decode_mapped_u8` (invert the byte-map permutation)."""
    return decode_mapped_u8(arr_u8, byte_map)


# ---------------------------------------------------------------------------
# Brotli stream packing / unpacking
# ---------------------------------------------------------------------------

def pack_brotli_stream(raw: bytes, *, quality: int = 11) -> bytes:
    """Compress one schema-window of bytes via brotli at the given quality."""
    return brotli.compress(raw, quality=quality)


def decompress_brotli_streams(data: bytes, n_streams: int) -> bytes:
    """Decompress ``n_streams`` concatenated brotli streams (verbatim port)."""
    outputs: list[bytes] = []
    pos = 0
    for _ in range(n_streams):
        dec = brotli.Decompressor()
        chunks: list[bytes] = []
        while pos < len(data) and not dec.is_finished():
            chunks.append(dec.process(data[pos:pos + 1]))
            pos += 1
        if not dec.is_finished():
            raise Pr101SplitBrotliCodecError("truncated compact decoder payload")
        outputs.append(b"".join(chunks))
    if pos != len(data):
        raise Pr101SplitBrotliCodecError("trailing compact decoder payload")
    return b"".join(outputs)


# ---------------------------------------------------------------------------
# Quantization (per-tensor symmetric INT8 with fp16 scale)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _QuantizedTensor:
    name: str
    shape: tuple[int, ...]
    q_i8: np.ndarray  # int8, shape == ``shape``
    scale: float  # stored as float16 in the blob


def _quantize_tensor(name: str, tensor: torch.Tensor, *, n_quant: int = N_QUANT) -> _QuantizedTensor:
    """Per-tensor symmetric INT8 quant. Identical to PR106's quantize_state_dict."""
    t = tensor.detach().cpu().float()
    abs_max = t.abs().max().item()
    if abs_max > 0:
        scale = abs_max / n_quant
    else:
        scale = 1.0
    q = (t / scale).round().clamp(-n_quant, n_quant).to(torch.int8).numpy()
    return _QuantizedTensor(
        name=name,
        shape=tuple(int(s) for s in tensor.shape),
        q_i8=q,
        scale=float(scale),
    )


# ---------------------------------------------------------------------------
# Top-level encode / decode
# ---------------------------------------------------------------------------

def _build_per_tensor_payload(qt: _QuantizedTensor, idx: int) -> bytes:
    """Apply byte-map (+ optional 4D permutation) and append fp16 scale."""
    name, expected_shape = FIXED_STATE_SCHEMA[idx]
    if qt.name != name:
        raise Pr101SplitBrotliCodecError(
            f"schema mismatch at idx {idx}: expected {name!r}, got {qt.name!r}"
        )
    if qt.shape != expected_shape:
        raise Pr101SplitBrotliCodecError(
            f"shape mismatch for {name!r}: schema {expected_shape}, tensor {qt.shape}"
        )
    q = qt.q_i8
    if len(qt.shape) == 4 and idx in CONV4_STORAGE_PERMS:
        # Permute to storage order; flatten via the storage-order axes.
        q_storage = np.transpose(q, CONV4_STORAGE_PERMS[idx]).copy()
        flat = q_storage.reshape(-1)
    else:
        flat = q.reshape(-1)
    byte_map = DECODER_BYTE_MAPS.get(idx, "zig")
    mapped = _encode_mapped_u8(flat, byte_map)
    # fp16 scale (2 bytes, little-endian); fp16 is the on-disk format PR101 uses.
    scale_bytes = np.array([qt.scale], dtype=np.float16).tobytes()
    return mapped.tobytes() + scale_bytes


def encode_decoder_compact(
    state_dict: dict[str, torch.Tensor],
    *,
    brotli_quality: int = 11,
) -> bytes:
    """Encode a torch ``state_dict`` to a PR101-compatible ``decoder_blob``.

    The output bytes can be losslessly decoded by :func:`decode_decoder_compact`
    AND by PR101's own ``decode_decoder_compact`` (in their codec.py).

    Args:
        state_dict: HNeRVDecoder state dict keyed by FIXED_STATE_SCHEMA names.
        brotli_quality: Brotli compression level (PR101 ships at 11).

    Returns:
        Concatenated bytes of len(DECODER_STREAM_ENDS) brotli streams.
    """
    # 1) Quantize every tensor in schema order.
    quantized: list[_QuantizedTensor] = []
    schema_names = {name for name, _ in FIXED_STATE_SCHEMA}
    for name in schema_names:
        if name not in state_dict:
            raise Pr101SplitBrotliCodecError(f"missing tensor {name!r} in state_dict")
    for idx, (name, _shape) in enumerate(FIXED_STATE_SCHEMA):
        quantized.append(_quantize_tensor(name, state_dict[name]))

    # 2) Build per-tensor on-disk payloads (mapped bytes + fp16 scale), in
    #    DECODER_STORAGE_ORDER (NOT schema order).
    parts_by_storage: list[bytes] = []
    for storage_idx in DECODER_STORAGE_ORDER:
        parts_by_storage.append(_build_per_tensor_payload(quantized[storage_idx], storage_idx))

    # 3) Group payloads into split-stream windows and brotli each window.
    streams: list[bytes] = []
    start = 0
    for end in DECODER_STREAM_ENDS:
        window_raw = b"".join(parts_by_storage[start:end])
        streams.append(pack_brotli_stream(window_raw, quality=brotli_quality))
        start = end

    return b"".join(streams)


def decode_decoder_compact(data: bytes) -> dict[str, torch.Tensor]:
    """Decode a PR101 ``decoder_blob`` to a torch ``state_dict``.

    Verbatim port of PR101's ``decode_decoder_compact`` (codec.py:259-292),
    minus the ``HNeRVDecoder``-instantiation step (we use ``FIXED_STATE_SCHEMA``
    directly to avoid pulling in the model module).
    """
    raw = decompress_brotli_streams(data, len(DECODER_STREAM_ENDS))
    pos = 0
    sd: dict[str, torch.Tensor] = {}

    for idx in DECODER_STORAGE_ORDER:
        name, shape = FIXED_STATE_SCHEMA[idx]
        numel = int(np.prod(shape))
        zz = np.frombuffer(raw, dtype=np.uint8, count=numel, offset=pos)
        pos += numel
        scale = np.frombuffer(raw, dtype=np.float16, count=1, offset=pos)[0]
        pos += 2

        q = decode_mapped_u8(zz, DECODER_BYTE_MAPS.get(idx, "zig"))
        if len(shape) == 4:
            storage_perm = CONV4_STORAGE_PERMS[idx]
            inverse_perm = CONV4_INVERSE_PERMS[idx]
            stored_shape = tuple(shape[i] for i in storage_perm)
            q = q.reshape(stored_shape)
            q = np.transpose(q, inverse_perm).copy()
        else:
            q = q.reshape(shape)
        sd[name] = torch.from_numpy(q.astype(np.float32)) * float(scale)

    if pos != len(raw):
        raise Pr101SplitBrotliCodecError("trailing or truncated compact decoder payload")
    return sd


# ---------------------------------------------------------------------------
# Contrarian gate: validate that every byte_map actually saves bytes
# ---------------------------------------------------------------------------

def validate_byte_map_savings(
    state_dict: dict[str, torch.Tensor],
    *,
    brotli_quality: int = 11,
) -> dict[int, dict[str, int]]:
    """For each tensor with a non-default ``byte_map``, measure the per-tensor
    brotli output bytes WITH vs WITHOUT the byte_map.

    Returns a dict keyed by tensor schema-index containing:
        with_map_bytes:      brotli(mapped(q) + scale_fp16)
        without_map_bytes:   brotli(zigzag(q) + scale_fp16) — i.e. default 'zig'
        delta_bytes:         with_map_bytes - without_map_bytes
                             (negative means byte_map saves bytes)
        byte_map:            the assigned byte_map string

    If ``delta_bytes > 0`` for any tensor, a WARNING is logged. Operators should
    decide whether to skip that byte_map for that tensor on the input
    state_dict (PR101's authored byte_maps are chosen for THEIR weights —
    different weights can regress).
    """
    quantized = [
        _quantize_tensor(name, state_dict[name])
        for name, _shape in FIXED_STATE_SCHEMA
    ]
    results: dict[int, dict[str, int]] = {}
    for idx in sorted(DECODER_BYTE_MAPS.keys()):
        # With the assigned byte_map.
        with_payload = _build_per_tensor_payload(quantized[idx], idx)
        with_bytes = len(pack_brotli_stream(with_payload, quality=brotli_quality))

        # Without (force default 'zig').
        qt = quantized[idx]
        if len(qt.shape) == 4 and idx in CONV4_STORAGE_PERMS:
            q_storage = np.transpose(qt.q_i8, CONV4_STORAGE_PERMS[idx]).copy()
            flat = q_storage.reshape(-1)
        else:
            flat = qt.q_i8.reshape(-1)
        zig_mapped = _encode_mapped_u8(flat, "zig")
        scale_bytes = np.array([qt.scale], dtype=np.float16).tobytes()
        without_payload = zig_mapped.tobytes() + scale_bytes
        without_bytes = len(pack_brotli_stream(without_payload, quality=brotli_quality))

        delta = with_bytes - without_bytes
        results[idx] = {
            "with_map_bytes": int(with_bytes),
            "without_map_bytes": int(without_bytes),
            "delta_bytes": int(delta),
            "byte_map": DECODER_BYTE_MAPS[idx],
        }
        if delta > 0:
            logger.warning(
                "PR101 byte_map %s for tensor idx=%d (%s) REGRESSES brotli "
                "by %d bytes vs default 'zig' on this state_dict; consider "
                "skipping it for this checkpoint.",
                DECODER_BYTE_MAPS[idx],
                idx,
                FIXED_STATE_SCHEMA[idx][0],
                delta,
            )
    return results


__all__ = [
    "BASE_CHANNELS",
    "CONV4_INVERSE_PERMS",
    "CONV4_STORAGE_PERMS",
    "DECODER_BLOB_LEN",
    "DECODER_BYTE_MAPS",
    "DECODER_STORAGE_ORDER",
    "DECODER_STREAM_ENDS",
    "EVAL_SIZE",
    "FIXED_STATE_SCHEMA",
    "LATENT_BLOB_LEN",
    "LATENT_DIM",
    "N_PAIRS",
    "N_QUANT",
    "Pr101SplitBrotliCodecError",
    "apply_byte_map_inverse",
    "apply_conv4_perm",
    "decode_decoder_compact",
    "decode_mapped_u8",
    "decompress_brotli_streams",
    "encode_decoder_compact",
    "pack_brotli_stream",
    "validate_byte_map_savings",
]
