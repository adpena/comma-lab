# SPDX-License-Identifier: MIT
"""Lossless bit-level layouts for integer packet payloads.

These are pre-entropy-coder transforms: they do not change the represented
integer symbols, but they can expose lower-entropy planes to Brotli/range/ANS
style coders. Decoders must know the original byte length from the tensor or
section manifest; no hidden length authority is embedded here.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np

IntPayloadLayout = Literal["flat", "nibble_planes", "bitplanes_lsb"]
DEFAULT_INT_PAYLOAD_LAYOUTS: Final[tuple[IntPayloadLayout, ...]] = (
    "flat",
    "nibble_planes",
    "bitplanes_lsb",
)

VALID_INT_PAYLOAD_LAYOUTS: Final[frozenset[str]] = frozenset(
    DEFAULT_INT_PAYLOAD_LAYOUTS
)


def encode_int_payload_layout(payload: bytes, layout: IntPayloadLayout) -> bytes:
    """Encode a byte payload under a lossless bit-level layout."""

    raw = _payload_u8(payload)
    if layout == "flat":
        return bytes(payload)
    if layout == "nibble_planes":
        hi = (raw >> 4) & 0xF
        lo = raw & 0xF
        return _pack_nibbles_padded(hi) + _pack_nibbles_padded(lo)
    if layout == "bitplanes_lsb":
        planes = []
        for bit in range(8):
            plane = ((raw >> bit) & 1).astype(np.uint8, copy=False)
            planes.append(np.packbits(plane, bitorder="little").tobytes())
        return b"".join(planes)
    raise ValueError(f"unknown int payload layout: {layout!r}")


def decode_int_payload_layout(
    encoded: bytes,
    *,
    layout: IntPayloadLayout,
    raw_len: int,
) -> bytes:
    """Decode bytes produced by :func:`encode_int_payload_layout`."""

    if raw_len < 0:
        raise ValueError(f"raw_len must be >= 0; got {raw_len}")
    packed = _payload_u8(encoded)
    if layout == "flat":
        if packed.size != raw_len:
            raise ValueError(
                f"flat payload length {packed.size} does not match raw_len {raw_len}"
            )
        return bytes(encoded)
    if layout == "nibble_planes":
        half = (raw_len + 1) // 2
        expected = 2 * half
        if packed.size != expected:
            raise ValueError(
                f"nibble_planes length {packed.size} does not match expected {expected}"
            )
        hi = _unpack_nibbles(packed[:half].tobytes(), raw_len)
        lo = _unpack_nibbles(packed[half:].tobytes(), raw_len)
        return (((hi << 4) | lo).astype(np.uint8, copy=False)).tobytes()
    if layout == "bitplanes_lsb":
        plane_len = (raw_len + 7) // 8
        expected = 8 * plane_len
        if packed.size != expected:
            raise ValueError(
                f"bitplanes_lsb length {packed.size} does not match expected {expected}"
            )
        out = np.zeros(raw_len, dtype=np.uint8)
        for bit in range(8):
            start = bit * plane_len
            stop = start + plane_len
            bits = np.unpackbits(packed[start:stop], bitorder="little")[:raw_len]
            out |= (bits.astype(np.uint8, copy=False) << bit)
        return out.tobytes()
    raise ValueError(f"unknown int payload layout: {layout!r}")


def _payload_u8(payload: bytes) -> np.ndarray:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"payload must be bytes-like; got {type(payload)!r}")
    return np.frombuffer(bytes(payload), dtype=np.uint8)


def _pack_nibbles_padded(nibbles: np.ndarray) -> bytes:
    raw = np.asarray(nibbles, dtype=np.uint8).reshape(-1)
    if raw.size and int(raw.max()) > 0xF:
        raise ValueError("nibble values must fit in 4 bits")
    if raw.size & 1:
        raw = np.pad(raw, (0, 1), constant_values=0).astype(np.uint8, copy=False)
    hi = raw[0::2] & 0xF
    lo = raw[1::2] & 0xF
    return ((hi << 4) | lo).astype(np.uint8, copy=False).tobytes()


def _unpack_nibbles(packed: bytes, count: int) -> np.ndarray:
    if count < 0:
        raise ValueError(f"count must be >= 0; got {count}")
    raw = _payload_u8(packed)
    if count > 2 * raw.size:
        raise ValueError(f"count {count} exceeds available nibbles {2 * raw.size}")
    out = np.empty(raw.size * 2, dtype=np.uint8)
    out[0::2] = (raw >> 4) & 0xF
    out[1::2] = raw & 0xF
    return out[:count]


__all__ = [
    "DEFAULT_INT_PAYLOAD_LAYOUTS",
    "VALID_INT_PAYLOAD_LAYOUTS",
    "IntPayloadLayout",
    "decode_int_payload_layout",
    "encode_int_payload_layout",
]
