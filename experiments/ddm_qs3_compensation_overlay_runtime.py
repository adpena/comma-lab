"""Four-bit shared compensation overlay used by the QS3 scale bracket.

Q3C1 widens QS2's measured three-bit delta alphabet from [-3, 4] to [-8, 7]
so all nine unique retained QS1 Schur rows are representable.  The format is a
strict canonical payload codec.  It is not production receiver authority until
the unchanged contest runtime is explicitly adapted and parse-back tested.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

MAGIC = b"Q3C1"
VERSION = 1
PAIR_COUNT = 600
DIMENSIONS = 12
MIN_DELTA = -8
MAX_DELTA = 7
DELTA_BITS = 4


class CompensationOverlayError(ValueError):
    """The Q3C1 payload is noncanonical or outside the sealed geometry."""


def _append_bits(bits: list[int], value: int, width: int) -> None:
    if width <= 0 or not 0 <= value < 1 << width:
        raise CompensationOverlayError("bit field value is out of range")
    bits.extend((value >> shift) & 1 for shift in range(width - 1, -1, -1))


def _take_bits(bits: np.ndarray, cursor: int, width: int) -> tuple[int, int]:
    if width <= 0 or cursor + width > bits.size:
        raise CompensationOverlayError("truncated compensation bitstream")
    value = 0
    for bit in bits[cursor : cursor + width]:
        value = (value << 1) | int(bit)
    return value, cursor + width


def encode_compensation_overlay(
    pair_indices: Sequence[int], deltas: np.ndarray
) -> bytes:
    """Encode sorted n600 pair indices and sparse four-bit int12 deltas."""
    pairs = tuple(int(value) for value in pair_indices)
    values = np.asarray(deltas, dtype=np.int32)
    if not 1 <= len(pairs) <= 15 or values.shape != (len(pairs), DIMENSIONS):
        raise CompensationOverlayError("overlay geometry differs")
    if tuple(sorted(pairs)) != pairs or len(set(pairs)) != len(pairs):
        raise CompensationOverlayError("pair indices must be sorted and unique")
    if pairs[0] < 0 or pairs[-1] >= PAIR_COUNT:
        raise CompensationOverlayError("pair index exceeds the n600 domain")
    if np.any(values < MIN_DELTA) or np.any(values > MAX_DELTA):
        raise CompensationOverlayError("delta exceeds the four-bit domain")
    masks = np.zeros(len(pairs), dtype=np.uint16)
    for row in range(len(pairs)):
        for dimension in range(DIMENSIONS):
            if int(values[row, dimension]):
                masks[row] |= np.uint16(1 << dimension)
        if not int(masks[row]):
            raise CompensationOverlayError("zero-support pair is noncanonical")
    bits: list[int] = []
    for pair in pairs:
        _append_bits(bits, pair, 10)
    for mask in masks:
        _append_bits(bits, int(mask), DIMENSIONS)
    for row, mask in enumerate(masks):
        for dimension in range(DIMENSIONS):
            if int(mask) & (1 << dimension):
                _append_bits(bits, int(values[row, dimension]) - MIN_DELTA, DELTA_BITS)
    return MAGIC + bytes(((VERSION << 4) | len(pairs),)) + np.packbits(
        np.asarray(bits, dtype=np.uint8), bitorder="big"
    ).tobytes()


def decode_compensation_overlay(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode Q3C1 while rejecting aliases, truncation, and nonzero padding."""
    if len(payload) < len(MAGIC) + 2 or not payload.startswith(MAGIC):
        raise CompensationOverlayError("invalid compensation overlay magic or length")
    version_count = payload[len(MAGIC)]
    version, count = version_count >> 4, version_count & 0x0F
    if version != VERSION or not 1 <= count <= 15:
        raise CompensationOverlayError("unsupported compensation overlay header")
    bits = np.unpackbits(
        np.frombuffer(payload[len(MAGIC) + 1 :], dtype=np.uint8), bitorder="big"
    )
    cursor = 0
    pairs = np.empty(count, dtype=np.int16)
    for index in range(count):
        pairs[index], cursor = _take_bits(bits, cursor, 10)
    if np.any(pairs < 0) or np.any(pairs >= PAIR_COUNT) or np.any(pairs[1:] <= pairs[:-1]):
        raise CompensationOverlayError("pair index order or domain differs")
    masks = np.empty(count, dtype=np.uint16)
    for index in range(count):
        masks[index], cursor = _take_bits(bits, cursor, DIMENSIONS)
    if np.any(masks == 0):
        raise CompensationOverlayError("zero-support pair is noncanonical")
    values = np.zeros((count, DIMENSIONS), dtype=np.int32)
    for row, mask in enumerate(masks):
        for dimension in range(DIMENSIONS):
            if int(mask) & (1 << dimension):
                encoded, cursor = _take_bits(bits, cursor, DELTA_BITS)
                delta = encoded + MIN_DELTA
                if not MIN_DELTA <= delta <= MAX_DELTA or delta == 0:
                    raise CompensationOverlayError("noncanonical encoded compensation delta")
                values[row, dimension] = delta
    expected_bytes = (cursor + 7) // 8
    if len(payload) != len(MAGIC) + 1 + expected_bytes:
        raise CompensationOverlayError("compensation overlay has trailing bytes")
    if cursor < bits.size and np.any(bits[cursor:]):
        raise CompensationOverlayError("compensation overlay has nonzero padding")
    return pairs, values


def apply_compensation_overlay(base_codes: np.ndarray, payload: bytes) -> np.ndarray:
    """Apply a Q3C1 payload to the real 600x12 signed-int12 lattice."""
    codes = np.asarray(base_codes, dtype=np.int32)
    if codes.shape != (PAIR_COUNT, DIMENSIONS):
        raise CompensationOverlayError("base carrier code geometry differs")
    pairs, deltas = decode_compensation_overlay(payload)
    result = codes.copy()
    result[pairs.astype(np.int64)] += deltas
    if np.any(result < -2048) or np.any(result > 2047):
        raise CompensationOverlayError("overlay pushes carrier outside signed-int12")
    return result
