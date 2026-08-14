"""Counted temporal control lattice for the PK3 frame-0 pose receiver.

The payload stores signed four-bit controls at uniformly spaced temporal knots.
The receiver expands them with deterministic integer linear interpolation and
adds the result to CP135's real 600x12 signed-int12 frame-0 coefficient lattice.
No PoseNet targets, pixels, or learned basis values are embedded in this code.
"""

from __future__ import annotations

import math
import struct

import numpy as np

MAGIC = b"P0J1"
VERSION = 1
PAIR_COUNT = 600
DIMENSIONS = 12
MIN_CONTROL = -7
MAX_CONTROL = 7
_HEADER = struct.Struct("<4sBB")
_SELECTOR_HEADER = struct.Struct("<4sBH")


class Frame0PoseOverlayError(ValueError):
    """The counted temporal overlay or its receiver lattice is invalid."""


def _round_ratio(numerator: int, denominator: int) -> int:
    """Round a signed rational to nearest, away from zero on exact ties."""
    if denominator <= 0:
        raise Frame0PoseOverlayError("interpolation denominator must be positive")
    magnitude, remainder = divmod(abs(int(numerator)), int(denominator))
    if remainder * 2 >= denominator:
        magnitude += 1
    return -magnitude if numerator < 0 else magnitude


def encode_pose_overlay(controls: np.ndarray) -> bytes:
    """Encode a canonical Kx12 signed-four-bit temporal control lattice."""
    values = np.asarray(controls, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != DIMENSIONS:
        raise Frame0PoseOverlayError("control geometry differs")
    knots = int(values.shape[0])
    if not 2 <= knots <= 64:
        raise Frame0PoseOverlayError("knot count must be in [2, 64]")
    if np.any(values < MIN_CONTROL) or np.any(values > MAX_CONTROL):
        raise Frame0PoseOverlayError("control exceeds signed four-bit domain")
    encoded = (values - MIN_CONTROL).astype(np.uint8, copy=False).ravel()
    packed = ((encoded[0::2] << 4) | encoded[1::2]).tobytes()
    return _HEADER.pack(MAGIC, VERSION, knots) + packed


def decode_pose_overlay(payload: bytes) -> np.ndarray:
    """Decode the strict overlay, rejecting aliases and trailing bytes."""
    if len(payload) < _HEADER.size:
        raise Frame0PoseOverlayError("truncated pose overlay")
    magic, version, knots = _HEADER.unpack_from(payload)
    if magic != MAGIC or version != VERSION or not 2 <= knots <= 64:
        raise Frame0PoseOverlayError("unsupported pose overlay header")
    expected = _HEADER.size + knots * DIMENSIONS // 2
    if len(payload) != expected:
        raise Frame0PoseOverlayError("pose overlay length differs")
    packed = np.frombuffer(payload[_HEADER.size :], dtype=np.uint8)
    encoded = np.empty(knots * DIMENSIONS, dtype=np.uint8)
    encoded[0::2] = packed >> 4
    encoded[1::2] = packed & 0x0F
    if np.any(encoded > MAX_CONTROL - MIN_CONTROL):
        raise Frame0PoseOverlayError("reserved four-bit control code is non-canonical")
    return (encoded.astype(np.int32) + MIN_CONTROL).reshape(knots, DIMENSIONS)


def expand_pose_controls(controls: np.ndarray) -> np.ndarray:
    """Expand controls to the n600 integer coefficient-delta lattice."""
    values = np.asarray(controls, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != DIMENSIONS:
        raise Frame0PoseOverlayError("control geometry differs")
    knots = int(values.shape[0])
    if not 2 <= knots <= 64:
        raise Frame0PoseOverlayError("knot count must be in [2, 64]")
    if np.any(values < MIN_CONTROL) or np.any(values > MAX_CONTROL):
        raise Frame0PoseOverlayError("control exceeds signed four-bit domain")
    result = np.empty((PAIR_COUNT, DIMENSIONS), dtype=np.int32)
    denominator = PAIR_COUNT - 1
    for pair in range(PAIR_COUNT):
        position = pair * (knots - 1)
        left = min(position // denominator, knots - 2)
        remainder = position - left * denominator
        for dimension in range(DIMENSIONS):
            numerator = (
                int(values[left, dimension]) * (denominator - remainder)
                + int(values[left + 1, dimension]) * remainder
            )
            result[pair, dimension] = _round_ratio(numerator, denominator)
    return result


def apply_compensation_overlay(base_codes: np.ndarray, payload: bytes) -> np.ndarray:
    """Apply the overlay to the actual signed-int12 CP135 receiver lattice."""
    codes = np.asarray(base_codes, dtype=np.int32)
    if codes.shape != (PAIR_COUNT, DIMENSIONS):
        raise Frame0PoseOverlayError("base carrier code geometry differs")
    result = codes + expand_pose_controls(decode_pose_overlay(payload))
    if np.any(result < -2048) or np.any(result > 2047):
        raise Frame0PoseOverlayError("overlay pushes carrier outside signed-int12")
    return result


def selector_payload_bytes(payload: bytes) -> int:
    """Return the exact F0E1 selector prefix length before an overlay."""
    if len(payload) < _SELECTOR_HEADER.size:
        raise Frame0PoseOverlayError("truncated frame-0 selector")
    magic, version, count = _SELECTOR_HEADER.unpack_from(payload)
    if magic != b"F0E1" or version != 1 or not 1 <= count <= PAIR_COUNT:
        raise Frame0PoseOverlayError("invalid frame-0 selector header")
    limit = math.comb(PAIR_COUNT, count)
    rank_bytes = ((limit - 1).bit_length() + 7) // 8
    label_bytes = (count * 3 + 7) // 8
    expected = _SELECTOR_HEADER.size + rank_bytes + label_bytes
    if len(payload) < expected:
        raise Frame0PoseOverlayError("truncated frame-0 selector payload")
    return expected


def split_selector_compensation(payload: bytes) -> tuple[bytes, bytes | None]:
    """Split one strict F0E1 selector from an optional strict PK3 overlay."""
    selector_bytes = selector_payload_bytes(payload)
    selector = payload[:selector_bytes]
    overlay = payload[selector_bytes:]
    if not overlay:
        return selector, None
    decode_pose_overlay(overlay)
    return selector, overlay
