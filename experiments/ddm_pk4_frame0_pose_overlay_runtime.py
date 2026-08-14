"""Strict counted temporal control lattice for PK4 frame-0 pose control.

P0J2 differs from PK3's P0J1 only in using a uint16 knot count.  This makes
the charter's approximately 1 KiB rung representable without changing the
signed-four-bit controls or deterministic integer interpolation.  The module
contains generic receiver code only; all video-derived controls remain in the
counted archive payload.
"""

from __future__ import annotations

import math
import struct

import numpy as np

MAGIC = b"P0J2"
VERSION = 2
PAIR_COUNT = 600
DIMENSIONS = 12
MIN_CONTROL = -7
MAX_CONTROL = 7
MAX_KNOTS = 599
_HEADER = struct.Struct("<4sBH")
_SELECTOR_HEADER = struct.Struct("<4sBH")


class Frame0PoseOverlayError(ValueError):
    """The counted temporal overlay or resulting int12 lattice is invalid."""


def _round_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise Frame0PoseOverlayError("interpolation denominator must be positive")
    magnitude, remainder = divmod(abs(int(numerator)), int(denominator))
    if remainder * 2 >= denominator:
        magnitude += 1
    return -magnitude if numerator < 0 else magnitude


def encoded_bytes_for_knots(knots: int) -> int:
    """Return the exact raw counted P0J2 byte length."""
    if not 2 <= int(knots) <= MAX_KNOTS:
        raise Frame0PoseOverlayError("knot count is outside the P0J2 domain")
    return _HEADER.size + int(knots) * DIMENSIONS // 2


def encode_pose_overlay(controls: np.ndarray) -> bytes:
    values = np.asarray(controls, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != DIMENSIONS:
        raise Frame0PoseOverlayError("control geometry differs")
    knots = int(values.shape[0])
    encoded_bytes_for_knots(knots)
    if np.any(values < MIN_CONTROL) or np.any(values > MAX_CONTROL):
        raise Frame0PoseOverlayError("control exceeds signed-four-bit domain")
    encoded = (values - MIN_CONTROL).astype(np.uint8, copy=False).ravel()
    packed = ((encoded[0::2] << 4) | encoded[1::2]).tobytes()
    return _HEADER.pack(MAGIC, VERSION, knots) + packed


def decode_pose_overlay(payload: bytes) -> np.ndarray:
    if len(payload) < _HEADER.size:
        raise Frame0PoseOverlayError("truncated pose overlay")
    magic, version, knots = _HEADER.unpack_from(payload)
    if magic != MAGIC or version != VERSION:
        raise Frame0PoseOverlayError("unsupported pose overlay header")
    expected = encoded_bytes_for_knots(knots)
    if len(payload) != expected:
        raise Frame0PoseOverlayError("pose overlay length differs")
    packed = np.frombuffer(payload[_HEADER.size :], dtype=np.uint8)
    encoded = np.empty(knots * DIMENSIONS, dtype=np.uint8)
    encoded[0::2] = packed >> 4
    encoded[1::2] = packed & 0x0F
    if np.any(encoded > MAX_CONTROL - MIN_CONTROL):
        raise Frame0PoseOverlayError("reserved control code is non-canonical")
    return (encoded.astype(np.int32) + MIN_CONTROL).reshape(knots, DIMENSIONS)


def expand_pose_controls(controls: np.ndarray) -> np.ndarray:
    values = np.asarray(controls, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != DIMENSIONS:
        raise Frame0PoseOverlayError("control geometry differs")
    knots = int(values.shape[0])
    encoded_bytes_for_knots(knots)
    if np.any(values < MIN_CONTROL) or np.any(values > MAX_CONTROL):
        raise Frame0PoseOverlayError("control exceeds signed-four-bit domain")
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
    codes = np.asarray(base_codes, dtype=np.int32)
    if codes.shape != (PAIR_COUNT, DIMENSIONS):
        raise Frame0PoseOverlayError("base carrier geometry differs")
    result = codes + expand_pose_controls(decode_pose_overlay(payload))
    if np.any(result < -2048) or np.any(result > 2047):
        raise Frame0PoseOverlayError("overlay pushes carrier outside signed int12")
    return result


def selector_payload_bytes(payload: bytes) -> int:
    if len(payload) < _SELECTOR_HEADER.size:
        raise Frame0PoseOverlayError("truncated frame-0 selector")
    magic, version, count = _SELECTOR_HEADER.unpack_from(payload)
    if magic != b"F0E1" or version != 1 or not 1 <= count <= PAIR_COUNT:
        raise Frame0PoseOverlayError("invalid frame-0 selector header")
    rank_bytes = ((math.comb(PAIR_COUNT, count) - 1).bit_length() + 7) // 8
    label_bytes = (count * 3 + 7) // 8
    expected = _SELECTOR_HEADER.size + rank_bytes + label_bytes
    if len(payload) < expected:
        raise Frame0PoseOverlayError("truncated frame-0 selector payload")
    return expected


def split_selector_compensation(payload: bytes) -> tuple[bytes, bytes | None]:
    selector_bytes = selector_payload_bytes(payload)
    selector = payload[:selector_bytes]
    overlay = payload[selector_bytes:]
    if not overlay:
        return selector, None
    decode_pose_overlay(overlay)
    return selector, overlay
