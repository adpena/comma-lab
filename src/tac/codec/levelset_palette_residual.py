# SPDX-License-Identifier: MIT
"""Strict EKPR1 per-pair/per-class RGB residual sections for LVLS1.

EKPR1 stores one signed-int8 RGB delta for every ``(pair, class)``.  The
receiver selects the class with the level-set generator's own ``phi.argmax``
and adds the corresponding delta to frame 1 before antialiasing, optional lane
composition, and the contest resize/round operator.  No scorer state is needed
at decode time.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

EKPR1_MAGIC = b"EKPR1\x00"
EKPR1_VERSION = 1
EKPR1_DTYPE_CODE_INT8 = 1
EKPR1_CHANNELS = 3
EKPR1_CODEC = "EKPR1"
EKPR1_DTYPE = "int8"
EKPR1_APPLICATION = "frame1_phi_argmax_pre_aa_lane_r_add_clip_uint8_domain"

# magic, explicit version, n_pairs, n_classes, channels, dtype code, payload bytes
_HEADER = struct.Struct("<6sBIHBBI")
_MANIFEST_KEYS = frozenset({"codec", "version", "shape", "dtype", "application"})


class LevelsetPaletteResidualError(ValueError):
    """Raised when an EKPR1 section or its LVLS1 binding is malformed."""


@dataclass(frozen=True)
class PaletteResidualSection:
    """Parsed EKPR1 section with immutable wire metadata."""

    residuals: np.ndarray
    raw: bytes

    @property
    def n_pairs(self) -> int:
        return int(self.residuals.shape[0])

    @property
    def n_classes(self) -> int:
        return int(self.residuals.shape[1])

    @property
    def manifest(self) -> dict[str, Any]:
        return palette_residual_manifest(self.residuals)


def encode_palette_residual(residuals: np.ndarray) -> bytes:
    """Encode a contiguous ``[n_pairs, n_classes, 3]`` signed-int8 tensor."""

    array = _validate_residual_array(residuals)
    n_pairs, n_classes, channels = (int(value) for value in array.shape)
    payload = array.tobytes(order="C")
    return (
        _HEADER.pack(
            EKPR1_MAGIC,
            EKPR1_VERSION,
            n_pairs,
            n_classes,
            channels,
            EKPR1_DTYPE_CODE_INT8,
            len(payload),
        )
        + payload
    )


def decode_palette_residual(
    raw: bytes | bytearray | memoryview,
    *,
    expected_n_pairs: int | None = None,
    expected_n_classes: int | None = None,
) -> PaletteResidualSection:
    """Parse EKPR1 with strict magic, version, shape, and exact-length checks."""

    payload = bytes(raw)
    if len(payload) < _HEADER.size:
        raise LevelsetPaletteResidualError(f"EKPR1 truncated header: got {len(payload)} B, need {_HEADER.size} B")
    magic, version, n_pairs, n_classes, channels, dtype_code, payload_bytes = _HEADER.unpack_from(payload)
    if magic != EKPR1_MAGIC:
        raise LevelsetPaletteResidualError(f"bad EKPR1 magic: {magic!r}")
    if version != EKPR1_VERSION:
        raise LevelsetPaletteResidualError(f"unsupported EKPR1 version {version}; expected {EKPR1_VERSION}")
    if n_pairs <= 0 or n_classes <= 0:
        raise LevelsetPaletteResidualError("EKPR1 shape dimensions must be positive")
    if channels != EKPR1_CHANNELS:
        raise LevelsetPaletteResidualError(f"EKPR1 channels must be {EKPR1_CHANNELS}; got {channels}")
    if dtype_code != EKPR1_DTYPE_CODE_INT8:
        raise LevelsetPaletteResidualError(
            f"EKPR1 dtype code must be signed-int8 ({EKPR1_DTYPE_CODE_INT8}); got {dtype_code}"
        )
    expected_payload_bytes = int(n_pairs) * int(n_classes) * EKPR1_CHANNELS
    if payload_bytes != expected_payload_bytes:
        raise LevelsetPaletteResidualError(
            f"EKPR1 payload length field is {payload_bytes} B; shape requires {expected_payload_bytes} B"
        )
    expected_total = _HEADER.size + expected_payload_bytes
    if len(payload) < expected_total:
        raise LevelsetPaletteResidualError(
            f"EKPR1 truncated payload: got {len(payload)} B, expected {expected_total} B"
        )
    if len(payload) > expected_total:
        raise LevelsetPaletteResidualError(f"EKPR1 has {len(payload) - expected_total} trailing byte(s)")
    _validate_expected_dimension(expected_n_pairs, int(n_pairs), "n_pairs")
    _validate_expected_dimension(expected_n_classes, int(n_classes), "n_classes")
    residuals = (
        np.frombuffer(payload, dtype=np.int8, offset=_HEADER.size)
        .reshape(int(n_pairs), int(n_classes), EKPR1_CHANNELS)
        .copy()
    )
    residuals.flags.writeable = False
    return PaletteResidualSection(residuals=residuals, raw=payload)


def palette_residual_manifest(residuals: np.ndarray) -> dict[str, Any]:
    """Return the canonical LVLS1 manifest entry for one EKPR1 tensor."""

    array = _validate_residual_array(residuals)
    return {
        "codec": EKPR1_CODEC,
        "version": EKPR1_VERSION,
        "shape": [int(value) for value in array.shape],
        "dtype": EKPR1_DTYPE,
        "application": EKPR1_APPLICATION,
    }


def validate_palette_residual_binding(
    manifest_entry: Mapping[str, Any] | None,
    section: bytes | bytearray | memoryview | None,
    *,
    expected_n_pairs: int,
    expected_n_classes: int,
) -> PaletteResidualSection | None:
    """Enforce the LVLS1 manifest/section bijection and exact shape identity."""

    if manifest_entry is None and section is None:
        return None
    if manifest_entry is None:
        raise LevelsetPaletteResidualError("EKPR1 section is present without palette_residual manifest")
    if section is None:
        raise LevelsetPaletteResidualError("palette_residual manifest is present without EKPR1 section")
    if not isinstance(manifest_entry, Mapping):
        raise LevelsetPaletteResidualError("palette_residual manifest entry must be a mapping")
    keys = frozenset(manifest_entry)
    if keys != _MANIFEST_KEYS:
        missing = sorted(_MANIFEST_KEYS - keys)
        extra = sorted(keys - _MANIFEST_KEYS)
        raise LevelsetPaletteResidualError(f"palette_residual manifest keys mismatch: missing={missing} extra={extra}")
    if manifest_entry.get("codec") != EKPR1_CODEC:
        raise LevelsetPaletteResidualError("palette_residual codec must be EKPR1")
    if manifest_entry.get("version") != EKPR1_VERSION:
        raise LevelsetPaletteResidualError("palette_residual manifest version mismatch")
    if manifest_entry.get("dtype") != EKPR1_DTYPE:
        raise LevelsetPaletteResidualError("palette_residual manifest dtype must be int8")
    if manifest_entry.get("application") != EKPR1_APPLICATION:
        raise LevelsetPaletteResidualError("palette_residual application semantics mismatch")
    parsed = decode_palette_residual(
        section,
        expected_n_pairs=expected_n_pairs,
        expected_n_classes=expected_n_classes,
    )
    if manifest_entry.get("shape") != list(parsed.residuals.shape):
        raise LevelsetPaletteResidualError("palette_residual manifest/section shape mismatch")
    return parsed


def cap_palette_residual(raw: bytes, n_pairs: int) -> bytes:
    """Return a canonical EKPR1 prefix containing exactly ``n_pairs`` rows."""

    parsed = decode_palette_residual(raw)
    if isinstance(n_pairs, bool) or not isinstance(n_pairs, int) or n_pairs <= 0:
        raise LevelsetPaletteResidualError("EKPR1 cap n_pairs must be a positive integer")
    if n_pairs > parsed.n_pairs:
        raise LevelsetPaletteResidualError(f"EKPR1 cap n_pairs {n_pairs} exceeds section n_pairs {parsed.n_pairs}")
    return encode_palette_residual(parsed.residuals[:n_pairs])


def apply_palette_residual(
    rgb: np.ndarray,
    phi: np.ndarray,
    residuals: np.ndarray,
    *,
    pair_index: int,
) -> np.ndarray:
    """Apply the EKPR1 frame-1 residual selected by ``phi.argmax``.

    ``rgb`` remains in the generator's floating-point uint8 domain.  Clipping is
    performed immediately after addition, before any downstream AA/lane/R step.
    """

    rgb_array = np.asarray(rgb)
    phi_array = np.asarray(phi)
    residual_array = _validate_residual_array(residuals)
    if rgb_array.ndim != 2 or rgb_array.shape[1] != EKPR1_CHANNELS:
        raise LevelsetPaletteResidualError("EKPR1 rgb must have shape [pixels,3]")
    if phi_array.ndim != 2 or phi_array.shape[0] != rgb_array.shape[0]:
        raise LevelsetPaletteResidualError("EKPR1 phi must have shape [pixels,n_classes]")
    if phi_array.shape[1] != residual_array.shape[1]:
        raise LevelsetPaletteResidualError("EKPR1 phi class count does not match residual section")
    if not np.issubdtype(rgb_array.dtype, np.floating) or not np.issubdtype(phi_array.dtype, np.floating):
        raise LevelsetPaletteResidualError("EKPR1 rgb and phi must be floating-point arrays")
    if not np.isfinite(rgb_array).all() or not np.isfinite(phi_array).all():
        raise LevelsetPaletteResidualError("EKPR1 rgb and phi must contain only finite values")
    if isinstance(pair_index, bool) or not isinstance(pair_index, int):
        raise LevelsetPaletteResidualError("EKPR1 pair_index must be an integer")
    if pair_index < 0 or pair_index >= residual_array.shape[0]:
        raise LevelsetPaletteResidualError("EKPR1 pair_index is outside the section")
    labels = phi_array.argmax(axis=-1)
    deltas = residual_array[pair_index, labels].astype(rgb_array.dtype, copy=False)
    return np.clip(rgb_array + deltas, 0.0, 255.0).astype(rgb_array.dtype, copy=False)


def _validate_residual_array(residuals: np.ndarray) -> np.ndarray:
    array = np.asarray(residuals)
    if array.dtype != np.int8:
        raise LevelsetPaletteResidualError(f"EKPR1 residuals must have dtype int8; got {array.dtype}")
    if array.ndim != 3 or array.shape[2] != EKPR1_CHANNELS:
        raise LevelsetPaletteResidualError("EKPR1 residuals must have shape [n_pairs,n_classes,3]")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise LevelsetPaletteResidualError("EKPR1 residual dimensions must be positive")
    return np.ascontiguousarray(array)


def _validate_expected_dimension(expected: int | None, actual: int, field: str) -> None:
    if expected is None:
        return
    if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
        raise LevelsetPaletteResidualError(f"expected_{field} must be a positive integer")
    if actual != expected:
        raise LevelsetPaletteResidualError(f"EKPR1 {field} {actual} does not match LVLS1 {field} {expected}")


__all__ = [
    "EKPR1_APPLICATION",
    "EKPR1_CHANNELS",
    "EKPR1_CODEC",
    "EKPR1_DTYPE",
    "EKPR1_MAGIC",
    "EKPR1_VERSION",
    "LevelsetPaletteResidualError",
    "PaletteResidualSection",
    "apply_palette_residual",
    "cap_palette_residual",
    "decode_palette_residual",
    "encode_palette_residual",
    "palette_residual_manifest",
    "validate_palette_residual_binding",
]
