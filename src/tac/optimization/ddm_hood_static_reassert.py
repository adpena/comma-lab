# SPDX-License-Identifier: MIT
"""Pure primitives for DDM static-hood reassertion.

The operation is deliberately ordered: first render the MENU1 paint winner,
then restore V19C frame-1 bytes on a self-detected ego-hood support.  Frame 0
is immutable.  Stored supports use an exact parse-backable bit-mask payload;
decoder-derived supports carry no new video-derived bytes.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np

from tac.boundary_math.hood_static_component import (
    compute_static_hood_mask,
    identify_static_hood_class,
)

_MAGIC = b"HMK1"
_HEADER = struct.Struct("<4sBIII")


class HoodStaticReassertError(ValueError):
    """Raised when the support or ordered composition violates custody."""


@dataclass(frozen=True)
class HoodSupportSet:
    """Self-detected per-frame and single-static scorer-cell supports."""

    hood_class: int
    per_frame: np.ndarray
    static: np.ndarray
    evidence: tuple[dict[str, object], ...]
    static_mean_frame_iou: float
    static_min_frame_iou: float


def derive_hood_supports(base_cells: np.ndarray) -> HoodSupportSet:
    """Derive both support interpretations without a hard-coded class index."""

    cells = np.asarray(base_cells)
    if cells.dtype != np.uint8 or cells.ndim != 3 or cells.shape[0] != 600:
        raise HoodStaticReassertError("base_cells must be uint8 [600,H,W]")
    hood_class, raw_evidence = identify_static_hood_class(cells)
    static = compute_static_hood_mask(cells, hood_cls=hood_class, agg="majority")
    per_frame = np.ascontiguousarray(cells == hood_class)
    evidence = tuple(
        {
            "class_id": int(row.cls),
            "static_iou": float(row.static_iou),
            "frac_of_frame": float(row.frac_of_frame),
            "bottom_share": float(row.bottom_share),
            "majority_row_span": [int(v) for v in row.maj_row_span],
        }
        for row in raw_evidence
    )
    return HoodSupportSet(
        hood_class=int(hood_class),
        per_frame=per_frame,
        static=np.ascontiguousarray(static.mask),
        evidence=evidence,
        static_mean_frame_iou=float(static.mean_frame_iou),
        static_min_frame_iou=float(static.min_frame_iou),
    )


def encode_stored_support(mask: np.ndarray) -> bytes:
    """Encode a 2-D or 3-D bool support with deterministic zlib parse-back."""

    value = np.asarray(mask, dtype=bool)
    if value.ndim not in (2, 3) or any(int(v) <= 0 for v in value.shape):
        raise HoodStaticReassertError("stored support must be nonempty 2-D or 3-D")
    n, h, w = ((1, *value.shape) if value.ndim == 2 else value.shape)
    packed = np.packbits(value.reshape(-1), bitorder="little").tobytes()
    body = zlib.compress(packed, level=9)
    return _HEADER.pack(_MAGIC, value.ndim, int(n), int(h), int(w)) + body


def decode_stored_support(payload: bytes) -> np.ndarray:
    """Decode and validate an exact stored-support payload."""

    if len(payload) < _HEADER.size:
        raise HoodStaticReassertError("stored support payload is truncated")
    magic, ndim, n, h, w = _HEADER.unpack_from(payload)
    if magic != _MAGIC or ndim not in (2, 3) or min(n, h, w) <= 0:
        raise HoodStaticReassertError("stored support header differs")
    total = int(n) * int(h) * int(w)
    packed = zlib.decompress(payload[_HEADER.size :])
    expected = (total + 7) // 8
    if len(packed) != expected:
        raise HoodStaticReassertError("stored support packed length differs")
    bits = np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8), bitorder="little", count=total
    ).astype(bool, copy=False)
    shape = (int(h), int(w)) if ndim == 2 else (int(n), int(h), int(w))
    return np.ascontiguousarray(bits.reshape(shape))


def expand_support_to_camera(
    support: np.ndarray,
    *,
    batch_size: int,
    camera_hw: tuple[int, int],
) -> np.ndarray:
    """Nearest-cell expand one static or per-frame support to camera pixels."""

    cells = np.asarray(support, dtype=bool)
    if cells.ndim == 2:
        cells = np.broadcast_to(cells, (int(batch_size), *cells.shape))
    if cells.ndim != 3 or cells.shape[0] != int(batch_size):
        raise HoodStaticReassertError("support batch geometry differs")
    h, w = cells.shape[1:]
    ch, cw = (int(v) for v in camera_hw)
    ys = (np.arange(ch, dtype=np.int64) * h // ch).clip(0, h - 1)
    xs = (np.arange(cw, dtype=np.int64) * w // cw).clip(0, w - 1)
    return np.ascontiguousarray(cells[:, ys[:, None], xs[None, :]])


def reassert_frame1(
    *,
    winner_camera: np.ndarray,
    base_camera: np.ndarray,
    camera_support: np.ndarray,
) -> np.ndarray:
    """Restore base frame-1 bytes on support and prove the ordered composition."""

    winner = np.asarray(winner_camera)
    base = np.asarray(base_camera)
    support = np.asarray(camera_support, dtype=bool)
    if (
        winner.dtype != np.uint8
        or base.dtype != np.uint8
        or winner.shape != base.shape
        or winner.ndim != 5
        or winner.shape[1] != 2
        or support.shape != winner[:, 1].shape[:3]
    ):
        raise HoodStaticReassertError("camera/support geometry differs")
    if not np.array_equal(winner[:, 0], base[:, 0]):
        raise HoodStaticReassertError("MENU1 winner changed frame 0")
    result = winner.copy()
    result[:, 1][support] = base[:, 1][support]
    if not np.array_equal(result[:, 0], base[:, 0]):
        raise HoodStaticReassertError("hood reassert changed frame 0")
    if not np.array_equal(result[:, 1][support], base[:, 1][support]):
        raise HoodStaticReassertError("hood support did not restore base bytes")
    if not np.array_equal(result[:, 1][~support], winner[:, 1][~support]):
        raise HoodStaticReassertError("hood reassert changed bytes outside support")
    return result


def class_transition_rows(
    *,
    before: np.ndarray,
    after: np.ndarray,
    target: np.ndarray,
    class_names: dict[int, str],
) -> dict[str, dict[str, int | float]]:
    """Decompose correction/introduction by target class."""

    parent = np.asarray(before)
    child = np.asarray(after)
    truth = np.asarray(target)
    if parent.shape != child.shape or child.shape != truth.shape:
        raise HoodStaticReassertError("transition geometry differs")
    rows: dict[str, dict[str, int | float]] = {}
    for class_id, name in class_names.items():
        mask = truth == int(class_id)
        sites = int(np.count_nonzero(mask))
        before_errors = int(np.count_nonzero((parent != truth) & mask))
        after_errors = int(np.count_nonzero((child != truth) & mask))
        corrected = int(np.count_nonzero((parent != truth) & (child == truth) & mask))
        introduced = int(np.count_nonzero((parent == truth) & (child != truth) & mask))
        rows[name] = {
            "sites": sites,
            "errors_before": before_errors,
            "errors_after": after_errors,
            "errors_corrected": corrected,
            "errors_introduced": introduced,
            "delta_errors_realized": corrected - introduced,
            "d_seg_after": after_errors / sites if sites else 0.0,
        }
    return rows


__all__ = [
    "HoodStaticReassertError",
    "HoodSupportSet",
    "class_transition_rows",
    "decode_stored_support",
    "derive_hood_supports",
    "encode_stored_support",
    "expand_support_to_camera",
    "reassert_frame1",
]
