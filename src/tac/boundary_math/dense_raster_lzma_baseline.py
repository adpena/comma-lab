# SPDX-License-Identifier: MIT
"""Dense raster LZMA baseline for lossless SegNet argmax partitions.

This module encodes the whole ``(H, W)`` uint8 label raster in row-major order and
compresses it with a deterministic RAW-LZMA2 filter chain. It is a reversible
baseline for storing ``L*`` exactly; it is NOT a boundary-edge, chain-code, or
contour codec. Smooth interiors compress because LZMA sees long repeated raster
runs, but the transmitted object is still the dense label map.

Real boundary-oriented surfaces live elsewhere:

* ``tac.boundary_math.partition`` builds the region adjacency/edge structure.
* ``tac.boundary_math.boundary_solver`` contains contour-normal and MDL contour
  repair candidates.
* ``tac.boundary_math.context_partition_codec`` is the context/range-coded
  replacement for this dense-raster baseline.

NO FAKE: ``encode_partition`` -> ``decode_partition`` is bit-exact identity, and
``partition_description_bytes`` returns the actual compressed payload length.
"""

from __future__ import annotations

import lzma
from dataclasses import dataclass

import numpy as np

N_SEG_CLASSES = 5

# Deterministic LZMA filter chain (FORMAT_RAW strips container headers; lc/lp/pb
# are tuned for small-alphabet label rasters).
_LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]


@dataclass(frozen=True)
class DenseRasterLzmaCode:
    """A reversible, byte-exact dense-raster encoding of a partition."""

    payload: bytes
    shape: tuple[int, int]
    n_classes: int

    @property
    def n_bytes(self) -> int:
        return len(self.payload)


# Back-compatible public type name. The object is dense-raster LZMA; the alias is
# retained because archive grammar and Rust parity artifacts already serialize this
# shape under the older name.
ContourCode = DenseRasterLzmaCode


def encode_partition(argmax_hw: np.ndarray, n_classes: int = N_SEG_CLASSES) -> DenseRasterLzmaCode:
    """Losslessly encode the dense label map with RAW-LZMA2."""

    a = np.asarray(argmax_hw)
    if a.ndim != 2:
        raise ValueError(f"argmax must be (H, W); got {a.shape}")
    if a.min() < 0 or a.max() >= n_classes:
        raise ValueError(f"class ids must be in [0,{n_classes})")
    raw = a.astype(np.uint8).tobytes(order="C")
    payload = lzma.compress(raw, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS)
    return DenseRasterLzmaCode(payload=payload, shape=(int(a.shape[0]), int(a.shape[1])), n_classes=n_classes)


def decode_partition(code: DenseRasterLzmaCode) -> np.ndarray:
    """Decode back to the bit-exact ``(H, W)`` int64 label map."""

    raw = lzma.decompress(code.payload, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(code.shape).astype(np.int64)
    return arr


def partition_description_bytes(argmax_hw: np.ndarray, n_classes: int = N_SEG_CLASSES) -> int:
    """Actual compressed payload length in bytes under this dense-raster baseline."""

    return encode_partition(argmax_hw, n_classes).n_bytes


__all__ = [
    "N_SEG_CLASSES",
    "_LZMA_FILTERS",
    "ContourCode",
    "DenseRasterLzmaCode",
    "decode_partition",
    "encode_partition",
    "partition_description_bytes",
]
