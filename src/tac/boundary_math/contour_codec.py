# SPDX-License-Identifier: MIT
"""Compatibility shim for the old ``contour_codec`` import path.

The implementation was renamed to
``tac.boundary_math.dense_raster_lzma_baseline`` because it stores the dense
uint8 label raster and RAW-LZMA2-compresses it. It is not a boundary-edge,
chain-code, or contour codec.

This shim remains because external Rust parity/golden-vector artifacts cite the
old Python oracle path. New Python code should import the honest module name.
"""

from __future__ import annotations

from tac.boundary_math.dense_raster_lzma_baseline import (
    _LZMA_FILTERS,
    N_SEG_CLASSES,
    ContourCode,
    DenseRasterLzmaCode,
    decode_partition,
    encode_partition,
    partition_description_bytes,
)

__all__ = [
    "N_SEG_CLASSES",
    "_LZMA_FILTERS",
    "ContourCode",
    "DenseRasterLzmaCode",
    "decode_partition",
    "encode_partition",
    "partition_description_bytes",
]
