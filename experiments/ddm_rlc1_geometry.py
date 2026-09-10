"""Counted-config Q16 geometry, strict group-causal native adapter.

No default semantic class, spatial band, or geometry tuning is supplied here.
Native Q16 truncation and nearest-even distance are predeclared generic rules.
"""

from __future__ import annotations

import ctypes
import itertools
import os
import struct
from pathlib import Path

import numpy as np

H, W, K = 384, 512, 5
Y, X = np.indices((H, W))
GROUP = X % 64 + 2 * (Y % 64)
POSITIONS = tuple(np.flatnonzero(GROUP.reshape(-1) == g) for g in range(190))
FORMAT = struct.Struct("<BHH8B6B")


def parse_config(payload):
    if len(payload) != FORMAT.size:
        raise ValueError("incorrect counted geometry config length")
    v = FORMAT.unpack(payload)
    if not (v[0] < K and 0 <= v[1] < v[2] <= H):
        raise ValueError("invalid counted class/band")
    # Bounds limit generic integer arithmetic, not preferred content values.
    if not (
        1 <= v[3] <= 16
        and 1 <= v[4] <= 4
        and 1 <= v[5] <= 255
        and 1 <= v[6] <= 255
        and 1 <= v[7] <= 2
        and 1 <= v[8] <= 255
        and 1 <= v[9] <= 255
        and v[10] < 64
    ):
        raise ValueError("geometry outside proven integer domain")
    if any(a >= b for a, b in itertools.pairwise(v[11:])):
        raise ValueError("distance edges must be strictly increasing")
    return np.array(v, dtype=np.int64)


class LaneGeometry:
    """Allocate native causal state; only observe can advance the decoded prefix."""

    def __init__(self, previous, config):
        self.config = parse_config(config)
        if previous is not None:
            previous = np.asarray(previous)
            if (
                previous.shape != (H, W)
                or previous.dtype.kind not in "iu"
                or np.any(previous < 0)
                or np.any(previous >= K)
            ):
                raise ValueError("previous must be complete decoded class plane")
            previous = np.ascontiguousarray(previous, dtype=np.uint8)
        library = Path(os.environ["RLC1_GEOMETRY_LIBRARY"])
        self.lib = ctypes.CDLL(str(library))
        ints = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        octets = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        self.lib.rlc_new.argtypes = [ints, ctypes.c_void_p]
        self.lib.rlc_new.restype = ctypes.c_void_p
        self.lib.rlc_free.argtypes = [ctypes.c_void_p]
        self.lib.rlc_free.restype = None
        self.lib.rlc_contexts.argtypes = [ctypes.c_void_p, ctypes.c_int, ints, octets]
        self.lib.rlc_contexts.restype = None
        self.lib.rlc_observe.argtypes = [ctypes.c_void_p, ctypes.c_int, ints, ints]
        self.lib.rlc_observe.restype = None
        self.handle = self.lib.rlc_new(self.config, None if previous is None else previous.ctypes.data)
        if not self.handle:
            raise MemoryError("geometry allocation")
        self.plane = np.full((H, W), K, dtype=np.uint8)
        self.group = 0
        self.pending = False

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.rlc_free(self.handle)
            self.handle = None

    def _positions(self, positions):
        positions = np.asarray(positions)
        if (
            self.group >= len(POSITIONS)
            or positions.dtype.kind not in "iu"
            or not np.array_equal(positions, POSITIONS[self.group])
        ):
            raise ValueError("complete next HPAC group required")
        return np.ascontiguousarray(positions, dtype=np.int64)

    def contexts(self, positions):
        positions = self._positions(positions)
        if self.pending:
            raise ValueError("observe required")
        result = np.empty(len(positions), dtype=np.uint8)
        self.lib.rlc_contexts(self.handle, len(positions), positions, result)
        self.pending = True
        return result

    def observe(self, positions, symbols):
        positions = self._positions(positions)
        symbols = np.asarray(symbols)
        if (
            not self.pending
            or symbols.shape != positions.shape
            or symbols.dtype.kind not in "iu"
            or np.any(symbols < 0)
            or np.any(symbols >= K)
        ):
            raise ValueError("valid decoded group required")
        symbols = np.ascontiguousarray(symbols, dtype=np.int64)
        self.lib.rlc_observe(self.handle, len(positions), positions, symbols)
        self.plane.reshape(-1)[positions] = symbols
        self.group += 1
        self.pending = False
