#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_tk1: scorer-free semantic-stream coder race for tq1c argmax maps.

This tool prices 5-class SegNet argmax maps that already exist on disk.  It
does not import scorers, decode videos, edit receivers, or call
``upstream/evaluate.py``.  The learned-prior row counts every video-derived
model byte in a self-describing frame and proves the range stream decodes back
to the exact label raster.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import lzma
import math
import os
import platform
import struct
import sys
import time
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
for entry in (str(REPO), str(REPO / "src"), str(REPO / "experiments"), str(REPO / "tools")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments import ddm_pp1_direct_partition_coder as pp1  # noqa: E402

try:
    import brotli
except ImportError:  # pragma: no cover - optional environment dependency
    brotli = None  # type: ignore[assignment]


SCHEMA: Final = "ddm_tk1_semantic_stream_race.v1"
AXIS: Final = "[macOS-CPU byte-only scorer-free]"
RATE_DENOMINATOR: Final = 37_545_489
LEVELS: Final = 5
SENTINEL: Final = 5
CONTEXT_RADIX: Final = 6
DEFAULT_PATCH: Final = 32
DEFAULT_GROUP_DELTA: Final = 2
DEFAULT_HASH_CONTEXT_ROWS: Final = 2000
DEFAULT_MAX_MODEL_BYTES: Final = 10_000

DEFAULT_TQ1C_LABELS: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/parent_tq1c_argmax_n600.npy"
)
DEFAULT_TQ1C_DIGESTS: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/"
    "stage_checkpoints/n600_scorer/move_0023_snap_r00_c12_L13"
)
DEFAULT_GT_LABELS: Final = Path("/Volumes/VertigoDataTier/pact/ddm_ph1_lstars_u8.npy")
DEFAULT_RECEIPT_DIR: Final = REPO / ".omx/research/ddm_tk1_20260806"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_tk1_20260806")

EXPECTED_TQ1C_RAW_SHA256: Final = "a7dd6f4271eedfa877f6499348de5f9dae2d97311f9e98f4f534908eb66e044e"
EXPECTED_TQ1C_NPY_SHA256: Final = "764a244c4890b22a67c4dbe95a959e970c29328778d41ffe4deb85f5b650eee6"
EXPECTED_GT_RAW_SHA256: Final = "f2c8be94774780bda718adf337900403a8533b6ffa1352b5aae19e200a005557"
EXPECTED_GT_NPY_SHA256: Final = "b74a14b226a5aceb5824899898bcb06e5413c54b7db2441268da7bc91a10db5d"

TK1_MAGIC: Final = b"TK1LAB1!"
TK1_VERSION: Final = 1
TK1_HEADER: Final = struct.Struct("<8sBBBBBHHHHII32s")
CONTEXT_MODE_IDS: Final = {
    "prev": 1,
    "prev_left_up": 2,
    "prev_left_up_ul": 3,
    "hash_prev_spatial": 4,
}
ID_CONTEXT_MODES: Final = {value: key for key, value in CONTEXT_MODE_IDS.items()}
DEFAULT_CONTEXT_MODE_RACE: Final = (
    "prev",
    "prev_left_up",
    "prev_left_up_ul",
    "hash_prev_spatial",
)

STATE_BITS: Final = 32
FULL_RANGE: Final = 1 << STATE_BITS
HALF: Final = FULL_RANGE >> 1
QUARTER: Final = HALF >> 1
THREE_QUARTERS: Final = 3 * QUARTER
RAW_LZMA_FILTERS: Final = [{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}]


class TK1Error(ValueError):
    """The TK1 semantic stream or measurement precondition failed closed."""


class _BitWriter:
    def __init__(self) -> None:
        self.output = bytearray()
        self.current = 0
        self.used = 0

    def write(self, bit: int) -> None:
        self.current = (self.current << 1) | (bit & 1)
        self.used += 1
        if self.used == 8:
            self.output.append(self.current)
            self.current = 0
            self.used = 0

    def finish(self) -> bytes:
        if self.used:
            self.output.append(self.current << (8 - self.used))
            self.current = 0
            self.used = 0
        return bytes(self.output)


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.byte_index = 0
        self.bit_index = 0

    def read(self) -> int:
        if self.byte_index >= len(self.payload):
            return 0
        value = (self.payload[self.byte_index] >> (7 - self.bit_index)) & 1
        self.bit_index += 1
        if self.bit_index == 8:
            self.bit_index = 0
            self.byte_index += 1
        return value


class _RangeEncoder:
    def __init__(self) -> None:
        self.writer = _BitWriter()
        self.low = 0
        self.high = FULL_RANGE - 1
        self.pending = 0

    def _emit(self, bit: int) -> None:
        self.writer.write(bit)
        while self.pending:
            self.writer.write(1 - bit)
            self.pending -= 1

    def encode(self, symbol: int, cumulative: Sequence[int], total: int) -> None:
        width = self.high - self.low + 1
        self.high = self.low + (width * int(cumulative[symbol + 1]) // total) - 1
        self.low = self.low + (width * int(cumulative[symbol]) // total)
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTERS:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> bytes:
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        return self.writer.finish()


class _RangeDecoder:
    def __init__(self, payload: bytes) -> None:
        if not payload:
            raise TK1Error("range stream is empty")
        self.reader = _BitReader(payload)
        self.low = 0
        self.high = FULL_RANGE - 1
        self.code = 0
        for _ in range(STATE_BITS):
            self.code = (self.code << 1) | self.reader.read()

    def decode(self, cumulative: Sequence[int], total: int) -> int:
        width = self.high - self.low + 1
        target = ((self.code - self.low + 1) * total - 1) // width
        symbol = bisect_right(cumulative, target) - 1
        if symbol < 0 or symbol + 1 >= len(cumulative):
            raise TK1Error("range target escaped the frequency row")
        self.high = self.low + (width * int(cumulative[symbol + 1]) // total) - 1
        self.low = self.low + (width * int(cumulative[symbol]) // total)
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTERS:
                self.low -= QUARTER
                self.high -= QUARTER
                self.code -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = (self.code << 1) | self.reader.read()
        return symbol


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def score_rate(bytes_count: int) -> float:
    return 25.0 * int(bytes_count) / RATE_DENOMINATOR


def _validate_labels(
    labels: np.ndarray,
    *,
    n: int | None = None,
    expected_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    array = np.ascontiguousarray(labels[:n] if n is not None else labels, dtype=np.uint8)
    if array.ndim != 3 or not array.size:
        raise TK1Error("label array must be non-empty uint8 [P,H,W]")
    if expected_hw is not None and array.shape[1:] != expected_hw:
        raise TK1Error(f"expected {expected_hw[0]}x{expected_hw[1]} labels, got {array.shape}")
    if int(array.max()) >= LEVELS:
        raise TK1Error("label array escapes the 5-symbol alphabet")
    return array


def load_label_array(
    path: Path,
    *,
    n: int,
    npz_key: str = "lstars",
    expected_raw_sha256: str | None = None,
    expected_file_sha256: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    file_bytes, file_sha = sha256_file(path)
    if expected_file_sha256 and file_sha != expected_file_sha256:
        raise TK1Error(f"{path} sha256 {file_sha} != expected {expected_file_sha256}")
    if path.suffix == ".npz":
        loaded = np.load(path, mmap_mode="r")[npz_key]
    else:
        loaded = np.load(path, mmap_mode="r")
    labels = _validate_labels(loaded, n=n, expected_hw=(384, 512))
    raw = labels.tobytes()
    raw_sha = sha256_bytes(raw)
    if expected_raw_sha256 and raw_sha != expected_raw_sha256:
        raise TK1Error(f"{path} raw sha256 {raw_sha} != expected {expected_raw_sha256}")
    return labels, {
        "path": str(path),
        "file_bytes": file_bytes,
        "file_sha256": file_sha,
        "raw_uint8_bytes": int(labels.nbytes),
        "raw_sha256": raw_sha,
        "shape": [int(value) for value in labels.shape],
        "dtype": str(labels.dtype),
        "alphabet": [int(value) for value in np.unique(labels)],
    }


def verify_batch_digests(labels: np.ndarray, checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    checked = []
    for path in sorted(checkpoint_dir.glob("batch_*.json")):
        row = json.loads(path.read_text())
        start, stop = [int(value) for value in row["pair_range"]]
        digest = sha256_bytes(np.ascontiguousarray(labels[start:stop], dtype=np.uint8).tobytes())
        checked.append(
            {
                "file": str(path),
                "pair_range": [start, stop],
                "expected_cells_sha256": row["cells_sha256"],
                "actual_cells_sha256": digest,
                "pass": digest == row["cells_sha256"],
            }
        )
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "batch_count": len(checked),
        "all_pass": bool(checked) and all(item["pass"] for item in checked),
        "checked_first": checked[0] if checked else None,
        "checked_last": checked[-1] if checked else None,
        "mismatches": [item for item in checked if not item["pass"]],
    }


def _lzma_raw(payload: bytes) -> bytes:
    return lzma.compress(payload, format=lzma.FORMAT_RAW, filters=RAW_LZMA_FILTERS)


def generic_baselines(labels: np.ndarray, *, stream_name: str) -> dict[str, Any]:
    raw = labels.tobytes()
    print(f"[tk1] {stream_name}: LZMA1-x9e raw baseline", file=sys.stderr, flush=True)
    lzma_blob = _lzma_raw(raw)
    if lzma.decompress(lzma_blob, format=lzma.FORMAT_RAW, filters=RAW_LZMA_FILTERS) != raw:
        raise TK1Error("raw LZMA round-trip failed")
    print(f"[tk1] {stream_name}: zlib-9 raw baseline", file=sys.stderr, flush=True)
    zlib_blob = zlib_compress(raw)
    if zlib_decompress(zlib_blob) != raw:
        raise TK1Error("raw zlib round-trip failed")
    print(f"[tk1] {stream_name}: bz2-9 raw baseline", file=sys.stderr, flush=True)
    bz2_blob = bz2.compress(raw, 9)
    if bz2.decompress(bz2_blob) != raw:
        raise TK1Error("raw bz2 round-trip failed")
    out: dict[str, Any] = {
        "raw_uint8_bytes": len(raw),
        "lzma1_x9e": {"bytes": len(lzma_blob), "roundtrip": True, "sha256": sha256_bytes(lzma_blob)},
        "zlib_9": {"bytes": len(zlib_blob), "roundtrip": True, "sha256": sha256_bytes(zlib_blob)},
        "bz2_9": {"bytes": len(bz2_blob), "roundtrip": True, "sha256": sha256_bytes(bz2_blob)},
    }
    if brotli is not None:
        print(f"[tk1] {stream_name}: Brotli-q11 raw baseline", file=sys.stderr, flush=True)
        br = brotli.compress(raw, quality=11)
        if brotli.decompress(br) != raw:
            raise TK1Error("raw Brotli round-trip failed")
        out["brotli_11"] = {"bytes": len(br), "roundtrip": True, "sha256": sha256_bytes(br)}
    else:
        out["brotli_11"] = {"bytes": None, "roundtrip": False, "reason": "brotli unavailable"}
    return out


def zlib_compress(payload: bytes) -> bytes:
    import zlib

    return zlib.compress(payload, 9)


def zlib_decompress(payload: bytes) -> bytes:
    import zlib

    return zlib.decompress(payload)


def context_arith_race(labels: np.ndarray) -> dict[str, Any]:
    templates = {
        "intra_o4": pp1._INTRA_O4,
        "intra_o6": pp1._INTRA_O6,
        "intra_o8": pp1._INTRA_O8,
        "temporal_o8_prev5": pp1._INTRA_O8 + pp1._PREV5,
    }
    rows: dict[str, Any] = {}
    for name, template in templates.items():
        kt_bytes, n_contexts = pp1.adaptive_code_bytes(labels.astype(np.int64), template, alpha=0.5)
        laplace_bytes, _ = pp1.adaptive_code_bytes(labels.astype(np.int64), template, alpha=1.0)
        rows[name] = {
            "kt_bytes": kt_bytes,
            "kt_bytes_ceiled": int(math.ceil(kt_bytes)),
            "laplace_bytes": laplace_bytes,
            "n_contexts": n_contexts,
        }
    best_name = min(rows, key=lambda key: float(rows[key]["kt_bytes"]))
    proof = pp1.roundtrip_proof(labels.astype(np.int64), templates[best_name], nf=min(6, labels.shape[0]))
    return {
        "source": "experiments/ddm_pp1_direct_partition_coder.py adaptive_code_bytes + roundtrip_proof",
        "rows": rows,
        "best_context": best_name,
        "best_kt_bytes": rows[best_name]["kt_bytes"],
        "best_kt_bytes_ceiled": rows[best_name]["kt_bytes_ceiled"],
        "roundtrip_proof": proof,
    }


def context_rows_for_mode(mode: str, hash_context_rows: int = DEFAULT_HASH_CONTEXT_ROWS) -> int:
    if mode == "prev":
        return CONTEXT_RADIX
    if mode == "prev_left_up":
        return CONTEXT_RADIX ** 3
    if mode == "prev_left_up_ul":
        return CONTEXT_RADIX ** 4
    if mode == "hash_prev_spatial":
        return hash_context_rows
    raise TK1Error(f"unknown context mode {mode!r}")


def _shift(a: np.ndarray, dy: int, dx: int, fill: int = SENTINEL) -> np.ndarray:
    frames, height, width = a.shape
    out = np.full((frames, height, width), fill, dtype=np.uint8)
    ys0, ye0 = max(0, dy), height + min(0, dy)
    xs0, xe0 = max(0, dx), width + min(0, dx)
    ys1, ye1 = max(0, -dy), height + min(0, -dy)
    xs1, xe1 = max(0, -dx), width + min(0, -dx)
    out[:, ys0:ye0, xs0:xe0] = a[:, ys1:ye1, xs1:xe1]
    return out


def _tshift(a: np.ndarray, k: int, fill: int = SENTINEL) -> np.ndarray:
    out = np.empty_like(a)
    out[:k] = fill
    out[k:] = a[:-k]
    return out


def build_context_ids(
    labels: np.ndarray,
    *,
    mode: str,
    context_rows: int,
    patch: int = DEFAULT_PATCH,
) -> np.ndarray:
    arr = _validate_labels(labels)
    prev = _tshift(arr, 1)
    if mode == "prev":
        return prev.astype(np.uint16, copy=False)
    left = _shift(arr, 0, -1)
    up = _shift(arr, -1, 0)
    if mode == "prev_left_up":
        return ((prev.astype(np.uint16) * CONTEXT_RADIX + left) * CONTEXT_RADIX + up).astype(
            np.uint16
        )
    ul = _shift(arr, -1, -1)
    if mode == "prev_left_up_ul":
        return (
            ((prev.astype(np.uint16) * CONTEXT_RADIX + left) * CONTEXT_RADIX + up)
            * CONTEXT_RADIX
            + ul
        ).astype(np.uint16)
    if mode != "hash_prev_spatial":
        raise TK1Error(f"unknown context mode {mode!r}")
    frames, height, width = arr.shape
    yy = (np.arange(height, dtype=np.uint16) // patch)[None, :, None]
    xx = (np.arange(width, dtype=np.uint16) // patch)[None, None, :]
    ctx = (
        prev.astype(np.uint32) * 131
        + left.astype(np.uint32) * 67
        + up.astype(np.uint32) * 31
        + ul.astype(np.uint32) * 17
        + yy.astype(np.uint32) * 5
        + xx.astype(np.uint32)
    ) % int(context_rows)
    return ctx.astype(np.uint16)


def train_label_context_table(
    labels: np.ndarray,
    *,
    mode: str,
    context_rows: int,
    patch: int = DEFAULT_PATCH,
) -> tuple[bytes, dict[str, Any]]:
    start = time.monotonic()
    ctx = build_context_ids(labels, mode=mode, context_rows=context_rows, patch=patch).reshape(-1)
    symbols = labels.reshape(-1).astype(np.int64)
    counts = np.bincount(ctx.astype(np.int64) * LEVELS + symbols, minlength=context_rows * LEVELS)
    counts = counts.reshape(context_rows, LEVELS).astype(np.uint64)
    model = np.empty((context_rows, LEVELS), dtype=np.uint8)
    active_rows = 0
    for row_index, row in enumerate(counts):
        if int(row.sum()) == 0:
            model[row_index] = 48
            continue
        active_rows += 1
        smoothed = row.astype(np.float64) + 0.5
        scaled = np.rint(smoothed / smoothed.sum() * 240.0).astype(np.int64)
        scaled = np.maximum(scaled, 1)
        scaled = np.minimum(scaled, 255)
        model[row_index] = scaled.astype(np.uint8)
    return model.tobytes(), {
        "context_mode": mode,
        "context_rows": context_rows,
        "model_raw_bytes": int(model.size),
        "active_context_rows": active_rows,
        "patch": patch,
        "train_seconds": time.monotonic() - start,
        "counts_total": int(counts.sum()),
    }


def _cumulative_rows(model_bytes: bytes, context_rows: int) -> tuple[list[list[int]], list[int]]:
    expected = context_rows * LEVELS
    if len(model_bytes) != expected:
        raise TK1Error(f"model length {len(model_bytes)} != expected {expected}")
    model = np.frombuffer(model_bytes, dtype=np.uint8).reshape(context_rows, LEVELS)
    if np.any(model == 0):
        raise TK1Error("model frequency table contains zero")
    rows: list[list[int]] = []
    totals: list[int] = []
    for row in model:
        cumulative = [0]
        running = 0
        for value in row.tolist():
            running += int(value)
            cumulative.append(running)
        rows.append(cumulative)
        totals.append(running)
    return rows, totals


def estimated_static_model_bits(
    labels: np.ndarray,
    model_bytes: bytes,
    *,
    mode: str,
    context_rows: int,
    patch: int = DEFAULT_PATCH,
) -> float:
    ctx = build_context_ids(labels, mode=mode, context_rows=context_rows, patch=patch).reshape(-1)
    symbols = labels.reshape(-1).astype(np.int64)
    model = np.frombuffer(model_bytes, dtype=np.uint8).reshape(context_rows, LEVELS).astype(np.float64)
    totals = model.sum(axis=1)
    probs = model[ctx, symbols] / totals[ctx]
    return float(-np.log2(probs).sum())


def patch_group_order(
    height: int,
    width: int,
    *,
    patch: int = DEFAULT_PATCH,
    delta: int = DEFAULT_GROUP_DELTA,
) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for block_y in range(0, height, patch):
        ph = min(patch, height - block_y)
        for block_x in range(0, width, patch):
            pw = min(patch, width - block_x)
            max_group = (pw - 1) + delta * (ph - 1)
            for group in range(max_group + 1):
                for local_y in range(ph):
                    local_x = group - delta * local_y
                    if 0 <= local_x < pw:
                        order.append((block_y + local_y, block_x + local_x))
    if len(order) != height * width:
        raise TK1Error("patch-group order does not cover the full raster exactly once")
    return order


def patch_group_causality_receipt(
    *,
    height: int = 384,
    width: int = 512,
    patch: int = DEFAULT_PATCH,
    delta: int = DEFAULT_GROUP_DELTA,
) -> dict[str, Any]:
    seen = np.zeros((height, width), dtype=bool)
    violations = []
    for y, x in patch_group_order(height, width, patch=patch, delta=delta):
        checks = []
        if x > 0:
            checks.append((y, x - 1, "left"))
        if y > 0:
            checks.append((y - 1, x, "up"))
        if y > 0 and x > 0:
            checks.append((y - 1, x - 1, "up_left"))
        for cy, cx, name in checks:
            if not seen[cy, cx]:
                violations.append({"pixel": [y, x], "dependency": [cy, cx], "name": name})
                if len(violations) >= 8:
                    return {
                        "patch": patch,
                        "delta": delta,
                        "steps_per_patch": (patch - 1) + delta * (patch - 1) + 1,
                        "causal_for_left_up_ul": False,
                        "violations": violations,
                    }
        seen[y, x] = True
    return {
        "patch": patch,
        "delta": delta,
        "steps_per_patch": (patch - 1) + delta * (patch - 1) + 1,
        "causal_for_left_up_ul": True,
        "violations": [],
        "patches_per_frame": (height // patch) * (width // patch),
    }


def _context_from_decoded(
    out: np.ndarray,
    p: int,
    y: int,
    x: int,
    mode: str,
    context_rows: int,
    patch: int,
) -> int:
    prev = int(out[p - 1, y, x]) if p else SENTINEL
    if mode == "prev":
        return prev
    left = int(out[p, y, x - 1]) if x else SENTINEL
    up = int(out[p, y - 1, x]) if y else SENTINEL
    if mode == "prev_left_up":
        return (prev * CONTEXT_RADIX + left) * CONTEXT_RADIX + up
    ul = int(out[p, y - 1, x - 1]) if y and x else SENTINEL
    if mode == "prev_left_up_ul":
        return ((prev * CONTEXT_RADIX + left) * CONTEXT_RADIX + up) * CONTEXT_RADIX + ul
    if mode != "hash_prev_spatial":
        raise TK1Error(f"unknown context mode {mode!r}")
    return (
        prev * 131
        + left * 67
        + up * 31
        + ul * 17
        + (y // patch) * 5
        + (x // patch)
    ) % context_rows


def encode_labels_with_model(
    labels: np.ndarray,
    model_bytes: bytes,
    *,
    mode: str,
    context_rows: int,
    patch: int = DEFAULT_PATCH,
    group_delta: int = DEFAULT_GROUP_DELTA,
) -> bytes:
    arr = _validate_labels(labels)
    cum_rows, totals = _cumulative_rows(model_bytes, context_rows)
    order = patch_group_order(arr.shape[1], arr.shape[2], patch=patch, delta=group_delta)
    encoder = _RangeEncoder()
    for p in range(arr.shape[0]):
        for y, x in order:
            ctx = _context_from_decoded(arr, p, y, x, mode, context_rows, patch)
            encoder.encode(int(arr[p, y, x]), cum_rows[ctx], totals[ctx])
    return encoder.finish()


def decode_labels_with_model(
    stream: bytes,
    shape: tuple[int, int, int],
    model_bytes: bytes,
    *,
    mode: str,
    context_rows: int,
    patch: int = DEFAULT_PATCH,
    group_delta: int = DEFAULT_GROUP_DELTA,
) -> np.ndarray:
    if len(shape) != 3 or any(int(value) <= 0 or int(value) > 65535 for value in shape):
        raise TK1Error("declared label shape is invalid")
    cum_rows, totals = _cumulative_rows(model_bytes, context_rows)
    order = patch_group_order(shape[1], shape[2], patch=patch, delta=group_delta)
    decoder = _RangeDecoder(stream)
    out = np.zeros(shape, dtype=np.uint8)
    for p in range(shape[0]):
        for y, x in order:
            ctx = _context_from_decoded(out, p, y, x, mode, context_rows, patch)
            out[p, y, x] = decoder.decode(cum_rows[ctx], totals[ctx])
    return np.ascontiguousarray(out)


def build_tk1_frame(
    labels: np.ndarray,
    model_bytes: bytes,
    stream: bytes,
    *,
    mode: str,
    context_rows: int,
    patch: int = DEFAULT_PATCH,
    group_delta: int = DEFAULT_GROUP_DELTA,
) -> bytes:
    arr = _validate_labels(labels)
    mode_id = CONTEXT_MODE_IDS.get(mode)
    if mode_id is None:
        raise TK1Error(f"unknown context mode {mode!r}")
    if len(model_bytes) != context_rows * LEVELS:
        raise TK1Error("model bytes do not match context_rows*levels")
    header = TK1_HEADER.pack(
        TK1_MAGIC,
        TK1_VERSION,
        LEVELS,
        mode_id,
        patch,
        group_delta,
        arr.shape[0],
        arr.shape[1],
        arr.shape[2],
        context_rows,
        len(model_bytes),
        len(stream),
        hashlib.sha256(arr.tobytes()).digest(),
    )
    return header + model_bytes + stream


def decode_tk1_frame(frame: bytes, *, verify_canonical: bool = True) -> np.ndarray:
    if len(frame) < TK1_HEADER.size:
        raise TK1Error("TK1 frame is truncated")
    (
        magic,
        version,
        levels,
        mode_id,
        patch,
        group_delta,
        n_frames,
        height,
        width,
        context_rows,
        model_len,
        stream_len,
        digest,
    ) = TK1_HEADER.unpack_from(frame)
    if magic != TK1_MAGIC or version != TK1_VERSION or levels != LEVELS:
        raise TK1Error("TK1 frame magic/version/levels differ")
    mode = ID_CONTEXT_MODES.get(mode_id)
    if mode is None:
        raise TK1Error("TK1 context mode id differs")
    offset = TK1_HEADER.size
    model = frame[offset : offset + model_len]
    offset += model_len
    stream = frame[offset : offset + stream_len]
    offset += stream_len
    if offset != len(frame) or len(model) != model_len or len(stream) != stream_len:
        raise TK1Error("TK1 frame lengths do not close")
    shape = (int(n_frames), int(height), int(width))
    decoded = decode_labels_with_model(
        stream,
        shape,
        model,
        mode=mode,
        context_rows=int(context_rows),
        patch=int(patch),
        group_delta=int(group_delta),
    )
    if hashlib.sha256(decoded.tobytes()).digest() != digest:
        raise TK1Error("TK1 decoded label SHA-256 differs")
    if verify_canonical:
        recoded = encode_labels_with_model(
            decoded,
            model,
            mode=mode,
            context_rows=int(context_rows),
            patch=int(patch),
            group_delta=int(group_delta),
        )
        if recoded != stream:
            raise TK1Error("TK1 frame is noncanonical")
    return decoded


def learned_prior_race(
    labels: np.ndarray,
    *,
    ssd_dir: Path,
    stream_name: str,
    context_modes: Sequence[str] = DEFAULT_CONTEXT_MODE_RACE,
    hash_context_rows: int = DEFAULT_HASH_CONTEXT_ROWS,
    max_model_bytes: int = DEFAULT_MAX_MODEL_BYTES,
    patch: int = DEFAULT_PATCH,
    group_delta: int = DEFAULT_GROUP_DELTA,
    full_range: bool = True,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    best_estimated: dict[str, Any] | None = None
    best_model: bytes | None = None
    for mode in context_modes:
        context_rows = context_rows_for_mode(mode, hash_context_rows)
        model_raw_bytes = context_rows * LEVELS
        if model_raw_bytes > max_model_bytes:
            rows.append(
                {
                    "context_mode": mode,
                    "context_rows": context_rows,
                    "model_raw_bytes": model_raw_bytes,
                    "skipped": True,
                    "reason": f"model exceeds max_model_bytes={max_model_bytes}",
                }
            )
            continue
        model, info = train_label_context_table(
            labels, mode=mode, context_rows=context_rows, patch=patch
        )
        bits = estimated_static_model_bits(
            labels, model, mode=mode, context_rows=context_rows, patch=patch
        )
        row = {
            "context_mode": mode,
            "context_rows": context_rows,
            "model_raw_bytes": len(model),
            "header_bytes": TK1_HEADER.size,
            "estimated_range_bytes": int(math.ceil(bits / 8.0)),
            "estimated_frame_bytes": int(TK1_HEADER.size + len(model) + math.ceil(bits / 8.0)),
            "estimated_bits_per_symbol": float(bits / labels.size),
            "model_info": info,
            "skipped": False,
        }
        rows.append(row)
        if best_estimated is None or int(row["estimated_frame_bytes"]) < int(
            best_estimated["estimated_frame_bytes"]
        ):
            best_estimated = row
            best_model = model
    if best_estimated is None or best_model is None:
        raise TK1Error("no learned-prior context mode was admissible")

    full_row = dict(best_estimated)
    if full_range:
        encode_start = time.monotonic()
        stream = encode_labels_with_model(
            labels,
            best_model,
            mode=str(best_estimated["context_mode"]),
            context_rows=int(best_estimated["context_rows"]),
            patch=patch,
            group_delta=group_delta,
        )
        encode_seconds = time.monotonic() - encode_start
        frame = build_tk1_frame(
            labels,
            best_model,
            stream,
            mode=str(best_estimated["context_mode"]),
            context_rows=int(best_estimated["context_rows"]),
            patch=patch,
            group_delta=group_delta,
        )
        decode_start = time.monotonic()
        decoded = decode_tk1_frame(frame, verify_canonical=True)
        decode_seconds = time.monotonic() - decode_start
        if not np.array_equal(decoded, labels):
            raise TK1Error("learned prior full-frame decode equality failed")
        frame_path = Path(ssd_dir) / f"{stream_name}_learned_prior.tk1"
        write_bytes(frame_path, frame)
        frame_bytes, frame_sha = sha256_file(frame_path)
        full_row.update(
            {
                "full_range_measured": True,
                "range_stream_bytes": len(stream),
                "frame_bytes": frame_bytes,
                "frame_sha256": frame_sha,
                "frame_path": str(frame_path),
                "encode_seconds": encode_seconds,
                "decode_seconds_including_canonical_reencode": decode_seconds,
                "decode_equal": True,
                "canonical_reencode_equal": True,
                "range_vs_estimated_ratio": len(stream)
                / max(1, int(best_estimated["estimated_range_bytes"])),
            }
        )
    else:
        subset = labels[: min(6, labels.shape[0])]
        stream = encode_labels_with_model(
            subset,
            best_model,
            mode=str(best_estimated["context_mode"]),
            context_rows=int(best_estimated["context_rows"]),
            patch=patch,
            group_delta=group_delta,
        )
        frame = build_tk1_frame(
            subset,
            best_model,
            stream,
            mode=str(best_estimated["context_mode"]),
            context_rows=int(best_estimated["context_rows"]),
            patch=patch,
            group_delta=group_delta,
        )
        decoded = decode_tk1_frame(frame, verify_canonical=True)
        if not np.array_equal(decoded, subset):
            raise TK1Error("learned prior subset round-trip failed")
        full_row.update(
            {
                "full_range_measured": False,
                "frame_bytes": int(best_estimated["estimated_frame_bytes"]),
                "range_stream_bytes": int(best_estimated["estimated_range_bytes"]),
                "subset_roundtrip_frames": int(subset.shape[0]),
                "subset_range_stream_bytes": len(stream),
                "subset_frame_bytes": len(frame),
                "decode_equal": True,
                "canonical_reencode_equal": True,
            }
        )
    return {
        "family": "small counted static conditional-prior table over 5-symbol labels + exact range coder",
        "patch_group_causality": patch_group_causality_receipt(
            height=labels.shape[1], width=labels.shape[2], patch=patch, delta=group_delta
        ),
        "raced_context_modes": rows,
        "best": full_row,
    }


def summarize_stream(
    name: str,
    labels: np.ndarray,
    source: Mapping[str, Any],
    *,
    digest_receipt: Mapping[str, Any] | None,
    ssd_dir: Path,
    full_range: bool,
    learned: bool,
) -> dict[str, Any]:
    start = time.monotonic()
    generic = generic_baselines(labels, stream_name=name)
    print(f"[tk1] {name}: PP1 KT context-arith race", file=sys.stderr, flush=True)
    context = context_arith_race(labels)
    print(f"[tk1] {name}: learned counted prior race", file=sys.stderr, flush=True)
    learned_result = (
        learned_prior_race(labels, ssd_dir=ssd_dir, stream_name=name, full_range=full_range)
        if learned
        else None
    )
    candidates = [
        ("kt_context_arith", int(context["best_kt_bytes_ceiled"])),
        ("lzma1_x9e", int(generic["lzma1_x9e"]["bytes"])),
        ("bz2_9", int(generic["bz2_9"]["bytes"])),
        ("zlib_9", int(generic["zlib_9"]["bytes"])),
    ]
    if generic["brotli_11"]["bytes"] is not None:
        candidates.append(("brotli_11", int(generic["brotli_11"]["bytes"])))
    if learned_result is not None:
        candidates.append(("learned_static_prior", int(learned_result["best"]["frame_bytes"])))
    ladder = [
        {
            "rank": rank,
            "name": item[0],
            "bytes": item[1],
            "rate_S": score_rate(item[1]),
        }
        for rank, item in enumerate(sorted(candidates, key=lambda kv: kv[1]), start=1)
    ]
    return {
        "name": name,
        "source": dict(source),
        "digest_receipt": digest_receipt,
        "class_fractions": {pp1.CLASS_NAMES[c]: float((labels == c).mean()) for c in range(LEVELS)},
        "boundary_px_per_frame_mean": boundary_px_per_frame(labels),
        "temporal_disagree_frac": float((labels[1:] != labels[:-1]).mean()) if labels.shape[0] > 1 else 0.0,
        "generic_baselines": generic,
        "context_arith": context,
        "learned_prior": learned_result,
        "byte_ladder": ladder,
        "best_bytes": ladder[0]["bytes"],
        "best_coder": ladder[0]["name"],
        "wall_seconds": time.monotonic() - start,
    }


def boundary_px_per_frame(labels: np.ndarray) -> float:
    db_l = np.zeros_like(labels, dtype=bool)
    db_l[:, :, 1:] = labels[:, :, 1:] != labels[:, :, :-1]
    db_u = np.zeros_like(labels, dtype=bool)
    db_u[:, 1:, :] = labels[:, 1:, :] != labels[:, :-1, :]
    return float((db_l | db_u).reshape(labels.shape[0], -1).sum(1).mean())


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def run_tk1(
    *,
    tq1c_labels_path: Path = DEFAULT_TQ1C_LABELS,
    tq1c_digest_dir: Path = DEFAULT_TQ1C_DIGESTS,
    gt_labels_path: Path = DEFAULT_GT_LABELS,
    receipt_dir: Path = DEFAULT_RECEIPT_DIR,
    ssd_dir: Path = DEFAULT_SSD_DIR,
    n: int = 600,
    full_range: bool = True,
) -> dict[str, Any]:
    run_start = time.monotonic()
    tq1c, tq1c_source = load_label_array(
        tq1c_labels_path,
        n=n,
        expected_raw_sha256=EXPECTED_TQ1C_RAW_SHA256 if n == 600 else None,
        expected_file_sha256=EXPECTED_TQ1C_NPY_SHA256 if n == 600 else None,
    )
    gt, gt_source = load_label_array(
        gt_labels_path,
        n=n,
        expected_raw_sha256=EXPECTED_GT_RAW_SHA256 if n == 600 else None,
        expected_file_sha256=EXPECTED_GT_NPY_SHA256 if n == 600 and gt_labels_path.suffix == ".npy" else None,
    )
    digest_receipt = verify_batch_digests(tq1c, tq1c_digest_dir) if n == 600 else None
    if digest_receipt is not None and not digest_receipt["all_pass"]:
        raise TK1Error("tq1c labels do not match batch cells_sha256 checkpoints")

    receipt_path = Path(receipt_dir) / "semantic_stream_race.json"
    partial_path = Path(receipt_dir) / "semantic_stream_race.partial.json"
    base_result: dict[str, Any] = {
        "schema": SCHEMA,
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "scorer_forwards_run": 0,
        "ran_upstream_evaluate_py": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "n_frames": n,
        "rate_denominator": RATE_DENOMINATOR,
        "full_range_learned_prior_requested": full_range,
        "streams": {},
        "progress_state": "sources_verified",
        "source_receipts": {
            "tq1c_parent_argmax": tq1c_source,
            "gt_lstars": gt_source,
            "tq1c_digest_receipt": digest_receipt,
        },
    }
    write_json(partial_path, base_result)

    print("[tk1] tq1c_parent_argmax: starting stream summary", file=sys.stderr, flush=True)
    tq1c_result = summarize_stream(
        "tq1c_parent_argmax",
        tq1c,
        tq1c_source,
        digest_receipt=digest_receipt,
        ssd_dir=ssd_dir,
        full_range=full_range,
        learned=True,
    )
    base_result["streams"]["tq1c_parent_argmax"] = tq1c_result
    base_result["progress_state"] = "tq1c_complete"
    base_result["run_seconds_so_far"] = time.monotonic() - run_start
    write_json(partial_path, base_result)

    print("[tk1] gt_lstars: starting stream summary", file=sys.stderr, flush=True)
    gt_result = summarize_stream(
        "gt_lstars",
        gt,
        gt_source,
        digest_receipt=None,
        ssd_dir=ssd_dir,
        full_range=full_range,
        learned=True,
    )
    result = {
        **base_result,
        "streams": {
            **base_result["streams"],
            "gt_lstars": gt_result,
        },
        "progress_state": "complete",
        "pp1_delta": {
            "gt_pp1_best_kt_bytes": int(gt_result["context_arith"]["best_kt_bytes_ceiled"]),
            "gt_learned_prior_bytes": int(gt_result["learned_prior"]["best"]["frame_bytes"]),
            "gt_learned_minus_pp1_kt_bytes": int(gt_result["learned_prior"]["best"]["frame_bytes"])
            - int(gt_result["context_arith"]["best_kt_bytes_ceiled"]),
        },
        "byte_ladder_first": tq1c_result["byte_ladder"],
        "run_seconds": time.monotonic() - run_start,
    }
    write_json(receipt_path, result)
    return result


def _self_test() -> None:
    rng = np.random.default_rng(20260806)
    base = rng.integers(0, LEVELS, size=(1, 8, 8), dtype=np.uint8)
    labels = np.repeat(base, 5, axis=0)
    noise = rng.integers(0, LEVELS, size=labels.shape, dtype=np.uint8)
    labels = np.where(rng.random(labels.shape) < 0.08, noise, labels).astype(np.uint8)
    model, _info = train_label_context_table(
        labels, mode="prev_left_up_ul", context_rows=context_rows_for_mode("prev_left_up_ul"), patch=4
    )
    stream = encode_labels_with_model(
        labels,
        model,
        mode="prev_left_up_ul",
        context_rows=context_rows_for_mode("prev_left_up_ul"),
        patch=4,
    )
    frame = build_tk1_frame(
        labels,
        model,
        stream,
        mode="prev_left_up_ul",
        context_rows=context_rows_for_mode("prev_left_up_ul"),
        patch=4,
    )
    decoded = decode_tk1_frame(frame)
    if not np.array_equal(decoded, labels):
        raise AssertionError("self-test TK1 frame roundtrip failed")
    receipt = patch_group_causality_receipt(height=8, width=8, patch=4, delta=2)
    if not receipt["causal_for_left_up_ul"]:
        raise AssertionError("patch-group order is not causal")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tq1c-labels", type=Path, default=DEFAULT_TQ1C_LABELS)
    parser.add_argument("--tq1c-digest-dir", type=Path, default=DEFAULT_TQ1C_DIGESTS)
    parser.add_argument("--gt-labels", type=Path, default=DEFAULT_GT_LABELS)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPT_DIR)
    parser.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        print(json.dumps({"schema": SCHEMA, "self_test": "ok"}, sort_keys=True))
        return 0
    result = run_tk1(
        tq1c_labels_path=args.tq1c_labels,
        tq1c_digest_dir=args.tq1c_digest_dir,
        gt_labels_path=args.gt_labels,
        receipt_dir=args.receipt_dir,
        ssd_dir=args.ssd_dir,
        n=args.n,
        full_range=not args.estimate_only,
    )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "tq1c_best_coder": result["streams"]["tq1c_parent_argmax"]["best_coder"],
                "tq1c_best_bytes": result["streams"]["tq1c_parent_argmax"]["best_bytes"],
                "gt_best_coder": result["streams"]["gt_lstars"]["best_coder"],
                "gt_best_bytes": result["streams"]["gt_lstars"]["best_bytes"],
                "receipt": str(Path(args.receipt_dir) / "semantic_stream_race.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
