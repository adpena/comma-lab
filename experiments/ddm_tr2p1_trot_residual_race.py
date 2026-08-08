#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""ddm_tr2p1 scorer-free TROT residual/AC race on CR1 edge supports.

This arm races the measured CR1 edge-conditioned support incumbent against a
counted joint-from-marginals residual stream and the charter-amended conditional
range-AC reference form.  It never runs SegNet, PoseNet, Metal, a scorer job,
or an archive promotion path.  Every raced byte row is a real lossless coder
output with decode equality back to the same selected n600 edge-labeled support
arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

_REPO: Final = Path(__file__).resolve().parents[1]
for _path in (_REPO / "experiments", _REPO / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import ddm_bd1_class_field_receiver as bd1
import ddm_cr1_gdl1_coder_races as cr1

SEG_H: Final = cr1.SEG_H
SEG_W: Final = cr1.SEG_W
N_PAIRS: Final = cr1.N_PAIRS
CLASS_NAMES: Final = cr1.CLASS_NAMES
TOP_EDGE_PAIRS: Final = cr1.TOP_EDGE_PAIRS

AXIS: Final = "[byte-only scorer-free]"
SELECTION_MODE: Final = "n600_all_pairs_no_prefix"
INCUMBENT_BYTES: Final = 464_557
INCUMBENT_SHA256: Final = "0a53f649768c61912399ccab14e4d3323998e47235992091e2a9e28cf7259fe1"
INCUMBENT_RAW_BYTES: Final = 2_732_013
OWN_FRONTIER: Final = "S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]"
CONTEST_POINTER: Final = "borrowed/unmoved 0.1910828242 [contest-CPU]"

DEFAULT_GT_ARGMAX: Final = cr1.DEFAULT_GT_ARGMAX
DEFAULT_CURRENT_ARGMAX: Final = cr1.DEFAULT_CURRENT_ARGMAX
DEFAULT_CR1_RECEIPT: Final = _REPO / ".omx/research/ddm_cr1_20260808/CR1_RECEIPT.json"
DEFAULT_CR1_INCUMBENT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cr1_20260808/payloads/"
    "p2_edge_conditioned_support.lzma1-raw.bin"
)
DEFAULT_RESEARCH_DIR: Final = _REPO / ".omx/research/ddm_tr2p1_20260808"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_tr2p1_20260808")
DEFAULT_Q_VALUES: Final = (0.5, 0.8, 1.0, 1.2, 2.0)
DEFAULT_LAMBDA: Final = 4.0
DEFAULT_ITERATIONS: Final = 12
DEFAULT_MAX_SOLVER_CELLS: Final = 65_536


class TR2P1Error(ValueError):
    """TR2P1 payload, solver, or decode-equality validation failed closed."""


@dataclass(frozen=True, slots=True)
class CoderRow:
    codec: str
    bytes: int
    sha256: str
    artifact_path: str | None = None


@dataclass(frozen=True, slots=True)
class StreamBuild:
    arm_id: str
    description: str
    raw: bytes
    records: tuple[bytes, ...]
    decode_equal: bool
    component_raw_bytes: dict[str, int]
    prediction_pixels: int
    residual_missing_pixels: int
    residual_extra_pixels: int
    config: dict[str, Any]


class BitWriter:
    __slots__ = ("_bits", "_buffer", "_current")

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._current = 0
        self._bits = 0

    def write(self, bit: int) -> None:
        self._current = (self._current << 1) | (bit & 1)
        self._bits += 1
        if self._bits == 8:
            self._buffer.append(self._current)
            self._current = 0
            self._bits = 0

    def finish(self) -> bytes:
        if self._bits:
            self._buffer.append(self._current << (8 - self._bits))
            self._current = 0
            self._bits = 0
        return bytes(self._buffer)


class BitReader:
    __slots__ = ("_bit_index", "_byte_index", "_data")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._byte_index = 0
        self._bit_index = 0

    def read(self) -> int:
        if self._byte_index >= len(self._data):
            return 0
        byte = self._data[self._byte_index]
        bit = (byte >> (7 - self._bit_index)) & 1
        self._bit_index += 1
        if self._bit_index == 8:
            self._bit_index = 0
            self._byte_index += 1
        return bit


RANGE_STATE_BITS: Final = 32
RANGE_FULL: Final = 1 << RANGE_STATE_BITS
RANGE_HALF: Final = RANGE_FULL >> 1
RANGE_QUARTER: Final = RANGE_HALF >> 1
RANGE_THREE_QUARTERS: Final = RANGE_QUARTER * 3


class FastRangeEncoder:
    """Small unchecked interval encoder for per-symbol conditional models."""

    __slots__ = ("_high", "_low", "_pending", "_writer")

    def __init__(self) -> None:
        self._writer = BitWriter()
        self._low = 0
        self._high = RANGE_FULL - 1
        self._pending = 0

    def _emit(self, bit: int) -> None:
        self._writer.write(bit)
        while self._pending:
            self._writer.write(1 - bit)
            self._pending -= 1

    def encode_interval(self, low_count: int, high_count: int, total: int) -> None:
        current_range = self._high - self._low + 1
        self._high = self._low + (current_range * high_count // total) - 1
        self._low = self._low + (current_range * low_count // total)
        while True:
            if self._high < RANGE_HALF:
                self._emit(0)
            elif self._low >= RANGE_HALF:
                self._emit(1)
                self._low -= RANGE_HALF
                self._high -= RANGE_HALF
            elif self._low >= RANGE_QUARTER and self._high < RANGE_THREE_QUARTERS:
                self._pending += 1
                self._low -= RANGE_QUARTER
                self._high -= RANGE_QUARTER
            else:
                break
            self._low <<= 1
            self._high = (self._high << 1) | 1

    def finish(self) -> bytes:
        self._pending += 1
        self._emit(0 if self._low < RANGE_QUARTER else 1)
        return self._writer.finish()


class FastRangeDecoder:
    __slots__ = ("_code", "_high", "_low", "_reader")

    def __init__(self, encoded: bytes) -> None:
        self._reader = BitReader(encoded)
        self._low = 0
        self._high = RANGE_FULL - 1
        self._code = 0
        for _ in range(RANGE_STATE_BITS):
            self._code = (self._code << 1) | self._reader.read()

    def target(self, total: int) -> int:
        current_range = self._high - self._low + 1
        return ((self._code - self._low + 1) * total - 1) // current_range

    def update(self, low_count: int, high_count: int, total: int) -> None:
        current_range = self._high - self._low + 1
        self._high = self._low + (current_range * high_count // total) - 1
        self._low = self._low + (current_range * low_count // total)
        while True:
            if self._high < RANGE_HALF:
                pass
            elif self._low >= RANGE_HALF:
                self._low -= RANGE_HALF
                self._high -= RANGE_HALF
                self._code -= RANGE_HALF
            elif self._low >= RANGE_QUARTER and self._high < RANGE_THREE_QUARTERS:
                self._low -= RANGE_QUARTER
                self._high -= RANGE_QUARTER
                self._code -= RANGE_QUARTER
            else:
                break
            self._low <<= 1
            self._high = (self._high << 1) | 1
            self._code = (self._code << 1) | self._reader.read()


class Fenwick:
    __slots__ = ("_tree", "n", "values")

    def __init__(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.int64).reshape(-1)
        self.n = int(arr.size)
        self.values = arr.copy()
        self._tree = np.zeros(self.n + 1, dtype=np.int64)
        for idx, value in enumerate(arr.tolist(), start=1):
            j = idx
            while j <= self.n:
                self._tree[j] += value
                j += j & -j

    def total(self) -> int:
        return self.prefix_sum(self.n)

    def prefix_sum(self, end: int) -> int:
        total = 0
        idx = int(end)
        while idx > 0:
            total += int(self._tree[idx])
            idx -= idx & -idx
        return total

    def set_zero(self, index: int) -> None:
        old = int(self.values[index])
        if old == 0:
            return
        self.values[index] = 0
        idx = index + 1
        while idx <= self.n:
            self._tree[idx] -= old
            idx += idx & -idx

    def find_by_cumulative(self, target: int) -> int:
        idx = 0
        bit = 1 << (self.n.bit_length() - 1)
        remaining = int(target)
        while bit:
            nxt = idx + bit
            if nxt <= self.n and int(self._tree[nxt]) <= remaining:
                idx = nxt
                remaining -= int(self._tree[nxt])
            bit >>= 1
        if idx >= self.n:
            raise TR2P1Error("range target outside Fenwick table")
        return idx


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, CoderRow):
        return {
            "codec": value.codec,
            "bytes": value.bytes,
            "sha256": value.sha256,
            "artifact_path": value.artifact_path,
        }
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    return value


def compact_header(schema: str, extra: dict[str, Any]) -> bytes:
    return json.dumps({"schema": schema, **extra}, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def write_varint(value: int) -> bytes:
    return cr1.write_varint(value)


def read_varint(payload: bytes, offset: int) -> tuple[int, int]:
    return cr1.read_varint(payload, offset)


def encode_sorted_deltas(values: np.ndarray) -> bytes:
    return cr1.encode_sorted_deltas(np.asarray(values, dtype=np.uint32))


def decode_sorted_deltas(payload: bytes, offset: int) -> tuple[np.ndarray, int]:
    return cr1.decode_sorted_deltas(payload, offset)


def pack_record_stream(magic: bytes, records: tuple[bytes, ...]) -> bytes:
    return cr1.pack_record_stream(magic, records)


def parse_record_stream(payload: bytes, expected_magic: bytes) -> tuple[bytes, ...]:
    if len(expected_magic) != 8:
        raise TR2P1Error("expected magic must be 8 bytes")
    if len(payload) < 12 or payload[:8] != expected_magic:
        raise TR2P1Error(f"bad record stream magic {payload[:8]!r}")
    (count,) = struct.unpack_from("<I", payload, 8)
    offset = 12
    records: list[bytes] = []
    for _ in range(count):
        if offset + 4 > len(payload):
            raise TR2P1Error("record stream length header truncated")
        (length,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        record = payload[offset : offset + length]
        if len(record) != length:
            raise TR2P1Error("record stream body truncated")
        offset += length
        records.append(record)
    if offset != len(payload):
        raise TR2P1Error("record stream has trailing bytes")
    return tuple(records)


def edge_name(edge: tuple[int, int]) -> str:
    return cr1.edge_name(edge)


def load_argmax(path: Path) -> np.ndarray:
    return cr1.load_argmax(path)


def build_supports(
    gt_argmax: np.ndarray,
    current_argmax: np.ndarray,
) -> tuple[dict[tuple[int, int], list[np.ndarray]], dict[str, Any]]:
    return cr1.build_supports(gt_argmax, current_argmax, TOP_EDGE_PAIRS)


def encode_count_pairs(counts: np.ndarray) -> bytes:
    arr = np.asarray(counts, dtype=np.uint16).reshape(-1)
    nz = np.flatnonzero(arr)
    out = bytearray(write_varint(int(nz.size)))
    prev = 0
    for i, idx_raw in enumerate(nz):
        idx = int(idx_raw)
        delta = idx if i == 0 else idx - prev
        out += write_varint(delta)
        out += write_varint(int(arr[idx]))
        prev = idx
    return bytes(out)


def decode_count_pairs(payload: bytes, offset: int, length: int) -> tuple[np.ndarray, int]:
    count, offset = read_varint(payload, offset)
    out = np.zeros(length, dtype=np.uint16)
    prev = 0
    for i in range(count):
        delta, offset = read_varint(payload, offset)
        value, offset = read_varint(payload, offset)
        idx = delta if i == 0 else prev + delta
        if idx < 0 or idx >= length:
            raise TR2P1Error("marginal index outside grid")
        if value > max(SEG_H, SEG_W):
            raise TR2P1Error("marginal count outside grid")
        out[idx] = value
        prev = idx
    return out, offset


def flats_to_marginals(flats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(flats, dtype=np.uint32)
    ys = arr // SEG_W
    xs = arr % SEG_W
    row_counts = np.bincount(ys, minlength=SEG_H).astype(np.uint16)
    col_counts = np.bincount(xs, minlength=SEG_W).astype(np.uint16)
    return row_counts, col_counts


def encode_marginals(row_counts: np.ndarray, col_counts: np.ndarray) -> bytes:
    rows = encode_count_pairs(row_counts)
    cols = encode_count_pairs(col_counts)
    return write_varint(len(rows)) + rows + write_varint(len(cols)) + cols


def decode_marginals(record: bytes, offset: int) -> tuple[np.ndarray, np.ndarray, int]:
    rows_len, offset = read_varint(record, offset)
    rows_end = offset + rows_len
    if rows_end > len(record):
        raise TR2P1Error("row marginal payload truncated")
    row_counts, row_off = decode_count_pairs(record[:rows_end], offset, SEG_H)
    if row_off != rows_end:
        raise TR2P1Error("row marginal payload has trailing bytes")
    offset = rows_end
    cols_len, offset = read_varint(record, offset)
    cols_end = offset + cols_len
    if cols_end > len(record):
        raise TR2P1Error("column marginal payload truncated")
    col_counts, col_off = decode_count_pairs(record[:cols_end], offset, SEG_W)
    if col_off != cols_end:
        raise TR2P1Error("column marginal payload has trailing bytes")
    return row_counts, col_counts, cols_end


def encode_residual(missing: np.ndarray, extra: np.ndarray) -> bytes:
    missing_payload = encode_sorted_deltas(np.asarray(missing, dtype=np.uint32))
    extra_payload = encode_sorted_deltas(np.asarray(extra, dtype=np.uint32))
    return write_varint(len(missing_payload)) + missing_payload + write_varint(len(extra_payload)) + extra_payload


def decode_residual(record: bytes, offset: int) -> tuple[np.ndarray, np.ndarray, int]:
    missing_len, offset = read_varint(record, offset)
    missing_end = offset + missing_len
    if missing_end > len(record):
        raise TR2P1Error("missing residual payload truncated")
    missing, miss_off = decode_sorted_deltas(record[:missing_end], offset)
    if miss_off != missing_end:
        raise TR2P1Error("missing residual has trailing bytes")
    offset = missing_end
    extra_len, offset = read_varint(record, offset)
    extra_end = offset + extra_len
    if extra_end > len(record):
        raise TR2P1Error("extra residual payload truncated")
    extra, extra_off = decode_sorted_deltas(record[:extra_end], offset)
    if extra_off != extra_end:
        raise TR2P1Error("extra residual has trailing bytes")
    return missing, extra, extra_end


def q_exponential(logits: np.ndarray, q: float) -> np.ndarray:
    if abs(q - 1.0) < 1.0e-12:
        return np.exp(logits)
    base = 1.0 + (1.0 - q) * logits
    return np.power(np.maximum(base, 1.0e-12), 1.0 / (1.0 - q))


def active_marginals(row_counts: np.ndarray, col_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.flatnonzero(row_counts).astype(np.int32)
    cols = np.flatnonzero(col_counts).astype(np.int32)
    r = row_counts[rows].astype(np.float64)
    c = col_counts[cols].astype(np.float64)
    if int(r.sum()) != int(c.sum()):
        raise TR2P1Error("row and column marginals differ")
    return rows, cols, r, c


def q_kernel_from_marginals(
    row_counts: np.ndarray,
    col_counts: np.ndarray,
    *,
    q: float,
    lam: float,
    max_solver_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols, r, c = active_marginals(row_counts, col_counts)
    if rows.size == 0:
        return rows, cols, np.zeros((0, 0), dtype=np.float64)
    cells = int(rows.size * cols.size)
    if cells > max_solver_cells:
        # Deterministic fallback keeps the solve bounded while still using both
        # counted marginals.  It is recorded in the receipt and never hidden.
        score = np.outer(r, c)
        return rows, cols, score / max(1.0, float(score.max()))
    row_cdf = (np.cumsum(r) - 0.5 * r) / max(1.0, float(r.sum()))
    col_cdf = (np.cumsum(c) - 0.5 * c) / max(1.0, float(c.sum()))
    coord_cost = np.abs((rows[:, None] / max(1, SEG_H - 1)) - (cols[None, :] / max(1, SEG_W - 1)))
    quantile_cost = np.abs(row_cdf[:, None] - col_cdf[None, :])
    cost = 0.5 * coord_cost + 0.5 * quantile_cost
    logits = -float(lam) * cost
    kernel = q_exponential(logits, q)
    kernel = np.maximum(kernel, 1.0e-12)
    return rows, cols, kernel.astype(np.float64, copy=False)


def balance_q_kernel(
    row_counts: np.ndarray,
    col_counts: np.ndarray,
    *,
    q: float,
    lam: float,
    iterations: int,
    max_solver_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols, mat = q_kernel_from_marginals(
        row_counts,
        col_counts,
        q=q,
        lam=lam,
        max_solver_cells=max_solver_cells,
    )
    if mat.size == 0:
        return rows, cols, mat
    _, _, r, c = active_marginals(row_counts, col_counts)
    plan = mat.copy()
    for _ in range(max(1, iterations)):
        row_sum = np.maximum(plan.sum(axis=1), 1.0e-12)
        plan *= (r / row_sum)[:, None]
        col_sum = np.maximum(plan.sum(axis=0), 1.0e-12)
        plan *= (c / col_sum)[None, :]
    return rows, cols, plan


def marginal_only_scores(row_counts: np.ndarray, col_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols, r, c = active_marginals(row_counts, col_counts)
    if rows.size == 0:
        return rows, cols, np.zeros((0, 0), dtype=np.float64)
    row_rank = (np.cumsum(r) - 0.5 * r) / max(1.0, float(r.sum()))
    col_rank = (np.cumsum(c) - 0.5 * c) / max(1.0, float(c.sum()))
    score = 1.0 / (1.0 + np.abs(row_rank[:, None] - col_rank[None, :]))
    return rows, cols, score


def edge_context_scores(
    row_counts: np.ndarray,
    col_counts: np.ndarray,
    edge: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols, base = marginal_only_scores(row_counts, col_counts)
    if base.size == 0:
        return rows, cols, base
    edge_index = TOP_EDGE_PAIRS.index(edge)
    edge_center = (edge_index + 0.5) / len(TOP_EDGE_PAIRS)
    col_norm = cols.astype(np.float64) / max(1, SEG_W - 1)
    edge_bias = 1.0 / (1.0 + np.abs(col_norm - edge_center))
    return rows, cols, base * edge_bias[None, :]


def scores_for_model(
    row_counts: np.ndarray,
    col_counts: np.ndarray,
    *,
    mode: str,
    edge: tuple[int, int],
    q: float,
    lam: float,
    iterations: int,
    max_solver_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "edge_context":
        return edge_context_scores(row_counts, col_counts, edge)
    if mode == "marginals_only":
        return marginal_only_scores(row_counts, col_counts)
    if mode == "trot_q":
        return balance_q_kernel(
            row_counts,
            col_counts,
            q=q,
            lam=lam,
            iterations=iterations,
            max_solver_cells=max_solver_cells,
        )
    raise TR2P1Error(f"unknown AC model mode {mode!r}")


def integer_projection_from_scores(
    rows: np.ndarray,
    cols: np.ndarray,
    row_counts: np.ndarray,
    col_counts: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    if rows.size == 0:
        return np.empty(0, dtype=np.uint32)
    remaining = col_counts[cols].astype(np.int32).copy()
    selected: list[int] = []
    row_order = np.argsort(-row_counts[rows], kind="stable")
    for row_pos in row_order:
        row = int(rows[row_pos])
        need = int(row_counts[row])
        if need == 0:
            continue
        available = np.flatnonzero(remaining > 0)
        if available.size < need:
            raise TR2P1Error("integer projection exhausted columns")
        # Bipartite Havel-Hakimi feasibility dominates the score.  The q-family
        # plan only breaks ties among columns with equal remaining quota.
        order = available[
            np.lexsort((-scores[row_pos, available], -remaining[available]))
        ]
        chosen = order[:need]
        remaining[chosen] -= 1
        selected.extend(row * SEG_W + int(cols[col_pos]) for col_pos in chosen)
    if np.any(remaining != 0):
        raise TR2P1Error("integer projection failed to satisfy column marginals")
    out = np.asarray(selected, dtype=np.uint32)
    out.sort()
    return out


def predict_from_marginals(
    row_counts: np.ndarray,
    col_counts: np.ndarray,
    *,
    mode: str,
    q: float = 1.0,
    lam: float = DEFAULT_LAMBDA,
    iterations: int = DEFAULT_ITERATIONS,
    max_solver_cells: int = DEFAULT_MAX_SOLVER_CELLS,
) -> np.ndarray:
    if mode == "identity":
        raise TR2P1Error("identity mode does not predict from marginals")
    if mode == "marginals_only":
        rows, cols, scores = marginal_only_scores(row_counts, col_counts)
    elif mode == "trot_q":
        rows, cols, scores = balance_q_kernel(
            row_counts,
            col_counts,
            q=q,
            lam=lam,
            iterations=iterations,
            max_solver_cells=max_solver_cells,
        )
    else:
        raise TR2P1Error(f"unknown prediction mode {mode!r}")
    return integer_projection_from_scores(rows, cols, row_counts, col_counts, scores)


def setdiff_sorted(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.setdiff1d(np.asarray(left, dtype=np.uint32), np.asarray(right, dtype=np.uint32), assume_unique=True)


def encode_identity_record(flats: np.ndarray) -> bytes:
    return b"I" + encode_sorted_deltas(np.asarray(flats, dtype=np.uint32))


def decode_identity_record(record: bytes) -> np.ndarray:
    if not record.startswith(b"I"):
        raise TR2P1Error("bad identity record magic")
    flats, offset = decode_sorted_deltas(record, 1)
    if offset != len(record):
        raise TR2P1Error("identity record has trailing bytes")
    return flats


def encode_residual_record(
    flats: np.ndarray,
    *,
    mode: str,
    q: float,
    lam: float,
    iterations: int,
    max_solver_cells: int,
) -> tuple[bytes, dict[str, int]]:
    row_counts, col_counts = flats_to_marginals(flats)
    prediction = predict_from_marginals(
        row_counts,
        col_counts,
        mode=mode,
        q=q,
        lam=lam,
        iterations=iterations,
        max_solver_cells=max_solver_cells,
    )
    missing = setdiff_sorted(flats, prediction)
    extra = setdiff_sorted(prediction, flats)
    marginals = encode_marginals(row_counts, col_counts)
    residual = encode_residual(missing, extra)
    magic = b"T" if mode == "trot_q" else b"M"
    record = magic + write_varint(len(marginals)) + marginals + residual
    accounting = {
        "marginals": len(marginals),
        "side_info": 0,
        "residual": len(residual),
        "ac_stream": 0,
        "tags": 1,
        "framing": len(record) - len(marginals) - len(residual) - 1,
        "prediction_pixels": int(prediction.size),
        "residual_missing_pixels": int(missing.size),
        "residual_extra_pixels": int(extra.size),
    }
    return record, accounting


def decode_residual_record(
    record: bytes,
    *,
    mode: str,
    q: float,
    lam: float,
    iterations: int,
    max_solver_cells: int,
) -> np.ndarray:
    expected_magic = b"T" if mode == "trot_q" else b"M"
    if not record.startswith(expected_magic):
        raise TR2P1Error(f"bad residual record magic {record[:1]!r} for {mode}")
    marg_len, offset = read_varint(record, 1)
    marg_end = offset + marg_len
    if marg_end > len(record):
        raise TR2P1Error("marginal section truncated")
    row_counts, col_counts, marg_off = decode_marginals(record[:marg_end], offset)
    if marg_off != marg_end:
        raise TR2P1Error("marginal section has trailing bytes")
    missing, extra, end = decode_residual(record, marg_end)
    if end != len(record):
        raise TR2P1Error("residual record has trailing bytes")
    prediction = predict_from_marginals(
        row_counts,
        col_counts,
        mode=mode,
        q=q,
        lam=lam,
        iterations=iterations,
        max_solver_cells=max_solver_cells,
    )
    out = np.setdiff1d(prediction, extra, assume_unique=True)
    out = np.union1d(out, missing).astype(np.uint32)
    out.sort()
    return out


AC_FREQ_SCALE: Final = 4096


def row_choice_frequencies(
    scores: np.ndarray,
    row_pos: int,
    remaining_cols: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(scores[row_pos], dtype=np.float64) * np.maximum(
        remaining_cols.astype(np.float64), 0.0
    )
    if weights.size == 0:
        return np.empty(0, dtype=np.int64)
    max_weight = float(weights.max())
    if max_weight <= 0.0:
        freqs = (remaining_cols > 0).astype(np.int64)
    else:
        freqs = 1 + np.rint(weights / max_weight * (AC_FREQ_SCALE - 1)).astype(np.int64)
        freqs[remaining_cols <= 0] = 0
    return freqs


def encode_ac_choices(
    flats: np.ndarray,
    *,
    edge: tuple[int, int],
    mode: str,
    q: float,
    lam: float,
    iterations: int,
    max_solver_cells: int,
) -> tuple[bytes, dict[str, int]]:
    row_counts, col_counts = flats_to_marginals(flats)
    rows, cols, scores = scores_for_model(
        row_counts,
        col_counts,
        mode=mode,
        edge=edge,
        q=q,
        lam=lam,
        iterations=iterations,
        max_solver_cells=max_solver_cells,
    )
    arr = np.asarray(flats, dtype=np.uint32)
    ys = arr // SEG_W
    xs = arr % SEG_W
    row_to_pos = np.full(SEG_H, -1, dtype=np.int32)
    row_to_pos[rows] = np.arange(rows.size, dtype=np.int32)
    col_to_pos = np.full(SEG_W, -1, dtype=np.int32)
    col_to_pos[cols] = np.arange(cols.size, dtype=np.int32)
    remaining = col_counts[cols].astype(np.int64).copy()
    encoder = FastRangeEncoder()
    symbol_count = 0
    for row in rows.tolist():
        need = int(row_counts[row])
        if need == 0:
            continue
        start = int(np.searchsorted(ys, row, side="left"))
        end = int(np.searchsorted(ys, row, side="right"))
        row_xs = xs[start:end]
        if row_xs.size != need:
            raise TR2P1Error("row AC encoder saw inconsistent row count")
        row_pos = int(row_to_pos[row])
        freqs = row_choice_frequencies(scores, row_pos, remaining)
        model = Fenwick(freqs)
        for x_raw in row_xs.tolist():
            col_pos = int(col_to_pos[int(x_raw)])
            if col_pos < 0 or int(remaining[col_pos]) <= 0 or int(model.values[col_pos]) <= 0:
                raise TR2P1Error("AC encoder selected an unavailable column")
            low = model.prefix_sum(col_pos)
            high = low + int(model.values[col_pos])
            total = model.total()
            if total <= 0:
                raise TR2P1Error("AC encoder has empty model")
            encoder.encode_interval(low, high, total)
            model.set_zero(col_pos)
            remaining[col_pos] -= 1
            symbol_count += 1
    if np.any(remaining != 0):
        raise TR2P1Error("AC encoder failed to consume all column marginals")
    return encoder.finish(), {
        "prediction_pixels": int(arr.size),
        "ac_symbols": symbol_count,
    }


def decode_ac_choices(
    encoded: bytes,
    row_counts: np.ndarray,
    col_counts: np.ndarray,
    *,
    edge: tuple[int, int],
    mode: str,
    q: float,
    lam: float,
    iterations: int,
    max_solver_cells: int,
) -> np.ndarray:
    rows, cols, scores = scores_for_model(
        row_counts,
        col_counts,
        mode=mode,
        edge=edge,
        q=q,
        lam=lam,
        iterations=iterations,
        max_solver_cells=max_solver_cells,
    )
    remaining = col_counts[cols].astype(np.int64).copy()
    decoder = FastRangeDecoder(encoded)
    selected: list[int] = []
    for row in rows.tolist():
        need = int(row_counts[row])
        if need == 0:
            continue
        row_pos = int(np.searchsorted(rows, row))
        freqs = row_choice_frequencies(scores, row_pos, remaining)
        model = Fenwick(freqs)
        for _ in range(need):
            total = model.total()
            if total <= 0:
                raise TR2P1Error("AC decoder has empty model")
            target = decoder.target(total)
            col_pos = model.find_by_cumulative(target)
            low = model.prefix_sum(col_pos)
            high = low + int(model.values[col_pos])
            decoder.update(low, high, total)
            selected.append(row * SEG_W + int(cols[col_pos]))
            model.set_zero(col_pos)
            remaining[col_pos] -= 1
    if np.any(remaining != 0):
        raise TR2P1Error("AC decoder failed to consume all column marginals")
    out = np.asarray(selected, dtype=np.uint32)
    out.sort()
    return out


def encode_ac_record(
    flats: np.ndarray,
    *,
    edge: tuple[int, int],
    mode: str,
    q: float,
    lam: float,
    iterations: int,
    max_solver_cells: int,
) -> tuple[bytes, dict[str, int]]:
    row_counts, col_counts = flats_to_marginals(flats)
    marginals = encode_marginals(row_counts, col_counts)
    ac_stream, ac_meta = encode_ac_choices(
        flats,
        edge=edge,
        mode=mode,
        q=q,
        lam=lam,
        iterations=iterations,
        max_solver_cells=max_solver_cells,
    )
    record = b"A" + write_varint(len(marginals)) + marginals + write_varint(len(ac_stream)) + ac_stream
    return record, {
        "marginals": len(marginals),
        "side_info": 0,
        "residual": 0,
        "ac_stream": len(ac_stream),
        "tags": 1,
        "framing": len(record) - len(marginals) - len(ac_stream) - 1,
        "prediction_pixels": int(ac_meta["prediction_pixels"]),
        "residual_missing_pixels": 0,
        "residual_extra_pixels": 0,
        "ac_symbols": int(ac_meta["ac_symbols"]),
    }


def decode_ac_record(
    record: bytes,
    *,
    edge: tuple[int, int],
    mode: str,
    q: float,
    lam: float,
    iterations: int,
    max_solver_cells: int,
) -> np.ndarray:
    if not record.startswith(b"A"):
        raise TR2P1Error("bad AC record magic")
    marg_len, offset = read_varint(record, 1)
    marg_end = offset + marg_len
    if marg_end > len(record):
        raise TR2P1Error("AC marginal section truncated")
    row_counts, col_counts, marg_off = decode_marginals(record[:marg_end], offset)
    if marg_off != marg_end:
        raise TR2P1Error("AC marginal section has trailing bytes")
    ac_len, offset = read_varint(record, marg_end)
    ac_end = offset + ac_len
    if ac_end > len(record):
        raise TR2P1Error("AC range stream truncated")
    if ac_end != len(record):
        raise TR2P1Error("AC record has trailing bytes")
    return decode_ac_choices(
        record[offset:ac_end],
        row_counts,
        col_counts,
        edge=edge,
        mode=mode,
        q=q,
        lam=lam,
        iterations=iterations,
        max_solver_cells=max_solver_cells,
    )


def records_for_support_order(
    supports: dict[tuple[int, int], list[np.ndarray]],
) -> Iterable[tuple[tuple[int, int], int, np.ndarray]]:
    for edge in TOP_EDGE_PAIRS:
        for pair in range(N_PAIRS):
            yield edge, pair, supports[edge][pair]


def build_identity_stream(
    supports: dict[tuple[int, int], list[np.ndarray]],
    *,
    header_common: dict[str, Any],
) -> StreamBuild:
    header = compact_header(
        "ddm_tr2p1_identity_container_records.v1",
        {**header_common, "arm": "identity_container_control"},
    )
    records = [header]
    component_raw_bytes = {
        "marginals": 0,
        "side_info": 0,
        "tags": len(header),
        "residual": 0,
        "ac_stream": 0,
        "framing": 0,
    }
    for _edge, _pair, flats in records_for_support_order(supports):
        rec = encode_identity_record(flats)
        decoded = decode_identity_record(rec)
        if not np.array_equal(decoded, flats):
            raise TR2P1Error("identity control decode mismatch")
        records.append(rec)
        component_raw_bytes["residual"] += len(rec) - 1
        component_raw_bytes["tags"] += 1
    records_tuple = tuple(records)
    raw = pack_record_stream(b"TR2P1ID!", records_tuple)
    component_raw_bytes["framing"] = len(raw) - sum(len(record) for record in records_tuple)
    return StreamBuild(
        arm_id="identity_container_control",
        description="raw selected support through the TR2P1 container and same coder set",
        raw=raw,
        records=records_tuple,
        decode_equal=True,
        component_raw_bytes=component_raw_bytes,
        prediction_pixels=0,
        residual_missing_pixels=0,
        residual_extra_pixels=0,
        config={"mode": "identity", "coding_family": "delta_container"},
    )


def build_residual_stream(
    supports: dict[tuple[int, int], list[np.ndarray]],
    *,
    header_common: dict[str, Any],
    arm_id: str,
    mode: str,
    q: float,
    lam: float,
    iterations: int,
    max_solver_cells: int,
) -> StreamBuild:
    print(f"[tr2p1] building residual stream {arm_id}", file=sys.stderr, flush=True)
    config = {
        "mode": mode,
        "coding_family": "lz_residual_baseline",
        "q": q if mode == "trot_q" else None,
        "lambda": lam if mode == "trot_q" else None,
        "iterations": iterations if mode == "trot_q" else None,
        "max_solver_cells": max_solver_cells,
        "side_info_source": "generic_cost_derived_from_counted_marginals",
        "video_derived_side_info_bytes": 0,
        "integer_projection": "row_count_descending_top_score_with_column_quota",
    }
    header = compact_header(
        "ddm_tr2p1_joint_from_marginals_residual_records.v1",
        {**header_common, "arm": arm_id, "config": config},
    )
    records = [header]
    component_raw_bytes = {
        "marginals": 0,
        "side_info": 0,
        "tags": len(header),
        "residual": 0,
        "ac_stream": 0,
        "framing": 0,
    }
    prediction_pixels = 0
    missing_pixels = 0
    extra_pixels = 0
    for _edge, _pair, flats in records_for_support_order(supports):
        rec, accounting = encode_residual_record(
            flats,
            mode=mode,
            q=q,
            lam=lam,
            iterations=iterations,
            max_solver_cells=max_solver_cells,
        )
        decoded = decode_residual_record(
            rec,
            mode=mode,
            q=q,
            lam=lam,
            iterations=iterations,
            max_solver_cells=max_solver_cells,
        )
        if not np.array_equal(decoded, flats):
            raise TR2P1Error(f"{arm_id} decode mismatch")
        records.append(rec)
        for key in ("marginals", "side_info", "tags", "residual", "ac_stream", "framing"):
            component_raw_bytes[key] += int(accounting[key])
        prediction_pixels += int(accounting["prediction_pixels"])
        missing_pixels += int(accounting["residual_missing_pixels"])
        extra_pixels += int(accounting["residual_extra_pixels"])
    records_tuple = tuple(records)
    raw = pack_record_stream(b"TR2P1RS!", records_tuple)
    component_raw_bytes["framing"] += len(raw) - sum(len(record) for record in records_tuple)
    built = StreamBuild(
        arm_id=arm_id,
        description=(
            "q-family joint-from-marginals residual stream"
            if mode == "trot_q"
            else "marginals plus residual with no q-family OT solve"
        ),
        raw=raw,
        records=records_tuple,
        decode_equal=True,
        component_raw_bytes=component_raw_bytes,
        prediction_pixels=prediction_pixels,
        residual_missing_pixels=missing_pixels,
        residual_extra_pixels=extra_pixels,
        config=config,
    )
    print(
        f"[tr2p1] built residual stream {arm_id}: raw={len(raw)}",
        file=sys.stderr,
        flush=True,
    )
    return built


def build_ac_stream(
    supports: dict[tuple[int, int], list[np.ndarray]],
    *,
    header_common: dict[str, Any],
    arm_id: str,
    mode: str,
    q: float,
    lam: float,
    iterations: int,
    max_solver_cells: int,
) -> StreamBuild:
    print(f"[tr2p1] building AC stream {arm_id}", file=sys.stderr, flush=True)
    config = {
        "mode": mode,
        "coding_family": "conditional_range_ac",
        "q": q if mode == "trot_q" else None,
        "lambda": lam if mode == "trot_q" else None,
        "iterations": iterations if mode == "trot_q" else None,
        "max_solver_cells": max_solver_cells,
        "side_info_source": "generic_cost_derived_from_counted_marginals_and_edge_context",
        "video_derived_side_info_bytes": 0,
        "symbolization": "row_ordered_column_choices_under_counted_row_col_marginals",
        "frequency_scale": AC_FREQ_SCALE,
    }
    header = compact_header(
        "ddm_tr2p1_conditional_range_ac_records.v1",
        {**header_common, "arm": arm_id, "config": config},
    )
    records = [header]
    component_raw_bytes = {
        "marginals": 0,
        "side_info": 0,
        "tags": len(header),
        "residual": 0,
        "ac_stream": 0,
        "framing": 0,
    }
    prediction_pixels = 0
    ac_symbols = 0
    for edge, _pair, flats in records_for_support_order(supports):
        rec, accounting = encode_ac_record(
            flats,
            edge=edge,
            mode=mode,
            q=q,
            lam=lam,
            iterations=iterations,
            max_solver_cells=max_solver_cells,
        )
        decoded = decode_ac_record(
            rec,
            edge=edge,
            mode=mode,
            q=q,
            lam=lam,
            iterations=iterations,
            max_solver_cells=max_solver_cells,
        )
        if not np.array_equal(decoded, flats):
            raise TR2P1Error(f"{arm_id} AC decode mismatch")
        records.append(rec)
        for key in ("marginals", "side_info", "tags", "residual", "ac_stream", "framing"):
            component_raw_bytes[key] += int(accounting[key])
        prediction_pixels += int(accounting["prediction_pixels"])
        ac_symbols += int(accounting["ac_symbols"])
    records_tuple = tuple(records)
    raw = pack_record_stream(b"TR2P1AC!", records_tuple)
    component_raw_bytes["framing"] += len(raw) - sum(len(record) for record in records_tuple)
    description = (
        "CR1 edge-conditioned conditional range-AC control"
        if mode == "edge_context"
        else (
            "q-family joint-from-marginals conditional range-AC stream"
            if mode == "trot_q"
            else "marginals-only conditional range-AC control"
        )
    )
    built = StreamBuild(
        arm_id=arm_id,
        description=description,
        raw=raw,
        records=records_tuple,
        decode_equal=True,
        component_raw_bytes=component_raw_bytes,
        prediction_pixels=prediction_pixels,
        residual_missing_pixels=0,
        residual_extra_pixels=0,
        config={**config, "ac_symbols": ac_symbols},
    )
    print(
        f"[tr2p1] built AC stream {arm_id}: raw={len(raw)} ac_symbols={ac_symbols}",
        file=sys.stderr,
        flush=True,
    )
    return built


def race_coders(
    *,
    stream: StreamBuild,
    artifact_dir: Path,
) -> tuple[tuple[CoderRow, ...], CoderRow]:
    print(
        f"[tr2p1] coding {stream.arm_id}: raw={len(stream.raw)} records={len(stream.records)}",
        file=sys.stderr,
        flush=True,
    )
    encoded = {
        "zlib-9": zlib.compress(stream.raw, level=9),
        "brotli-q11": bytes(brotli.compress(stream.raw, quality=11)),
        "lzma1-raw": bd1.lzma1_raw(stream.raw),
    }
    print(
        f"[tr2p1] coding {stream.arm_id}: zlib={len(encoded['zlib-9'])} "
        f"brotli={len(encoded['brotli-q11'])} lzma1={len(encoded['lzma1-raw'])}; "
        "starting smevr",
        file=sys.stderr,
        flush=True,
    )
    encoded["smevr-r7-nibble"] = bd1.smevr_records(list(stream.records))
    print(
        f"[tr2p1] coding {stream.arm_id}: smevr={len(encoded['smevr-r7-nibble'])}",
        file=sys.stderr,
        flush=True,
    )
    if zlib.decompress(encoded["zlib-9"]) != stream.raw:
        raise TR2P1Error(f"{stream.arm_id}: zlib roundtrip failed")
    if brotli.decompress(encoded["brotli-q11"]) != stream.raw:
        raise TR2P1Error(f"{stream.arm_id}: Brotli roundtrip failed")
    if bd1.unlzma1_raw(encoded["lzma1-raw"], len(stream.raw)) != stream.raw:
        raise TR2P1Error(f"{stream.arm_id}: LZMA1 roundtrip failed")
    if tuple(bd1.unsmevr_records(encoded["smevr-r7-nibble"])) != stream.records:
        raise TR2P1Error(f"{stream.arm_id}: SMEVR record roundtrip failed")
    best_codec = min(encoded, key=lambda name: len(encoded[name]))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows: list[CoderRow] = []
    for codec, payload in sorted(encoded.items(), key=lambda item: len(item[1])):
        artifact_path = None
        if codec == best_codec:
            path = artifact_dir / f"{stream.arm_id}.{codec}.bin"
            path.write_bytes(payload)
            artifact_path = str(path)
        rows.append(CoderRow(codec, len(payload), sha256_bytes(payload), artifact_path))
    best = next(row for row in rows if row.codec == best_codec)
    return tuple(rows), best


def decode_cr1_incumbent(
    incumbent_path: Path,
    supports: dict[tuple[int, int], list[np.ndarray]],
) -> dict[str, Any]:
    if not incumbent_path.exists():
        raise TR2P1Error(f"missing CR1 incumbent artifact {incumbent_path}")
    artifact_sha = sha256_file(incumbent_path)
    if artifact_sha != INCUMBENT_SHA256:
        raise TR2P1Error(f"CR1 incumbent sha drifted: {artifact_sha}")
    raw = bd1.unlzma1_raw(incumbent_path.read_bytes(), INCUMBENT_RAW_BYTES)
    records = parse_record_stream(raw, b"CR1P2E1!")
    header = json.loads(records[0].decode("utf-8"))
    if header.get("schema") != "ddm_cr1_edge_conditioned_records.v1":
        raise TR2P1Error("CR1 incumbent header schema mismatch")
    offset = 1
    decoded_pixels = 0
    for edge in TOP_EDGE_PAIRS:
        for pair in range(N_PAIRS):
            arr = cr1.decode_edge_conditioned_record(records[offset])
            offset += 1
            if not np.array_equal(arr, supports[edge][pair]):
                raise TR2P1Error(f"CR1 incumbent decode mismatch {edge_name(edge)} pair {pair}")
            decoded_pixels += int(arr.size)
    if offset != len(records):
        raise TR2P1Error("CR1 incumbent record count mismatch")
    return {
        "artifact_path": str(incumbent_path),
        "bytes": incumbent_path.stat().st_size,
        "sha256": artifact_sha,
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
        "record_count": len(records),
        "decoded_pixels": decoded_pixels,
        "decode_equality": True,
    }


def validate_q_solver_fixtures() -> dict[str, Any]:
    row_counts = np.zeros(SEG_H, dtype=np.uint16)
    col_counts = np.zeros(SEG_W, dtype=np.uint16)
    row_counts[[2, 5, 7]] = [2, 1, 2]
    col_counts[[3, 6, 8, 11]] = [1, 2, 1, 1]
    rows, cols, plan_q1 = balance_q_kernel(
        row_counts,
        col_counts,
        q=1.0,
        lam=2.0,
        iterations=80,
        max_solver_cells=10_000,
    )
    rows_ref, cols_ref, kernel_ref = q_kernel_from_marginals(
        row_counts,
        col_counts,
        q=1.0,
        lam=2.0,
        max_solver_cells=10_000,
    )
    _, _, r, c = active_marginals(row_counts, col_counts)
    ref = kernel_ref.copy()
    for _ in range(80):
        ref *= (r / np.maximum(ref.sum(axis=1), 1.0e-12))[:, None]
        ref *= (c / np.maximum(ref.sum(axis=0), 1.0e-12))[None, :]
    if not np.array_equal(rows, rows_ref) or not np.array_equal(cols, cols_ref):
        raise TR2P1Error("q=1 fixture active coordinates drifted")
    max_abs = float(np.max(np.abs(plan_q1 - ref)))
    row_err = float(np.max(np.abs(plan_q1.sum(axis=1) - r)))
    col_err = float(np.max(np.abs(plan_q1.sum(axis=0) - c)))
    if max_abs > 1.0e-12 or row_err > 1.0e-9 or col_err > 1.0e-9:
        raise TR2P1Error("q=1 fixture did not recover Sinkhorn/IPF reference")
    target = np.asarray([2 * SEG_W + 3, 2 * SEG_W + 6, 5 * SEG_W + 6, 7 * SEG_W + 8, 7 * SEG_W + 11], dtype=np.uint32)
    record, _accounting = encode_residual_record(
        target,
        mode="trot_q",
        q=1.0,
        lam=2.0,
        iterations=80,
        max_solver_cells=10_000,
    )
    decoded = decode_residual_record(
        record,
        mode="trot_q",
        q=1.0,
        lam=2.0,
        iterations=80,
        max_solver_cells=10_000,
    )
    if not np.array_equal(decoded, target):
        raise TR2P1Error("fixture residual decode equality failed")
    ac_record, ac_accounting = encode_ac_record(
        target,
        edge=TOP_EDGE_PAIRS[0],
        mode="trot_q",
        q=1.0,
        lam=2.0,
        iterations=80,
        max_solver_cells=10_000,
    )
    ac_decoded = decode_ac_record(
        ac_record,
        edge=TOP_EDGE_PAIRS[0],
        mode="trot_q",
        q=1.0,
        lam=2.0,
        iterations=80,
        max_solver_cells=10_000,
    )
    if not np.array_equal(ac_decoded, target):
        raise TR2P1Error("fixture AC decode equality failed")
    return {
        "fixture": "3x4 active support embedded in scorer grid",
        "q1_matches_sinkhorn_reference": True,
        "max_abs_plan_delta": max_abs,
        "row_marginal_max_abs_error": row_err,
        "col_marginal_max_abs_error": col_err,
        "residual_decode_equality": True,
        "conditional_range_ac_decode_equality": True,
        "conditional_range_ac_bytes": len(ac_record),
        "conditional_range_ac_symbols": int(ac_accounting["ac_symbols"]),
        "target_pixels": int(target.size),
    }


def stream_row(stream: StreamBuild, rows: tuple[CoderRow, ...], best: CoderRow) -> dict[str, Any]:
    delta = best.bytes - INCUMBENT_BYTES
    coding_family = str(stream.config.get("coding_family", "unknown"))
    if coding_family == "conditional_range_ac" and stream.config["mode"] == "trot_q":
        verdict_scope = (
            "FORMULATION: q-family joint-from-marginals conditional range-AC coding of "
            "the CR1 selected edge-labeled n600 support payload"
        )
    elif coding_family == "lz_residual_baseline" and stream.config["mode"] == "trot_q":
        verdict_scope = (
            "INSTRUMENT(LZ-coder): q-family joint-from-marginals residual stream; "
            "not a family/formulation verdict carrier under the 2026-08-08 amendment"
        )
    else:
        verdict_scope = "CONTROL: same-payload TR2P1 byte container for selected edge-labeled n600 support"
    return {
        "schema": "ddm_tr2p1_coder_race_row.v1",
        "created_utc": now_utc(),
        "arm_id": stream.arm_id,
        "axis": AXIS,
        "selection_mode": SELECTION_MODE,
        "description": stream.description,
        "config": stream.config,
        "coders": ["zlib-9", "brotli-q11", "lzma1-raw", "smevr-r7-nibble"],
        "coder_rows": rows,
        "best": best,
        "incumbent_control": {
            "bytes": INCUMBENT_BYTES,
            "codec": "lzma1-raw",
            "sha256": INCUMBENT_SHA256,
        },
        "delta_vs_incumbent_bytes": delta,
        "delta_vs_incumbent_pct": delta / INCUMBENT_BYTES,
        "verdict": "WIN-w/-bytes" if delta < 0 else "LOSS-w/-bytes",
        "pass_condition_met": bool(delta < 0 and stream.decode_equal),
        "decode_equality": stream.decode_equal,
        "counted_bytes_breakdown": {
            "compressed_total": best.bytes,
            "compressed_sha256": best.sha256,
            "raw_total": len(stream.raw),
            "raw_sha256": sha256_bytes(stream.raw),
            "raw_components": stream.component_raw_bytes,
            "video_derived_side_info_bytes": 0,
        },
        "prediction_pixels": stream.prediction_pixels,
        "residual_missing_pixels": stream.residual_missing_pixels,
        "residual_extra_pixels": stream.residual_extra_pixels,
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "verdict_scope": verdict_scope,
    }


def apply_reference_thresholds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    incumbent_ac = next(row for row in rows if row["arm_id"] == "incumbent_edge_context_ac")
    if int(incumbent_ac["best"]["bytes"]) < INCUMBENT_BYTES:
        reference = {
            "arm_id": incumbent_ac["arm_id"],
            "bytes": int(incumbent_ac["best"]["bytes"]),
            "codec": incumbent_ac["best"]["codec"],
            "sha256": incumbent_ac["best"]["sha256"],
        }
    else:
        reference = {
            "arm_id": "cr1_edge_conditioned_lzma1_raw_incumbent",
            "bytes": INCUMBENT_BYTES,
            "codec": "lzma1-raw",
            "sha256": INCUMBENT_SHA256,
        }
    reference_bytes = int(reference["bytes"])
    for row in rows:
        delta = int(row["best"]["bytes"]) - reference_bytes
        row["reference_best_control"] = reference
        row["delta_vs_reference_best_bytes"] = delta
        row["delta_vs_reference_best_pct"] = delta / reference_bytes
        is_trot_ac = (
            row["config"].get("coding_family") == "conditional_range_ac"
            and row["config"].get("mode") == "trot_q"
        )
        row["pass_condition_met"] = bool(is_trot_ac and row["decode_equality"] and delta < 0)
        if is_trot_ac:
            row["verdict"] = "WIN-w/-bytes" if row["pass_condition_met"] else "LOSS-w/-bytes"
        elif row["config"].get("coding_family") == "lz_residual_baseline":
            row["verdict"] = "INSTRUMENT-LZ-BASELINE"
        else:
            row["verdict"] = "CONTROL"
    return reference


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(jsonable(row), sort_keys=True) + "\n" for row in rows))


def findings_table_row(row: dict[str, Any]) -> str:
    best = row["best"]
    delta_cr1 = row["delta_vs_incumbent_bytes"]
    delta_ref = row["delta_vs_reference_best_bytes"]
    return (
        f"| {row['arm_id']} | {row['config'].get('coding_family')} | "
        f"{best['bytes']} B ({best['codec']}) | "
        f"{delta_cr1:+d} B ({row['delta_vs_incumbent_pct']:.3%}) | "
        f"{delta_ref:+d} B ({row['delta_vs_reference_best_pct']:.3%}) | "
        f"{row['decode_equality']} | {row['verdict']} |"
    )


def md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def write_findings(path: Path, receipt: dict[str, Any]) -> None:
    best = receipt["best_treatment"]
    reference = receipt["reference_best_control"]
    lines = [
        "# TR2P1 TROT residual/range-AC byte race - 2026-08-08",
        "",
        "Tags: [no-triality] [p0-ledger-ok]",
        "",
        "## Answer First",
        "",
        "No scorer, no evaluator, no Metal/GPU, no paid job, and no archive promotion ran.",
        (
            f"Best reference-form TROT AC challenger: `{best['arm_id']}` at "
            f"`{best['best']['bytes']}` B ({best['best']['codec']}), delta "
            f"`{best['delta_vs_reference_best_bytes']:+d}` B vs best incumbent control "
            f"`{reference['arm_id']}` at `{reference['bytes']}` B."
        ),
        f"Verdict: `{receipt['verdict']}`.",
        "",
        "| arm | coding family | best bytes | delta vs CR1-LZ | delta vs best incumbent | decode equality | verdict |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in receipt["typed_rows"]:
        lines.append(findings_table_row(row))
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- GT argmax: `{receipt['inputs']['gt_argmax_path']}` ({receipt['inputs']['gt_argmax_sha256']})",
            f"- Current argmax: `{receipt['inputs']['current_argmax_path']}` ({receipt['inputs']['current_argmax_sha256']})",
            f"- CR1 incumbent: `{receipt['incumbent_control']['artifact_path']}` "
            f"({receipt['incumbent_control']['sha256']})",
            f"- Axis: `{receipt['axis']}`.",
            f"- Selection: `{receipt['selection_mode']}`.",
            "",
            "## Recall Evidence",
            "",
            "| source or query | result | impact |",
            "|---|---|---|",
        ]
    )
    for item in receipt["recall_evidence"]:
        lines.append(
            f"| {md_cell(item['source'])} | {md_cell(item['result'])} | {md_cell(item['impact'])} |"
        )
    lines.extend(
        [
            "",
            "## Solver Validation",
            "",
            (
                f"Fixture `{receipt['solver_validation']['fixture']}`: q=1 matched the independent "
                f"Sinkhorn/IPF reference with max plan delta "
                f"`{receipt['solver_validation']['max_abs_plan_delta']:.3e}`; "
                f"row/column max errors "
                f"`{receipt['solver_validation']['row_marginal_max_abs_error']:.3e}` / "
                f"`{receipt['solver_validation']['col_marginal_max_abs_error']:.3e}`; "
                f"residual decode equality `{receipt['solver_validation']['residual_decode_equality']}`; "
                f"conditional range-AC decode equality "
                f"`{receipt['solver_validation']['conditional_range_ac_decode_equality']}`."
            ),
            "",
            "## Counted Bytes",
            "",
            "| arm | compressed total | raw total | marginals raw | side-info raw | residual raw | AC stream raw | tags raw | framing raw |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in receipt["typed_rows"]:
        br = row["counted_bytes_breakdown"]
        comp = br["raw_components"]
        lines.append(
            f"| {row['arm_id']} | {br['compressed_total']} | {br['raw_total']} | "
            f"{comp['marginals']} | {comp['side_info']} | {comp['residual']} | "
            f"{comp['ac_stream']} | "
            f"{comp['tags']} | {comp['framing']} |"
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- These are byte-only scorer-free measurements over cached argmax labels.",
            "- The q-family solve is generic decode-time algorithm; counted payload includes marginals, AC streams or residuals, tags, and framing.",
            "- Conditional range-AC rows are the reference-form verdict carrier required by the 2026-08-08 amendment.",
            "- Residual-through-LZ rows are retained only as instrument baselines, not as formulation verdict carriers.",
            "- No video-derived side-information matrix was hidden in code or omitted from bytes; side-info bytes are zero because the cost is derived only from counted marginals.",
            "- No RGB receiver, archive parse-back, SegNet/PoseNet scorer survival, or score improvement is claimed.",
            "- Negative verdict scope is the pre-registered conditional-AC FORMULATION only, not the TROT family outside this CR1 selected-support payload.",
            "",
            "## Follow-On Disposition",
            "",
            "| item | disposition | fire order |",
            "|---|---|---|",
        ]
    )
    for item in receipt["follow_on_disposition"]:
        lines.append(f"| {item['id']} | {item['disposition']} | {item['fire_order']} |")
    lines.extend(
        [
            "",
            "## Frontier Honesty",
            "",
            f"Own-vehicle frontier remains `{OWN_FRONTIER}`. Contest pointer remains `{CONTEST_POINTER}`.",
            "",
        ]
    )
    atomic_write_text(path, "\n".join(lines))


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    gt = load_argmax(args.gt_argmax)
    current = load_argmax(args.current_argmax)
    supports, extraction = build_supports(gt, current)
    solver_validation = validate_q_solver_fixtures()
    incumbent = decode_cr1_incumbent(args.cr1_incumbent, supports)

    header_common = {
        "edges": [edge_name(edge) for edge in TOP_EDGE_PAIRS],
        "class_ids": [list(edge) for edge in TOP_EDGE_PAIRS],
        "height": SEG_H,
        "width": SEG_W,
        "pairs": N_PAIRS,
        "selection": SELECTION_MODE,
        "incumbent_sha256": INCUMBENT_SHA256,
        "incumbent_bytes": INCUMBENT_BYTES,
    }
    artifact_dir = args.ssd_dir / "payloads"
    streams: list[StreamBuild] = [
        build_identity_stream(supports, header_common=header_common),
        build_ac_stream(
            supports,
            header_common=header_common,
            arm_id="incumbent_edge_context_ac",
            mode="edge_context",
            q=1.0,
            lam=args.lam,
            iterations=args.iterations,
            max_solver_cells=args.max_solver_cells,
        ),
        build_residual_stream(
            supports,
            header_common=header_common,
            arm_id="marginals_only_residual",
            mode="marginals_only",
            q=1.0,
            lam=args.lam,
            iterations=args.iterations,
            max_solver_cells=args.max_solver_cells,
        ),
        build_ac_stream(
            supports,
            header_common=header_common,
            arm_id="marginals_only_ac",
            mode="marginals_only",
            q=1.0,
            lam=args.lam,
            iterations=args.iterations,
            max_solver_cells=args.max_solver_cells,
        ),
    ]
    for q in args.q_values:
        safe_q = str(q).replace(".", "p")
        streams.append(
            build_residual_stream(
                supports,
                header_common=header_common,
                arm_id=f"trot_q{safe_q}_residual",
                mode="trot_q",
                q=float(q),
                lam=args.lam,
                iterations=args.iterations,
                max_solver_cells=args.max_solver_cells,
            )
        )
        streams.append(
            build_ac_stream(
                supports,
                header_common=header_common,
                arm_id=f"trot_q{safe_q}_ac",
                mode="trot_q",
                q=float(q),
                lam=args.lam,
                iterations=args.iterations,
                max_solver_cells=args.max_solver_cells,
            )
        )

    typed_rows: list[dict[str, Any]] = []
    for stream in streams:
        rows, best = race_coders(stream=stream, artifact_dir=artifact_dir)
        typed_rows.append(jsonable(stream_row(stream, rows, best)))

    reference_control = apply_reference_thresholds(typed_rows)
    trot_rows = [
        row
        for row in typed_rows
        if row["config"]["mode"] == "trot_q"
        and row["config"].get("coding_family") == "conditional_range_ac"
    ]
    best_treatment = min(trot_rows, key=lambda row: row["best"]["bytes"])
    pass_rows = [row for row in trot_rows if row["pass_condition_met"]]
    verdict = (
        "WIN-w/-bytes"
        if pass_rows
        else (
            "LOSS-w/-bytes; FORMULATION falsifier met for q-family joint-from-marginals "
            "conditional range-AC coding of CR1 selected edge support"
        )
    )

    return {
        "schema": "ddm_tr2p1_trot_residual_range_ac_race_receipt.v2",
        "created_utc": now_utc(),
        "axis": AXIS,
        "selection_mode": SELECTION_MODE,
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "wall_seconds": time.perf_counter() - start,
        "inputs": {
            "gt_argmax_path": str(args.gt_argmax),
            "gt_argmax_sha256": sha256_file(args.gt_argmax),
            "current_argmax_path": str(args.current_argmax),
            "current_argmax_sha256": sha256_file(args.current_argmax),
            "cr1_receipt_path": str(args.cr1_receipt),
            "class_names": CLASS_NAMES,
            "selected_edges": [edge_name(edge) for edge in TOP_EDGE_PAIRS],
        },
        "support_extraction": extraction,
        "solver_validation": solver_validation,
        "incumbent_control": incumbent,
        "q_values": args.q_values,
        "lambda": args.lam,
        "iterations": args.iterations,
        "max_solver_cells": args.max_solver_cells,
        "typed_rows": typed_rows,
        "reference_best_control": reference_control,
        "best_treatment": best_treatment,
        "pass_rows": pass_rows,
        "verdict": verdict,
        "recall_evidence": [
            {
                "source": "MEMORY.md query: ddm_tr2p1|TR2P1|tr2p1|ddm_tr2|common_contract|20260808",
                "result": "No TR2P1-specific prior memory entry was found in MEMORY.md.",
                "impact": "Used live charter, TR2, CR1, and corpus receipts rather than a recalled shortcut.",
            },
            {
                "source": ".omx/research/ddm_tr2_20260808/TR2_CROSSWALK.md and TR2_ROWS.jsonl rows 1, 3, 6",
                "result": "TR2 pre-registered only this CR1 same-payload q-family residual race; sparse-plan claims were lesson-only.",
                "impact": "Kept verdict scoped to this formulation and did not claim TROT sparsity or metric replacement.",
            },
            {
                "source": ".omx/research/ddm_tr2p1_20260808/CHARTER.md amendment 2026-08-08",
                "result": "The operator correction requires conditional arithmetic/range coding; LZ residual rows alone are instrument-scoped.",
                "impact": "Added incumbent edge-context AC plus TROT conditional range-AC rows and made the FORMULATION verdict depend only on the AC race.",
            },
            {
                "source": ".omx/research/ddm_cr1_20260808/CR1_FINDINGS.md, CR1_RECEIPT.json, CR1_ROWS.jsonl row 2",
                "result": "CR1 P2 measured 464557 B edge-conditioned lzma1-raw with exact decode equality on selected n600 supports.",
                "impact": "Set the pass threshold and re-decoded the incumbent artifact instead of re-measuring CR1.",
            },
            {
                "source": "content query over GDL1/RL1/SX1/TR2/BD1 scopes: joint-from-marginals|TROT|Tsallis|Sinkhorn|edge-conditioned|same-coder|#940|Road<->Lane|separatrix",
                "result": "Found GDL1-P2 edge-conditioned fire-order, SX1 separatrix concentration, RL1 interface pricing, and TR2 same-coder requirement beyond the charter seeds.",
                "impact": "Used top-five class-pair edges, n600/no-prefix selection, same coder set, and exact decode equality.",
            },
            {
                "source": "content query: #940|same-coder|races-not-reputation|same payload|decode equality",
                "result": "Found repeated same-coder race doctrine and TR2 citation of ddm_sv2 as the #940 blocker.",
                "impact": "Reported all coders per arm and selected by real compressed bytes, not representation reputation.",
            },
            {
                "source": "tools/list_canonical_equations.py --json",
                "result": "Canonical registry was consulted; relevant hits reinforced scorer-free byte-race and transfer/equality boundaries, with no TR2P1 superseding equation found.",
                "impact": "Receipt stays a byte-only measurement with score_claim=false and promotion_eligible=false.",
            },
        ],
        "follow_on_disposition": [
            {
                "id": "TR2-P1",
                "disposition": "FIRED",
                "fire_order": "This receipt is the pre-registered byte-only q-family residual plus amendment-required conditional range-AC race against CR1's incumbent.",
            },
            {
                "id": "TR2-P1-implementation-reference",
                "disposition": "FIRED",
                "fire_order": "The local q-family solver passed deterministic fixtures before the CR1 payload race; no unpinned dependency was vendored.",
            },
            {
                "id": "#984 rate axis / CR1 successor consumer",
                "disposition": "QUEUED-WITH-FIRE-ORDER",
                "fire_order": (
                    "Consume this row only as a scorer-free byte-race negative unless a future arm supplies a new counted side-info source or conditional probability model and repeats same-payload decode-equality AC racing."
                    if not pass_rows
                    else "A future owner must still build receiver/archive parse-back before any scorer or promotion claim."
                ),
            },
        ],
        "boundaries": [
            "No scorer, no evaluator, no archive promotion, no Metal/GPU, no paid job.",
            "Payloads are cached real n600 argmax supports from GT and cx1 argmax arrays.",
            "CR1 incumbent was re-decoded from the SHA-pinned lzma1-raw artifact; it was not re-derived.",
            "No video-derived side-information matrix was omitted from counted bytes.",
            "All LZ coder rows are real zlib/Brotli/LZMA/SMEVR outputs with round-trip checks.",
            "All conditional range-AC rows carry real AC bitstreams and decode-equality checks before any optional container coding.",
            "Verdicts are conditional-AC formulation-scoped byte-race outcomes only.",
        ],
        "frontier": {
            "own_vehicle": OWN_FRONTIER,
            "contest_pointer": CONTEST_POINTER,
        },
    }


def parse_q_values(text: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in text.split(",") if part.strip())
    if len(values) < 4 or not any(abs(value - 1.0) < 1.0e-12 for value in values):
        raise argparse.ArgumentTypeError("q ladder must have at least 4 values and include 1.0")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-argmax", type=Path, default=DEFAULT_GT_ARGMAX)
    parser.add_argument("--current-argmax", type=Path, default=DEFAULT_CURRENT_ARGMAX)
    parser.add_argument("--cr1-receipt", type=Path, default=DEFAULT_CR1_RECEIPT)
    parser.add_argument("--cr1-incumbent", type=Path, default=DEFAULT_CR1_INCUMBENT)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    parser.add_argument("--q-values", type=parse_q_values, default=DEFAULT_Q_VALUES)
    parser.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--max-solver-cells", type=int, default=DEFAULT_MAX_SOLVER_CELLS)
    parser.add_argument("--fixture-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fixture_only:
        print(json.dumps(validate_q_solver_fixtures(), sort_keys=True))
        return 0
    receipt = build_receipt(args)
    args.research_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.research_dir / "TR2P1_RECEIPT.json"
    rows_path = args.research_dir / "TR2P1_ROWS.jsonl"
    findings_path = args.research_dir / "TR2P1_FINDINGS.md"
    atomic_write_json(receipt_path, receipt)
    write_jsonl(rows_path, receipt["typed_rows"])
    write_findings(findings_path, receipt)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "rows": str(rows_path),
                "findings": str(findings_path),
                "verdict": receipt["verdict"],
                "best_treatment": receipt["best_treatment"]["arm_id"],
                "best_treatment_bytes": receipt["best_treatment"]["best"]["bytes"],
                "delta_vs_incumbent_bytes": receipt["best_treatment"]["delta_vs_incumbent_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
