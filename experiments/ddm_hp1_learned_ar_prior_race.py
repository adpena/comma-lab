# SPDX-License-Identifier: MIT
"""ddm_hp1: learned conditional-prior byte race on the live IX2 token stream.

This is a scorer-free rate-axis experiment.  It opens a byte-closed tq1c archive
read-only, extracts the receiver-input ``IX2TOK01`` token lattice, trains a small
counted 10K-int8 conditional table on that lattice, range-codes the lattice, and
verifies exact decode equality.  The learned table is video-derived and counted
inside the HP1 frame; no scorer outputs or hidden receiver tables are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import os
import platform
import struct
import sys
import time
import zipfile
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
for entry in (str(REPO), str(REPO / "src"), str(REPO / "experiments")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments import ddm_r7_token_coder as r7
from tac.optimization import ddm_ix2_archive_container as ix2

try:
    import brotli
except ImportError:  # pragma: no cover - brotli is an environment dependency.
    brotli = None  # type: ignore[assignment]


SCHEMA: Final = "ddm_hp1_learned_ar_prior_race.v1"
AXIS: Final = "[macOS-CPU byte-only scorer-free]"
RATE_DENOMINATOR: Final = 37_545_489
DEFAULT_ARCHIVE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/"
    "candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes"
)
EXPECTED_TQ1C_SHA256: Final = "b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06"
DEFAULT_RECEIPT_DIR: Final = REPO / ".omx/research/ddm_hp1_20260806"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_hp1_20260806")

HP1_MAGIC: Final = b"HP1LAP1!"
HP1_VERSION: Final = 1
HP1_HEADER: Final = struct.Struct("<8sBBBBHHHHHII32s")
LEVELS: Final = 16
DEFAULT_CONTEXT_ROWS: Final = 625
DEFAULT_PATCH: Final = 4
DEFAULT_MAX_MODEL_BYTES: Final = 10_000
CONTEXT_MODE_IDS: Final = {
    "prev_k": 1,
    "prev_left": 2,
    "prev_up": 3,
    "prev_chan": 4,
    "hash_prev_spatial": 5,
}
ID_CONTEXT_MODES: Final = {value: key for key, value in CONTEXT_MODE_IDS.items()}
DEFAULT_CONTEXT_MODE_RACE: Final = (
    "prev_k",
    "prev_left",
    "prev_up",
    "prev_chan",
    "hash_prev_spatial",
)

STATE_BITS: Final = 32
FULL_RANGE: Final = 1 << STATE_BITS
HALF: Final = FULL_RANGE >> 1
QUARTER: Final = HALF >> 1
THREE_QUARTERS: Final = 3 * QUARTER

_IX2_TOKEN_HEADER: Final = struct.Struct("<BBBBHHHHII")
_IX2_LZMA_FILTERS: Final = [
    {"id": lzma.FILTER_LZMA1, "dict_size": 1 << 24, "lc": 3, "lp": 0, "pb": 0}
]
_RAW_LZMA_FILTERS: Final = [
    {"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}
]


class HP1Error(ValueError):
    """Raised when the HP1 stream or measurement precondition fails closed."""


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
            raise HP1Error("range stream is empty")
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
            raise HP1Error("range target escaped the frequency row")
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
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def score_rate_delta(byte_delta: int) -> float:
    return 25.0 * int(byte_delta) / RATE_DENOMINATOR


def _validate_codes(codes: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(codes, dtype=np.uint8)
    if array.ndim != 4 or not array.size:
        raise HP1Error("token lattice must be non-empty uint8 [P,R,C,K]")
    if any(int(value) <= 0 or int(value) > 65535 for value in array.shape):
        raise HP1Error("token lattice shape is outside uint16 bounds")
    if int(array.max()) >= LEVELS:
        raise HP1Error("token lattice escapes the 16-symbol alphabet")
    return array


def load_live_token_stream(archive_path: Path, expected_sha256: str | None = EXPECTED_TQ1C_SHA256) -> dict[str, Any]:
    archive_path = Path(archive_path)
    archive_bytes, archive_sha = sha256_file(archive_path)
    if expected_sha256 and archive_sha != expected_sha256:
        raise HP1Error(f"archive sha256 differs: {archive_sha} != {expected_sha256}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        infos = zf.infolist()
        if tuple(info.filename for info in infos) != ("0.bin",):
            raise HP1Error(f"expected single 0.bin archive, got {[info.filename for info in infos]!r}")
        if infos[0].compress_type != zipfile.ZIP_STORED or infos[0].is_dir():
            raise HP1Error("0.bin is not a stored file member")
        member = zf.read("0.bin")
    bulk, joint = ix2.parse_payload(member)
    if bulk[: len(ix2.TOKEN_FRAME_MAGIC)] != ix2.TOKEN_FRAME_MAGIC:
        raise HP1Error("bulk section is not IX2TOK01")
    start = time.monotonic()
    codes = _validate_codes(ix2.decode_token_frame(bulk))
    decode_seconds = time.monotonic() - start
    reencoded = ix2.encode_token_frame(codes)
    if reencoded != bulk:
        raise HP1Error("IX2 token frame does not re-encode canonically")
    return {
        "archive_path": str(archive_path),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "member_bytes": len(member),
        "member_sha256": sha256_bytes(member),
        "token_bulk": bulk,
        "token_bulk_bytes": len(bulk),
        "token_bulk_sha256": sha256_bytes(bulk),
        "joint_section_sizes": [len(section) for section in joint],
        "joint_section_sha256": [sha256_bytes(section) for section in joint],
        "ix2_decode_seconds": decode_seconds,
        "codes": codes,
    }


def context_rows_for_mode(
    mode: str,
    *,
    channels: int,
    hash_context_rows: int = DEFAULT_CONTEXT_ROWS,
) -> int:
    if mode == "prev_k":
        return 17 * channels
    if mode in {"prev_left", "prev_up", "prev_chan"}:
        return 17 * 17
    if mode == "hash_prev_spatial":
        return hash_context_rows
    raise HP1Error(f"unknown context mode {mode!r}")


def _context_id(
    array: np.ndarray,
    p: int,
    r: int,
    c: int,
    k: int,
    context_rows: int,
    patch: int,
    mode: str,
) -> int:
    prev = int(array[p - 1, r, c, k]) if p else LEVELS
    left = int(array[p, r, c - 1, k]) if c else LEVELS
    up = int(array[p, r - 1, c, k]) if r else LEVELS
    chan = int(array[p, r, c, k - 1]) if k else LEVELS
    if mode == "prev_k":
        return prev * int(array.shape[3]) + k
    if mode == "prev_left":
        return prev * 17 + left
    if mode == "prev_up":
        return prev * 17 + up
    if mode == "prev_chan":
        return prev * 17 + chan
    if mode != "hash_prev_spatial":
        raise HP1Error(f"unknown context mode {mode!r}")
    patch_r = r // patch
    patch_c = c // patch
    return (
        prev * 131
        + left * 67
        + up * 31
        + chan * 17
        + k * 11
        + patch_r * 5
        + patch_c
    ) % context_rows


def _iter_patch_cells(rows: int, cols: int, patch: int):
    for pr in range(0, rows, patch):
        for pc in range(0, cols, patch):
            for r in range(pr, min(pr + patch, rows)):
                for c in range(pc, min(pc + patch, cols)):
                    yield r, c


def train_context_table(
    codes: np.ndarray,
    *,
    context_rows: int | None = None,
    patch: int = DEFAULT_PATCH,
    mode: str = "hash_prev_spatial",
) -> tuple[bytes, dict[str, Any]]:
    """Train a counted 625x16 uint8 conditional table on the real token lattice."""

    array = _validate_codes(codes)
    if context_rows is None:
        context_rows = context_rows_for_mode(
            mode,
            channels=int(array.shape[3]),
            hash_context_rows=DEFAULT_CONTEXT_ROWS,
        )
    if context_rows <= 0 or context_rows > 65535:
        raise HP1Error("context_rows is outside uint16 bounds")
    p_count, row_count, col_count, channel_count = array.shape
    counts = np.zeros((context_rows, LEVELS), dtype=np.uint32)
    start = time.monotonic()
    for r, c in _iter_patch_cells(row_count, col_count, patch):
        for k in range(channel_count):
            for p in range(p_count):
                ctx = _context_id(array, p, r, c, k, context_rows, patch, mode)
                counts[ctx, int(array[p, r, c, k])] += 1
    train_seconds = time.monotonic() - start

    model = np.empty((context_rows, LEVELS), dtype=np.uint8)
    active_rows = 0
    for row_index, row in enumerate(counts):
        if int(row.sum()) == 0:
            model[row_index] = 16
            continue
        active_rows += 1
        smoothed = row.astype(np.float64) + 0.5
        scaled = np.rint(smoothed / smoothed.sum() * 240.0).astype(np.int64)
        scaled = np.maximum(scaled, 1)
        scaled = np.minimum(scaled, 255)
        model[row_index] = scaled.astype(np.uint8)
    return model.tobytes(), {
        "context_rows": context_rows,
        "model_raw_bytes": int(model.size),
        "active_context_rows": active_rows,
        "patch": patch,
        "context_mode": mode,
        "train_seconds": train_seconds,
    }


def _cumulative_rows(model_bytes: bytes, context_rows: int) -> tuple[list[list[int]], list[int]]:
    expected = context_rows * LEVELS
    if len(model_bytes) != expected:
        raise HP1Error(f"model length {len(model_bytes)} != expected {expected}")
    model = np.frombuffer(model_bytes, dtype=np.uint8).reshape(context_rows, LEVELS)
    if np.any(model == 0):
        raise HP1Error("model frequency table contains zero")
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


def encode_tokens_with_model(
    codes: np.ndarray,
    model_bytes: bytes,
    *,
    context_rows: int = DEFAULT_CONTEXT_ROWS,
    patch: int = DEFAULT_PATCH,
    mode: str = "hash_prev_spatial",
) -> bytes:
    array = _validate_codes(codes)
    cum_rows, totals = _cumulative_rows(model_bytes, context_rows)
    encoder = _RangeEncoder()
    p_count, row_count, col_count, channel_count = array.shape
    for r, c in _iter_patch_cells(row_count, col_count, patch):
        for k in range(channel_count):
            for p in range(p_count):
                ctx = _context_id(array, p, r, c, k, context_rows, patch, mode)
                encoder.encode(int(array[p, r, c, k]), cum_rows[ctx], totals[ctx])
    return encoder.finish()


def decode_tokens_with_model(
    stream: bytes,
    shape: tuple[int, int, int, int],
    model_bytes: bytes,
    *,
    context_rows: int = DEFAULT_CONTEXT_ROWS,
    patch: int = DEFAULT_PATCH,
    mode: str = "hash_prev_spatial",
) -> np.ndarray:
    if any(int(value) <= 0 or int(value) > 65535 for value in shape):
        raise HP1Error("declared shape is outside uint16 bounds")
    cum_rows, totals = _cumulative_rows(model_bytes, context_rows)
    decoder = _RangeDecoder(stream)
    out = np.zeros(shape, dtype=np.uint8)
    p_count, row_count, col_count, channel_count = shape
    for r, c in _iter_patch_cells(row_count, col_count, patch):
        for k in range(channel_count):
            for p in range(p_count):
                ctx = _context_id(out, p, r, c, k, context_rows, patch, mode)
                out[p, r, c, k] = decoder.decode(cum_rows[ctx], totals[ctx])
    return np.ascontiguousarray(out)


def build_hp1_frame(
    codes: np.ndarray,
    model_bytes: bytes,
    stream: bytes,
    *,
    context_rows: int = DEFAULT_CONTEXT_ROWS,
    patch: int = DEFAULT_PATCH,
    mode: str = "hash_prev_spatial",
) -> bytes:
    array = _validate_codes(codes)
    p_count, row_count, col_count, channel_count = array.shape
    if len(model_bytes) != context_rows * LEVELS:
        raise HP1Error("model bytes do not match context_rows*16")
    mode_id = CONTEXT_MODE_IDS.get(mode)
    if mode_id is None:
        raise HP1Error(f"unknown context mode {mode!r}")
    header = HP1_HEADER.pack(
        HP1_MAGIC,
        HP1_VERSION,
        LEVELS,
        mode_id,
        patch,
        p_count,
        row_count,
        col_count,
        channel_count,
        context_rows,
        len(model_bytes),
        len(stream),
        hashlib.sha256(array.tobytes()).digest(),
    )
    return header + model_bytes + stream


def decode_hp1_frame(frame: bytes, *, verify_canonical: bool = True) -> np.ndarray:
    if len(frame) < HP1_HEADER.size:
        raise HP1Error("HP1 frame is truncated")
    (
        magic,
        version,
        levels,
        mode_id,
        patch,
        p_count,
        row_count,
        col_count,
        channel_count,
        context_rows,
        model_len,
        stream_len,
        digest,
    ) = HP1_HEADER.unpack_from(frame)
    if magic != HP1_MAGIC or version != HP1_VERSION or levels != LEVELS:
        raise HP1Error("HP1 frame magic/version/levels differ")
    mode = ID_CONTEXT_MODES.get(mode_id)
    if mode is None:
        raise HP1Error("HP1 context mode id differs")
    off = HP1_HEADER.size
    model = frame[off : off + model_len]
    off += model_len
    stream = frame[off : off + stream_len]
    off += stream_len
    if off != len(frame) or len(model) != model_len or len(stream) != stream_len:
        raise HP1Error("HP1 frame lengths do not close")
    shape = (p_count, row_count, col_count, channel_count)
    decoded = decode_tokens_with_model(
        stream,
        shape,
        model,
        context_rows=context_rows,
        patch=patch,
        mode=mode,
    )
    if hashlib.sha256(decoded.tobytes()).digest() != digest:
        raise HP1Error("HP1 decoded token SHA-256 differs")
    if verify_canonical:
        recoded = encode_tokens_with_model(
            decoded,
            model,
            context_rows=context_rows,
            patch=patch,
            mode=mode,
        )
        if recoded != stream:
            raise HP1Error("HP1 frame is noncanonical")
    return decoded


def empirical_entropy_from_counts(counts: np.ndarray) -> tuple[float, int]:
    total = int(counts.sum())
    if total <= 0:
        raise HP1Error("entropy count table is empty")
    probs = counts[counts > 0].astype(np.float64) / float(total)
    entropy = float(-(probs * np.log2(probs)).sum())
    return entropy, total


def entropy_baselines(codes: np.ndarray) -> dict[str, Any]:
    array = _validate_codes(codes)
    p_count, row_count, col_count, channel_count = array.shape
    order0_counts = np.bincount(array.reshape(-1), minlength=LEVELS).astype(np.uint64)
    order0_h, total = empirical_entropy_from_counts(order0_counts)

    prev_counts = np.zeros((17 * channel_count, LEVELS), dtype=np.uint64)
    spatial_counts = np.zeros((17 * 17 * 17 * channel_count, LEVELS), dtype=np.uint64)
    prev_spatial_counts = np.zeros((17 * 17 * 17 * 17 * channel_count, LEVELS), dtype=np.uint64)

    for p in range(p_count):
        for r in range(row_count):
            for c in range(col_count):
                for k in range(channel_count):
                    symbol = int(array[p, r, c, k])
                    prev = int(array[p - 1, r, c, k]) if p else LEVELS
                    left = int(array[p, r, c - 1, k]) if c else LEVELS
                    up = int(array[p, r - 1, c, k]) if r else LEVELS
                    chan = int(array[p, r, c, k - 1]) if k else LEVELS
                    prev_counts[prev * channel_count + k, symbol] += 1
                    spatial_idx = (((left * 17) + up) * 17 + chan) * channel_count + k
                    spatial_counts[spatial_idx, symbol] += 1
                    both_idx = ((((prev * 17) + left) * 17 + up) * 17 + chan) * channel_count + k
                    prev_spatial_counts[both_idx, symbol] += 1

    def cond(table: np.ndarray) -> dict[str, float | int]:
        weighted = 0.0
        active = 0
        for row in table:
            row_total = int(row.sum())
            if row_total:
                active += 1
                h, _ = empirical_entropy_from_counts(row)
                weighted += h * row_total
        bits = weighted / total
        return {
            "bits_per_symbol": bits,
            "ideal_bytes": bits * total / 8.0,
            "active_contexts": active,
        }

    return {
        "symbol_count": total,
        "alphabet": [int(value) for value in np.flatnonzero(order0_counts)],
        "order0": {
            "bits_per_symbol": order0_h,
            "ideal_bytes": order0_h * total / 8.0,
            "active_symbols": int((order0_counts > 0).sum()),
        },
        "spatial_context_left_up_channelprev": cond(spatial_counts),
        "prev_pair_conditioned_same_cell": cond(prev_counts),
        "prev_pair_plus_spatial_context": cond(prev_spatial_counts),
    }


def _lzma1_raw(payload: bytes) -> bytes:
    return lzma.compress(payload, format=lzma.FORMAT_RAW, filters=_RAW_LZMA_FILTERS)


def _ix2_lzma1(payload: bytes) -> bytes:
    return lzma.compress(payload, format=lzma.FORMAT_RAW, filters=_IX2_LZMA_FILTERS)


def forced_lzma_ix2_token_frame(codes: np.ndarray) -> bytes:
    array = _validate_codes(codes)
    base, delta = r7.factor_mode_delta(array, LEVELS)
    residual = np.ascontiguousarray(np.transpose(delta, (1, 2, 3, 0)))
    block_r = _ix2_lzma1(r7.pack_nibbles(residual.reshape(-1)))
    block_b = _ix2_lzma1(r7.pack_nibbles(base.reshape(-1)))
    p_count, row_count, col_count, channel_count = array.shape
    header = _IX2_TOKEN_HEADER.pack(
        LEVELS,
        3,
        3,
        0,
        p_count,
        row_count,
        col_count,
        channel_count,
        len(block_r),
        len(block_b),
    )
    frame = ix2.TOKEN_FRAME_MAGIC + header + block_r + block_b
    if not np.array_equal(ix2.decode_token_frame(frame), array):
        raise HP1Error("forced LZMA IX2 frame decode differs")
    return frame


def raw_token_frame(codes: np.ndarray) -> bytes:
    array = _validate_codes(codes)
    return struct.pack("<4H", *array.shape) + array.tobytes()


def measure_baseline_coders(
    codes: np.ndarray,
    shipped_bulk: bytes,
    *,
    retained_dir: Path,
) -> dict[str, Any]:
    if brotli is None:
        raise HP1Error("brotli dependency unavailable")
    raw = raw_token_frame(codes)
    raw_lzma = _lzma1_raw(raw)
    raw_brotli = brotli.compress(raw, quality=11, lgwin=24)
    if lzma.decompress(raw_lzma, format=lzma.FORMAT_RAW, filters=_RAW_LZMA_FILTERS) != raw:
        raise HP1Error("raw token LZMA baseline decode differs")
    if brotli.decompress(raw_brotli) != raw:
        raise HP1Error("raw token Brotli baseline decode differs")
    ix2_lzma = forced_lzma_ix2_token_frame(codes)
    retained = {
        "raw_token_frame": retain_payload(retained_dir / "raw_token_frame.bin", raw),
        "shipped_ix2_brotli_q11": retain_payload(
            retained_dir / "shipped_ix2_brotli_q11.bin", shipped_bulk
        ),
        "forced_ix2_lzma1": retain_payload(retained_dir / "forced_ix2_lzma1.bin", ix2_lzma),
        "raw_token_frame_lzma1": retain_payload(
            retained_dir / "raw_token_frame_lzma1.bin", raw_lzma
        ),
        "raw_token_frame_brotli_q11": retain_payload(
            retained_dir / "raw_token_frame_brotli_q11.bin", raw_brotli
        ),
    }
    return {
        "raw_token_frame": retained["raw_token_frame"],
        "shipped_ix2_brotli_q11": {
            "bytes": len(shipped_bulk),
            "sha256": sha256_bytes(shipped_bulk),
            "retained_payload": retained["shipped_ix2_brotli_q11"],
            "note": "exact IX2TOK01 bulk section from tq1c archive; both residual and base blocks use coder_id=2 Brotli",
        },
        "forced_ix2_lzma1": {
            "bytes": len(ix2_lzma),
            "sha256": sha256_bytes(ix2_lzma),
            "retained_payload": retained["forced_ix2_lzma1"],
            "delta_vs_shipped_bytes": len(ix2_lzma) - len(shipped_bulk),
            "delta_vs_shipped_s_rate": score_rate_delta(len(ix2_lzma) - len(shipped_bulk)),
        },
        "raw_token_frame_lzma1": {
            "bytes": len(raw_lzma),
            "sha256": sha256_bytes(raw_lzma),
            "retained_payload": retained["raw_token_frame_lzma1"],
            "delta_vs_shipped_bytes": len(raw_lzma) - len(shipped_bulk),
            "delta_vs_shipped_s_rate": score_rate_delta(len(raw_lzma) - len(shipped_bulk)),
        },
        "raw_token_frame_brotli_q11": {
            "bytes": len(raw_brotli),
            "sha256": sha256_bytes(raw_brotli),
            "retained_payload": retained["raw_token_frame_brotli_q11"],
            "delta_vs_shipped_bytes": len(raw_brotli) - len(shipped_bulk),
            "delta_vs_shipped_s_rate": score_rate_delta(len(raw_brotli) - len(shipped_bulk)),
        },
    }


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


def retain_payload(path: Path, payload: bytes) -> dict[str, Any]:
    """Persist one materialized candidate before its scalar metrics are reported."""
    write_bytes(path, payload)
    nbytes, digest = sha256_file(path)
    expected = sha256_bytes(payload)
    if nbytes != len(payload) or digest != expected:
        raise HP1Error(f"retained payload verification failed for {path}")
    return {"path": str(path), "bytes": nbytes, "sha256": digest}


def run_hp1(
    *,
    archive: Path = DEFAULT_ARCHIVE,
    receipt_dir: Path = DEFAULT_RECEIPT_DIR,
    ssd_dir: Path = DEFAULT_SSD_DIR,
    expected_sha256: str | None = EXPECTED_TQ1C_SHA256,
    context_rows: int = DEFAULT_CONTEXT_ROWS,
    patch: int = DEFAULT_PATCH,
    max_model_bytes: int = DEFAULT_MAX_MODEL_BYTES,
) -> dict[str, Any]:
    run_start = time.monotonic()
    live = load_live_token_stream(archive, expected_sha256)
    codes = live["codes"]
    token_sha = sha256_bytes(codes.tobytes())

    entropy_start = time.monotonic()
    entropy = entropy_baselines(codes)
    entropy_seconds = time.monotonic() - entropy_start

    learned_rows: list[dict[str, Any]] = []
    best_frame: bytes | None = None
    best_row: dict[str, Any] | None = None
    for mode in DEFAULT_CONTEXT_MODE_RACE:
        rows = context_rows_for_mode(
            mode,
            channels=int(codes.shape[3]),
            hash_context_rows=context_rows,
        )
        model_raw_bytes = rows * LEVELS
        if model_raw_bytes > max_model_bytes:
            learned_rows.append(
                {
                    "context_mode": mode,
                    "context_rows": rows,
                    "model_raw_bytes": model_raw_bytes,
                    "skipped": True,
                    "reason": f"model exceeds max_model_bytes={max_model_bytes}",
                }
            )
            continue
        model_bytes, model_info = train_context_table(
            codes,
            context_rows=rows,
            patch=patch,
            mode=mode,
        )
        encode_start = time.monotonic()
        learned_stream = encode_tokens_with_model(
            codes,
            model_bytes,
            context_rows=rows,
            patch=patch,
            mode=mode,
        )
        encode_seconds = time.monotonic() - encode_start
        frame = build_hp1_frame(
            codes,
            model_bytes,
            learned_stream,
            context_rows=rows,
            patch=patch,
            mode=mode,
        )
        decode_start = time.monotonic()
        decoded = decode_hp1_frame(frame, verify_canonical=True)
        decode_seconds = time.monotonic() - decode_start
        if not np.array_equal(decoded, codes):
            raise HP1Error(f"HP1 decode equality failed for context mode {mode}")
        retained_frame = retain_payload(
            ssd_dir / "retained" / "learned_contexts" / f"{mode}.hp1",
            frame,
        )
        row = {
            "context_mode": mode,
            "context_rows": rows,
            "model_raw_bytes": len(model_bytes),
            "header_bytes": HP1_HEADER.size,
            "range_stream_bytes": len(learned_stream),
            "frame_bytes": len(frame),
            "frame_sha256": sha256_bytes(frame),
            "retained_payload": retained_frame,
            "model_info": model_info,
            "encode_seconds": encode_seconds,
            "decode_seconds_including_canonical_reencode": decode_seconds,
            "decode_equal": True,
            "canonical_reencode_equal": True,
            "skipped": False,
        }
        learned_rows.append(row)
        if best_row is None or int(row["frame_bytes"]) < int(best_row["frame_bytes"]):
            best_row = row
            best_frame = frame
    if best_row is None or best_frame is None:
        raise HP1Error("no learned-prior context mode was measured")

    baselines = measure_baseline_coders(
        codes,
        live["token_bulk"],
        retained_dir=ssd_dir / "retained" / "baseline_coders",
    )
    frame_path = ssd_dir / "hp1_learned_prior.hp1"
    write_bytes(frame_path, best_frame)
    frame_bytes, frame_sha = sha256_file(frame_path)

    shipped_bytes = int(baselines["shipped_ix2_brotli_q11"]["bytes"])
    learned_delta = frame_bytes - shipped_bytes
    lzma_bytes = int(baselines["forced_ix2_lzma1"]["bytes"])
    raw_symbols = int(np.prod(codes.shape))
    patch_groups = int(math.ceil(codes.shape[1] / patch) * math.ceil(codes.shape[2] / patch))
    stream_count = int(codes.shape[1] * codes.shape[2] * codes.shape[3])

    result = {
        "schema": SCHEMA,
        "axis": AXIS,
        "score_claim": False,
        "scorer_forwards_run": 0,
        "ran_upstream_evaluate_py": False,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "source_archive": {
            key: value
            for key, value in live.items()
            if key not in {"codes", "token_bulk"}
        },
        "token_stream": {
            "definition": "receiver-input IX2TOK01 token lattice decoded from tq1c 0.bin bulk section",
            "not_semantic_labels": True,
            "shape": [int(value) for value in codes.shape],
            "dtype": str(codes.dtype),
            "alphabet": [int(value) for value in np.unique(codes)],
            "raw_symbol_count": raw_symbols,
            "raw_uint8_bytes": int(codes.nbytes),
            "sha256": token_sha,
        },
        "entropy": {
            **entropy,
            "measurement_seconds": entropy_seconds,
            "condition_on_prev_not_diffs": True,
        },
        "baselines": baselines,
        "learned_prior": {
            "family": "small counted learned conditional prior with static int8 frequency table and exact range coder",
            "context_mode": best_row["context_mode"],
            "context_rows": best_row["context_rows"],
            "levels": LEVELS,
            "max_model_bytes": max_model_bytes,
            "model_raw_bytes": best_row["model_raw_bytes"],
            "header_bytes": HP1_HEADER.size,
            "range_stream_bytes": best_row["range_stream_bytes"],
            "frame_bytes": frame_bytes,
            "frame_sha256": frame_sha,
            "frame_path": str(frame_path),
            "delta_vs_shipped_bytes": learned_delta,
            "delta_vs_shipped_s_rate": score_rate_delta(learned_delta),
            "model_info": best_row["model_info"],
            "encode_seconds": best_row["encode_seconds"],
            "decode_seconds_including_canonical_reencode": best_row["decode_seconds_including_canonical_reencode"],
            "decode_equal": True,
            "canonical_reencode_equal": True,
            "raced_context_modes": learned_rows,
        },
        "byte_ladder": [
            {
                "rank": 1,
                "name": "shipped_ix2_brotli_q11",
                "bytes": shipped_bytes,
                "delta_vs_shipped_bytes": 0,
                "delta_vs_shipped_s_rate": 0.0,
            },
            {
                "rank": 2,
                "name": "forced_ix2_lzma1",
                "bytes": lzma_bytes,
                "delta_vs_shipped_bytes": lzma_bytes - shipped_bytes,
                "delta_vs_shipped_s_rate": score_rate_delta(lzma_bytes - shipped_bytes),
            },
            {
                "rank": 3,
                "name": "hp1_learned_prior_plus_counted_model",
                "bytes": frame_bytes,
                "delta_vs_shipped_bytes": learned_delta,
                "delta_vs_shipped_s_rate": score_rate_delta(learned_delta),
            },
        ],
        "decode_feasibility": {
            "patch": patch,
            "patch_groups_per_pair": patch_groups,
            "cell_channel_temporal_streams": stream_count,
            "scalar_range_updates_full_stream": raw_symbols,
            "measured_full_decode_seconds_including_canonical_reencode": best_row["decode_seconds_including_canonical_reencode"],
            "thirty_minute_budget_seconds": 1800,
            "within_budget_on_this_host": bool(float(best_row["decode_seconds_including_canonical_reencode"]) < 1800),
            "no_94_step_claim": True,
            "note": "prototype decodes scalar range symbols in Python; a production receiver could batch by patch group, but this result claims only the measured scalar path.",
        },
        "scope_regrade": {
            "task_918_scope": "INSTRUMENT-SCOPED to explicit LZ/rank/basis/token-coder families already raced",
            "learned_conditional_prior_was_not_measured_by_918": True,
            "this_hp1_family_verdict_scope": "FAMILY on this exact tq1c IX2 token stream for this 10K-int8 static-context learned prior",
            "lz_family_closure_stands": True,
        },
        "falsifier": {
            "net_delta_bytes_ge_0": learned_delta >= 0,
            "decode_projection_exceeds_budget": float(best_row["decode_seconds_including_canonical_reencode"]) >= 1800,
            "verdict": "FAMILY_NEGATIVE_ON_THIS_STREAM" if learned_delta >= 0 else "NET_POSITIVE_NEEDS_MAIN_COMPOSITION",
        },
        "run_seconds": time.monotonic() - run_start,
    }
    result_path = receipt_dir / "hp1_results.json"
    write_json(result_path, result)
    return result


def _self_test() -> None:
    rng = np.random.default_rng(20260806)
    base = rng.integers(0, LEVELS, size=(1, 4, 4, 2), dtype=np.uint8)
    codes = np.repeat(base, 9, axis=0)
    noise = rng.integers(0, LEVELS, size=codes.shape, dtype=np.uint8)
    mask = rng.random(codes.shape) < 0.08
    codes = np.where(mask, noise, codes).astype(np.uint8)
    model, _info = train_context_table(codes, context_rows=37, patch=2, mode="hash_prev_spatial")
    stream = encode_tokens_with_model(codes, model, context_rows=37, patch=2, mode="hash_prev_spatial")
    frame = build_hp1_frame(codes, model, stream, context_rows=37, patch=2, mode="hash_prev_spatial")
    restored = decode_hp1_frame(frame)
    if not np.array_equal(restored, codes):
        raise AssertionError("self-test HP1 frame roundtrip failed")
    entropy = entropy_baselines(codes)
    if not (
        entropy["prev_pair_conditioned_same_cell"]["bits_per_symbol"]
        < entropy["order0"]["bits_per_symbol"]
    ):
        raise AssertionError("self-test temporal context did not lower entropy")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--expected-sha256", default=EXPECTED_TQ1C_SHA256)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPT_DIR)
    parser.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    parser.add_argument("--context-rows", type=int, default=DEFAULT_CONTEXT_ROWS)
    parser.add_argument("--patch", type=int, default=DEFAULT_PATCH)
    parser.add_argument("--max-model-bytes", type=int, default=DEFAULT_MAX_MODEL_BYTES)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "ok", "schema": SCHEMA}, sort_keys=True))
        return 0
    result = run_hp1(
        archive=args.archive,
        receipt_dir=args.receipt_dir,
        ssd_dir=args.ssd_dir,
        expected_sha256=args.expected_sha256 or None,
        context_rows=args.context_rows,
        patch=args.patch,
        max_model_bytes=args.max_model_bytes,
    )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "shipped_bytes": result["byte_ladder"][0]["bytes"],
                "lzma_bytes": result["byte_ladder"][1]["bytes"],
                "learned_plus_model_bytes": result["byte_ladder"][2]["bytes"],
                "learned_delta_vs_shipped_bytes": result["byte_ladder"][2]["delta_vs_shipped_bytes"],
                "learned_delta_vs_shipped_s_rate": result["byte_ladder"][2]["delta_vs_shipped_s_rate"],
                "verdict": result["falsifier"]["verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
