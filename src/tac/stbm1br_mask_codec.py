"""Standalone decoder for PR90 ``STBM1BR`` semantic topband mask segments.

The public PR90 qrepro archive stores its semantic masks as::

    STBM1BR\0 + brotli(QTBM* topband/road-boundary stream)

This module reimplements the narrow decode surface needed to prove and replay
that mask stream inside PR85-family local candidate pipelines.  It intentionally
does not import from the PR90 source checkout or touch any scorer/runtime model.
"""

from __future__ import annotations

import bz2
import hashlib
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli
import numpy as np


STBM1BR_MAGIC = b"STBM1BR\0"
QTBM_MAGICS = (b"QTBM5\0", b"QTBM4\0", b"QTBM3\0", b"QTBM2\0", b"QTBM1\0")
N_CLASSES = 5
N_SYM = N_CLASSES - 1
DEFAULT_SHAPE = (600, 384, 512)

FEAT_DIAG_TLTL = 0
FEAT_LEFT_LEFT = 1
FEAT_TOP_TOP_TOP = 2
FEAT_PREV_PREV_PREV = 3
FEAT_DIAG_TRTR = 4
FEAT_PREV_LEFT = 5
FEAT_PREV_RIGHT = 6
FEAT_PREV_TOP = 7
FEAT_PREV_BOTTOM = 8
FEAT_PREV2_LEFT = 9
FEAT_PREV2_RIGHT = 10
FEAT_PREV_BOTTOM_RIGHT = 11
FEAT_PREV_BOTTOM_LEFT = 12
FEAT_PREV_TOP_RIGHT = 13
FEAT_PREV_BOTTOM2 = 14
FEAT_PREV_RIGHT2 = 15
FEAT_X_BIN5 = 16
FEAT_Y_BIN5 = 17
FEAT_X_BIN5_SHIFT = 20
FEAT_PEEL_DIST42 = 30
FEAT_PEEL_BOUND5 = 31
FEAT_PEEL_SLOPE5 = 32


class STBM1BRError(ValueError):
    """Raised when an STBM1BR payload violates the self-contained contract."""


@dataclass(frozen=True)
class STBM1BRMetadata:
    """Parsed metadata for a self-describing STBM1BR mask segment."""

    segment_bytes: int
    segment_sha256: str
    brotli_body_bytes: int
    brotli_body_sha256: str
    qtbm_blob_bytes: int
    qtbm_blob_sha256: str
    qtbm_magic: str
    n_pairs: int
    height: int
    width: int
    precision: int
    top_bins: int
    boundary_xbins: int
    shift_dy: int
    shift_dx: int
    residual_order: tuple[int, ...] | None
    top_payload_bytes: int
    road_payload_bytes: int
    spatial_table_bytes: int
    m5_table_bytes: int
    sparse_feature_ids: tuple[int, ...]
    sparse_table_bytes: int
    residual_bitstream_bytes: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _RangeDecoder:
    TOP = 0xFFFFFFFF
    HALF = 0x80000000
    QUARTER = 0x40000000
    THREE_QUARTER = 0xC0000000

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0
        self.low = 0
        self.high = self.TOP
        self.code = 0
        for _ in range(32):
            self.code = ((self.code << 1) | self._read_bit()) & self.TOP

    def _read_bit(self) -> int:
        if self.byte_pos >= len(self.data):
            return 0
        byte = self.data[self.byte_pos]
        bit = (byte >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
        return bit

    def decode_target(self, total: int) -> int:
        rng = self.high - self.low + 1
        return ((self.code - self.low + 1) * total - 1) // rng

    def advance(self, cum_low: int, cum_high: int, total: int) -> None:
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cum_high) // total - 1
        self.low = self.low + (rng * cum_low) // total

        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.low -= self.HALF
                self.high -= self.HALF
                self.code -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREE_QUARTER:
                self.low -= self.QUARTER
                self.high -= self.QUARTER
                self.code -= self.QUARTER
            else:
                break
            self.low = (self.low << 1) & self.TOP
            self.high = ((self.high << 1) | 1) & self.TOP
            self.code = ((self.code << 1) | self._read_bit()) & self.TOP


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise STBM1BRError(message)


def _m5_ctx(top_v: int, left_v: int, tl_v: int, tr_v: int, prev_v: int) -> int:
    return ((((top_v * 5 + left_v) * 5 + tl_v) * 5 + tr_v) * 5 + prev_v)


def _leb128_decode_big_deltas(buf: bytes, pos: int, count: int) -> tuple[np.ndarray, int]:
    if count < 0:
        raise STBM1BRError(f"negative LEB128 count: {count}")
    deltas = np.empty(count, dtype=np.int64)
    for i in range(count):
        result = 0
        shift = 0
        while True:
            if pos >= len(buf):
                raise STBM1BRError("truncated LEB128 delta stream")
            byte = buf[pos]
            pos += 1
            result |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
            if shift > 63:
                raise STBM1BRError("overlong LEB128 delta stream")
        deltas[i] = result
    return deltas, pos


def decode_boundary_mask_payload(payload: bytes, n_pairs: int, height: int, width: int) -> np.ndarray:
    if payload[:5] == b"QBD1\0":
        pos = 5
        _require(pos + 16 <= len(payload), "QBD1 boundary payload is truncated")
        first_len, dx_len, err_len, err_count = struct.unpack_from("<IIII", payload, pos)
        pos += 16
        _require(pos + first_len <= len(payload), "QBD1 first-boundary stream overruns payload")
        first = np.frombuffer(
            bz2.decompress(payload[pos : pos + first_len]), dtype="<u2", count=n_pairs
        ).astype(np.int16)
        pos += first_len
        _require(pos + dx_len <= len(payload), "QBD1 delta stream overruns payload")
        dx = np.frombuffer(
            bz2.decompress(payload[pos : pos + dx_len]), dtype=np.int8, count=n_pairs * (width - 1)
        ).reshape(n_pairs, width - 1)
        pos += dx_len
        _require(pos + err_len <= len(payload), "QBD1 exception stream overruns payload")
        err_raw = brotli.decompress(payload[pos : pos + err_len])
        pos += err_len
    elif payload[:5] == b"QBD2\0":
        pos = 5
        _require(pos + 3 + 16 <= len(payload), "QBD2 boundary payload is truncated")
        bins, dx_nsym, dx_offset = struct.unpack_from("<BBB", payload, pos)
        pos += 3
        first_len, dx_len, err_len, err_count = struct.unpack_from("<IIII", payload, pos)
        pos += 16
        _require(bins > 0 and dx_nsym > 0, "QBD2 has empty bin or symbol table")
        _require(pos + first_len <= len(payload), "QBD2 first-boundary stream overruns payload")
        first = np.frombuffer(
            bz2.decompress(payload[pos : pos + first_len]), dtype="<u2", count=n_pairs
        ).astype(np.int16)
        pos += first_len
        freq_bytes = bins * dx_nsym * 2
        _require(pos + freq_bytes <= len(payload), "QBD2 frequency table overruns payload")
        freqs = (
            np.frombuffer(payload, dtype="<u2", count=bins * dx_nsym, offset=pos)
            .astype(np.int64)
            .reshape(bins, dx_nsym)
        )
        pos += freq_bytes
        _require(pos + dx_len <= len(payload), "QBD2 arithmetic delta stream overruns payload")
        bitstream = payload[pos : pos + dx_len]
        pos += dx_len
        _require(pos + err_len <= len(payload), "QBD2 exception stream overruns payload")
        err_raw = brotli.decompress(payload[pos : pos + err_len])
        pos += err_len
        cdf = np.zeros((bins, dx_nsym + 1), dtype=np.int64)
        cdf[:, 1:] = np.cumsum(freqs, axis=1)
        total = int(cdf[0, -1])
        _require(total > 0, "QBD2 frequency total must be positive")
        dec = _RangeDecoder(bitstream)
        dx = np.empty((n_pairs, width - 1), dtype=np.int16)
        for fi in range(n_pairs):
            for x in range(width - 1):
                row = cdf[(x * bins) // (width - 1)]
                target = dec.decode_target(total)
                sym = int(np.searchsorted(row, target, side="right") - 1)
                if sym < 0 or sym >= dx_nsym:
                    raise STBM1BRError(f"QBD2 decoded symbol out of range: {sym}")
                dx[fi, x] = sym - int(dx_offset)
                dec.advance(int(row[sym]), int(row[sym + 1]), total)
    else:
        raise STBM1BRError(f"bad QBD boundary payload magic: {payload[:5]!r}")

    _require(pos == len(payload), "QBD boundary payload has trailing bytes")
    _require(first.size == n_pairs, "QBD first-boundary count mismatch")
    bounds = np.empty((n_pairs, width), dtype=np.int16)
    bounds[:, 0] = first
    bounds[:, 1:] = first[:, None] + np.cumsum(dx.astype(np.int16), axis=1)
    yy = np.arange(height, dtype=np.int16)[None, :, None]
    out = (yy >= bounds[:, None, :]).astype(np.uint8)

    if err_count:
        deltas, used = _leb128_decode_big_deltas(err_raw, 0, err_count)
        _require(used == len(err_raw), "trailing QBD exception bytes")
        idx = np.cumsum(deltas, dtype=np.int64) - 1
        _require(idx.size == 0 or (int(idx[0]) >= 0 and int(idx[-1]) < out.size), "QBD exception index out of range")
        flat = out.reshape(-1)
        flat[idx] ^= 1
    return out


def decode_topband_payload(payload: bytes, n_pairs: int, height: int, width: int) -> np.ndarray:
    if payload[:5] not in (b"QTB1\0", b"QTB2\0", b"QTB3\0", b"QTBZ\0"):
        raise STBM1BRError(f"bad QTB top-band payload magic: {payload[:5]!r}")
    version = payload[:5]
    pos = 5
    err_len = 0
    if version == b"QTBZ\0":
        _require(pos + struct.calcsize("<HHHI") <= len(payload), "QTBZ payload is truncated")
        n2, w2, bins, bounds_len = struct.unpack_from("<HHHI", payload, pos)
        pos += struct.calcsize("<HHHI")
        _require(pos + bounds_len <= len(payload), "QTBZ boundary stream overruns payload")
        bounds_raw = bz2.decompress(payload[pos : pos + bounds_len])
        pos += bounds_len
        bounds = np.frombuffer(bounds_raw, dtype="<u2", count=n_pairs * bins).reshape(n_pairs, bins)
    elif version == b"QTB2\0":
        _require(pos + struct.calcsize("<HHHI") <= len(payload), "QTB2 payload is truncated")
        n2, w2, bins, err_len = struct.unpack_from("<HHHI", payload, pos)
        pos += struct.calcsize("<HHHI")
        bounds_dtype = "<u2"
    elif version == b"QTB3\0":
        _require(pos + struct.calcsize("<HHH") <= len(payload), "QTB3 payload is truncated")
        n2, w2, bins = struct.unpack_from("<HHH", payload, pos)
        pos += struct.calcsize("<HHH")
        bounds_dtype = "u1"
    else:
        _require(pos + struct.calcsize("<HHH") <= len(payload), "QTB1 payload is truncated")
        n2, w2, bins = struct.unpack_from("<HHH", payload, pos)
        pos += struct.calcsize("<HHH")
        bounds_dtype = "<u2"
    _require(n2 == n_pairs and w2 == width, "QTB top-band dimensions do not match stream")
    _require(bins > 0, "QTB top-band bins must be positive")
    if version != b"QTBZ\0":
        dtype = np.dtype(bounds_dtype)
        bounds_bytes = n_pairs * bins * dtype.itemsize
        _require(pos + bounds_bytes <= len(payload), "QTB bounds overrun payload")
        bounds = np.frombuffer(payload, dtype=dtype, count=n_pairs * bins, offset=pos).reshape(n_pairs, bins)
        pos += bounds_bytes
    out = np.zeros((n_pairs, height, width), dtype=np.uint8)
    for b in range(bins):
        x0 = (b * width) // bins
        x1 = ((b + 1) * width) // bins
        for fi in range(n_pairs):
            out[fi, : int(bounds[fi, b]), x0:x1] = 1
    if err_len:
        _require(pos + err_len <= len(payload), "QTB exception stream overruns payload")
        err_raw = brotli.decompress(payload[pos : pos + err_len])
        pos += err_len
        _require(len(err_raw) >= 4, "QTB exception stream is truncated")
        (err_count,) = struct.unpack_from("<I", err_raw, 0)
        deltas, used = _leb128_decode_big_deltas(err_raw, 4, err_count)
        _require(used == len(err_raw), "trailing QTB exception bytes")
        idx = np.cumsum(deltas, dtype=np.int64) - 1
        _require(idx.size == 0 or (int(idx[0]) >= 0 and int(idx[-1]) < out.size), "QTB exception index out of range")
        out.reshape(-1)[idx] = 0
    _require(pos == len(payload), "QTB top-band payload has trailing bytes")
    return out


def unpack_sparse_big(compressed: bytes, precision: int) -> tuple[np.ndarray, np.ndarray]:
    raw = zlib.decompress(compressed, -15)
    return unpack_sparse_big_plain(raw, precision)


def unpack_sparse_big_plain(raw: bytes, precision: int) -> tuple[np.ndarray, np.ndarray]:
    _require(len(raw) >= 4, "sparse table is truncated before count")
    (n_fired,) = struct.unpack_from("<I", raw, 0)
    if n_fired == 0:
        _require(len(raw) == 4, "empty sparse table has trailing bytes")
        return np.zeros((0,), dtype=np.int64), np.zeros((0, N_SYM), dtype=np.uint16)
    pos = 4
    deltas, pos = _leb128_decode_big_deltas(raw, pos, n_fired)
    table_bytes = n_fired * (N_SYM - 1) * 2
    _require(pos + table_bytes == len(raw), "sparse table length mismatch")
    partial = (
        np.frombuffer(raw, dtype="<u2", count=n_fired * (N_SYM - 1), offset=pos)
        .reshape(n_fired, N_SYM - 1)
        .copy()
    )
    last = (1 << precision) - partial.astype(np.int64).sum(axis=1)
    _require(not (last < 0).any() and not (last > 0xFFFF).any(), "invalid sparse frequency row")
    freqs = np.empty((n_fired, N_SYM), dtype=np.uint16)
    freqs[:, : N_SYM - 1] = partial
    freqs[:, N_SYM - 1] = last.astype(np.uint16)
    fired_idx = (np.cumsum(deltas, dtype=np.int64) - 1).astype(np.int64)
    return fired_idx, freqs


def unpack_sparse_big_plain_colsfirst(raw: bytes, precision: int) -> tuple[np.ndarray, np.ndarray]:
    _require(len(raw) >= 4, "cols-first sparse table is truncated before count")
    (n_fired,) = struct.unpack_from("<I", raw, 0)
    if n_fired == 0:
        _require(len(raw) == 4, "empty cols-first sparse table has trailing bytes")
        return np.zeros((0,), dtype=np.int64), np.zeros((0, N_SYM), dtype=np.uint16)
    pos = 4
    table_bytes = n_fired * (N_SYM - 1) * 2
    _require(pos + table_bytes <= len(raw), "cols-first sparse table overruns payload")
    partial = (
        np.frombuffer(raw, dtype="<u2", count=n_fired * (N_SYM - 1), offset=pos)
        .reshape(N_SYM - 1, n_fired)
        .T.copy()
    )
    pos += table_bytes
    deltas, pos = _leb128_decode_big_deltas(raw, pos, n_fired)
    _require(pos == len(raw), "cols-first sparse table has trailing bytes")
    last = (1 << precision) - partial.astype(np.int64).sum(axis=1)
    _require(not (last < 0).any() and not (last > 0xFFFF).any(), "invalid cols-first sparse frequency row")
    freqs = np.empty((n_fired, N_SYM), dtype=np.uint16)
    freqs[:, : N_SYM - 1] = partial
    freqs[:, N_SYM - 1] = last.astype(np.uint16)
    fired_idx = (np.cumsum(deltas, dtype=np.int64) - 1).astype(np.int64)
    return fired_idx, freqs


def _decode_frame_topband(
    dec: _RangeDecoder,
    frame: np.ndarray,
    prev_frame: list[list[int]] | None,
    prev_prev_frame: list[list[int]] | None,
    prev_prev_prev_frame: list[list[int]] | None,
    top_support_frame: np.ndarray,
    road_frame: np.ndarray,
    spatial_cdf_flat_py: list[list[int]],
    m5_cdf_flat_py: list[list[int]],
    fired_cdf_flat_py: list[list[int]],
    m12_to_slot_dict: dict[int, int],
    feat_ids: tuple[int, ...],
    total: int,
    height: int,
    width: int,
    shift_dy: int,
    shift_dx: int,
    inv: list[int],
) -> list[list[int]]:
    top_support = np.asarray(top_support_frame, dtype=bool)
    road = np.asarray(road_frame, dtype=bool)
    active = ~(top_support | road)
    base = np.zeros((height, width), dtype=np.uint8)
    base[road] = 4
    base[top_support] = 2
    frame_list = base.tolist()
    active_xs_by_row = [np.flatnonzero(active[y]).tolist() for y in range(height)]
    prev_list = prev_frame
    pp_list = prev_prev_frame
    ppp_list = prev_prev_prev_frame
    get_target = dec.decode_target
    advance = dec.advance
    m12_get = m12_to_slot_dict.get
    specialized_sparse_features = feat_ids == (
        FEAT_DIAG_TLTL,
        FEAT_PREV_RIGHT2,
        FEAT_PREV_BOTTOM2,
        FEAT_X_BIN5_SHIFT,
        FEAT_PEEL_DIST42,
    )
    x_bin5_shift = []
    if specialized_sparse_features:
        x_shift = width // 10
        for bx in range(width):
            value = ((bx + x_shift) * 5) // width
            x_bin5_shift.append(4 if value > 4 else value)

    rev_road = road[::-1]
    rev_nonroad = ~rev_road
    has_nonroad = rev_nonroad.any(axis=0)
    first_nonroad_from_bottom = np.argmax(rev_nonroad, axis=0)
    bottom_is_road = road[height - 1]
    road_bounds_arr = np.full(width, height, dtype=np.int16)
    road_bounds_arr[bottom_is_road] = np.where(
        has_nonroad[bottom_is_road],
        height - first_nonroad_from_bottom[bottom_is_road],
        0,
    )
    road_bounds = road_bounds_arr.tolist()

    for y in range(height):
        prev_row = prev_list[y] if prev_list is not None else None
        pp_row = pp_list[y] if pp_list is not None else None
        prev_row_above = prev_list[y - 1] if (prev_list is not None and y > 0) else None
        prev_row_below = prev_list[y + 1] if (prev_list is not None and y + 1 < height) else None
        ppp_row = ppp_list[y] if ppp_list is not None else None
        for x in active_xs_by_row[y]:
            top_v = frame_list[y - 1][x] if y > 0 else 0
            left_v = frame_list[y][x - 1] if x > 0 else 0
            tl_v = frame_list[y - 1][x - 1] if (y > 0 and x > 0) else 0
            tr_v = frame_list[y - 1][x + 1] if (y > 0 and x + 1 < width) else 0
            if prev_list is None:
                cdf_row = spatial_cdf_flat_py[((top_v * 5 + left_v) * 5 + tl_v) * 5 + tr_v]
            else:
                sy = y + shift_dy
                sx = x + shift_dx
                if 0 <= sy < height and 0 <= sx < width:
                    prev_v = prev_list[sy][sx]
                    pp_v = pp_list[sy][sx] if pp_list is not None else 0
                else:
                    prev_v = 0
                    pp_v = 0
                tt_v = frame_list[y - 2][x] if y > 1 else 0
                m5_ctx = ((((top_v * 5 + left_v) * 5 + tl_v) * 5 + tr_v) * 5 + prev_v)
                cdf_row = m5_cdf_flat_py[m5_ctx]
                if pp_list is not None and fired_cdf_flat_py:
                    m12_ctx = ((m5_ctx * 5 + pp_v) * 5 + tt_v)
                    if specialized_sparse_features:
                        fv0 = frame_list[y - 2][x - 2] if (y >= 2 and x >= 2) else 0
                        fv1 = prev_row[x + 2] if x + 2 < width else 0
                        fv2 = prev_list[y + 2][x] if y + 2 < height else 0
                        dist = road_bounds[x] - y
                        if dist <= 0:
                            fv4 = 0
                        else:
                            fv4 = ((dist - 1) // 42) + 1
                            if fv4 > 4:
                                fv4 = 4
                        m12_ctx = (
                            (((m12_ctx * 5 + fv0) * 5 + fv1) * 5 + fv2) * 5
                            + x_bin5_shift[x]
                        ) * 5 + fv4
                    else:
                        for fid in feat_ids:
                            if fid == FEAT_DIAG_TLTL:
                                fv = frame_list[y - 2][x - 2] if (y >= 2 and x >= 2) else 0
                            elif fid == FEAT_LEFT_LEFT:
                                fv = frame_list[y][x - 2] if x >= 2 else 0
                            elif fid == FEAT_TOP_TOP_TOP:
                                fv = frame_list[y - 3][x] if y >= 3 else 0
                            elif fid == FEAT_PREV_PREV_PREV:
                                fv = ppp_row[x] if ppp_row is not None else 0
                            elif fid == FEAT_DIAG_TRTR:
                                fv = frame_list[y - 2][x + 2] if (y >= 2 and x + 2 < width) else 0
                            elif fid == FEAT_PREV_LEFT:
                                fv = prev_row[x - 1] if (prev_row is not None and x >= 1) else 0
                            elif fid == FEAT_PREV_RIGHT:
                                fv = prev_row[x + 1] if (prev_row is not None and x + 1 < width) else 0
                            elif fid == FEAT_PREV_TOP:
                                fv = prev_row_above[x] if prev_row_above is not None else 0
                            elif fid == FEAT_PREV_BOTTOM:
                                fv = prev_row_below[x] if prev_row_below is not None else 0
                            elif fid == FEAT_PREV2_LEFT:
                                fv = pp_row[x - 1] if (pp_row is not None and x >= 1) else 0
                            elif fid == FEAT_PREV2_RIGHT:
                                fv = pp_row[x + 1] if (pp_row is not None and x + 1 < width) else 0
                            elif fid == FEAT_PREV_BOTTOM_RIGHT:
                                fv = prev_row_below[x + 1] if (prev_row_below is not None and x + 1 < width) else 0
                            elif fid == FEAT_PREV_BOTTOM_LEFT:
                                fv = prev_row_below[x - 1] if (prev_row_below is not None and x >= 1) else 0
                            elif fid == FEAT_PREV_TOP_RIGHT:
                                fv = prev_row_above[x + 1] if (prev_row_above is not None and x + 1 < width) else 0
                            elif fid == FEAT_PREV_BOTTOM2:
                                fv = prev_list[y + 2][x] if (prev_list is not None and y + 2 < height) else 0
                            elif fid == FEAT_PREV_RIGHT2:
                                fv = prev_row[x + 2] if (prev_row is not None and x + 2 < width) else 0
                            elif fid == FEAT_X_BIN5:
                                fv = (x * 5) // width
                            elif fid == FEAT_Y_BIN5:
                                fv = (y * 5) // height
                            elif fid == FEAT_X_BIN5_SHIFT:
                                fv = ((x + width // 10) * 5) // width
                                if fv > 4:
                                    fv = 4
                            elif fid == FEAT_PEEL_DIST42:
                                dist = road_bounds[x] - y
                                if dist <= 0:
                                    fv = 0
                                else:
                                    fv = ((dist - 1) // 42) + 1
                                    if fv > 4:
                                        fv = 4
                            elif fid == FEAT_PEEL_BOUND5:
                                fv = (road_bounds[x] * 5) // height
                                if fv > 4:
                                    fv = 4
                            elif fid == FEAT_PEEL_SLOPE5:
                                prev_bound = road_bounds[x - 1] if x >= 1 else road_bounds[x]
                                delta = road_bounds[x] - prev_bound
                                if delta > 2:
                                    delta = 2
                                fv = delta + 2
                                if fv < 0:
                                    fv = 0
                            else:
                                raise STBM1BRError(f"unsupported STBM sparse feature id: {fid}")
                            m12_ctx = m12_ctx * 5 + fv
                    slot = m12_get(m12_ctx)
                    if slot is not None:
                        cdf_row = fired_cdf_flat_py[slot]

            target = get_target(total)
            sym = 0
            while cdf_row[sym + 1] <= target:
                sym += 1
                if sym >= N_SYM:
                    raise STBM1BRError(f"decoded residual symbol out of range: {sym}")
            frame_list[y][x] = inv[sym]
            advance(cdf_row[sym], cdf_row[sym + 1], total)
    for y in range(height):
        frame[y] = frame_list[y]
    return frame_list


def _parse_qtbm_blob(blob: bytes) -> tuple[dict[str, Any], int]:
    pos = 0
    magic = None
    for candidate in QTBM_MAGICS:
        if blob.startswith(candidate):
            magic = candidate
            pos = len(candidate)
            break
    if magic is None:
        raise STBM1BRError(f"bad QTBM magic: {blob[:8]!r}")
    _require(pos + struct.calcsize("<HHHBBBbb") <= len(blob), "QTBM blob is truncated before dimensions")
    n_pairs, height, width, precision, top_bins, boundary_xbins, shift_dy, shift_dx = struct.unpack_from(
        "<HHHBBBbb", blob, pos
    )
    pos += struct.calcsize("<HHHBBBbb")
    has_residual_order = magic != b"QTBM1\0"
    residual_order = None
    if has_residual_order:
        _require(pos + N_SYM <= len(blob), "QTBM blob is truncated before residual order")
        residual_order = tuple(int(v) for v in blob[pos : pos + N_SYM])
        _require(sorted(residual_order) == [0, 1, 2, 3], f"invalid QTBM residual order: {residual_order!r}")
        pos += N_SYM
    _require(pos + 8 <= len(blob), "QTBM blob is truncated before top/road lengths")
    top_len, road_len = struct.unpack_from("<II", blob, pos)
    pos += 8
    top_start = pos
    top_end = top_start + int(top_len)
    road_start = top_end
    road_end = road_start + int(road_len)
    _require(road_end <= len(blob), "QTBM top/road payloads overrun blob")
    pos = road_end
    _require(pos + 4 <= len(blob), "QTBM blob is truncated before table lengths")
    spatial_size, m5_size = struct.unpack_from("<HH", blob, pos)
    pos += 4
    spatial_start = pos
    spatial_end = spatial_start + int(spatial_size)
    m5_start = spatial_end
    m5_end = m5_start + int(m5_size)
    _require(m5_end <= len(blob), "QTBM sparse frequency tables overrun blob")
    pos = m5_end
    _require(pos < len(blob), "QTBM blob is missing sparse feature count")
    n_feats = int(blob[pos])
    pos += 1
    _require(pos + n_feats <= len(blob), "QTBM blob is truncated in sparse feature ids")
    feat_ids = tuple(int(v) for v in blob[pos : pos + n_feats])
    pos += n_feats
    _require(pos + 6 <= len(blob), "QTBM blob is truncated before sparse table")
    threshold_q8, sparse_len = struct.unpack_from("<HI", blob, pos)
    pos += 6
    sparse_start = pos
    sparse_end = sparse_start + int(sparse_len)
    _require(sparse_end + 4 <= len(blob), "QTBM sparse table overruns blob")
    pos = sparse_end
    (bitstream_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    bitstream_start = pos
    bitstream_end = bitstream_start + int(bitstream_len)
    _require(bitstream_end == len(blob), "QTBM residual bitstream does not consume blob exactly")
    meta = {
        "magic": magic,
        "n_pairs": int(n_pairs),
        "height": int(height),
        "width": int(width),
        "precision": int(precision),
        "top_bins": int(top_bins),
        "boundary_xbins": int(boundary_xbins),
        "shift_dy": int(shift_dy),
        "shift_dx": int(shift_dx),
        "residual_order": residual_order,
        "top_payload": blob[top_start:top_end],
        "road_payload": blob[road_start:road_end],
        "spatial_table": blob[spatial_start:spatial_end],
        "m5_table": blob[m5_start:m5_end],
        "sparse_feature_ids": feat_ids,
        "threshold_q8": int(threshold_q8),
        "sparse_table": blob[sparse_start:sparse_end],
        "residual_bitstream": blob[bitstream_start:bitstream_end],
    }
    return meta, bitstream_end


def parse_stbm1br_metadata(segment: bytes) -> STBM1BRMetadata:
    """Parse an STBM1BR segment and return fail-closed byte metadata."""

    if not segment.startswith(STBM1BR_MAGIC):
        raise STBM1BRError(f"bad STBM1BR magic: {segment[:8]!r}")
    brotli_body = segment[len(STBM1BR_MAGIC) :]
    if not brotli_body:
        raise STBM1BRError("STBM1BR segment has an empty Brotli body")
    try:
        blob = brotli.decompress(brotli_body)
    except brotli.error as exc:
        raise STBM1BRError("STBM1BR Brotli body failed to decompress") from exc
    qtbm, consumed = _parse_qtbm_blob(blob)
    _require(consumed == len(blob), "QTBM parser did not consume the decoded blob")
    return STBM1BRMetadata(
        segment_bytes=len(segment),
        segment_sha256=sha256_bytes(segment),
        brotli_body_bytes=len(brotli_body),
        brotli_body_sha256=sha256_bytes(brotli_body),
        qtbm_blob_bytes=len(blob),
        qtbm_blob_sha256=sha256_bytes(blob),
        qtbm_magic=qtbm["magic"].decode("ascii", errors="replace"),
        n_pairs=qtbm["n_pairs"],
        height=qtbm["height"],
        width=qtbm["width"],
        precision=qtbm["precision"],
        top_bins=qtbm["top_bins"],
        boundary_xbins=qtbm["boundary_xbins"],
        shift_dy=qtbm["shift_dy"],
        shift_dx=qtbm["shift_dx"],
        residual_order=qtbm["residual_order"],
        top_payload_bytes=len(qtbm["top_payload"]),
        road_payload_bytes=len(qtbm["road_payload"]),
        spatial_table_bytes=len(qtbm["spatial_table"]),
        m5_table_bytes=len(qtbm["m5_table"]),
        sparse_feature_ids=qtbm["sparse_feature_ids"],
        sparse_table_bytes=len(qtbm["sparse_table"]),
        residual_bitstream_bytes=len(qtbm["residual_bitstream"]),
    )


def decode_qtbm_topband_blob(blob: bytes) -> np.ndarray:
    """Decode a decompressed QTBM topband/road-boundary blob to uint8 masks."""

    qtbm, _consumed = _parse_qtbm_blob(blob)
    n_pairs = qtbm["n_pairs"]
    height = qtbm["height"]
    width = qtbm["width"]
    precision = qtbm["precision"]
    inv = list(qtbm["residual_order"] or (0, 1, 2, 3))

    top_support = decode_topband_payload(qtbm["top_payload"], n_pairs, height, width)
    road_mask = decode_boundary_mask_payload(qtbm["road_payload"], n_pairs, height, width)

    spatial_colsfirst = qtbm["magic"] == b"QTBM5\0"
    sparse_colsfirst = qtbm["magic"] in (b"QTBM4\0", b"QTBM5\0")
    sparse_plain = qtbm["magic"] in (b"QTBM3\0", b"QTBM4\0", b"QTBM5\0")
    if spatial_colsfirst:
        spatial_idx, spatial_rows = unpack_sparse_big_plain_colsfirst(qtbm["spatial_table"], precision)
    else:
        spatial_idx, spatial_rows = unpack_sparse_big_plain(qtbm["spatial_table"], precision)
    m5_idx, m5_rows = unpack_sparse_big_plain(qtbm["m5_table"], precision)
    total = 1 << precision
    _require(total > 3, "QTBM precision total is too small")
    default = np.array([1, 1, 1, total - 3], dtype=np.uint16)
    spatial_freqs = np.broadcast_to(default, (5**4, N_SYM)).copy()
    m5_freqs = np.broadcast_to(default, (5**5, N_SYM)).copy()
    _require(spatial_idx.size == 0 or (int(spatial_idx[0]) >= 0 and int(spatial_idx[-1]) < 5**4), "QTBM spatial index out of range")
    _require(m5_idx.size == 0 or (int(m5_idx[0]) >= 0 and int(m5_idx[-1]) < 5**5), "QTBM M5 index out of range")
    spatial_freqs[spatial_idx] = spatial_rows
    m5_freqs[m5_idx] = m5_rows

    spatial_cdf_flat = np.zeros((5**4, N_SYM + 1), dtype=np.int64)
    spatial_cdf_flat[:, 1:] = np.cumsum(spatial_freqs.astype(np.int64), axis=-1)
    m5_cdf_flat = np.zeros((5**5, N_SYM + 1), dtype=np.int64)
    m5_cdf_flat[:, 1:] = np.cumsum(m5_freqs.astype(np.int64), axis=-1)

    if sparse_colsfirst:
        fired_idx, fired_freqs = unpack_sparse_big_plain_colsfirst(qtbm["sparse_table"], precision)
    elif sparse_plain:
        fired_idx, fired_freqs = unpack_sparse_big_plain(qtbm["sparse_table"], precision)
    else:
        fired_idx, fired_freqs = unpack_sparse_big(qtbm["sparse_table"], precision)
    fired_cdf_flat_py: list[list[int]] = []
    if fired_idx.size > 0:
        fired_cdf = np.zeros((fired_idx.size, N_SYM + 1), dtype=np.int64)
        fired_cdf[:, 1:] = np.cumsum(fired_freqs.astype(np.int64), axis=-1)
        fired_cdf_flat_py = fired_cdf.tolist()
    m12_to_slot = {int(ctx): i for i, ctx in enumerate(fired_idx.tolist())}

    dec = _RangeDecoder(qtbm["residual_bitstream"])
    out = np.zeros((n_pairs, height, width), dtype=np.uint8)
    spatial_py = spatial_cdf_flat.tolist()
    m5_py = m5_cdf_flat.tolist()
    previous_frames: list[list[list[int]]] = []
    for fi in range(n_pairs):
        frame_list = _decode_frame_topband(
            dec,
            out[fi],
            previous_frames[-1] if len(previous_frames) >= 1 else None,
            previous_frames[-2] if len(previous_frames) >= 2 else None,
            previous_frames[-3] if len(previous_frames) >= 3 else None,
            top_support[fi],
            road_mask[fi],
            spatial_py,
            m5_py,
            fired_cdf_flat_py,
            m12_to_slot,
            qtbm["sparse_feature_ids"],
            total,
            height,
            width,
            qtbm["shift_dy"],
            qtbm["shift_dx"],
            inv,
        )
        previous_frames.append(frame_list)
        if len(previous_frames) > 3:
            previous_frames.pop(0)
    return out


def decode_stbm1br_mask_segment(
    segment: bytes,
    *,
    expected_shape: tuple[int, int, int] | None = DEFAULT_SHAPE,
) -> np.ndarray:
    """Decode ``STBM1BR\0`` + Brotli(QTBM*) bytes to render-order masks.

    The return value is ``uint8`` with shape ``(pairs, height, width)``.  When
    ``expected_shape`` is not ``None`` the shape is enforced before returning.
    """

    if not segment.startswith(STBM1BR_MAGIC):
        raise STBM1BRError(f"bad STBM1BR magic: {segment[:8]!r}")
    body = segment[len(STBM1BR_MAGIC) :]
    if not body:
        raise STBM1BRError("STBM1BR segment has no Brotli body")
    try:
        blob = brotli.decompress(body)
    except brotli.error as exc:
        raise STBM1BRError("STBM1BR Brotli body failed to decompress") from exc
    decoded = decode_qtbm_topband_blob(blob)
    if expected_shape is not None and tuple(int(v) for v in decoded.shape) != tuple(expected_shape):
        raise STBM1BRError(f"decoded STBM shape {tuple(decoded.shape)} != expected {expected_shape}")
    if decoded.dtype != np.uint8:
        raise STBM1BRError(f"decoded STBM dtype {decoded.dtype} != uint8")
    max_value = int(decoded.max()) if decoded.size else -1
    min_value = int(decoded.min()) if decoded.size else -1
    if min_value < 0 or max_value >= N_CLASSES:
        raise STBM1BRError(f"decoded STBM class range is invalid: min={min_value} max={max_value}")
    return decoded


def decode_stbm1br_mask_file(
    path: Path | str,
    *,
    expected_shape: tuple[int, int, int] | None = DEFAULT_SHAPE,
) -> np.ndarray:
    return decode_stbm1br_mask_segment(Path(path).read_bytes(), expected_shape=expected_shape)


def metadata_as_dict(metadata: STBM1BRMetadata) -> dict[str, Any]:
    payload = metadata.__dict__.copy()
    if metadata.residual_order is not None:
        payload["residual_order"] = list(metadata.residual_order)
    payload["sparse_feature_ids"] = list(metadata.sparse_feature_ids)
    return payload
