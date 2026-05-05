"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``56:19: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``stbm1br_mask_codec.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/stbm1br_mask_codec.py'
__recovery_spec__ = 'stbm1br_mask_codec.recovery_spec.json'
__recovery_ast_error__ = '56:19: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: stbm1br_mask_codec.cpython-312.pyc (Python 3.12)

'''Standalone decoder for PR90 ``STBM1BR`` semantic topband mask segments.

The public PR90 qrepro archive stores its semantic masks as::

    STBM1BR\x00 + brotli(QTBM* topband/road-boundary stream)

This module reimplements the narrow decode surface needed to prove and replay
that mask stream inside PR85-family local candidate pipelines.  It intentionally
does not import from the PR90 source checkout or touch any scorer/runtime model.
'''
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
STBM1BR_MAGIC = b'STBM1BR\x00'
QTBM_MAGICS = (b'QTBM5\x00', b'QTBM4\x00', b'QTBM3\x00', b'QTBM2\x00', b'QTBM1\x00')
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
    '''Raised when an STBM1BR payload violates the self-contained contract.'''
    pass

STBM1BRMetadata = <NODE:12>()

def sha256_bytes(data = None):
    return hashlib.sha256(data).hexdigest()


class _RangeDecoder:
    TOP = 0xFFFFFFFF
    HALF = 0x80000000
    QUARTER = 1073741824
    THREE_QUARTER = 0xC0000000
    
    def __init__(self = None, data = None):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0
        self.low = 0
        self.high = self.TOP
        self.code = 0
        for _ in range(32):
            self.code = (self.code << 1 | self._read_bit()) & self.TOP

    
    def _read_bit(self = None):
        if self.byte_pos >= len(self.data):
            return 0
        byte = self.data[self.byte_pos]
        bit = byte >> 7 - self.bit_pos & 1
        if self.bit_pos == 8:
            0 = self, self.bit_pos += 1, .bit_pos
        return bit

    
    def decode_target(self = None, total = None):
        rng = (self.high - self.low) + 1
        return (((self.code - self.low) + 1) * total - 1) // rng

    
    def advance(self = None, cum_low = None, cum_high = None, total = ('cum_low', 'int', 'cum_high', 'int', 'total', 'int', 'return', 'None')):
        rng = (self.high - self.low) + 1
        self.high = self.low + rng * cum_high // total - 1
        self.low = self.low + rng * cum_low // total
        if self.high < self.HALF:
            pass
        elif self.low >= self.HALF:
            pass
        elif self.low >= self.QUARTER and self.high < self.THREE_QUARTER:
            pass
        else:
            return None
        self.low << 1 & self.TOP = self, self.code -= self.QUARTER, .code
        self.high = (self.high << 1 | 1) & self.TOP
        self.code = (self.code << 1 | self._read_bit()) & self.TOP
        continue



def _require(condition = None, message = None):
    if not condition:
        raise STBM1BRError(message)


def _m5_ctx(top_v, left_v = None, tl_v = None, tr_v = None, prev_v = ('top_v', 'int', 'left_v', 'int', 'tl_v', 'int', 'tr_v', 'int', 'prev_v', 'int', 'return', 'int')):
    return (((top_v * 5 + left_v) * 5 + tl_v) * 5 + tr_v) * 5 + prev_v


def _leb128_decode_big_deltas(buf = None, pos = None, count = None):
    if count < 0:
        raise STBM1BRError(f'''negative LEB128 count: {count}''')
    deltas = np.empty(count, dtype = np.int64)
    for i in range(count):
        result = 0
        shift = 0
        if pos >= len(buf):
            raise STBM1BRError('truncated LEB128 delta stream')
        byte = buf[pos]
        pos += 1
        result |= (byte & 127) << shift
        if not byte & 128:
            pass
        else:
            shift += 7
            if shift > 63:
                raise STBM1BRError('overlong LEB128 delta stream')
        deltas[i] = result
    return (deltas, pos)


def decode_boundary_mask_payload(payload = None, n_pairs = None, height = None, width = ('payload', 'bytes', 'n_pairs', 'int', 'height', 'int', 'width', 'int', 'return', 'np.ndarray')):
    if payload[:5] == b'QBD1\x00':
        pos = 5
        _require(pos + 16 <= len(payload), 'QBD1 boundary payload is truncated')
        (first_len, dx_len, err_len, err_count) = struct.unpack_from('<IIII', payload, pos)
        pos += 16
        _require(pos + first_len <= len(payload), 'QBD1 first-boundary stream overruns payload')
        first = np.frombuffer(bz2.decompress(payload[pos:pos + first_len]), dtype = '<u2', count = n_pairs).astype(np.int16)
        pos += first_len
        _require(pos + dx_len <= len(payload), 'QBD1 delta stream overruns payload')
        dx = np.frombuffer(bz2.decompress(payload[pos:pos + dx_len]), dtype = np.int8, count = n_pairs * (width - 1)).reshape(n_pairs, width - 1)
        pos += dx_len
        _require(pos + err_len <= len(payload), 'QBD1 exception stream overruns payload')
        err_raw = brotli.decompress(payload[pos:pos + err_len])
        pos += err_len
    elif payload[:5] == b'QBD2\x00':
        pos = 5
        _require(pos + 3 + 16 <= len(payload), 'QBD2 boundary payload is truncated')
        (bins, dx_nsym, dx_offset) = struct.unpack_from('<BBB', payload, pos)
        pos += 3
        (first_len, dx_len, err_len, err_count) = struct.unpack_from('<IIII', payload, pos)
        pos += 16
        if bins > 0:
            bins > 0
        _require(dx_nsym > 0, 'QBD2 has empty bin or symbol table')
        _require(pos + first_len <= len(payload), 'QBD2 first-boundary stream overruns payload')
        first = np.frombuffer(bz2.decompress(payload[pos:pos + first_len]), dtype = '<u2', count = n_pairs).astype(np.int16)
        pos += first_len
        freq_bytes = bins * dx_nsym * 2
        _require(pos + freq_bytes <= len(payload), 'QBD2 frequency table overruns payload')
        freqs = np.frombuffer(payload, dtype = '<u2', count = bins * dx_nsym, offset = pos).astype(np.int64).reshape(bins, dx_nsym)
        pos += freq_bytes
        _require(pos + dx_len <= len(payload), 'QBD2 arithmetic delta stream overruns payload')
        bitstream = payload[pos:pos + dx_len]
        pos += dx_len
        _require(pos + err_len <= len(payload), 'QBD2 exception stream overruns payload')
        err_raw = brotli.decompress(payload[pos:pos + err_len])
        pos += err_len
        cdf = np.zeros((bins, dx_nsym + 1), dtype = np.int64)
        cdf[(:, 1:)] = np.cumsum(freqs, axis = 1)
        total = int(cdf[(0, -1)])
        _require(total > 0, 'QBD2 frequency total must be positive')
        dec = _RangeDecoder(bitstream)
        dx = np.empty((n_pairs, width - 1), dtype = np.int16)
        for fi in range(n_pairs):
            for x in range(width - 1):
                row = cdf[x * bins // (width - 1)]
                target = dec.decode_target(total)
                sym = int(np.searchsorted(row, target, side = 'right') - 1)
                if sym < 0 or sym >= dx_nsym:
                    raise STBM1BRError(f'''QBD2 decoded symbol out of range: {sym}''')
                dx[(fi, x)] = sym - int(dx_offset)
                dec.advance(int(row[sym]), int(row[sym + 1]), total)
    else:
        raise STBM1BRError(f'''bad QBD boundary payload magic: {payload[:5]!r}''')
    _require(pos == len(payload), 'QBD boundary payload has trailing bytes')
    _require(first.size == n_pairs, 'QBD first-boundary count mismatch')
    bounds = np.empty((n_pairs, width), dtype = np.int16)
    bounds[(:, 0)] = first
    bounds[(:, 1:)] = first[(:, None)] + np.cumsum(dx.astype(np.int16), axis = 1)
    yy = np.arange(height, dtype = np.int16)[(None, :, None)]
    out = (yy >= bounds[(:, None, :)]).astype(np.uint8)
    if err_count:
        (deltas, used) = _leb128_decode_big_deltas(err_raw, 0, err_count)
        _require(used == len(err_raw), 'trailing QBD exception bytes')
        idx = np.cumsum(deltas, dtype = np.int64) - 1
        if not idx.size == 0:
            idx.size == 0
            if int(idx[0]) >= 0:
                int(idx[0]) >= 0
        _require(int(idx[-1]) < out.size, 'QBD exception index out of range')
        flat = out.reshape(-1)
    return out


def decode_topband_payload(payload = None, n_pairs = None, height = None, width = ('payload', 'bytes', 'n_pairs', 'int', 'height', 'int', 'width', 'int', 'return', 'np.ndarray')):
    if payload[:5] not in (b'QTB1\x00', b'QTB2\x00', b'QTB3\x00', b'QTBZ\x00'):
        raise STBM1BRError(f'''bad QTB top-band payload magic: {payload[:5]!r}''')
    version = payload[:5]
    pos = 5
    err_len = 0
    if version == b'QTBZ\x00':
        _require(pos + struct.calcsize('<HHHI') <= len(payload), 'QTBZ payload is truncated')
        (n2, w2, bins, bounds_len) = struct.unpack_from('<HHHI', payload, pos)
        pos += struct.calcsize('<HHHI')
        _require(pos + bounds_len <= len(payload), 'QTBZ boundary stream overruns payload')
        bounds_raw = bz2.decompress(payload[pos:pos + bounds_len])
        pos += bounds_len
        bounds = np.frombuffer(bounds_raw, dtype = '<u2', count = n_pairs * bins).reshape(n_pairs, bins)
    elif version == b'QTB2\x00':
        _require(pos + struct.calcsize('<HHHI') <= len(payload), 'QTB2 payload is truncated')
        (n2, w2, bins, err_len) = struct.unpack_from('<HHHI', payload, pos)
        pos += struct.calcsize('<HHHI')
        bounds_dtype = '<u2'
    elif version == b'QTB3\x00':
        _require(pos + struct.calcsize('<HHH') <= len(payload), 'QTB3 payload is truncated')
        (n2, w2, bins) = struct.unpack_from('<HHH', payload, pos)
        pos += struct.calcsize('<HHH')
        bounds_dtype = 'u1'
    else:
        _require(pos + struct.calcsize('<HHH') <= len(payload), 'QTB1 payload is truncated')
        (n2, w2, bins) = struct.unpack_from('<HHH', payload, pos)
        pos += struct.calcsize('<HHH')
        bounds_dtype = '<u2'
    if n2 == n_pairs:
        n2 == n_pairs
    _require(w2 == width, 'QTB top-band dimensions do not match stream')
    _require(bins > 0, 'QTB top-band bins must be positive')
# WARNING: Decompyle incomplete


def unpack_sparse_big(compressed = None, precision = None):
    raw = zlib.decompress(compressed, -15)
    return unpack_sparse_big_plain(raw, precision)


def unpack_sparse_big_plain(raw = None, precision = None):
    _require(len(raw) >= 4, 'sparse table is truncated before count')
    (n_fired,) = struct.unpack_from('<I', raw, 0)
    if n_fired == 0:
        _require(len(raw) == 4, 'empty sparse table has trailing bytes')
        return (np.zeros((0,), dtype = np.int64), np.zeros((0, N_SYM), dtype = np.uint16))
    pos = None
    (deltas, pos) = _leb128_decode_big_deltas(raw, pos, n_fired)
    table_bytes = n_fired * (N_SYM - 1) * 2
    _require(pos + table_bytes == len(raw), 'sparse table length mismatch')
    partial = np.frombuffer(raw, dtype = '<u2', count = n_fired * (N_SYM - 1), offset = pos).reshape(n_fired, N_SYM - 1).copy()
    last = (1 << precision) - partial.astype(np.int64).sum(axis = 1)
    if not (last < 0).any():
        not (last < 0).any()
    _require(not (last > 65535).any(), 'invalid sparse frequency row')
    freqs = np.empty((n_fired, N_SYM), dtype = np.uint16)
    freqs[(:, :N_SYM - 1)] = partial
    freqs[(:, N_SYM - 1)] = last.astype(np.uint16)
    fired_idx = (np.cumsum(deltas, dtype = np.int64) - 1).astype(np.int64)
    return (fired_idx, freqs)


def unpack_sparse_big_plain_colsfirst(raw = None, precision = None):
    _require(len(raw) >= 4, 'cols-first sparse table is truncated before count')
    (n_fired,) = struct.unpack_from('<I', raw, 0)
    if n_fired == 0:
        _require(len(raw) == 4, 'empty cols-first sparse table has trailing bytes')
        return (np.zeros((0,), dtype = np.int64), np.zeros((0, N_SYM), dtype = np.uint16))
    pos = None
    table_bytes = n_fired * (N_SYM - 1) * 2
    _require(pos + table_bytes <= len(raw), 'cols-first sparse table overruns payload')
    partial = np.frombuffer(raw, dtype = '<u2', count = n_fired * (N_SYM - 1), offset = pos).reshape(N_SYM - 1, n_fired).T.copy()
    pos += table_bytes
    (deltas, pos) = _leb128_decode_big_deltas(raw, pos, n_fired)
    _require(pos == len(raw), 'cols-first sparse table has trailing bytes')
    last = (1 << precision) - partial.astype(np.int64).sum(axis = 1)
    if not (last < 0).any():
        not (last < 0).any()
    _require(not (last > 65535).any(), 'invalid cols-first sparse frequency row')
    freqs = np.empty((n_fired, N_SYM), dtype = np.uint16)
    freqs[(:, :N_SYM - 1)] = partial
    freqs[(:, N_SYM - 1)] = last.astype(np.uint16)
    fired_idx = (np.cumsum(deltas, dtype = np.int64) - 1).astype(np.int64)
    return (fired_idx, freqs)


def _decode_frame_topband(dec, frame, prev_frame, prev_prev_frame, prev_prev_prev_frame, top_support_frame, road_frame, spatial_cdf_flat_py, m5_cdf_flat_py, fired_cdf_flat_py, m12_to_slot_dict, feat_ids, total, height, width = None, shift_dy = None, shift_dx = None, inv = ('dec', '_RangeDecoder', 'frame', 'np.ndarray', 'prev_frame', 'list[list[int]] | None', 'prev_prev_frame', 'list[list[int]] | None', 'prev_prev_prev_frame', 'list[list[int]] | None', 'top_support_frame', 'np.ndarray', 'road_frame', 'np.ndarray', 'spatial_cdf_flat_py', 'list[list[int]]', 'm5_cdf_flat_py', 'list[list[int]]', 'fired_cdf_flat_py', 'list[list[int]]', 'm12_to_slot_dict', 'dict[int, int]', 'feat_ids', 'tuple[int, ...]', 'total', 'int', 'height', 'int', 'width', 'int', 'shift_dy', 'int', 'shift_dx', 'int', 'inv', 'list[int]', 'return', 'list[list[int]]')):
    top_support = np.asarray(top_support_frame, dtype = bool)
    road = np.asarray(road_frame, dtype = bool)
    active = ~(top_support | road)
    base = np.zeros((height, width), dtype = np.uint8)
    base[road] = 4
    base[top_support] = 2
    frame_list = base.tolist()
# WARNING: Decompyle incomplete


def _parse_qtbm_blob(blob = None):
    pos = 0
    magic = None
    for candidate in QTBM_MAGICS:
        if not blob.startswith(candidate):
            continue
        magic = candidate
        pos = len(candidate)
        QTBM_MAGICS
# WARNING: Decompyle incomplete


def parse_stbm1br_metadata(segment = None):
    '''Parse an STBM1BR segment and return fail-closed byte metadata.'''
    if not segment.startswith(STBM1BR_MAGIC):
        raise STBM1BRError(f'''bad STBM1BR magic: {segment[:8]!r}''')
    brotli_body = segment[len(STBM1BR_MAGIC):]
    if not brotli_body:
        raise STBM1BRError('STBM1BR segment has an empty Brotli body')
# WARNING: Decompyle incomplete


def decode_qtbm_topband_blob(blob = None):
    '''Decode a decompressed QTBM topband/road-boundary blob to uint8 masks.'''
    (qtbm, _consumed) = _parse_qtbm_blob(blob)
    n_pairs = qtbm['n_pairs']
    height = qtbm['height']
    width = qtbm['width']
    precision = qtbm['precision']
    if not qtbm['residual_order']:
        qtbm['residual_order']
    inv = list((0, 1, 2, 3))
    top_support = decode_topband_payload(qtbm['top_payload'], n_pairs, height, width)
    road_mask = decode_boundary_mask_payload(qtbm['road_payload'], n_pairs, height, width)
    spatial_colsfirst = qtbm['magic'] == b'QTBM5\x00'
    sparse_colsfirst = qtbm['magic'] in (b'QTBM4\x00', b'QTBM5\x00')
    sparse_plain = qtbm['magic'] in (b'QTBM3\x00', b'QTBM4\x00', b'QTBM5\x00')
    if spatial_colsfirst:
        (spatial_idx, spatial_rows) = unpack_sparse_big_plain_colsfirst(qtbm['spatial_table'], precision)
    else:
        (spatial_idx, spatial_rows) = unpack_sparse_big_plain(qtbm['spatial_table'], precision)
    (m5_idx, m5_rows) = unpack_sparse_big_plain(qtbm['m5_table'], precision)
    total = 1 << precision
    _require(total > 3, 'QTBM precision total is too small')
    default = np.array([
        1,
        1,
        1,
        total - 3], dtype = np.uint16)
    spatial_freqs = np.broadcast_to(default, (625, N_SYM)).copy()
    m5_freqs = np.broadcast_to(default, (3125, N_SYM)).copy()
    if not spatial_idx.size == 0:
        spatial_idx.size == 0
        if int(spatial_idx[0]) >= 0:
            int(spatial_idx[0]) >= 0
    _require(int(spatial_idx[-1]) < 625, 'QTBM spatial index out of range')
    if not m5_idx.size == 0:
        m5_idx.size == 0
        if int(m5_idx[0]) >= 0:
            int(m5_idx[0]) >= 0
    _require(int(m5_idx[-1]) < 3125, 'QTBM M5 index out of range')
    spatial_freqs[spatial_idx] = spatial_rows
    m5_freqs[m5_idx] = m5_rows
    spatial_cdf_flat = np.zeros((625, N_SYM + 1), dtype = np.int64)
    spatial_cdf_flat[(:, 1:)] = np.cumsum(spatial_freqs.astype(np.int64), axis = -1)
    m5_cdf_flat = np.zeros((3125, N_SYM + 1), dtype = np.int64)
    m5_cdf_flat[(:, 1:)] = np.cumsum(m5_freqs.astype(np.int64), axis = -1)
    if sparse_colsfirst:
        (fired_idx, fired_freqs) = unpack_sparse_big_plain_colsfirst(qtbm['sparse_table'], precision)
    elif sparse_plain:
        (fired_idx, fired_freqs) = unpack_sparse_big_plain(qtbm['sparse_table'], precision)
    else:
        (fired_idx, fired_freqs) = unpack_sparse_big(qtbm['sparse_table'], precision)
    fired_cdf_flat_py = []
    if fired_idx.size > 0:
        fired_cdf = np.zeros((fired_idx.size, N_SYM + 1), dtype = np.int64)
        fired_cdf[(:, 1:)] = np.cumsum(fired_freqs.astype(np.int64), axis = -1)
        fired_cdf_flat_py = fired_cdf.tolist()
# WARNING: Decompyle incomplete


def decode_stbm1br_mask_segment(segment = None, *, expected_shape):
    '''Decode ``STBM1BR\x00`` + Brotli(QTBM*) bytes to render-order masks.

    The return value is ``uint8`` with shape ``(pairs, height, width)``.  When
    ``expected_shape`` is not ``None`` the shape is enforced before returning.
    '''
    if not segment.startswith(STBM1BR_MAGIC):
        raise STBM1BRError(f'''bad STBM1BR magic: {segment[:8]!r}''')
    body = segment[len(STBM1BR_MAGIC):]
    if not body:
        raise STBM1BRError('STBM1BR segment has no Brotli body')
# WARNING: Decompyle incomplete


def decode_stbm1br_mask_file(path = None, *, expected_shape):
    return decode_stbm1br_mask_segment(Path(path).read_bytes(), expected_shape = expected_shape)


def metadata_as_dict(metadata = None):
    payload = metadata.__dict__.copy()
# WARNING: Decompyle incomplete


"""
