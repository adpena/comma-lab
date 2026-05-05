"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``58:19: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``seg_sparse_m5_codec.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/results/public_pr90_intake_20260504_worker/pr90_src/submissions/qrepro/seg_sparse_m5_codec.py'
__recovery_spec__ = 'seg_sparse_m5_codec.recovery_spec.json'
__recovery_ast_error__ = '58:19: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: seg_sparse_m5_codec.cpython-312.pyc (Python 3.12)

from __future__ import annotations
import struct
import bz2
from pathlib import Path
import brotli
import numpy as np
from encode_seg_c2split_purepy import N_CLASSES, compute_spatial_contexts, decode_mask_payload, make_remap_tables
from encode_seg_c2split_tr_purepy import compute_tr
from range_coder import RangeDecoder
from seg_sparse_m10_codec import FEAT_DIAG_TLTL, FEAT_DIAG_TRTR, FEAT_LEFT_LEFT, FEAT_PREV2_LEFT, FEAT_PREV2_RIGHT, FEAT_PREV_BOTTOM, FEAT_PREV_BOTTOM2, FEAT_PREV_BOTTOM_LEFT, FEAT_PREV_BOTTOM_RIGHT, FEAT_PREV_LEFT, FEAT_PREV_PREV_PREV, FEAT_PREV_RIGHT, FEAT_PREV_RIGHT2, FEAT_PREV_TOP_RIGHT, FEAT_PREV_TOP, FEAT_PEEL_DIST42, FEAT_PEEL_BOUND5, FEAT_PEEL_SLOPE5, FEAT_TOP_TOP_TOP, FEAT_X_BIN5, FEAT_X_BIN5_SHIFT, FEAT_Y_BIN5, M10_VERSION, N_SYM, unpack_sparse_m10
MAGIC = b'QSM5\x00'
MAGIC_SHIFT = b'QSM5S\x00'
MAGIC_SHIFT_BIG = b'QSM5S7\x00'
MAGIC_SHIFT_BIG3 = b'QSM5S8\x00'
MAGIC_SHIFT_BIG4 = b'QSM5S9\x00'
MAGIC_SHIFT_BIG5 = b'QSM5SA\x00'
MAGIC_TOPBAND = b'QTBM1\x00'
MAGIC_TOPBAND2 = b'QTBM2\x00'
MAGIC_TOPBAND3 = b'QTBM3\x00'
MAGIC_TOPBAND4 = b'QTBM4\x00'
MAGIC_TOPBAND5 = b'QTBM5\x00'
BINARY_MASK_FORMAT = 255
BOUNDARY_MASK_FORMAT = 254
BINARY_FEATURES = {
    0: 'top',
    1: 'left',
    2: 'tl',
    3: 'tr',
    4: 'tt',
    5: 'll',
    6: 'prev',
    7: 'prev2',
    8: 'prev_bottom',
    9: 'prev_right',
    10: 'prev_left',
    11: 'prev_top',
    12: 'prev_bottom_right',
    13: 'prev_right2',
    14: 'prev_bottom2' }

def _m5_ctx(top_v, left_v = None, tl_v = None, tr_v = None, prev_v = ('top_v', 'int', 'left_v', 'int', 'tl_v', 'int', 'tr_v', 'int', 'prev_v', 'int', 'return', 'int')):
    return (((top_v * 5 + left_v) * 5 + tl_v) * 5 + tr_v) * 5 + prev_v


def _binary_ctx_frame(mask = None, fi = None, feat_ids = None):
    frame = mask[fi]
    (h, w) = frame.shape
    out = np.zeros((h, w), dtype = np.int64)
    for fid in feat_ids:
        name = BINARY_FEATURES[fid]
        feat = np.zeros((h, w), dtype = np.int64)
        if name == 'top':
            feat[1:] = frame[:-1]
        elif name == 'left':
            feat[(:, 1:)] = frame[(:, :-1)]
        elif name == 'tl':
            feat[(1:, 1:)] = frame[(:-1, :-1)]
        elif name == 'tr':
            feat[(1:, :-1)] = frame[(:-1, 1:)]
        elif name == 'tt':
            feat[2:] = frame[:-2]
        elif name == 'll':
            feat[(:, 2:)] = frame[(:, :-2)]
        elif name == 'prev' or fi >= 1:
            feat = mask[fi - 1].astype(np.int64)
        elif name == 'prev2' or fi >= 2:
            feat = mask[fi - 2].astype(np.int64)
        elif name == 'prev_bottom' or fi >= 1:
            feat[:-1] = mask[(fi - 1, 1:)]
        elif name == 'prev_right' or fi >= 1:
            feat[(:, :-1)] = mask[(fi - 1, :, 1:)]
        elif name == 'prev_left' or fi >= 1:
            feat[(:, 1:)] = mask[(fi - 1, :, :-1)]
        elif name == 'prev_top' or fi >= 1:
            feat[1:] = mask[(fi - 1, :-1)]
        else:
            raise ValueError(name)
        out = out * 2 + feat
    return out


def decode_binary_mask_payload(payload = None, n_pairs = None, h = None, w = ('payload', 'bytes', 'n_pairs', 'int', 'h', 'int', 'w', 'int', 'return', 'np.ndarray')):
    if payload[:5] != b'QBM1\x00':
        raise ValueError('bad QBM1 mask payload')
    pos = 5
    (precision, n0, n) = struct.unpack_from('<BBB', payload, pos)
    pos += 3
    feat0 = tuple(payload[pos:pos + n0])
    pos += n0
    feat = tuple(payload[pos:pos + n])
    pos += n
    (freq0_size, freq_size, bs_len) = struct.unpack_from('<HHI', payload, pos)
    pos += struct.calcsize('<HHI')
    freq0 = np.frombuffer(payload, dtype = '<u2', count = freq0_size // 2, offset = pos).reshape(2 ** n0, 2).copy()
    pos += freq0_size
    freq = np.frombuffer(payload, dtype = '<u2', count = freq_size // 2, offset = pos).reshape(2 ** n, 2).copy()
    pos += freq_size
    bitstream = payload[pos:pos + bs_len]
    cdf0 = np.zeros((2 ** n0, 3), dtype = np.int64)
    cdf0[(:, 1:)] = np.cumsum(freq0.astype(np.int64), axis = 1)
    cdf = np.zeros((2 ** n, 3), dtype = np.int64)
    cdf[(:, 1:)] = np.cumsum(freq.astype(np.int64), axis = 1)
    dec = RangeDecoder(bitstream)
    total = 1 << precision
    out = np.zeros((n_pairs, h, w), dtype = np.uint8)
    for fi in range(n_pairs):
        feats = feat0 if fi == 0 else feat
        rows = cdf0 if fi == 0 else cdf
        for y in range(h):
            for x in range(w):
                ctx = 0
                for fid in feats:
                    name = BINARY_FEATURES[fid]
                    if name == 'top':
                        fv = int(out[(fi, y - 1, x)]) if y > 0 else 0
                    elif name == 'left':
                        fv = int(out[(fi, y, x - 1)]) if x > 0 else 0
                    elif name == 'tl':
                        fv = int(out[(fi, y - 1, x - 1)]) if y > 0 and x > 0 else 0
                    elif name == 'tr':
                        fv = int(out[(fi, y - 1, x + 1)]) if y > 0 and x + 1 < w else 0
                    elif name == 'tt':
                        fv = int(out[(fi, y - 2, x)]) if y > 1 else 0
                    elif name == 'll':
                        fv = int(out[(fi, y, x - 2)]) if x > 1 else 0
                    elif name == 'prev':
                        fv = int(out[(fi - 1, y, x)]) if fi >= 1 else 0
                    elif name == 'prev2':
                        fv = int(out[(fi - 2, y, x)]) if fi >= 2 else 0
                    elif name == 'prev_bottom':
                        fv = int(out[(fi - 1, y + 1, x)]) if fi >= 1 and y + 1 < h else 0
                    elif name == 'prev_right':
                        fv = int(out[(fi - 1, y, x + 1)]) if fi >= 1 and x + 1 < w else 0
                    elif name == 'prev_left':
                        fv = int(out[(fi - 1, y, x - 1)]) if fi >= 1 and x >= 1 else 0
                    elif name == 'prev_top':
                        fv = int(out[(fi - 1, y - 1, x)]) if fi >= 1 and y >= 1 else 0
                    elif name == 'prev_bottom_right':
                        fv = int(out[(fi - 1, y + 1, x + 1)]) if fi >= 1 and y + 1 < h and x + 1 < w else 0
                    elif name == 'prev_right2':
                        fv = int(out[(fi - 1, y, x + 2)]) if fi >= 1 and x + 2 < w else 0
                    elif name == 'prev_bottom2':
                        fv = int(out[(fi - 1, y + 2, x)]) if fi >= 1 and y + 2 < h else 0
                    else:
                        raise ValueError(name)
                    ctx = ctx * 2 + fv
                row = rows[ctx]
                target = dec.decode_target(total)
                sym = 0 if row[1] > target else 1
                out[(fi, y, x)] = sym
                dec.advance(int(row[sym]), int(row[sym + 1]), total)
        if not fi % 50 == 0:
            continue
        if not fi:
            continue
        print(f'''  decoded binary mask frame {fi}/{n_pairs}''', flush = True)
    return out


def _leb128_decode_big_deltas(buf = None, pos = None, count = None):
    deltas = np.empty(count, dtype = np.int64)
    for i in range(count):
        result = 0
        shift = 0
        byte = buf[pos]
        pos += 1
        result |= (byte & 127) << shift
        if not byte & 128:
            pass
        else:
            shift += 7
        deltas[i] = result
    return (deltas, pos)


def decode_boundary_mask_payload(payload = None, n_pairs = None, h = None, w = ('payload', 'bytes', 'n_pairs', 'int', 'h', 'int', 'w', 'int', 'return', 'np.ndarray')):
    if payload[:5] == b'QBD1\x00':
        pos = 5
        (first_len, dx_len, err_len, err_count) = struct.unpack_from('<IIII', payload, pos)
        pos += 16
        first = np.frombuffer(bz2.decompress(payload[pos:pos + first_len]), dtype = '<u2', count = n_pairs).astype(np.int16)
        pos += first_len
        dx = np.frombuffer(bz2.decompress(payload[pos:pos + dx_len]), dtype = np.int8, count = n_pairs * (w - 1)).reshape(n_pairs, w - 1)
        pos += dx_len
        err_raw = brotli.decompress(payload[pos:pos + err_len])
    elif payload[:5] == b'QBD2\x00':
        pos = 5
        (bins, dx_nsym, dx_offset) = struct.unpack_from('<BBB', payload, pos)
        pos += 3
        (first_len, dx_len, err_len, err_count) = struct.unpack_from('<IIII', payload, pos)
        pos += 16
        first = np.frombuffer(bz2.decompress(payload[pos:pos + first_len]), dtype = '<u2', count = n_pairs).astype(np.int16)
        pos += first_len
        freqs = np.frombuffer(payload, dtype = '<u2', count = bins * dx_nsym, offset = pos).astype(np.int64).reshape(bins, dx_nsym)
        pos += bins * dx_nsym * 2
        bitstream = payload[pos:pos + dx_len]
        pos += dx_len
        err_raw = brotli.decompress(payload[pos:pos + err_len])
        cdf = np.zeros((bins, dx_nsym + 1), dtype = np.int64)
        cdf[(:, 1:)] = np.cumsum(freqs, axis = 1)
        total = int(cdf[(0, -1)])
        dec = RangeDecoder(bitstream)
        dx = np.empty((n_pairs, w - 1), dtype = np.int16)
        for fi in range(n_pairs):
            for x in range(w - 1):
                row = cdf[x * bins // (w - 1)]
                target = dec.decode_target(total)
                sym = int(np.searchsorted(row, target, side = 'right') - 1)
                dx[(fi, x)] = sym - int(dx_offset)
                dec.advance(int(row[sym]), int(row[sym + 1]), total)
    else:
        raise ValueError('bad QBD mask payload')
    bounds = np.empty((n_pairs, w), dtype = np.int16)
    bounds[(:, 0)] = first
    bounds[(:, 1:)] = first[(:, None)] + np.cumsum(dx.astype(np.int16), axis = 1)
    yy = np.arange(h, dtype = np.int16)[(None, :, None)]
    out = (yy >= bounds[(:, None, :)]).astype(np.uint8)
    if err_count:
        (deltas, used) = _leb128_decode_big_deltas(err_raw, 0, err_count)
        if used != len(err_raw):
            raise ValueError('trailing QBD1 error bytes')
        idx = np.cumsum(deltas, dtype = np.int64) - 1
        flat = out.reshape(-1)
    return out


def decode_topband_payload(payload = None, n_pairs = None, h = None, w = ('payload', 'bytes', 'n_pairs', 'int', 'h', 'int', 'w', 'int', 'return', 'np.ndarray')):
    if payload[:5] not in (b'QTB1\x00', b'QTB2\x00', b'QTB3\x00', b'QTBZ\x00'):
        raise ValueError('bad QTB1 top-band payload')
    version = payload[:5]
    pos = 5
    if version == b'QTBZ\x00':
        (n2, w2, bins, bounds_len) = struct.unpack_from('<HHHI', payload, pos)
        pos += struct.calcsize('<HHHI')
        bounds_raw = bz2.decompress(payload[pos:pos + bounds_len])
        pos += bounds_len
        bounds = np.frombuffer(bounds_raw, dtype = '<u2', count = n_pairs * bins).reshape(n_pairs, bins)
        err_len = 0
    elif version == b'QTB2\x00':
        (n2, w2, bins, err_len) = struct.unpack_from('<HHHI', payload, pos)
        pos += struct.calcsize('<HHHI')
        bounds_dtype = '<u2'
    elif version == b'QTB3\x00':
        (n2, w2, bins) = struct.unpack_from('<HHH', payload, pos)
        err_len = 0
        bounds_dtype = 'u1'
        pos += struct.calcsize('<HHH')
    else:
        (n2, w2, bins) = struct.unpack_from('<HHH', payload, pos)
        err_len = 0
        bounds_dtype = '<u2'
        pos += struct.calcsize('<HHH')
    if n2 != n_pairs or w2 != w:
        raise ValueError('top-band dimensions do not match stream')
# WARNING: Decompyle incomplete


def unpack_sparse_big(compressed = None, precision = None):
    import zlib
    raw = zlib.decompress(compressed, -15)
    (n_fired,) = struct.unpack_from('<I', raw, 0)
    if n_fired == 0:
        return (np.zeros((0,), dtype = np.int64), np.zeros((0, N_SYM), dtype = np.uint16))
    pos = None
    (deltas, pos) = _leb128_decode_big_deltas(raw, pos, n_fired)
    partial = np.frombuffer(raw, dtype = '<u2', count = n_fired * (N_SYM - 1), offset = pos).reshape(n_fired, N_SYM - 1).copy()
    last = (1 << precision) - partial.astype(np.int64).sum(axis = 1)
    if (last < 0).any() or (last > 65535).any():
        raise ValueError('invalid sparse-big frequency row')
    freqs = np.empty((n_fired, N_SYM), dtype = np.uint16)
    freqs[(:, :N_SYM - 1)] = partial
    freqs[(:, N_SYM - 1)] = last.astype(np.uint16)
    fired_idx = (np.cumsum(deltas, dtype = np.int64) - 1).astype(np.int64)
    return (fired_idx, freqs)


def unpack_sparse_big_plain(raw = None, precision = None):
    (n_fired,) = struct.unpack_from('<I', raw, 0)
    if n_fired == 0:
        return (np.zeros((0,), dtype = np.int64), np.zeros((0, N_SYM), dtype = np.uint16))
    pos = None
    (deltas, pos) = _leb128_decode_big_deltas(raw, pos, n_fired)
    partial = np.frombuffer(raw, dtype = '<u2', count = n_fired * (N_SYM - 1), offset = pos).reshape(n_fired, N_SYM - 1).copy()
    last = (1 << precision) - partial.astype(np.int64).sum(axis = 1)
    if (last < 0).any() or (last > 65535).any():
        raise ValueError('invalid sparse-big-plain frequency row')
    freqs = np.empty((n_fired, N_SYM), dtype = np.uint16)
    freqs[(:, :N_SYM - 1)] = partial
    freqs[(:, N_SYM - 1)] = last.astype(np.uint16)
    fired_idx = (np.cumsum(deltas, dtype = np.int64) - 1).astype(np.int64)
    return (fired_idx, freqs)


def unpack_sparse_big_plain_colsfirst(raw = None, precision = None):
    (n_fired,) = struct.unpack_from('<I', raw, 0)
    if n_fired == 0:
        return (np.zeros((0,), dtype = np.int64), np.zeros((0, N_SYM), dtype = np.uint16))
    pos = None
    partial = np.frombuffer(raw, dtype = '<u2', count = n_fired * (N_SYM - 1), offset = pos).reshape(N_SYM - 1, n_fired).T.copy()
    pos += n_fired * (N_SYM - 1) * 2
    (deltas, pos) = _leb128_decode_big_deltas(raw, pos, n_fired)
    if pos != len(raw):
        raise ValueError('trailing sparse-big-plain-colsfirst bytes')
    last = (1 << precision) - partial.astype(np.int64).sum(axis = 1)
    if (last < 0).any() or (last > 65535).any():
        raise ValueError('invalid sparse-big-plain-colsfirst frequency row')
    freqs = np.empty((n_fired, N_SYM), dtype = np.uint16)
    freqs[(:, :N_SYM - 1)] = partial
    freqs[(:, N_SYM - 1)] = last.astype(np.uint16)
    fired_idx = (np.cumsum(deltas, dtype = np.int64) - 1).astype(np.int64)
    return (fired_idx, freqs)


def _decode_frame_m5fallback(dec, frame, prev_frame, prev_prev_frame, prev_prev_prev_frame, mask_frame, m5_cdf_flat_py, fired_cdf_flat_py, m12_to_slot_dict, feat_ids, total, h, w = None, peel_class = None, inv_remap_py = None, shift_dy = (0, 0), shift_dx = ('dec', 'RangeDecoder', 'frame', 'np.ndarray', 'prev_frame', 'np.ndarray', 'prev_prev_frame', 'np.ndarray', 'prev_prev_prev_frame', 'np.ndarray | None', 'mask_frame', 'np.ndarray', 'm5_cdf_flat_py', 'list', 'fired_cdf_flat_py', 'list', 'm12_to_slot_dict', 'dict[int, int]', 'feat_ids', 'tuple[int, ...]', 'total', 'int', 'h', 'int', 'w', 'int', 'peel_class', 'int', 'inv_remap_py', 'list', 'shift_dy', 'int', 'shift_dx', 'int', 'return', 'None')):
    pass
# WARNING: Decompyle incomplete


def _decode_frame_topband(dec, frame, prev_frame, prev_prev_frame, prev_prev_prev_frame, top_support_frame, road_frame, spatial_cdf_flat_py, m5_cdf_flat_py, fired_cdf_flat_py, m12_to_slot_dict, feat_ids, total, h, w = None, shift_dy = None, shift_dx = None, inv = ('dec', 'RangeDecoder', 'frame', 'np.ndarray', 'prev_frame', 'np.ndarray | None', 'prev_prev_frame', 'np.ndarray | None', 'prev_prev_prev_frame', 'np.ndarray | None', 'top_support_frame', 'np.ndarray', 'road_frame', 'np.ndarray', 'spatial_cdf_flat_py', 'list', 'm5_cdf_flat_py', 'list', 'fired_cdf_flat_py', 'list', 'm12_to_slot_dict', 'dict[int, int]', 'feat_ids', 'tuple[int, ...]', 'total', 'int', 'h', 'int', 'w', 'int', 'shift_dy', 'int', 'shift_dx', 'int', 'inv', 'list[int]', 'return', 'None')):
    pass
# WARNING: Decompyle incomplete


def _decode_frame_m5_shift(dec, frame, prev_frame, mask_frame, m5_cdf_flat_py, total, h, w, peel_class = None, inv_remap_py = None, shift_dy = None, shift_dx = ('dec', 'RangeDecoder', 'frame', 'np.ndarray', 'prev_frame', 'np.ndarray', 'mask_frame', 'np.ndarray', 'm5_cdf_flat_py', 'list', 'total', 'int', 'h', 'int', 'w', 'int', 'peel_class', 'int', 'inv_remap_py', 'list', 'shift_dy', 'int', 'shift_dx', 'int', 'return', 'None')):
    pass
# WARNING: Decompyle incomplete


def decode_seg_split_m5(path = None):
    blob = Path(path).read_bytes()
    pos = 0
    shifted = False
    big_sparse = False
    compact_tables = False
    sparse_dense_tables = False
    sparse_dense_plain = False
    shift_dy = 0
    shift_dx = 0
    if blob[:len(MAGIC_SHIFT_BIG5)] == MAGIC_SHIFT_BIG5:
        shifted = True
        big_sparse = True
        sparse_dense_tables = True
        sparse_dense_plain = True
        pos += len(MAGIC_SHIFT_BIG5)
    elif blob[:len(MAGIC_SHIFT_BIG4)] == MAGIC_SHIFT_BIG4:
        shifted = True
        big_sparse = True
        sparse_dense_tables = True
        sparse_dense_plain = False
        pos += len(MAGIC_SHIFT_BIG4)
    elif blob[:len(MAGIC_SHIFT_BIG3)] == MAGIC_SHIFT_BIG3:
        shifted = True
        big_sparse = True
        compact_tables = True
        sparse_dense_plain = False
        pos += len(MAGIC_SHIFT_BIG3)
    elif blob[:len(MAGIC_SHIFT_BIG)] == MAGIC_SHIFT_BIG:
        shifted = True
        big_sparse = True
        pos += len(MAGIC_SHIFT_BIG)
    elif blob[:len(MAGIC_SHIFT)] == MAGIC_SHIFT:
        shifted = True
        pos += len(MAGIC_SHIFT)
    elif blob[:len(MAGIC)] == MAGIC:
        pos += len(MAGIC)
    else:
        raise ValueError('bad QSM5 magic')
    if shifted:
        (n_pairs, h, w, precision, peel_class, mask_format, shift_dy, shift_dx) = struct.unpack_from('<HHHBBBbb', blob, pos)
        pos += struct.calcsize('<HHHBBBbb')
    else:
        (n_pairs, h, w, precision, peel_class, mask_format) = struct.unpack_from('<HHHBBB', blob, pos)
        pos += struct.calcsize('<HHHBBB')
    (mask_len,) = struct.unpack_from('<I', blob, pos)
    pos += 4
    mask_payload = blob[pos:pos + mask_len]
    pos += mask_len
    (spatial_size, m5_size) = struct.unpack_from('<HH', blob, pos)
    pos += struct.calcsize('<HH')
    if sparse_dense_tables:
        unpack_dense = unpack_sparse_big_plain if sparse_dense_plain else unpack_sparse_big
        (spatial_idx, spatial_rows) = unpack_dense(blob[pos:pos + spatial_size], precision = precision)
        pos += spatial_size
        (m5_idx, m5_rows) = unpack_dense(blob[pos:pos + m5_size], precision = precision)
        pos += m5_size
        total = 1 << precision
        default = np.array([
            1,
            1,
            1,
            total - 3], dtype = np.uint16)
        spatial_freqs = np.broadcast_to(default, (625, N_SYM)).copy()
        m5_freqs = np.broadcast_to(default, (3125, N_SYM)).copy()
        spatial_freqs[spatial_idx] = spatial_rows
        m5_freqs[m5_idx] = m5_rows
        spatial_freqs = spatial_freqs.reshape((N_CLASSES,) * 4 + (N_SYM,))
        m5_freqs = m5_freqs.reshape((N_CLASSES,) * 5 + (N_SYM,))
    elif compact_tables:
        spatial_part = np.frombuffer(blob, dtype = '<u2', count = spatial_size // 2, offset = pos).reshape(625, N_SYM - 1).copy()
        pos += spatial_size
        m5_part = np.frombuffer(blob, dtype = '<u2', count = m5_size // 2, offset = pos).reshape(3125, N_SYM - 1).copy()
        pos += m5_size
        total = 1 << precision
        spatial_last = total - spatial_part.astype(np.int64).sum(axis = 1)
        m5_last = total - m5_part.astype(np.int64).sum(axis = 1)
        if (spatial_last < 0).any() or (m5_last < 0).any():
            raise ValueError('invalid compact frequency table')
        spatial_freqs = np.empty((625, N_SYM), dtype = np.uint16)
        spatial_freqs[(:, :N_SYM - 1)] = spatial_part
        spatial_freqs[(:, N_SYM - 1)] = spatial_last.astype(np.uint16)
        spatial_freqs = spatial_freqs.reshape((N_CLASSES,) * 4 + (N_SYM,))
        m5_freqs = np.empty((3125, N_SYM), dtype = np.uint16)
        m5_freqs[(:, :N_SYM - 1)] = m5_part
        m5_freqs[(:, N_SYM - 1)] = m5_last.astype(np.uint16)
        m5_freqs = m5_freqs.reshape((N_CLASSES,) * 5 + (N_SYM,))
    else:
        spatial_freqs = np.frombuffer(blob, dtype = '<u2', count = spatial_size // 2, offset = pos).reshape((N_CLASSES,) * 4 + (N_SYM,)).copy()
        pos += spatial_size
        m5_freqs = np.frombuffer(blob, dtype = '<u2', count = m5_size // 2, offset = pos).reshape((N_CLASSES,) * 5 + (N_SYM,)).copy()
        pos += m5_size
    n_feats = blob[pos]
    pos += 1
    feat_ids = tuple(blob[pos:pos + n_feats])
    pos += n_feats
    (_thr_q8, sparse_len) = struct.unpack_from('<HI', blob, pos)
    pos += 6
    sparse = blob[pos:pos + sparse_len]
    pos += sparse_len
    (bs_len,) = struct.unpack_from('<I', blob, pos)
    pos += 4
    bitstream = blob[pos:pos + bs_len]
    if mask_format == BINARY_MASK_FORMAT:
        mask = decode_binary_mask_payload(mask_payload, n_pairs, h, w)
    elif mask_format == BOUNDARY_MASK_FORMAT:
        mask = decode_boundary_mask_payload(mask_payload, n_pairs, h, w)
    else:
        mask = decode_mask_payload(mask_payload, mask_format, n_pairs, h, w)
    (_, inverse) = make_remap_tables(peel_class)
    inv_remap_py = inverse.tolist()
    total = 1 << precision
    spatial_cdf = np.zeros((N_CLASSES,) * 4 + (N_SYM + 1,), dtype = np.int64)
    spatial_cdf[(..., 1:)] = np.cumsum(spatial_freqs.astype(np.int64), axis = -1)
    spatial_py = spatial_cdf.tolist()
    m5_cdf_flat = np.zeros((3125, N_SYM + 1), dtype = np.int64)
    m5_cdf_flat[(:, 1:)] = np.cumsum(m5_freqs.reshape(3125, N_SYM).astype(np.int64), axis = -1)
    m5_py = m5_cdf_flat.reshape((N_CLASSES,) * 5 + (N_SYM + 1,)).tolist()
    m5_flat_py = m5_cdf_flat.tolist()
    if big_sparse:
        (fired_idx, fired_freqs) = unpack_sparse_big(sparse, precision = precision)
    else:
        (fired_idx, fired_freqs, _n_ctx) = unpack_sparse_m10(sparse, version = M10_VERSION, precision = precision)
    fired_cdf_flat_py = []
    if fired_idx.size > 0:
        fired_cdf = np.zeros((fired_idx.size, N_SYM + 1), dtype = np.int64)
        fired_cdf[(:, 1:)] = np.cumsum(fired_freqs.astype(np.int64), axis = -1)
        fired_cdf_flat_py = fired_cdf.tolist()
# WARNING: Decompyle incomplete


def decode_seg_topband(path = None):
    blob = Path(path).read_bytes()
    pos = 0
    sparse_colsfirst = False
    spatial_colsfirst = False
    if blob[:len(MAGIC_TOPBAND5)] == MAGIC_TOPBAND5:
        has_residual_order = True
        sparse_plain = True
        sparse_colsfirst = True
        spatial_colsfirst = True
        pos += len(MAGIC_TOPBAND5)
    elif blob[:len(MAGIC_TOPBAND4)] == MAGIC_TOPBAND4:
        has_residual_order = True
        sparse_plain = True
        sparse_colsfirst = True
        pos += len(MAGIC_TOPBAND4)
    elif blob[:len(MAGIC_TOPBAND3)] == MAGIC_TOPBAND3:
        has_residual_order = True
        sparse_plain = True
        pos += len(MAGIC_TOPBAND3)
    elif blob[:len(MAGIC_TOPBAND2)] == MAGIC_TOPBAND2:
        has_residual_order = True
        sparse_plain = False
        pos += len(MAGIC_TOPBAND2)
    elif blob[:len(MAGIC_TOPBAND)] == MAGIC_TOPBAND:
        has_residual_order = False
        sparse_plain = False
        pos += len(MAGIC_TOPBAND)
    else:
        raise ValueError('bad QTBM1 magic')
    (n_pairs, h, w, precision, _top_bins, _boundary_xbins, shift_dy, shift_dx) = struct.unpack_from('<HHHBBBbb', blob, pos)
    pos += struct.calcsize('<HHHBBBbb')
    if has_residual_order:
        inv = list(blob[pos:pos + N_SYM])
        pos += N_SYM
        if sorted(inv) != [
            0,
            1,
            2,
            3]:
            raise ValueError('invalid QTBM2 residual order')
    inv = [
        0,
        1,
        2,
        3]
    (top_len, road_len) = struct.unpack_from('<II', blob, pos)
    pos += 8
    top_payload = blob[pos:pos + top_len]
    pos += top_len
    road_payload = blob[pos:pos + road_len]
    pos += road_len
    (spatial_size, m5_size) = struct.unpack_from('<HH', blob, pos)
    pos += struct.calcsize('<HH')
    if spatial_colsfirst:
        (spatial_idx, spatial_rows) = unpack_sparse_big_plain_colsfirst(blob[pos:pos + spatial_size], precision = precision)
    else:
        (spatial_idx, spatial_rows) = unpack_sparse_big_plain(blob[pos:pos + spatial_size], precision = precision)
    pos += spatial_size
    (m5_idx, m5_rows) = unpack_sparse_big_plain(blob[pos:pos + m5_size], precision = precision)
    pos += m5_size
    total = 1 << precision
    default = np.array([
        1,
        1,
        1,
        total - 3], dtype = np.uint16)
    spatial_freqs = np.broadcast_to(default, (625, N_SYM)).copy()
    m5_freqs = np.broadcast_to(default, (3125, N_SYM)).copy()
    spatial_freqs[spatial_idx] = spatial_rows
    m5_freqs[m5_idx] = m5_rows
    n_feats = blob[pos]
    pos += 1
    feat_ids = tuple(blob[pos:pos + n_feats])
    pos += n_feats
    (_thr_q8, sparse_len) = struct.unpack_from('<HI', blob, pos)
    pos += 6
    sparse = blob[pos:pos + sparse_len]
    pos += sparse_len
    (bs_len,) = struct.unpack_from('<I', blob, pos)
    pos += 4
    bitstream = blob[pos:pos + bs_len]
    top_support = decode_topband_payload(top_payload, n_pairs, h, w)
    road_mask = decode_boundary_mask_payload(road_payload, n_pairs, h, w)
    spatial_cdf_flat = np.zeros((625, N_SYM + 1), dtype = np.int64)
    spatial_cdf_flat[(:, 1:)] = np.cumsum(spatial_freqs.astype(np.int64), axis = -1)
    m5_cdf_flat = np.zeros((3125, N_SYM + 1), dtype = np.int64)
    m5_cdf_flat[(:, 1:)] = np.cumsum(m5_freqs.astype(np.int64), axis = -1)
    if sparse_colsfirst:
        (fired_idx, fired_freqs) = unpack_sparse_big_plain_colsfirst(sparse, precision = precision)
    elif sparse_plain:
        (fired_idx, fired_freqs) = unpack_sparse_big_plain(sparse, precision = precision)
    else:
        (fired_idx, fired_freqs) = unpack_sparse_big(sparse, precision = precision)
    fired_cdf_flat_py = []
    if fired_idx.size > 0:
        fired_cdf = np.zeros((fired_idx.size, N_SYM + 1), dtype = np.int64)
        fired_cdf[(:, 1:)] = np.cumsum(fired_freqs.astype(np.int64), axis = -1)
        fired_cdf_flat_py = fired_cdf.tolist()
# WARNING: Decompyle incomplete


"""
