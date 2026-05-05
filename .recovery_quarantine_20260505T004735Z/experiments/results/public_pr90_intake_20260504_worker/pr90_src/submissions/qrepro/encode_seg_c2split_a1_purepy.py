"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``66:10: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``encode_seg_c2split_a1_purepy.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/results/public_pr90_intake_20260504_worker/pr90_src/submissions/qrepro/encode_seg_c2split_a1_purepy.py'
__recovery_spec__ = 'encode_seg_c2split_a1_purepy.recovery_spec.json'
__recovery_ast_error__ = '66:10: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: encode_seg_c2split_a1_purepy.cpython-312.pyc (Python 3.12)

\"\"\"a1 Markov seg_targets codec: +top_top stacked on trpp M6 (task #16 Gate 1).

Stacks candidate a1 (+top_top = (-2,0,cur)) as an additional dim on top of trpp's
M6 context, turning frames 2+ into Markov-7. Frames 0 and 1 are unchanged.

  - frame 0:   P(sym | top, left, tl, tr)                           — Markov-4
  - frame 1:   P(sym | top, left, tl, tr, prev)                     — Markov-5
  - frame 2+:  P(sym | top, left, tl, tr, prev, prev_prev, top_top) — a1 Markov-7

Three CDF tables (all uint16 freqs, precision = PRECISION):

  spatial M4   : (5,5,5,5,4)         = 2,500 rows × 4 = 5,000 B raw
  temporal M5  : (5,5,5,5,5,4)       = 12,500 rows × 4 = 25,000 B raw
  a1      M7   : (5,5,5,5,5,5,5,4)   = 312,500 rows × 4 = 625,000 B raw

Counts are populated cleanly by segment:
  - M4 from frame 0 only
  - M5 from frame 1 only (clean separation from M6; avoids table-redundancy)
  - M6 from frames 2+ only

DEFLATE is expected to absorb most of the 100 KB M6 CDF raw-byte growth (per the
DEFLATE-93% rule, soft-math-6156). Earlier scoping projected -6,506 B archive
delta for B alone; stacking on A should be near-additive under the same rule.

On-disk format (little-endian):

    uint16 n_pairs, uint16 H, uint16 W
    uint8  precision
    uint8  peel_class
    uint8  mask_format
    uint32 mask_payload_len
    <mask_payload bytes>
    uint16 spatial_size_bytes       # 2 * 5**4 * 4
    uint16 temporal_m5_size_bytes   # 2 * 5**5 * 4
    uint32 temporal_m6_size_bytes   # 2 * 5**7 * 4 (M7 = 625,000 B raw)
    spatial freqs    : uint16[5,5,5,5,4]
    temporal_m5 freqs: uint16[5,5,5,5,5,4]
    temporal_m6 freqs: uint16[5,5,5,5,5,5,5,4]
    uint32 bitstream_length
    <bitstream bytes>

Decoder (seg_c2split_a1_codec.py) MUST match exactly.

Usage:
    python encode_seg_c2split_a1_purepy.py --src seg_targets.bin --peel 4 --roundtrip
\"\"\"
from __future__ import annotations
import argparse
import struct
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from range_coder import RangeDecoder, RangeEncoder
from encode_seg_c2split_purepy import N_CLASSES, BORDER, PRECISION, MASK_FORMAT_BZ2_RAW, MASK_FORMAT_BZ2_PACKBITS, MASK_FORMAT_LZMA_RAW, MASK_FORMAT_LZMA_PACKBITS, load_seg_targets_lzma, make_remap_tables, compute_spatial_contexts, quantize_freqs, encode_mask_best, decode_mask_payload
from encode_seg_c2split_tr_purepy import compute_tr

def compute_tt(frame = None):
    '''Top-top neighbour (y-2, x). Causal under raster. BORDER at edges.'''
    (h, w) = frame.shape
    tt = np.full((h, w), BORDER, dtype = frame.dtype)
    tt[(2:, :)] = frame[(:-2, :)]
    return tt


def build_spatial_counts_m4(seg = None, peel_class = None):
    '''Frame-0 spatial M4 counts: (top, left, tl, tr). Shape (5,5,5,5,4).'''
    (forward, _) = make_remap_tables(peel_class)
    counts = np.zeros((N_CLASSES,) * 4 + (N_CLASSES - 1,), dtype = np.int64)
    frame = seg[0]
    (top, left, tl) = compute_spatial_contexts(frame)
    tr = compute_tr(frame)
    non_peel = frame != peel_class
    target = forward[frame[non_peel]]
    np.add.at(counts, (top[non_peel].ravel(), left[non_peel].ravel(), tl[non_peel].ravel(), tr[non_peel].ravel(), target.ravel()), 1)
    return counts


def build_temporal_counts_m5(seg = None, peel_class = None):
    '''Frame-1 temporal M5 counts: (top, left, tl, tr, prev). Shape (5,5,5,5,5,4).

    Populated by frame 1 ONLY — frames 2+ go to the M6 table instead.
    '''
    (forward, _) = make_remap_tables(peel_class)
    counts = np.zeros((N_CLASSES,) * 5 + (N_CLASSES - 1,), dtype = np.int64)
    if seg.shape[0] < 2:
        return counts
    frame = None[1]
    prev = seg[0]
    (top, left, tl) = compute_spatial_contexts(frame)
    tr = compute_tr(frame)
    non_peel = frame != peel_class
    target = forward[frame[non_peel]]
    np.add.at(counts, (top[non_peel].ravel(), left[non_peel].ravel(), tl[non_peel].ravel(), tr[non_peel].ravel(), prev[non_peel].ravel(), target.ravel()), 1)
    return counts


def build_temporal_counts_m6(seg = None, peel_class = None):
    '''Frames 2+ a1 M7 counts: (top, left, tl, tr, prev, prev_prev, top_top).

    Shape (5,5,5,5,5,5,5,4) = 312,500 contexts × 4 residual symbols.
    '''
    (forward, _) = make_remap_tables(peel_class)
    counts = np.zeros((N_CLASSES,) * 7 + (N_CLASSES - 1,), dtype = np.int64)
    for i in range(2, seg.shape[0]):
        frame = seg[i]
        prev = seg[i - 1]
        prev_prev = seg[i - 2]
        (top, left, tl) = compute_spatial_contexts(frame)
        tr = compute_tr(frame)
        tt = compute_tt(frame)
        non_peel = frame != peel_class
        target = forward[frame[non_peel]]
        np.add.at(counts, (top[non_peel].ravel(), left[non_peel].ravel(), tl[non_peel].ravel(), tr[non_peel].ravel(), prev[non_peel].ravel(), prev_prev[non_peel].ravel(), tt[non_peel].ravel(), target.ravel()), 1)
    return counts


def encode_seg_split_a1(seg, peel_class, spatial_freqs = None, temporal_m5_freqs = None, temporal_m6_freqs = None, precision = ('seg', 'np.ndarray', 'peel_class', 'int', 'spatial_freqs', 'np.ndarray', 'temporal_m5_freqs', 'np.ndarray', 'temporal_m6_freqs', 'np.ndarray', 'precision', 'int', 'return', 'bytes')):
    '''Range-encode the 4-class residual. Frame 0 M4, frame 1 M5, frames 2+ M6.'''
    pass
# WARNING: Decompyle incomplete


def pack_archive_blob(n_pairs, h, w, precision, peel_class, mask_payload, mask_format, spatial_freqs = None, temporal_m5_freqs = None, temporal_m6_freqs = None, bitstream = ('n_pairs', 'int', 'h', 'int', 'w', 'int', 'precision', 'int', 'peel_class', 'int', 'mask_payload', 'bytes', 'mask_format', 'int', 'spatial_freqs', 'np.ndarray', 'temporal_m5_freqs', 'np.ndarray', 'temporal_m6_freqs', 'np.ndarray', 'bitstream', 'bytes', 'return', 'bytes')):
    '''Assemble the on-disk seg_targets.bin with three CDF sections.'''
    header = struct.pack('<HHHBBB', n_pairs, h, w, precision, peel_class, mask_format)
    mask_len = struct.pack('<I', len(mask_payload))
    spatial_bytes = spatial_freqs.astype('<u2').tobytes()
    m5_bytes = temporal_m5_freqs.astype('<u2').tobytes()
    m6_bytes = temporal_m6_freqs.astype('<u2').tobytes()
    sizes = struct.pack('<HHI', len(spatial_bytes), len(m5_bytes), len(m6_bytes))
    bslen = struct.pack('<I', len(bitstream))
    return header + mask_len + mask_payload + sizes + spatial_bytes + m5_bytes + m6_bytes + bslen + bitstream


def unpack_archive_blob(blob = None):
    '''Inverse of pack_archive_blob for the +trpp variant.'''
    pos = 0
    (n_pairs, h, w, precision, peel_class, mask_format) = struct.unpack_from('<HHHBBB', blob, pos)
    pos += struct.calcsize('<HHHBBB')
    (mask_len,) = struct.unpack_from('<I', blob, pos)
    pos += 4
    mask_payload = blob[pos:pos + mask_len]
    pos += mask_len
    (spatial_size, m5_size, m6_size) = struct.unpack_from('<HHI', blob, pos)
    pos += struct.calcsize('<HHI')
    spatial_freqs = np.frombuffer(blob, dtype = '<u2', count = spatial_size // 2, offset = pos).reshape((N_CLASSES,) * 4 + (N_CLASSES - 1,))
    pos += spatial_size
    m5_freqs = np.frombuffer(blob, dtype = '<u2', count = m5_size // 2, offset = pos).reshape((N_CLASSES,) * 5 + (N_CLASSES - 1,))
    pos += m5_size
    m6_freqs = np.frombuffer(blob, dtype = '<u2', count = m6_size // 2, offset = pos).reshape((N_CLASSES,) * 7 + (N_CLASSES - 1,))
    pos += m6_size
    (bs_len,) = struct.unpack_from('<I', blob, pos)
    pos += 4
    bitstream = blob[pos:pos + bs_len]
    return {
        'n_pairs': n_pairs,
        'h': h,
        'w': w,
        'precision': precision,
        'peel_class': peel_class,
        'mask_format': mask_format,
        'mask_payload': mask_payload,
        'spatial_freqs': spatial_freqs,
        'temporal_m5_freqs': m5_freqs,
        'temporal_m6_freqs': m6_freqs,
        'bitstream': bitstream }


def _decode_frame_spatial_m4(dec, frame, mask_frame, cdf_py, total, h = None, w = None, peel_class = None, inv_remap_py = ('dec', 'RangeDecoder', 'frame', 'np.ndarray', 'mask_frame', 'np.ndarray', 'cdf_py', 'list', 'total', 'int', 'h', 'int', 'w', 'int', 'peel_class', 'int', 'inv_remap_py', 'list', 'return', 'None')):
    '''Frame-0 M4-spatial decode.'''
    pass
# WARNING: Decompyle incomplete


def _decode_frame_temporal_m5(dec, frame, prev_frame, mask_frame, cdf_py, total, h = None, w = None, peel_class = None, inv_remap_py = ('dec', 'RangeDecoder', 'frame', 'np.ndarray', 'prev_frame', 'np.ndarray', 'mask_frame', 'np.ndarray', 'cdf_py', 'list', 'total', 'int', 'h', 'int', 'w', 'int', 'peel_class', 'int', 'inv_remap_py', 'list', 'return', 'None')):
    '''Frame-1 M5-temporal decode: (top, left, tl, tr, prev).'''
    pass
# WARNING: Decompyle incomplete


def _decode_frame_temporal_m6(dec, frame, prev_frame, prev_prev_frame, mask_frame, cdf_py, total, h = None, w = None, peel_class = None, inv_remap_py = ('dec', 'RangeDecoder', 'frame', 'np.ndarray', 'prev_frame', 'np.ndarray', 'prev_prev_frame', 'np.ndarray', 'mask_frame', 'np.ndarray', 'cdf_py', 'list', 'total', 'int', 'h', 'int', 'w', 'int', 'peel_class', 'int', 'inv_remap_py', 'list', 'return', 'None')):
    '''Frames 2+ a1 M7 decode: (top, left, tl, tr, prev, prev_prev, top_top).'''
    pass
# WARNING: Decompyle incomplete


def decode_seg_split_a1(blob = None):
    '''Decode the a1 c2split format to a (n_pairs, H, W) uint8 array.'''
    fields = unpack_archive_blob(blob)
    n_pairs = fields['n_pairs']
    h = fields['h']
    w = fields['w']
    precision = fields['precision']
    peel_class = fields['peel_class']
    spatial_freqs = fields['spatial_freqs']
    m5_freqs = fields['temporal_m5_freqs']
    m6_freqs = fields['temporal_m6_freqs']
    bitstream = fields['bitstream']
    mask = decode_mask_payload(fields['mask_payload'], fields['mask_format'], n_pairs, h, w)
    (_, inverse) = make_remap_tables(peel_class)
    inv_remap_py = inverse.tolist()
    total = 1 << precision
    n_other = N_CLASSES - 1
    spatial_cdf = np.zeros((N_CLASSES,) * 4 + (n_other + 1,), dtype = np.int64)
    spatial_cdf[(..., 1:)] = np.cumsum(spatial_freqs.astype(np.int64), axis = -1)
    m5_cdf = np.zeros((N_CLASSES,) * 5 + (n_other + 1,), dtype = np.int64)
    m5_cdf[(..., 1:)] = np.cumsum(m5_freqs.astype(np.int64), axis = -1)
    m6_cdf = np.zeros((N_CLASSES,) * 7 + (n_other + 1,), dtype = np.int64)
    m6_cdf[(..., 1:)] = np.cumsum(m6_freqs.astype(np.int64), axis = -1)
    spatial_py = spatial_cdf.tolist()
    m5_py = m5_cdf.tolist()
    m6_py = m6_cdf.tolist()
    dec = RangeDecoder(bitstream)
    out = np.zeros((n_pairs, h, w), dtype = np.uint8)
    _decode_frame_spatial_m4(dec, out[0], mask[0], spatial_py, total, h, w, peel_class, inv_remap_py)
    if n_pairs >= 2:
        _decode_frame_temporal_m5(dec, out[1], out[0], mask[1], m5_py, total, h, w, peel_class, inv_remap_py)
    for fi in range(2, n_pairs):
        _decode_frame_temporal_m6(dec, out[fi], out[fi - 1], out[fi - 2], mask[fi], m6_py, total, h, w, peel_class, inv_remap_py)
    return out


def build_blob_from_seg(seg = None, peel_class = None, precision = None):
    '''End-to-end: counts → freqs → mask + bitstream → packed blob (M4/M5/M6 stacked).'''
    print('  building M4-spatial counts (+tr), frame 0...', flush = True)
    spatial_counts = build_spatial_counts_m4(seg, peel_class)
    print('  building M5-temporal counts (+tr +prev), frame 1...', flush = True)
    m5_counts = build_temporal_counts_m5(seg, peel_class)
    print('  building a1 M7 counts (trpp + top_top), frames 2+...', flush = True)
    m6_counts = build_temporal_counts_m6(seg, peel_class)
    print('  quantizing freqs...', flush = True)
    spatial_freqs = quantize_freqs(spatial_counts, precision)
    m5_freqs = quantize_freqs(m5_counts, precision)
    m6_freqs = quantize_freqs(m6_counts, precision)
    print(f'''  encoding peel-class binary mask (peel=c{peel_class})...''', flush = True)
    is_peel = (seg == peel_class).astype(np.uint8).reshape(-1)
    (mask_payload, mask_format) = encode_mask_best(is_peel)
    mask_format_names = {
        MASK_FORMAT_LZMA_PACKBITS: 'lzma-packbits',
        MASK_FORMAT_LZMA_RAW: 'lzma-raw',
        MASK_FORMAT_BZ2_PACKBITS: 'bz2-packbits',
        MASK_FORMAT_BZ2_RAW: 'bz2-raw' }
    print(f'''    chose {mask_format_names[mask_format]}: {len(mask_payload):,} B''')
    print('  encoding 4-class residual via pure-python range coder (M4/M5/a1 M7)...', flush = True)
    t0 = time.time()
    bitstream = encode_seg_split_a1(seg, peel_class, spatial_freqs, m5_freqs, m6_freqs, precision)
    elapsed = time.time() - t0
    n_non_peel = int((seg != peel_class).sum())
    print(f'''    encoded {n_non_peel} non-peel symbols in {elapsed:.1f}s ({n_non_peel / max(elapsed, 1e-06):.0f} sym/s, {len(bitstream):,} B bitstream)''')
    blob = pack_archive_blob(seg.shape[0], seg.shape[1], seg.shape[2], precision, peel_class, mask_payload, mask_format, spatial_freqs, m5_freqs, m6_freqs, bitstream)
    return blob


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--peel', type = int, default = 4, choices = list(range(N_CLASSES)), help = 'peel class (default 4 to match shipped archive)')
    parser.add_argument('--src', type = str, default = 'seg_targets.bin', help = 'lzma+rle source seg_targets.bin')
    parser.add_argument('--out', type = str, default = None, help = 'output path for the new seg_targets.bin')
    parser.add_argument('--roundtrip', action = 'store_true', help = 'decode + assert byte-identical reconstruction')
    parser.add_argument('--limit', type = int, default = None, help = 'encode only first N pairs (smoke)')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    seg_path = Path(args.src)
    if not seg_path.is_absolute():
        seg_path = Path.cwd() / seg_path
    out_path = Path(args.out) if args.out else Path(__file__).resolve().with_name('seg_targets_a1.bin')
    out_path.parent.mkdir(parents = True, exist_ok = True)
    print(f'''loading seg from {seg_path.relative_to(repo)}''')
    seg = load_seg_targets_lzma(seg_path)
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    main()
    return None

"""
