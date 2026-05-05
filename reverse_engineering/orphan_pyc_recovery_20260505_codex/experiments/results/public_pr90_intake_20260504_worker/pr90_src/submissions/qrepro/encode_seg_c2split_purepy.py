"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``106:11: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``encode_seg_c2split_purepy.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/results/public_pr90_intake_20260504_worker/pr90_src/submissions/qrepro/encode_seg_c2split_purepy.py'
__recovery_spec__ = 'encode_seg_c2split_purepy.recovery_spec.json'
__recovery_ast_error__ = '106:11: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: encode_seg_c2split_purepy.cpython-312.pyc (Python 3.12)

'''Build seg_targets.bin via single-class peel split + 4-class AC residual.

Direction 1 (c2 variant) from the parallel-research synthesis 2026-04-11.
Sister module to encode_seg_v3_purepy.py — same range coder, same Markov-3
spatial / Markov-4 temporal context structure, but the dominant peel class
(default c2 = sky/background, 49.5% of pixels) is pulled out into a
separately compressed binary mask. The arithmetic-coded residual then has
a 4-symbol alphabet over the 50.5% of pixels that aren\'t c2.

Crucial detail: the residual encoder uses the FULL 5-class context (top,
left, top-left, prev) — i.e. the model knows when a neighbour was the peel
class — even though it only ever EMITS one of the 4 non-peel symbols.
That makes it strictly tighter than the c2_entropy.py Strategy B estimate
which collapsed contexts to 4 classes.

On-disk format ("seg_targets.bin"):

    uint16 n_pairs, uint16 H, uint16 W
    uint8  precision         # typically 16
    uint8  peel_class        # 0..N_CLASSES-1
    uint8  mask_format       # 0=bz2-raw, 1=bz2-packbits, 2=lzma-raw, 3=lzma-packbits
    uint32 mask_payload_len
    <mask_payload bytes>
    uint16 spatial_size_bytes  # 2 * 5**3 * 4
    uint16 temporal_size_bytes # 2 * 5**4 * 4
    spatial freqs  : uint16[5,5,5,4]    P(remapped_target | top,left,tl)  frame 0
    temporal freqs : uint16[5,5,5,5,4]  P(remapped_target | top,left,tl,prev) frames 1+
    uint32 bitstream_length
    <bitstream bytes>

Each freq row sums to exactly 2**precision so the decoder infers totals
without storing them. The remap maps the 4 non-peel labels into {0,1,2,3}
in increasing-original-label order (so for peel=2, residual symbols
{0,1,2,3} mean original classes {0,1,3,4}). The context indices on the
spatial/temporal axes are still the full 5-class labels.

Usage:
    python encode_seg_c2split_purepy.py --src seg_targets.bin --peel 2 --roundtrip
'''
from __future__ import annotations
import argparse
import bz2
import lzma
import struct
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from range_coder import RangeDecoder, RangeEncoder
N_CLASSES = 5
BORDER = 0
PRECISION = 16
MASK_FORMAT_BZ2_RAW = 0
MASK_FORMAT_BZ2_PACKBITS = 1
MASK_FORMAT_LZMA_RAW = 2
MASK_FORMAT_LZMA_PACKBITS = 3

def load_seg_targets_lzma(path = None):
    '''Read the v2 lzma+rle seg_targets.bin into a (n_pairs, H, W) uint8 array.'''
    
    def rle_decode(data = None, output_size = None):
        result = np.empty(output_size, dtype = np.uint8)
        pos = 0
        i = 0
        if i < len(data):
            result[pos:pos + data[i]] = data[i + 1]
            pos += data[i]
            i += 2
            if i < len(data):
                continue
        return result

    f = open(path, 'rb')
    (n_pairs, h, w) = struct.unpack('<HHH', f.read(6))
    rle = lzma.decompress(f.read())
    None(None, None)
# WARNING: Decompyle incomplete


def make_remap_tables(peel_class = None):
    '''Build forward / inverse maps between 5-class labels and 4-class residual indices.

    forward[c] -> 0..3 for non-peel classes, 255 sentinel for peel class
    inverse[i] -> original 5-class label for residual index i in [0,4)
    '''
    forward = np.full(N_CLASSES, 255, dtype = np.uint8)
    inverse = np.zeros(N_CLASSES - 1, dtype = np.uint8)
    new_idx = 0
    for c in range(N_CLASSES):
        if c == peel_class:
            continue
        forward[c] = new_idx
        inverse[new_idx] = c
        new_idx += 1
    return (forward, inverse)


def compute_spatial_contexts(frame = None):
    '''Top / left / top-left neighbour planes with BORDER padding at edges.'''
    (h, w) = frame.shape
    top = np.full((h, w), BORDER, dtype = np.uint8)
    top[(1:, :)] = frame[(:-1, :)]
    left = np.full((h, w), BORDER, dtype = np.uint8)
    left[(:, 1:)] = frame[(:, :-1)]
    tl = np.full((h, w), BORDER, dtype = np.uint8)
    tl[(1:, 1:)] = frame[(:-1, :-1)]
    return (top, left, tl)


def build_spatial_counts_split(seg = None, peel_class = None):
    \"\"\"3-neighbour counts over the 4-class residual, full 5-class context.

    Counts are accumulated only at non-peel positions (the encoder skips
    peel positions). Context indices are still 5-class because that's what
    the decoder reconstructs from the mask + previously decoded pixels.
    \"\"\"
    (forward, _) = make_remap_tables(peel_class)
    counts = np.zeros((N_CLASSES, N_CLASSES, N_CLASSES, N_CLASSES - 1), dtype = np.int64)
    for i in range(seg.shape[0]):
        frame = seg[i]
        (top, left, tl) = compute_spatial_contexts(frame)
        non_peel = frame != peel_class
        target = forward[frame[non_peel]]
        np.add.at(counts, (top[non_peel].ravel(), left[non_peel].ravel(), tl[non_peel].ravel(), target.ravel()), 1)
    return counts


def build_temporal_counts_split(seg = None, peel_class = None):
    '''4-neighbour counts (top, left, tl, prev) over the 4-class residual.'''
    (forward, _) = make_remap_tables(peel_class)
    counts = np.zeros((N_CLASSES, N_CLASSES, N_CLASSES, N_CLASSES, N_CLASSES - 1), dtype = np.int64)
    for i in range(1, seg.shape[0]):
        frame = seg[i]
        prev = seg[i - 1]
        (top, left, tl) = compute_spatial_contexts(frame)
        non_peel = frame != peel_class
        target = forward[frame[non_peel]]
        np.add.at(counts, (top[non_peel].ravel(), left[non_peel].ravel(), tl[non_peel].ravel(), prev[non_peel].ravel(), target.ravel()), 1)
    return counts


def quantize_freqs(counts = None, precision = None):
    '''Convert raw counts to uint16 freqs summing to exactly 2**precision per row.

    Last axis is the symbol alphabet. Identical to encode_seg_v3_purepy.quantize_freqs
    but kept here so the c2-split encoder is fully self-contained for the
    inflate-side decoder.
    '''
    denom = 1 << precision
    smoothed = counts.astype(np.int64) + 1
    row_sums = smoothed.sum(axis = -1, keepdims = True)
    freqs = np.maximum(smoothed * denom // row_sums, 1)
    it = np.nditer(freqs[(..., 0)], flags = [
        'multi_index'])
# WARNING: Decompyle incomplete


def encode_mask_best(mask_uint8 = None):
    '''Try four binary-mask compressors and return the smallest payload + format tag.

    mask_uint8 must be a flat 0/1 uint8 array. Returns (payload_bytes, mask_format).
    '''
    raw = mask_uint8.tobytes()
    packed = np.packbits(mask_uint8).tobytes()
    options = [
        (MASK_FORMAT_BZ2_RAW, bz2.compress(raw, compresslevel = 9)),
        (MASK_FORMAT_BZ2_PACKBITS, bz2.compress(packed, compresslevel = 9)),
        (MASK_FORMAT_LZMA_RAW, lzma.compress(raw, preset = 9 | lzma.PRESET_EXTREME)),
        (MASK_FORMAT_LZMA_PACKBITS, lzma.compress(packed, preset = 9 | lzma.PRESET_EXTREME))]
    (best_format, best_payload) = min(options, key = (lambda kv: len(kv[1])))
    return (best_payload, best_format)


def decode_mask_payload(payload, mask_format = None, n_pairs = None, h = None, w = ('payload', 'bytes', 'mask_format', 'int', 'n_pairs', 'int', 'h', 'int', 'w', 'int', 'return', 'np.ndarray')):
    '''Inverse of encode_mask_best. Returns a (n_pairs, H, W) uint8 0/1 array.'''
    n_pix = n_pairs * h * w
    if mask_format == MASK_FORMAT_BZ2_RAW:
        raw = bz2.decompress(payload)
        flat = np.frombuffer(raw, dtype = np.uint8)
    elif mask_format == MASK_FORMAT_BZ2_PACKBITS:
        packed = bz2.decompress(payload)
        flat = np.unpackbits(np.frombuffer(packed, dtype = np.uint8))[:n_pix]
    elif mask_format == MASK_FORMAT_LZMA_RAW:
        raw = lzma.decompress(payload)
        flat = np.frombuffer(raw, dtype = np.uint8)
    elif mask_format == MASK_FORMAT_LZMA_PACKBITS:
        packed = lzma.decompress(payload)
        flat = np.unpackbits(np.frombuffer(packed, dtype = np.uint8))[:n_pix]
    else:
        raise ValueError(f'''unknown mask_format: {mask_format}''')
    if flat.size != n_pix:
        raise ValueError(f'''mask payload decoded to {flat.size} bytes, expected {n_pix}''')
    return flat.reshape(n_pairs, h, w)


def encode_seg_split(seg, peel_class = None, spatial_freqs = None, temporal_freqs = None, precision = ('seg', 'np.ndarray', 'peel_class', 'int', 'spatial_freqs', 'np.ndarray', 'temporal_freqs', 'np.ndarray', 'precision', 'int', 'return', 'bytes')):
    '''Range-encode the 4-class residual at non-peel positions only.

    Frame 0 uses Markov-3 (top, left, top-left); frames 1+ use Markov-4
    (top, left, top-left, prev). Peel positions are skipped — the decoder
    reconstructs them from the bz2 mask.
    '''
    pass
# WARNING: Decompyle incomplete


def pack_archive_blob(n_pairs, h, w, precision, peel_class, mask_payload, mask_format = None, spatial_freqs = None, temporal_freqs = None, bitstream = ('n_pairs', 'int', 'h', 'int', 'w', 'int', 'precision', 'int', 'peel_class', 'int', 'mask_payload', 'bytes', 'mask_format', 'int', 'spatial_freqs', 'np.ndarray', 'temporal_freqs', 'np.ndarray', 'bitstream', 'bytes', 'return', 'bytes')):
    '''Assemble the on-disk seg_targets.bin for the c2-split format.'''
    header = struct.pack('<HHHBBB', n_pairs, h, w, precision, peel_class, mask_format)
    mask_len = struct.pack('<I', len(mask_payload))
    spatial_bytes = spatial_freqs.astype('<u2').tobytes()
    temporal_bytes = temporal_freqs.astype('<u2').tobytes()
    sizes = struct.pack('<HH', len(spatial_bytes), len(temporal_bytes))
    bslen = struct.pack('<I', len(bitstream))
    return header + mask_len + mask_payload + sizes + spatial_bytes + temporal_bytes + bslen + bitstream


def unpack_archive_blob(blob = None):
    '''Inverse of pack_archive_blob. Returns a dict of decoded fields for decode_seg_split.'''
    pos = 0
    (n_pairs, h, w, precision, peel_class, mask_format) = struct.unpack_from('<HHHBBB', blob, pos)
    pos += struct.calcsize('<HHHBBB')
    (mask_len,) = struct.unpack_from('<I', blob, pos)
    pos += 4
    mask_payload = blob[pos:pos + mask_len]
    pos += mask_len
    (spatial_size, temporal_size) = struct.unpack_from('<HH', blob, pos)
    pos += 4
    spatial_freqs = np.frombuffer(blob, dtype = '<u2', count = spatial_size // 2, offset = pos).reshape(N_CLASSES, N_CLASSES, N_CLASSES, N_CLASSES - 1)
    pos += spatial_size
    temporal_freqs = np.frombuffer(blob, dtype = '<u2', count = temporal_size // 2, offset = pos).reshape(N_CLASSES, N_CLASSES, N_CLASSES, N_CLASSES, N_CLASSES - 1)
    pos += temporal_size
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
        'temporal_freqs': temporal_freqs,
        'bitstream': bitstream }


def _decode_frame_spatial(dec, frame, mask_frame, cdf_py, total, h = None, w = None, peel_class = None, inv_remap_py = ('dec', 'RangeDecoder', 'frame', 'np.ndarray', 'mask_frame', 'np.ndarray', 'cdf_py', 'list', 'total', 'int', 'h', 'int', 'w', 'int', 'peel_class', 'int', 'inv_remap_py', 'list', 'return', 'None')):
    \"\"\"Inverse of the encoder's frame-0 spatial Markov-3 path. Pure python hot loop.\"\"\"
    pass
# WARNING: Decompyle incomplete


def _decode_frame_temporal(dec, frame, prev_frame, mask_frame, cdf_py, total, h = None, w = None, peel_class = None, inv_remap_py = ('dec', 'RangeDecoder', 'frame', 'np.ndarray', 'prev_frame', 'np.ndarray', 'mask_frame', 'np.ndarray', 'cdf_py', 'list', 'total', 'int', 'h', 'int', 'w', 'int', 'peel_class', 'int', 'inv_remap_py', 'list', 'return', 'None')):
    \"\"\"Inverse of the encoder's frame-1+ temporal Markov-4 path.\"\"\"
    pass
# WARNING: Decompyle incomplete


def decode_seg_split(blob = None):
    '''Decode the c2-split format back to a (n_pairs, H, W) uint8 array.'''
    fields = unpack_archive_blob(blob)
    n_pairs = fields['n_pairs']
    h = fields['h']
    w = fields['w']
    precision = fields['precision']
    peel_class = fields['peel_class']
    spatial_freqs = fields['spatial_freqs']
    temporal_freqs = fields['temporal_freqs']
    bitstream = fields['bitstream']
    mask = decode_mask_payload(fields['mask_payload'], fields['mask_format'], n_pairs, h, w)
    (_, inverse) = make_remap_tables(peel_class)
    inv_remap_py = inverse.tolist()
    total = 1 << precision
    n_other = N_CLASSES - 1
    spatial_cdf = np.zeros((N_CLASSES, N_CLASSES, N_CLASSES, n_other + 1), dtype = np.int64)
    spatial_cdf[(..., 1:)] = np.cumsum(spatial_freqs.astype(np.int64), axis = -1)
    temporal_cdf = np.zeros((N_CLASSES, N_CLASSES, N_CLASSES, N_CLASSES, n_other + 1), dtype = np.int64)
    temporal_cdf[(..., 1:)] = np.cumsum(temporal_freqs.astype(np.int64), axis = -1)
    spatial_py = spatial_cdf.tolist()
    temporal_py = temporal_cdf.tolist()
    dec = RangeDecoder(bitstream)
    out = np.zeros((n_pairs, h, w), dtype = np.uint8)
    _decode_frame_spatial(dec, out[0], mask[0], spatial_py, total, h, w, peel_class, inv_remap_py)
    for fi in range(1, n_pairs):
        _decode_frame_temporal(dec, out[fi], out[fi - 1], mask[fi], temporal_py, total, h, w, peel_class, inv_remap_py)
    return out


def build_blob_from_seg(seg = None, peel_class = None, precision = None):
    '''End-to-end: counts -> freqs -> mask + bitstream -> packed blob.'''
    print('  building spatial counts (Markov-3, frame 0 model)...', flush = True)
    spatial_counts = build_spatial_counts_split(seg, peel_class)
    print('  building temporal counts (Markov-4, frames 1+ model)...', flush = True)
    temporal_counts = build_temporal_counts_split(seg, peel_class)
    print('  quantizing freqs...', flush = True)
    spatial_freqs = quantize_freqs(spatial_counts, precision)
    temporal_freqs = quantize_freqs(temporal_counts, precision)
    print(f'''  encoding peel-class binary mask (peel=c{peel_class})...''', flush = True)
    is_peel = (seg == peel_class).astype(np.uint8).reshape(-1)
    (mask_payload, mask_format) = encode_mask_best(is_peel)
    mask_format_names = {
        MASK_FORMAT_LZMA_PACKBITS: 'lzma-packbits',
        MASK_FORMAT_LZMA_RAW: 'lzma-raw',
        MASK_FORMAT_BZ2_PACKBITS: 'bz2-packbits',
        MASK_FORMAT_BZ2_RAW: 'bz2-raw' }
    print(f'''    chose {mask_format_names[mask_format]}: {len(mask_payload):,} B''')
    print('  encoding 4-class residual via pure-python range coder...', flush = True)
    t0 = time.time()
    bitstream = encode_seg_split(seg, peel_class, spatial_freqs, temporal_freqs, precision)
    elapsed = time.time() - t0
    n_non_peel = int((seg != peel_class).sum())
    print(f'''    encoded {n_non_peel} non-peel symbols in {elapsed:.1f}s ({n_non_peel / max(elapsed, 1e-06):.0f} sym/s)''')
    blob = pack_archive_blob(seg.shape[0], seg.shape[1], seg.shape[2], precision, peel_class, mask_payload, mask_format, spatial_freqs, temporal_freqs, bitstream)
    return blob


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--peel', type = int, default = 2, choices = list(range(N_CLASSES)), help = 'class to peel into a separate binary mask (default 2 = sky/background)')
    parser.add_argument('--src', type = str, default = 'seg_targets.bin', help = 'source seg_targets.bin')
    parser.add_argument('--out', type = str, default = None, help = 'output path (default: seg_targets_c2split.bin next to this file)')
    parser.add_argument('--roundtrip', action = 'store_true', help = 'after writing the blob, decode it and assert byte-identical reconstruction')
    parser.add_argument('--limit', type = int, default = None, help = 'encode only first N pairs (smoke test)')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    seg_path = Path(args.src)
    if not seg_path.is_absolute():
        seg_path = Path.cwd() / seg_path
    out_path = Path(args.out) if args.out else Path(__file__).resolve().with_name('seg_targets_c2split.bin')
    out_path.parent.mkdir(parents = True, exist_ok = True)
    print(f'''loading seg from {seg_path.relative_to(repo)}''')
    seg = load_seg_targets_lzma(seg_path)
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    main()
    return None

"""
