"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``35:14: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``pr85_bundle.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/pr85_bundle.py'
__recovery_spec__ = 'pr85_bundle.recovery_spec.json'
__recovery_ast_error__ = '35:14: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: pr85_bundle.cpython-312.pyc (Python 3.12)

'''Canonical parser/serializer for PR85 single-member ``x`` bundles.

PR85 uses a compact archive grammar: one ZIP member named ``x`` whose body is a
concatenation of a small length header and eleven charged payload segments.
This module is deliberately runtime-light. It does not import the public replay
inflater, load models, or decode scorer-dependent tensors; it only validates
and slices bytes so later optimization passes can share one fail-closed
contract.
'''
from __future__ import annotations
import hashlib
import struct
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping
SEGMENT_ORDER: 'tuple[str, ...]' = ('mask', 'model', 'pose', 'post', 'shift', 'frac', 'frac2', 'frac3', 'bias', 'region', 'randmulti')
HEADER_V5_SEGMENTS: 'tuple[str, ...]' = SEGMENT_ORDER[:8]
HEADER_EXPLICIT30_SEGMENTS: 'tuple[str, ...]' = SEGMENT_ORDER[:10]
FIXED_V5_LENGTHS: 'dict[str, int]' = {
    'bias': 223,
    'region': 273 }
QPOST_MAGIC = b'QPS1'
HPM1_MAGIC = b'HPM1'
HPM1_HEADER_BYTES = 48
QPOST_STREAM_NAMES: 'tuple[str, ...]' = ('post', 'shift', 'frac', 'frac2', 'frac3', 'bias', 'region', 'randmulti')
PR85_HEADERLESS_RANDMULTI_SPECS: 'tuple[tuple[int, int, int, int], ...]' = ((24, 32, 1, 12), (12, 16, 1, 1), (6, 8, 1, 1), (3, 4, 1, 1), (2, 2, 1, 1), (8, 8, 1, 1), (4, 4, 1, 1), (4, 8, 1, 1), (2, 4, 1, 1), (2, 8, 1, 1), (1, 2, 1, 1), (1, 4, 1, 1), (2, 1, 1, 1), (4, 1, 1, 1), (8, 1, 1, 1), (1, 8, 1, 1), (16, 1, 1, 1), (1, 16, 1, 1), (32, 1, 1, 1), (64, 1, 1, 1), (256, 1, 1, 1), (1024, 1, 1, 1), (2048, 1, 1, 1), (4096, 1, 1, 1), (8192, 1, 1, 1), (8192, 1, 1, 1), (16384, 1, 1, 1), (32768, 1, 1, 1), (65536, 1, 1, 1), (131072, 1, 1, 1), (262144, 1, 1, 1), (524288, 1, 1, 1), (1048576, 1, 1, 1), (874, 1, 1, 1), (874, 1, 1, 1), (2097152, 1, 1, 1), (875, 1, 1, 1), (876, 1, 1, 1), (877, 1, 1, 1), (1164, 1, 1, 1), (878, 1, 1, 1), (879, 1, 1, 1), (880, 1, 1, 1), (881, 1, 1, 1), (882, 1, 1, 1), (512, 2, 1, 1), (256, 2, 1, 1), (128, 2, 1, 1), (64, 2, 1, 1), (32, 2, 1, 1), (16, 2, 1, 1), (8, 2, 1, 1), (4, 2, 1, 1), (4, 4, 1, 1), (8, 4, 1, 1), (16, 4, 1, 1), (32, 4, 1, 1), (64, 4, 1, 1), (128, 4, 1, 1), (64, 8, 1, 1), (32, 8, 1, 1), (222, 222, 4, 1), (222, 223, 4, 1), (223, 222, 2, 1), (223, 223, 4, 1), (223, 221, 4, 1), (223, 224, 4, 1), (223, 221, 4, 1), (223, 219, 4, 1), (64, 16, 1, 1), (223, 218, 4, 1), (224, 222, 4, 1))

class Pr85BundleError(ValueError):
    '''Raised when a PR85 bundle violates the byte grammar.'''
    pass

Pr85Bundle = <NODE:12>()
Pr85RuntimeExpansion = <NODE:12>()
Pr85SegmentContract = <NODE:12>()

def _sha256(data = dataclass(frozen = True)):
    return hashlib.sha256(data).hexdigest()


def _brotli_decompress(data = None, name = None):
    
    try:
        import brotli
        
        try:
            return brotli.decompress(data)
            except ImportError:
                exc = None
                raise Pr85BundleError('PR85 runtime expansion requires the brotli package'), exc
                exc = None
                del exc
        except brotli.error:
            exc = None
            raise Pr85BundleError(f'''PR85 segment {name!r} is not Brotli-decodable'''), exc
            exc = None
            del exc




def _brotli_compress(data = None, *, quality, lgwin):
    
    try:
        import brotli
        return brotli.compress(data, quality = quality, lgwin = lgwin)
    except ImportError:
        exc = None
        raise Pr85BundleError('PR85 runtime expansion requires the brotli package'), exc
        exc = None
        del exc



def decode_rmb1_randmulti_payload(encoded = None):
    '''Decode PR92 ``RMB1`` randmulti bytes to PR85 headerless sparse rows.'''
    if len(encoded) < 6 or encoded[:4] != b'RMB1':
        raise Pr85BundleError('bad RMB1 randmulti payload')
    mask_len = int.from_bytes(encoded[4:6], 'little')
    mask_br = encoded[6:6 + mask_len]
    vals_br = encoded[6 + mask_len:]
    if not mask_br or vals_br:
        raise Pr85BundleError('truncated RMB1 randmulti payload')
    mask = _brotli_decompress(mask_br, 'randmulti_rmb1_mask')
    vals = _brotli_decompress(vals_br, 'randmulti_rmb1_values')
    if len(mask) % 75:
        raise Pr85BundleError('bad RMB1 mask length')
    out = bytearray()
    vals_pos = 0
    for row_start in range(0, len(mask), 75):
        row_mask = mask[row_start:row_start + 75]
        indices = []
        row_values = []
        for byte_i, byte in enumerate(row_mask):
            for bit in range(8):
                frame_i = byte_i * 8 + bit
                if frame_i >= 600:
                    range(8)
                    continue
                if not byte & 1 << bit:
                    continue
                if vals_pos >= len(vals):
                    raise Pr85BundleError('truncated RMB1 values')
                indices.append(frame_i)
                row_values.append(vals[vals_pos])
                vals_pos += 1
        count = len(indices)
        if count < 255:
            out.append(count)
        else:
            out.append(255)
            out.extend(count.to_bytes(2, 'little'))
        last = -1
        for idx in indices:
            delta = idx - last - 1
            last = idx
            byte = delta & 127
            delta >>= 7
            if delta:
                out.append(byte | 128)
            else:
                out.append(byte)
        out.extend(row_values)
    if vals_pos != len(vals):
        raise Pr85BundleError('unused RMB1 values')
    return bytes(out)


def decode_pr85_randmulti_to_headerless_rows(encoded_randmulti = None):
    '''Decode supported PR85-family randmulti containers to headerless rows.'''
    if encoded_randmulti.startswith(b'RMB1'):
        decoded = decode_rmb1_randmulti_payload(encoded_randmulti)
        codec = 'RMB1_bitmask_value_randmulti'
    else:
        decoded = _brotli_decompress(encoded_randmulti, 'randmulti')
        codec = 'brotli_headerless_sparse_randmulti'
    (groups, profile) = _slice_pr85_randmulti_group_payloads(decoded)
    profile.update({
        'codec': codec,
        'encoded_bytes': len(encoded_randmulti),
        'encoded_sha256': _sha256(encoded_randmulti),
        'decoded_sha256': _sha256(decoded),
        'decoded_group_count': len(groups) })
    return (decoded, profile)


def compare_pr85_randmulti_decoded_rows(reference_encoded_randmulti = None, candidate_encoded_randmulti = None):
    \"\"\"Compare PR85-family randmulti streams after normalizing to decoded rows.

    PR92's ``RMB1`` container is a rate-only recode when its decoded sparse rows
    match the original PR85/STBM randmulti rows.  Builders can use this helper
    to prove that an RMB1 transplant changes only representation bytes, not the
    randmulti action schedule consumed by the runtime.
    \"\"\"
    (reference_decoded, reference_profile) = decode_pr85_randmulti_to_headerless_rows(reference_encoded_randmulti)
    (candidate_decoded, candidate_profile) = decode_pr85_randmulti_to_headerless_rows(candidate_encoded_randmulti)
    reference_sha = _sha256(reference_decoded)
    candidate_sha = _sha256(candidate_decoded)
    decoded_rows_match = reference_decoded == candidate_decoded
    return {
        'schema': 'pr85_randmulti_decoded_rows_parity_v1',
        'parity_status': 'passed' if decoded_rows_match else 'failed',
        'decoded_rows_match': decoded_rows_match,
        'decoded_rows_bytes': len(candidate_decoded),
        'decoded_rows_sha256': candidate_sha,
        'reference': {
            'codec': reference_profile['codec'],
            'encoded_bytes': reference_profile['encoded_bytes'],
            'encoded_sha256': reference_profile['encoded_sha256'],
            'decoded_rows_bytes': len(reference_decoded),
            'decoded_rows_sha256': reference_sha,
            'group_count': reference_profile['group_count'],
            'sparse_row_count': reference_profile['sparse_row_count'],
            'nonzero_entries': reference_profile['nonzero_entries'] },
        'candidate': {
            'codec': candidate_profile['codec'],
            'encoded_bytes': candidate_profile['encoded_bytes'],
            'encoded_sha256': candidate_profile['encoded_sha256'],
            'decoded_rows_bytes': len(candidate_decoded),
            'decoded_rows_sha256': candidate_sha,
            'group_count': candidate_profile['group_count'],
            'sparse_row_count': candidate_profile['sparse_row_count'],
            'nonzero_entries': candidate_profile['nonzero_entries'] } }


def _u24le(data = None, offset = None):
    if offset < 0 or offset + 3 > len(data):
        raise Pr85BundleError(f'''uint24 read at {offset} exceeds buffer length {len(data)}''')
    return int.from_bytes(data[offset:offset + 3], 'little')


def _pack_u24le(value = None):
    if not  < 0, int(value) or 0, int(value) <= 16777215:
        pass
    
    raise Pr85BundleError(f'''segment length cannot be encoded as positive uint24: {value!r}''')
    return int(value).to_bytes(3, 'little')


def _slice(raw = None, lengths = None, *, header_bytes, fmt):
    pos = header_bytes
    offsets = { }
    segments = { }
    for name in SEGMENT_ORDER[:-1]:
        size = int(lengths[name])
        if size <= 0:
            raise Pr85BundleError(f'''{fmt} segment {name!r} has non-positive length {size}''')
        end = pos + size
        if end > len(raw):
            raise Pr85BundleError(f'''{fmt} segment {name!r} is truncated: end={end} raw={len(raw)}''')
        offsets[name] = pos
        segments[name] = raw[pos:end]
        pos = end
    if pos >= len(raw):
        raise Pr85BundleError(f'''{fmt} bundle is missing nonempty randmulti tail''')
    offsets['randmulti'] = pos
    segments['randmulti'] = raw[pos:]
    if header_bytes == 30:
        return Pr85Bundle(format = fmt, header_bytes = header_bytes, segments = segments, segment_offsets = offsets, fixed_length_segments = { })
    return None(format = Pr85Bundle, header_bytes = fmt, segments = header_bytes, segment_offsets = segments, fixed_length_segments = offsets(FIXED_V5_LENGTHS))


def parse_hpm1_mask_segment(segment = None):
    '''Parse PR91 ``HPM1`` mask bytes into a typed fail-closed contract.'''
    if not segment.startswith(HPM1_MAGIC):
        raise Pr85BundleError(f'''HPM1 mask segment has wrong magic: {segment[:4]!r}''')
    if len(segment) < HPM1_HEADER_BYTES:
        raise Pr85BundleError(f'''HPM1 mask segment is shorter than its {HPM1_HEADER_BYTES}-byte header''')
    (n_frames, height, width, predictor_count, delta, channels, use_spm, hpac_d_film, tokens_len, hpac_len, ppmd_order) = struct.unpack_from('<IIIIIIIIIII', segment, len(HPM1_MAGIC))
    token_start = HPM1_HEADER_BYTES
    token_end = token_start + int(tokens_len)
    hpac_end = token_end + int(hpac_len)
    if int(tokens_len) <= 0:
        raise Pr85BundleError('HPM1 token stream length must be positive')
    if int(hpac_len) <= 0:
        raise Pr85BundleError('HPM1 HPAC model length must be positive')
    if token_end > len(segment) or hpac_end > len(segment):
        raise Pr85BundleError(f'''HPM1 declared token/model lengths exceed segment bytes: tokens_end={token_end} hpac_end={hpac_end} segment={len(segment)}''')
    tail_bytes = len(segment) - hpac_end
    if tail_bytes != 0:
        raise Pr85BundleError(f'''HPM1 segment has unsupported trailing bytes: {tail_bytes}''')
    tokens = segment[token_start:token_end]
    hpac = segment[token_end:hpac_end]
    tokens_uint32_aligned = int(tokens_len) % 4 == 0
# WARNING: Decompyle incomplete


def infer_pr85_segment_contract(name = None, segment = None):
    '''Infer the local byte contract for a PR85-family segment.'''
    if name == 'mask' and segment.startswith(HPM1_MAGIC):
        return parse_hpm1_mask_segment(segment)
    if None == 'mask' and segment.startswith(b'QMA9'):
        codec = 'QMA9'
        magic = 'QMA9'
    elif name == 'model':
        codec = 'brotli_qh_model'
        magic = segment[:4].hex()
    elif name == 'pose':
        codec = 'brotli_p1d1_pose'
        magic = segment[:4].hex()
    elif name in QPOST_STREAM_NAMES:
        codec = 'brotli_qpost_sidechannel'
        magic = segment[:4].hex()
    else:
        codec = 'opaque'
        magic = segment[:4].hex()
    return Pr85SegmentContract(name = name, codec = codec, bytes = len(segment), sha256 = _sha256(segment), magic = magic, metadata = { })


def parse_pr85_bundle(raw = None):
    '''Parse PR85 v5 or explicit-30 bundle bytes.

    The public PR85 v5 grammar stores the first eight segment lengths in a
    24-byte header, then relies on fixed bias/region byte counts and assigns the
    remaining tail to randmulti. Some local recode experiments use an explicit
    30-byte header that also charges bias and region lengths. This parser
    accepts both and chooses explicit-30 only when the header is internally
    plausible.
    '''
    pass
# WARNING: Decompyle incomplete


def pack_pr85_bundle(segments = None, *, header_mode):
    \"\"\"Serialize validated PR85 segment bytes.

    ``header_mode='v5'`` preserves the public PR85 contract and requires fixed
    bias/region lengths. ``header_mode='explicit_30'`` writes ten uint24
    lengths and permits changed bias/region sizes for local recode candidates.
    \"\"\"
    pass
# WARNING: Decompyle incomplete


def validate_pr85_member_name(name = None):
    '''Validate the strict PR85 archive member contract.'''
    path = PurePosixPath(name)
    if name != 'x' and path.is_absolute() or '..' in path.parts:
        raise Pr85BundleError(f'''PR85 archive must contain exactly one safe member named \'x\', got {name!r}''')
    return name


def _read_vlq(data = None, cursor = None):
    value = 0
    shift = 0
    if cursor < len(data):
        byte = data[cursor]
        cursor += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return (value, cursor)
        None += 7
        if shift > 63:
            raise Pr85BundleError('truncated or overlong PR85 VLQ stream')
        if cursor < len(data):
            continue
    raise Pr85BundleError('truncated or overlong PR85 VLQ stream')


def _consume_sparse_randmulti_row(raw = None, cursor = None):
    start = cursor
    if cursor >= len(raw):
        raise Pr85BundleError('PR85 randmulti stream ended before count byte')
    count = int(raw[cursor])
    cursor += 1
    if count == 255:
        if cursor + 2 > len(raw):
            raise Pr85BundleError('PR85 randmulti extended count is truncated')
        count = int.from_bytes(raw[cursor:cursor + 2], 'little')
        cursor += 2
    previous = -1
    for _ in range(count):
        (delta, cursor) = _read_vlq(raw, cursor)
        previous += delta + 1
        if not previous < 0 and previous >= 600:
            continue
        raise Pr85BundleError(f'''PR85 randmulti sparse index out of range: {previous}''')
    values_end = cursor + count
    if values_end > len(raw):
        raise Pr85BundleError('PR85 randmulti value stream is truncated')
    return (values_end, count, values_end - start)


def _slice_pr85_randmulti_group_payloads(decoded = None):
    cursor = 0
    groups = []
    nonzero_entries = 0
    max_nonzero = 0
    row_count = 0
    for _height, _width, _amplitude, scount in enumerate(PR85_HEADERLESS_RANDMULTI_SPECS):
        group_start = cursor
        for _ in range(int(scount)):
            (cursor, count, _row_bytes) = _consume_sparse_randmulti_row(decoded, cursor)
            row_count += 1
            nonzero_entries += int(count)
            max_nonzero = max(max_nonzero, int(count))
        groups.append(decoded[group_start:cursor])
    if cursor != len(decoded):
        raise Pr85BundleError('PR85 randmulti stream has trailing bytes for v5 schedule')
    return (groups, {
        'decoded_bytes': len(decoded),
        'group_count': len(groups),
        'sparse_row_count': row_count,
        'nonzero_entries': nonzero_entries,
        'max_nonzero_entries_in_row': max_nonzero })


def transcode_pr85_randmulti_to_qrm1(encoded_randmulti = None):
    \"\"\"Wrap PR85's headerless sparse randmulti stream as runtime QRM1.

    The sparse row payloads are not reinterpreted or densified.  They are
    copied byte-for-byte behind explicit group ids so robust_current's reviewed
    QRM1 decoder can apply the full 72-group PR85 schedule.
    \"\"\"
    if encoded_randmulti.startswith(b'RMB1'):
        decoded = decode_rmb1_randmulti_payload(encoded_randmulti)
        source_codec = 'RMB1_bitmask_value_randmulti'
    else:
        decoded = _brotli_decompress(encoded_randmulti, 'randmulti')
        source_codec = 'brotli_headerless_sparse_randmulti'
    if decoded.startswith(b'QRM1'):
        return (encoded_randmulti, {
            'input_already_qrm1': True,
            'source_codec': 'brotli_qrm1_sparse_group_id_stream',
            'encoded_bytes': len(encoded_randmulti),
            'encoded_sha256': _sha256(encoded_randmulti),
            'decoded_bytes': len(decoded),
            'decoded_sha256': _sha256(decoded) })
    (groups, profile) = None(decoded)
    raw = bytearray(b'QRM1')
    raw.extend(len(groups).to_bytes(2, 'little'))
    for group_id, payload in enumerate(groups):
        raw.extend(int(group_id).to_bytes(2, 'little'))
        raw.extend(payload)
    encoded = _brotli_compress(bytes(raw), quality = 11, lgwin = 24)
    profile.update({
        'input_already_qrm1': False,
        'source_encoded_bytes': len(encoded_randmulti),
        'source_encoded_sha256': _sha256(encoded_randmulti),
        'source_codec': source_codec,
        'source_decoded_sha256': _sha256(decoded),
        'qrm1_raw_bytes': len(raw),
        'qrm1_raw_sha256': _sha256(raw),
        'encoded_bytes': len(encoded),
        'encoded_sha256': _sha256(encoded),
        'byte_delta_vs_source_encoded': len(encoded) - len(encoded_randmulti),
        'runtime_contract': 'QRM1_sparse_group_id_stream' })
    return (encoded, profile)


def build_pr85_qpost_bin(segments = None, *, transcode_randmulti_qrm1):
    '''Build robust_current ``qpost.bin`` bytes from PR85 side channels.'''
    pass
# WARNING: Decompyle incomplete


def decode_pr85_p1d1_pose_to_fp16(encoded_pose = None):
    '''Decode a Brotli(P1D1) pose segment to raw fp16 ``optimized_poses.bin``.'''
    raw = _brotli_decompress(encoded_pose, 'pose')
    if not raw.startswith(b'P1D1'):
        raise Pr85BundleError(f'''PR85 pose stream is not P1D1: {raw[:4]!r}''')
    cursor = 4
    if cursor >= len(raw):
        raise Pr85BundleError('P1D1 dimension count is missing')
    dim_count = int(raw[cursor])
    cursor += 1
    if dim_count <= 0 or dim_count > 6:
        raise Pr85BundleError(f'''P1D1 dimension count must be in [1,6], got {dim_count}''')
    dims = []
    lengths = []
    seen_dims = set()
    for _ in range(dim_count):
        if cursor + 3 > len(raw):
            raise Pr85BundleError('P1D1 dimension header is truncated')
        dim = int(raw[cursor])
        n_bytes = int.from_bytes(raw[cursor + 1:cursor + 3], 'little')
        cursor += 3
        if dim < 0 or dim >= 6:
            raise Pr85BundleError(f'''P1D1 dimension out of range: {dim}''')
        if dim in seen_dims:
            raise Pr85BundleError(f'''P1D1 duplicate dimension: {dim}''')
        if n_bytes <= 0:
            raise Pr85BundleError(f'''P1D1 dimension {dim} has non-positive stream length''')
        seen_dims.add(dim)
        dims.append(dim)
        lengths.append(n_bytes)
# WARNING: Decompyle incomplete


def expand_pr85_bundle_to_runtime_members(raw = None, *, transcode_randmulti_qrm1):
    '''Expand a PR85 ``x`` payload into robust_current logical members.

    The returned member map is intended for archive-builder/preflight use:
    ``masks.qma9`` is copied as charged QMA9 bytes, ``renderer.bin`` is the
    decoded QH0 renderer payload, ``optimized_poses.bin`` is raw fp16 decoded
    from P1D1, and ``qpost.bin`` is a QPS1 side-channel container.
    '''
    bundle = parse_pr85_bundle(raw)
    segments = bundle.segments
    renderer = _brotli_decompress(bytes(segments['model']), 'model')
    if not renderer.startswith((b'QH0', b'QH1')):
        raise Pr85BundleError(f'''PR85 model decoded to unexpected magic: {renderer[:4]!r}''')
    (poses, pose_meta) = decode_pr85_p1d1_pose_to_fp16(bytes(segments['pose']))
    (qpost, qpost_meta) = build_pr85_qpost_bin(segments, transcode_randmulti_qrm1 = transcode_randmulti_qrm1)
    members = {
        'masks.qma9': bytes(segments['mask']),
        'renderer.bin': renderer,
        'optimized_poses.bin': poses,
        'qpost.bin': qpost }
# WARNING: Decompyle incomplete


"""
