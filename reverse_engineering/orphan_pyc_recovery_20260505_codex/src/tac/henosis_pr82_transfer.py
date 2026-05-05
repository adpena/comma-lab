"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``37:22: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``henosis_pr82_transfer.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/henosis_pr82_transfer.py'
__recovery_spec__ = 'henosis_pr82_transfer.recovery_spec.json'
__recovery_ast_error__ = '37:22: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: henosis_pr82_transfer.cpython-312.pyc (Python 3.12)

'''PR82/Henosis atom-transfer helpers.

The helpers in this module are intentionally scorer-free.  They parse the
public PR82 compact bundle, expose deterministic per-pair activity summaries,
and build runtime-compatible ``QPS1`` postprocess sidecars for local archive
screening.
'''
from __future__ import annotations
import ast
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import brotli
import numpy as np
QPOST_MAGIC = b'QPS1'
QPOST_STREAM_NAMES = ('post', 'shift', 'frac', 'frac2', 'frac3', 'bias', 'region', 'randmulti')
PR82_HEADER_STREAM_NAMES = ('mask', 'model', 'pose', 'post', 'shift', 'frac', 'frac2', 'frac3')
QPOST_DEFAULTS = {
    'post': 0,
    'shift': 40,
    'frac': 4,
    'frac2': 4,
    'frac3': 4,
    'bias': 13,
    'region': 0,
    'randmulti': 0 }

class HenosisPr82TransferError(ValueError):
    '''Raised when PR82 atom parsing or transfer fails a closed guard.'''
    pass

Pr82ReplayContract = <NODE:12>()
Pr82Bundle = <NODE:12>()
Pr82RandmultiGroup = <NODE:12>()
PR82_RANDMULTI_SPECIAL_SEMANTICS: 'dict[tuple[int, int, int], str]' = {
    (224, 222, 4): 'replay_special_f2_tile_bias_2x2_channel_radius4',
    (223, 223, 4): 'replay_special_f2_boundary_all_channel_radius4',
    (223, 222, 2): 'replay_special_f2_class_conditioned_channel_radius2',
    (223, 224, 4): 'replay_special_f2_boundary_class_channel_radius4',
    (223, 219, 4): 'replay_special_f2_width2_boundary_channel_class_radius4',
    (223, 218, 4): 'replay_special_f2_width3_boundary_channel_class_radius4',
    (223, 221, 4): 'replay_special_f2_class_conditioned_channel_radius4',
    (222, 223, 4): 'replay_special_f2_class_conditioned_all_channel_radius4',
    (222, 222, 4): 'replay_special_f2_global_rgb_bias_radius4' }

def sha256_bytes(data = dataclass(frozen = True)):
    return hashlib.sha256(data).hexdigest()


def sha256_path(path = None):
    pass
# WARNING: Decompyle incomplete


def parse_replay_contract(path = None):
    '''Parse fixed tail lengths and randmulti specs from replay ``inflate.py``.'''
    pass
# WARNING: Decompyle incomplete


def parse_pr82_bundle(raw = None, contract = None):
    '''Split a PR82 compact ``x`` payload into encoded stream bytes.'''
    if len(raw) < 24:
        raise HenosisPr82TransferError('PR82 payload is too short for 8x u24 header')
# WARNING: Decompyle incomplete


def brotli_decompress_segment(encoded = None, name = None):
    
    try:
        return brotli.decompress(encoded)
    except brotli.error:
        exc = None
        raise HenosisPr82TransferError(f'''PR82 segment {name!r} is not Brotli-decodable'''), exc
        exc = None
        del exc



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
            raise HenosisPr82TransferError('truncated or overlong VLQ stream')
        if cursor < len(data):
            continue
    raise HenosisPr82TransferError('truncated or overlong VLQ stream')


def _vlq_indices_values(raw = None, cursor = None, count = None):
    idx = -1
    indices = []
    for _ in range(count):
        (delta, cursor) = _read_vlq(raw, cursor)
        idx += delta + 1
        if idx < 0 or idx >= 600:
            raise HenosisPr82TransferError(f'''sparse qpost index out of range: {idx}''')
        indices.append(idx)
    end = cursor + count
    if end > len(raw):
        raise HenosisPr82TransferError('sparse qpost value stream is truncated')
    values = np.frombuffer(raw, dtype = np.uint8, count = count, offset = cursor).astype(np.uint8)
    return (indices, values, end)


def randmulti_semantic_label(height = None, width = None, amplitude = None):
    return PR82_RANDMULTI_SPECIAL_SEMANTICS.get((int(height), int(width), int(amplitude)), 'generic_frame0_nearest_random_pattern')


def randmulti_group_qps1_nm2_compatible(group = None):
    '''Whether current QPS1 ``NM2`` can represent and replay this group exactly.'''
    if randmulti_semantic_label(group.height, group.width, group.amplitude) != 'generic_frame0_nearest_random_pattern':
        return False
    return (lambda .0: pass# WARNING: Decompyle incomplete
)((group.height, group.width, group.amplitude, group.scount)())


def _decode_randmulti_rows(raw = None, cursor = None, scount = None):
    rows = np.zeros((int(scount), 600), dtype = np.uint8)
    for row_index in range(int(scount)):
        if cursor >= len(raw):
            raise HenosisPr82TransferError('randmulti stream ended before count byte')
        count = int(raw[cursor])
        cursor += 1
        if count == 255:
            if cursor + 2 > len(raw):
                raise HenosisPr82TransferError('randmulti extended count is truncated')
            count = int.from_bytes(raw[cursor:cursor + 2], 'little')
            cursor += 2
        (indices, values, cursor) = _vlq_indices_values(raw, cursor, count)
        if not count:
            continue
        rows[(row_index, np.asarray(indices, dtype = np.int64))] = values
    return (rows, cursor)


def _write_vlq(value = None):
    if value < 0:
        raise HenosisPr82TransferError(f'''cannot VLQ-encode negative value: {value}''')
    out = bytearray()
    byte = int(value) & 127
    value >>= 7
    if value:
        out.append(byte | 128)
    else:
        out.append(byte)
        return bytes(out)


def _encode_randmulti_rows(rows = None):
    if rows.ndim != 2 or rows.shape[1] != 600:
        raise HenosisPr82TransferError(f'''randmulti rows must have shape (scount, 600), got {rows.shape}''')
    out = bytearray()
    for row in rows.astype(np.uint8, copy = False):
        indices = np.flatnonzero(row)
        count = int(indices.size)
        if count > 65535:
            raise HenosisPr82TransferError(f'''randmulti sparse row has too many choices: {count}''')
        previous = -1
        for index in indices:
            out.extend(_write_vlq(int(index) - previous - 1))
            previous = int(index)
        out.extend(row[indices].astype(np.uint8, copy = False).tobytes())
    return bytes(out)


def decode_randmulti_groups(encoded = None, specs = None):
    \"\"\"Decode PR82 headerless sparse randmulti rows.

    PR82's replay runtime has a hard-coded 72-group spec table.  This decoder
    keeps that table out of the archive bytes and validates that the sparse
    stream closes exactly under the supplied replay contract.
    \"\"\"
    raw = brotli_decompress_segment(encoded, 'randmulti')
    if raw[:3] in frozenset({b'NM1', b'NM2'}):
        raise HenosisPr82TransferError('PR82 randmulti deconstruction expects headerless sparse payload')
    cursor = 0
    groups = []
# WARNING: Decompyle incomplete


def encode_randmulti_qrm1(groups = None, *, pair_indices):
    \"\"\"Encode PR82-native sparse randmulti groups as a charged ``QRM1`` stream.

    ``QRM1`` is the minimal self-describing extension needed for PR82-native
    groups: the archive carries explicit replay group ids and sparse rows,
    while the runtime supplies the reviewed PR82 group semantics for those ids.
    It avoids ``NM2``'s u8 dimension limit and keeps the original sparse
    economics instead of expanding large groups to dense 600-column rows.
    \"\"\"
    if len(groups) > 65535:
        raise HenosisPr82TransferError('QRM1 supports at most 65535 randmulti groups')
    keep = None
# WARNING: Decompyle incomplete


def decode_randmulti_qrm1(encoded = None, specs = None):
    '''Decode charged ``QRM1`` sparse randmulti groups using PR82 replay specs.'''
    raw = brotli_decompress_segment(encoded, 'randmulti')
    if raw[:4] != b'QRM1':
        raise HenosisPr82TransferError(f'''randmulti stream is not QRM1: {raw[:4]!r}''')
    if len(raw) < 6:
        raise HenosisPr82TransferError('QRM1 stream is truncated')
    cursor = 4
    group_count = int.from_bytes(raw[cursor:cursor + 2], 'little')
    cursor += 2
    groups = []
    seen = set()
# WARNING: Decompyle incomplete


def randmulti_qrm1_parity_profile(original_groups = None, decoded_groups = None, *, encoded, source_encoded):
    '''Summarize local group-row parity for a ``QRM1`` encoded stream.'''
    pass
# WARNING: Decompyle incomplete


def randmulti_group_summary(group = None):
    sparse_choice_total = int(np.count_nonzero(group.rows))
    semantic = randmulti_semantic_label(group.height, group.width, group.amplitude)
    return {
        'amplitude': int(group.amplitude),
        'group_index': int(group.group_index),
        'height': int(group.height),
        'nonzero_choice_total': sparse_choice_total,
        'payload_bytes': int(group.payload_bytes),
        'qps1_nm2_runtime_compatible': randmulti_group_qps1_nm2_compatible(group),
        'scount': int(group.scount),
        'semantic': semantic,
        'width': int(group.width) }


def encode_randmulti_nm2(groups = None, *, pair_indices):
    '''Encode current-runtime-compatible generic randmulti groups as ``NM2``.'''
    if len(groups) > 255:
        raise HenosisPr82TransferError('NM2 supports at most 255 randmulti groups')
    keep = None
# WARNING: Decompyle incomplete


def _decode_dense_or_delta(raw = None, *, full, delta, default, sparse):
    magic = raw[:3]
    if magic == full:
        arr = np.frombuffer(raw, dtype = np.uint8, offset = 3).copy()
    elif magic == delta:
        encoded = np.frombuffer(raw, dtype = np.uint8, offset = 3).astype(np.int64)
        arr = np.where(encoded == 0, default, encoded - 1).astype(np.uint8)
# WARNING: Decompyle incomplete


def decode_postprocess(encoded = None):
    raw = brotli_decompress_segment(encoded, 'post')
    if raw[:4] == b'PCD1':
        cursor = 5
        stages = []
        for _ in range(raw[4]):
            if cursor + 3 > len(raw):
                raise HenosisPr82TransferError('PCD1 postprocess header is truncated')
            _stage_id = raw[cursor]
            n = struct.unpack_from('<H', raw, cursor + 1)[0]
            cursor += 3
            choices = np.frombuffer(raw, dtype = np.uint8, count = n, offset = cursor).copy()
            cursor += n
            if choices.shape != (600,):
                raise HenosisPr82TransferError('PCD1 postprocess stage does not contain 600 choices')
            stages.append(choices)
        if cursor != len(raw):
            raise HenosisPr82TransferError('PCD1 postprocess has trailing bytes')
        return np.stack(stages)
    if None(raw) % 600 != 0 or len(raw) // 600 not in (3, 4):
        raise HenosisPr82TransferError('headerless postprocess must be 3 or 4 stages of 600 choices')
    return np.frombuffer(raw, dtype = np.uint8).copy().reshape(len(raw) // 600, 600)


def decode_control_arrays(encoded_segments = None):
    '''Decode runtime-compatible PR82 qpost controls to per-pair arrays.'''
    post = decode_postprocess(encoded_segments['post'])
    shift = _decode_dense_or_delta(brotli_decompress_segment(encoded_segments['shift'], 'shift'), full = b'SH4', delta = b'SD4', default = 40)
    frac = _decode_dense_or_delta(brotli_decompress_segment(encoded_segments['frac'], 'frac'), full = b'FH1', delta = b'FD1', sparse = b'FV1', default = 4)
    frac2 = _decode_dense_or_delta(brotli_decompress_segment(encoded_segments['frac2'], 'frac2'), full = b'FH2', delta = b'FD2', default = 4)
    frac3 = _decode_dense_or_delta(brotli_decompress_segment(encoded_segments['frac3'], 'frac3'), full = b'FH3', delta = b'FD3', default = 4)
    bias = _decode_dense_or_delta(brotli_decompress_segment(encoded_segments['bias'], 'bias'), full = b'BH1', delta = b'BD1', sparse = b'BV1', default = 13)
    region = _decode_dense_or_delta(brotli_decompress_segment(encoded_segments['region'], 'region'), full = b'RH1', delta = b'RD1', sparse = b'RV1', default = 0)
    return {
        'post': post,
        'shift': shift,
        'frac': frac,
        'frac2': frac2,
        'frac3': frac3,
        'bias': bias,
        'region': region }


def decode_randmulti_activity(encoded = None, specs = None):
    '''Return per-pair nonzero counts for PR82 headerless randmulti.'''
    raw = brotli_decompress_segment(encoded, 'randmulti')
    counts_by_pair = np.zeros(600, dtype = np.int32)
    decoded_groups = decode_randmulti_groups(encoded, specs)
    groups = []
    for group in decoded_groups:
        rows = group.rows
        counts_by_pair += np.count_nonzero(rows, axis = 0).astype(np.int32)
        groups.append(randmulti_group_summary(group))
# WARNING: Decompyle incomplete


def summarize_pair_activity(arrays = None, *, randmulti_counts):
    rows = []
# WARNING: Decompyle incomplete


def rank_pairs_by_activity(pair_rows = None):
    pass
# WARNING: Decompyle incomplete


def _keep_mask(pair_indices = None):
    keep = np.zeros(600, dtype = bool)
    for pair in pair_indices:
        if pair < 0 or pair >= 600:
            raise HenosisPr82TransferError(f'''pair index out of range: {pair}''')
        keep[int(pair)] = True
    return keep


def filter_qpost_streams_to_pairs(encoded_segments = None, pair_indices = None, *, include_streams):
    '''Build QPS1 stream blobs where non-selected pairs decode to identity.'''
    unknown = sorted(set(include_streams) - set(QPOST_STREAM_NAMES))
    if unknown:
        raise HenosisPr82TransferError(f'''unknown qpost stream(s): {unknown}''')
    if 'randmulti' in include_streams:
        raise HenosisPr82TransferError('PR82 randmulti has 72 replay groups and is not QPS1-runtime-compatible')
    arrays = decode_control_arrays(encoded_segments)
    keep = _keep_mask(pair_indices)
    include = set(include_streams)
    out = { }
    for name in QPOST_STREAM_NAMES:
        if name not in include:
            out[name] = b''
            continue
        if name == 'post':
            arr = arrays[name].copy()
            arr[(:, ~keep)] = 0
            out[name] = brotli.compress(arr.astype(np.uint8, copy = False).tobytes(), quality = 11)
            continue
        if name == 'shift':
            arr = arrays[name].copy()
            arr[~keep] = 40
            out[name] = brotli.compress(b'SH4' + arr.astype(np.uint8, copy = False).tobytes(), quality = 11)
            continue
        if name in frozenset({'frac', 'frac2', 'frac3'}):
            arr = arrays[name].copy()
            arr[~keep] = 4
            magic = {
                'frac': b'FH1',
                'frac2': b'FH2',
                'frac3': b'FH3' }[name]
            out[name] = brotli.compress(magic + arr.astype(np.uint8, copy = False).tobytes(), quality = 11)
            continue
        if name == 'bias':
            arr = arrays[name].copy()
            arr[~keep] = 13
            out[name] = brotli.compress(b'BH1' + arr.astype(np.uint8, copy = False).tobytes(), quality = 11)
            continue
        if not name == 'region':
            continue
        arr = arrays[name].copy()
        arr[~keep] = 0
        out[name] = brotli.compress(b'RH1' + arr.astype(np.uint8, copy = False).tobytes(), quality = 11)
    return out


def encode_qpost(streams = None):
    pass
# WARNING: Decompyle incomplete


def qpost_stream_summary(streams = None, original = None):
    pass
# WARNING: Decompyle incomplete


def decode_pr82_p1d1_pose(encoded_pose = None):
    raw = brotli_decompress_segment(encoded_pose, 'pose')
    if not raw.startswith(b'P1D1'):
        raise HenosisPr82TransferError(f'''PR82 pose stream is not P1D1: {raw[:4]!r}''')
    cursor = 4
    dim_count = raw[cursor]
    cursor += 1
    dims = []
    lengths = []
    for _ in range(dim_count):
        if cursor + 3 > len(raw):
            raise HenosisPr82TransferError('P1D1 dimension header is truncated')
        dims.append(int(raw[cursor]))
        lengths.append(int.from_bytes(raw[cursor + 1:cursor + 3], 'little'))
        cursor += 3
    pose = np.zeros((600, 6), dtype = np.float32)
    for dim, n_bytes in zip(dims, lengths):
        stream = raw[cursor:cursor + n_bytes]
        if len(stream) != n_bytes:
            raise HenosisPr82TransferError('P1D1 dimension stream is truncated')
        cursor += n_bytes
        values = []
        pos = 0
        if pos < len(stream):
            (zz, pos) = _read_vlq(stream, pos)
            delta = zz >> 1 ^ -(zz & 1)
            previous = values[-1] if values else 0
            values.append(previous + delta)
            if pos < len(stream):
                continue
        if len(values) != 600:
            raise HenosisPr82TransferError(f'''P1D1 dim {dim} decoded {len(values)} values, expected 600''')
        q = np.asarray(values, dtype = np.int32)
        if dim == 0:
            pose[(:, 0)] = q.astype(np.float32) / 512 + 20
            continue
        pose[(:, dim)] = q.clip(-32768, 32767).astype(np.int16).astype(np.float32) / 2048
    if cursor != len(raw):
        raise HenosisPr82TransferError('P1D1 pose stream has trailing bytes')
    return pose


def pose_velocity_atom_ranking(source_pose = None, pr82_pose = None):
    if source_pose.shape[0] != 600 or pr82_pose.shape[0] != 600:
        raise HenosisPr82TransferError('pose atom ranking expects 600 pose rows')
    delta_q = np.rint((pr82_pose[(:, 0)] - source_pose[(:, 0)]) * 512).astype(np.int64)
# WARNING: Decompyle incomplete


"""
