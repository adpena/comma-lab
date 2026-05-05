# pyc-recovery: STUB unreconstructible -- see .recovery_spec.json for dis() ground-truth
# pycdc could not produce parseable output; raw decompiled text preserved in _PYCDC_PARTIAL_OUTPUT below.
"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``42:1: cannot assign to literal``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``apply_qzs3_postprocess.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'submissions/robust_current/apply_qzs3_postprocess.py'
__recovery_spec__ = 'apply_qzs3_postprocess.recovery_spec.json'
__recovery_ast_error__ = '42:1: cannot assign to literal'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: apply_qzs3_postprocess.cpython-312.pyc (Python 3.12)

'''Apply counted qpose/QZS3 postprocess atoms to inflated raw frames.

This is a submission-runtime helper.  It does not load PoseNet or SegNet; it
only applies deterministic, archive-carried image transforms after the normal
renderer has written raw RGB frames.

``qpost.bin`` format:

    b"QPS1" + uint32[8] little-endian lengths + concatenated streams

The eight streams mirror the public PR65/henosis postprocess bundle:
post-code stages, integer frame-0 shifts, 0.5/0.25/0.125px frame-1 shifts,
frame-1 RGB bias, frame-1 regional bias, and multiscale random patterns.
Each stream remains Brotli-compressed inside ``qpost.bin`` and is charged as
archive bytes.

``randmulti`` additionally accepts ``QRM1`` streams emitted by
``tac.henosis_pr82_transfer.encode_randmulti_qrm1``.  QRM1 carries PR82 replay
group ids plus sparse per-pair rows.  This runtime supports generic random
patterns plus the PR82 tile/global and source-mask-conditioned frame-1 bias
branches.  Source-mask-conditioned branches fail closed at apply time unless
the charged archive mask stream can be loaded from the archive directory.
'''
from __future__ import annotations
import argparse
import importlib.util as importlib
import json
import os
import struct
import sys
import zipfile
from pathlib import Path
from typing import NamedTuple
import brotli
import numpy as np
import torch

functional
b'QPS1' = import torch.nn.functional, nn
STREAM_NAMES = ('post', 'shift', 'frac', 'frac2', 'frac3', 'bias', 'region', 'randmulti')
HEADER = '<' + 'I' * len(STREAM_NAMES)
PR82_QRM1_RANDMULTI_SPECS: 'tuple[tuple[int, int, int, int], ...]' = ((24, 32, 1, 12), (12, 16, 1, 1), (6, 8, 1, 1), (3, 4, 1, 1), (2, 2, 1, 1), (8, 8, 1, 1), (4, 4, 1, 1), (4, 8, 1, 1), (2, 4, 1, 1), (2, 8, 1, 1), (1, 2, 1, 1), (1, 4, 1, 1), (2, 1, 1, 1), (4, 1, 1, 1), (8, 1, 1, 1), (1, 8, 1, 1), (16, 1, 1, 1), (1, 16, 1, 1), (32, 1, 1, 1), (64, 1, 1, 1), (256, 1, 1, 1), (1024, 1, 1, 1), (2048, 1, 1, 1), (4096, 1, 1, 1), (8192, 1, 1, 1), (8192, 1, 1, 1), (16384, 1, 1, 1), (32768, 1, 1, 1), (65536, 1, 1, 1), (131072, 1, 1, 1), (262144, 1, 1, 1), (524288, 1, 1, 1), (1048576, 1, 1, 1), (874, 1, 1, 1), (874, 1, 1, 1), (2097152, 1, 1, 1), (875, 1, 1, 1), (876, 1, 1, 1), (877, 1, 1, 1), (1164, 1, 1, 1), (878, 1, 1, 1), (879, 1, 1, 1), (880, 1, 1, 1), (881, 1, 1, 1), (882, 1, 1, 1), (512, 2, 1, 1), (256, 2, 1, 1), (128, 2, 1, 1), (64, 2, 1, 1), (32, 2, 1, 1), (16, 2, 1, 1), (8, 2, 1, 1), (4, 2, 1, 1), (4, 4, 1, 1), (8, 4, 1, 1), (16, 4, 1, 1), (32, 4, 1, 1), (64, 4, 1, 1), (128, 4, 1, 1), (64, 8, 1, 1), (32, 8, 1, 1), (222, 222, 4, 1), (222, 223, 4, 1), (223, 222, 2, 1), (223, 223, 4, 1), (223, 221, 4, 1), (223, 224, 4, 1), (223, 221, 4, 1), (223, 219, 4, 1), (64, 16, 1, 1), (223, 218, 4, 1), (224, 222, 4, 1))
QRM1_SUPPORTED_SPECIAL_MAX_CHOICE = {
    (222, 222, 4): 32,
    (222, 223, 4): 40,
    (223, 222, 2): 60,
    (223, 223, 4): 48,
    (223, 221, 4): 120,
    (223, 224, 4): 144,
    (223, 219, 4): 144,
    (223, 218, 4): 144,
    (224, 222, 4): 96 }
QRM1_SOURCE_MASK_PR82_SPECIAL_RANDMULTI = {
    (222, 223, 4): 'class-conditioned all-channel bias',
    (223, 222, 2): 'class-conditioned channel bias',
    (223, 223, 4): 'boundary all-channel bias',
    (223, 221, 4): 'class-conditioned channel bias',
    (223, 224, 4): 'boundary/class-boundary channel bias',
    (223, 219, 4): 'width-2 boundary channel/class bias',
    (223, 218, 4): 'width-3 boundary channel/class bias' }

class QPostState(NamedTuple):
    f1_randmulti: 'list[tuple[torch.Tensor, int, int, int]] | None' = 'QPostState'


def _qrm1_support_reason(lh = None, lw = None, amp = None):
    return (True, None)


def classify_qrm1_randmulti_stream(blob = None):
    '''Classify a Brotli-compressed QRM1 randmulti stream for runtime support.

    This is a fail-closed preflight helper for PR82-derived candidates.  It
    does not relax runtime validation and does not apply any transform; it
    proves which carried PR82 group ids are locally supported by this raw-frame
    postprocess helper and which require the source mask tensor from the public
    PR82 replay path.
    '''
    pass
# WARNING: Decompyle incomplete


def classify_qpost_qrm1_support(qpost_path = None):
    raw = qpost_path.read_bytes()
    header_size = len(MAGIC) + struct.calcsize(HEADER)
    if len(raw) < header_size or raw[:4] != MAGIC:
        raise ValueError(f'''bad qpost magic in {qpost_path}''')
    lengths = struct.unpack_from(HEADER, raw, len(MAGIC))
    pos = header_size
    streams = { }
    for name, n in zip(STREAM_NAMES, lengths):
        end = pos + int(n)
        if end > len(raw):
            raise ValueError(f'''qpost stream {name} overruns payload''')
        streams[name] = raw[pos:end]
        pos = end
    if pos != len(raw):
        raise ValueError(f'''qpost has {len(raw) - pos} trailing bytes''')
    if not streams['randmulti']:
        return {
            'contract': 'no_randmulti',
            'dispatchable_qrm1': True,
            'supported_group_ids': [],
            'unsupported_group_ids': [] }
    return None(streams['randmulti'])


def classify_archive_qrm1_support(archive_path = None):
    \"\"\"Return QRM1 support classification for a candidate archive's qpost.bin.\"\"\"
    zf = zipfile.ZipFile(archive_path, 'r')
    names = (lambda .0: pass# WARNING: Decompyle incomplete
)(zf.infolist()())
    if names.count('qpost.bin') > 1:
        raise ValueError(f'''{archive_path} contains duplicate qpost.bin members''')
    if 'qpost.bin' not in names:
        None(None, None)
        return 
    None(None, None)
    archive_path.with_suffix(archive_path.suffix + '.qpost.inspect.tmp') = sorted.read('qpost.bin')
# WARNING: Decompyle incomplete


def _post_pair_tensor(value = None, default = None, device = None):
    pass
# WARNING: Decompyle incomplete


def _B(r, g, b = (0, 0, 0)):
    return (float(r), float(g), float(b))


def _PB(f0, f1 = ((0, 0, 0), (0, 0, 0))):
    pass
# WARNING: Decompyle incomplete


def _post_defs(stage_id = None):
    if stage_id == 1:
        return [
            (None, None),
            (None, _B(2, 0, 0)),
            (None, _B(1, 1, 1)),
            (None, _B(0, 0, -2)),
            (None, _B(0, 0, 2)),
            (None, _B(-1, -1, -1)),
            (None, _B(-2, 0, 0)),
            (None, _B(2, 2, 2)),
            (None, _B(0, -2, 0)),
            (None, _B(0, 2, 0)),
            (_B(1.01, 1.01, 1.01), None),
            (_B(0.99, 0.99, 0.99), None)]
    if None == 2:
        defs = [
            (None, None)]
        for val in (-4, -3, -2, -1, 1, 2, 3, 4):
            defs += [
                (None, _B(val, val, val)),
                (None, _B(val, 0, 0)),
                (None, _B(0, val, 0)),
                (None, _B(0, 0, val))]
        for frame in (0, 1):
            for val in (-2, -1, 1, 2):
                for _chan, ci in (('all', -1), ('r', 0), ('g', 1), ('b', 2)):
                    f1 = [
                        0,
                        0,
                        0]
                    f0 = [
                        0,
                        0,
                        0]
                    target = f0 if frame == 0 else f1
                    defs.append((None, _PB(f0, f1)))
        return defs
    if None == 3:
        defs = [
            (None, None)]
        for r in (-2, -1, 0, 1, 2):
            for g in (-2, -1, 0, 1, 2):
                for b in (-2, -1, 0, 1, 2):
                    if not (r, g, b) != (0, 0, 0):
                        continue
                    defs.append((None, _PB((r, g, b), (0, 0, 0))))
        return defs
    if None == 4:
        defs = [
            (None, None)]
        for r in (-1, 0, 1):
            for g in (-1, 0, 1):
                for b in (-1, 0, 1):
                    if not (r, g, b) != (0, 0, 0):
                        continue
                    defs.append((None, _PB((r, g, b), (0, 0, 0))))
        return defs
    raise None(f'''unknown compact post stage id {stage_id}''')


def _post_stage_from_defs(defs = None, choices_data = None, device = None):
    pass
# WARNING: Decompyle incomplete


def _load_post_codes(blob = None, device = None):
    if not blob:
        return None
    raw = brotli.decompress(blob)
    stages = []
    if raw[:4] == b'PCD1':
        pos = 4
        stage_count = raw[pos]
        pos += 1
        for _ in range(stage_count):
            stage_id = raw[pos]
            pos += 1
            n = struct.unpack_from('<H', raw, pos)[0]
            pos += 2
            choices = raw[pos:pos + n]
            pos += n
            stages.append(_post_stage_from_defs(_post_defs(stage_id), choices, device))
    else:
        pairs_per_file = 600
        if len(raw) % pairs_per_file != 0:
            raise ValueError('bad headerless post_codes length')
        stage_count = len(raw) // pairs_per_file
        if stage_count not in (3, 4):
            raise ValueError(f'''bad headerless post_codes stage count {stage_count}''')
        pos = 0
        for stage_id in range(1, stage_count + 1):
            choices = raw[pos:pos + pairs_per_file]
            pos += pairs_per_file
            stages.append(_post_stage_from_defs(_post_defs(stage_id), choices, device))
    if not stages:
        stages


def _decode_dense_or_delta(blob = None, *, magic_full, magic_delta, default, center, device):
    if not blob:
        return None
    raw = brotli.decompress(blob)
    magic = raw[:3]
    if magic == magic_full:
        arr = np.frombuffer(raw, dtype = np.uint8, offset = 3).astype(np.int64)
    elif magic == magic_delta:
        d = np.frombuffer(raw, dtype = np.uint8, offset = 3).astype(np.int64)
        arr = np.where(d == 0, default, d - 1).astype(np.int64)
# WARNING: Decompyle incomplete


def _decode_frac(blob = None, device = None):
    if not blob:
        return None
    raw = brotli.decompress(blob)
    magic = raw[:3]
    if magic == b'FH1':
        arr = np.frombuffer(raw, dtype = np.uint8, offset = 3).astype(np.int64)
    elif magic == b'FV1':
        cnt = int.from_bytes(raw[3:5], 'little')
        pos = 5
        arr = np.full(600, 4, dtype = np.int64)
        idx = -1
        inds = []
        for _ in range(cnt):
            acc = 0
            sh = 0
            by = raw[pos]
            pos += 1
            acc |= (by & 127) << sh
            if by & 128:
                sh += 7
            
        continue
        idx += acc + 1
        inds.append(idx)
        continue
        vals = np.frombuffer(raw, dtype = np.uint8, count = cnt, offset = pos).astype(np.int64)
        for ii, vv in zip(inds, vals):
            arr[ii] = vv - 1
    else:
        raise ValueError('bad f1 fractional shift payload')
    return torch.from_numpy(arr).to(device)


def _decode_region(blob = None, device = None):
    if not blob:
        return None
    raw = brotli.decompress(blob)
    magic = raw[:3]
    if magic == b'RH1':
        arr = np.frombuffer(raw, dtype = np.uint8, offset = 3).astype(np.int64)
    elif magic == b'RD1':
        d = np.frombuffer(raw, dtype = np.uint8, offset = 3).astype(np.int64)
        arr = np.where(d == 0, 0, d - 1).astype(np.int64)
    elif magic == b'RV1':
        cnt = int.from_bytes(raw[3:5], 'little')
        pos = 5
        arr = np.zeros(600, dtype = np.int64)
        idx = -1
        inds = []
        for _ in range(cnt):
            acc = 0
            sh = 0
            by = raw[pos]
            pos += 1
            acc |= (by & 127) << sh
            if by & 128:
                sh += 7
            
        continue
        idx += acc + 1
        inds.append(idx)
        continue
        vals = np.frombuffer(raw, dtype = np.uint8, count = cnt, offset = pos).astype(np.int64)
        for ii, vv in zip(inds, vals):
            arr[ii] = vv - 1
    else:
        raise ValueError('bad f1 region-bias payload')
    return torch.from_numpy(arr).to(device)


def _read_vlq(data = None, pos = None):
    value = 0
    shift = 0
    if pos < len(data):
        by = data[pos]
        pos += 1
        value |= (by & 127) << shift
        if by < 128:
            return (value, pos)
        None += 7
        if shift > 63:
            raise ValueError('truncated or overlong randmulti VLQ')
        if pos < len(data):
            continue
    raise ValueError('truncated or overlong randmulti VLQ')


def _decode_sparse_randmulti_rows(raw = None, pos = None, scount = None):
    rows = np.zeros((int(scount), 600), dtype = np.uint8)
    for si in range(int(scount)):
        if pos >= len(raw):
            raise ValueError('randmulti stream ended before count byte')
        cnt = int(raw[pos])
        pos += 1
        if cnt == 255:
            if pos + 2 > len(raw):
                raise ValueError('randmulti extended count is truncated')
            cnt = int.from_bytes(raw[pos:pos + 2], 'little')
            pos += 2
        idx = -1
        inds = []
        for _ in range(cnt):
            (acc, pos) = _read_vlq(raw, pos)
            idx += acc + 1
            if idx < 0 or idx >= 600:
                raise ValueError(f'''randmulti sparse index out of range: {idx}''')
            inds.append(idx)
        end = pos + cnt
        if end > len(raw):
            raise ValueError('randmulti value stream is truncated')
        vals = np.frombuffer(raw, dtype = np.uint8, count = cnt, offset = pos)
        pos = end
        if not cnt:
            continue
        rows[(si, np.array(inds, dtype = np.int64))] = vals
    return (rows, pos)


def _validate_qrm1_group_choices(rows = None, lh = None, lw = None, amp = ('rows', 'np.ndarray', 'lh', 'int', 'lw', 'int', 'amp', 'int', 'return', 'None')):
    spec_key = (int(lh), int(lw), int(amp))
    (supported, reason) = _qrm1_support_reason(lh, lw, amp)
    if supported and int(np.count_nonzero(rows)) > 0:
        raise ValueError(f'''unsupported QRM1 PR82 randmulti branch {spec_key}: {reason}''')
    max_choice = QRM1_SUPPORTED_SPECIAL_MAX_CHOICE.get(spec_key)
# WARNING: Decompyle incomplete


def _randmulti_requires_source_masks(state = None):
    pass
# WARNING: Decompyle incomplete


def qpost_requires_source_masks(state = None):
    '''Whether this parsed qpost needs source masks for non-noop randmulti.'''
    return _randmulti_requires_source_masks(state)


def _decode_qrm1_randmulti(raw = None, device = None):
    if len(raw) < 6:
        raise ValueError('QRM1 randmulti stream is truncated')
    pos = 4
    gcount = int.from_bytes(raw[pos:pos + 2], 'little')
    pos += 2
    out = []
    seen = set()
    for _ in range(gcount):
        if pos + 2 > len(raw):
            raise ValueError('QRM1 randmulti group id is truncated')
        group_id = int.from_bytes(raw[pos:pos + 2], 'little')
        pos += 2
        if group_id in seen:
            raise ValueError(f'''QRM1 duplicate randmulti group id: {group_id}''')
        seen.add(group_id)
        if group_id >= len(PR82_QRM1_RANDMULTI_SPECS):
            raise ValueError(f'''QRM1 randmulti group id outside PR82 replay specs: {group_id}''')
        (lh, lw, amp, scount) = PR82_QRM1_RANDMULTI_SPECS[group_id]
        (rows, pos) = _decode_sparse_randmulti_rows(raw, pos, scount)
        _validate_qrm1_group_choices(rows, lh, lw, amp)
        out.append((torch.from_numpy(rows.astype(np.int64)).to(device), lh, lw, amp))
    if pos != len(raw):
        raise ValueError('QRM1 randmulti stream has trailing bytes')
    if not out:
        out


def _decode_rmb1_randmulti_payload(encoded = None):
    \"\"\"Decode PR92's charged RMB1 bitmask+value randmulti container.\"\"\"
    if len(encoded) < 6 or encoded[:4] != b'RMB1':
        raise ValueError('bad RMB1 randmulti payload')
    mask_len = int.from_bytes(encoded[4:6], 'little')
    mask_br = encoded[6:6 + mask_len]
    vals_br = encoded[6 + mask_len:]
    if not mask_br or vals_br:
        raise ValueError('truncated RMB1 randmulti payload')
    
    try:
        mask = brotli.decompress(mask_br)
        vals = brotli.decompress(vals_br)
        if len(mask) % 75:
            raise ValueError('bad RMB1 mask length')
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
                        raise ValueError('truncated RMB1 values')
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
            raise ValueError('unused RMB1 values')
        return bytes(out)
    except brotli.error:
        exc = None
        raise ValueError('RMB1 randmulti substream is not Brotli-decodable'), exc
        exc = None
        del exc



def _decode_randmulti(blob = None, device = None):
    if not blob:
        return None
    if blob[:4] == b'RMB1':
        raw = _decode_rmb1_randmulti_payload(blob)
    else:
        
        try:
            raw = brotli.decompress(blob)
            out = []
            if raw[:4] == b'QRM1':
                return _decode_qrm1_randmulti(raw, device)
            if None[:3] == b'NM1':
                if len(raw) < 4:
                    raise ValueError('NM1 randmulti stream is truncated')
                scount = int(raw[3])
                if len(raw) != 4 + scount * 600:
                    raise ValueError('NM1 randmulti payload length does not match scount')
                arr = np.frombuffer(raw, dtype = np.uint8, count = scount * 600, offset = 4).reshape(scount, 600).astype(np.int64)
                out.append((torch.from_numpy(arr).to(device), 24, 32, 1))
            elif raw[:3] == b'NM2':
                if len(raw) < 4:
                    raise ValueError('NM2 randmulti stream is truncated')
                pos = 4
                gcount = int(raw[3])
                for _ in range(gcount):
                    if pos + 4 > len(raw):
                        raise ValueError('NM2 randmulti group header is truncated')
                    (lh, lw, amp, scount) = (int(raw[pos]), int(raw[pos + 1]), int(raw[pos + 2]), int(raw[pos + 3]))
                    pos += 4
                    if pos + scount * 600 > len(raw):
                        raise ValueError('NM2 randmulti dense rows are truncated')
                    arr = np.frombuffer(raw, dtype = np.uint8, count = scount * 600, offset = pos).reshape(scount, 600).astype(np.int64)
                    pos += scount * 600
                    out.append((torch.from_numpy(arr).to(device), lh, lw, amp))
                if pos != len(raw):
                    raise ValueError('NM2 randmulti stream has trailing bytes')
            specs = [
                (24, 32, 1, 12),
                (12, 16, 1, 1),
                (6, 8, 1, 1),
                (3, 4, 1, 1),
                (2, 2, 1, 1),
                (8, 8, 1, 1),
                (4, 4, 1, 1),
                (4, 8, 1, 1),
                (2, 4, 1, 1),
                (2, 8, 1, 1),
                (1, 2, 1, 1),
                (1, 4, 1, 1),
                (2, 1, 1, 1),
                (4, 1, 1, 1),
                (8, 1, 1, 1),
                (1, 8, 1, 1)]
            pos = 0
            for lh, lw, amp, scount in specs:
                (rows, pos) = _decode_sparse_randmulti_rows(raw, pos, scount)
                out.append((torch.from_numpy(rows.astype(np.int64)).to(device), lh, lw, amp))
            if pos != len(raw):
                raise ValueError('bad headerless f1 randmulti payload')
            if not out:
                out
            return None
        except brotli.error:
            exc = None
            raise ValueError('randmulti stream is not Brotli-decodable'), exc
            exc = None
            del exc



def read_qpost(path = None, device = None):
    raw = path.read_bytes()
    header_size = len(MAGIC) + struct.calcsize(HEADER)
    if len(raw) < header_size or raw[:4] != MAGIC:
        raise ValueError(f'''bad qpost magic in {path}''')
    lengths = struct.unpack_from(HEADER, raw, len(MAGIC))
    pos = header_size
    streams = { }
    for name, n in zip(STREAM_NAMES, lengths):
        if n < 0:
            raise ValueError(f'''negative qpost stream length for {name}''')
        end = pos + n
        if end > len(raw):
            raise ValueError(f'''qpost stream {name} overruns payload''')
        streams[name] = raw[pos:end]
        pos = end
    if pos != len(raw):
        raise ValueError(f'''qpost has {len(raw) - pos} trailing bytes''')
    return QPostState(postprocess = _load_post_codes(streams['post'], device), f0_shift = _decode_dense_or_delta(streams['shift'], magic_full = b'SH4', magic_delta = b'SD4', default = 40, center = None, device = device), f1_frac = _decode_frac(streams['frac'], device), f1_frac2 = _decode_dense_or_delta(streams['frac2'], magic_full = b'FH2', magic_delta = b'FD2', default = 4, center = None, device = device), f1_frac3 = _decode_dense_or_delta(streams['frac3'], magic_full = b'FH3', magic_delta = b'FD3', default = 4, center = None, device = device), f1_bias = _decode_dense_or_delta(streams['bias'], magic_full = b'BH1', magic_delta = b'BD1', default = 13, center = 13, device = device), f1_region = _decode_region(streams['region'], device), f1_randmulti = _decode_randmulti(streams['randmulti'], device))


def _shift_grid(cache, ch, step = None, h = None, w = None, device = ('cache', 'dict[tuple[float, int, int, int], torch.Tensor]', 'ch', 'int', 'step', 'float', 'h', 'int', 'w', 'int', 'device', 'torch.device', 'return', 'torch.Tensor')):
    key = (step, ch, h, w)
    if key not in cache:
        dy = (ch // 3 - 1) * step
        dx = (ch % 3 - 1) * step
        (yy, xx) = torch.meshgrid(torch.arange(h, device = device, dtype = torch.float32), torch.arange(w, device = device, dtype = torch.float32), indexing = 'ij')
        gx = ((xx - dx) + 0.5) * 2 / w - 1
        gy = ((yy - dy) + 0.5) * 2 / h - 1
        cache[key] = torch.stack([
            gx,
            gy], dim = -1).unsqueeze(0)
    return cache[key]


def apply_qpost_batch(batch_hwc = None, *, pair_start, state, grid_cache, randpat_cache, source_masks):
    pass
# WARNING: Decompyle incomplete


def _load_inflate_renderer_module():
    module_path = Path(__file__).with_name('inflate_renderer.py')
    spec = importlib.util.spec_from_file_location('robust_current_inflate_renderer_for_qpost', module_path)
# WARNING: Decompyle incomplete


def load_source_pair_masks_from_archive(archive_dir = None, *, expected_pairs):
    '''Load the charged source mask tensor used by PR82 mask-aware randmulti.'''
    runtime = _load_inflate_renderer_module()
    mask_path = runtime._resolve_mask_path(archive_dir, 'masks.mkv')
    masks = runtime._load_archive_masks_with_optional_amr1_repair(archive_dir, mask_path)
    if isinstance(masks, torch.Tensor) or masks.ndim != 3:
        raise ValueError(f'''archive masks must decode to (frames,H,W), got {type(masks).__name__}''')
    if int(masks.shape[0]) == int(expected_pairs):
        pair_masks = masks
    elif int(masks.shape[0]) == int(expected_pairs) * 2:
        pair_masks = masks[1::2]
    else:
        raise ValueError(f'''archive source masks have {masks.shape[0]} frames; expected {expected_pairs} pair masks or {expected_pairs * 2} full-frame masks''')
    if pair_masks.numel():
        if int(pair_masks.min().item()) < 0 or int(pair_masks.max().item()) > 4:
            raise ValueError('archive source masks must contain class ids in [0,4]')
    return pair_masks.cpu().long().contiguous()


def apply_qpost_to_raw(raw_path = None, state = None, *, height, width, batch_pairs, device, source_masks):
    frame_bytes = height * width * 3
    size = raw_path.stat().st_size
    if size % 2 * frame_bytes != 0:
        raise ValueError(f'''{raw_path} size {size} is not an even number of RGB frames for {width}x{height}''')
    pair_count = size // 2 * frame_bytes
# WARNING: Decompyle incomplete


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('qpost', type = Path)
    parser.add_argument('inflated_dir', type = Path)
    parser.add_argument('video_names_file', type = Path)
    parser.add_argument('--height', type = int, default = 874)
    parser.add_argument('--width', type = int, default = 1164)
    parser.add_argument('--batch-pairs', type = int, default = 8)
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    state = read_qpost(args.qpost, device)
    source_pair_masks = None
    source_pair_cursor = 0
    if qpost_requires_source_masks(state):
        raw_pair_total = 0
        for line in args.video_names_file.read_text().splitlines():
            if not line.strip():
                continue
            raw_path = args.inflated_dir / (Path(line.strip()).stem + '.raw')
            if not raw_path.exists():
                raise FileNotFoundError(f'''raw output missing before qpost apply: {raw_path}''')
            frame_bytes = args.height * args.width * 3
            raw_size = raw_path.stat().st_size
            if raw_size % 2 * frame_bytes != 0:
                raise ValueError(f'''{raw_path} size {raw_size} is not an even number of RGB frames''')
            raw_pair_total += raw_size // 2 * frame_bytes
        source_pair_masks = load_source_pair_masks_from_archive(args.qpost.parent, expected_pairs = raw_pair_total)
    records = []
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    raise SystemExit(main())

"""
