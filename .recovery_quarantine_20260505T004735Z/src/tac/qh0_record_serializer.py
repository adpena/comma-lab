"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``25:13: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``qh0_record_serializer.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/qh0_record_serializer.py'
__recovery_spec__ = 'qh0_record_serializer.recovery_spec.json'
__recovery_ast_error__ = '25:13: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: qh0_record_serializer.cpython-312.pyc (Python 3.12)

'''Deterministic QH0/QM0 record parser and serializer.

This module is intentionally narrow: it preserves the reviewed PR85/QH0 record
order and only rewrites between the two runtime-supported byte layouts:

* ``QH0``: high/low nibble split for FP4 payloads and even/odd byte split for
  fp16 scale/value tensors.
* ``QM0``: the same records in direct byte order.

It does not introduce a new runtime grammar. Any caller that wants a different
container must prove runtime support separately before dispatch.
'''
from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import torch
from tac.qh0_renderer_codec import QH0_MAGIC, QM0_MAGIC, QH0CodecError, decode_qh0_state_dict, reconstruct_qh1_payload
from tac.quantizr_faithful_renderer import JointFrameGenerator, build_quantizr_faithful_renderer
SUPPORTED_OUTPUT_MAGICS = (QH0_MAGIC, QM0_MAGIC)
QH0Record = <NODE:12>()
QH0RecordSet = <NODE:12>()
QH0SerializedVariant = <NODE:12>()

class QH0SerializerError(ValueError):
    '''Raised when a QH0/QM0 record stream cannot be serialized safely.'''
    pass


def sha256_bytes(data = dataclass(frozen = True)):
    '''Return the SHA-256 hex digest for ``data``.'''
    return hashlib.sha256(data).hexdigest()


def unsplit_even_odd_bytes(data = None):
    '''Undo QH0 even/odd byte split used for fp16 payloads.'''
    raw = bytes(data)
    half = (len(raw) + 1) // 2
    out = bytearray(len(raw))
    out[0::2] = raw[:half]
    out[1::2] = raw[half:]
    return bytes(out)


def split_even_odd_bytes(data = None):
    '''Apply QH0 even/odd byte split used for fp16 payloads.'''
    raw = bytes(data)
    return raw[0::2] + raw[1::2]


def unpack_hilo_fp4_bytes(data = None, packed_len = None):
    '''Undo QH0 high/low nibble split into direct packed FP4 bytes.'''
    pass
# WARNING: Decompyle incomplete


def pack_hilo_fp4_bytes(packed = None):
    '''Apply QH0 high/low nibble split to direct packed FP4 bytes.'''
    pass
# WARNING: Decompyle incomplete


def _module_weight_order(model = None):
    ordered = []
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Embedding)):
            continue
        ordered.append((name, module))
    return ordered


def _require(raw = None, pos = None, nbytes = None, label = ('raw', 'bytes', 'pos', 'int', 'nbytes', 'int', 'label', 'str', 'return', 'bytes')):
    if pos < 0 and nbytes < 0 or pos + nbytes > len(raw):
        raise QH0SerializerError(f'''QH0 record stream truncated while reading {label}: pos={pos} nbytes={nbytes} payload={len(raw)}''')
    return raw[pos:pos + nbytes]


def _fp16_layout_bytes(raw = None, pos = None, nbytes = None, *, hilo_split, label):
    source = _require(raw, pos, nbytes, label)
    direct = unsplit_even_odd_bytes(source) if hilo_split else source
    qh0 = split_even_odd_bytes(direct)
    return (direct, qh0, pos + nbytes)


def parse_qh0_record_set(payload = None):
    '''Parse QH0/QM0 bytes into deterministic runtime records.'''
    raw = reconstruct_qh1_payload(payload)
    if len(raw) < 3:
        raise QH0SerializerError('QH0/QM0 payload is shorter than the 3-byte magic')
    magic = raw[:3]
    if magic not in SUPPORTED_OUTPUT_MAGICS:
        raise QH0SerializerError(f'''unsupported QH0 serializer magic: {magic!r}''')
    hilo_split = magic == QH0_MAGIC
    model = build_quantizr_faithful_renderer()
    pos = 3
    records = []
    covered = set()
# WARNING: Decompyle incomplete


def _make_record(*, name, category, record_kind, raw, start, end, direct_record, qh0_record, tensor_shape, element_count, kind_byte, source_magic):
    source_record = raw[start:end]
    expected = qh0_record if source_magic == QH0_MAGIC else direct_record
    if source_record != expected:
        raise QH0SerializerError(f'''internal serializer mismatch for {name}: source layout does not round-trip''')
    return QH0Record(name = name, category = category, record_kind = record_kind, offset = start, source_nbytes = end - start, direct_record = bytes(direct_record), qh0_record = bytes(qh0_record), tensor_shape = tensor_shape, element_count = element_count, kind_byte = kind_byte)


def serialize_records(records = None, *, magic):
    '''Serialize records as ``QH0`` or ``QM0`` bytes.'''
    if magic == QH0_MAGIC:
        return b''.join + (lambda .0: pass# WARNING: Decompyle incomplete
)(records())
    if None == QM0_MAGIC:
        return b''.join + (lambda .0: pass# WARNING: Decompyle incomplete
)(records())
    raise None(f'''unsupported serializer output magic: {magic!r}''')


def build_serialized_variants(payload = None):
    '''Build deterministic ``QH0`` and ``QM0`` payload variants.'''
    raw = reconstruct_qh1_payload(payload)
    record_set = parse_qh0_record_set(raw)
    variants = []
    for variant_id, magic in (('qh0_canonical', QH0_MAGIC), ('qm0_direct', QM0_MAGIC)):
        encoded = serialize_records(record_set.records, magic = magic)
        variants.append(QH0SerializedVariant(variant_id = variant_id, magic = magic.decode('ascii'), payload = encoded, payload_bytes = len(encoded), payload_sha256 = sha256_bytes(encoded), same_as_source = encoded == raw))
    return (record_set, tuple(variants))


def prove_decoded_tensor_parity(source_payload = None, candidate_payload = None, *, device):
    '''Decode two payloads through the reviewed loader and prove tensor equality.'''
    (source_state, source_report) = decode_qh0_state_dict(source_payload, device = device)
    (candidate_state, candidate_report) = decode_qh0_state_dict(candidate_payload, device = device)
    mismatches = []
    if set(source_state) != set(candidate_state):
        missing = sorted(set(source_state) - set(candidate_state))
        extra = sorted(set(candidate_state) - set(source_state))
        mismatches.append({
            'kind': 'key_set',
            'missing': missing,
            'extra': extra })
    for key in sorted(set(source_state) & set(candidate_state)):
        left = source_state[key]
        right = candidate_state[key]
        if tuple(left.shape) != tuple(right.shape):
            mismatches.append({
                'kind': 'shape',
                'name': key,
                'source_shape': list(left.shape),
                'candidate_shape': list(right.shape) })
            continue
        if left.dtype != right.dtype:
            mismatches.append({
                'kind': 'dtype',
                'name': key,
                'source_dtype': str(left.dtype),
                'candidate_dtype': str(right.dtype) })
            continue
        if torch.equal(left.cpu(), right.cpu()):
            continue
        diff = (left.cpu() - right.cpu()).abs()
        mismatches.append({
            'kind': 'value',
            'name': key,
            'max_abs_diff': float(diff.max().item()) if diff.numel() else 0,
            'changed_elements': int((diff != 0).sum().item()) })
        if not len(mismatches) >= 8:
            continue
        sorted(set(source_state) & set(candidate_state))
    return {
        'decoded_tensor_parity': not mismatches,
        'mismatch_count': len(mismatches),
        'mismatches': mismatches,
        'source_report': source_report.__dict__,
        'candidate_report': candidate_report.__dict__,
        'source_tensor_count': len(source_state),
        'candidate_tensor_count': len(candidate_state) }


def record_set_summary(record_set = None):
    '''Return byte accounting for a parsed record set.'''
    by_category = { }
    by_kind = { }
    for record in record_set.records:
        by_category[record.category] = by_category.get(record.category, 0) + record.source_nbytes
        by_kind[record.record_kind] = by_kind.get(record.record_kind, 0) + record.source_nbytes
# WARNING: Decompyle incomplete


def choose_byte_win_candidates(candidates = None, *, require_runtime_compatible):
    '''Filter candidate rows to real byte wins that are runtime compatible.'''
    out = []
    for candidate in candidates:
        model_delta = int(candidate.get('candidate_model_delta_bytes_vs_source', 0))
        runtime = candidate.get('runtime_compatibility', { })
        compatible = bool(runtime.get('runtime_can_decode_without_edits', False)) if isinstance(runtime, Mapping) else False
        if not model_delta < 0:
            continue
        if compatible and require_runtime_compatible:
            continue
        out.append(candidate)
    return out


"""
