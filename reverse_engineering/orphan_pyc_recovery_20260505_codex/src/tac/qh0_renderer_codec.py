"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``43:19: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``qh0_renderer_codec.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/qh0_renderer_codec.py'
__recovery_spec__ = 'qh0_renderer_codec.recovery_spec.json'
__recovery_ast_error__ = '43:19: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: qh0_renderer_codec.cpython-312.pyc (Python 3.12)

'''QH0/QM0 renderer codec for PR85-style JointFrameGenerator payloads.

The public PR85/PR89 family stores a Quantizr-faithful
``JointFrameGenerator`` as a compact custom byte stream.  This module decodes
that stream without importing the public submission runtime or using pickle.
It is intentionally narrow: the output is the existing
``tac.quantizr_faithful_renderer.JointFrameGenerator`` used by robust_current.
'''
from __future__ import annotations
import hashlib
import json
import lzma
import math
import struct
import zlib
from dataclasses import dataclass
import numpy as np
import torch
from tac.quantizr_faithful_renderer import JointFrameGenerator, build_quantizr_faithful_renderer
QH0_MAGIC = b'QH0'
QM0_MAGIC = b'QM0'
QH1_MAGIC = b'QH1'
QH0_SUPPORTED_MAGICS = (QH0_MAGIC, QM0_MAGIC)
QH1_HEADER_STRUCT = struct.Struct('<I')
QH1_SCHEMA = 'qh1_record_repack_v1'
FP4_POS_LEVELS = torch.tensor([
    0,
    0.5,
    1,
    1.5,
    2,
    3,
    4,
    6], dtype = torch.float32)

class QH0CodecError(ValueError):
    '''Raised when a QH0/QM0 payload is malformed or unsupported.'''
    pass

QH0DecodeReport = <NODE:12>()

def unpack_nibbles(packed = None, count = None):
    '''Unpack high/low 4-bit nibbles in the public PR85 order.'''
    if count < 0:
        raise QH0CodecError(f'''nibble count must be non-negative, got {count}''')
    flat = packed.reshape(-1).to(torch.uint8)
    out = torch.empty(flat.numel() * 2, dtype = torch.uint8, device = packed.device)
    out[0::2] = flat >> 4 & 15
    out[1::2] = flat & 15
    return out[:count]


def _require_available(raw = None, pos = None, nbytes = None, label = ('raw', 'bytes', 'pos', 'int', 'nbytes', 'int', 'label', 'str', 'return', 'None')):
    if nbytes < 0 and pos < 0 or pos + nbytes > len(raw):
        raise QH0CodecError(f'''QH0 payload truncated while reading {label}: pos={pos} nbytes={nbytes} payload={len(raw)}''')


def _read_u8(raw = None, pos = None, label = None):
    _require_available(raw, pos, 1, label)
    return (int(raw[pos]), pos + 1)


def _sha256(data = None):
    return hashlib.sha256(data).hexdigest()


def _decode_qh1_record(data = None, codec = None, label = None):
    if codec == 'raw':
        return data
    if None == 'zlib':
        return zlib.decompress(data)
    if None == 'lzma':
        return lzma.decompress(data)
    if None == 'brotli':
        
        try:
            import brotli
            
            try:
                return brotli.decompress(data)
                raise QH0CodecError(f'''{label}: unsupported QH1 record codec {codec!r}''')
                except ImportError:
                    exc = None
                    raise QH0CodecError(f'''{label}: QH1 brotli record requires brotli'''), exc
                    exc = None
                    del exc
            except brotli.error:
                exc = None
                raise QH0CodecError(f'''{label}: invalid QH1 brotli record'''), exc
                exc = None
                del exc




def reconstruct_qh1_payload(payload = None):
    '''Reconstruct original QH0/QM0 bytes from a QH1 record-repack payload.

    QH1 is intentionally lossless: it patches compressed record slices back into
    a stored QH0/QM0 source byte stream, then the normal reviewed QH0 decoder
    handles model tensors. It is a byte-packer only, not a renderer mutation.
    '''
    pass
# WARNING: Decompyle incomplete


def _unsplit_bytes_to_tensor(raw, pos, nbytes, dtype = None, shape = None, device = None, label = ('raw', 'bytes', 'pos', 'int', 'nbytes', 'int', 'dtype', 'torch.dtype', 'shape', 'tuple[int, ...]', 'device', 'torch.device | str', 'label', 'str', 'return', 'tuple[torch.Tensor, int]')):
    \"\"\"Undo PR85's even/odd byte split before constructing a tensor.\"\"\"
    _require_available(raw, pos, nbytes, label)
    half = (nbytes + 1) // 2
    source = np.frombuffer(raw[pos:pos + nbytes], dtype = np.uint8)
    out = np.empty(nbytes, dtype = np.uint8)
    out[0::2] = source[:half]
    out[1::2] = source[half:]
    tensor = torch.frombuffer(bytearray(out.tobytes()), dtype = dtype).clone()
    return (tensor.reshape(shape).to(device), pos + nbytes)


def _read_tensor_bytes(raw, pos, nbytes, dtype = None, shape = None, device = None, label = ('raw', 'bytes', 'pos', 'int', 'nbytes', 'int', 'dtype', 'torch.dtype', 'shape', 'tuple[int, ...]', 'device', 'torch.device | str', 'label', 'str', 'return', 'tuple[torch.Tensor, int]')):
    _require_available(raw, pos, nbytes, label)
    tensor = torch.frombuffer(bytearray(raw[pos:pos + nbytes]), dtype = dtype).clone()
    return (tensor.reshape(shape).to(device), pos + nbytes)


def _unhilo_packed(raw, pos = None, packed_len = None, device = None, label = ('raw', 'bytes', 'pos', 'int', 'packed_len', 'int', 'device', 'torch.device | str', 'label', 'str', 'return', 'tuple[torch.Tensor, int]')):
    '''Undo PR85 QH0 high/low nibble split into packed FP4 bytes.'''
    if packed_len % 2:
        raise QH0CodecError(f'''{label} packed_len must be even for QH0, got {packed_len}''')
    half = packed_len // 2
    _require_available(raw, pos, half * 2, label)
    hi_packed = np.frombuffer(raw[pos:pos + half], dtype = np.uint8)
    pos += half
    lo_packed = np.frombuffer(raw[pos:pos + half], dtype = np.uint8)
    pos += half
    hi = np.empty(half * 2, dtype = np.uint8)
    lo = np.empty(half * 2, dtype = np.uint8)
    hi[0::2] = hi_packed >> 4 & 15
    hi[1::2] = hi_packed & 15
    lo[0::2] = lo_packed >> 4 & 15
    lo[1::2] = lo_packed & 15
    packed = (hi[:packed_len] << 4 | lo[:packed_len]).astype(np.uint8)
    tensor = torch.frombuffer(bytearray(packed.tobytes()), dtype = torch.uint8).clone()
    return (tensor.to(device), pos)


def _dequantize_fp4_from_nibbles(nibbles = None, scales = None, shape = None):
    flat_n = math.prod(shape)
    if scales.numel() <= 0:
        raise QH0CodecError('FP4 record has no scales')
    if nibbles.numel() % scales.numel() != 0:
        raise QH0CodecError(f'''FP4 nibbles/scales mismatch: nibbles={nibbles.numel()} scales={scales.numel()}''')
    block_size = nibbles.numel() // scales.numel()
    nib = nibbles.view(-1, block_size)
    signs = (nib >> 3).to(torch.int64)
    mag_idx = (nib & 7).to(torch.int64)
    levels = FP4_POS_LEVELS.to(scales.device, torch.float32)
    q = levels[mag_idx]
    q = torch.where(signs.bool(), -q, q)
    dq = q * scales[(:, None)].to(torch.float32)
    return dq.reshape(-1)[:flat_n].reshape(shape)


def _module_weight_order(model = None):
    ordered = []
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Embedding)):
            continue
        ordered.append((name, module))
    return ordered


def decode_qh0_state_dict(payload = None, *, device):
    '''Decode PR85/PR89 QH0 or QM0 bytes into a JointFrameGenerator state dict.'''
    raw = reconstruct_qh1_payload(payload)
    if len(raw) < 3:
        raise QH0CodecError('QH0 payload is shorter than the 3-byte magic')
    magic = raw[:3]
    if magic not in QH0_SUPPORTED_MAGICS:
        raise QH0CodecError(f'''unsupported QH0 renderer magic: {magic!r}''')
    hilo_split = magic == QH0_MAGIC
    pos = 3
    model = build_quantizr_faithful_renderer()
    state = { }
    covered = set()
    fp4_count = 0
    fp16_count = 0
    int8_count = 0
# WARNING: Decompyle incomplete


def load_qh0(payload = None, *, device):
    '''Load QH0/QM0 bytes into a Quantizr-faithful JointFrameGenerator.'''
    (state, _report) = decode_qh0_state_dict(payload, device = device)
    model = build_quantizr_faithful_renderer()
    model.load_state_dict(state, strict = True)
    return model.to(device).eval()


"""
