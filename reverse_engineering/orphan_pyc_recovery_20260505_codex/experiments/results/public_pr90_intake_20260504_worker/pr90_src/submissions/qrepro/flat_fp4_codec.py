# pyc-recovery: STUB unreconstructible -- see .recovery_spec.json for dis() ground-truth
# pycdc could not produce parseable output; raw decompiled text preserved in _PYCDC_PARTIAL_OUTPUT below.
"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``180:41: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``flat_fp4_codec.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/results/public_pr90_intake_20260504_worker/pr90_src/submissions/qrepro/flat_fp4_codec.py'
__recovery_spec__ = 'flat_fp4_codec.recovery_spec.json'
__recovery_ast_error__ = '180:41: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: flat_fp4_codec.cpython-312.pyc (Python 3.12)

'''Flat FP4 serializer for Ship 4 model.pt.br polish sweep.

Replaces torch.save legacy format (23.3 KB envelope, 25.9% of payload)
with a compact [manifest_json | concat_tensor_bytes] layout, brotli-compressed.

Compatible with Ship 4 state_dict:
  - __codebook__ (fp32 [8])
  - quantized: dict of per-module records (conv2d / embedding / linear), with
    weight_kind in {fp4_packed, fp16} + optional bias_fp16
  - dense_fp16: residual dict of un-quantized norm/conv weights

Round-trip is bit-identical: no fp16 round-off, no lossy steps. Only the
serialization envelope changes.
'''
from __future__ import annotations
import json
import struct
from typing import Any
import brotli
import numpy as np
import torch
FLAT_MAGIC = b'QFP4'
FLAT_VERSION = 1
_DTYPE_MAP: 'dict[str, tuple[np.dtype, int]]' = {
    'f32': (np.dtype('<f4'), 4),
    'f16': (np.dtype('<f2'), 2),
    'u8': (np.dtype('<u1'), 1),
    'i64': (np.dtype('<i8'), 8) }

def _tensor_bytes(t = None):
    return t.detach().contiguous().cpu().numpy().tobytes()


def _torch_dtype_to_name(dt = None):
    if dt == torch.float16:
        return 'f16'
    if dt == torch.float32:
        return 'f32'
    if dt == torch.uint8:
        return 'u8'
    if dt == torch.int64:
        return 'i64'
    raise ValueError(f'''unsupported dtype: {dt}''')


def encode_flat(sd = None):
    '''Encode Ship-4-style FP4 state dict into flat brotli-compressed bytes.'''
    manifest = []
    payload = bytearray()
    cb = sd['__codebook__']
    manifest.append({
        't': 'cb',
        'o': len(payload),
        'l': cb.numel() * 4,
        's': list(cb.shape) })
    payload += _tensor_bytes(cb.float())
# WARNING: Decompyle incomplete


def decode_flat(blob = None):
    '''Decode flat blob into a load_state_dict-compatible dict.'''
    raw = brotli.decompress(blob)
    return decode_flat_raw(raw)


def decode_flat_raw(raw = None):
    '''Decode an already-decompressed flat blob.'''
    pass
# WARNING: Decompyle incomplete


def decode_payload_only_for_model(blob = None, model = None, block_size = None):
    \"\"\"Decode a manifest-free payload using the fixed qpose model structure.

    This mirrors ``encode_flat``'s tensor byte order for this architecture, but
    omits the JSON manifest from the archive. The decoder can recover shapes and
    dense state keys from the instantiated model.
    \"\"\"
    pay = memoryview(brotli.decompress(blob))
    offset = 0
    codebook_arr = np.frombuffer(pay[offset:offset + 32], dtype = '<f4').copy()
    codebook = torch.from_numpy(codebook_arr)
    offset += 32
    state_dict = { }
    covered = set()
    omitted_zero_init = {
        'frame2_head.block1.film_proj.weight',
        'frame2_head.block1.film_proj.bias'}
# WARNING: Decompyle incomplete


def decode_qrow_payload_for_model(blob = None, model = None, block_size = None):
    '''Decode QFPL plus row-wise uint8 storage for selected frame1 FiLM rows.'''
    pay = memoryview(brotli.decompress(blob))
    offset = 0
    codebook_arr = np.frombuffer(pay[offset:offset + 32], dtype = '<f4').copy()
    codebook = torch.from_numpy(codebook_arr)
    offset += 32
    state_dict = { }
    covered = set()
    omitted_zero_init = {
        'frame2_head.block1.film_proj.weight',
        'frame2_head.block1.film_proj.bias'}
# WARNING: Decompyle incomplete


def decode_qrow_grouped_payload_for_model(blob = None, model = None, block_size = None):
    '''Decode QFQ2: the same tensors as QFQ1, grouped by storage kind.

    Grouping homogeneous byte streams gives Brotli slightly better context than
    interleaving packed weights, scales, and fp16 tensors module-by-module.
    '''
    pay = memoryview(brotli.decompress(blob))
    offset = 0
    codebook_arr = np.frombuffer(pay[offset:offset + 32], dtype = '<f4').copy()
    codebook = torch.from_numpy(codebook_arr)
    offset += 32
    state_dict = { }
    covered = set()
    packed_modules = []
    fp16_modules = []
    bias_modules = []
    omitted_zero_init = {
        'frame2_head.block1.film_proj.weight',
        'frame2_head.block1.film_proj.bias'}
# WARNING: Decompyle incomplete


def decode_qrow_grouped3_payload_for_model(blob = None, model = None, block_size = None):
    '''Decode QFQ3: packed weights, one fp16 byte-plane group, then raw tensors.'''
    pay = memoryview(brotli.decompress(blob))
    offset = 0
    state_dict = { }
    covered = set()
    packed_modules = []
    fp16_modules = []
    bias_modules = []
    omitted_zero_init = {
        'frame2_head.block1.film_proj.weight',
        'frame2_head.block1.film_proj.bias'}
# WARNING: Decompyle incomplete


def decode_qrow_grouped4_payload_for_model(blob = None, model = None, block_size = None):
    '''Decode QFQ4: QFQ3 tensors with better Brotli-local byte ordering.'''
    pay = memoryview(brotli.decompress(blob))
    offset = 0
    state_dict = { }
    covered = set()
    packed_modules = []
    fp16_modules = []
    bias_modules = []
    omitted_zero_init = {
        'frame2_head.block1.film_proj.weight',
        'frame2_head.block1.film_proj.bias'}
# WARNING: Decompyle incomplete


def _unpack_nibbles(packed = None, count = None):
    hi = packed >> 4 & 15
    lo = packed & 15
    out = np.empty(packed.size * 2, dtype = np.uint8)
    out[0::2] = hi
    out[1::2] = lo
    return out[:count]


def _dequant_fp4(nibbles = None, scales = None, shape = None, codebook = ('nibbles', 'np.ndarray', 'scales', 'np.ndarray', 'shape', 'list[int]', 'codebook', 'torch.Tensor', 'return', 'torch.Tensor')):
    flat_n = int(np.prod(shape))
    block_size = nibbles.size // scales.size
    nib = nibbles.reshape(-1, block_size).astype(np.int64)
    signs = (nib >> 3).astype(np.int64)
    mag = (nib & 7).astype(np.int64)
    levels = codebook.numpy().astype(np.float32)
    q = levels[mag]
    q = np.where(signs.astype(bool), -q, q)
    dq = q * scales.astype(np.float32)[(:, None)]
    return torch.from_numpy(dq.reshape(-1)[:flat_n].reshape(shape).copy()).float()


"""
