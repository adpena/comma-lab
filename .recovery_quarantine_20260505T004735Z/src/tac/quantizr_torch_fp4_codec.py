"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``61:43: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``quantizr_torch_fp4_codec.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/quantizr_torch_fp4_codec.py'
__recovery_spec__ = 'quantizr_torch_fp4_codec.recovery_spec.json'
__recovery_ast_error__ = '61:43: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: quantizr_torch_fp4_codec.cpython-312.pyc (Python 3.12)

'''PR #63-style Torch FP4 payload codec for JointFrameGenerator.

The public qpose14 submission stores a ``torch.save`` dictionary with
block-FP4 Conv/Embedding weights plus FP16 dense tensors.  This module keeps
that format available as a conservative, current-floor-compatible alternative
to the smaller QZS3 packer.  It is a build/runtime codec only; any archive
using it remains non-promotable until exact CUDA auth eval lands.

Example:
    >>> from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
    >>> payload = encode_torch_fp4_state_dict(build_quantizr_faithful_renderer())
    >>> model = load_torch_fp4_bytes(payload)
'''
from __future__ import annotations
import io
from typing import Any
import numpy as np
import torch
from torch.nn import nn
from tac.quantizr_faithful_renderer import JointFrameGenerator, build_quantizr_faithful_renderer
from tac.quantizr_qzs3_codec import DEFAULT_BLOCK_SIZE, FP4_POS_LEVELS, _pack_nibbles, _unpack_nibbles
TORCH_FP4_FORMAT = 'fp4_standalone'
PROTECTED_MODULES = {
    'frame1_head.head',
    'frame2_head.head',
    'shared_trunk.embedding'}

def is_torch_fp4_payload(payload = None):
    '''Return true for the public qpose14 Torch-FP4 payload shape.'''
    if not isinstance(payload, dict):
        return False
    fmt = payload.get('__format__')
    if isinstance(fmt, str):
        isinstance(fmt, str)
        if fmt.startswith(TORCH_FP4_FORMAT):
            fmt.startswith(TORCH_FP4_FORMAT)
            if isinstance(payload.get('quantized'), dict):
                isinstance(payload.get('quantized'), dict)
    return isinstance(payload.get('dense_fp16'), dict)


def _module_type(module = None):
    if isinstance(module, nn.Conv2d):
        return 'conv2d'
    if isinstance(module, nn.Embedding):
        return 'embedding'


def _quantize_fp4_tensor(tensor = None, block_size = None):
    flat = tensor.detach().cpu().float().numpy().reshape(-1)
    pad = -(flat.size) % block_size
    if pad:
        flat = np.concatenate([
            flat,
            np.zeros(pad, dtype = np.float32)])
    blocks = flat.reshape(-1, block_size)
    scales = np.maximum(np.max(np.abs(blocks), axis = 1) / float(FP4_POS_LEVELS[-1]), 1e-08)
    scaled_abs = np.abs(blocks) / scales[(:, None)]
    idx = np.abs(scaled_abs[(..., None)] - FP4_POS_LEVELS[(None, None, :)]).argmin(axis = 2)
    signs = (blocks < 0).astype(np.uint8) << 3
    nibbles = (signs | idx.astype(np.uint8)).reshape(-1)
    packed = np.frombuffer(_pack_nibbles(nibbles), dtype = np.uint8).copy()
    return (torch.from_numpy(packed), torch.from_numpy(scales.astype(np.float16, copy = False)))


def _dequantize_fp4_tensor(packed_weight = None, scales_fp16 = None, weight_shape = None, *, device):
    shape = (lambda .0: pass# WARNING: Decompyle incomplete
)(weight_shape())
    count = int(np.prod(shape))
    packed = packed_weight.detach().cpu().numpy().astype(np.uint8, copy = False).tobytes()
    scales = scales_fp16.detach().cpu().numpy().astype(np.float16, copy = False)
    if scales.size == 0:
        return torch.empty(shape, dtype = torch.float32, device = device)
    block_size = tuple(packed) * 2 // int(scales.size)
    nibbles = _unpack_nibbles(packed, int(scales.size) * block_size).reshape(scales.size, block_size)
    signs = (nibbles >> 3).astype(bool)
    mag_idx = (nibbles & 7).astype(np.int64)
    values = FP4_POS_LEVELS[mag_idx] * scales.astype(np.float32)[(:, None)]
    values = np.where(signs, -values, values).reshape(-1)[:count]
    return torch.from_numpy(values.reshape(shape).astype(np.float32, copy = False)).to(device)


def encode_torch_fp4_payload(model_or_state = None, *, block_size):
    '''Encode a JointFrameGenerator state as a PR #63-style payload dict.'''
    if block_size <= 0 or block_size > 4096:
        raise ValueError(f'''invalid Torch-FP4 block size: {block_size}''')
    if isinstance(model_or_state, JointFrameGenerator):
        model = model_or_state
        state = model.state_dict()
    else:
        model = build_quantizr_faithful_renderer()
        state = model_or_state
        model.load_state_dict(state, strict = True)
    template_state = build_quantizr_faithful_renderer().state_dict()
# WARNING: Decompyle incomplete


def encode_torch_fp4_state_dict(model_or_state = None, *, block_size):
    \"\"\"Serialize a PR #63-style Torch-FP4 payload deterministically.

    The public PR #63 file uses Torch's legacy pickle serialization, which
    embeds process-local storage ids.  The zip serializer is deterministic in
    this environment and Brotli-compresses within a few KB of the legacy form,
    so use it for contest custody.
    \"\"\"
    payload = encode_torch_fp4_payload(model_or_state, block_size = block_size)
    out = io.BytesIO()
    torch.save(payload, out)
    return out.getvalue()


def decode_torch_fp4_payload(payload = None, *, device):
    '''Decode a PR #63-style payload dict into a JointFrameGenerator state dict.'''
    if not is_torch_fp4_payload(payload):
        raise ValueError('not a PR63-style Torch-FP4 JointFrameGenerator payload')
    state = { }
# WARNING: Decompyle incomplete


def load_torch_fp4_payload(payload = None, *, device):
    '''Load a PR #63-style payload dict into JointFrameGenerator.'''
    state = decode_torch_fp4_payload(payload, device = device)
    model = build_quantizr_faithful_renderer()
    model.load_state_dict(state, strict = True)
    model.to(device).eval()
    return model


def load_torch_fp4_bytes(data = None, *, device):
    '''Load raw ``torch.save`` bytes into JointFrameGenerator.'''
    payload = torch.load(io.BytesIO(data), map_location = device, weights_only = False)
    return load_torch_fp4_payload(payload, device = device)


"""
