"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``32:16: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``sidecar.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/results/leaderboard_intel_20260504_codex/pr99_runtime/sidecar.py'
__recovery_spec__ = 'sidecar.recovery_spec.json'
__recovery_ast_error__ = '32:16: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: sidecar.cpython-312.pyc (Python 3.12)

\"\"\"Latent-correction sidecar for hnerv_repack_latent.

Wire format (single blob, brotli'd):
  u16 n_pairs
  per pair: u8 dim_idx (0..27, or 255 = no correction), i8 delta_quantized (real = i8 * DELTA_SCALE)

At inflate time, for each pair p:
  if dim_idx[p] != 255:
      latents[p, dim_idx[p]] += delta_quantized[p] * DELTA_SCALE
\"\"\"
import struct
import numpy as np
DELTA_SCALE = 0.01

def encode_corrections(out_dim, out_delta_q):
    \"\"\"out_dim, out_delta_q: int8 arrays of length 600. dim=0 + delta_q=0 means 'no correction'.
    Returns brotli-compressed blob.\"\"\"
    import brotli
    n = len(out_dim)
# WARNING: Decompyle incomplete


def decode_corrections(blob):
    '''Returns (dim_arr (n, int8), delta_q_arr (n, int8)). dim==255 means no correction.'''
    import brotli
    raw = brotli.decompress(blob)
    n = struct.unpack_from('<H', raw, 0)[0]
    arr = np.frombuffer(raw[2:2 + 2 * n], dtype = np.uint8).reshape(n, 2)
    dim = arr[(:, 0)]
    delta_q = arr[(:, 1)].view(np.int8)
    return (dim, delta_q)


def apply_corrections(latents_tensor, dim_arr, delta_q_arr, scale = (DELTA_SCALE,)):
    '''In-place add correction to latents_tensor (n, latent_dim). dim==255 means no-op.'''
    import torch
    n = latents_tensor.shape[0]
    for p in range(n):
        d = int(dim_arr[p])
        if d == 255:
            continue
        latents_tensor[(p, d)] = latents_tensor[(p, d)] + float(delta_q_arr[p]) * scale
    return latents_tensor


"""
