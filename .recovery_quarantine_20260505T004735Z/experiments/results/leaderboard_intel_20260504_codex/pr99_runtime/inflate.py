"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``100:66: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``inflate.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/results/leaderboard_intel_20260504_codex/pr99_runtime/inflate.py'
__recovery_spec__ = 'inflate.recovery_spec.json'
__recovery_ast_error__ = '100:66: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: inflate.cpython-312.pyc (Python 3.12)

\"\"\"hnerv_repack_latent inflate: load our compact archive, run AaronLeslie138's HNeRV decoder.

Archive format (single 0.bin file):
  u32 dec_len   | dec_blob (brotli)   — concatenated INT8 codes (schema-driven)
  u32 sca_len   | sca_blob            — fp16 scales, one per tensor in schema order
  u32 lat_len   | lat_blob (brotli)   — per-dim asym uint8 + delta + lo/hi split
  u32 wrp_len   | wrp_blob (brotli)   — per-pair (u8 dim, i8 quant_delta), dim=255 means no-op

Credits: HNeRV decoder weights and architecture by AaronLeslie138 (PR #95 / hnerv_muon).
This submission re-packs his archive ~470 B smaller via schema-driven layer names + fp16 scales,
and adds a ~1.2 KB latent-correction sidecar (per-pair single-dim perturbation chosen to
minimize SegNet+PoseNet distortion).
\"\"\"
import io
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

def _ensure_brotli():
    
    try:
        import brotli
        return None
    except ImportError:
        subprocess.check_call([
            sys.executable,
            '-m',
            'pip',
            'install',
            'brotli'])
        return None


_ensure_brotli()
import brotli
import torch

functional
from hnerv_model import HNeRVDecoder
import torch.nn.functional, nn
from schema import SCHEMA, META
from sidecar import decode_corrections, apply_corrections
(CAMERA_H, CAMERA_W) = (874, 1164)
(NATIVE_H, NATIVE_W) = META['eval_size']

def split_archive(b):
    o = 0
    parts = []
    for _ in range(4):
        L = struct.unpack_from('<I', b, o)[0]
        o += 4
        parts.append(b[o:o + L])
        o += L
    if o != len(b):
        raise RuntimeError(f'''archive trailing: {o} vs {len(b)}''')
    return parts


def decode_decoder(blob, sca_blob):
    raw = brotli.decompress(blob)
    codes = np.frombuffer(raw, dtype = np.int8)
    scales = np.frombuffer(sca_blob, dtype = np.float16)
    sd = { }
    o = 0
    for name, shape in enumerate(SCHEMA):
        n_el = int(np.prod(shape))
        chunk = codes[o:o + n_el].reshape(shape)
        sd[name] = torch.from_numpy(chunk.astype(np.float32) * float(scales[i]))
        o += n_el
    if o != codes.size:
        raise RuntimeError(f'''decoder leftover: {o} vs {codes.size}''')
    return sd


def decode_latents(blob):
    raw = brotli.decompress(blob)
    buf = io.BytesIO(raw)
    (n, d) = struct.unpack('<II', buf.read(8))
    mins = np.frombuffer(buf.read(d * 2), dtype = np.float16).astype(np.float32)
    scales = np.frombuffer(buf.read(d * 2), dtype = np.float16).astype(np.float32)
    total = n * d
    lo = np.frombuffer(buf.read(total), dtype = np.uint8).astype(np.uint16)
    hi = np.frombuffer(buf.read(total), dtype = np.uint8).astype(np.uint16)
    delta_zz = (hi << 8 | lo).reshape(n, d)
    delta = np.where(delta_zz % 2 == 0, delta_zz.astype(np.int32) // 2, -(delta_zz.astype(np.int32) // 2) - 1).astype(np.int16)
    q = np.empty_like(delta, dtype = np.int32)
    q[0] = delta[0]
    for i in range(1, n):
        q[i] = q[i - 1] + delta[i]
    return torch.from_numpy(q.astype(np.float32) * scales[(None, :)] + mins[(None, :)])


def inflate(src_bin = None, dst_raw = None):
    f = open(src_bin, 'rb')
    archive_bytes = f.read()
    None(None, None)
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Usage: python -m submissions.hnerv_muon_lc.inflate <src.bin> <dst.raw>')
    inflate(sys.argv[1], sys.argv[2])
    return None

"""
