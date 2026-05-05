"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``17:2: cannot assign to literal``.

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
__recovery_orphan__ = 'experiments/results/leaderboard_intel_20260504_codex/pr96_runtime/inflate.py'
__recovery_spec__ = 'inflate.recovery_spec.json'
__recovery_ast_error__ = '17:2: cannot assign to literal'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: inflate.cpython-312.pyc (Python 3.12)

import io
import lzma
import struct
import sys
from collections import OrderedDict
from pathlib import Path
import brotli
import constriction
import numpy as np
import torch
from torch.nn import nn

functional
(1164, 874) = import torch.nn.functional, nn
LATENT_DIM = 28
BASE_CHANNELS = 36
EVAL_SIZE = (384, 512)

def _ac_decode(coded, count, hist_u16):
    probs = hist_u16.astype(np.float64)
    probs = np.maximum(probs, 1e-10)
    probs /= probs.sum()
    cat = constriction.stream.model.Categorical(probs, perfect = False)
    dec = constriction.stream.queue.RangeDecoder(np.frombuffer(coded, dtype = np.uint32))
    out = np.zeros(count, dtype = np.int32)
    for i in range(count):
        out[i] = dec.decode(cat)
    return out


def _decompress_hist(comp_id, blob):
    if comp_id == 0:
        return lzma.decompress(blob)
    if None == 2:
        return brotli.decompress(blob)
    if None == 1:
        import zstandard
        return zstandard.ZstdDecompressor().decompress(blob)
    raise None(f'''unknown histogram codec id {comp_id}''')


def _decode_decoder(blob):
    pass
# WARNING: Decompyle incomplete


def _dequantize(quantized):
    sd = OrderedDict()
    for q, scale, shape in quantized.items():
        sd[name] = torch.from_numpy(q.astype(np.float32).reshape(shape)) * scale
    return sd


def _decode_latents(blob):
    h = io.BytesIO(blob)
    (n_rows, n_dim) = struct.unpack('<II', h.read(8))
    mins = torch.from_numpy(np.frombuffer(h.read(n_dim * 2), dtype = np.float16).copy()).float()
    scales = torch.from_numpy(np.frombuffer(h.read(n_dim * 2), dtype = np.float16).copy()).float()
    q = torch.from_numpy(np.frombuffer(h.read(n_rows * n_dim), dtype = np.uint8).reshape(n_rows, n_dim).copy()).float()
    return q * scales.unsqueeze(0) + mins.unsqueeze(0)


class Decoder(nn.Module):
    pass
# WARNING: Decompyle incomplete


def main():
    if len(sys.argv) != 4:
        raise SystemExit('usage: inflate.py <archive_dir> <output_dir> <video_names_file>')
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    video_names_file = Path(sys.argv[3])
    output_dir.mkdir(parents = True, exist_ok = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = _dequantize(_decode_decoder((archive_dir / 'decoder.bin').read_bytes()))
    latents = _decode_latents((archive_dir / 'latents.bin').read_bytes()).to(device)
    decoder = Decoder(LATENT_DIM, BASE_CHANNELS, EVAL_SIZE).to(device)
    decoder.load_state_dict(state_dict)
    decoder.eval()
    (width, height) = CAMERA_SIZE
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    main()
    return None

"""
