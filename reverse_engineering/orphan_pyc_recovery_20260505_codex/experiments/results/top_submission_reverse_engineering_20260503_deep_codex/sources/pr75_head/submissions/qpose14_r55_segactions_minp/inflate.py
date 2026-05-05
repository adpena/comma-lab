"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``31:104: invalid syntax``.

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
__recovery_orphan__ = 'experiments/results/top_submission_reverse_engineering_20260503_deep_codex/sources/pr75_head/submissions/qpose14_r55_segactions_minp/inflate.py'
__recovery_spec__ = 'inflate.recovery_spec.json'
__recovery_ast_error__ = '31:104: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: inflate.cpython-312.pyc (Python 3.12)

import io
import os
import struct
import sys
import tempfile
from pathlib import Path
import av
import brotli
import einops
import numpy as np
import torch
from torch.nn import nn

functional
from tqdm import tqdm
import torch.nn.functional, nn

class FP4Codebook:
    pos_levels = torch.tensor([
        0,
        0.5,
        1,
        1.5,
        2,
        3,
        4,
        6], dtype = torch.float32)
    dequantize_from_nibbles = (lambda nibbles = None, scales = None, orig_shape = staticmethod: flat_n = int(torch.tensor(orig_shape).prod().item())block_size = nibbles.numel() // scales.numel()nibbles = nibbles.view(-1, block_size)signs = (nibbles >> 3).to(torch.int64)mag_idx = (nibbles & 7).to(torch.int64)levels = FP4Codebook.pos_levels.to(scales.device, torch.float32)q = levels[mag_idx]q = torch.where(signs.bool(), -q, q)dq = q * scales[(:, None)].to(torch.float32)dq.view(-1)[:flat_n].reshape(orig_shape))()


def unpack_nibbles(packed = None, count = None):
    flat = packed.reshape(-1)
    hi = flat >> 4 & 15
    lo = flat & 15
    out = torch.empty(flat.numel() * 2, dtype = torch.uint8, device = packed.device)
    out[0::2] = hi
    out[1::2] = lo
    return out[:count]


def get_decoded_state_dict(payload_data = None, device = None):
    if payload_data.startswith(b'QZS3'):
        return get_grouped_qv_state_dict(payload_data, device)
    if None.startswith(b'QZS2'):
        return get_grouped_q10_state_dict(payload_data, device)
    if None.startswith(b'QZS1'):
        return get_grouped_compact_state_dict(payload_data, device)
    if None.startswith(b'QZC1') and payload_data.startswith(b'QZC2') or payload_data.startswith(b'QZC3'):
        return get_compact_state_dict(payload_data, device)
    data = None.load(io.BytesIO(payload_data), map_location = device)
    state_dict = { }
# WARNING: Decompyle incomplete


def get_compact_state_dict(payload_data = None, device = None):
    pass
# WARNING: Decompyle incomplete


def get_grouped_compact_state_dict(payload_data = None, device = None):
    pass
# WARNING: Decompyle incomplete


def unpack_q10(data = None, count = None):
    raw = np.frombuffer(data, dtype = np.uint8)
    out = np.empty(count, dtype = np.uint16)
    acc = 0
    bits = 0
    j = 0
    for byte in raw:
        acc |= int(byte) << bits
        bits += 8
        if not bits >= 10:
            continue
        if not j < count:
            continue
        out[j] = acc & 1023
        acc >>= 10
        bits -= 10
        j += 1
        if not bits >= 10:
            continue
        if j < count:
            continue
    continue
    return torch.from_numpy(out.astype(np.float32, copy = False))


def unpack_qbits(data = None, count = None, width = None):
    raw = np.frombuffer(data, dtype = np.uint8)
    mask = (1 << width) - 1
    out = np.empty(count, dtype = np.uint16)
    acc = 0
    bits = 0
    j = 0
    for byte in raw:
        acc |= int(byte) << bits
        bits += 8
        if not bits >= width:
            continue
        if not j < count:
            continue
        out[j] = acc & mask
        acc >>= width
        bits -= width
        j += 1
        if not bits >= width:
            continue
        if j < count:
            continue
    continue
    return torch.from_numpy(out.astype(np.float32, copy = False))


def get_qv_specs():
    specs = {
        'frame1_head.block1.film_proj.weight': (9, False),
        'pose_mlp.2.weight': (10, True) }
    for key in ('frame1_head.block1.conv1.norm.weight', 'frame1_head.block1.conv1.norm.bias', 'frame1_head.block1.norm2.weight', 'frame1_head.block1.norm2.bias', 'frame1_head.block1.film_proj.bias', 'frame1_head.block2.conv1.norm.weight', 'frame1_head.block2.conv1.norm.bias', 'frame1_head.block2.norm2.weight', 'frame1_head.block2.norm2.bias', 'frame1_head.pre.norm.weight', 'frame1_head.pre.norm.bias'):
        specs[key] = (8, False)
    for key in ('frame2_head.block1.conv1.norm.weight', 'frame2_head.block1.conv1.norm.bias', 'frame2_head.block1.norm2.weight', 'frame2_head.block1.norm2.bias', 'frame2_head.block2.conv1.norm.weight', 'frame2_head.block2.conv1.norm.bias', 'frame2_head.block2.norm2.weight', 'frame2_head.block2.norm2.bias', 'frame2_head.pre.norm.weight', 'frame2_head.pre.norm.bias'):
        specs[key] = (8, False)
    return specs


def get_grouped_qv_state_dict(payload_data = None, device = None):
    pass
# WARNING: Decompyle incomplete


def get_grouped_q10_state_dict(payload_data = None, device = None):
    pass
# WARNING: Decompyle incomplete


class QConv2d(nn.Conv2d):
    pass
# WARNING: Decompyle incomplete


class QEmbedding(nn.Embedding):
    pass
# WARNING: Decompyle incomplete


class SepConvGNAct(nn.Module):
    pass
# WARNING: Decompyle incomplete


class SepConv(nn.Module):
    pass
# WARNING: Decompyle incomplete


class SepResBlock(nn.Module):
    pass
# WARNING: Decompyle incomplete


class FiLMSepResBlock(nn.Module):
    pass
# WARNING: Decompyle incomplete


class SharedMaskDecoder(nn.Module):
    pass
# WARNING: Decompyle incomplete


class Frame2StaticHead(nn.Module):
    pass
# WARNING: Decompyle incomplete


class FrameHead(nn.Module):
    pass
# WARNING: Decompyle incomplete


class JointFrameGenerator(nn.Module):
    pass
# WARNING: Decompyle incomplete


def make_coord_grid(batch, height = None, width = None, device = None, dtype = ('batch', int, 'height', int, 'width', int, 'return', torch.Tensor)):
    ys = (torch.arange(height, device = device, dtype = dtype) + 0.5) / height
    xs = (torch.arange(width, device = device, dtype = dtype) + 0.5) / width
    (yy, xx) = torch.meshgrid(ys, xs, indexing = 'ij')
    grid = torch.stack([
        xx * 2 - 1,
        yy * 2 - 1], dim = 0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)


def make_dct_basis(k = None, h = None, w = None, device = ('k', int, 'h', int, 'w', int, 'device', torch.device, 'return', torch.Tensor)):
    ys = (torch.arange(h, device = device, dtype = torch.float32) + 0.5) / h
    xs = (torch.arange(w, device = device, dtype = torch.float32) + 0.5) / w
    (yy, xx) = torch.meshgrid(ys, xs, indexing = 'ij')
    freqs = []
    max_freq = 16
    for fy in range(max_freq):
        for fx in range(max_freq):
            if fx == 0 and fy == 0:
                continue
            freqs.append((fx, fy, fx * fx + fy * fy))
    freqs.sort(key = (lambda item: item[2]))
    patterns = []
    for channel in range(3):
        for fx, fy, _ in freqs:
            pat = torch.cos(np.pi * fx * xx) * torch.cos(np.pi * fy * yy)
            chans = torch.zeros(3, h, w, device = device)
            chans[channel] = pat
            patterns.append(chans)
            if not len(patterns) >= k:
                continue
            basis = torch.stack(patterns, dim = 0)
            
            
            return range(3), freqs, basis / basis.flatten(1).std(dim = 1).clamp_min(1e-06).view(-1, 1, 1, 1)
    raise ValueError(f'''not enough DCT basis patterns for k={k}''')


def load_actuator(path = None, device = None):
    if not path.exists():
        return None
    f = open(path, 'rb')
    payload = np.load(io.BytesIO(brotli.decompress(f.read())))
    None(None, None)
# WARNING: Decompyle incomplete


def seg_tile_action_specs(device = None):
    specs = []
    directions = [
        (1, 1, 1),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (0, 1, 1),
        (1, 0, 1),
        (-0.35, 0.15, 0.45),
        (0.25, 0.15, -0.2)]
    for vec in directions:
        v = torch.tensor(vec, dtype = torch.float32, device = device).view(3, 1, 1)
        v = v / v.abs().max().clamp_min(1e-06)
        for amp in (2, 4, 6, 8, 12, 16):
            specs.append(v * amp)
            specs.append(-v * amp)
    return torch.stack(specs, dim = 0)


def load_seg_tile_actions_data(data = None, device = None):
    raw = brotli.decompress(data)
    records = []
    if len(raw) % 4 == 0:
        for i in range(0, len(raw), 4):
            frame = int.from_bytes(raw[i:i + 2], 'little')
            tile = raw[i + 2]
            action = raw[i + 3]
            records.append((frame, tile, action))
    elif len(raw) % 5 == 0:
        for i in range(0, len(raw), 5):
            frame = int.from_bytes(raw[i:i + 2], 'little')
            tile = int.from_bytes(raw[i + 2:i + 4], 'little')
            action = raw[i + 4]
            records.append((frame, tile, action))
    else:
        raise ValueError(f'''unsupported seg tile action payload length: {len(raw)}''')
    by_frame = { }
    for frame, tile, action in records:
        by_frame.setdefault(frame, []).append((tile, action))
    return {
        'by_frame': by_frame,
        'deltas': seg_tile_action_specs(device) }


def load_seg_tile_actions(path = None, device = None):
    if not path.exists():
        return None
    return load_seg_tile_actions_data(path.read_bytes(), device)


def load_smooth_pose(path = None):
    if not path.exists():
        return None
    f = open(path, 'rb')
    payload = np.load(io.BytesIO(brotli.decompress(f.read())))
    None(None, None)
# WARNING: Decompyle incomplete


def make_smooth_pose_basis(num_pairs = None, basis_kind = None):
    t = np.linspace(-1, 1, num_pairs, dtype = np.float32)
    cols = [
        np.ones_like(t),
        t,
        t * t,
        t * t * t]
    if basis_kind == 'poly_fourier':
        u = (t + 1) * 0.5
        for f in (1, 2, 3, 4):
            cols.append(np.sin(np.float32(2 * np.pi * f) * u))
            cols.append(np.cos(np.float32(2 * np.pi * f) * u))
    elif basis_kind != 'poly':
        raise ValueError(f'''unsupported smooth pose basis: {basis_kind}''')
    return np.stack(cols, axis = 1).astype(np.float32)


def load_encoded_mask_video(path = None):
    container = av.open(path)
    frames = []
    for frame in container.decode(video = 0):
        img = frame.to_ndarray(format = 'gray')
        cls_img = np.round(img / 63).astype(np.uint8)
        cls_img = np.clip(cls_img, 0, 4)
        frames.append(cls_img)
    container.close()
    return torch.from_numpy(np.stack(frames)).contiguous()


def main():
    if len(sys.argv) < 4:
        print('Usage: python inflate.py <data_dir> <output_dir> <file_list_txt>')
        sys.exit(1)
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    out_dir.mkdir(parents = True, exist_ok = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    main()
    return None

"""
