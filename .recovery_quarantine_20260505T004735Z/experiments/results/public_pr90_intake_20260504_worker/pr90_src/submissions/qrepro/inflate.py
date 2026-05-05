"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``65:104: invalid syntax``.

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
__recovery_orphan__ = 'experiments/results/public_pr90_intake_20260504_worker/pr90_src/submissions/qrepro/inflate.py'
__recovery_spec__ = 'inflate.recovery_spec.json'
__recovery_ast_error__ = '65:104: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: inflate.cpython-312.pyc (Python 3.12)

import io
import bz2
import lzma
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
GEO_MAGIC = b'QGEO1\x00'
PACK_MAGIC = b'QPK1\x00'
RGB_CONTROL_MAGIC = b'QRGB1\x00'
RGB_BAND_CONTROL_MAGIC = b'QRGB2\x00'
RGB_SPARSE_CONTROL_MAGIC = b'QRGB3\x00'
RGB_HBAND_CONTROL_MAGIC = b'QRGB4\x00'
RGB_HBAND_PLANE_ORDER = (5, 2, 3, 0, 11, 16, 22, 18, 6, 19, 13, 20, 7, 10, 21, 12, 1, 9, 23, 17, 24, 4, 25, 8, 28, 15, 14, 27, 26, 29)
RGB_QUAD_CONTROL_MAGIC = b'QRGB5\x00'
RGB_QUAD_PLANE_ORDER = (5, 2, 3, 0, 22, 11, 18, 19, 16, 6, 20, 13, 23, 24, 7, 10, 21, 25, 12, 1, 28, 9, 17, 4, 27, 8, 15, 32, 26, 30, 14, 29, 31, 35, 33, 34)
RGB_VDETAIL_CONTROL_MAGIC = b'QRGB6\x00'
RGB_VDETAIL_PLANE_ORDER = (5, 2, 3, 0, 22, 11, 18, 40, 19, 16, 6, 20, 13, 37, 23, 24, 7, 10, 21, 25, 12, 1, 36, 28, 9, 17, 4, 46, 43, 27, 8, 39, 41, 15, 32, 42, 26, 30, 14, 38, 29, 31, 35, 44, 33, 47, 45, 34)
RGB_COMPACT_CONTROL_BYTES = 5058
RGB_COMPACT_PLANE_ORDER = (34, 33, 29, 45, 47, 44, 35, 31, 26, 14, 38, 27, 30, 15, 42, 25, 32, 8, 43, 28, 4, 17, 46, 24, 1, 41, 21, 12, 36, 39, 13, 9, 10, 7, 16, 20, 19, 37, 6, 3, 18, 23, 11, 22, 40, 5, 0, 2)
RGB_HDETAIL_CONTROL_MAGIC = b'QRGB7\x00'
RGB_HDETAIL_PLANE_ORDER = RGB_VDETAIL_PLANE_ORDER + tuple(range(48, 60))
COMPACT_MASK_BODY_BYTES = 152431
COMPACT_MODEL_BODY_BYTES = 56385
SEM_M11_BR_MAGIC = b'SM11BR\x00'
SEM_M5_BR_MAGIC = b'SM5BR\x00'
SEM_M5_SHIFT_BR_MAGIC = b'SM5SBR\x00'
SEM_M5_SHIFT_BIG_BR_MAGIC = b'SM5S7BR\x00'
SEM_M5_SHIFT_BIG3_BR_MAGIC = b'SM5S8BR\x00'
SEM_M5_SHIFT_BIG5_BR_MAGIC = b'SM5SABR\x00'
SEM_TOPBAND_BR_MAGIC = b'STBM1BR\x00'
FLAT_MODEL_BR_MAGIC = b'QFBR\x00'
PAYLOAD_ONLY_MODEL_BR_MAGIC = b'QFPL\x00'
QROW_MODEL_BR_MAGIC = b'QFQ1\x00'
QROW_GROUPED_MODEL_BR_MAGIC = b'QFQ2\x00'
QROW_GROUPED3_MODEL_BR_MAGIC = b'QFQ3\x00'
QROW_GROUPED4_MODEL_BR_MAGIC = b'QFQ4\x00'
(NET_H, NET_W) = (384, 512)
PAIRS_PER_FILE = 600

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
    data = torch.load(io.BytesIO(payload_data), map_location = device)
    state_dict = { }
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


def load_encoded_mask_video(path = None, soft = None):
    container = av.open(path)
    frames = []
    for frame in container.decode(video = 0):
        img = frame.to_ndarray(format = 'gray')
        scaled = img.astype(np.float32) / 63
        frames.append(cls_img)
    container.close()
    return torch.from_numpy(np.stack(frames)).contiguous()


def decode_geo_topbot(payload = None):
    encoded = np.frombuffer(bz2.decompress(payload), dtype = np.uint16).reshape(PAIRS_PER_FILE, 2, NET_W)
    dx = (encoded >> 1).astype(np.int32) ^ -(encoded & 1).astype(np.int32)
    bounds = np.cumsum(dx, axis = 2).astype(np.int16)
    tops = bounds[(:, 0)].clip(0, NET_H)
    bots = bounds[(:, 1)].clip(0, NET_H)
    masks = np.zeros((PAIRS_PER_FILE, NET_H, NET_W), dtype = np.uint8)
    for t in range(PAIRS_PER_FILE):
        for x in range(NET_W):
            top = int(tops[(t, x)])
            bot = int(bots[(t, x)])
            masks[(t, :top, x)] = 2
            masks[(t, top:bot, x)] = 0
            masks[(t, bot:, x)] = 4
    return masks


def render_geo_components(masks = None, payload = None, cls = None):
    raw = lzma.decompress(payload)
    offset = 0
    for t in range(PAIRS_PER_FILE):
        (num_components,) = struct.unpack_from('<H', raw, offset)
        offset += 2
        for _ in range(num_components):
            (num_samples,) = struct.unpack_from('<H', raw, offset)
            offset += 2
            samples = []
            y = 0
            left = 0
            right = 0
            for _ in range(num_samples):
                (dy, dl, dr) = struct.unpack_from('<hhh', raw, offset)
                offset += 6
                y += dy
                left += dl
                right += dr
                samples.append((y, left, right))
            if not samples:
                continue
            for i in range(len(samples) - 1):
                (y0, l0, r0) = samples[i]
                (y1, l1, r1) = samples[i + 1]
                dy = max(1, y1 - y0)
                for yy in range(y0, y1):
                    a = (yy - y0) / dy
                    lft = round(l0 + (l1 - l0) * a)
                    rgt = round(r0 + (r1 - r0) * a)
                    masks[(t, yy, max(0, lft):min(NET_W, rgt))] = cls
            (yy, lft, rgt) = samples[-1]
            masks[(t, yy, max(0, lft):min(NET_W, rgt))] = cls


def decode_geo_masks(top_payload = None, vehicle_payload = None, lane_payload = None):
    masks = decode_geo_topbot(top_payload)
    render_geo_components(masks, vehicle_payload, 3)
    render_geo_components(masks, lane_payload, 1)
    return torch.from_numpy(masks).contiguous()


def split_geo_payload(payload = None):
    offset = len(GEO_MAGIC)
    (model_len, pose_len, top_len, vehicle_len, lane_len) = struct.unpack_from('<IIIII', payload, offset)
    offset += 20
    model_data = payload[offset:offset + model_len]
    offset += model_len
    pose_data = payload[offset:offset + pose_len]
    offset += pose_len
    top_data = payload[offset:offset + top_len]
    offset += top_len
    vehicle_data = payload[offset:offset + vehicle_len]
    offset += vehicle_len
    lane_data = payload[offset:offset + lane_len]
    return (model_data, pose_data, top_data, vehicle_data, lane_data)


def split_qpack_payload(payload = None):
    offset = len(PACK_MAGIC)
    (mask_len, model_len, pose_len) = struct.unpack_from('<III', payload, offset)
    offset += 12
    mask_data = payload[offset:offset + mask_len]
    offset += mask_len
    model_data = payload[offset:offset + model_len]
    offset += model_len
    pose_data = payload[offset:offset + pose_len]
    if len(mask_data) != mask_len and len(model_data) != model_len or len(pose_data) != pose_len:
        raise ValueError('short QPK1 payload')
    return (mask_data, model_data, pose_data)


def split_compact_payload(payload = None):
    mask_end = COMPACT_MASK_BODY_BYTES
    model_end = mask_end + COMPACT_MODEL_BODY_BYTES
    if len(payload) <= model_end:
        raise ValueError('short compact payload')
    mask_data = SEM_TOPBAND_BR_MAGIC + payload[:mask_end]
    model_data = QROW_GROUPED4_MODEL_BR_MAGIC + payload[mask_end:model_end]
    pose_data = payload[model_end:]
    return (mask_data, model_data, pose_data)


def _decode_qpose_raw(q = None):
    poses = np.empty(q.shape, dtype = np.float32)
    poses[(:, 0)] = q[(:, 0)].astype(np.float32) / 512 + 20
    poses[(:, 1:)] = q[(:, 1:)].view(np.int16).astype(np.float32) / 2048
    return torch.from_numpy(poses).float()


def decode_control_payload(payload = None):
    pass
# WARNING: Decompyle incomplete


def decode_poseq_payload(payload = None):
    (poses, _biases) = decode_control_payload(payload)
    return poses


def load_pose_frames(data_dir = None):
    poseq_path = data_dir / 'poseq.bin.br'
    short_poseq_path = data_dir / 'c'
    q8_path = data_dir / 'pose.q8.br'
    q12_path = data_dir / 'pose.q12.br'
    legacy_path = data_dir / 'pose.npy.br'
    if poseq_path.exists():
        return decode_poseq_payload(poseq_path.read_bytes())
    if None.exists():
        return decode_poseq_payload(short_poseq_path.read_bytes())
    if None.exists():
        raw = brotli.decompress(q8_path.read_bytes())
        magic = b'QPOSE8\x00'
        if not raw.startswith(magic):
            raise ValueError('invalid pose.q8.br header')
        offset = len(magic)
        mins = np.frombuffer(raw, dtype = np.float32, count = 6, offset = offset).copy()
        offset += 24
        scales = np.frombuffer(raw, dtype = np.float32, count = 6, offset = offset).copy()
        offset += 24
        q = np.frombuffer(raw, dtype = np.uint8, offset = offset).reshape(-1, 6)
        poses = q.astype(np.float32) * scales[(None, :)] + mins[(None, :)]
        return torch.from_numpy(poses).float()
    if None.exists():
        raw = brotli.decompress(q12_path.read_bytes())
        magic = b'QPOSE12\x00'
        if not raw.startswith(magic):
            raise ValueError('invalid pose.q12.br header')
        offset = len(magic)
        mins = np.frombuffer(raw, dtype = np.float32, count = 6, offset = offset).copy()
        offset += 24
        scales = np.frombuffer(raw, dtype = np.float32, count = 6, offset = offset).copy()
        offset += 24
        packed = np.frombuffer(raw, dtype = np.uint8, offset = offset).reshape(-1, 3)
        a = packed[(:, 0)].astype(np.uint16) | (packed[(:, 1)].astype(np.uint16) & 15) << 8
        b = packed[(:, 1)].astype(np.uint16) >> 4 & 15 | packed[(:, 2)].astype(np.uint16) << 4
        q = np.empty(packed.shape[0] * 2, dtype = np.uint16)
        q[0::2] = a
        q[1::2] = b
        q = q.reshape(-1, 6)
        poses = q.astype(np.float32) * scales[(None, :)] + mins[(None, :)]
        return torch.from_numpy(poses).float()
    f = None(legacy_path, 'rb')
    pose_bytes = brotli.decompress(f.read())
    None(None, None)
# WARNING: Decompyle incomplete


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
