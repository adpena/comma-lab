# pyc-recovery pass2: STUB unreconstructible -- DERIVATIVE source never committed
# Sherlock pass2 investigation (2026-05-05):
#   - Searched git rev-list --all + git fsck --lost-found: only STUB blobs exist
#   - This file was a runtime ARTIFACT produced by experiments/build_pr85_stbm1br_mask_recode_candidate.py
#     (orig blob 8b931a313de3d3e605d4d607cc7a3b888a2a8aee, accessible via git cat-file -p)
#   - The runtime is OUR derivative of PR85's inflate.py with a narrow STBM1BR mask branch added
#   - Both source-of-truth pieces are intact:
#       1. PR85 inflate.py (vendored from upstream): experiments/results/public_pr85_intake_20260503_codex/replay_submission/inflate.py
#       2. STBM1BR codec module: src/tac/stbm1br_mask_codec.py (esp. decode_stbm1br_mask_segment)
#   - To rebuild this runtime: take PR85 inflate.py, identify the mask-loading function,
#     add a magic-byte sniff for STBM1BR_MAGIC = b"STBM1BR\x00", route those segments
#     to tac.stbm1br_mask_codec.decode_stbm1br_mask_segment, all other branches unchanged
#   - The .recovery_spec.json companion has dis() bytecode ground truth for every code object
#     including FP4Codebook, JointFrameGenerator, load_stbm1br_mask, load_compact_archive_bundle, main
#   - Memory ref: .omx/research/pr85_stbm1br_mask_recode_20260504_worker.md (see "Mitigation landed:" section)
"""RECOVERY STUB - DERIVATIVE runtime, hand-reconstruct from PR85 inflate + tac.stbm1br_mask_codec.

This file is a runtime ARTIFACT (not source-controlled) produced locally by the
build_pr85_stbm1br_mask_recode_candidate.py builder. The .pyc was the only
forensic trace.

Reconstruction recipe (Sherlock pass2 verdict, NOT executed):

    pr85_inflate = Path("experiments/results/public_pr85_intake_20260503_codex/replay_submission/inflate.py").read_text()
    # Find PR85's mask-loading function. Add a STBM1BR_MAGIC sniff branch
    # that calls tac.stbm1br_mask_codec.decode_stbm1br_mask_segment when seen.
    # All other PR85 inflate logic unchanged.

The companion ``inflate.recovery_spec.json`` contains dis() bytecode for every
code object including the model classes (FP4Codebook, SepResBlock,
JointFrameGenerator) and the mask loaders (load_range_mask, load_stbm1br_mask,
load_encoded_mask_video) and the full main() function.

This stub remains a no-op import; replace once exact reconstruction is done.
Until then, the running pyc + pyc artifact in __pycache__/ are still callable
on the same Python 3.12 ABI.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/results/pr85_stbm1br_mask_recode_20260504_worker/replay_submission_stbm/inflate.py'
__recovery_spec__ = 'inflate.recovery_spec.json'
__recovery_ast_error__ = '34:104: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: inflate.cpython-312.pyc (Python 3.12)

import io
import json
import os
import shutil
import struct
import subprocess
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


def _unsplit_bytes_to_tensor(payload_data, pos, nbytes = None, dtype = None, shape = None, device = ('payload_data', bytes, 'pos', int, 'nbytes', int)):
    half = (nbytes + 1) // 2
    a = np.frombuffer(payload_data[pos:pos + nbytes], dtype = np.uint8)
    out = np.empty(nbytes, dtype = np.uint8)
    out[0::2] = a[:half]
    out[1::2] = a[half:]
    t = torch.frombuffer(bytearray(out.tobytes()), dtype = dtype).clone().reshape(shape).to(device)
    return (t, pos + nbytes)


def _unhilo_packed(payload_data = None, pos = None, packed_len = None, device = ('payload_data', bytes, 'pos', int, 'packed_len', int)):
    half = packed_len // 2
    hp = np.frombuffer(payload_data[pos:pos + half], dtype = np.uint8)
    pos += half
    lp = np.frombuffer(payload_data[pos:pos + half], dtype = np.uint8)
    pos += half
    hi = np.empty(half * 2, dtype = np.uint8)
    lo = np.empty(half * 2, dtype = np.uint8)
    hi[0::2] = hp >> 4 & 15
    hi[1::2] = hp & 15
    lo[0::2] = lp >> 4 & 15
    lo[1::2] = lp & 15
    packed = (hi[:packed_len] << 4 | lo[:packed_len]).astype(np.uint8)
    return (torch.frombuffer(bytearray(packed.tobytes()), dtype = torch.uint8).clone().to(device), pos)


def get_decoded_state_dict_custom(payload_data = None, device = None):
    magic = payload_data[:3]
    if magic not in (b'QM0', b'QH0'):
        return None
    pos = 3
    hilosplit = magic == b'QH0'
    state_dict = { }
    probe = JointFrameGenerator()
    covered = set()
# WARNING: Decompyle incomplete


def get_decoded_state_dict(payload_data = None, device = None):
    custom = get_decoded_state_dict_custom(payload_data, device)
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


def load_range_mask(mask_payload = None):
    codec_src = Path(__file__).with_name('range_mask_codec.cpp')
    if not codec_src.exists():
        raise RuntimeError(f'''missing range-mask decoder source: {codec_src}''')
    if len(mask_payload) < 20:
        raise RuntimeError('truncated range-mask payload')
    t_count = int.from_bytes(mask_payload[4:8], 'little')
    mask_h = int.from_bytes(mask_payload[8:12], 'little')
    mask_w = int.from_bytes(mask_payload[12:16], 'little')
    td = tempfile.TemporaryDirectory()
    td_path = Path(td)
    exe = td_path / 'range_mask_codec'
    compiler_candidates = [
        os.environ.get('CXX', ''),
        'c++',
        'g++',
        'clang++']
    last_error = None
# WARNING: Decompyle incomplete


def load_stbm1br_mask(mask_payload = None):
    decode_stbm1br_mask_segment = decode_stbm1br_mask_segment
    import tac.stbm1br_mask_codec
    arr = decode_stbm1br_mask_segment(mask_payload, expected_shape = (600, 384, 512))
    return torch.from_numpy(arr).contiguous()


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


def unpack_pose_q12(raw = None):
    if raw[:4] == b'PQB1':
        (n, d, bits) = struct.unpack_from('<HHBxxx', raw, 4)
        off = 12
        mn = np.frombuffer(raw, dtype = '<f4', count = d, offset = off).astype(np.float32)
        off += 4 * d
        scale = np.frombuffer(raw, dtype = '<f4', count = d, offset = off).astype(np.float32)
        off += 4 * d
        data = np.frombuffer(raw, dtype = np.uint8, offset = off)
        total = n * d
        vals = np.empty(total, dtype = np.uint16)
        bitbuf = 0
        bitcount = 0
        byte_i = 0
        mask = (1 << bits) - 1
        for i in range(total):
            if bitcount < bits:
                bitbuf |= int(data[byte_i]) << bitcount
                byte_i += 1
                bitcount += 8
                if bitcount < bits:
                    continue
            vals[i] = bitbuf & mask
            bitbuf >>= bits
            bitcount -= bits
        return mn[(None, :)] + vals.reshape(n, d).astype(np.float32) * scale[(None, :)]
    if None[:4] != b'PQ12':
        raise ValueError('bad pose_q magic')
    (n, d) = struct.unpack_from('<HH', raw, 4)
    off = 8
    mn = np.frombuffer(raw, dtype = '<f4', count = d, offset = off).astype(np.float32)
    off += 4 * d
    scale = np.frombuffer(raw, dtype = '<f4', count = d, offset = off).astype(np.float32)
    off += 4 * d
    p = np.frombuffer(raw, dtype = np.uint8, offset = off).reshape(-1, 3).astype(np.uint16)
    q0 = p[(:, 0)] | (p[(:, 1)] & 15) << 8
    q1 = p[(:, 1)] >> 4 | p[(:, 2)] << 4
    q = np.empty(p.shape[0] * 2, dtype = np.uint16)
    q[0::2] = q0
    q[1::2] = q1
    return mn[(None, :)] + q[:n * d].reshape(n, d).astype(np.float32) * scale[(None, :)]


def load_pose_frames(path_or_dir = None):
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / 'pose.npy.br'
# WARNING: Decompyle incomplete


def load_pose_frames_from_payload(encoded_pose = None):
    pose_bytes = encoded_pose if encoded_pose[:4] in (b'PQ12', b'PQB1') else brotli.decompress(encoded_pose)
    if pose_bytes[:4] in (b'PQ12', b'PQB1'):
        return torch.from_numpy(unpack_pose_q12(pose_bytes)).float()
    if None[:4] == b'P1D1':
        n = 600
        pos = 4
        cnt = pose_bytes[pos]
        pos += 1
        dims = []
        lens = []
        for _ in range(cnt):
            dims.append(int(pose_bytes[pos]))
            pos += 1
            lens.append(int.from_bytes(pose_bytes[pos:pos + 2], 'little'))
            pos += 2
        pose_np = np.zeros((n, 6), dtype = np.float32)
        for d, ln in zip(dims, lens):
            stream = pose_bytes[pos:pos + ln]
            pos += ln
            vals = np.empty(n, dtype = np.uint32)
            acc = 0
            shift = 0
            j = 0
            for byte in stream:
                acc |= (int(byte) & 127) << shift
                if byte & 128:
                    shift += 7
                    continue
                vals[j] = acc
                j += 1
                acc = 0
                shift = 0
                if not j >= n:
                    continue
                stream
        delta = (vals.astype(np.int32) >> 1 ^ -(vals.astype(np.int32) & 1)).astype(np.int32)
        q = np.cumsum(delta)
        if int(d) == 0:
            pose_np[(:, 0)] = q.astype(np.float32) / 512 + 20
            continue
        q = q.clip(-32768, 32767).astype(np.int16)
        pose_np[(:, int(d))] = q.astype(np.float32) / 2048
        continue
        return torch.from_numpy(pose_np).float()
    return torch.from_numpy(np.load(io.BytesIO(pose_bytes))).float()


def _post_pair_tensor(value = None, default = None, device = None):
    pass
# WARNING: Decompyle incomplete


def _post_stage_from_defs(defs = None, choices_data = None, device = None):
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
                for chan, ci in (('all', -1), ('r', 0), ('g', 1), ('b', 2)):
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


def load_post_codes(data_dir = None, device = None, encoded_post_codes = None):
    path = data_dir / 'post_codes.br'
# WARNING: Decompyle incomplete


def load_postprocess(data_dir = None, device = None, encoded_post_codes = None):
    pass
# WARNING: Decompyle incomplete


def load_compact_archive_bundle(data_dir = None):
    '''Load one-file ZIP member used by the smallest packaged variants.

    Member ``x`` stores three little-endian uint32 lengths followed by the
    brotli-compressed mask, model, pose, and post-code payloads.  Keeping these
    already-compressed blobs in one ZIP member avoids three ZIP local/central
    headers and long member names.
    '''
    pass
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
