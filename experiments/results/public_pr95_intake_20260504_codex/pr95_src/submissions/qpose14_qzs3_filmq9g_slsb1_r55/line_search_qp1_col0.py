#!/usr/bin/env python
import argparse
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import brotli
import einops
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frame_utils import DaliVideoDataset
from modules import PoseNet, posenet_sd_path
from inflate import JointFrameGenerator, get_decoded_state_dict, load_encoded_mask_video

ORIGINAL_SIZE = 37_545_489
MASK_BYTES = 219_472
MODEL_BYTES = 56093


def decode_qp1(data: bytes) -> np.ndarray:
    vals = [int(np.frombuffer(data[3:5], dtype=np.uint16, count=1)[0])]
    cursor = 5
    while cursor < len(data):
        shift = 0
        acc = 0
        while True:
            byte = data[cursor]
            cursor += 1
            acc |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
        vals.append(vals[-1] + ((acc >> 1) ^ -(acc & 1)))
    return np.asarray(vals, dtype=np.int64)


def encode_qp1(col0: np.ndarray) -> bytes:
    col0 = np.asarray(col0, dtype=np.int64)
    out = bytearray(b"QP1")
    out.extend(int(col0[0]).to_bytes(2, "little"))
    for d in np.diff(col0).astype(np.int64):
        x = int((d << 1) ^ (d >> 63))
        while x >= 128:
            out.append((x & 127) | 128)
            x >>= 7
        out.append(x)
    return brotli.compress(bytes(out), quality=11, lgwin=24)


def pose_from_col0(col0, device):
    pose = torch.zeros((col0.shape[0], 6), device=device)
    pose[:, 0] = col0.to(device=device, dtype=torch.float32) / 512.0 + 20.0
    return pose


def load_masks(mask_br):
    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp:
        tmp.write(brotli.decompress(mask_br))
        tmp_path = tmp.name
    try:
        return load_encoded_mask_video(tmp_path)
    finally:
        os.remove(tmp_path)


def make_frames(generator, masks, pose):
    f1, f2 = generator(masks.long(), pose.float())
    f1 = F.interpolate(f1, size=(874, 1164), mode="bilinear", align_corners=False)
    f2 = F.interpolate(f2, size=(874, 1164), mode="bilinear", align_corners=False)
    return torch.stack([f1, f2], dim=1).clamp(0, 255).round()


def pose_outputs(posenet, pairs_bhwc):
    x = einops.rearrange(pairs_bhwc, "b t h w c -> b t c h w").float()
    return posenet(posenet.preprocess_input(x))["pose"][..., :6].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--candidate-chunk", type=int, default=32)
    ap.add_argument("--radii", default="1,2,3,5,8")
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args()
    device = torch.device(args.device)
    archive_dir = args.submission_dir / "archive"
    payload = (archive_dir / "p").read_bytes()
    mask_br = payload[:MASK_BYTES]
    model_br = payload[MASK_BYTES:MASK_BYTES + MODEL_BYTES]
    pose_br = payload[MASK_BYTES + MODEL_BYTES:]
    col0 = decode_qp1(brotli.decompress(pose_br))

    masks = load_masks(mask_br)
    generator = JointFrameGenerator().to(device).eval()
    generator.load_state_dict(get_decoded_state_dict(brotli.decompress(model_br), device), strict=True)
    posenet = PoseNet().to(device).eval()
    posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    for p in list(generator.parameters()) + list(posenet.parameters()):
        p.requires_grad_(False)

    ds = DaliVideoDataset(["0.mkv"], data_dir=ROOT / "videos", batch_size=args.batch_size, device=device)
    ds.prepare_data()
    gt = torch.cat([batch.cpu() for _, _, batch in tqdm(ds, desc="GT")]).contiguous()
    target = []
    with torch.inference_mode():
        for start in range(0, gt.shape[0], args.batch_size):
            target.append(pose_outputs(posenet, gt[start:start + args.batch_size].to(device)))
    target = torch.cat(target)
    n = gt.shape[0]

    def eval_col(vals):
        total = 0.0
        with torch.inference_mode():
            for start in range(0, n, args.batch_size):
                end = min(n, start + args.batch_size)
                idx = torch.arange(start, end)
                pose = pose_from_col0(torch.from_numpy(vals[start:end]), device)
                frames = make_frames(generator, masks.index_select(0, idx).to(device), pose)
                pred = pose_outputs(posenet, einops.rearrange(frames, "b t c h w -> b t h w c"))
                total += (pred - target[start:end]).pow(2).mean(dim=1).sum().item()
        return total / n

    def objective(vals):
        pose = eval_col(vals)
        pose_bytes = len(encode_qp1(vals))
        archive_size = MASK_BYTES + MODEL_BYTES + pose_bytes + 100
        return math.sqrt(10.0 * pose) + 25.0 * archive_size / ORIGINAL_SIZE, pose, pose_bytes, archive_size

    best = col0.copy()
    best_obj, best_pose, best_bytes, best_size = objective(best)
    print(f"baseline obj={best_obj:.9f} pose={best_pose:.12f} pose_bytes={best_bytes} size={best_size}", flush=True)
    cur = best.copy()
    for radius in [int(x) for x in args.radii.split(",") if x.strip()]:
        deltas = torch.arange(-radius, radius + 1, device=device)
        for pass_idx in range(args.passes):
            changed = 0
            for start in tqdm(range(0, n, args.batch_size), desc=f"r{radius} p{pass_idx+1}", leave=False):
                end = min(n, start + args.batch_size)
                b = end - start
                base = torch.from_numpy(cur[start:end]).to(device)
                cand = torch.clamp(base[:, None] + deltas[None, :], 0, 65535)
                c = cand.shape[1]
                best_dist = torch.full((b,), float("inf"), device=device)
                best_val = base.clone()
                masks_batch = masks[start:end].to(device)
                target_batch = target[start:end]
                for c_start in range(0, c, args.candidate_chunk):
                    c_end = min(c, c_start + args.candidate_chunk)
                    cand_chunk = cand[:, c_start:c_end]
                    cc = c_end - c_start
                    masks_rep = masks_batch.repeat_interleave(cc, dim=0)
                    frames = make_frames(generator, masks_rep, pose_from_col0(cand_chunk.reshape(-1), device))
                    pred = pose_outputs(posenet, einops.rearrange(frames, "b t c h w -> b t h w c"))
                    dist = (pred - target_batch.repeat_interleave(cc, dim=0)).pow(2).mean(dim=1).reshape(b, cc)
                    chunk_dist, chunk_pick = dist.min(dim=1)
                    better = chunk_dist < best_dist
                    best_dist = torch.where(better, chunk_dist, best_dist)
                    best_val = torch.where(better, cand_chunk[torch.arange(b, device=device), chunk_pick], best_val)
                    del frames, pred, dist
                new = best_val.cpu().numpy()
                changed += int((new != cur[start:end]).sum())
                cur[start:end] = new
            obj, pose, byte_count, size = objective(cur)
            print(f"radius={radius} pass={pass_idx+1} obj={obj:.9f} pose={pose:.12f} pose_bytes={byte_count} size={size} changed={changed} best={best_obj:.9f}", flush=True)
            if obj < best_obj:
                best_obj, best_pose, best_bytes, best_size = obj, pose, byte_count, size
                best = cur.copy()
            else:
                cur = best.copy()
                break

    new_pose_br = encode_qp1(best)
    new_payload = mask_br + model_br + new_pose_br
    (archive_dir / "p").write_bytes(new_payload)
    with zipfile.ZipFile(args.submission_dir / "archive.zip", "w", compression=zipfile.ZIP_STORED) as z:
        z.write(archive_dir / "p", arcname="p")
    print(f"saved obj={best_obj:.9f} pose={best_pose:.12f} pose_bytes={best_bytes} size={(args.submission_dir / 'archive.zip').stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
