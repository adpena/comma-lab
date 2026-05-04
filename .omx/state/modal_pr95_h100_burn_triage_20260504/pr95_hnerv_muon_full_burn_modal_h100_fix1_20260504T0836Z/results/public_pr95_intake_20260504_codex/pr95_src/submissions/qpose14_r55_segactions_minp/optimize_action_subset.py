#!/usr/bin/env python
import argparse
import io
import math
import os
import struct
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

ROOT = Path(__file__).resolve().parents[2]
SUB = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SUB))

from frame_utils import AVVideoDataset, DaliVideoDataset  # noqa: E402
from inflate import JointFrameGenerator, get_decoded_state_dict, load_encoded_mask_video  # noqa: E402
from modules import PoseNet, SegNet, posenet_sd_path, segnet_sd_path  # noqa: E402

ORIG_SIZE = 37_545_489
MASK_BYTES = 219_472


def read_payload(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        return zf.read("p")


def split_known_payload(payload: bytes):
    if payload.startswith(b"P3"):
        mask_len, model_len, action_len = struct.unpack_from("<IHH", payload, 2)
        cursor = 10
        mask = payload[cursor:cursor + mask_len]
        cursor += mask_len
        model = payload[cursor:cursor + model_len]
        cursor += model_len
        actions = payload[cursor:cursor + action_len]
        pose = payload[cursor + action_len:]
        return mask, model, actions, pose
    if len(payload) == 276362:
        return (
            payload[:MASK_BYTES],
            payload[MASK_BYTES:MASK_BYTES + 55_756],
            payload[MASK_BYTES + 55_756:MASK_BYTES + 55_756 + 236],
            payload[MASK_BYTES + 55_756 + 236:],
        )
    if len(payload) == 276520:
        return (
            payload[:MASK_BYTES],
            payload[MASK_BYTES:MASK_BYTES + 55_914],
            payload[MASK_BYTES + 55_914:MASK_BYTES + 55_914 + 236],
            payload[MASK_BYTES + 55_914 + 236:],
        )
    if len(payload) == 276641:
        return (
            payload[:MASK_BYTES],
            payload[MASK_BYTES:MASK_BYTES + 56_034],
            payload[MASK_BYTES + 56_034:MASK_BYTES + 56_034 + 236],
            payload[MASK_BYTES + 56_034 + 236:],
        )
    # PR #77-style split: legacy mask, compact qzs3 model, SG action stream, QP1 pose.
    for model_len in (55_756, 55_757, 55_914):
        for pose_len in (898, 899):
            if MASK_BYTES + model_len + pose_len >= len(payload):
                continue
            model = payload[MASK_BYTES:MASK_BYTES + model_len]
            actions = payload[MASK_BYTES + model_len:len(payload) - pose_len]
            pose = payload[len(payload) - pose_len:]
            try:
                brotli.decompress(model)
                brotli.decompress(actions)
                brotli.decompress(pose)
            except Exception:
                continue
            return payload[:MASK_BYTES], model, actions, pose
    raise ValueError(f"unknown packed payload length: {len(payload)}")


def read_uvarint(raw: bytes, cursor: int):
    shift = 0
    value = 0
    while True:
        byte = raw[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, cursor
        shift += 7


def write_uvarint(value: int, out: bytearray):
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return


def unpack_actions(actions_br: bytes):
    raw = brotli.decompress(actions_br)
    records = []
    if raw.startswith(b"SG2") or (len(raw) % 4 != 0 and len(raw) % 5 != 0):
        cursor = 3 if raw.startswith(b"SG2") else 0
        while cursor < len(raw):
            tile, cursor = read_uvarint(raw, cursor)
            count, cursor = read_uvarint(raw, cursor)
            frame = 0
            for idx in range(count):
                delta, cursor = read_uvarint(raw, cursor)
                frame = delta if idx == 0 else frame + delta
                action = raw[cursor]
                cursor += 1
                records.append((frame, tile, action))
    elif len(raw) % 4 == 0:
        for i in range(0, len(raw), 4):
            records.append((int.from_bytes(raw[i:i + 2], "little"), raw[i + 2], raw[i + 3]))
    elif len(raw) % 5 == 0:
        for i in range(0, len(raw), 5):
            records.append((int.from_bytes(raw[i:i + 2], "little"), int.from_bytes(raw[i + 2:i + 4], "little"), raw[i + 4]))
    return [r for r in records if 0 <= r[0] < 600]


def pack_fixed4(records):
    raw = bytearray()
    for frame, tile, action in records:
        raw += int(frame).to_bytes(2, "little")
        raw += int(tile).to_bytes(1, "little")
        raw += int(action).to_bytes(1, "little")
    return brotli.compress(bytes(raw), quality=11)


def pack_sg(records, header=False):
    by_tile = {}
    for frame, tile, action in records:
        by_tile.setdefault(tile, []).append((frame, action))
    raw = bytearray(b"SG2" if header else b"")
    for tile in sorted(by_tile):
        vals = sorted(by_tile[tile])
        write_uvarint(tile, raw)
        write_uvarint(len(vals), raw)
        prev = 0
        for idx, (frame, action) in enumerate(vals):
            write_uvarint(frame if idx == 0 else frame - prev, raw)
            raw.append(action)
            prev = frame
    return brotli.compress(bytes(raw), quality=11)


def pack_best(records):
    fixed = pack_fixed4(records)
    sg = pack_sg(records, header=False)
    sg2 = pack_sg(records, header=True)
    return min((fixed, sg, sg2), key=len)


def decode_pose(pose_br: bytes):
    raw = brotli.decompress(pose_br)
    if not raw.startswith(b"QP1"):
        raise ValueError("expected QP1 pose")
    first = np.frombuffer(raw[3:5], dtype=np.uint16, count=1)[0]
    vals = [int(first)]
    cursor = 5
    while cursor < len(raw):
        shift = 0
        acc = 0
        while True:
            byte = raw[cursor]
            cursor += 1
            acc |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
        delta = (acc >> 1) ^ -(acc & 1)
        vals.append(vals[-1] + delta)
    q = np.zeros((len(vals), 6), dtype=np.uint16)
    q[:, 0] = np.asarray(vals, dtype=np.uint16)
    pose = np.empty(q.shape, dtype=np.float32)
    pose[:, 0] = q[:, 0].astype(np.float32) / 512.0 + 20.0
    pose[:, 1:] = q[:, 1:].view(np.int16).astype(np.float32) / 2048.0
    return torch.from_numpy(pose).float()


def raw_cycle_384(x):
    x_up = F.interpolate(x, size=(874, 1164), mode="bilinear", align_corners=False)
    x_round = x_up.clamp(0, 255).round()
    return F.interpolate(x_round, size=(384, 512), mode="bilinear", align_corners=False)


def action_specs(device):
    specs = []
    directions = [
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (-0.35, 0.15, 0.45),
        (0.25, 0.15, -0.20),
    ]
    for vec in directions:
        v = torch.tensor(vec, dtype=torch.float32, device=device).view(3, 1, 1)
        v = v / v.abs().max().clamp_min(1e-6)
        for amp in (2.0, 4.0, 6.0, 8.0, 12.0, 16.0):
            specs.append(v * amp)
            specs.append(-v * amp)
    return torch.stack(specs, 0)


def load_base(archive: Path, device, batch_size):
    mask_br, model_br, actions_br, pose_br = split_known_payload(read_payload(archive))
    gen = JointFrameGenerator().to(device)
    gen.load_state_dict(get_decoded_state_dict(brotli.decompress(model_br), device), strict=True)
    gen.eval()

    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp:
        tmp.write(brotli.decompress(mask_br))
        tmp_path = tmp.name
    masks = load_encoded_mask_video(tmp_path).long()
    os.remove(tmp_path)
    poses = decode_pose(pose_br)

    f1s, f2s = [], []
    with torch.no_grad():
        for start in range(0, masks.shape[0], batch_size):
            m = masks[start:start + batch_size].to(device)
            p = poses[start:start + batch_size].to(device)
            f1, f2 = gen(m, p)
            f1s.append(raw_cycle_384(f1).cpu())
            f2s.append(raw_cycle_384(f2).cpu())
    return mask_br, model_br, actions_br, pose_br, torch.cat(f1s, 0), torch.cat(f2s, 0)


@torch.inference_mode()
def load_targets(device, batch_size):
    files = (ROOT / "public_test_video_names.txt").read_text().splitlines()
    ds_cls = DaliVideoDataset if device.type == "cuda" else AVVideoDataset
    ds = ds_cls(files, data_dir=ROOT / "videos", batch_size=batch_size, device=device)
    ds.prepare_data()
    seg = SegNet().eval().to(device)
    seg.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    pose = PoseNet().eval().to(device)
    pose.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    masks = []
    poses = []
    for _, _, batch in ds:
        batch_bct = einops.rearrange(batch, "b t h w c -> b t c h w").float().to(device)
        masks.append(seg(seg.preprocess_input(batch_bct)).argmax(1).cpu())
        poses.append(pose(pose.preprocess_input(batch_bct))["pose"][..., :6].cpu())
    return seg, pose, torch.cat(masks, 0), torch.cat(poses, 0)


def frame_from_records(base2, records_by_frame, frame, specs, tile):
    out = base2[frame].clone()
    tw = out.shape[-1] // tile
    for tile_id, action in records_by_frame.get(frame, []):
        y0 = (tile_id // tw) * tile
        x0 = (tile_id % tw) * tile
        out[:, y0:y0 + tile, x0:x0 + tile] = (out[:, y0:y0 + tile, x0:x0 + tile] + specs[action].cpu()).clamp(0, 255)
    return out


def predict_one(seg, pose, target_mask, target_pose, f1, f2, device):
    with torch.no_grad():
        pred_seg = seg(f2.unsqueeze(0).to(device)).argmax(1).cpu()[0]
        pair = torch.stack([f1, f2], 0).unsqueeze(0).to(device)
        pred_pose = pose(pose.preprocess_input(pair))["pose"][..., :6].cpu()[0]
    bad = (pred_seg != target_mask).float().mean().item()
    pose_err = (pred_pose - target_pose).pow(2).mean().item()
    return bad, pose_err


def estimate_score(seg_bad, pose_err, archive_bytes):
    return 100.0 * seg_bad + math.sqrt(10.0 * pose_err) + 25.0 * archive_bytes / ORIG_SIZE


def zip_size_for_payload(payload):
    return len(payload) + 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--candidate-archive", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tile", type=int, default=32)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--min-gain", type=float, default=0.0000002)
    ap.add_argument("--out", type=Path, default=SUB / "action_subset_archive.zip")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    mask_br, model_br, actions_br, pose_br, base1, base2 = load_base(args.archive, device, args.batch_size)
    _, _, cand_actions_br, _ = split_known_payload(read_payload(args.candidate_archive))
    target_seg_model, target_pose_model, target_masks, target_pose = load_targets(device, args.batch_size)
    specs = action_specs(device)

    current = list(dict.fromkeys(unpack_actions(actions_br)))
    candidates = list(dict.fromkeys(unpack_actions(cand_actions_br)))
    pool = list(dict.fromkeys(current + candidates))
    records_by_frame = {}
    for frame, tile_id, action in current:
        records_by_frame.setdefault(frame, []).append((tile_id, action))

    frame_seg = np.zeros(base2.shape[0], dtype=np.float64)
    frame_pose = np.zeros(base2.shape[0], dtype=np.float64)
    for frame in range(base2.shape[0]):
        f2 = frame_from_records(base2, records_by_frame, frame, specs, args.tile)
        s, p = predict_one(target_seg_model, target_pose_model, target_masks[frame], target_pose[frame], base1[frame], f2, device)
        frame_seg[frame] = s
        frame_pose[frame] = p
    payload = mask_br + model_br + pack_best(current) + pose_br
    cur_score = estimate_score(frame_seg.mean(), frame_pose.mean(), zip_size_for_payload(payload))
    print(f"start records={len(current)} pool={len(pool)} actions_bytes={len(pack_best(current))} archive={zip_size_for_payload(payload)} seg={frame_seg.mean():.8f} pose={frame_pose.mean():.8f} score={cur_score:.8f}", flush=True)

    current_set = set(current)
    for pass_idx in range(args.passes):
        improved = 0
        for rec in pool:
            if rec in current_set:
                trial_set = current_set - {rec}
            else:
                trial_set = current_set | {rec}
            frame = rec[0]
            trial_records = sorted(trial_set)
            trial_by_frame = dict(records_by_frame)
            trial_by_frame[frame] = [(t, a) for f, t, a in trial_records if f == frame]
            f2 = frame_from_records(base2, trial_by_frame, frame, specs, args.tile)
            new_frame_seg, new_frame_pose = predict_one(
                target_seg_model, target_pose_model, target_masks[frame], target_pose[frame], base1[frame], f2, device
            )
            trial_seg = frame_seg.mean() + (new_frame_seg - frame_seg[frame]) / len(frame_seg)
            trial_pose = frame_pose.mean() + (new_frame_pose - frame_pose[frame]) / len(frame_pose)
            trial_payload = mask_br + model_br + pack_best(trial_records) + pose_br
            trial_score = estimate_score(trial_seg, trial_pose, zip_size_for_payload(trial_payload))
            gain = cur_score - trial_score
            if gain > args.min_gain:
                action = "drop" if rec in current_set else "add"
                current_set = trial_set
                current = trial_records
                records_by_frame = {}
                for f, t, a in current:
                    records_by_frame.setdefault(f, []).append((t, a))
                frame_seg[frame] = new_frame_seg
                frame_pose[frame] = new_frame_pose
                cur_score = trial_score
                improved += 1
                print(f"{action} pass={pass_idx} rec={rec} records={len(current)} actions_bytes={len(pack_best(current))} seg={frame_seg.mean():.8f} pose={frame_pose.mean():.8f} score={cur_score:.8f} gain={gain:.8f}", flush=True)
        print(f"pass_done={pass_idx} improved={improved} records={len(current)} score={cur_score:.8f}", flush=True)
        if improved == 0:
            break

    final_actions = pack_best(current)
    final_payload = mask_br + model_br + final_actions + pose_br
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("p", final_payload)
    print(f"wrote {args.out} archive={args.out.stat().st_size} actions_bytes={len(final_actions)} records={len(current)}", flush=True)


if __name__ == "__main__":
    main()
