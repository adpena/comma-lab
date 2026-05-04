#!/usr/bin/env python
import argparse
import io
import os
import sys
import tempfile
import zipfile
import struct
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

from frame_utils import DaliVideoDataset, AVVideoDataset, camera_size, seq_len  # noqa: E402
from modules import PoseNet, SegNet, posenet_sd_path, segnet_sd_path  # noqa: E402
from inflate import (  # noqa: E402
    JointFrameGenerator,
    get_decoded_state_dict,
    load_encoded_mask_video,
)

ORIG_SIZE = 37_545_489
BASE_ARCHIVE_BYTES = 276_564
BASE_SEG = 0.00061000
BASE_POSE = 0.00048597
MINP_ARCHIVE_BYTES = 276_741
MASK_BYTES = 219_472
MODEL_BYTES = 56_034
ACTION_BYTES = 236
ACTION_FREE_ARCHIVE_BYTES = MINP_ARCHIVE_BYTES - ACTION_BYTES


def read_packed_archive(submission_dir: Path):
    global ACTION_FREE_ARCHIVE_BYTES
    with zipfile.ZipFile(submission_dir / "archive.zip", "r") as zf:
        payload = zf.read("p")
    if len(payload) in (276641, 276520, 276362, 276381, 276379, 276574, 276749) or 276900 <= len(payload) <= 278000:
        if len(payload) == 276641:
            model_bytes = MODEL_BYTES
            action_bytes = 236
        elif len(payload) == 276520:
            model_bytes = 55_914
            action_bytes = 236
        elif len(payload) == 276362:
            model_bytes = 55_756
            action_bytes = 236
        elif len(payload) == 276381:
            model_bytes = 55_756
            action_bytes = 255
        elif len(payload) == 276574:
            model_bytes = 55_756
            action_bytes = 448
        elif len(payload) == 276749:
            model_bytes = 55_756
            action_bytes = 623
        elif 276900 <= len(payload) <= 278000:
            model_bytes = 55_756
            action_bytes = len(payload) - MASK_BYTES - model_bytes - 898
        else:
            model_bytes = 55_756
            action_bytes = 253
        cursor = 0
        mask_br_data = payload[cursor:cursor + MASK_BYTES]
        cursor += MASK_BYTES
        model_br_data = payload[cursor:cursor + model_bytes]
        cursor += model_bytes
        actions_br_data = payload[cursor:cursor + action_bytes]
        pose_q_br_data = payload[cursor + action_bytes:]
        ACTION_FREE_ARCHIVE_BYTES = len(payload) + 100 - len(actions_br_data)
        return mask_br_data, model_br_data, actions_br_data, pose_q_br_data
    if payload.startswith(b"P3"):
        mask_len, model_len, action_len = struct.unpack_from("<IHH", payload, 2)
        cursor = 10
        mask_br_data = payload[cursor:cursor + mask_len]
        cursor += mask_len
        model_br_data = payload[cursor:cursor + model_len]
        cursor += model_len
        actions_br_data = payload[cursor:cursor + action_len]
        pose_q_br_data = payload[cursor + action_len:]
        ACTION_FREE_ARCHIVE_BYTES = len(payload) + 100 - len(actions_br_data)
        return mask_br_data, model_br_data, actions_br_data, pose_q_br_data
    raise ValueError(f"unexpected minp payload length/header: {len(payload)}")


def decode_qp1_pose(pose_q_br_data: bytes) -> torch.Tensor:
    pose_raw = brotli.decompress(pose_q_br_data)
    if not pose_raw.startswith(b"QP1"):
        raise ValueError("expected QP1 pose payload")
    first = np.frombuffer(pose_raw[3:5], dtype=np.uint16, count=1)[0]
    vals = [int(first)]
    cursor = 5
    while cursor < len(pose_raw):
        shift = 0
        acc = 0
        while True:
            byte = pose_raw[cursor]
            cursor += 1
            acc |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
        delta = (acc >> 1) ^ -(acc & 1)
        vals.append(vals[-1] + delta)
    q_pose = np.zeros((len(vals), 6), dtype=np.uint16)
    q_pose[:, 0] = np.asarray(vals, dtype=np.uint16)
    pose_np = np.empty(q_pose.shape, dtype=np.float32)
    pose_np[:, 0] = q_pose[:, 0].astype(np.float32) / 512.0 + 20.0
    pose_np[:, 1:] = q_pose[:, 1:].view(np.int16).astype(np.float32) / 2048.0
    return torch.from_numpy(pose_np).float()


def load_generator(device: torch.device):
    mask_br_data, model_br_data, actions_br_data, pose_q_br_data = read_packed_archive(SUB)
    gen = JointFrameGenerator().to(device)
    gen.load_state_dict(get_decoded_state_dict(brotli.decompress(model_br_data), device), strict=True)
    gen.eval()

    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp:
        tmp.write(brotli.decompress(mask_br_data))
        tmp_path = tmp.name
    masks = load_encoded_mask_video(tmp_path).long()
    os.remove(tmp_path)
    poses = decode_qp1_pose(pose_q_br_data)
    return gen, masks, actions_br_data, poses


def unpack_records(actions_br_data: bytes):
    raw = brotli.decompress(actions_br_data)
    records = []
    def read_uvarint(cursor: int):
        shift = 0
        value = 0
        while True:
            byte = raw[cursor]
            cursor += 1
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                return value, cursor
            shift += 7

    if raw.startswith(b"SG2") or (len(raw) % 4 != 0 and len(raw) % 5 != 0):
        cursor = 3 if raw.startswith(b"SG2") else 0
        while cursor < len(raw):
            tile, cursor = read_uvarint(cursor)
            count, cursor = read_uvarint(cursor)
            frame = 0
            for idx in range(count):
                delta, cursor = read_uvarint(cursor)
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
    else:
        raise ValueError(f"unsupported action payload length: {len(raw)}")
    return records


def raw_cycle_384(x: torch.Tensor) -> torch.Tensor:
    x_up = F.interpolate(x, size=(874, 1164), mode="bilinear", align_corners=False)
    x_round = x_up.clamp(0, 255).round()
    return F.interpolate(x_round, size=(384, 512), mode="bilinear", align_corners=False)


def generate_eval_frames(gen, masks, poses, device, batch_size: int):
    f1s, f2s = [], []
    with torch.no_grad():
        for start in range(0, masks.shape[0], batch_size):
            m = masks[start:start + batch_size].to(device)
            p = poses[start:start + batch_size].to(device)
            f1, f2 = gen(m, p)
            f1s.append(raw_cycle_384(f1).cpu())
            f2s.append(raw_cycle_384(f2).cpu())
    return torch.cat(f1s, 0).clone(), torch.cat(f2s, 0).clone()


@torch.inference_mode()
def load_targets(device: torch.device, batch_size: int):
    files = (ROOT / "public_test_video_names.txt").read_text().splitlines()
    ds_cls = DaliVideoDataset if device.type == "cuda" else AVVideoDataset
    ds = ds_cls(files, data_dir=ROOT / "videos", batch_size=batch_size, device=device)
    ds.prepare_data()

    seg = SegNet().eval().to(device)
    seg.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    pose = PoseNet().eval().to(device)
    pose.load_state_dict(load_file(posenet_sd_path, device=str(device)))

    target_masks, target_pose, real_batches = [], [], []
    for _, _, batch in ds:
        batch_bct = einops.rearrange(batch, "b t h w c -> b t c h w").float().to(device)
        seg_in = seg.preprocess_input(batch_bct)
        target_masks.append(seg(seg_in).argmax(1).cpu())
        pose_in = pose.preprocess_input(batch_bct)
        target_pose.append(pose(pose_in)["pose"][..., :6].cpu())
        real_batches.append(raw_cycle_384(batch_bct.flatten(0, 1)).view(batch_bct.shape[0], 2, 3, 384, 512).cpu())
    return seg, pose, torch.cat(target_masks, 0), torch.cat(target_pose, 0), torch.cat(real_batches, 0)


def action_specs():
    specs = []
    directions = [
        ("y", (1.0, 1.0, 1.0)),
        ("r", (1.0, 0.0, 0.0)),
        ("g", (0.0, 1.0, 0.0)),
        ("b", (0.0, 0.0, 1.0)),
        ("rg", (1.0, 1.0, 0.0)),
        ("gb", (0.0, 1.0, 1.0)),
        ("rb", (1.0, 0.0, 1.0)),
        ("road", (-0.35, 0.15, 0.45)),
        ("sky", (0.25, 0.15, -0.20)),
    ]
    for name, vec in directions:
        v = torch.tensor(vec, dtype=torch.float32).view(3, 1, 1)
        v = v / v.abs().max().clamp_min(1e-6)
        for amp in (2.0, 4.0, 6.0, 8.0, 12.0, 16.0):
            specs.append((f"{name}+{amp:g}", v * amp))
            specs.append((f"{name}-{amp:g}", -v * amp))
    return specs


def pack_records(records):
    arr = bytearray()
    for frame, tile, action in records:
        arr += int(frame).to_bytes(2, "little")
        if tile < 256:
            arr += int(tile).to_bytes(1, "little")
            arr += int(action).to_bytes(1, "little")
        else:
            arr += int(tile).to_bytes(2, "little")
            arr += int(action).to_bytes(1, "little")
    return brotli.compress(bytes(arr), quality=11)


def write_uvarint(value: int, out: bytearray):
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return


def pack_records_sg(records, header=False):
    by_tile = {}
    for frame, tile, action in records:
        by_tile.setdefault(tile, []).append((frame, action))
    arr = bytearray(b"SG2" if header else b"")
    for tile in sorted(by_tile):
        vals = sorted(by_tile[tile])
        write_uvarint(int(tile), arr)
        write_uvarint(len(vals), arr)
        prev = 0
        for idx, (frame, action) in enumerate(vals):
            write_uvarint(int(frame) if idx == 0 else int(frame) - prev, arr)
            arr.append(int(action))
            prev = int(frame)
    return brotli.compress(bytes(arr), quality=11)


def pack_records_best(records):
    return min(
        (pack_records(records), pack_records_sg(records, False), pack_records_sg(records, True)),
        key=len,
    )


def apply_records_to_fake2(fake2, records, specs, tile):
    h, w = fake2.shape[-2:]
    tw = w // tile
    for frame, tile_id, action in records:
        y0 = (tile_id // tw) * tile
        x0 = (tile_id % tw) * tile
        fake2[frame, :, y0:y0 + tile, x0:x0 + tile] = (
            fake2[frame, :, y0:y0 + tile, x0:x0 + tile] + specs[action][1]
        ).clamp(0, 255)
    return fake2


def seg_predict(seg, frames, device, batch_size):
    outs = []
    with torch.no_grad():
        for start in range(0, frames.shape[0], batch_size):
            outs.append(seg(frames[start:start + batch_size].to(device)).argmax(1).cpu())
    return torch.cat(outs, 0)


def pose_dist(pose, target_pose, f1_eval, f2_eval, device, batch_size):
    vals = []
    with torch.no_grad():
        for start in range(0, f1_eval.shape[0], batch_size):
            pair = torch.stack([
                f1_eval[start:start + batch_size],
                f2_eval[start:start + batch_size],
            ], dim=1).to(device)
            pred = pose(pose.preprocess_input(pair))["pose"][..., :6].cpu()
            vals.append((pred - target_pose[start:start + batch_size]).pow(2).mean(dim=1))
    return torch.cat(vals).mean().item()


def pose_err_one(pose, target_pose_one, f1_one, f2_one, device):
    with torch.no_grad():
        pair = torch.stack([f1_one, f2_one], dim=0).unsqueeze(0).to(device)
        pred = pose(pose.preprocess_input(pair))["pose"][..., :6].cpu()[0]
    return (pred - target_pose_one).pow(2).mean().item()


def score(seg_dist, pose_distortion, payload_bytes):
    return 100.0 * seg_dist + (10.0 * pose_distortion) ** 0.5 + 25.0 * (ACTION_FREE_ARCHIVE_BYTES + payload_bytes) / ORIG_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--tile", type=int, default=32)
    ap.add_argument("--top-tiles", type=int, default=3)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--max-actions", type=int, default=600)
    ap.add_argument("--min-gain", type=float, default=0.00002)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--pose-gate", action="store_true")
    ap.add_argument("--pose-check", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    gen, masks, actions_br_data, poses = load_generator(device)
    seg, pose, target_masks, target_pose, _ = load_targets(device, args.batch_size)
    fake1, fake2 = generate_eval_frames(gen, masks, poses, device, args.batch_size)

    specs = action_specs()
    records = unpack_records(actions_br_data)
    fake2 = apply_records_to_fake2(fake2, records, specs, args.tile)
    cur_pred = seg_predict(seg, fake2, device, args.batch_size)
    cur_bad = cur_pred != target_masks
    cur_seg = cur_bad.float().mean().item()
    cur_pose_errs = None
    if args.pose_gate:
        cur_pose_errs = []
        for idx in range(fake1.shape[0]):
            cur_pose_errs.append(pose_err_one(pose, target_pose[idx], fake1[idx], fake2[idx], device))
        cur_pose_errs = np.asarray(cur_pose_errs, dtype=np.float64)
        cur_pose = float(cur_pose_errs.mean())
    else:
        cur_pose = BASE_POSE
    base_score = score(cur_seg, cur_pose, len(pack_records_best(records)))
    cur_score = base_score
    print(
        f"minp_start existing_actions={len(records)} action_payload={len(pack_records_best(records))} "
        f"seg_probe={cur_seg:.8f} pose={cur_pose:.8f} score_est={cur_score:.8f}",
        flush=True,
    )

    n, _, h, w = fake2.shape
    tile = args.tile
    th, tw = h // tile, w // tile
    for pass_idx in range(args.passes):
        accepted = 0
        tile_counts = F.avg_pool2d(cur_bad.float().unsqueeze(1), kernel_size=tile, stride=tile).squeeze(1)
        end_frame = n if args.end_frame is None else min(args.end_frame, n)
        for frame_idx in range(max(args.start_frame, 0), end_frame):
            if frame_idx % args.progress_every == 0:
                payload_now = len(pack_records_best(records))
                seg_now = cur_bad.float().mean().item()
                print(
                    f"scan pass={pass_idx} frame={frame_idx}/{n} actions={len(records)} "
                    f"seg={seg_now:.8f} payload={payload_now}",
                    flush=True,
                )
            top = torch.topk(tile_counts[frame_idx].flatten(), k=min(args.top_tiles, tile_counts.shape[1] * tile_counts.shape[2])).indices.tolist()
            for tile_id in top:
                if len(records) >= args.max_actions:
                    break
                y0 = (tile_id // tw) * tile
                x0 = (tile_id % tw) * tile
                if cur_bad[frame_idx, y0:y0 + tile, x0:x0 + tile].sum().item() == 0:
                    continue

                candidates = []
                candidate_meta = []
                base_img = fake2[frame_idx:frame_idx + 1].repeat(len(specs), 1, 1, 1)
                for action_idx, (_, delta) in enumerate(specs):
                    img = fake2[frame_idx].clone()
                    img[:, y0:y0 + tile, x0:x0 + tile] = (img[:, y0:y0 + tile, x0:x0 + tile] + delta).clamp(0, 255)
                    candidates.append(img)
                    candidate_meta.append(action_idx)
                cand = torch.stack(candidates, 0)
                pred = seg_predict(seg, cand, device, args.batch_size)
                bad_counts = (pred != target_masks[frame_idx:frame_idx + 1]).flatten(1).sum(1)
                old_bad_count = cur_bad[frame_idx].sum().item()
                best_pos = int(torch.argmin(bad_counts).item())
                new_bad_count = int(bad_counts[best_pos].item())
                pixel_gain = old_bad_count - new_bad_count
                if pixel_gain <= 0:
                    continue

                trial_records = records + [(frame_idx, tile_id, candidate_meta[best_pos])]
                payload = len(pack_records_best(trial_records))
                new_seg = cur_bad.float().mean().item() - pixel_gain / (n * h * w)
                new_pose = cur_pose
                if args.pose_gate:
                    new_frame_pose_err = pose_err_one(
                        pose,
                        target_pose[frame_idx],
                        fake1[frame_idx],
                        cand[best_pos].cpu(),
                        device,
                    )
                    new_pose = cur_pose + (new_frame_pose_err - cur_pose_errs[frame_idx]) / n
                old_score = score(cur_seg, cur_pose, len(pack_records_best(records)))
                new_score = score(new_seg, new_pose, payload)
                net = old_score - new_score
                if net < args.min_gain:
                    continue

                fake2[frame_idx] = cand[best_pos].cpu()
                cur_pred[frame_idx] = pred[best_pos].cpu()
                cur_bad[frame_idx] = cur_pred[frame_idx] != target_masks[frame_idx]
                cur_seg = new_seg
                if args.pose_gate:
                    cur_pose_errs[frame_idx] = new_frame_pose_err
                    cur_pose = new_pose
                records = trial_records
                accepted += 1
                payload_now = len(pack_records_best(records))
                print(
                    f"accept pass={pass_idx} frame={frame_idx} tile={tile_id} "
                    f"action={specs[candidate_meta[best_pos]][0]} pixel_gain={pixel_gain} "
                    f"net={net:.8f} actions={len(records)} seg={cur_seg:.8f} pose={cur_pose:.8f} "
                    f"payload={payload_now} est_score={score(cur_seg, cur_pose, payload_now):.8f}",
                    flush=True,
                )
            if len(records) >= args.max_actions:
                break

        payload_now = len(pack_records_best(records))
        seg_now = cur_bad.float().mean().item()
        print(
            f"pass_done={pass_idx} accepted={accepted} actions={len(records)} seg={seg_now:.8f} "
            f"payload={payload_now} est_score_no_pose={score(seg_now, BASE_POSE, payload_now):.8f}",
            flush=True,
        )
        if accepted == 0:
            break

    payload = len(pack_records_best(records))
    final_seg = cur_bad.float().mean().item()
    final_pose = BASE_POSE
    if args.pose_check and records:
        final_pose = pose_dist(pose, target_pose, fake1, fake2, device, args.batch_size)
    print(
        f"FINAL actions={len(records)} payload={payload} seg={final_seg:.8f} "
        f"pose={final_pose:.8f} est_score={score(final_seg, final_pose, payload):.8f}",
        flush=True,
    )
    if records:
        out = SUB / "seg_tile_actions_probe.br"
        out.write_bytes(pack_records_best(records))
        print(f"wrote {out} ({out.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
