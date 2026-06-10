#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""LEVER-B targets builder — the EXACT score-native targets the smoke descends on.

Lever B of GOAL_standing_v3 (`.omx/research/original_innovation_offensive_plan_20260610.md`
§4 the $0 descent-smoke spec). Builds, from the FROZEN scorer on the 600 GT frame1s:

  1. The 600 SegNet 5-class ARGMAX maps at 384x512 (the LITERAL d_seg target).
     d_seg = mean over 196,608 px of (argmax(out_pred) != argmax(out_gt)), per pair.
  2. The per-pixel MARGIN field (top1 - top2 logit gap) of the GT argmax — lever G:
     error is FREE inside the margin; the carrier may dump error in the large-margin
     interior. Stored so the generator's hinge loss can weight by it.
  3. The 600x6 PoseNet pose trajectory (first-6-of-12 dims) — the direct pose carrier
     section + lever F's free byproduct (its entropy / KB cost).

EXACT path (NO GUESSING — line-cited modules.py + frame_utils.py read in full):
  - GT decode: `frame_utils.yuv420_to_rgb` (pyav, BT.601 limited range). NEVER MPS.
  - SegNet: `x[:,-1,...]` (frame1 only) -> interpolate bilinear to (384,512) -> smp.Unet
    'tu-efficientnet_b2' 5-class, RAW [0,255] (NO mean/std). argmax over 5 classes.
  - PoseNet: BOTH frames -> interpolate (384,512) -> rgb_to_yuv6 (12ch) -> (x-127.5)/63.75
    -> fastvit_t12 -> Hydra pose head 12 dims -> FIRST 6 scored.

EVIDENCE DISCIPLINE: all [macOS-CPU advisory]; promotion_eligible=false. Real pyav decode
of the real 0.mkv, real frozen DistortionNet forward. Disk hygiene: targets cached on the
SSD tier (/Volumes/VertigoDataTier/pact/...), NEVER /tmp.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "upstream"
for p in (REPO_ROOT, REPO_ROOT / "src", UPSTREAM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path, field: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field}={path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


def build_targets(
    video_path: Path,
    out_dir: Path,
    num_pairs: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Decode GT frame pairs, run frozen SegNet+PoseNet, cache argmax maps + margins + pose."""
    import av
    import torch
    from frame_utils import seq_len, yuv420_to_rgb
    from modules import (
        SegNet,
        PoseNet,
        posenet_sd_path,
        segnet_sd_path,
    )
    from safetensors.torch import load_file

    if device != "cpu":
        # MPS corrupts the scorer (CLAUDE.md). This builder is CPU-exact only.
        raise ValueError("targets builder is CPU-exact only; MPS/CUDA not used for the GT scorer here")

    _refuse_tmp(out_dir, "out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_in_h, seg_in_w = 384, 512  # segnet_model_input_size = (512, 384) -> (H, W)

    # --- decode the first (num_pairs*2) GT frames, EXACT upstream path ---
    t0 = time.time()
    n_frames_needed = num_pairs * seq_len
    frames: list[np.ndarray] = []
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        arr = yuv420_to_rgb(frame).cpu().numpy().astype(np.uint8)  # (H,W,3) uint8 camera-res
        frames.append(arr)
        if len(frames) >= n_frames_needed:
            break
    container.close()
    n = (len(frames) // seq_len) * seq_len
    n_pairs = n // seq_len
    if n_pairs == 0:
        raise ValueError("no full pairs decoded")
    decode_s = time.time() - t0

    # --- load frozen scorers (CPU, eval) ---
    seg = SegNet().eval().to(device)
    seg.load_state_dict(load_file(segnet_sd_path, device=device))
    pose = PoseNet().eval().to(device)
    pose.load_state_dict(load_file(posenet_sd_path, device=device))

    # Output stores
    argmax_maps = np.memmap(
        out_dir / "gt_segnet_argmax.u8", dtype=np.uint8, mode="w+",
        shape=(n_pairs, seg_in_h, seg_in_w),
    )
    margin_maps = np.memmap(
        out_dir / "gt_segnet_margin.f16", dtype=np.float16, mode="w+",
        shape=(n_pairs, seg_in_h, seg_in_w),
    )
    pose_traj = np.zeros((n_pairs, 6), dtype=np.float32)

    t1 = time.time()
    with torch.inference_mode():
        for pi in range(n_pairs):
            f0 = pi * seq_len
            # frame1 (LAST frame) for SegNet
            pair = np.stack(frames[f0 : f0 + seq_len])  # (2,H,W,3) uint8
            pair_t = torch.from_numpy(pair).unsqueeze(0).float()  # (1,2,H,W,3)
            # SegNet: preprocess_input takes (b,t,c,h,w); modules.SegNet.preprocess slices x[:,-1]
            import einops
            x_bt_chw = einops.rearrange(pair_t, "b t h w c -> b t c h w")
            seg_in = seg.preprocess_input(x_bt_chw)  # (1,3,384,512) frame1 resized
            seg_logits = seg(seg_in)  # (1,5,384,512)
            top2 = torch.topk(seg_logits, k=2, dim=1).values  # (1,2,384,512)
            argmax = seg_logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()  # (384,512)
            margin = (top2[:, 0] - top2[:, 1])[0].cpu().numpy().astype(np.float16)  # (384,512)
            argmax_maps[pi] = argmax
            margin_maps[pi] = margin

            # PoseNet: both frames
            pose_in = pose.preprocess_input(x_bt_chw)  # (1,12,192,256)
            pose_out = pose(pose_in)["pose"][0, :6].cpu().numpy().astype(np.float32)  # (6,)
            pose_traj[pi] = pose_out

    argmax_maps.flush()
    margin_maps.flush()
    del argmax_maps, margin_maps
    np.save(out_dir / "gt_posenet_pose6.npy", pose_traj)
    score_s = time.time() - t1

    # --- pose-trajectory byte accounting (lever F free byproduct) ---
    pose_raw_fp16 = pose_traj.astype(np.float16).tobytes()
    import brotli  # type: ignore
    import lzma

    pose_fp16_brotli = len(brotli.compress(pose_raw_fp16, quality=11))
    # temporal-delta fp16 (smooth driving trajectory)
    delta = np.diff(pose_traj, axis=0, prepend=pose_traj[:1])
    pose_delta_fp16_brotli = len(brotli.compress(delta.astype(np.float16).tobytes(), quality=11))
    pose_delta_fp16_lzma = len(
        lzma.compress(delta.astype(np.float16).tobytes(), format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    )

    # --- margin distribution (lever G interior budget) ---
    margins = np.asarray(margin_maps_ro(out_dir, n_pairs, seg_in_h, seg_in_w))
    margin_pos_frac = float((margins > 0).mean())  # should be ~1 (argmax => top1>top2)
    interior_bands = {
        f"frac_margin_gt_{b}": float((margins > b).mean())
        for b in (0.5, 1.0, 2.0, 5.0)
    }

    meta = {
        "subagent": "lever_b_score_native_argmax_smoke_20260610",
        "utc": _utc(),
        "evidence_grade": "[macOS-CPU advisory]",
        "promotion_eligible": False,
        "score_claim": False,
        "video": str(video_path),
        "num_pairs_requested": num_pairs,
        "num_pairs_built": n_pairs,
        "frames_decoded": len(frames),
        "seg_input_hw": [seg_in_h, seg_in_w],
        "seg_argmax_px_per_map": seg_in_h * seg_in_w,
        "decode_seconds": round(decode_s, 2),
        "score_seconds": round(score_s, 2),
        "pose_trajectory": {
            "shape": [n_pairs, 6],
            "raw_fp16_bytes": len(pose_raw_fp16),
            "fp16_brotli_q11_bytes": pose_fp16_brotli,
            "delta_fp16_brotli_q11_bytes": pose_delta_fp16_brotli,
            "delta_fp16_lzma_xz_bytes": pose_delta_fp16_lzma,
            "best_pose_carrier_bytes": min(pose_fp16_brotli, pose_delta_fp16_brotli, pose_delta_fp16_lzma),
        },
        "segnet_margin": {
            "margin_positive_fraction": margin_pos_frac,
            **interior_bands,
            "note": "lever G interior budget: fraction of px with large margin = free logit room",
        },
        "argmax_class_histogram": _class_hist(out_dir, n_pairs, seg_in_h, seg_in_w),
        "artifacts": {
            "argmax_u8": str(out_dir / "gt_segnet_argmax.u8"),
            "margin_f16": str(out_dir / "gt_segnet_margin.f16"),
            "pose6_npy": str(out_dir / "gt_posenet_pose6.npy"),
        },
    }
    (out_dir / "targets_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def margin_maps_ro(out_dir: Path, n_pairs: int, h: int, w: int) -> np.memmap:
    return np.memmap(out_dir / "gt_segnet_margin.f16", dtype=np.float16, mode="r", shape=(n_pairs, h, w))


def _class_hist(out_dir: Path, n_pairs: int, h: int, w: int) -> dict[str, float]:
    am = np.memmap(out_dir / "gt_segnet_argmax.u8", dtype=np.uint8, mode="r", shape=(n_pairs, h, w))
    total = n_pairs * h * w
    return {f"class_{k}_frac": float((np.asarray(am) == k).sum()) / total for k in range(5)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LEVER-B score-native targets builder (frozen scorer)")
    ap.add_argument("--video", type=Path, default=UPSTREAM / "videos" / "0.mkv")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610/targets"),
    )
    ap.add_argument("--num-pairs", type=int, default=600)
    args = ap.parse_args(argv)

    meta = build_targets(args.video, args.out_dir, args.num_pairs)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
