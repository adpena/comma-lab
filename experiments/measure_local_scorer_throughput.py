#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""LOCAL-THROUGHPUT-ATTACK: micro-benchmark the canonical contest scorer
forward+backward cost on the LOCAL torch-CPU substrate (the d_seg gradient
authority per CLAUDE.md "local CPU + MLX GPU good").

What this measures (NOT a score — a TIMING benchmark):
  - The per-batch wall-clock of a representative score-aware training gradient:
    render-frames -> SegNet (d_seg surrogate: soft-argmax CE-style MSE) and
    PoseNet (d_pose MSE) forward + backward through the FROZEN scorer weights.
  - SegNet-only vs PoseNet-only vs both, so we know which scorer dominates the
    backward (load-bearing for the distilled-surrogate angle).

NO-FAKE: this runs the EXACT upstream/modules.py SegNet (EfficientNet-B2 U-Net)
+ PoseNet (FastViT-T12) with the REAL safetensors weights at the canonical
512x384 model input. It measures TIME, not score. fp16/bf16 variants are
THROUGHPUT tools and are labeled as such; no d_seg authority claim is made here.

Device default: cpu (the M5 torch-CPU anchor). --device mps is FORBIDDEN as an
authority but allowed here ONLY as a throughput-curiosity (tagged advisory) —
this script makes NO score claim on any device.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "upstream"))

from modules import SegNet, PoseNet  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

SEG_W, SEG_H = 512, 384  # segnet_model_input_size (W, H)


def _build_scorers(device: str):
    seg = SegNet().to(device)
    seg.load_state_dict(load_file(REPO / "upstream/models/segnet.safetensors", device=device))
    pose = PoseNet().to(device)
    pose.load_state_dict(load_file(REPO / "upstream/models/posenet.safetensors", device=device))
    # Frozen scorer: weights do NOT require grad; the gradient flows back to the
    # rendered pixels (the renderer params), exactly as in score-aware training.
    for p in seg.parameters():
        p.requires_grad_(False)
    for p in pose.parameters():
        p.requires_grad_(False)
    seg.eval()
    pose.eval()
    return seg, pose


def _seg_loss(seg, frame_pred, seg_target_logits):
    # frame_pred: (B, C=3, H, W) leaf with grad (the "render"); SegNet takes
    # last-frame only via preprocess, but here we feed the model body directly
    # at model-input resolution to isolate the scorer fwd+bwd cost.
    out = seg(frame_pred)
    # Differentiable d_seg surrogate: MSE on logits vs frozen target logits
    # (a stand-in for soft-CE/KL; the COST is what we measure, not the value).
    return ((out - seg_target_logits) ** 2).mean()


def _pose_loss(pose, yuv12_pred, pose_target):
    out = pose(yuv12_pred)["pose"][..., :6]
    return ((out - pose_target) ** 2).mean()


def _time_loop(fn, n_warmup: int, n_iters: int):
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    dt = (time.perf_counter() - t0) / n_iters
    return dt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    device = args.device
    B = args.batch_size

    seg, pose = _build_scorers(device)

    autocast_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    # --- SegNet input: (B, 3, 384, 512) rendered frame leaf (requires grad) ---
    def make_seg_inputs():
        x = torch.rand(B, 3, SEG_H, SEG_W, device=device, requires_grad=True)
        with torch.no_grad():
            tgt = seg(torch.rand(B, 3, SEG_H, SEG_W, device=device))
        return x, tgt

    seg_x, seg_tgt = make_seg_inputs()

    def seg_step():
        seg_x.grad = None
        if autocast_dtype is not None and device != "mps":
            with torch.autocast(device_type="cpu", dtype=autocast_dtype):
                loss = _seg_loss(seg, seg_x, seg_tgt)
        else:
            loss = _seg_loss(seg, seg_x, seg_tgt)
        loss.backward()
        return loss

    # --- PoseNet input: (B, 12, 384, 512) yuv6x2 leaf (requires grad) ---
    def make_pose_inputs():
        x = torch.rand(B, 12, SEG_H, SEG_W, device=device, requires_grad=True)
        tgt = torch.rand(B, 6, device=device)
        return x, tgt

    pose_x, pose_tgt = make_pose_inputs()

    def pose_step():
        pose_x.grad = None
        if autocast_dtype is not None and device != "mps":
            with torch.autocast(device_type="cpu", dtype=autocast_dtype):
                loss = _pose_loss(pose, pose_x, pose_tgt)
        else:
            loss = _pose_loss(pose, pose_x, pose_tgt)
        loss.backward()
        return loss

    def both_step():
        seg_step()
        pose_step()

    print(f"=== LOCAL scorer throughput: device={device} B={B} dtype={args.dtype} threads={torch.get_num_threads()} ===")
    seg_dt = _time_loop(seg_step, args.warmup, args.iters)
    print(f"SegNet  fwd+bwd: {seg_dt:.3f} s/batch  ({B/seg_dt:.3f} pairs/s)")
    pose_dt = _time_loop(pose_step, args.warmup, args.iters)
    print(f"PoseNet fwd+bwd: {pose_dt:.3f} s/batch  ({B/pose_dt:.3f} pairs/s)")
    both_dt = seg_dt + pose_dt
    print(f"BOTH (seg+pose): {both_dt:.3f} s/batch  ({B/both_dt:.3f} pairs/s)")
    print(f"  SegNet share of backward cost:  {seg_dt/both_dt*100:.1f}%")
    print(f"  PoseNet share of backward cost: {pose_dt/both_dt*100:.1f}%")
    # n600 projection: 600 pairs / B = steps/epoch
    steps = 600 / B
    print(f"  n600 epoch projection (600 pairs, B={B}): {both_dt*steps/60:.1f} min/epoch")


if __name__ == "__main__":
    main()
