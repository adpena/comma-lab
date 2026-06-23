#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Gate #128 headroom: can a per-pair frame_1 selector mode cut d_seg on the frontier?

The frontier's FECa selector (222 bytes) picks palette idx 0-15 per pair -- ALL of
which target frame_0 or are 'none' (they tune d_pose, not d_seg). The SegNet d_seg
is scored on frame_1 (the LAST frame). Palette modes 22-30 are tiny frame_1 pixel
biases (luma +-1/-2, RGB bias, blue-chroma) that DO touch the SegNet-scored frame.

This probe measures the EXACT d_seg headroom of pointing the selector at d_seg:
for each pair, apply each frame_1 mode to the comp last-frame (the same camera-res
pixel op the inflate applies via apply_pr101_selector_to_frames, BEFORE the uint8
cast), re-run the frozen SegNet, and record the best achievable per-pair d_seg.

Reports:
  - baseline d_seg (frontier),
  - best-frame1-mode d_seg (oracle per-pair pick over modes 22-30 + 'none'),
  - the d_seg delta, and the byte cost of encoding the d_seg-only selector.

NO FAKE: every d_seg is the real frozen SegNet argmax through the exact preprocess.
Camera-res mode ops mirror frame_selector.apply_frame0_mode (rgb_bias/blue_chroma).
[contest-CPU advisory] single video 0 == the 600-pair eval locally.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
UP = REPO / "upstream"
SUB = REPO / "experiments/results/pr110_payload_entropy_recode_20260610/submission_dir"
sys.path.insert(0, str(UP))
sys.path.insert(0, str(SUB / "src"))

CAMERA_H, CAMERA_W = 874, 1164
SEG_W, SEG_H = 512, 384
OUT_DIR = REPO / "experiments/results/indep_dseg_bets_20260623_inflated"
INFLATED_RAW = OUT_DIR / "0.raw"

# frame_1 modes (palette idx 22-30) -> camera-res pixel op, mirroring apply_frame0_mode.
# Each entry: (label, family, params). Plus 'none' identity.
FRAME1_MODES = [
    ("none", "identity", ()),
    ("frame1_rgb_bias_p2_m1_m1", "rgb_bias", (2, -1, -1)),
    ("frame1_rgb_bias_m2_p1_p1", "rgb_bias", (-2, 1, 1)),
    ("frame1_luma_bias_-1", "rgb_bias", (-1, -1, -1)),
    ("frame1_blue_chroma_amp_3", "blue_chroma", (3,)),
    ("frame1_rgb_bias_p0_m1_p1", "rgb_bias", (0, -1, 1)),
    ("frame1_rgb_bias_p0_p1_m1", "rgb_bias", (0, 1, -1)),
    ("frame1_luma_bias_+1", "rgb_bias", (1, 1, 1)),
    ("frame1_blue_chroma_amp_1", "blue_chroma", (1,)),
    ("frame1_luma_bias_-2", "rgb_bias", (-2, -2, -2)),
]


def load_segnet():
    from modules import SegNet, segnet_sd_path  # type: ignore
    from safetensors.torch import load_file

    seg = SegNet().eval()
    seg.load_state_dict(load_file(str(segnet_sd_path), device="cpu"))
    for p in seg.parameters():
        p.requires_grad_(False)
    return seg


def _blue_tile(h, w):
    from frame_selector import _blue_tile as bt  # type: ignore
    return bt(h, w, device=torch.device("cpu"), dtype=torch.float32)


def apply_mode_camera(frame_chw_f32: torch.Tensor, family: str, params) -> torch.Tensor:
    """Mirror frame_selector.apply_frame0_mode at camera-res, then clamp+round+uint8.

    Input/out: (3,H,W) float32. The frontier applies the mode to the rounded float
    frame, then casts to uint8. We replicate: op -> clamp(0,255) -> round -> (kept f32).
    """
    if family == "identity":
        out = frame_chw_f32
    elif family == "rgb_bias":
        delta = torch.tensor(params, dtype=torch.float32).view(3, 1, 1)
        out = frame_chw_f32 + delta
    elif family == "blue_chroma":
        amp = float(params[0])
        out = frame_chw_f32.clone()
        tile = _blue_tile(frame_chw_f32.shape[1], frame_chw_f32.shape[2])
        out[0].add_(tile * amp)
        out[2].sub_(tile * amp)
    else:
        raise ValueError(family)
    return out.clamp(0, 255).round()


def segnet_argmax(seg, frame_chw_f32: torch.Tensor) -> torch.Tensor:
    x = frame_chw_f32.unsqueeze(0)  # (1,3,H,W)
    x = F.interpolate(x, size=(SEG_H, SEG_W), mode="bilinear")
    with torch.inference_mode():
        return seg(x).argmax(dim=1).squeeze(0).to(torch.uint8)


def main(n_pairs: int):
    torch.set_num_threads(2)
    seg = load_segnet()
    cache = np.load(OUT_DIR / "seg_argmaps.npz")
    gt = cache["gt"]  # (600,SEG_H,SEG_W) GT argmax
    fb = CAMERA_H * CAMERA_W * 3
    N = INFLATED_RAW.stat().st_size // fb
    mm = np.memmap(INFLATED_RAW, dtype=np.uint8, mode="r", shape=(N, CAMERA_H, CAMERA_W, 3))

    n_pairs = min(n_pairs, gt.shape[0])
    base_total = 0.0
    best_total = 0.0
    mode_wins = {m[0]: 0 for m in FRAME1_MODES}
    npx = SEG_H * SEG_W

    for pair in range(n_pairs):
        gt_arg = torch.from_numpy(gt[pair].astype(np.int64))
        comp = mm[2 * pair + 1]  # (H,W,3) uint8 -- the rounded uint8 last frame
        comp_chw = torch.from_numpy(np.ascontiguousarray(comp)).to(torch.float32).permute(2, 0, 1)
        # NOTE: the frontier's selector ALREADY applied its chosen frame0/none mode to
        # frame_0; frame_1 in the inflated raw is the UN-selected (base) last frame for
        # all pairs whose mode targets frame_0/none -- which is ALL frontier pairs.
        # So comp_chw here is exactly the pre-frame1-selector last frame. Applying a
        # frame_1 mode on top is the faithful candidate.
        best_d = None
        best_m = "none"
        for label, fam, pp in FRAME1_MODES:
            cand = apply_mode_camera(comp_chw, fam, pp)
            arg = segnet_argmax(seg, cand)
            d = float((arg.to(torch.int64) != gt_arg).float().mean().item())
            if best_d is None or d < best_d:
                best_d = d
                best_m = label
            if label == "none":
                base_d = d
        base_total += base_d
        best_total += best_d
        mode_wins[best_m] += 1
        if (pair + 1) % 25 == 0:
            print(f"  {pair+1}/{n_pairs} base={base_total/(pair+1):.8f} best={best_total/(pair+1):.8f}", flush=True)

    base = base_total / n_pairs
    best = best_total / n_pairs
    # byte cost of a d_seg-only selector: per-pair pick over 10 modes ~ log2(10)=3.32 bits
    # but the frontier ALREADY spends 222 bytes on a frame0 selector. A frame1 d_seg
    # selector would be ADDITIONAL bytes (can't reuse: frame0 picks tune d_pose).
    import math
    sel_bits = n_pairs * math.log2(len(FRAME1_MODES))
    sel_bytes_raw = math.ceil(sel_bits / 8)
    res = {
        "stage": "gate128_frame1_selector_headroom",
        "n_pairs": n_pairs,
        "baseline_d_seg_none": base,
        "best_frame1_mode_d_seg_oracle": best,
        "d_seg_delta": best - base,
        "d_seg_rel_reduction": (base - best) / base if base else 0.0,
        "mode_win_histogram": mode_wins,
        "added_selector_bytes_raw_estimate": sel_bytes_raw,
        "note": "oracle per-pair pick; selector bytes are ADDITIONAL (frame0 selector tunes d_pose, cannot be reused for d_seg)",
    }
    print(json.dumps(res, indent=2))
    (OUT_DIR / "gate128_frame1_selector_headroom.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    main(n)
