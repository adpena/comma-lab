#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""$0 independent d_seg probes on the existing 0.19110 CPU frontier.

Authority: the EXACT d_seg = per-pixel argmax-disagreement through the frozen CPU
SegNet (upstream/models/segnet.safetensors) on the LAST frame of each pair, after
the contest preprocess (bilinear-resize camera-res RGB -> 512x384, SegNet, argmax
over 5 classes). GT decoded via upstream frame_utils.yuv420_to_rgb (NEVER PyAV
rgb24). Frontier reconstruction read from the inflated 0.raw (camera-res uint8).

This script measures three $0 gates that need NO trainer:
  #139 hood static-region clamp  : force the fixed bottom hood band of the comp
                                   last-frame to a known-constant value; measure
                                   the exact d_seg delta + per-region flip frac.
  #138 road/lane geometric prior : measure argmax-flip fraction inside the
                                   ground-plane road trapezoid; $0 gate = does
                                   forcing comp argmax->GT argmax there cut d_seg
                                   (an UPPER bound on any geometric-prior win),
                                   and is that region byte-closeable for ~0 bytes.
  #128 frontier sub-linear lever : reported separately (selector/decoder-sub).

All measurements are [contest-CPU advisory] (single video 0 == the 600-pair eval
locally). NO FAKE: every reported d_seg comes from the real frozen SegNet through
the exact preprocess. Honest negatives welcome.

Usage:
  python experiments/probe_indep_dseg_bets_frontier.py --stage prepare --pairs 600
  python experiments/probe_indep_dseg_bets_frontier.py --stage hood --pairs 600
  python experiments/probe_indep_dseg_bets_frontier.py --stage geo  --pairs 600
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
UP = REPO / "upstream"
sys.path.insert(0, str(UP))

CAMERA_H, CAMERA_W = 874, 1164
SEG_W, SEG_H = 512, 384  # segnet_model_input_size = (W, H)
N_FRAMES_PER_PAIR = 2

INFLATED_RAW = REPO / "experiments/results/indep_dseg_bets_20260623_inflated/0.raw"
GT_VIDEO = UP / "videos/0.mkv"
OUT_DIR = REPO / "experiments/results/indep_dseg_bets_20260623_inflated"
ARGMAP_NPZ = OUT_DIR / "seg_argmaps.npz"  # cached GT + comp argmax maps


def load_segnet(device: str = "cpu"):
    """Load the frozen contest SegNet (frozen weights, eval mode)."""
    from modules import SegNet, segnet_sd_path  # type: ignore
    from safetensors.torch import load_file

    seg = SegNet().eval().to(device=device)
    sd = load_file(str(segnet_sd_path), device=device)
    seg.load_state_dict(sd)
    for p in seg.parameters():
        p.requires_grad_(False)
    return seg


def segnet_argmax_from_camera_rgb(seg, frame_uint8_hw3: torch.Tensor) -> torch.Tensor:
    """frame_uint8_hw3: (H_cam, W_cam, 3) uint8 on cpu -> argmax (SEG_H, SEG_W) uint8.

    Mirrors SegNet.preprocess_input on the LAST frame: rearrange to (1,3,H,W) float,
    bilinear-resize to (SEG_H, SEG_W), run SegNet, argmax over channel dim.
    """
    x = frame_uint8_hw3.to(torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    x = F.interpolate(x, size=(SEG_H, SEG_W), mode="bilinear")
    with torch.inference_mode():
        out = seg(x)  # (1, 5, SEG_H, SEG_W)
    return out.argmax(dim=1).squeeze(0).to(torch.uint8)  # (SEG_H, SEG_W)


def iter_gt_last_frames(n_pairs: int):
    """Yield GT last-frame (frame index 2*i+1) camera-res RGB uint8 (H,W,3) via yuv420."""
    import av
    from frame_utils import yuv420_to_rgb  # type: ignore

    container = av.open(str(GT_VIDEO))
    stream = container.streams.video[0]
    idx = 0
    pair = 0
    buf = None
    for frame in container.decode(stream):
        if pair >= n_pairs:
            break
        if idx % 2 == 1:  # last frame of pair (0-indexed: frame 1,3,5,...)
            rgb = yuv420_to_rgb(frame)  # (H,W,3) uint8 torch
            yield pair, rgb
            pair += 1
        idx += 1
    container.close()


def comp_last_frame(mm: np.memmap, pair: int) -> torch.Tensor:
    """Comp last-frame (frame 2*pair+1) from inflated mmap -> (H,W,3) uint8 torch."""
    fr = mm[2 * pair + 1]  # (H,W,3) uint8
    return torch.from_numpy(np.ascontiguousarray(fr))


def open_inflated() -> np.memmap:
    fb = CAMERA_H * CAMERA_W * 3
    nbytes = INFLATED_RAW.stat().st_size
    N = nbytes // fb
    return np.memmap(INFLATED_RAW, dtype=np.uint8, mode="r", shape=(N, CAMERA_H, CAMERA_W, 3))


# ---------------------------------------------------------------------------
# Stage: prepare  -> cache GT + comp argmax maps for n_pairs, report baseline d_seg
# ---------------------------------------------------------------------------
def stage_prepare(n_pairs: int):
    seg = load_segnet("cpu")
    mm = open_inflated()
    gt_maps = np.zeros((n_pairs, SEG_H, SEG_W), dtype=np.uint8)
    comp_maps = np.zeros((n_pairs, SEG_H, SEG_W), dtype=np.uint8)
    per_pair = np.zeros((n_pairs,), dtype=np.float64)

    for pair, gt_rgb in iter_gt_last_frames(n_pairs):
        gt_arg = segnet_argmax_from_camera_rgb(seg, gt_rgb)
        comp_rgb = comp_last_frame(mm, pair)
        comp_arg = segnet_argmax_from_camera_rgb(seg, comp_rgb)
        gt_maps[pair] = gt_arg.numpy()
        comp_maps[pair] = comp_arg.numpy()
        per_pair[pair] = float((gt_arg != comp_arg).float().mean().item())
        if (pair + 1) % 50 == 0:
            print(f"  prepared {pair+1}/{n_pairs}  running d_seg={per_pair[:pair+1].mean():.8f}", flush=True)

    baseline = float(per_pair.mean())
    np.savez_compressed(ARGMAP_NPZ, gt=gt_maps, comp=comp_maps, per_pair=per_pair)
    print(json.dumps({
        "stage": "prepare",
        "n_pairs": n_pairs,
        "baseline_d_seg_local": baseline,
        "report_d_seg_600": 0.00055978,
        "argmap_cache": str(ARGMAP_NPZ),
    }, indent=2))


# ---------------------------------------------------------------------------
# Stage: hood (#139) -> measure flips inside fixed bottom hood band; test $0 clamp
# ---------------------------------------------------------------------------
def stage_hood(n_pairs: int, hood_rows_frac_list=(0.84, 0.88, 0.90, 0.92, 0.94)):
    """The hood is a static bottom band of every camera frame. In the 384x512
    SegNet grid, rows below frac*SEG_H map to the hood. Measure:
      (a) flip fraction inside the hood band (comp vs GT argmax),
      (b) what fraction of TOTAL d_seg the hood band contributes,
      (c) the d_seg if we force comp argmax == GT argmax inside the hood band
          (UPPER bound on any 0-byte hood-clamp win), AND the d_seg if we force
          the hood band to a single constant class (a true 0-byte inflate-time
          clamp -- both comp and GT? no: GT is fixed; we can only change COMP).

    A 0-byte hood clamp can only force the COMP hood region to a constant class.
    The win exists only if the GT SegNet argmax in the hood is ALSO that constant
    (i.e. GT hood is near-uniform) AND comp currently flips there. We measure the
    realistic win: force comp-hood -> the per-pixel GT-hood majority class is NOT
    0-byte (needs GT). The true 0-byte clamp forces comp-hood -> the GLOBAL hood
    majority class (data-independent). We report both the upper bound and the
    realistic 0-byte clamp.
    """
    d = np.load(ARGMAP_NPZ)
    gt = d["gt"]; comp = d["comp"]; per_pair = d["per_pair"]
    n_pairs = min(n_pairs, gt.shape[0])
    gt = gt[:n_pairs]; comp = comp[:n_pairs]

    total_px = SEG_H * SEG_W
    flips = (gt != comp)  # (n,SEG_H,SEG_W) bool
    baseline = float(flips.mean())

    results = {"stage": "hood", "n_pairs": n_pairs, "baseline_d_seg": baseline, "bands": []}

    for frac in hood_rows_frac_list:
        row0 = int(round(frac * SEG_H))
        band = slice(row0, SEG_H)
        band_px = (SEG_H - row0) * SEG_W
        band_flip_frac = float(flips[:, band, :].mean())  # flips within band
        band_share_of_total = float(flips[:, band, :].sum()) / float(flips.sum() + 1e-12)

        # GT hood uniformity: dominant class fraction inside band, per pair averaged
        gt_band = gt[:, band, :].reshape(n_pairs, -1)
        # global hood majority class (data-independent constant for 0-byte clamp)
        global_counts = np.bincount(gt_band.reshape(-1), minlength=5)
        global_major = int(np.argmax(global_counts))
        global_major_frac = float(global_counts[global_major] / global_counts.sum())

        # (a) UPPER bound: comp_band := gt_band  (cannot be done at 0 byte -> needs GT)
        new_flips_upper = flips.copy()
        new_flips_upper[:, band, :] = False
        d_seg_upper = float(new_flips_upper.mean())

        # (b) realistic 0-byte clamp: force comp argmax in band -> global_major.
        #     new flip there = (gt != global_major). (comp is overwritten.)
        comp_clamped = comp.copy()
        comp_clamped[:, band, :] = global_major
        new_flips_clamp = (gt != comp_clamped)
        d_seg_clamp = float(new_flips_clamp.mean())

        results["bands"].append({
            "hood_row_frac": frac,
            "seg_rows": [row0, SEG_H],
            "band_px_per_frame": band_px,
            "band_frac_of_frame": band_px / total_px,
            "flip_frac_in_band": band_flip_frac,
            "band_share_of_total_dseg": band_share_of_total,
            "gt_hood_global_major_class": global_major,
            "gt_hood_global_major_frac": global_major_frac,
            "d_seg_upper_bound_force_gt": d_seg_upper,
            "d_seg_delta_upper": d_seg_upper - baseline,
            "d_seg_realistic_0byte_clamp": d_seg_clamp,
            "d_seg_delta_0byte_clamp": d_seg_clamp - baseline,
        })

    print(json.dumps(results, indent=2))
    (OUT_DIR / "hood_gate_139.json").write_text(json.dumps(results, indent=2))


# ---------------------------------------------------------------------------
# Stage: geo (#138) -> road/lane geometric prior region flip + 0-byte feasibility
# ---------------------------------------------------------------------------
def stage_geo(n_pairs: int):
    """Ground-plane road trapezoid in the SegNet grid. comma10k classes (SegNet
    5-class): typical ordering [road, lane-markings/undrivable, movable, my-car,
    sky/other] -- we do NOT assume the index; instead we measure the per-class
    flip contribution + the geometric road-trapezoid region, and report:
      (a) flip fraction inside the road trapezoid,
      (b) per-class share of total d_seg (which classes drive the flips),
      (c) UPPER bound d_seg if we force comp argmax->GT inside the trapezoid
          (an upper bound on ANY geometric-prior road win),
      (d) 0-byte feasibility note (a geometric prior is data-independent; it can
          force comp argmax->road inside the trapezoid only where GT is road).

    The trapezoid: horizon row ~0.50*SEG_H, widening to full width at hood top.
    """
    d = np.load(ARGMAP_NPZ)
    gt = d["gt"]; comp = d["comp"]
    n_pairs = min(n_pairs, gt.shape[0])
    gt = gt[:n_pairs]; comp = comp[:n_pairs]
    flips = (gt != comp)
    baseline = float(flips.mean())

    # per-class share of d_seg (by GT class at flipped pixels, and by comp class)
    gt_at_flip = gt[flips]
    comp_at_flip = comp[flips]
    gt_share = {int(c): float((gt_at_flip == c).mean()) for c in range(5)}
    comp_share = {int(c): float((comp_at_flip == c).mean()) for c in range(5)}

    # road trapezoid mask (data-independent geometric prior).
    rr, cc = np.meshgrid(np.arange(SEG_H), np.arange(SEG_W), indexing="ij")
    horizon = 0.50 * SEG_H
    hood_top = 0.90 * SEG_H
    # width fraction grows linearly from 0.18 at horizon to 0.95 at hood_top
    t = np.clip((rr - horizon) / (hood_top - horizon + 1e-9), 0.0, 1.0)
    half_w = (0.09 + (0.475 - 0.09) * t) * SEG_W
    center = 0.5 * SEG_W
    in_band = (rr >= horizon) & (rr < hood_top)
    trap = in_band & (np.abs(cc - center) <= half_w)  # (SEG_H, SEG_W) bool

    trap_px = int(trap.sum())
    trap_flip_frac = float(flips[:, trap].mean())
    trap_share_of_total = float(flips[:, trap].sum()) / float(flips.sum() + 1e-12)

    # GT dominant class inside trapezoid (should be 'road' if our geometry is right)
    gt_trap = gt[:, trap].reshape(-1)
    trap_counts = np.bincount(gt_trap, minlength=5)
    trap_major = int(np.argmax(trap_counts))
    trap_major_frac = float(trap_counts[trap_major] / trap_counts.sum())

    # (c) UPPER bound: force comp->GT inside trapezoid
    new_flips_upper = flips.copy()
    new_flips_upper[:, trap] = False
    d_seg_upper = float(new_flips_upper.mean())

    # (d) 0-byte geometric prior: force comp argmax->trap_major inside trapezoid.
    comp_clamped = comp.copy()
    comp_clamped[:, trap] = trap_major
    new_flips_clamp = (gt != comp_clamped)
    d_seg_clamp = float(new_flips_clamp.mean())

    results = {
        "stage": "geo", "n_pairs": n_pairs, "baseline_d_seg": baseline,
        "gt_class_share_of_dseg": gt_share,
        "comp_class_share_of_dseg": comp_share,
        "trapezoid": {
            "px_per_frame": trap_px, "frac_of_frame": trap_px / (SEG_H * SEG_W),
            "flip_frac_in_trap": trap_flip_frac,
            "trap_share_of_total_dseg": trap_share_of_total,
            "gt_trap_major_class": trap_major, "gt_trap_major_frac": trap_major_frac,
            "d_seg_upper_bound_force_gt": d_seg_upper,
            "d_seg_delta_upper": d_seg_upper - baseline,
            "d_seg_realistic_0byte_prior": d_seg_clamp,
            "d_seg_delta_0byte_prior": d_seg_clamp - baseline,
        },
    }
    print(json.dumps(results, indent=2))
    (OUT_DIR / "geo_gate_138.json").write_text(json.dumps(results, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prepare", "hood", "geo"])
    ap.add_argument("--pairs", type=int, default=600)
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        stage_prepare(args.pairs)
    elif args.stage == "hood":
        stage_hood(args.pairs)
    elif args.stage == "geo":
        stage_geo(args.pairs)


if __name__ == "__main__":
    main()
