#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure delta_R: the R-chain SegNet-margin noise floor (task #360, force #2).

WHAT / WHY
----------
Force #2 (MARGIN-BAND SATISFICING) needs a principled hinge threshold delta_R:
the margin level below which the frozen SegNet's top1-top2 logit gap CANNOT be
controlled, because the contest R operator's uint8 quantization at camera
resolution (874x1164) perturbs the margin by an uncontrollable amount. Pushing a
boundary pixel's margin above delta_R buys nothing (uint8 noise can still flip
it); pushing it there is wasted gradient that should be reallocated to pixels
still inside the band. delta_R is a high quantile of the uint8-induced margin
perturbation distribution, measured on the annulus (where d_seg lives).

THE MEASUREMENT (uint8-isolation, deterministic)
------------------------------------------------
The contest R chain (per pr95_hnerv_mlx_training.apply_contest_faithful_roundtrip
_nhwc): render(base) -> bicubic up to CAMERA(874x1164) -> uint8 round/clamp at
CAMERA -> bilinear down to SEG(384x512) -> SegNet. The ONLY non-smooth,
uncontrollable step is the uint8 round at camera res. We isolate it:

  x_c  = bicubic( bilinear(gt_f1 -> 384x512) -> 874x1164 )     # continuous, witness-reachable
  m0   = margin( segnet( bilinear(x_c            -> 384x512) ) )   # NO uint8
  m1   = margin( segnet( bilinear(round(x_c)     -> 384x512) ) )   # WITH uint8 (clamp 0..255)
  delta = m1 - m0                                              # per-pixel margin perturbation

delta_R := p95 of |delta| over annulus pixels (|GT margin| < band).
Rationale for p95 (not max, not p50): the hinge must stop pushing once the margin
is safe against the TYPICAL worst-case R-noise; p95 means "R-noise exceeds this
only 5% of the time" -> a pixel above delta_R*(1+headroom) is R-robustly safe.

AUTHORITY: frozen CPU-torch SegNet (numpy-fp32 verdict authority). NEVER MPS.
This produces a delta_R NUMBER that sets a lever default -> CPU only.
Output tagged [macOS-CPU advisory . NON-PROMOTABLE]. Pointer 0.19110 UNMOVED.

USAGE
-----
  .venv/bin/python tools/measure_delta_R_noise_floor.py \
      --gt-npz experiments/results/mlx_fleet_gt_cache/gt_n96.npz \
      --band 1.0 --n 96 --out reports/delta_R_noise_floor.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_segnet_cpu(upstream: str):
    from tac.scorer import load_default_segnet

    return load_default_segnet(upstream, device="cpu")


def _margin_from_logits(logits):  # logits: (B,5,H,W) torch
    import torch

    top2 = torch.topk(logits, 2, dim=1).values  # (B,2,H,W)
    return (top2[:, 0] - top2[:, 1])  # (B,H,W)


def main(argv=None) -> int:
    import torch
    import torch.nn.functional as F

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-npz", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--upstream", default="upstream")
    ap.add_argument("--band", type=float, default=1.0,
                    help="annulus = |GT margin| < band (matches subpix_band / seg_chroma_boundary margin_band=1.0)")
    ap.add_argument("--n", type=int, default=96, help="number of frames to use (caps at cache size)")
    ap.add_argument("--out", default="reports/delta_R_noise_floor.json")
    args = ap.parse_args(argv)

    torch.set_grad_enabled(False)
    # CPU authority only — MPS is NEVER a score/margin authority (CLAUDE.md).
    seg = _load_segnet_cpu(args.upstream)

    z = np.load(args.gt_npz)
    f1 = z["gt_f1"]  # (N,874,1164,3) uint8 — SegNet scores the last frame of the pair
    gt_margin = z["margins"] if "margins" in z.files else None  # (N,384,512) fp32 GT signed margin
    N = min(args.n, f1.shape[0])

    CAM_HW = (874, 1164)
    SEG_HW = (384, 512)

    abs_delta_annulus = []
    abs_delta_all = []
    # cross-check accumulator: full-R margin vs GT-direct margin (includes bicubic up-down, not just uint8)
    abs_fullR_annulus = []
    n_annulus_px = 0
    n_total_px = 0

    for i in range(N):
        frame = torch.from_numpy(f1[i]).float().permute(2, 0, 1)[None]  # (1,3,874,1164) [0,255]
        # witness-reachable continuous camera frame: 384-content upsampled to camera (pre-uint8)
        at_seg = F.interpolate(frame, size=SEG_HW, mode="bilinear", align_corners=False)
        x_c = F.interpolate(at_seg, size=CAM_HW, mode="bicubic", align_corners=False)  # (1,3,874,1164) float

        # m0: NO uint8
        seg_in0 = F.interpolate(x_c, size=SEG_HW, mode="bilinear", align_corners=False)
        m0 = _margin_from_logits(seg(seg_in0))[0].numpy()  # (384,512)

        # m1: WITH uint8 (the only uncontrollable step)
        x_q = torch.clamp(torch.round(x_c), 0, 255)
        seg_in1 = F.interpolate(x_q, size=SEG_HW, mode="bilinear", align_corners=False)
        m1 = _margin_from_logits(seg(seg_in1))[0].numpy()

        delta = np.abs(m1 - m0)  # (384,512) per-pixel uint8-induced margin perturbation

        # annulus = low-|GT margin| band (fallback to m0 proxy if cache lacks margins)
        ann = (np.abs(gt_margin[i]) < args.band) if gt_margin is not None else (np.abs(m0) < args.band)
        abs_delta_annulus.append(delta[ann])
        abs_delta_all.append(delta.reshape(-1))
        n_annulus_px += int(ann.sum())
        n_total_px += delta.size

        # cross-check: GT frame -> direct SegNet margin vs full-R margin
        seg_in_direct = F.interpolate(frame, size=SEG_HW, mode="bilinear", align_corners=False)
        m_direct = _margin_from_logits(seg(seg_in_direct))[0].numpy()
        abs_fullR_annulus.append(np.abs(m1 - m_direct)[ann])

    ann_cat = np.concatenate(abs_delta_annulus)
    all_cat = np.concatenate(abs_delta_all)
    fullR_cat = np.concatenate(abs_fullR_annulus)

    def _q(a):
        return {
            "mean": float(np.mean(a)),
            "p50": float(np.quantile(a, 0.50)),
            "p90": float(np.quantile(a, 0.90)),
            "p95": float(np.quantile(a, 0.95)),
            "p99": float(np.quantile(a, 0.99)),
            "max": float(np.max(a)),
        }

    result = {
        "measurement": "delta_R_noise_floor",
        "axis": "[macOS-CPU advisory . NON-PROMOTABLE]",
        "gt_npz": args.gt_npz,
        "n_frames": N,
        "band": args.band,
        "annulus_area_frac": n_annulus_px / max(n_total_px, 1),
        "uint8_isolation": {
            "annulus": _q(ann_cat),
            "all_px": _q(all_cat),
            "note": "|margin(uint8-round(x_c)) - margin(x_c)| ; the pure uint8-at-camera perturbation",
        },
        "delta_R": float(np.quantile(ann_cat, 0.95)),
        "delta_R_def": "p95 of |uint8-induced margin perturbation| over annulus (|GT margin|<band)",
        "cross_check_full_R_vs_gt_direct": {
            "annulus": _q(fullR_cat),
            "note": "|margin(full-R of 384-content) - margin(GT-direct)| ; includes bicubic up/down + uint8 (upper bound)",
        },
        "gt_margin_scale_note": "GT annulus margin p10~0.172 p50~0.897 (run-1 telemetry); compare delta_R to these",
    }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
