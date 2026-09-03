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
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# Receipt-only constants (they never enter the delta_R measurement itself).
# Sub-band divisors give the annulus-definition sensitivity of delta_R for free
# from the same pass: band/1, band/2, band/4 are nested subsets of the annulus.
RECEIPT_SUB_BAND_DIVISORS = (1.0, 2.0, 4.0)
# comma10k canonical order, MEASURED 2026-06-27 (CLAUDE.md): never luma-sorted.
SEG_CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")


def _load_segnet_cpu(upstream: str):
    from tac.scorer import load_default_segnet

    return load_default_segnet(upstream, device="cpu")


def _margin_from_logits(logits):  # logits: (B,5,H,W) torch
    import torch

    top2 = torch.topk(logits, 2, dim=1).values  # (B,2,H,W)
    return (top2[:, 0] - top2[:, 1])  # (B,H,W)


def _quantile_summary(a):
    """Canonical quantile block. Shared by the result JSON and the receipts."""
    return {
        "mean": float(np.mean(a)),
        "p50": float(np.quantile(a, 0.50)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


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
    ap.add_argument("--receipts-out", default=None,
                    help="RECEIPTS ONLY (does not change delta_R): write per-frame, "
                         "per-class-annulus and sub-band receipt rows to this JSON path")
    ap.add_argument("--retain-dir", default=None,
                    help="ALWAYS KEEP THE PAYLOAD: persist the per-frame m0/m1 margin "
                         "arrays (float32, (N,384,512)) as .npy under this directory")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch.set_num_threads(N); default leaves the torch default alone")
    args = ap.parse_args(argv)

    torch.set_grad_enabled(False)
    if args.threads is not None:
        torch.set_num_threads(int(args.threads))
    # CPU authority only — MPS is NEVER a score/margin authority (CLAUDE.md).
    seg = _load_segnet_cpu(args.upstream)

    z = np.load(args.gt_npz)
    f1 = z["gt_f1"]  # (N,874,1164,3) uint8 — SegNet scores the last frame of the pair
    gt_margin = z["margins"] if "margins" in z.files else None  # (N,384,512) fp32 GT signed margin
    lstars = z["lstars"] if "lstars" in z.files else None  # (N,384,512) int GT argmax class
    N = min(args.n, f1.shape[0])

    CAM_HW = (874, 1164)
    SEG_HW = (384, 512)

    # ---- receipts / retention scaffolding (inert unless the flags are given) ----
    receipts_on = args.receipts_out is not None
    per_frame_rows = []
    class_pool = {c: [] for c in range(len(SEG_CLASS_NAMES))} if receipts_on else {}
    sub_band_pool = {d: [] for d in RECEIPT_SUB_BAND_DIVISORS} if receipts_on else {}
    m0_mm = m1_mm = None
    retain_dir = None
    if args.retain_dir is not None:
        retain_dir = Path(args.retain_dir)
        retain_dir.mkdir(parents=True, exist_ok=True)
        m0_mm = np.lib.format.open_memmap(
            retain_dir / "m0_no_uint8.npy", mode="w+", dtype=np.float32,
            shape=(N, SEG_HW[0], SEG_HW[1]))
        m1_mm = np.lib.format.open_memmap(
            retain_dir / "m1_with_uint8.npy", mode="w+", dtype=np.float32,
            shape=(N, SEG_HW[0], SEG_HW[1]))

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

        # ---- receipts / retention (read-only over the arrays above) ----
        if m0_mm is not None:
            m0_mm[i] = m0
            m1_mm[i] = m1
        if receipts_on:
            ref = gt_margin[i] if gt_margin is not None else m0
            row = {"frame": i, "n_annulus_px": int(ann.sum())}
            row["annulus"] = (
                _quantile_summary(delta[ann]) if ann.any() else None
            )
            per_class = None
            if lstars is not None:
                per_class = {}
                for c, cname in enumerate(SEG_CLASS_NAMES):
                    m = ann & (lstars[i] == c)
                    n_c = int(m.sum())
                    per_class[cname] = {
                        "n_px": n_c,
                        "p95": float(np.quantile(delta[m], 0.95)) if n_c else None,
                    }
                    if n_c:
                        class_pool[c].append(delta[m])
            row["per_class_annulus"] = per_class
            for d in RECEIPT_SUB_BAND_DIVISORS:
                sb = np.abs(ref) < (args.band / d)
                if sb.any():
                    sub_band_pool[d].append(delta[sb])
            per_frame_rows.append(row)

    ann_cat = np.concatenate(abs_delta_annulus)
    all_cat = np.concatenate(abs_delta_all)
    fullR_cat = np.concatenate(abs_fullR_annulus)

    _q = _quantile_summary

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

    retained = {}
    if m0_mm is not None:
        m0_mm.flush()
        m1_mm.flush()
        del m0_mm, m1_mm
        for name in ("m0_no_uint8.npy", "m1_with_uint8.npy"):
            p = retain_dir / name
            retained[name] = {"bytes": p.stat().st_size, "sha256": _sha256_file(p)}
        print(f"[retain] {retain_dir}: {json.dumps(retained)}")

    if receipts_on:
        receipts = {
            "measurement": "delta_R_noise_floor_receipts",
            "note": "RECEIPTS ONLY — derived from the same arrays as --out; "
                    "delta_R itself is unchanged by these flags",
            "axis": result["axis"],
            "gt_npz": args.gt_npz,
            "n_frames": N,
            "band": args.band,
            "delta_R": result["delta_R"],
            "torch_num_threads": int(torch.get_num_threads()),
            "per_class_annulus_pooled": {
                SEG_CLASS_NAMES[c]: (
                    {
                        "n_px": int(sum(a.size for a in class_pool[c])),
                        **_quantile_summary(np.concatenate(class_pool[c])),
                    }
                    if class_pool[c]
                    else None
                )
                for c in range(len(SEG_CLASS_NAMES))
            } if lstars is not None else None,
            "sub_band_sensitivity": {
                f"band_{args.band / d:g}": (
                    {
                        "n_px": int(sum(a.size for a in sub_band_pool[d])),
                        **_quantile_summary(np.concatenate(sub_band_pool[d])),
                    }
                    if sub_band_pool[d]
                    else None
                )
                for d in RECEIPT_SUB_BAND_DIVISORS
            },
            "retained_payloads": retained or None,
            "retain_dir": str(retain_dir) if retain_dir is not None else None,
            "per_frame": per_frame_rows,
        }
        rp = Path(args.receipts_out)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(receipts, indent=2))
        print(f"[receipts] wrote {rp} ({len(per_frame_rows)} frame rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
