#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""$0 horizon-band-localized d_seg lever probe on the 0.19110 CPU frontier.

Tests the operator's stark-edge-under-resolution hypothesis AND a horizon-band
flip-residual sidecar (Lever-D localized) — the highest-prior untested d_seg path,
since 97.8% of the frontier residual d_seg lives in the horizon band (SegNet rows
96-288), peaking at rows ~185-195 (camera row ~421, near calibration cy=437).

Authority: exact d_seg = per-pixel argmax-disagreement through the frozen CPU
SegNet on the LAST frame of each pair, after the contest preprocess. GT decoded via
upstream frame_utils.yuv420_to_rgb (NEVER PyAV rgb24). Reuses the cached argmaps
(gt, comp) from probe_indep_dseg_bets_frontier.py prepare stage — those argmaps
ARE the exact frozen-SegNet output (validated: cached d_seg=0.00055989 vs report
0.00055978, Δ=1e-7). All score/break-even math via tac.contest_score (canonical).

[contest-CPU advisory] — single video 0 (1200 frames=600 pairs) == the 600-sample
eval locally. NON-PROMOTABLE; a pointer move still needs byte-close + exact eval.

Stages:
  margin  : Task B — reconstruction luma/chroma error vs camera row + logit-margin
            shallowness at horizon flips (is the stark sky/ground edge under-resolved
            and are the flips shallow-margin = training-fixable?).
  sidecar : Task C — horizon-band flip-residual sidecar: oracle Δd_seg (force comp
            argmax->GT argmax in the band) + entropy-coded byte cost; GO/NO-GO vs
            the canonical break-even.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
UP = REPO / "upstream"
sys.path.insert(0, str(UP))

from tac.contest_score import (
    break_even_d_seg,
    compute_contest_score,
    rate_term,
)

CAMERA_H, CAMERA_W = 874, 1164
SEG_W, SEG_H = 512, 384

INFLATED_DIR = REPO / "experiments/results/indep_dseg_bets_20260623_inflated"
INFLATED_RAW = INFLATED_DIR / "0.raw"
ARGMAP_NPZ = INFLATED_DIR / "seg_argmaps.npz"
GT_VIDEO = UP / "videos/0.mkv"
OUT_DIR = REPO / "experiments/results/horizon_band_dseg_lever_20260623"

# Frontier operating point (from report.txt of the canonical frontier archive).
FRONTIER_D_SEG = 0.00055978
FRONTIER_D_POSE = 0.00002942
FRONTIER_BYTES = 177169
FRONTIER_S = 0.19109982419209975

# Horizon band in SegNet-grid rows (carries 97.8% of d_seg per prior probe).
HB_LO, HB_HI = 96, 288


def _open_inflated() -> np.memmap:
    fb = CAMERA_H * CAMERA_W * 3
    nbytes = INFLATED_RAW.stat().st_size
    n = nbytes // fb
    return np.memmap(
        INFLATED_RAW, dtype=np.uint8, mode="r", shape=(n, CAMERA_H, CAMERA_W, 3)
    )


def _load_segnet():
    from modules import SegNet, segnet_sd_path  # type: ignore
    from safetensors.torch import load_file

    seg = SegNet().eval()
    sd = load_file(str(segnet_sd_path), device="cpu")
    seg.load_state_dict(sd)
    for p in seg.parameters():
        p.requires_grad_(False)
    return seg


def _seg_logits_from_camera_rgb(seg, frame_uint8_hw3: torch.Tensor) -> torch.Tensor:
    """(H,W,3) uint8 -> (5, SEG_H, SEG_W) logits, exact contest preprocess."""
    x = frame_uint8_hw3.to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=(SEG_H, SEG_W), mode="bilinear")
    with torch.inference_mode():
        out = seg(x)  # (1,5,SEG_H,SEG_W)
    return out.squeeze(0)


def _iter_gt_last_frames(n_pairs: int):
    import av
    from frame_utils import yuv420_to_rgb  # type: ignore

    container = av.open(str(GT_VIDEO))
    stream = container.streams.video[0]
    idx = 0
    pair = 0
    for frame in container.decode(stream):
        if pair >= n_pairs:
            break
        if idx % 2 == 1:
            rgb = yuv420_to_rgb(frame)  # (H,W,3) uint8 torch
            yield pair, rgb
            pair += 1
        idx += 1
    container.close()


# ---------------------------------------------------------------------------
# Task B: stark-edge mechanism — luma/chroma recon error vs row + flip margins
# ---------------------------------------------------------------------------
def stage_margin(n_pairs: int):
    """Measure (1) per-camera-row luma+chroma |comp-GT| error; (2) the SegNet
    softmax-margin at horizon-band flip pixels vs non-flip pixels (shallow margin =
    training/fidelity-fixable). Uses real GT decode + real frozen SegNet logits."""
    d = np.load(ARGMAP_NPZ)
    gt_arg, comp_arg = d["gt"], d["comp"]  # (600,SEG_H,SEG_W) uint8
    seg = _load_segnet()
    mm = _open_inflated()

    # BT.601 luma weights (recon error decomposed into luma vs chroma proxies).
    luma_w = torch.tensor([0.299, 0.587, 0.114])

    row_luma_err = np.zeros(CAMERA_H, dtype=np.float64)
    row_chroma_err = np.zeros(CAMERA_H, dtype=np.float64)
    row_count = 0

    # margin accumulation (softmax top1-top2 gap at flip vs non-flip, horizon band)
    flip_margins, nonflip_margins = [], []
    # edge-shift proxy: vertical luma gradient peak row in GT vs comp (horizon edge)
    gt_edge_rows, comp_edge_rows = [], []

    for pair, gt_rgb in _iter_gt_last_frames(n_pairs):
        comp_rgb = torch.from_numpy(np.ascontiguousarray(mm[2 * pair + 1]))
        gt_f = gt_rgb.to(torch.float32)
        comp_f = comp_rgb.to(torch.float32)
        # per-row luma error and chroma (R-G,B-G as cheap chroma proxy) error
        gt_luma = (gt_f * luma_w).sum(-1)  # (H,W)
        comp_luma = (comp_f * luma_w).sum(-1)
        row_luma_err += (comp_luma - gt_luma).abs().mean(dim=1).numpy()
        gt_chroma = torch.stack(
            [gt_f[..., 0] - gt_f[..., 1], gt_f[..., 2] - gt_f[..., 1]], -1
        )
        comp_chroma = torch.stack(
            [comp_f[..., 0] - comp_f[..., 1], comp_f[..., 2] - comp_f[..., 1]], -1
        )
        row_chroma_err += (comp_chroma - gt_chroma).abs().mean(dim=(1, 2)).numpy()
        row_count += 1

        # edge-shift: vertical luma gradient magnitude profile (mean over cols),
        # peak row in the horizon camera band (rows 350-500) for GT vs comp.
        band_lo_cam, band_hi_cam = 350, 520
        gl = gt_luma[band_lo_cam:band_hi_cam]
        cl = comp_luma[band_lo_cam:band_hi_cam]
        gt_grad = (gl[1:] - gl[:-1]).abs().mean(dim=1)
        comp_grad = (cl[1:] - cl[:-1]).abs().mean(dim=1)
        gt_edge_rows.append(band_lo_cam + int(gt_grad.argmax().item()))
        comp_edge_rows.append(band_lo_cam + int(comp_grad.argmax().item()))

        # margins on the COMP last-frame logits (what SegNet actually decides on).
        logits = _seg_logits_from_camera_rgb(seg, comp_rgb)  # (5,SEG_H,SEG_W)
        sm = F.softmax(logits, dim=0)  # (5,SEG_H,SEG_W)
        top2 = sm.topk(2, dim=0).values  # (2,SEG_H,SEG_W)
        margin = (top2[0] - top2[1])  # (SEG_H,SEG_W) in [0,1]
        hb = slice(HB_LO, HB_HI)
        m_hb = margin[hb].numpy()
        flips_hb = gt_arg[pair, hb] != comp_arg[pair, hb]
        flip_margins.append(m_hb[flips_hb])
        # sample equal count of non-flip pixels for a fair comparison
        nf = m_hb[~flips_hb]
        if nf.size > flips_hb.sum() and flips_hb.sum() > 0:
            sel = np.random.default_rng(pair).choice(
                nf.size, int(flips_hb.sum()), replace=False
            )
            nonflip_margins.append(nf[sel])
        if (pair + 1) % 50 == 0:
            print(f"  margin {pair+1}/{n_pairs}", flush=True)

    row_luma_err /= row_count
    row_chroma_err /= row_count
    fm = np.concatenate(flip_margins) if flip_margins else np.array([0.0])
    nfm = np.concatenate(nonflip_margins) if nonflip_margins else np.array([0.0])
    gt_edge = np.array(gt_edge_rows)
    comp_edge = np.array(comp_edge_rows)

    # horizon camera-row band = SegNet rows 185-195 mapped: row_cam = row_seg*(874/384)
    hr_lo = int(180 * CAMERA_H / SEG_H)
    hr_hi = int(200 * CAMERA_H / SEG_H)
    rep = {
        "n_pairs": row_count,
        "horizon_camera_rows_band": [hr_lo, hr_hi],
        "luma_err_horizon_band_mean": float(row_luma_err[hr_lo:hr_hi].mean()),
        "luma_err_whole_frame_mean": float(row_luma_err.mean()),
        "luma_err_horizon_vs_frame_ratio": float(
            row_luma_err[hr_lo:hr_hi].mean() / max(row_luma_err.mean(), 1e-9)
        ),
        "chroma_err_horizon_band_mean": float(row_chroma_err[hr_lo:hr_hi].mean()),
        "chroma_err_whole_frame_mean": float(row_chroma_err.mean()),
        "chroma_err_horizon_vs_frame_ratio": float(
            row_chroma_err[hr_lo:hr_hi].mean() / max(row_chroma_err.mean(), 1e-9)
        ),
        "luma_err_peak_row": int(row_luma_err.argmax()),
        "luma_err_peak_value": float(row_luma_err.max()),
        "edge_shift_rows_mean_abs": float(np.abs(gt_edge - comp_edge).mean()),
        "edge_shift_rows_median_abs": float(np.median(np.abs(gt_edge - comp_edge))),
        "flip_margin_mean": float(fm.mean()),
        "flip_margin_median": float(np.median(fm)),
        "nonflip_margin_mean": float(nfm.mean()),
        "flip_margin_p90": float(np.percentile(fm, 90)),
        "flip_margin_frac_below_0p1": float((fm < 0.1).mean()),
        "flip_margin_frac_below_0p2": float((fm < 0.2).mean()),
    }
    # PAYLOAD WRITE ORDER (ddm_pl1, cured by ddm_ql2): the record went LAST here,
    # behind a bulk npz that ran BEFORE `OUT_DIR` was even created -- so on a cold
    # run the npz raised and `margin.json` never landed. Both bugs die together:
    # mkdir, then the cheap irreplaceable record, then the rebuildable row curves.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "margin.json").write_text(json.dumps(rep, indent=2))
    # per-row error curve for the memo
    np.savez_compressed(
        OUT_DIR / "margin_rowcurves.npz",
        row_luma_err=row_luma_err,
        row_chroma_err=row_chroma_err,
    )
    print(json.dumps(rep, indent=2))


# ---------------------------------------------------------------------------
# Task C: horizon-band flip-residual sidecar — oracle Δd_seg + byte cost + GO/NO-GO
# ---------------------------------------------------------------------------
def _entropy_bytes_for_flip_residual(gt_arg, comp_arg, band_lo, band_hi):
    """Estimate the byte cost of a flip-residual sidecar that stores, per pair, the
    horizon-band positions where comp!=gt AND the corrected class. Two encodings:
      (a) raw position bitmap + class bytes (loose upper bound);
      (b) entropy floor: -sum p log2 p over the per-pair flip-or-not Bernoulli +
          the corrected-class symbols, materialized via zlib on the actual symbol
          stream (a real, conservative coded size — NOT a hand-rolled formula).
    Returns dict of byte estimates + the flip count."""
    band = slice(band_lo, band_hi)
    flips = gt_arg[:, band, :] != comp_arg[:, band, :]  # (600, bandH, 512)
    n_flips = int(flips.sum())
    n_pairs, band_h, band_w = flips.shape
    band_pixels = n_pairs * band_h * band_w

    # (a) bitmap encoding: 1 bit per band pixel for position + 3 bits/flip for class.
    bitmap_bytes = math.ceil(band_pixels / 8) + math.ceil(n_flips * 3 / 8)

    # (b) REAL coded size: build the actual symbol stream and zlib it.
    #   - position stream: the flip mask as packed bits, zlib (exploits sparsity +
    #     spatial run structure — a real codec, conservative vs arithmetic coding).
    pos_packed = np.packbits(flips.reshape(-1))
    pos_coded = len(zlib.compress(pos_packed.tobytes(), 9))
    #   - class stream: the corrected (GT) class at each flip position, zlib.
    corrected = gt_arg[:, band, :][flips].astype(np.uint8)
    cls_coded = len(zlib.compress(corrected.tobytes(), 9))
    coded_bytes = pos_coded + cls_coded

    # entropy floor on the corrected-class symbols (for reference).
    if n_flips > 0:
        vals, cnts = np.unique(corrected, return_counts=True)
        p = cnts / cnts.sum()
        cls_entropy_bits = float(-(p * np.log2(p)).sum()) * n_flips
    else:
        cls_entropy_bits = 0.0
    # position entropy floor: Bernoulli(q) over band pixels.
    q = n_flips / max(band_pixels, 1)
    if 0 < q < 1:
        pos_entropy_bits = -(
            q * math.log2(q) + (1 - q) * math.log2(1 - q)
        ) * band_pixels
    else:
        pos_entropy_bits = 0.0
    entropy_floor_bytes = math.ceil((cls_entropy_bits + pos_entropy_bits) / 8)

    return {
        "n_flips": n_flips,
        "band_pixels": band_pixels,
        "flip_fraction_in_band": float(n_flips / max(band_pixels, 1)),
        "bitmap_upper_bound_bytes": int(bitmap_bytes),
        "zlib_coded_bytes": int(coded_bytes),
        "zlib_pos_bytes": int(pos_coded),
        "zlib_cls_bytes": int(cls_coded),
        "entropy_floor_bytes": int(entropy_floor_bytes),
    }


def stage_sidecar(n_pairs: int):
    d = np.load(ARGMAP_NPZ)
    gt_arg = d["gt"][:n_pairs]
    comp_arg = d["comp"][:n_pairs]

    baseline_dseg = float((gt_arg != comp_arg).mean())

    results = []
    # Test multiple band widths: tight (peak), horizon, full residual band.
    bands = {
        "peak_rows_180_200": (180, 200),
        "tight_rows_160_220": (160, 220),
        "horizon_rows_96_288": (HB_LO, HB_HI),
        "all_residual_rows_96_345": (96, 345),
    }
    for name, (lo, hi) in bands.items():
        band = slice(lo, hi)
        # oracle Δd_seg: force comp argmax -> GT argmax inside the band (this is
        # EXACTLY what a perfect flip-residual sidecar achieves — the max possible).
        corrected = comp_arg.copy()
        corrected[:, band, :] = gt_arg[:, band, :]
        new_dseg = float((gt_arg != corrected).mean())
        delta_dseg = new_dseg - baseline_dseg  # negative = improvement

        bcost = _entropy_bytes_for_flip_residual(gt_arg, comp_arg, lo, hi)

        # Net score impact via the CANONICAL helper (rate term grows with bytes).
        new_bytes = FRONTIER_BYTES + bcost["zlib_coded_bytes"]
        new_S = compute_contest_score(new_dseg, FRONTIER_D_POSE, new_bytes)
        net_delta_S = new_S - FRONTIER_S

        # break-even d_seg given the added bytes: how low must d_seg go to break even?
        be = break_even_d_seg(FRONTIER_S, FRONTIER_D_POSE, new_bytes)
        # Δd_seg per byte (oracle): how much d_seg per byte spent.
        dseg_per_byte = (
            delta_dseg / max(bcost["zlib_coded_bytes"], 1)
        )

        results.append(
            {
                "band": name,
                "seg_rows": [lo, hi],
                "oracle_new_dseg": new_dseg,
                "oracle_delta_dseg": delta_dseg,
                "oracle_rel_reduction": delta_dseg / max(baseline_dseg, 1e-12),
                "sidecar_bytes_zlib": bcost["zlib_coded_bytes"],
                "sidecar_bytes_entropy_floor": bcost["entropy_floor_bytes"],
                "sidecar_bytes_bitmap_ub": bcost["bitmap_upper_bound_bytes"],
                "n_flips_corrected": bcost["n_flips"],
                "delta_dseg_per_byte": dseg_per_byte,
                "rate_cost_in_dseg_units": rate_term(bcost["zlib_coded_bytes"])
                / 100.0,
                "net_new_score": new_S,
                "net_delta_S": net_delta_S,
                "break_even_dseg_given_bytes": be,
                "verdict": "GO" if net_delta_S < 0 else "NO-GO",
            }
        )

    rep = {
        "baseline_dseg": baseline_dseg,
        "frontier_S": FRONTIER_S,
        "frontier_bytes": FRONTIER_BYTES,
        "break_even_target": "net_delta_S < 0 (beats 0.19110)",
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sidecar.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["margin", "sidecar"])
    ap.add_argument("--pairs", type=int, default=600)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "margin":
        stage_margin(args.pairs)
    elif args.stage == "sidecar":
        stage_sidecar(args.pairs)


if __name__ == "__main__":
    main()
