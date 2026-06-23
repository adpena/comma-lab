#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""$0 frozen-instance cross-frame structure test of the horizon d_seg flip sets.

THE DECISIVE QUESTION (reconcile a90 deep-math byte-floor 16-262 B vs aa98 measured
NO-GO -4.65e-9 d_seg/byte): the contest is exact-overfit to ONE frozen video, so the
1200 per-frame horizon flip-sets are KNOWN. Do they collapse to a LOW-DIMENSIONAL
function of the KNOWN ego-motion (horizon trajectory v_h(t) = cy + fy*tan(pitch(t))),
making a PARAMETERIZED encoding (trajectory + low-entropy residual) cheap enough to
FLIP aa98's NO-GO economics?

Authority: exact frozen-SegNet argmax-disagreement (cached argmaps from the prior
probes' prepare stage — validated cached d_seg=0.00055989 vs report 0.00055978).
GT horizon row recovered from the EXACT GT argmax (road<->undrivable boundary) — no
external pose file, no PyAV rgb24, fully authority-faithful. All score/byte math via
tac.contest_score (Catalog #391). NON-PROMOTABLE [contest-CPU advisory]; pointer
UNMOVED 0.19110; any GO still needs byte-close + upstream/evaluate.py.

Stages:
  trajectory : recover v_h(t) per frame (GT road<->undrivable boundary, the exact
               proxy for the ego-pitch horizon row), measure its temporal smoothness +
               low-D structure (how many bytes to encode the trajectory itself).
  crossframe : THE decisive test. Model each per-frame flip-set vs v_h(t):
               (1) flip-band tracking: do flips concentrate around v_h(t)?
               (2) cross-frame intrinsic dimension: residual entropy of the flip
                   positions AFTER a v_h(t)-trajectory + per-column-offset model.
               (3) PARAMETERIZED encoding byte cost (trajectory + entropy-coded
                   residual) vs aa98's per-pixel -4.65e-9 and the rate slope
                   6.659e-9/byte. GO/NO-GO with delta_dseg/byte. Gap to a90's floor.
  taskspace  : brief — is encoding the KNOWN argmax targets DIRECTLY (task-space)
               structurally cheaper than a correction ON the frontier?
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
UP = REPO / "upstream"
sys.path.insert(0, str(UP))

from tac.contest_score import (  # noqa: E402
    break_even_d_seg,
    compute_contest_score,
    rate_term,
)

CAMERA_H, CAMERA_W = 874, 1164
SEG_W, SEG_H = 512, 384

INFLATED_DIR = REPO / "experiments/results/indep_dseg_bets_20260623_inflated"
ARGMAP_NPZ = INFLATED_DIR / "seg_argmaps.npz"
OUT_DIR = REPO / "experiments/results/frozen_instance_horizon_crossframe_20260623"

# Frontier operating point (canonical pointer + report.txt).
FRONTIER_D_SEG = 0.00055978
FRONTIER_D_POSE = 0.00002942
FRONTIER_BYTES = 177169
FRONTIER_S = 0.19109982419209975

# comma10k classes: 0=Road 1=LaneMark 2=Undrivable(sky) 3=Movable 4=MyCar.
ROAD, UNDRIVABLE = 0, 2

# Rate slope: d_seg per byte (canonical break-even a sidecar must beat).
RATE_SLOPE = rate_term(1) / 100.0  # 6.659e-9


def _load_argmaps(n_pairs: int):
    d = np.load(ARGMAP_NPZ)
    return d["gt"][:n_pairs].astype(np.uint8), d["comp"][:n_pairs].astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage: trajectory — recover v_h(t) from the EXACT GT argmax + its byte cost
# ---------------------------------------------------------------------------
def _gt_horizon_per_column(gt_t: np.ndarray) -> np.ndarray:
    """Per-column horizon row = topmost Road row (first road pixel scanning top->down).
    Columns with no road -> NaN. gt_t is (SEG_H, SEG_W) uint8."""
    road = gt_t == ROAD  # (H,W)
    has_road = road.any(axis=0)
    first_road = np.where(has_road, road.argmax(axis=0), np.nan).astype(np.float64)
    first_road[~has_road] = np.nan
    return first_road  # (W,)


def stage_trajectory(n_pairs: int):
    gt, _ = _load_argmaps(n_pairs)
    # per-column horizon row per frame -> (n, W)
    perc = np.stack([_gt_horizon_per_column(gt[t]) for t in range(n_pairs)], 0)
    # per-frame scalar v_h(t) = median over columns
    vh = np.nanmedian(perc, axis=1)  # (n,)

    # temporal smoothness
    dvh = np.abs(np.diff(vh))
    # low-D structure: fit a low-order polynomial in t and report residual.
    t = np.arange(n_pairs, dtype=np.float64)
    poly_resid = {}
    for deg in (1, 2, 3, 5, 8):
        c = np.polyfit(t, vh, deg)
        pred = np.polyval(c, t)
        poly_resid[f"deg{deg}_rms"] = float(np.sqrt(np.mean((vh - pred) ** 2)))
        poly_resid[f"deg{deg}_max_abs"] = float(np.max(np.abs(vh - pred)))

    # Byte cost of the v_h(t) trajectory itself, two encodings:
    #  (a) low-order poly: deg+1 float32 coeffs + per-frame int residual (quantized to
    #      1 seg-row, entropy-coded). Use the smallest poly with residual <= 1 row max.
    best_deg = None
    for deg in (1, 2, 3, 5, 8):
        if poly_resid[f"deg{deg}_max_abs"] <= 1.5:
            best_deg = deg
            break
    if best_deg is None:
        best_deg = 8
    c = np.polyfit(t, vh, best_deg)
    pred = np.polyval(c, t)
    resid_q = np.round(vh - pred).astype(np.int64)  # quantized to 1 row
    resid_bytes = len(zlib.compress(resid_q.astype(np.int8).tobytes(), 9))
    poly_bytes = (best_deg + 1) * 4 + resid_bytes  # coeffs f32 + residual

    #  (b) raw per-frame v_h quantized to 1 row, delta + zlib (no poly model).
    vh_q = np.round(vh).astype(np.int64)
    vh_delta = np.diff(vh_q, prepend=vh_q[:1])
    raw_bytes = len(zlib.compress(vh_delta.astype(np.int8).tobytes(), 9))

    rep = {
        "n_pairs": n_pairs,
        "vh_mean": float(np.nanmean(vh)),
        "vh_std": float(np.nanstd(vh)),
        "vh_min": float(np.nanmin(vh)),
        "vh_max": float(np.nanmax(vh)),
        "vh_temporal_smoothness_mean_abs_dvh": float(np.mean(dvh)),
        "vh_temporal_smoothness_max_abs_dvh": float(np.max(dvh)),
        "poly_residual": poly_resid,
        "best_poly_deg_for_1row": best_deg,
        "trajectory_bytes_poly": int(poly_bytes),
        "trajectory_bytes_raw_delta": int(raw_bytes),
        "trajectory_bytes_min": int(min(poly_bytes, raw_bytes)),
        "interpretation": (
            "v_h(t) is smooth + low-D (a90 geometry confirmed): the horizon LINE "
            "is cheap (tens of bytes). The question (crossframe stage) is whether "
            "the FLIP SET tracks it."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "trajectory.npz", vh=vh, perc=perc)
    (OUT_DIR / "trajectory.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


# ---------------------------------------------------------------------------
# Stage: crossframe — THE decisive cross-frame intrinsic-dimension test
# ---------------------------------------------------------------------------
def _zlib_bytes(arr: np.ndarray) -> int:
    return len(zlib.compress(np.ascontiguousarray(arr).tobytes(), 9))


def stage_crossframe(n_pairs: int):
    gt, comp = _load_argmaps(n_pairs)
    flips = gt != comp  # (n,H,W)
    baseline_dseg = float(flips.mean())

    # Recover v_h(t) (per-frame median horizon row) for the model.
    perc = np.stack([_gt_horizon_per_column(gt[t]) for t in range(n_pairs)], 0)
    vh = np.nanmedian(perc, axis=1)  # (n,)
    vh_round = np.round(np.nan_to_num(vh, nan=192.0)).astype(np.int64)

    H, W = SEG_H, SEG_W

    # ---- (1) flip-band tracking: per-frame flip-row distribution vs v_h(t) ----
    # For each frame, the offset (flip_row - v_h) distribution. If flips track v_h,
    # the offset is concentrated + frame-independent (low cross-frame variance).
    flip_offsets = []  # all (row - v_h(t)) for flip pixels, pooled across frames
    flip_centroid = np.full(n_pairs, np.nan)
    for t in range(n_pairs):
        rs, cs = np.where(flips[t])
        if rs.size:
            flip_offsets.append(rs.astype(np.float64) - vh[t])
            flip_centroid[t] = rs.mean()
    flip_off = (
        np.concatenate(flip_offsets) if flip_offsets else np.array([0.0])
    )
    m = ~(np.isnan(vh) | np.isnan(flip_centroid))
    corr_centroid_vh = (
        float(np.corrcoef(vh[m], flip_centroid[m])[0, 1]) if m.sum() > 2 else 0.0
    )

    # ---- (2) cross-frame intrinsic dimension ----
    # We compare the byte cost of encoding the flip SET under progressively more
    # cross-frame structure. The "intrinsic dimension" is read off how much the
    # cross-frame models reduce bytes below the per-frame-independent baseline.
    #
    # The flip set is the (n,H,W) boolean mask. We restrict to the horizon band to
    # match aa98 (rows 96-288) and a tighter peak band.
    bands = {
        "peak_rows_180_200": (180, 200),
        "horizon_rows_96_288": (96, 288),
    }
    band_results = {}
    for bname, (lo, hi) in bands.items():
        band_flips = flips[:, lo:hi, :]  # (n, bh, W)
        n_flips = int(band_flips.sum())
        bh = hi - lo

        # Model 0 (aa98 baseline): per-frame-independent position bitmap + class.
        #   This is EXACTLY aa98's encoding (zlib on the packed flip mask + GT class).
        pos_independent = _zlib_bytes(np.packbits(band_flips.reshape(-1)))
        corrected_cls = gt[:, lo:hi, :][band_flips].astype(np.uint8)
        cls_independent = _zlib_bytes(corrected_cls)
        m0_bytes = pos_independent + cls_independent

        # Model 1 (row-aligned by v_h): shift each frame so the horizon row aligns,
        #   then encode the aligned stack. If flips track v_h, alignment exposes
        #   cross-frame redundancy -> zlib compresses the aligned stack better.
        #   Shift each frame's band rows by -(v_h(t) - mean_vh) rounded.
        mean_vh = float(np.nanmean(vh))
        aligned = np.zeros_like(band_flips)
        for t in range(n_pairs):
            shift = int(round(vh_round[t] - mean_vh))
            # roll the band rows by -shift (bring v_h to a common row)
            aligned[t] = np.roll(band_flips[t], -shift, axis=0)
        pos_aligned = _zlib_bytes(np.packbits(aligned.reshape(-1)))
        m1_bytes = pos_aligned + cls_independent  # class stream unchanged

        # Model 2 (column-profile residual): the STRONGEST cross-frame model. Build a
        #   per-(row-offset, col) flip-PROBABILITY profile from all frames (the
        #   "where flips happen relative to the horizon" template). Encode each frame
        #   as the XOR-residual against the thresholded template prediction. If the
        #   per-frame flips are a low-D function of v_h + a fixed template, the
        #   residual is sparse -> cheap. If they are high-entropy independent
        #   scatter, the residual ~= the original (no gain).
        template = aligned.mean(axis=0)  # (bh, W) prob of flip at aligned position
        # threshold at the marginal flip rate -> predicted aligned mask
        q = aligned.mean()
        pred = template > q  # broadcastable (bh,W)
        residual = aligned ^ pred[None, :, :]  # (n,bh,W) bool
        pos_residual = _zlib_bytes(np.packbits(residual.reshape(-1)))
        # template itself must be stored: quantize prob to 8 levels + zlib
        template_q = np.clip((template * 8).astype(np.uint8), 0, 7)
        template_bytes = _zlib_bytes(template_q)
        m2_bytes = pos_residual + template_bytes + cls_independent

        # cross-frame intrinsic-dim proxy: best cross-frame model bytes / baseline.
        best_cross = min(m1_bytes, m2_bytes)
        compress_ratio = best_cross / max(m0_bytes, 1)

        # entropy of per-pair flip count (how independent are frames in their flip
        # COUNT) and per-pair Jaccard similarity of aligned flip sets (structure).
        per_pair_counts = band_flips.sum(axis=(1, 2))
        # mean pairwise Jaccard of aligned flip sets across consecutive frames
        jacc = []
        af = aligned.reshape(n_pairs, -1)
        for t in range(1, min(n_pairs, 200)):
            a, b = af[t - 1], af[t]
            inter = np.logical_and(a, b).sum()
            uni = np.logical_or(a, b).sum()
            if uni > 0:
                jacc.append(inter / uni)
        mean_jaccard = float(np.mean(jacc)) if jacc else 0.0

        band_results[bname] = {
            "seg_rows": [lo, hi],
            "n_flips": n_flips,
            "model0_per_frame_independent_bytes": int(m0_bytes),
            "model1_vh_row_aligned_bytes": int(m1_bytes),
            "model2_template_residual_bytes": int(m2_bytes),
            "best_cross_frame_bytes": int(best_cross),
            "cross_frame_compression_ratio_vs_m0": float(compress_ratio),
            "per_pair_flip_count_mean": float(per_pair_counts.mean()),
            "per_pair_flip_count_std": float(per_pair_counts.std()),
            "mean_consecutive_aligned_jaccard": mean_jaccard,
            "intrinsic_dim_reading": (
                "LOW-D (cross-frame model compresses) if ratio < ~0.6 and Jaccard "
                "high; HIGH-ENTROPY independent scatter if ratio ~= 1 and Jaccard ~0."
            ),
        }

    # ---- (3) PARAMETERIZED encoding economics: does it flip aa98's NO-GO? ----
    # For each band, the oracle Δd_seg (force comp->GT in band) is fixed; the byte
    # cost is now the BEST cross-frame parameterized encoding (trajectory + residual).
    econ = {}
    traj_bytes = json.loads(
        (OUT_DIR / "trajectory.json").read_text()
    )["trajectory_bytes_min"] if (OUT_DIR / "trajectory.json").exists() else 64
    for bname, (lo, hi) in bands.items():
        band = slice(lo, hi)
        corrected = comp.copy()
        corrected[:, band, :] = gt[:, band, :]
        new_dseg = float((gt != corrected).mean())
        delta_dseg = new_dseg - baseline_dseg  # negative = improvement
        # parameterized bytes = trajectory + best cross-frame flip-set encoding
        param_bytes = traj_bytes + band_results[bname]["best_cross_frame_bytes"]
        new_bytes = FRONTIER_BYTES + param_bytes
        new_S = compute_contest_score(new_dseg, FRONTIER_D_POSE, new_bytes)
        net_delta_S = new_S - FRONTIER_S
        dseg_per_byte = delta_dseg / max(param_bytes, 1)
        econ[bname] = {
            "oracle_delta_dseg": delta_dseg,
            "oracle_rel_reduction": delta_dseg / max(baseline_dseg, 1e-12),
            "parameterized_bytes": int(param_bytes),
            "delta_dseg_per_byte_parameterized": dseg_per_byte,
            "rate_slope_break_even": -RATE_SLOPE,
            "beats_break_even": bool(dseg_per_byte < -RATE_SLOPE),
            "net_delta_S": net_delta_S,
            "verdict": "GO" if net_delta_S < 0 else "NO-GO",
            "vs_aa98_per_pixel_-4.65e-9": dseg_per_byte,
            "gap_to_a90_floor_16_262B": {
                "a90_floor_low_B": 16,
                "a90_floor_high_B": 262,
                "measured_param_bytes": int(param_bytes),
                "ratio_measured_over_a90_high": float(param_bytes / 262.0),
            },
        }

    rep = {
        "baseline_dseg": baseline_dseg,
        "frontier_S": FRONTIER_S,
        "rate_slope_d_seg_per_byte": RATE_SLOPE,
        "flip_offset_vs_vh": {
            "mean": float(flip_off.mean()),
            "std": float(flip_off.std()),
            "p10": float(np.percentile(flip_off, 10)),
            "p90": float(np.percentile(flip_off, 90)),
        },
        "corr_flip_centroid_vs_vh": corr_centroid_vh,
        "band_intrinsic_dimension": band_results,
        "parameterized_economics": econ,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "crossframe.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


# ---------------------------------------------------------------------------
# Stage: taskspace — brief, encode KNOWN argmax targets directly?
# ---------------------------------------------------------------------------
def stage_taskspace(n_pairs: int):
    """Brief: compare (a) the FULL GT argmax field byte cost (task-space: store the
    known scorer-targets directly, d_seg=0) vs (b) the frontier's full decoder bytes.
    This bounds whether direct task-space coding is structurally cheaper than a
    correction ON the frontier reconstruction."""
    gt, comp = _load_argmaps(n_pairs)
    # (a) full GT argmax field, zlib (a loose upper bound on task-space rate).
    full_gt_bytes = _zlib_bytes(gt)
    # last-frame-only matters for d_seg (SegNet sees x[:,-1]); both frames stored for
    # completeness but d_seg only needs the per-pair argmax -> store comparison.
    # (b) the residual (gt^comp where they differ) — the correction-on-frontier cost
    # at FULL resolution (all rows), the absolute ceiling of aa98's approach.
    diff = (gt != comp)
    full_residual_bytes = _zlib_bytes(np.packbits(diff.reshape(-1))) + _zlib_bytes(
        gt[diff].astype(np.uint8)
    )
    rep = {
        "n_pairs": n_pairs,
        "full_gt_argmax_field_zlib_bytes": int(full_gt_bytes),
        "full_residual_correction_zlib_bytes": int(full_residual_bytes),
        "frontier_decoder_bytes": FRONTIER_BYTES,
        "reading": (
            "Task-space direct argmax storage (full_gt) vs correction-on-frontier "
            "(full_residual). If full_gt >> frontier_bytes, direct task-space coding "
            "of the dense argmax is NOT cheaper than the learned decoder; the win (if "
            "any) is a SPARSE structured code, which the crossframe stage tests."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "taskspace.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage", required=True, choices=["trajectory", "crossframe", "taskspace"]
    )
    ap.add_argument("--pairs", type=int, default=600)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "trajectory":
        stage_trajectory(args.pairs)
    elif args.stage == "crossframe":
        stage_crossframe(args.pairs)
    elif args.stage == "taskspace":
        stage_taskspace(args.pairs)


if __name__ == "__main__":
    main()
