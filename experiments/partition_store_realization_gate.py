# SPDX-License-Identifier: MIT
"""PHASE 1 — the DECISIVE realization gate for the non-neural partition STORE.

The prior probe (``experiments/yousfi_partition_topaiml_probe.py`` +
``reports/yousfi_partition_topaiml.json``) priced the partition store at
``d_seg=0`` because it stored ``L*`` (the SegNet argmax) losslessly.  But the
EVALUATOR never sees a partition — it runs SegNet on FRAMES at camera resolution
``(874, 1164, 3)`` and scores

    d_seg = mean[ argmax SegNet(comp_frame1) != argmax SegNet(gt_frame1) ]   (384x512)

So the store only helps if the inflate REALIZES the stored partition as a frame
whose SegNet argmax — THROUGH THE EXACT EVAL CHAIN — reproduces ``L*``.  The exact
chain a realized frame goes through (``upstream/modules.py:107`` +
``upstream/frame_utils.py``):

    painted camera-res frame (874x1164, uint8)
      -> SegNet.preprocess_input: take last frame, bilinear interpolate -> (384, 512)
      -> rgb_to_yuv6 / normalize -> SegNet -> argmax (384x512)
      -> compare to argmax SegNet(GT frame1)        == the REAL realized d_seg.

This probe MEASURES that realized d_seg (NO FAKE — never assumes d_seg=0):

  (a) per-class canonical RGB ``mu_c`` = mean GT-frame RGB over pixels whose
      GT-SegNet argmax (UPSAMPLED to camera res) is class c.  (PR#56 class-canonical
      / grayscale-LUT paradigm.)  Optional: optimize ``mu_c`` to maximise SegNet
      agreement.
  (b) paint the camera-res frame: pixel of class c <- mu_c, where the class map is
      ``L*`` nearest-upsampled to camera res.  Run it through the EXACT chain and
      measure realized d_seg + per-region (interior vs boundary) survival.
  (c) boundary-aware realization variants to LIFT survival: flat-fill (baseline),
      boundary-dilation, and a 2px boundary guard.
  (d) GATE: rate(0.182) + 100*realized_d_seg + pose(0.017) vs frontier 0.191 / 0.15.

Authority: real CPU-torch SegNet; GT decode via upstream ``yuv420_to_rgb`` ONLY;
NEVER MPS.  ``[contest-CPU advisory]`` NON-PROMOTABLE.  Every realized d_seg is the
REAL argmax-flip rate from the real SegNet through the exact resize/uint8 chain.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))

from tac.boundary_math.seg_core import (
    decode_gt_frame1_pairs,
    load_real_segnet,
)

N_CLASSES = 5
CAMERA_H, CAMERA_W = 874, 1164  # upstream camera_size = (1164 W, 874 H)
SEG_H, SEG_W = 384, 512  # upstream segnet_model_input_size = (512 W, 384 H)

# Frontier reference. [contest-CPU advisory] per CLAUDE.md pointer.
FRONTIER_TOTAL = 0.19110
FRONTIER_POSE_TERM = 0.017
SUB015_TARGET = 0.15
# The store's lossless rate term (best template = temporal), from the prior probe.
# Phase-1 gate uses this as the rate the store pays; Phase 2 tightens it.
STORE_RATE_LOSSLESS = 0.1824


def _segnet_argmax_camera(segnet, frame_camera_hwc_uint8: np.ndarray) -> np.ndarray:
    """Exact eval chain: camera-res frame -> SegNet.preprocess_input -> argmax (384x512).

    Mirrors ``upstream/modules.py`` SegNet.preprocess_input EXACTLY: build a
    (1, 2, H, W, 3) pair (SegNet reads only the last frame), permute to
    (b t c h w), bilinear-interpolate to (384, 512), forward, argmax.  CPU-torch.
    """
    import torch

    r = np.asarray(frame_camera_hwc_uint8)
    if r.ndim != 3 or r.shape[-1] != 3:
        raise ValueError(f"frame must be (H, W, 3); got {r.shape}")
    # (1, 2, H, W, 3) — duplicate last frame (SegNet uses only x[:, -1]).
    pair = torch.from_numpy(np.stack([r, r], axis=0)[None]).float()
    xp = pair.permute(0, 1, 4, 2, 3).contiguous().float()  # (1, 2, 3, H, W)
    with torch.inference_mode():
        seg_in = segnet.preprocess_input(xp)  # (1, 3, 384, 512)
        logits = segnet(seg_in)  # (1, 5, 384, 512)
        argmax = logits.argmax(dim=1)[0]
    return argmax.detach().cpu().numpy().astype(np.int64)


def _upsample_labels_nearest(lstar_seg: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour upsample a (384,512) label map to (h, w) (camera res)."""
    sh, sw = lstar_seg.shape
    ri = (np.arange(h) * sh / h).astype(np.int64).clip(0, sh - 1)
    ci = (np.arange(w) * sw / w).astype(np.int64).clip(0, sw - 1)
    return lstar_seg[ri][:, ci]


def _boundary_mask_seg(lstar_seg: np.ndarray) -> np.ndarray:
    """4-neighbour boundary pixels of the (384,512) partition (where the realized
    d_seg flips concentrate — the survival-wall locus)."""
    a = lstar_seg
    b = np.zeros_like(a, dtype=bool)
    b[:-1, :] |= a[:-1, :] != a[1:, :]
    b[1:, :] |= a[:-1, :] != a[1:, :]
    b[:, :-1] |= a[:, :-1] != a[:, 1:]
    b[:, 1:] |= a[:, :-1] != a[:, 1:]
    return b


def _canonical_mu_global(
    gt_frames: list[np.ndarray], gt_argmax_cam: list[np.ndarray], n_classes: int
) -> np.ndarray:
    """Per-class canonical RGB mu_c = mean GT-frame RGB over camera pixels whose
    GT-SegNet argmax (upsampled to camera res) is class c, pooled over all frames.

    PR#56 class-canonical paradigm: the single color SegNet most reliably maps to
    class c.  Mean over the actual GT appearance of class-c pixels is the MLE start.
    """
    sums = np.zeros((n_classes, 3), dtype=np.float64)
    cnts = np.zeros(n_classes, dtype=np.float64)
    for f, am in zip(gt_frames, gt_argmax_cam, strict=True):
        ff = f.astype(np.float64)
        for c in range(n_classes):
            m = am == c
            n = float(m.sum())
            if n > 0:
                sums[c] += ff[m].sum(axis=0)
                cnts[c] += n
    mu = np.zeros((n_classes, 3), dtype=np.float64)
    for c in range(n_classes):
        mu[c] = sums[c] / cnts[c] if cnts[c] > 0 else 128.0
    return mu


def _paint_flat(label_cam: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Flat-fill: each camera pixel of class c <- mu_c.  uint8 frame (874,1164,3)."""
    frame = mu[label_cam]  # (H, W, 3) float
    return np.clip(np.round(frame), 0, 255).astype(np.uint8)


def _dilate_majority(label_cam: np.ndarray, n_iter: int, n_classes: int) -> np.ndarray:
    """Morphological majority smoothing of the camera-res label map (boundary-aware
    realization): replace each boundary-adjacent pixel by the majority of its
    4-neighbours, ``n_iter`` times.  Thickens the dominant side of a boundary so the
    bilinear downsample lands more pixels on the correct class.
    """
    if n_iter <= 0:
        return label_cam
    out = label_cam.copy()
    for _ in range(n_iter):
        h, w = out.shape
        # one-hot then 4-neighbour sum -> argmax = local majority (incl. self).
        oh = np.zeros((n_classes, h, w), dtype=np.int32)
        for c in range(n_classes):
            oh[c] = (out == c).astype(np.int32)
        acc = oh.copy()
        acc[:, 1:, :] += oh[:, :-1, :]
        acc[:, :-1, :] += oh[:, 1:, :]
        acc[:, :, 1:] += oh[:, :, :-1]
        acc[:, :, :-1] += oh[:, :, 1:]
        out = acc.argmax(axis=0).astype(np.int64)
    return out


def _optimize_mu_for_segnet(
    seg,
    lstars_seg: list[np.ndarray],
    mu_init: np.ndarray,
    n_classes: int,
    *,
    n_probe: int = 4,
    rounds: int = 2,
) -> np.ndarray:
    """Greedily refine per-class canonical RGB to MAXIMISE SegNet agreement.

    For each class c, paint flat frames over a small probe subset trying a grid of
    candidate colors (the GT-mean plus saturated/dimmed/grayscale variants) and keep
    the color that yields the LOWEST realized d_seg across the probe frames.  This is
    the PR#56 'pick the color SegNet most reliably maps to class c' optimization, run
    against the REAL SegNet through the exact chain (NO FAKE).  Probe subset keeps it
    cheap; the chosen mu is then evaluated on the full set by the caller.
    """
    mu = mu_init.copy()
    probe_lstars = lstars_seg[: min(n_probe, len(lstars_seg))]

    def realized_dseg_for_mu(mu_try: np.ndarray) -> float:
        ds = []
        for lstar in probe_lstars:
            label_cam = _upsample_labels_nearest(lstar, CAMERA_H, CAMERA_W)
            painted = _paint_flat(label_cam, mu_try)
            realized = _segnet_argmax_camera(seg, painted)
            ds.append(d_seg(realized, lstar))
        return float(np.mean(ds))

    for _ in range(rounds):
        for c in range(n_classes):
            base = mu[c]
            # candidate palette for class c: GT-mean, brightness/saturation nudges,
            # and a couple of extreme anchors (SegNet boundaries are color-driven).
            cands = [base]
            for scale in (0.6, 0.8, 1.2, 1.4):
                cands.append(np.clip(base * scale, 0, 255))
            gray = np.full(3, float(np.clip(base.mean(), 0, 255)))
            cands.append(gray)
            for shift in (-40.0, 40.0):
                cands.append(np.clip(base + shift, 0, 255))
            best_c, best_d = base, np.inf
            for cand in cands:
                mu_try = mu.copy()
                mu_try[c] = cand
                d = realized_dseg_for_mu(mu_try)
                if d < best_d:
                    best_d, best_c = d, cand
            mu[c] = best_c
    return mu


def d_seg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.count_nonzero(a != b)) / a.size


def main() -> None:
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    t0 = time.time()
    seg = load_real_segnet("cpu")

    gt_frames: list[np.ndarray] = []
    lstars_seg: list[np.ndarray] = []  # the stored partition = SegNet(gt) argmax (384x512)
    gt_argmax_cam: list[np.ndarray] = []  # GT argmax upsampled to camera res (for mu)
    for _, _f0, f1 in decode_gt_frame1_pairs(n_pairs=n_pairs):
        f1 = np.asarray(f1)
        # The stored target L* = the EXACT SegNet argmax of the GT camera frame.
        lstar = _segnet_argmax_camera(seg, f1)  # (384, 512)
        gt_frames.append(f1)
        lstars_seg.append(lstar)
        gt_argmax_cam.append(_upsample_labels_nearest(lstar, CAMERA_H, CAMERA_W))
    n = len(lstars_seg)
    decode_secs = time.time() - t0

    # --- (a) canonical mu_c (global over all frames), then SegNet-optimized mu.
    mu = _canonical_mu_global(gt_frames, gt_argmax_cam, N_CLASSES)
    t_opt = time.time()
    mu_opt = _optimize_mu_for_segnet(seg, lstars_seg, mu, N_CLASSES)
    mu_opt_secs = time.time() - t_opt

    # --- realization variants. For each, paint -> exact chain -> realized d_seg.
    #   flat_fill      : mu = GT-mean per class (baseline).
    #   mu_optimized   : mu greedily refined to maximise SegNet agreement.
    #   dilate1/2      : boundary-aware morphological smoothing on the camera label map.
    #   gt_blend_0p5   : DIAGNOSTIC UPPER BOUND — blend painted-flat with the real GT
    #                    frame within each region (not a valid store; bounds how much
    #                    'natural texture' would help if it were free).
    variants = {
        "flat_fill": {"dilate": 0, "mu": "mean", "gt_blend": 0.0},
        "mu_optimized": {"dilate": 0, "mu": "opt", "gt_blend": 0.0},
        "dilate1": {"dilate": 1, "mu": "opt", "gt_blend": 0.0},
        "dilate2": {"dilate": 2, "mu": "opt", "gt_blend": 0.0},
        "gt_blend_0p5_diagnostic": {"dilate": 0, "mu": "opt", "gt_blend": 0.5},
    }
    results: dict[str, dict] = {}
    for vname, vcfg in variants.items():
        mu_use = mu_opt if vcfg["mu"] == "opt" else mu
        per_pair = []
        for lstar, _gt in zip(lstars_seg, gt_frames, strict=True):
            label_cam = _upsample_labels_nearest(lstar, CAMERA_H, CAMERA_W)
            label_cam = _dilate_majority(label_cam, vcfg["dilate"], N_CLASSES)
            painted = _paint_flat(label_cam, mu_use).astype(np.float64)  # (874,1164,3)
            if vcfg["gt_blend"] > 0.0:
                a = float(vcfg["gt_blend"])
                painted = (1.0 - a) * painted + a * _gt.astype(np.float64)
            painted = np.clip(np.round(painted), 0, 255).astype(np.uint8)
            realized = _segnet_argmax_camera(seg, painted)  # (384,512) argmax
            dd = d_seg(realized, lstar)
            # per-region survival: interior vs boundary of the stored partition.
            bmask = _boundary_mask_seg(lstar)
            interior = ~bmask
            flips = realized != lstar
            b_flip = float((flips & bmask).sum()) / max(1, int(bmask.sum()))
            i_flip = float((flips & interior).sum()) / max(1, int(interior.sum()))
            per_pair.append({
                "d_seg_realized": dd,
                "boundary_flip_rate": b_flip,
                "interior_flip_rate": i_flip,
                "boundary_px_frac": float(bmask.sum()) / lstar.size,
            })
        mean_dseg = float(np.mean([p["d_seg_realized"] for p in per_pair]))
        mean_bflip = float(np.mean([p["boundary_flip_rate"] for p in per_pair]))
        mean_iflip = float(np.mean([p["interior_flip_rate"] for p in per_pair]))
        mean_bfrac = float(np.mean([p["boundary_px_frac"] for p in per_pair]))
        store_score = STORE_RATE_LOSSLESS + 100.0 * mean_dseg + FRONTIER_POSE_TERM
        results[vname] = {
            "mean_d_seg_realized": mean_dseg,
            "mean_boundary_flip_rate": mean_bflip,
            "mean_interior_flip_rate": mean_iflip,
            "mean_boundary_px_frac": mean_bfrac,
            "store_score_rate0182_plus_100dseg_plus_pose": store_score,
            "beats_frontier": store_score < FRONTIER_TOTAL,
            "beats_sub015": store_score < SUB015_TARGET,
            "per_pair_sample": per_pair[: min(8, len(per_pair))],
        }

    # --- GATE synthesis: best realized d_seg, and what it implies.
    best_variant = min(results, key=lambda k: results[k]["mean_d_seg_realized"])
    best_dseg = results[best_variant]["mean_d_seg_realized"]
    best_store_score = results[best_variant][
        "store_score_rate0182_plus_100dseg_plus_pose"
    ]
    # The realized d_seg the store would need to beat each bar at the lossless rate.
    dseg_to_beat_frontier = (FRONTIER_TOTAL - STORE_RATE_LOSSLESS - FRONTIER_POSE_TERM) / 100.0
    dseg_to_beat_sub015 = (SUB015_TARGET - STORE_RATE_LOSSLESS - FRONTIER_POSE_TERM) / 100.0

    any_beats_frontier = any(r["beats_frontier"] for r in results.values())
    any_beats_sub015 = any(r["beats_sub015"] for r in results.values())

    if any_beats_frontier:
        gate = "PROCEED_TO_PHASE2_REALIZATION_VIABLE"
    elif best_dseg <= dseg_to_beat_frontier * 3:
        # close enough that a tighter coder (phase 2) could close it.
        gate = "BORDERLINE_TIGHTEN_CODER_PHASE2"
    else:
        gate = "DEFER_REALIZATION_WALL_DSEG_BELONGS_IN_TRAINING"

    out = {
        "authority": "contest-CPU-advisory",
        "promotable": False,
        "phase": 1,
        "note": (
            "PHASE 1 realization gate: paint the STORED partition (SegNet-argmax of "
            "GT) into a camera-res frame with per-class canonical RGB, run it through "
            "the EXACT eval chain (camera 874x1164 -> SegNet bilinear->384x512 -> "
            "argmax), and MEASURE the REAL realized d_seg (never assume d_seg=0)."
        ),
        "n_frames": n,
        "decode_and_lstar_seconds": round(decode_secs, 1),
        "camera_size_hw": [CAMERA_H, CAMERA_W],
        "seg_grid_hw": [SEG_H, SEG_W],
        "canonical_mu_rgb_per_class": mu.tolist(),
        "segnet_optimized_mu_rgb_per_class": mu_opt.tolist(),
        "mu_optimization_seconds": round(mu_opt_secs, 1),
        "frontier": {
            "total": FRONTIER_TOTAL,
            "sub015_target": SUB015_TARGET,
            "store_rate_lossless": STORE_RATE_LOSSLESS,
            "pose_term_inherited": FRONTIER_POSE_TERM,
        },
        "dseg_budget": {
            "realized_d_seg_to_beat_frontier_at_lossless_rate": dseg_to_beat_frontier,
            "realized_d_seg_to_beat_sub015_at_lossless_rate": dseg_to_beat_sub015,
        },
        "realization_variants": results,
        "gate": {
            "code": gate,
            "best_variant": best_variant,
            "best_realized_d_seg": best_dseg,
            "best_store_score": best_store_score,
            "any_variant_beats_frontier": any_beats_frontier,
            "any_variant_beats_sub015": any_beats_sub015,
            "interpretation": (
                f"Best realization '{best_variant}' yields realized d_seg="
                f"{best_dseg:.5f} (store score {best_store_score:.4f} vs frontier "
                f"{FRONTIER_TOTAL}). The store needs realized d_seg <= "
                f"{dseg_to_beat_frontier:.5f} to beat the frontier and <= "
                f"{dseg_to_beat_sub015:.5f} for sub-0.15 at the lossless rate. "
                + (
                    "Realization is viable — PROCEED to Phase 2 (tighten the coder)."
                    if any_beats_frontier else
                    "Realized d_seg exceeds the budget: the painted-partition store "
                    "faces a survival wall through the resize/uint8/SegNet chain — "
                    "interiors survive but the boundary band flips, exactly the "
                    "witness-sidecar failure mode. d_seg belongs in training."
                )
            ),
        },
    }
    outpath = REPO / "reports" / "partition_store_realization_gate.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
