"""ddm_sx2 G2 -- scorer-FREE hit-rate of GENERIC contour extractors against the SegNet separatrix.

Gate queued by ``ddm_sx1`` (.omx/research/ddm_sx1_separatrix_carrier_20260803.md §10 G2):

    "Run a generic contour extractor on decoded frames; score against ``lstars``; report
     rho = fraction of the 2,551,382 boundary px located within +/-1 px.
     rho > 0.5 makes S1's occupancy map effectively free."

This module runs the gate AND reports the quantity the gate's own conclusion actually needs.
Recall alone cannot make a map free: a predictor that fires everywhere has recall 1.0 and
carries no information. The decisive quantity is the RESIDUAL CELL-MAP COST -- the conditional
entropy of the true boundary-cell occupancy given the free prediction -- measured against
``sx1``'s iid upper bounds (36,798 B at c=16, 12,371 B at c=32).

Axis: [macOS-CPU advisory]. NO contest scorer forward is run anywhere in this file.
score_claim=false, promotion_eligible=false, rank_or_kill_eligible=false.

Substrate note (load-bearing, do not strip): the extractor is evaluated on GROUND-TRUTH RGB
resized through the exact ``SegNet.preprocess_input`` path. That is the vehicle-independent
CEILING for any decode -- a lossy decode can only lose edges relative to it. A ceiling that
fails is a decisive kill; a ceiling that passes does NOT establish the decoded-input number.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

# SegNet scorer grid, read from upstream (H, W).
SCORER_H = 384
SCORER_W = 512
DEFAULT_LSTARS = "experiments/results/ot_offset_n600_modal_20260709/gt_n600_lstars_slim.npz"
DEFAULT_FRAMES = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def _resize_to_scorer_grid(frame_hwc_u8: np.ndarray) -> np.ndarray:
    """Reproduce ``SegNet.preprocess_input`` exactly: bilinear interpolate to (384, 512).

    upstream/modules.py::SegNet.preprocess_input calls
    ``torch.nn.functional.interpolate(x, size=(H, W), mode='bilinear')`` on the float RGB
    tensor with ``align_corners`` at its default (False). We replicate that operator rather
    than substituting a cv2 resize, because the two do not agree at non-integer strides
    (1164/512 = 2.2734, 874/384 = 2.2760).
    """
    t = torch.from_numpy(frame_hwc_u8).permute(2, 0, 1).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, size=(SCORER_H, SCORER_W), mode="bilinear")
    return t.squeeze(0).permute(1, 2, 0).numpy()


def label_boundary_4conn(labels: np.ndarray) -> np.ndarray:
    """Boolean map: pixel differs from at least one 4-connected neighbour."""
    b = np.zeros(labels.shape, dtype=bool)
    b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    b[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    b[:-1, :] |= labels[:-1, :] != labels[1:, :]
    b[1:, :] |= labels[:-1, :] != labels[1:, :]
    return b


_K3 = np.ones((3, 3), np.uint8)


def dilate1(mask: np.ndarray) -> np.ndarray:
    """3x3 dilation == Chebyshev radius 1 == the gate's '+/-1 px' tolerance."""
    return cv2.dilate(mask.astype(np.uint8), _K3, iterations=1).astype(bool)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _topk_mask(score: np.ndarray, density: float) -> np.ndarray:
    """Keep the highest-scoring ``density`` fraction of pixels.

    Equal-count thresholding is the only fair way to race edge operators whose score scales
    differ. It fixes the predictor's budget to the GT boundary density so precision and recall
    are directly comparable across operators and against the random control.
    """
    n = score.size
    k = max(1, round(density * n))
    if k >= n:
        return np.ones(score.shape, dtype=bool)
    # argpartition (not a >= threshold) so ties cannot silently inflate the budget. The static
    # prior has huge tie mass at exactly 0, and a threshold comparison would hand it far more
    # than k pixels -- an unfair advantage over the continuous-valued gradient operators.
    keep = np.argpartition(score.ravel(), n - k)[n - k :]
    flat = np.zeros(n, dtype=bool)
    flat[keep] = True
    return flat.reshape(score.shape)


# --- generic extractors. Every one is a fixed algorithm with no video-derived table. ---


def ex_sobel_luma(rgb: np.ndarray, density: float) -> np.ndarray:
    y = _luma(rgb)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return _topk_mask(np.hypot(gx, gy), density)


def ex_sobel_rgbmax(rgb: np.ndarray, density: float) -> np.ndarray:
    best = None
    for c in range(3):
        ch = rgb[..., c].astype(np.float32)
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        m = np.hypot(gx, gy)
        best = m if best is None else np.maximum(best, m)
    return _topk_mask(best, density)


def ex_morph_gradient(rgb: np.ndarray, density: float) -> np.ndarray:
    y = _luma(rgb).astype(np.float32)
    g = cv2.morphologyEx(y, cv2.MORPH_GRADIENT, _K3)
    return _topk_mask(g, density)


def ex_laplacian(rgb: np.ndarray, density: float) -> np.ndarray:
    y = _luma(rgb).astype(np.float32)
    return _topk_mask(np.abs(cv2.Laplacian(y, cv2.CV_32F, ksize=3)), density)


def ex_canny(rgb: np.ndarray, density: float) -> np.ndarray:
    """Canny at a per-frame adaptive hysteresis, then equal-count trim by gradient magnitude.

    Canny's own output count is not controllable to an exact density, so we take its edge set
    and rank surviving pixels by Sobel magnitude to hit the same budget as the other operators.
    """
    y = np.clip(_luma(rgb), 0, 255).astype(np.uint8)
    hi = float(np.percentile(y, 90))
    lo = 0.4 * hi
    e = cv2.Canny(y, lo, hi, L2gradient=True) > 0
    gx = cv2.Sobel(y.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    score = np.hypot(gx, gy) * e
    return _topk_mask(score, density)


def ex_random(rgb: np.ndarray, density: float) -> np.ndarray:
    """CONTROL. A generic-triple control per the project's 'derived-or-raced' discipline.

    Seeded on the frame content so it is deterministic and reproducible.
    """
    rng = np.random.default_rng(abs(int(rgb[::97, ::89, 0].sum())) % (2**31))
    return _topk_mask(rng.random((SCORER_H, SCORER_W)).astype(np.float32), density)


EXTRACTORS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "sobel_luma": ex_sobel_luma,
    "sobel_rgbmax": ex_sobel_rgbmax,
    "morph_gradient": ex_morph_gradient,
    "laplacian": ex_laplacian,
    "canny": ex_canny,
    "random_control": ex_random,
}


def _binary_entropy_bits(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


@dataclass
class Accum:
    """Population accumulators. Every reported figure is a full-n600 total, never a mean of means."""

    gt_boundary_px: int = 0
    pred_px: int = 0
    gt_hit_px: int = 0  # GT boundary px within +/-1 of a predicted px  -> recall numerator
    pred_hit_px: int = 0  # predicted px within +/-1 of a GT boundary px -> precision numerator
    gt_hit_px_exact: int = 0  # exact-coincidence recall (tolerance 0)
    # cell-level confusion, per cell size
    cell_tp: dict[int, int] = field(default_factory=dict)
    cell_fp: dict[int, int] = field(default_factory=dict)
    cell_fn: dict[int, int] = field(default_factory=dict)
    cell_tn: dict[int, int] = field(default_factory=dict)
    frames: int = 0


def _cells_any(mask: np.ndarray, c: int) -> np.ndarray:
    h, w = mask.shape
    assert h % c == 0 and w % c == 0, f"cell size {c} must divide {h}x{w}"
    return mask.reshape(h // c, c, w // c, c).any(axis=(1, 3))


def run(
    n_pairs: int,
    cell_sizes: tuple[int, ...],
    lstars_path: str,
    frames_path: str,
    subset: str,
    out_json: str,
) -> dict:
    lz = np.load(lstars_path)
    lstars = lz["lstars"].astype(np.uint8)
    fz = np.load(frames_path)
    gt_f1 = fz["gt_f1"]
    total = int(min(n_pairs, lstars.shape[0], gt_f1.shape[0]))
    if subset == "all":
        idx = list(range(total))
    elif subset == "odd":
        idx = [i for i in range(total) if i % 2 == 1]
    elif subset == "even":
        idx = [i for i in range(total) if i % 2 == 0]
    else:
        raise ValueError(f"unknown subset {subset!r}")

    acc = {k: Accum() for k in EXTRACTORS}
    # static spatial prior: per-pixel boundary frequency, HELD OUT (built on even, scored on odd).
    prior_counts = np.zeros((SCORER_H, SCORER_W), dtype=np.int32)
    prior_frames = 0
    for i in range(total):
        if i % 2 == 0:
            prior_counts += label_boundary_4conn(lstars[i])
            prior_frames += 1
    prior_score = prior_counts.astype(np.float32) / max(1, prior_frames)

    density = None
    # first pass over the evaluation subset to fix the shared budget = observed GT density
    gt_b_total = 0
    for i in idx:
        gt_b_total += int(label_boundary_4conn(lstars[i]).sum())
    density = gt_b_total / (len(idx) * SCORER_H * SCORER_W)

    prior_mask = _topk_mask(prior_score, density)
    acc["static_prior_control"] = Accum()
    for cs in cell_sizes:
        for a in acc.values():
            a.cell_tp[cs] = a.cell_fp[cs] = a.cell_fn[cs] = a.cell_tn[cs] = 0

    for n, i in enumerate(idx):
        rgb = _resize_to_scorer_grid(gt_f1[i])
        gt_b = label_boundary_4conn(lstars[i])
        gt_b_d = dilate1(gt_b)
        gt_n = int(gt_b.sum())
        gt_cells = {cs: _cells_any(gt_b, cs) for cs in cell_sizes}

        for name in [*EXTRACTORS, "static_prior_control"]:
            pred = prior_mask if name == "static_prior_control" else EXTRACTORS[name](rgb, density)
            a = acc[name]
            pred_d = dilate1(pred)
            a.gt_boundary_px += gt_n
            a.pred_px += int(pred.sum())
            a.gt_hit_px += int((gt_b & pred_d).sum())
            a.gt_hit_px_exact += int((gt_b & pred).sum())
            a.pred_hit_px += int((pred & gt_b_d).sum())
            a.frames += 1
            for cs in cell_sizes:
                pc = _cells_any(pred, cs)
                gc = gt_cells[cs]
                a.cell_tp[cs] += int((pc & gc).sum())
                a.cell_fp[cs] += int((pc & ~gc).sum())
                a.cell_fn[cs] += int((~pc & gc).sum())
                a.cell_tn[cs] += int((~pc & ~gc).sum())
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(idx)} frames", flush=True)

    results = {}
    for name, a in acc.items():
        cells = {}
        for cs in cell_sizes:
            tp, fp, fn, tn = a.cell_tp[cs], a.cell_fp[cs], a.cell_fn[cs], a.cell_tn[cs]
            ncell = tp + fp + fn + tn
            occ = (tp + fn) / ncell
            pred_pos = (tp + fp) / ncell
            # marginal (transmit the map with no predictor): iid binary entropy of occupancy
            marginal_bits = _binary_entropy_bits(occ) * ncell
            # conditional (transmit only the correction to a FREE prediction)
            p_pos = tp + fp
            p_neg = fn + tn
            cond_bits = 0.0
            if p_pos:
                cond_bits += _binary_entropy_bits(tp / p_pos) * p_pos
            if p_neg:
                cond_bits += _binary_entropy_bits(fn / p_neg) * p_neg
            cells[cs] = {
                "n_cells": ncell,
                "gt_occupancy": occ,
                "pred_occupancy": pred_pos,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "cell_recall": tp / (tp + fn) if (tp + fn) else 0.0,
                "cell_precision": tp / (tp + fp) if (tp + fp) else 0.0,
                "marginal_map_bytes": marginal_bits / 8.0,
                "residual_map_bytes": cond_bits / 8.0,
                "map_bytes_saved": (marginal_bits - cond_bits) / 8.0,
                "map_free_fraction": 1.0 - (cond_bits / marginal_bits) if marginal_bits else 0.0,
            }
        results[name] = {
            "frames": a.frames,
            "gt_boundary_px": a.gt_boundary_px,
            "pred_px": a.pred_px,
            "rho_recall_pm1": a.gt_hit_px / a.gt_boundary_px if a.gt_boundary_px else 0.0,
            "recall_exact": a.gt_hit_px_exact / a.gt_boundary_px if a.gt_boundary_px else 0.0,
            "precision_pm1": a.pred_hit_px / a.pred_px if a.pred_px else 0.0,
            "cells": cells,
        }

    out = {
        "arm": "ddm_sx2",
        "gate": "G2",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "scorer_forwards_run": 0,
        "substrate": "GROUND-TRUTH frame_1 through SegNet.preprocess_input (CEILING, not decoded)",
        "subset": subset,
        "n_frames_evaluated": len(idx),
        "n_frames_available": total,
        "gt_boundary_density": density,
        "static_prior_built_on_frames": prior_frames,
        "results": results,
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--subset", choices=["all", "odd", "even"], default="odd")
    ap.add_argument("--cell-sizes", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--lstars", default=DEFAULT_LSTARS)
    ap.add_argument("--frames", default=DEFAULT_FRAMES)
    ap.add_argument("--out", default=".omx/research/ddm_sx2_g2_contour_hitrate.json")
    a = ap.parse_args()
    out = run(a.n_pairs, tuple(a.cell_sizes), a.lstars, a.frames, a.subset, a.out)
    print(f"\nsubset={out['subset']}  frames={out['n_frames_evaluated']}  density={out['gt_boundary_density']:.6f}")
    hdr = f"{'extractor':<22}{'rho(+-1)':>10}{'prec(+-1)':>11}"
    for cs in a.cell_sizes:
        hdr += f"{'c' + str(cs) + ' rec':>10}{'c' + str(cs) + ' resid B':>13}"
    print(hdr)
    for name, r in out["results"].items():
        line = f"{name:<22}{r['rho_recall_pm1']:>10.4f}{r['precision_pm1']:>11.4f}"
        for cs in a.cell_sizes:
            c = r["cells"][cs]
            line += f"{c['cell_recall']:>10.4f}{c['residual_map_bytes']:>13.0f}"
        print(line)


if __name__ == "__main__":
    main()
