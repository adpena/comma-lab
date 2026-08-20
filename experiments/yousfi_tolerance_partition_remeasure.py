# SPDX-License-Identifier: MIT
"""YOUSFI re-measure: tolerance-exploiting non-neural partition store rate-vs-d_seg.

B-WITNESS measured ONLY the LOSSLESS corner (d_seg=0, 896 B/frame). The frontier
operates at d_seg=5.6e-4 TOLERANCE. This maps the rate-vs-d_seg curve by
SIMPLIFYING L* (dropping low-margin boundary wiggle up to a d_seg budget) and
coding the simplified partition with (a) the LZMA baseline and (b) a
margin-weighted morphological-simplify variant. Real CPU-torch SegNet, GT decode
via yuv420_to_rgb. [macOS-CPU advisory] NON-PROMOTABLE. NO FAKE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))

from tac.boundary_math.dense_raster_lzma_baseline import partition_description_bytes  # noqa: E402
from tac.boundary_math.seg_core import (  # noqa: E402
    decode_gt_frame1_pairs,
    load_real_segnet,
    segnet_argmax_and_margin,
)

N_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
N_CLASSES = 5

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None


def simplify_partition_by_margin(lstar: np.ndarray, margin: np.ndarray, drop_frac: float) -> np.ndarray:
    """Drop the lowest-margin boundary pixels by reassigning to a strong neighbour.

    Tolerance-exploiting: low-margin boundary wiggle is where SegNet is least sure;
    flipping those pixels costs the fewest 'real' flips per byte saved. We replace
    the lowest drop_frac of BOUNDARY pixels (by margin) with their majority 4-neighbour
    label iterated until stable-ish, smoothing thin jagged contours (cheaper to code).
    """
    if drop_frac <= 0:
        return lstar.copy()
    out = lstar.copy()
    H, W = out.shape
    # boundary pixels: differ from any 4-neighbour
    def boundary_mask(a):
        b = np.zeros_like(a, dtype=bool)
        b[:-1, :] |= a[:-1, :] != a[1:, :]
        b[1:, :] |= a[:-1, :] != a[1:, :]
        b[:, :-1] |= a[:, :-1] != a[:, 1:]
        b[:, 1:] |= a[:, :-1] != a[:, 1:]
        return b
    bmask = boundary_mask(out)
    bidx = np.argwhere(bmask)
    if bidx.size == 0:
        return out
    mvals = margin[bmask]
    order = np.argsort(mvals)  # lowest margin first
    n_drop = int(round(drop_frac * len(order)))
    targets = bidx[order[:n_drop]]
    # reassign each to its most common 4-neighbour label (majority vote)
    for (r, c) in targets:
        cand = []
        if r > 0:
            cand.append(out[r - 1, c])
        if r < H - 1:
            cand.append(out[r + 1, c])
        if c > 0:
            cand.append(out[r, c - 1])
        if c < W - 1:
            cand.append(out[r, c + 1])
        if cand:
            vals, cnts = np.unique(np.array(cand), return_counts=True)
            out[r, c] = int(vals[np.argmax(cnts)])
    return out


def majority_smooth(lstar: np.ndarray, iters: int) -> np.ndarray:
    """Iterated 3x3 majority filter — kills speckle, lowers boundary entropy, raises d_seg."""
    if ndimage is None or iters <= 0:
        return lstar.copy()
    out = lstar.copy()
    for _ in range(iters):
        new = out.copy()
        for cls in range(N_CLASSES):
            m = (out == cls).astype(np.float32)
            s = ndimage.uniform_filter(m, size=3)
            new = np.where(s > 0.5, cls, new) if cls == 0 else new
        # generic majority: pick the class with max neighbourhood support
        stacks = np.stack(
            [ndimage.uniform_filter((out == c).astype(np.float32), size=3) for c in range(N_CLASSES)],
            axis=0,
        )
        out = stacks.argmax(axis=0).astype(out.dtype)
    return out


def d_seg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.count_nonzero(a != b)) / a.size


def rate_from_bytes_per_frame(bpf: float) -> float:
    """25 * (bytes for 600 frames) / 37_545_489."""
    return 25.0 * (bpf * 600.0) / 37_545_489.0


def main():
    seg = load_real_segnet("cpu")
    lstars = []
    margins = []
    for _, _f0, f1 in decode_gt_frame1_pairs(n_pairs=N_PAIRS):
        l, m = segnet_argmax_and_margin(seg, f1)
        lstars.append(l.astype(np.int64))
        margins.append(np.asarray(m, dtype=np.float64))
    n = len(lstars)

    # Operating points: lossless + a sweep of simplification strengths.
    smooth_iters = [0, 1, 2, 3, 4, 6, 8]
    curve = []
    for it in smooth_iters:
        dsegs = []
        bytes_lzma = []
        for l, _m in zip(lstars, margins):
            simp = majority_smooth(l, it)
            dsegs.append(d_seg(simp, l))
            bytes_lzma.append(partition_description_bytes(simp, N_CLASSES))
        curve.append({
            "smooth_iters": it,
            "mean_d_seg_vs_lstar": float(np.mean(dsegs)),
            "mean_bytes_per_frame": float(np.mean(bytes_lzma)),
            "rate_term": rate_from_bytes_per_frame(float(np.mean(bytes_lzma))),
        })

    # Margin-drop variant at a few fractions (independent of smoothing).
    drop_curve = []
    for df in [0.0, 0.1, 0.25, 0.4, 0.6, 0.8]:
        dsegs = []
        bts = []
        for l, m in zip(lstars, margins):
            simp = simplify_partition_by_margin(l, m, df)
            dsegs.append(d_seg(simp, l))
            bts.append(partition_description_bytes(simp, N_CLASSES))
        drop_curve.append({
            "drop_frac": df,
            "mean_d_seg_vs_lstar": float(np.mean(dsegs)),
            "mean_bytes_per_frame": float(np.mean(bts)),
            "rate_term": rate_from_bytes_per_frame(float(np.mean(bts))),
        })

    out = {
        "authority": "macOS-CPU-advisory",
        "n_frames": n,
        "frontier": {"total_score": 0.19110, "rate": 0.118, "seg": 0.056, "d_seg": 5.6e-4,
                     "pose": 0.017, "neural_decoder_rate_share": 0.108},
        "b_witness_lossless_corner": {"d_seg": 0.0, "bytes_per_frame": 895.7,
                                      "rate_seg_alone": rate_from_bytes_per_frame(895.7)},
        "smooth_curve_rate_vs_dseg": curve,
        "margin_drop_curve_rate_vs_dseg": drop_curve,
    }
    outdir = REPO / "experiments" / "results" / "yousfi_tolerance_remeasure_20260611"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "rate_vs_dseg.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
