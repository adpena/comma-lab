# SPDX-License-Identifier: MIT
"""Representation-audit $0 probe (Task #83).

Measures the two facts the right-vs-wrong representation table needs that are not
already pinned by prior tasks, on the EXACT frozen scorer + real ``0.mkv``:

  (A) SEG partition byte cost — run the real CPU-torch SegNet on GT frame1, extract
      the scored argmax partition ``L*``, contour-code it losslessly, report the
      description-length bytes + region count + roundtrip-exact.  This is the byte
      cost of STORING the scored partition directly (the "store the quotient" arm)
      vs the implicit cost of reconstructing it from a full RGB render.

  (B) MOTION temporal redundancy — measure frame-to-frame luma RMSE + the entropy of
      the frame0->frame1 delta vs the raw frame (how compressible the temporal
      residual is) on real GT pairs, and whether the pose pair built from
      (frame0, warped-frame0) differs from the (frame0, frame1) pose in d_pose
      (does motion/flow serve the 6 scored pose dims directly?).

Authority: ``[local CPU-torch advisory]`` — exact frozen scorer, GT via
``frame_utils.yuv420_to_rgb`` ONLY (NEVER MPS / rgb24).  NOT the contest 600-sample
harness -> non-promotable.  ``$0`` spend, no GPU, no dispatch.  NO FAKE: every number
is a real measurement (popcount, RMSE, zlib length, exact PoseNet MSE), not a
constant.  Run:  ``.venv/bin/python tools/representation_audit_probe.py --pairs 4``
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_UPSTREAM = _REPO / "upstream"
for _p in (_REPO / "src", _UPSTREAM):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _luma(frame_hwc_uint8: np.ndarray) -> np.ndarray:
    """BT.601 luma of an RGB frame (the channel the yuv6/pose path integrates)."""
    f = np.asarray(frame_hwc_uint8, dtype=np.float64)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _delta_entropy_bytes(delta_int: np.ndarray) -> int:
    """zlib-compressed length of an int delta map (a real coded-byte proxy)."""
    clipped = np.clip(np.round(delta_int), -127, 127).astype(np.int8)
    return len(zlib.compress(clipped.tobytes(), level=9))


def boundary_fraction(argmax_hw: np.ndarray) -> float:
    """Fraction of pixels on a 4-neighbour class contour (the O(boundary) seg cost).

    A pixel is "on a boundary" if its label differs from its right OR bottom
    neighbour.  This is the geometric quantity the contour codec captures: interiors
    are free constant-fill, only the boundary carries description length.
    """
    a = np.asarray(argmax_hw)
    if a.ndim != 2:
        raise ValueError(f"argmax must be (H, W); got {a.shape}")
    h_edge = np.zeros_like(a, dtype=bool)
    v_edge = np.zeros_like(a, dtype=bool)
    h_edge[:, :-1] = a[:, :-1] != a[:, 1:]
    v_edge[:-1, :] = a[:-1, :] != a[1:, :]
    return float((h_edge | v_edge).mean())


def partition_change_mask_bytes(lstar_t: np.ndarray, lstar_prev: np.ndarray) -> tuple[int, float]:
    """LZMA length of the cross-frame partition CHANGE (the seg analog of optical flow).

    Encodes the per-pixel change-mask (packed bits) + the new labels at changed
    positions.  Returns ``(delta_bytes, changed_pixel_fraction)``.  This is the
    "store the partition once, then per-frame boundary motion" representation arm.

    NO FAKE: this is the real LZMA-coded length of the change payload, not a constant.
    The audit's empirical finding is that this NAIVE change-mask delta is LARGER than
    re-encoding the whole partition (the scattered boundary change-mask is high-entropy
    while a full partition LZMA-compresses its constant interiors away) — so the test
    pins that a fully-changed partition costs MORE than a static one, and a static
    (no-change) partition costs a few bytes.
    """
    import lzma

    a = np.asarray(lstar_t)
    b = np.asarray(lstar_prev)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    changed = a != b
    changed_frac = float(changed.mean())
    mask_bits = np.packbits(changed.astype(np.uint8).ravel(order="C"))
    new_labels = a[changed].astype(np.uint8)
    raw = mask_bits.tobytes() + new_labels.tobytes()
    filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    payload = lzma.compress(raw, format=lzma.FORMAT_RAW, filters=filters)
    return len(payload), changed_frac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    from tac.boundary_math.seg_core import (
        build_and_measure_lstar,
        decode_gt_frame1_pairs,
        load_real_segnet,
    )

    segnet = load_real_segnet("cpu")

    # ---- (A) SEG partition byte cost on real GT frame1 + cross-frame redundancy ----
    from tac.boundary_math.seg_core import segnet_argmax_and_margin

    seg_rows: list[dict] = []
    seg_delta_rows: list[dict] = []
    prev_lstar = None
    for pair_idx, _f0, f1 in decode_gt_frame1_pairs(n_pairs=args.pairs):
        res = build_and_measure_lstar(segnet, f1, pair_idx=pair_idx)
        # recompute L* once for the boundary-fraction + cross-frame-delta numbers
        lstar, _ = segnet_argmax_and_margin(segnet, f1)
        seg_rows.append(
            {
                "pair": res.pair_idx,
                "partition_bytes": res.partition_bytes,
                "n_regions": res.n_regions,
                "roundtrip_exact": res.roundtrip_exact,
                "d_seg_lstar": res.d_seg_lstar,
                "boundary_fraction": boundary_fraction(lstar),
                "shape": list(res.shape),
            }
        )
        if prev_lstar is not None:
            d_bytes, changed_frac = partition_change_mask_bytes(lstar, prev_lstar)
            seg_delta_rows.append(
                {
                    "pair": pair_idx,
                    "independent_bytes": res.partition_bytes,
                    "change_mask_delta_bytes": d_bytes,
                    "changed_pixel_fraction": changed_frac,
                }
            )
        prev_lstar = lstar

    # ---- (B) MOTION temporal redundancy + flow-serves-pose ----
    # Reuse the EXACT frozen-PoseNet helpers from the pose probe (differentiable-yuv6
    # patch + canonical _pose6/_d_pose).  NEVER MPS.
    sys.path.insert(0, str(_REPO / "tools"))
    posenet = None
    _pose6 = None
    try:
        from pose_subspace_spectrum_probe import _load_posenet  # type: ignore
        from pose_subspace_spectrum_probe import _pose6 as _pose6_fn

        from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally  # type: ignore

        patch_upstream_yuv6_globally()
        posenet = _load_posenet()
        _pose6 = _pose6_fn
    except Exception as exc:  # pragma: no cover - degrade gracefully if scorer missing
        print(f"[pose-serve skipped: {exc}]", file=sys.stderr)
        posenet = None

    motion_rows: list[dict] = []
    pairs = list(decode_gt_frame1_pairs(n_pairs=args.pairs))
    for pair_idx, f0, f1 in pairs:
        y0, y1 = _luma(f0), _luma(f1)
        frame_rmse = float(np.sqrt(np.mean((y1 - y0) ** 2)))
        # raw frame entropy vs delta entropy (per-frame independent vs temporal residual)
        raw_bytes = len(zlib.compress(np.round(y1).clip(0, 255).astype(np.uint8).tobytes(), level=9))
        delta_bytes = _delta_entropy_bytes(y1 - y0)
        motion_rows.append(
            {
                "pair": pair_idx,
                "frame0to1_luma_rmse": frame_rmse,
                "raw_frame1_luma_zlib_bytes": raw_bytes,
                "delta_f0_to_f1_luma_zlib_bytes": delta_bytes,
                "delta_vs_raw_ratio": (delta_bytes / raw_bytes) if raw_bytes else None,
            }
        )

    # flow-serves-pose: is the scored pose a function of the MOTION between the two
    # frames?  Measure (i) d_pose of a STATIC pair (frame0,frame0) vs the GT moving pair
    # (frame0,frame1): if pose is the motion signal, the static pair reads a LARGE d_pose
    # (the scored quantity collapses when motion is removed); and (ii) d_pose when frame1
    # is replaced by frame0 SHIFTED by the mean optical translation (a 2-scalar "flow"
    # carrier) -- does a coarse global-translation flow already recover most of the pose?
    pose_serve = None
    if posenet is not None and _pose6 is not None and pairs:
        import torch

        def _chw(frame_hwc):
            return torch.from_numpy(np.asarray(frame_hwc, dtype=np.float64)).permute(2, 0, 1).float()

        def _dpose(p_comp, p_gt):
            return float(((p_comp - p_gt) ** 2).mean())

        def _global_shift(chw, dy, dx):
            # integer roll = a 2-scalar global-translation "flow" carrier (cheapest flow).
            return torch.roll(chw, shifts=(round(dy), round(dx)), dims=(1, 2))

        rows = []
        with torch.no_grad():
            for pair_idx, f0, f1 in pairs:
                f0c, f1c = _chw(f0), _chw(f1)
                p_gt = _pose6(posenet, f0c, f1c)
                # (i) static pair: motion removed
                p_static = _pose6(posenet, f0c, f0c)
                d_static = _dpose(p_static, p_gt)
                # (ii) estimate mean global translation of luma (phase-corr-free coarse flow)
                y0, y1 = _luma(f0), _luma(f1)
                # best integer global shift in a small window by min-RMSE (cheap proxy)
                best = (0, 0, np.inf)
                for dy in range(-6, 7, 2):
                    for dx in range(-6, 7, 2):
                        s = np.roll(np.roll(y0, dy, axis=0), dx, axis=1)
                        e = float(np.mean((s - y1) ** 2))
                        if e < best[2]:
                            best = (dy, dx, e)
                dy, dx, _ = best
                f1_flow = _global_shift(f0c, dy, dx)
                p_flow = _pose6(posenet, f0c, f1_flow)
                d_flow = _dpose(p_flow, p_gt)
                rows.append(
                    {
                        "pair": pair_idx,
                        "d_pose_static_pair": d_static,
                        "best_global_shift_dy_dx": [int(dy), int(dx)],
                        "d_pose_global_translation_flow": d_flow,
                    }
                )
        pose_serve = rows

    summary = {
        "authority": "local-CPU-torch-advisory",
        "promotable": False,
        "score_claim": False,
        "seg_partition": {
            "rows": seg_rows,
            "mean_partition_bytes": float(np.mean([r["partition_bytes"] for r in seg_rows])) if seg_rows else None,
            "mean_n_regions": float(np.mean([r["n_regions"] for r in seg_rows])) if seg_rows else None,
            "all_roundtrip_exact": all(r["roundtrip_exact"] for r in seg_rows) if seg_rows else None,
            "all_d_seg_zero": all(r["d_seg_lstar"] == 0.0 for r in seg_rows) if seg_rows else None,
            "mean_boundary_fraction": float(np.mean([r["boundary_fraction"] for r in seg_rows])) if seg_rows else None,
        },
        "seg_partition_cross_frame_delta": {
            "rows": seg_delta_rows,
            "mean_independent_bytes": float(np.mean([r["independent_bytes"] for r in seg_delta_rows])) if seg_delta_rows else None,
            "mean_change_mask_delta_bytes": float(np.mean([r["change_mask_delta_bytes"] for r in seg_delta_rows])) if seg_delta_rows else None,
            "mean_changed_pixel_fraction": float(np.mean([r["changed_pixel_fraction"] for r in seg_delta_rows])) if seg_delta_rows else None,
            "delta_vs_independent_ratio": (
                float(np.mean([r["change_mask_delta_bytes"] for r in seg_delta_rows]) / np.mean([r["independent_bytes"] for r in seg_delta_rows]))
                if seg_delta_rows else None
            ),
        },
        "motion": {
            "rows": motion_rows,
            "mean_frame0to1_luma_rmse": float(np.mean([r["frame0to1_luma_rmse"] for r in motion_rows])) if motion_rows else None,
            "mean_delta_vs_raw_ratio": float(np.mean([r["delta_vs_raw_ratio"] for r in motion_rows if r["delta_vs_raw_ratio"] is not None])) if motion_rows else None,
        },
        "pose_serve": pose_serve,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
