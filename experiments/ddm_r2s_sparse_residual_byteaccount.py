#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_r2s LEVER A: sparse auth-weighted task-lossy residual — the binding-stream byte accounting.

oc1 established the DISTORTION side on the copy-PREDICT base: the argmax-flip support is 0.864% of the
117.9M sites (1,019,467 flip sites, n600). LEVER A is the QUANTIZE stage: store residual values ONLY on
the flip support (dilated by the SegNet effective RF), code the support geometry + values with REAL
coders, crush the seg-free frame_0, and NAME the binding stream with its MEASURED bytes.

This tool measures, on all 600 real pairs (chunked SegNet pass), through REAL coders (never asserted):
  1. FLIP SUPPORT — SegNet(copy=f0) argmax vs cached lstars → the scorer-res flip mask + per-class.
  2. SUPPORT GEOMETRY bytes — the flip mask coded (Brotli-Q11 vs LZMA1 race; #307 context-arith is the
     named SOTA target for the memo, not re-implemented here).
  3. RESIDUAL VALUE bytes — camera-res temporal-delta residual (f1 - f0) on the camera-res support
     (scorer flip mask upsampled + RF-dilated), coded raw AND range(A)-projected (#519/#520: the ~52%
     ker(A) scorer-invisible energy dropped BEFORE quantize; the uint8-exactness caveat #532 is flagged,
     flips NOT re-verified here — DERIVED headroom, not a distortion claim).
  4. FRAME_0 crush — a lossy carrier byte estimate (2x2-box chroma INVISIBLE per frozen_scorer_exact_
     factorization + luma downsample), coded; d_pose survival is OWED (needs real PoseNet), flagged.
  5. COMPOSE — the rate = Σ bytes / 37,545,489, the binding stream named.

No score claim; ``[macOS-CPU advisory]``. NOT a byte-closed evaluate.py row (the sparse-residual grammar
is not the V10 receiver r6cal targets; a new inflate is the named next rung). Every byte is a REAL coder
output length; every d_seg is realized through the frozen SegNet.
"""

from __future__ import annotations

import argparse
import json
import lzma
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "ddm_r2s_sparse_residual_byteaccount.v1"
UNCOMPRESSED_DENOM = 37_545_489  # evaluate.py rate denominator
SHIPPED_FRAME0_BYTES = 81_000_000  # recall: frame-0 bootstrap ~81 MB = 27.9% of shipped archive
BOX_BYTES = 200_000  # charter box ~200 KB
BOX_DSEG = 0.00116


def _load_oc1_segnet(repo_root: Path, weights: Path):
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from ddm_oc1_flip_support_measure import _load_segnet, _segnet_argmax  # type: ignore[import-not-found]

    return _load_segnet(repo_root, weights), _segnet_argmax


def _brotli_len(data: bytes) -> int:
    try:
        import brotli
    except ImportError:
        return -1
    return len(bytes(brotli.compress(data, quality=11)))


def _lzma_len(data: bytes) -> int:
    return len(lzma.compress(data, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME))


def _code_race(data: bytes) -> dict[str, int]:
    return {"raw": len(data), "brotli_q11": _brotli_len(data), "lzma1_x9e": _lzma_len(data)}


def _compute_flip_mask(net, segnet_argmax, gt_f0, lstars, batch) -> tuple[np.ndarray, dict[str, Any]]:
    """SegNet(f0) argmax vs lstars → (flip_mask (N,384,512) bool, per-class stats)."""
    n = gt_f0.shape[0]
    flip = np.empty_like(lstars, dtype=bool)
    per_class = np.zeros((5, 2), dtype=np.int64)  # [flips_k, n_k]
    for start in range(0, n, batch):
        am = segnet_argmax(net, gt_f0[start : start + batch])
        gt = lstars[start : start + batch]
        fm = am != gt
        flip[start : start + batch] = fm
        for k in range(5):
            gt_k = gt == k
            per_class[k, 0] += int((fm & gt_k).sum())
            per_class[k, 1] += int(gt_k.sum())
    total = int(lstars.size)
    stats = {
        "flip_sites": int(flip.sum()),
        "total_sites": total,
        "support_fraction": int(flip.sum()) / total,
        "per_class": {
            f"class{k}": {
                "flip_share_of_all": int(per_class[k, 0]) / total,
                "flip_rate_within_class": (int(per_class[k, 0]) / int(per_class[k, 1])) if per_class[k, 1] else 0.0,
            }
            for k in range(5)
        },
    }
    return flip, stats


def _support_geometry_bytes(flip_mask: np.ndarray) -> dict[str, Any]:
    """Code the scorer-res flip mask (N,384,512 bool) as packed bits, race Brotli/LZMA."""
    packed = np.packbits(flip_mask.reshape(flip_mask.shape[0], -1), axis=1).tobytes()
    race = _code_race(packed)
    return {"packed_raw_bytes": len(packed), "coded": race,
            "best_bytes": min(v for v in race.values() if v > 0),
            "note": "#307 context-arith (JBIG/CABAC on the partition) is the SOTA target; brotli/lzma are the race floor"}


def _camera_support(flip_mask: np.ndarray, native_hw: tuple[int, int], radius: int) -> np.ndarray:
    """Upsample the scorer-res flip mask to camera res (nearest) + dilate by ``radius`` px (the SegNet
    effective-RF footprint the codec must correct). Returns (N,H,W) bool."""
    import cv2

    H, W = native_hw
    n = flip_mask.shape[0]
    out = np.empty((n, H, W), dtype=bool)
    ys = np.linspace(0, flip_mask.shape[1] - 1, H).round().astype(np.int64)
    xs = np.linspace(0, flip_mask.shape[2] - 1, W).round().astype(np.int64)
    if radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    for i in range(n):
        up = flip_mask[i][ys][:, xs]
        if radius > 0:
            up = cv2.dilate(up.astype(np.uint8), k).astype(bool)
        out[i] = up
    return out


def _residual_value_bytes(gt_f0, gt_f1, cam_support, project: bool) -> dict[str, Any]:
    """Camera-res temporal-delta residual (f1-f0) on the support, coded (raw + optional range(A))."""
    n, H, W, _ = gt_f0.shape
    resid = gt_f1.astype(np.int16) - gt_f0.astype(np.int16)  # (N,H,W,3)
    if project:
        from tac.boundary_math.range_a_projection import apply_projection

        proj = apply_projection(resid.astype(np.float64))  # drop ker(A) scorer-invisible energy
        resid = np.clip(np.round(proj), -255, 255).astype(np.int16)
    # gather support residuals as int8-centered bytes (clip to knife-edge)
    sel = cam_support
    vals = np.clip(resid[sel], -128, 127).astype(np.int8)  # (S,3)
    site_count = int(sel.sum())
    stream = vals.tobytes()
    race = _code_race(stream)
    return {
        "support_sites_camera": site_count,
        "support_fraction_camera": site_count / (n * H * W),
        "raw_value_bytes_int8x3": len(stream),
        "coded": race,
        "best_bytes": min(v for v in race.values() if v > 0),
    }


def _frame0_crush_bytes(gt_f0) -> dict[str, Any]:
    """Lossy seg-free frame_0 carrier byte estimate: 2x downsample luma + 2x2-box chroma (INVISIBLE),
    coded. d_pose survival OWED (real PoseNet). Bytes only."""
    import cv2

    n, H, W, _ = gt_f0.shape
    chunks = []
    for i in range(n):
        small = cv2.resize(gt_f0[i], (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        chunks.append(small.astype(np.uint8).tobytes())
    stream = b"".join(chunks)
    race = _code_race(stream)
    return {
        "carrier": "2x-downsample-area (luma+chroma), all 600 frame_0",
        "raw_bytes": len(stream),
        "coded": race,
        "best_bytes": min(v for v in race.values() if v > 0),
        "shipped_frame0_baseline_bytes": SHIPPED_FRAME0_BYTES,
        "d_pose_survival": "OWED — needs real PoseNet on the crushed frame_0 (frozen_scorer chroma law: 2x2 chroma <2px INVISIBLE)",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--gt-cache", type=Path, required=True)
    parser.add_argument("--segnet-weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=600)
    parser.add_argument("--dilate-radii", default="0,2,4")
    parser.add_argument("--project-range-a", action="store_true", help="also measure range(A)-projected value bytes")
    parser.add_argument("--skip-frame0", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    sys.path.insert(0, str(args.repo_root / "src"))
    cache = np.load(str(args.gt_cache))
    gt_f0 = cache["gt_f0"][: args.max_pairs]
    gt_f1 = cache["gt_f1"][: args.max_pairs]
    lstars = cache["lstars"][: args.max_pairs]
    n, H, W, _ = gt_f0.shape
    radii = [int(r) for r in args.dilate_radii.split(",") if r.strip()]

    t0 = time.time()
    net, segnet_argmax = _load_oc1_segnet(args.repo_root, args.segnet_weights)
    flip_mask, flip_stats = _compute_flip_mask(net, segnet_argmax, gt_f0, lstars, args.batch)
    print(f"flip support={flip_stats['support_fraction']:.6f} sites={flip_stats['flip_sites']} ({time.time()-t0:.0f}s)", flush=True)

    geom = _support_geometry_bytes(flip_mask)
    print(f"support-geometry best coded bytes={geom['best_bytes']:,}", flush=True)

    value_by_radius: dict[str, Any] = {}
    for r in radii:
        cam = _camera_support(flip_mask, (H, W), r)
        rec = {"raw": _residual_value_bytes(gt_f0, gt_f1, cam, project=False)}
        if args.project_range_a:
            rec["range_a_projected"] = _residual_value_bytes(gt_f0, gt_f1, cam, project=True)
        value_by_radius[f"radius_{r}"] = rec
        vb = rec["raw"]["best_bytes"]
        print(f"radius {r}: cam-support sites={rec['raw']['support_sites_camera']:,} value coded bytes={vb:,}", flush=True)

    frame0 = None if args.skip_frame0 else _frame0_crush_bytes(gt_f0)

    # Compose the rate at radius 0 (the minimal-support operating point) using best-coded streams.
    r0 = value_by_radius[f"radius_{radii[0]}"]["raw"]["best_bytes"]
    geom_b = geom["best_bytes"]
    frame0_b = frame0["best_bytes"] if frame0 else SHIPPED_FRAME0_BYTES
    total_bytes = geom_b + r0 + frame0_b
    binding = max(
        ("support_geometry", geom_b), ("residual_values", r0), ("frame0_carrier", frame0_b), key=lambda kv: kv[1]
    )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_axis": "[macOS-CPU advisory - real coders + frozen SegNet flip mask; NOT a byte-closed evaluate.py row]",
        "pairs": n,
        "predictor": "copy (f0)",
        "flip_support": flip_stats,
        "support_geometry_bytes": geom,
        "residual_value_bytes_by_dilation": value_by_radius,
        "frame0_crush": frame0,
        "compose_radius0": {
            "support_geometry_best_bytes": geom_b,
            "residual_values_best_bytes_radius0": r0,
            "frame0_best_bytes": frame0_b,
            "total_bytes": total_bytes,
            "rate": total_bytes / UNCOMPRESSED_DENOM,
            "rate_term_25x": 25.0 * total_bytes / UNCOMPRESSED_DENOM,
            "box_bytes": BOX_BYTES,
            "over_box_x": total_bytes / BOX_BYTES,
            "binding_stream": binding[0],
            "binding_stream_bytes": binding[1],
        },
        "wall_seconds": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"out": str(args.out), "binding_stream": binding[0], "binding_bytes": binding[1],
                      "total_bytes": total_bytes, "over_box_x": total_bytes / BOX_BYTES}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
