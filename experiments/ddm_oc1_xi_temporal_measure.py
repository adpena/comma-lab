#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DDM-OC1: measure the xi-temporal (ego-motion) rate lever on the real contest video.

The operator reframe (2026-07-27): 1200 frames = ONE static scene x ego-motion xi(t).
This tool MEASURES, on the real ``0.mkv`` at the scorer-relevant 384x512 description
resolution, how much cheaper a homography/ego-motion prediction is than the shipped
V10 description (fixed 1-2-1 blur, dense residual). Every number is measured from real
pixels; nothing is extrapolated. No score claim -- this prices the *description*, it
does not run the scorer. It sizes the #1 UNBUILT lever before any byte-close.

Two prediction scopes are contrasted, both against real per-pixel residual bytes coded
with the exact production codec (brotli-q11 of int16 residual):

  * INTRA-PAIR: predict frame1 from frame0 within each (2k, 2k+1) pair via a fitted
    homography -- the lever expressible in the *existing* per-pair V10 receiver
    (AFFINE6/PREVIOUS_PLANE modes). Attacks the 72.1% residual cost centre.
  * CROSS-FRAME ATLAS: predict frame t from a single keyframe warped by cumulative
    ego-motion over a temporal window -- the capstone lever that also attacks the
    27.85% bootstrap (600 keyframes -> few atlas keyframes). Needs a NEW cross-pair
    receiver grammar; this measures its potential.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "ddm_oc1_xi_temporal_measure.v1"
SCORER_H = 384
SCORER_W = 512
CHANNELS = 3
PLANE_VALUES = SCORER_H * SCORER_W * CHANNELS  # 589,824 -- one description plane
UNCOMPRESSED_BYTES = 37_545_489  # score rate denominator, measured
BOX_STRICT = 154_524
BOX_PLANNING = 200_000


_CODER_QUALITY = 11


def _brotli_len(payload: bytes) -> int:
    import brotli
    return len(brotli.compress(payload, quality=_CODER_QUALITY))


def _decode_frames(video_path: Path, max_frames: int) -> np.ndarray:
    """Decode the contest video to (N, 384, 512, 3) uint8 via a BT.601 YUV420 path.

    Matches the upstream ``yuv420_to_rgb`` colour math for camera-fidelity RGB, then
    area-downsamples to the description resolution the V10 archive actually encodes.
    """
    import av
    import cv2

    frames: list[np.ndarray] = []
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        yuv = frame.reformat(format="yuv420p")
        h, w = yuv.height, yuv.width
        y = np.frombuffer(yuv.planes[0], dtype=np.uint8).reshape(h, yuv.planes[0].line_size)[:, :w]
        u = np.frombuffer(yuv.planes[1], dtype=np.uint8).reshape(h // 2, yuv.planes[1].line_size)[:, : w // 2]
        v = np.frombuffer(yuv.planes[2], dtype=np.uint8).reshape(h // 2, yuv.planes[2].line_size)[:, : w // 2]
        u = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)
        yy = y.astype(np.float32)
        uu = u.astype(np.float32) - 128.0
        vv = v.astype(np.float32) - 128.0
        r = yy + 1.402 * vv
        g = yy - 0.344136 * uu - 0.714136 * vv
        b = yy + 1.772 * uu
        rgb = np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)
        small = cv2.resize(rgb, (SCORER_W, SCORER_H), interpolation=cv2.INTER_AREA)
        frames.append(small)
        if len(frames) >= max_frames:
            break
    container.close()
    return np.stack(frames, axis=0)


def _homography(prev: np.ndarray, cur: np.ndarray, orb: Any, matcher: Any) -> np.ndarray | None:
    """Fit a homography mapping ``prev`` pixel coords into ``cur`` via ORB + RANSAC."""
    import cv2

    g0 = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(k0) < 12 or len(k1) < 12:
        return None
    matches = matcher.knnMatch(d0, d1, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None
    src = np.float32([k0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or mask is None or int(mask.sum()) < 10:
        return None
    return H


def _warp(frame: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warp ``frame`` by homography ``H``; return (warped_uint8, valid_mask_bool)."""
    import cv2

    warped = cv2.warpPerspective(frame, H, (SCORER_W, SCORER_H), flags=cv2.INTER_LINEAR, borderValue=0)
    ones = np.ones((SCORER_H, SCORER_W), dtype=np.uint8)
    cover = cv2.warpPerspective(ones, H, (SCORER_W, SCORER_H), flags=cv2.INTER_NEAREST, borderValue=0)
    return warped, cover.astype(bool)


def _blur121(frame: np.ndarray) -> np.ndarray:
    """The shipped SPATIAL_SMOOTH_121 predictor: separable 1-2-1 blur of the plane."""
    import cv2

    kernel = np.array([1.0, 2.0, 1.0], dtype=np.float32) / 4.0
    out = cv2.sepFilter2D(frame.astype(np.float32), -1, kernel, kernel, borderType=cv2.BORDER_REFLECT)
    return np.clip(out, 0, 255).astype(np.uint8)


def _residual_bytes(target: np.ndarray, predict: np.ndarray) -> int:
    """Code a per-pixel int16 residual exactly as the production codec does (brotli-q11)."""
    resid = target.astype(np.int16) - predict.astype(np.int16)
    return _brotli_len(resid.astype("<i2").tobytes())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--atlas-window", type=int, default=64, help="cross-frame keyframe reach probe")
    parser.add_argument("--atlas-probe-keyframes", type=int, default=6, help="how many keyframe origins to probe")
    parser.add_argument("--coder-quality", type=int, default=11, help="brotli quality; 11=codec-faithful, 9=comparison-fast")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args(list(argv) if argv is not None else None)

    global _CODER_QUALITY
    _CODER_QUALITY = args.coder_quality

    import cv2

    t0 = time.time()
    frames = _decode_frames(args.video, args.max_frames)
    n = frames.shape[0]
    decode_s = time.time() - t0

    orb = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # -- INTRA-PAIR: predict frame1 from frame0 within each (2k, 2k+1) pair --
    n_pairs = n // 2
    intra: dict[str, list[float]] = {"homography": [], "blur121": [], "copy": [], "raw_f1": [], "cover": []}
    homog_fail = 0
    t1 = time.time()
    for k in range(n_pairs):
        f0 = frames[2 * k]
        f1 = frames[2 * k + 1]
        H = _homography(f0, f1, orb, matcher)
        if H is None:
            homog_fail += 1
            pred_h = f0  # fail-closed to copy so the number is never optimistic
            cover = np.ones((SCORER_H, SCORER_W), dtype=bool)
        else:
            pred_h, cover = _warp(f0, H)
            pred_h = np.where(cover[..., None], pred_h, f0)
        intra["homography"].append(_residual_bytes(f1, pred_h))
        intra["blur121"].append(_residual_bytes(f1, _blur121(f0)))
        intra["copy"].append(_residual_bytes(f1, f0))
        intra["raw_f1"].append(_brotli_len(f1.astype(np.uint8).tobytes()))
        intra["cover"].append(float(cover.mean()))
        if args.progress_every and (k + 1) % args.progress_every == 0:
            print(f"intra {k + 1}/{n_pairs} homog/blur={sum(intra['homography']) / max(1, sum(intra['blur121'])):.4f}", flush=True)

    intra_s = time.time() - t1

    # -- CROSS-FRAME ATLAS: predict frame t from a keyframe via cumulative homography --
    t2 = time.time()
    step_h: list[np.ndarray | None] = [None]
    for i in range(1, n):
        step_h.append(_homography(frames[i - 1], frames[i], orb, matcher))
    atlas_curves: list[dict[str, Any]] = []
    origins = np.linspace(0, max(0, n - args.atlas_window - 1), args.atlas_probe_keyframes, dtype=int)
    for origin in sorted({int(o) for o in origins}):
        curve = []
        H_cum = np.eye(3, dtype=np.float64)
        broke = False
        for d in range(1, args.atlas_window + 1):
            idx = origin + d
            if idx >= n or step_h[idx] is None:
                broke = True
                break
            H_cum = step_h[idx] @ H_cum
            pred, cover = _warp(frames[origin], H_cum)
            pred = np.where(cover[..., None], pred, frames[origin])
            curve.append({
                "dist": d,
                "residual_bytes": _residual_bytes(frames[idx], pred),
                "cover": float(cover.mean()),
                "l1": float(np.abs(frames[idx].astype(np.int16) - pred.astype(np.int16)).mean()),
            })
        atlas_curves.append({"origin": int(origin), "reached": len(curve), "broke": broke, "curve": curve})
    atlas_s = time.time() - t2

    def _tot(key: str) -> int:
        return int(sum(intra[key]))

    homography_resid = _tot("homography")
    blur_resid = _tot("blur121")
    copy_resid = _tot("copy")
    raw_f1_bytes = _tot("raw_f1")
    bootstrap_bytes = sum(_brotli_len(frames[2 * k].astype(np.uint8).tobytes()) for k in range(n_pairs))

    def _atlas_mean_at(dist: int) -> float | None:
        vals = [pt["residual_bytes"] for c in atlas_curves for pt in c["curve"] if pt["dist"] == dist]
        return float(np.mean(vals)) if vals else None

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_axis": "[measured-on-real-pixels description bytes; no scorer/distortion claim]",
        "video": str(args.video),
        "coder": f"brotli-q{args.coder_quality}",
        "coder_note": "q11 is codec-faithful; q9 is comparison-fast (~8% larger absolute, ratios faithful)",
        "frames_decoded": n,
        "pairs": n_pairs,
        "resolution": {"h": SCORER_H, "w": SCORER_W, "c": CHANNELS, "plane_values": PLANE_VALUES},
        "wall_seconds": {"decode": decode_s, "intra_pair": intra_s, "atlas": atlas_s},
        "homography_fail_pairs": homog_fail,
        "intra_pair_residual_bytes_total": {
            "homography_predict": homography_resid,
            "blur121_shipped_predictor": blur_resid,
            "previous_plane_copy": copy_resid,
            "raw_frame1_independent": raw_f1_bytes,
        },
        "intra_pair_per_pair_mean": {
            "homography_predict": homography_resid / n_pairs,
            "blur121_shipped_predictor": blur_resid / n_pairs,
            "previous_plane_copy": copy_resid / n_pairs,
        },
        "intra_pair_homography_vs_blur_ratio": homography_resid / blur_resid if blur_resid else None,
        "mean_warp_cover": float(np.mean(intra["cover"])),
        "bootstrap_bytes_independent_frame0": bootstrap_bytes,
        "description_projection": {
            "intra_pair_lever_bytes": bootstrap_bytes + homography_resid,
            "shipped_blur_lever_bytes": bootstrap_bytes + blur_resid,
            "note": "intra-pair lever leaves 600 independent bootstraps untouched; atlas attacks those",
        },
        "atlas_probe": {
            "window": args.atlas_window,
            "mean_residual_bytes_at_dist_1": _atlas_mean_at(1),
            "mean_residual_bytes_at_dist_8": _atlas_mean_at(8),
            "mean_residual_bytes_at_dist_32": _atlas_mean_at(32),
            "curves": atlas_curves,
        },
        "score_rate_context": {
            "shipped_archive_bytes": 291_205_400,
            "box_strict_bytes": BOX_STRICT,
            "box_planning_bytes": BOX_PLANNING,
            "rate_term_per_byte": 25.0 / UNCOMPRESSED_BYTES,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    summary = {
        "out": str(args.out),
        "pairs": n_pairs,
        "homography_resid_total": homography_resid,
        "blur_resid_total": blur_resid,
        "homography_vs_blur": report["intra_pair_homography_vs_blur_ratio"],
        "bootstrap_bytes": bootstrap_bytes,
        "intra_pair_lever_bytes": report["description_projection"]["intra_pair_lever_bytes"],
        "atlas_resid_d1": _atlas_mean_at(1),
        "atlas_resid_d8": _atlas_mean_at(8),
        "atlas_resid_d32": _atlas_mean_at(32),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
