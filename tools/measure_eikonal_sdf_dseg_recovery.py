#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Eikonal-SDF / control-point FULL-5-class partition d_seg recovery from a tiny
deterministic descriptor + a FREE generator.  $0 / CPU / numpy.

[macOS advisory / research-signal] — validates the witness RATE-HALF generator path.
NOT a contest score; pointer claim FORBIDDEN. score_claim=false, promotable=false.

QUESTION (lit-hunt H3): how much d_seg does a ZERO-LEARNED-BYTE deterministic
generator recover from a TINY boundary descriptor, and what residual is irreducible?
This sizes the witness rate-half before any GPU spend.

ARMS (each emits (d_seg, bytes/frame) -> the RD curve):
  A  lossless ceiling     LZMA-over-labels (contour_codec); d_seg=0 EXACT; boundary entropy.
  B  coarse-seed eikonal  block-majority seed grid + free NN/eikonal-Voronoi argmax recon.
                          (NN-upsample == euclidean-distance argmax from grid seeds; verified.)
  C  control points       per-class cv2 contour + approxPolyDP control points + free SDF-argmax.
  D  structured lane       FEED-dm IPM-polynomial lane-SDF re-verify (lane class only).

REUSES (no reinvention):
  tac.boundary_math.bitmask_dseg.d_seg_reference          (canonical d_seg AUTHORITY)
  tac.boundary_math.contour_codec.partition_description_bytes (lossless LZMA ceiling)
  tac.boundary_math.lever_b_levelset_generator.signed_distance_fields (eikonal SDF form)
  tac.boundary_math.lane_sdf_component.build_structured_lane_sdf + decompose (FEED-dj/dm)
  tac.contest_score (score-unit interpretation)

PRIOR-ART CROSS-CHECK (existence proofs; this tool ADDS the eikonal-seed + control-point
generators NOT previously curve'd):
  FEED-af (n600) FULL lossless = 255,288 B (425 B/frame temporal, SOTA arith codec), d_seg=0,
    rate 0.17; lossy region-drop sweep DOMINATED.
  FEED-dj (n24)  lossy margin-simplify curve — every lossy point RAISES score.
  FEED-dm (n48)  structured lane SDF continuous-band: d_seg 0.000415, post-R 0.000797 (R-SURVIVES).
  FEED-du (n96)  hood static: d_seg 0.000737, post-R 0.000677, 56 bytes total / 600 frames.
  FEED-ah        realization-through-R multiplies stored-partition d_seg ~170-350x (the
                 SEPARATE wall the TRAINED witness closes; SDFs survive it, palette-paint does not).

NO-FAKE: every d_seg is the canonical argmax-disagreement on the REAL frozen-SegNet L*
(gt_n96.npz). Reconstructions are REAL (scipy EDT / cv2 raster / np block-mode), not stubs.
Byte costs are REAL LZMA / quantized-vertex stream lengths. Advisory tag; no score claim.

Disk hygiene: writes one small JSON to experiments/results/eikonal_sdf_recovery_<utc>/ (gitignored,
rebuildable from this script + the committed gt cache). No bulk artifacts; no /tmp.
"""
from __future__ import annotations

import argparse
import json
import lzma
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "src"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tac.boundary_math.bitmask_dseg import d_seg_reference  # noqa: E402
from tac.boundary_math.contour_codec import _LZMA_FILTERS, partition_description_bytes  # noqa: E402
from tac.contest_score import UNCOMPRESSED_SIZE_BYTES  # noqa: E402

N_CLASSES = 5
H, W = 384, 512
SEG_FRAMES_FULL = 600  # contest: 600 pairs -> 600 seg-scored last-frames
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
DEFAULT_NPZ = str(_REPO / "experiments/results/mlx_fleet_gt_cache/gt_n96.npz")


def lzma_bytes(arr_u8: np.ndarray) -> int:
    raw = np.ascontiguousarray(arr_u8.astype(np.uint8)).tobytes()
    return len(lzma.compress(raw, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS))


# --- ARM B: coarse-seed eikonal/Voronoi (block-majority down + NN up) -------------
def block_majority_downsample(lab: np.ndarray, b: int) -> np.ndarray:
    """Vectorized block-majority (mode) downsample via one-hot count + argmax."""
    h, w = lab.shape
    gh, gw = (h + b - 1) // b, (w + b - 1) // b
    ph, pw = gh * b - h, gw * b - w
    if ph or pw:
        lab = np.pad(lab, ((0, ph), (0, pw)), mode="edge")
    onehot = (lab[..., None] == np.arange(N_CLASSES)).astype(np.int32)
    onehot = onehot.reshape(gh, b, gw, b, N_CLASSES)
    return onehot.sum(axis=(1, 3)).argmax(axis=-1).astype(np.uint8)


def nn_upsample(coarse: np.ndarray, b: int, h: int, w: int) -> np.ndarray:
    return np.repeat(np.repeat(coarse, b, axis=0), b, axis=1)[:h, :w]


def eikonal_argmax_from_seeds(coarse: np.ndarray, b: int, h: int, w: int) -> np.ndarray:
    """True per-class EDT-argmax reconstruction from block-center seeds — the literal
    'grow SDF from seeds via fast-marching, argmax->partition'. Equals nn_upsample for
    a regular grid (verified once)."""
    from scipy import ndimage

    gh, gw = coarse.shape
    seed = np.full((h, w), 255, np.uint8)
    for i in range(gh):
        for j in range(gw):
            cy = min(int(i * b + b // 2), h - 1)
            cx = min(int(j * b + b // 2), w - 1)
            seed[cy, cx] = coarse[i, j]
    _, (iy, ix) = ndimage.distance_transform_edt(seed == 255, return_indices=True)
    return seed[iy, ix].astype(np.int64)


# --- ARM C: control points (cv2 contour + approxPolyDP) + SDF-argmax -------------
def control_point_descriptor(lab: np.ndarray, eps: float, min_area: int = 8):
    import cv2

    polys = {c: [] for c in range(N_CLASSES)}
    nverts = 0
    for c in range(N_CLASSES):
        mask = (lab == c).astype(np.uint8)
        if mask.sum() == 0:
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < min_area:
                m = cnt.reshape(-1, 2).mean(axis=0).round().astype(np.int32)
                polys[c].append(m.reshape(1, 2))
                nverts += 1
                continue
            approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
            polys[c].append(approx.astype(np.int32))
            nverts += approx.shape[0]
    return polys, nverts


def rasterize_control_points_sdf_argmax(polys, h: int, w: int) -> np.ndarray:
    """FREE generator: rasterize each class's simplified polygons -> per-class mask ->
    signed distance -> argmax (the SDF argmax resolves gaps/overlaps)."""
    import cv2
    from scipy import ndimage

    sdf = np.full((h, w, N_CLASSES), -1e6, np.float32)
    for c in range(N_CLASSES):
        m = np.zeros((h, w), np.uint8)
        for poly in polys[c]:
            if poly.shape[0] >= 3:
                cv2.fillPoly(m, [poly.reshape(-1, 1, 2)], 1)
            else:
                for (x, y) in poly:
                    if 0 <= y < h and 0 <= x < w:
                        m[y, x] = 1
        mb = m.astype(bool)
        if mb.all():
            sdf[..., c] = float(max(h, w))
        elif not mb.any():
            sdf[..., c] = -float(max(h, w))
        else:
            sdf[..., c] = (ndimage.distance_transform_edt(mb)
                           - ndimage.distance_transform_edt(~mb)).astype(np.float32)
    return sdf.argmax(axis=-1).astype(np.int64)


def control_point_bytes(polys) -> int:
    """Honest reversible byte cost: per-class label + delta-coded int16 vertices, LZMA'd."""
    stream = []
    for c in range(N_CLASSES):
        for poly in polys[c]:
            stream.append(np.array([c, poly.shape[0]], np.int16))
            v = poly.astype(np.int16)
            d = v.copy()
            d[1:] = v[1:] - v[:-1]
            stream.append(d.reshape(-1))
    if not stream:
        return 0
    flat = np.concatenate([s.reshape(-1) for s in stream]).astype(np.int16)
    return len(lzma.compress(flat.tobytes(), format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS))


# --- score-unit interpretation ---------------------------------------------------
def score_contrib(d_seg: float, bytes_per_frame: float) -> dict:
    total_bytes = bytes_per_frame * SEG_FRAMES_FULL
    seg_term = 100.0 * d_seg
    rate_term = 25.0 * total_bytes / UNCOMPRESSED_SIZE_BYTES
    return {"seg_term": seg_term, "rate_term_seghalf": rate_term,
            "seghalf_score": seg_term + rate_term, "total_bytes_x600": total_bytes}


def per_class_residual(pred: np.ndarray, gt: np.ndarray) -> dict:
    n = float(gt.size)
    out = {}
    for c in range(N_CLASSES):
        is_c = gt == c
        fn = float(np.sum(is_c & (pred != c)) / n)
        fp = float(np.sum((~is_c) & (pred == c)) / n)
        out[CLASS_NAMES[c]] = {"area_frac": float(is_c.mean()), "fn": fn, "fp": fp,
                               "attributable": fn + fp}
    return out


BREAK_EVEN_DSEG_PER_BPF = 25.0 * SEG_FRAMES_FULL / (100.0 * UNCOMPRESSED_SIZE_BYTES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--frames-heavy", type=int, default=48, help="subsample for ARM C/D")
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    d = np.load(args.npz)
    lstars = d["lstars"][: args.frames].astype(np.int64)
    nfr = lstars.shape[0]
    arms = set(args.arms.split(","))
    results = {"provenance": {"npz": args.npz, "frames": nfr, "shape": [H, W],
                              "class_names": CLASS_NAMES, "seg_frames_full": SEG_FRAMES_FULL,
                              "uncompressed_bytes": UNCOMPRESSED_SIZE_BYTES,
                              "break_even_dseg_per_byte_per_frame": BREAK_EVEN_DSEG_PER_BPF,
                              "tag": "[macOS advisory / research-signal] score_claim=false promotable=false"},
               "arms": {}}

    if "A" in arms:
        per_frame = [partition_description_bytes(lstars[i]) for i in range(nfr)]
        seq_indep = sum(per_frame)
        seq_joint = len(lzma.compress(lstars.astype(np.uint8).tobytes(),
                                      format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS))
        results["arms"]["A_lossless_ceiling"] = {
            "desc": "LZMA lossless full-partition (d_seg=0 EXACT); boundary entropy. "
                    "SOTA context-arith codec gets this to ~425 B/frame (FEED-af).",
            "d_seg": 0.0,
            "bytes_per_frame_indep": float(np.mean(per_frame)),
            "bytes_per_frame_temporal": float(seq_joint / nfr),
            "temporal_ratio": float(seq_joint / max(seq_indep, 1)),
            "score_temporal": score_contrib(0.0, seq_joint / nfr),
        }

    if "B" in arms:
        c0 = block_majority_downsample(lstars[0], 8)
        eik_equiv = float(np.mean(nn_upsample(c0, 8, H, W) != eikonal_argmax_from_seeds(c0, 8, H, W)))
        curve = []
        for b in [2, 3, 4, 6, 8, 12, 16, 24, 32]:
            coarse_list, dsegs, bpf = [], [], []
            for i in range(nfr):
                coarse = block_majority_downsample(lstars[i], b)
                coarse_list.append(coarse)
                dsegs.append(d_seg_reference(nn_upsample(coarse, b, H, W), lstars[i]))
                bpf.append(lzma_bytes(coarse))
            shp = coarse_list[0].shape
            stack = np.stack(coarse_list).astype(np.uint8)
            joint = len(lzma.compress(stack.tobytes(), format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS))
            dmean = float(np.mean(dsegs))
            curve.append({"block": b, "coarse_grid": list(shp), "d_seg": dmean,
                          "bytes_per_frame_indep": float(np.mean(bpf)),
                          "bytes_per_frame_temporal": float(joint / nfr),
                          "score_temporal": score_contrib(dmean, joint / nfr)})
        results["arms"]["B_coarse_seed_eikonal"] = {
            "desc": "block-majority seed grid + free NN/eikonal-Voronoi argmax recon",
            "nn_vs_eikonal_disagree_frac": eik_equiv, "curve": curve}

    if "C" in arms:
        nh = min(args.frames_heavy, nfr)
        idx = np.linspace(0, nfr - 1, nh).round().astype(int)
        curve = []
        for eps in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
            dsegs, bpf, nverts = [], [], []
            for i in idx:
                polys, nv = control_point_descriptor(lstars[i], eps)
                dsegs.append(d_seg_reference(rasterize_control_points_sdf_argmax(polys, H, W), lstars[i]))
                bpf.append(control_point_bytes(polys))
                nverts.append(nv)
            dmean = float(np.mean(dsegs))
            curve.append({"approx_eps": eps, "d_seg": dmean,
                          "bytes_per_frame_indep": float(np.mean(bpf)),
                          "mean_vertices": float(np.mean(nverts)),
                          "score_indep": score_contrib(dmean, np.mean(bpf))})
        results["arms"]["C_control_points"] = {
            "desc": "per-class cv2 contour + approxPolyDP control points + free SDF-argmax recon",
            "frames_used": int(nh), "curve": curve}

    if "D" in arms:
        from tac.boundary_math.lane_sdf_component import (build_structured_lane_sdf,
                                                          decompose_argmax_disagreement)
        nh = min(args.frames_heavy, nfr)
        idx = np.linspace(0, nfr - 1, nh).round().astype(int)
        decs = []
        for i in idx:
            lstar = lstars[i]
            phi_lane, meta = build_structured_lane_sdf(lstar)
            pred = lstar.copy()
            pred[pred == 1] = 0
            pred[phi_lane > 0] = 1
            decs.append((decompose_argmax_disagreement(pred, lstar), meta))
        results["arms"]["D_structured_lane"] = {
            "desc": "FEED-dm structured lane SDF (IPM polynomial manifold) hard-band re-verify; "
                    "cite FEED-dm continuous-band optimal-form d_seg 0.000415 post-R 0.000797",
            "frames_used": int(nh),
            "lane_fn": float(np.mean([d.lane_fn for d, _ in decs])),
            "lane_fp_from_road": float(np.mean([d.lane_fp_from_road for d, _ in decs])),
            "lane_attributable_dseg": float(np.mean([d.lane_attributable for d, _ in decs])),
            "mean_floats_per_frame": float(np.mean([m["total_floats"] for _, m in decs]))}

    # per-class residual at a representative generic operating point (coarse block 2)
    if "B" in arms:
        pred = nn_upsample(block_majority_downsample(lstars[0], 2), 2, H, W)
        results["per_class_residual_block2_frame0"] = per_class_residual(pred, lstars[0])

    results["elapsed_sec"] = time.time() - t0
    out = args.out or str(_REPO / "experiments/results/eikonal_sdf_recovery_run/results.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWROTE {out}  ({results['elapsed_sec']:.1f}s)  break_even_dseg/bpf={BREAK_EVEN_DSEG_PER_BPF:.3e}")


if __name__ == "__main__":
    main()
