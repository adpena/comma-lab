#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_r2s LEVER B: stratified plane+parallax PREDICT flip-support, measured in the oc1 currency.

oc1 (merged) measured, on all 600 real pairs, the TASK-lossy residual support each PREDICT leaves:
d_seg with ZERO residual = fraction of the 117.9M SegNet-argmax sites where SegNet(predicted last-
frame) != the cached GT argmax (``lstars``). copy = 0.008642 (0.864%); a single GLOBAL homography is
task-NEGATIVE at 0.018672 (2.16x worse) -- one homography cannot model the multi-depth scene.

This tool tests the STRATIFIED reopener: warp f0 -> f1 with a DIFFERENT planar motion PER Morse-Smale
depth stratum (the argmax partition), so the plane where a homography IS exact (Road/Lane ground) is
warped by a ground-restricted homography, off-plane finite-depth cells (movable + undrivable-below-
horizon) get their own parallax homography, and the static hood + far sky stay copy. The homographies
are fit at ENCODE time from the REAL video via ORB matches RESTRICTED to each stratum's pixels
(unlimited encode compute is legal); this is the GENEROUS upper bound on the xi-parametric ground
predictor (GroundHomographyGeom H(xi) is a 6-DOF restriction of the 8-DOF data-fit homography, so a
NEGATIVE here kills the warp family at FAMILY scope, a POSITIVE motivates the xi-parametric codec).

FALSIFIER (charter): if stratified support >= copy's 0.864%, PREDICT stays closed at FAMILY scope for
warps; LEVER A proceeds with copy. Stratification uses the GT frame-1 argmax (encode-available; the
pixels come from f0, never from lstars, so the flip metric is not injected).

No score claim -- d_seg here is realized through the frozen SegNet on predicted frames but is NOT a
byte-closed archive S. ``[macOS-CPU advisory]``. Chunked (start-pair/max-pairs) for the harness window,
aggregated exactly by ``--aggregate``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "ddm_r2s_stratified_flip_support.v1"
SHIPPED_CODEC_DSEG = 0.00115997
COPY_N600_REFERENCE = 0.00864212883843316  # oc1 aggregate, n600 (self-check target for the copy mode)
GLOBAL_HOMOGRAPHY_N600_REFERENCE = 0.01867174784342448  # oc1 aggregate control
# Canonical comma10k order (CLAUDE.md L80, MEASURED): 0=Road 1=Lane 2=Undrivable(incl sky)
# 3=Movable 4=MyCar/hood.
GROUND_CLASSES = (0, 1)
OFFPLANE_FINITE_CLASS = 3  # movable (vertical, finite depth)
UNDRIVABLE_CLASS = 2  # sky above horizon + distant structure/barriers below
HOOD_CLASS = 4


def _load_oc1():
    """Reuse the oc1 SegNet load + argmax + flip-vs-lstars machinery (do-not-reinvent)."""
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from ddm_oc1_flip_support_measure import (  # type: ignore[import-not-found]
        _dseg_against_lstars,
        _load_segnet,
        _segnet_argmax,
    )

    return _load_segnet, _segnet_argmax, _dseg_against_lstars


def _resample_argmax_native(argmax: np.ndarray, native_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-resample a (384,512) argmax up to native camera (H,W) -- same rule as
    ``stratified_depth_warp.offplane_mask_from_partition``."""
    H, W = int(native_hw[0]), int(native_hw[1])
    a = np.asarray(argmax)
    if a.shape == (H, W):
        return a
    ys = np.linspace(0, a.shape[0] - 1, H).round().astype(np.int64)
    xs = np.linspace(0, a.shape[1] - 1, W).round().astype(np.int64)
    return a[ys][:, xs]


def _stratum_homography(
    prev: np.ndarray, cur: np.ndarray, region: np.ndarray, orb: Any, matcher: Any, *, min_matches: int = 12
) -> np.ndarray | None:
    """RANSAC homography f0->f1 fit from ORB matches whose f0 keypoint lies inside ``region`` (bool
    native-res mask). Restricting the fit to ONE depth stratum is what a single global homography
    cannot do -- it is the whole point of the stratified reopener."""
    import cv2

    g0 = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
    mask_u8 = (region.astype(np.uint8) * 255)
    k0, d0 = orb.detectAndCompute(g0, mask_u8)  # only keypoints inside the stratum on f0
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(k0) < min_matches or len(k1) < min_matches:
        return None
    matches = matcher.knnMatch(d0, d1, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < min_matches:
        return None
    src = np.float32([k0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or inl is None or int(inl.sum()) < 10:
        return None
    return H


def _warp_into(f0: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warp f0 into f1's frame by H (f0->f1); return (warped, cover_bool)."""
    import cv2

    h, w = f0.shape[:2]
    warped = cv2.warpPerspective(f0, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    cover = cv2.warpPerspective(
        np.ones((h, w), np.uint8), H, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
    ).astype(bool)
    return warped, cover


def _predict_stratified(
    f0: np.ndarray, f1: np.ndarray, argmax_native: np.ndarray, mode: str, orb: Any, matcher: Any
) -> tuple[np.ndarray, dict[str, int]]:
    """Build the predicted last-frame under ``mode``. ``argmax_native`` is the GT-f1 partition
    (encode-available) used ONLY to route each target pixel to its stratum's warp; pixel VALUES come
    from f0 (warped or copied), never from lstars. Returns (pred, per-stratum homography-fit status)."""
    h, w = f0.shape[:2]
    ground = np.isin(argmax_native, GROUND_CLASSES)
    rows = np.arange(h)[:, None]
    hz = h // 2
    offplane = (argmax_native == OFFPLANE_FINITE_CLASS) | ((argmax_native == UNDRIVABLE_CLASS) & (rows >= hz))
    upper = (argmax_native == UNDRIVABLE_CLASS) & (rows < hz)

    pred = f0.copy()
    status: dict[str, int] = {}

    def apply(region: np.ndarray, name: str) -> None:
        if not region.any():
            status[name] = -2  # empty stratum (nothing to fit)
            return
        H = _stratum_homography(f0, f1, region, orb, matcher)
        if H is None:
            status[name] = 0  # fit failed -> that stratum stays copy
            return
        warped, cover = _warp_into(f0, H)
        sel = region & cover
        pred[sel] = warped[sel]
        status[name] = int(sel.sum())

    if mode == "strat_ground":
        apply(ground, "ground")
    elif mode == "strat_full":
        apply(ground, "ground")
        apply(offplane, "offplane")
        apply(upper, "upper")
        # hood (4) + any residue stays copy by construction
    else:
        raise ValueError(mode)
    return pred, status


def _per_class_flips(pred_argmax: np.ndarray, lstars: np.ndarray) -> dict[str, Any]:
    """Attribute flips to the GT class at each flipped site + per-class flip rate within that class."""
    flip = pred_argmax != lstars
    total = int(flip.size)
    out: dict[str, Any] = {"flip_sites": int(flip.sum()), "total_sites": total}
    for k in range(5):
        gt_k = lstars == k
        n_k = int(gt_k.sum())
        flips_k = int((flip & gt_k).sum())
        out[f"class{k}_flip_share_of_all"] = flips_k / total if total else 0.0
        out[f"class{k}_flip_rate_within_class"] = (flips_k / n_k) if n_k else 0.0
    return out


def _run_modes(
    gt_f0: np.ndarray, gt_f1: np.ndarray, lstars: np.ndarray, modes: Sequence[str], batch: int,
    segnet_argmax, dseg_against_lstars, net,
) -> dict[str, Any]:
    import cv2

    orb = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    n, H, W = gt_f0.shape[0], gt_f0.shape[1], gt_f0.shape[2]
    total_sites = int(lstars.size)
    results: dict[str, Any] = {}
    for mode in modes:
        t = time.time()
        if mode == "copy":
            preds = gt_f0.copy()
            fit_status = {"note": "copy predictor: pred == f0"}
        else:
            preds = np.empty_like(gt_f1)
            agg_status: dict[str, list[int]] = {}
            for i in range(n):
                am = _resample_argmax_native(lstars[i], (H, W))
                p, st = _predict_stratified(gt_f0[i], gt_f1[i], am, mode, orb, matcher)
                preds[i] = p
                for kk, vv in st.items():
                    agg_status.setdefault(kk, []).append(vv)
            fit_status = {
                kk: {
                    "fits_ok": int(sum(1 for v in vv if v > 0)),
                    "fits_failed": int(sum(1 for v in vv if v == 0)),
                    "empty": int(sum(1 for v in vv if v == -2)),
                    "mean_px_overwritten": float(np.mean([v for v in vv if v > 0]) if any(v > 0 for v in vv) else 0.0),
                }
                for kk, vv in agg_status.items()
            }
        # ONE SegNet forward per mode: argmax -> flip total AND per-class attribution in the same pass.
        flip_total = 0
        per_class_accum = np.zeros((5, 2), dtype=np.int64)  # [flips_k, n_k]
        for start in range(0, n, batch):
            am = segnet_argmax(net, preds[start : start + batch])
            gt = lstars[start : start + batch]
            flip = am != gt
            flip_total += int(flip.sum())
            for k in range(5):
                gt_k = gt == k
                per_class_accum[k, 0] += int((flip & gt_k).sum())
                per_class_accum[k, 1] += int(gt_k.sum())
        pc = {
            f"class{k}": {
                "flip_share_of_all": int(per_class_accum[k, 0]) / total_sites,
                "flip_rate_within_class": (int(per_class_accum[k, 0]) / int(per_class_accum[k, 1]))
                if per_class_accum[k, 1]
                else 0.0,
            }
            for k in range(5)
        }
        results[mode] = {
            "d_seg_zero_residual": flip_total / total_sites,
            "flip_sites": flip_total,
            "total_sites": total_sites,
            "per_class": pc,
            "fit_status": fit_status,
            "wall_seconds": time.time() - t,
        }
        print(
            f"{mode}: d_seg(zero-residual)={flip_total / total_sites:.6f} "
            f"flip_sites={flip_total} wall={results[mode]['wall_seconds']:.1f}s",
            flush=True,
        )
    return results


def _aggregate(chunk_paths: Sequence[Path], out: Path) -> int:
    chunks = [json.loads(p.read_text()) for p in chunk_paths]
    modes = sorted({m for c in chunks for m in c["predict_only_dseg"]})
    agg: dict[str, Any] = {"schema": "ddm_r2s_stratified_flip_support_aggregate.v1", "score_claim": False,
                           "promotion_eligible": False,
                           "evidence_axis": "[macOS-CPU advisory - frozen SegNet on predicted frames; NOT a byte-closed S]",
                           "chunks": [str(p) for p in chunk_paths]}
    total_pairs = sum(int(c["pairs"]) for c in chunks)
    agg["pairs"] = total_pairs
    per_mode: dict[str, Any] = {}
    for mode in modes:
        flips = sum(int(c["predict_only_dseg"][mode]["flip_sites"]) for c in chunks)
        sites = sum(int(c["predict_only_dseg"][mode]["total_sites"]) for c in chunks)
        pc = {}
        for k in range(5):
            fk = sum(int(round(c["predict_only_dseg"][mode]["per_class"][f"class{k}"]["flip_share_of_all"]
                                * c["predict_only_dseg"][mode]["total_sites"])) for c in chunks)
            pc[f"class{k}_flip_share_of_all"] = fk / sites if sites else 0.0
        per_mode[mode] = {
            "d_seg_zero_residual": flips / sites if sites else 0.0,
            "flip_sites": flips,
            "total_sites": sites,
            "per_class_share_of_all": pc,
        }
    agg["predict_only_dseg"] = per_mode
    agg["copy_n600_reference"] = COPY_N600_REFERENCE
    agg["global_homography_n600_reference"] = GLOBAL_HOMOGRAPHY_N600_REFERENCE
    agg["shipped_codec_dseg_with_full_residual"] = SHIPPED_CODEC_DSEG
    if "copy" in per_mode:
        copy_supp = per_mode["copy"]["d_seg_zero_residual"]
    else:
        copy_supp = COPY_N600_REFERENCE
    verdict = {}
    for mode in modes:
        if mode == "copy":
            continue
        supp = per_mode[mode]["d_seg_zero_residual"]
        verdict[mode] = {
            "support_fraction": supp,
            "vs_copy_ratio": supp / copy_supp if copy_supp else float("inf"),
            "beats_copy": bool(supp < copy_supp),
            "falsifier_predict_stays_closed": bool(supp >= copy_supp),
        }
    agg["verdict"] = verdict
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, indent=2, sort_keys=True))
    print(json.dumps({"out": str(out), "copy_support": copy_supp, "verdict": verdict}, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--gt-cache", type=Path)
    parser.add_argument("--segnet-weights", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=600)
    parser.add_argument("--modes", default="copy,strat_ground,strat_full")
    parser.add_argument("--aggregate", nargs="+", type=Path, default=None,
                        help="aggregate the listed chunk JSONs into --out (no SegNet run)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.aggregate:
        if args.out is None:
            raise SystemExit("--aggregate requires --out")
        return _aggregate(args.aggregate, args.out)

    for req in ("repo_root", "gt_cache", "segnet_weights", "out"):
        if getattr(args, req) is None:
            raise SystemExit(f"--{req.replace('_','-')} required for a measurement run")

    load_segnet, segnet_argmax, dseg_against_lstars = _load_oc1()
    cache = np.load(str(args.gt_cache))
    lo, hi = args.start_pair, args.start_pair + args.max_pairs
    gt_f0 = cache["gt_f0"][lo:hi]
    gt_f1 = cache["gt_f1"][lo:hi]
    lstars = cache["lstars"][lo:hi]
    n = gt_f0.shape[0]
    net = load_segnet(args.repo_root, args.segnet_weights)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = _run_modes(gt_f0, gt_f1, lstars, modes, args.batch, segnet_argmax, dseg_against_lstars, net)

    report = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_axis": "[macOS-CPU advisory - frozen SegNet on predicted frames; NOT a byte-closed S]",
        "pairs": n,
        "start_pair": args.start_pair,
        "shipped_codec_dseg_with_full_residual": SHIPPED_CODEC_DSEG,
        "copy_n600_reference": COPY_N600_REFERENCE,
        "global_homography_n600_reference": GLOBAL_HOMOGRAPHY_N600_REFERENCE,
        "predict_only_dseg": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"out": str(args.out), "modes": modes,
                      "d_seg": {m: results[m]["d_seg_zero_residual"] for m in results}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
