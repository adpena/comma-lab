#!/usr/bin/env python
"""Frontier seg-repair flip-map builder (#51, the seg-repair pool attack).

THE MISSION: locate the frontier archive's (sha b7106c9b) ACTUAL flipped pixels.
d_seg = 5.5979e-04 = 0.056 score units = 29% of the 0.19199 total = the largest
single score pool. This tool inflates the frontier, renders the receiver comp
frames per pair (byte-faithful to inflate.py), decodes the contest-EXACT GT
(frame_utils.yuv420_to_rgb, NEVER PyAV rgb24 per the R3 GT-decode bug class),
scores N pairs through the EXACT upstream SegNet on local CPU, and records, per
pair: the flipped-pixel set (scorer grid 384x512 where comp argmax != GT argmax),
each flip's rendered top1-top2 margin (distance to flip back = recoverability),
and the GT-class identity at each flip.

AXIS: [macOS-CPU advisory]. This flip-map is a candidate-GENERATION prior. The R3
lesson stands: macOS-CPU per-pixel argmax can drift vs the Linux contest host at
the boundary, so the flip-map is the SEARCH SPACE; the on-host exact replay is the
admission authority. BUT: d_seg is an argmax-FLIP rate, far more robust to host FP
drift than pose~1e-5 (a flip needs the top1-top2 margin to cross zero, not a 1e-10
ordering). The flip-map's gross structure (which pairs carry flips, where) is a
sound prior; the on-host replay ratifies the final ΔS.

NO MPS. $0 local. Output -> SSD tier (VertigoDataTier).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Reuse the R2 contest-exact render + GT decode + scorer library (named reuse per
# the search-first doctrine; it carries the R3 contest-GT fix + render fidelity).
R2_ANALYSIS = (
    Path(__file__).resolve().parent.parent
    / "pr110pp_r2_nonmps_candidate_20260609" / "analysis"
)
sys.path.insert(0, str(R2_ANALYSIS))
import render_and_score_lib as L  # noqa: E402


def _segnet_argmax_and_margin(scorer, comp_chw_pair):
    """Run the EXACT SegNet on a comp pair (2,3,H,W camera-res); return
    (argmax_HW int64, margin_HW float64) at the scorer grid (384x512).

    Reproduces modules.py SegNet.preprocess_input: x[:, -1, ...] then bilinear
    resize camera->(384,512); argmax over 5 classes; margin = top1-top2 logit.
    """
    # comp -> (1,2,H,W,3) for DistortionNet.preprocess_input ('b t h w c -> b t c h w')
    comp_bthwc = L.comp_pair_to_bthwc(comp_chw_pair)  # (1,2,H,W,3)
    with torch.inference_mode():
        seg_in = scorer.net.segnet.preprocess_input(
            comp_bthwc.to(L.DEVICE).float().permute(0, 1, 4, 2, 3).contiguous()
        )  # (1,3,384,512)
        logits = scorer.net.segnet(seg_in)  # (1,5,384,512)
        top2 = torch.topk(logits, k=2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).clamp_min(0.0)[0]
        argmax = logits.argmax(dim=1)[0]
    return (
        argmax.detach().cpu().numpy().astype(np.int64),
        margin.detach().cpu().numpy().astype(np.float64),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", type=int, default=20, help="GT decode + score chunk size")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[flip-map] frontier archive sha (first 16): "
          f"{__import__('hashlib').sha256(L.ARCHIVE.read_bytes()).hexdigest()[:16]}", flush=True)
    renderer = L.FrontierRenderer()
    scorer = L.ExactScorer()
    n_total = renderer.n_pairs
    print(f"[flip-map] frontier n_pairs={n_total}; rendering+scoring "
          f"{args.n_pairs} from start={args.start}", flush=True)
    t_setup = time.time() - t0

    pair_indices = list(range(args.start, min(args.start + args.n_pairs, n_total)))

    per_pair = []
    flip_total = 0
    # margin histogram bins (logit units): tiny-margin = recoverable.
    margin_bins = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 1e9]
    margin_hist = [0] * (len(margin_bins) - 1)
    # recoverable threshold: a flip is "recoverable" if its rendered margin is small
    # (a small correction can flip it back). Use 2.0 logit units as the prior cut.
    RECOVERABLE_MARGIN = 2.0
    recoverable_flips = 0

    for ci in range(0, len(pair_indices), args.batch):
        chunk = pair_indices[ci: ci + args.batch]
        comp = renderer.render_baseline_pairs(chunk)  # dict pi->(2,3,H,W)
        gt = L.decode_gt_pairs(chunk)  # dict pi->(2,H,W,3) uint8
        for pi in chunk:
            comp_pair = comp[pi]  # (2,3,H,W) float rounded
            gt_pair = gt[pi].float()  # (2,H,W,3)
            # comp argmax/margin
            comp_argmax, comp_margin = _segnet_argmax_and_margin(scorer, comp_pair)
            # GT argmax: build a (2,3,H,W) gt comp-shaped tensor (chw) for the same path
            gt_chw = gt_pair.permute(0, 3, 1, 2).contiguous()  # (2,3,H,W)
            gt_argmax, _ = _segnet_argmax_and_margin(scorer, gt_chw)

            flip = comp_argmax != gt_argmax  # (384,512) bool
            n_flip = int(flip.sum())
            flip_total += n_flip
            d_seg_pair = float(flip.mean())

            if n_flip:
                fm = comp_margin[flip]
                # margin histogram over flipped pixels
                idx = np.digitize(fm, margin_bins) - 1
                idx = np.clip(idx, 0, len(margin_hist) - 1)
                for k in idx:
                    margin_hist[int(k)] += 1
                n_recov = int((fm <= RECOVERABLE_MARGIN).sum())
                recoverable_flips += n_recov
                margin_median = float(np.median(fm))
                margin_p10 = float(np.percentile(fm, 10))
                margin_p90 = float(np.percentile(fm, 90))
                # GT-class distribution at flips (which classes we'd repair toward)
                gt_at_flip = gt_argmax[flip]
                cls_counts = {int(c): int((gt_at_flip == c).sum())
                              for c in np.unique(gt_at_flip)}
            else:
                n_recov = 0
                margin_median = margin_p10 = margin_p90 = 0.0
                cls_counts = {}

            # store the flip mask compactly: flat indices of flipped pixels.
            flip_flat = np.where(flip.reshape(-1))[0].astype(np.int32)
            np.save(out_dir / f"flip_idx_pair{pi:04d}.npy", flip_flat)
            if n_flip:
                np.save(out_dir / f"flip_margin_pair{pi:04d}.npy",
                        comp_margin[flip].astype(np.float32))
                np.save(out_dir / f"flip_gtcls_pair{pi:04d}.npy",
                        gt_argmax[flip].astype(np.int8))

            per_pair.append({
                "pair_index": pi,
                "n_flip": n_flip,
                "d_seg_pair": d_seg_pair,
                "n_recoverable_margin_le2": n_recov,
                "margin_median": margin_median,
                "margin_p10": margin_p10,
                "margin_p90": margin_p90,
                "gt_class_counts_at_flips": cls_counts,
            })
        print(f"[flip-map] {ci + len(chunk)}/{len(pair_indices)} pairs "
              f"(flip_total={flip_total}, recoverable={recoverable_flips}, "
              f"elapsed={time.time() - t0:.1f}s)", flush=True)

    n_scored = len(pair_indices)
    d_seg_mean = flip_total / (n_scored * 384 * 512) if n_scored else 0.0
    summary = {
        "schema": "frontier_seg_repair_flip_map.v1",
        "frontier_archive_sha256_16": __import__('hashlib').sha256(
            L.ARCHIVE.read_bytes()).hexdigest()[:16],
        "n_pairs_scored": n_scored,
        "scorer_grid": [384, 512],
        "flip_total": flip_total,
        "d_seg_mean_recomputed": d_seg_mean,
        "recoverable_flips_margin_le2": recoverable_flips,
        "recoverable_fraction": (recoverable_flips / flip_total) if flip_total else 0.0,
        "flips_per_pair_mean": flip_total / n_scored if n_scored else 0.0,
        "margin_bins": margin_bins,
        "margin_hist_over_all_flips": margin_hist,
        "recoverable_margin_cut": RECOVERABLE_MARGIN,
        "setup_seconds": t_setup,
        "total_seconds": time.time() - t0,
        "axis_tag": "[macOS-CPU advisory]",
        "provenance": {
            "score_claim": False,
            "promotion_eligible": False,
            "hardware_substrate": "local_macos_cpu",
            "gt_decode": "frame_utils.yuv420_to_rgb (contest-exact)",
        },
    }
    (out_dir / "flip_map_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "flip_map_per_pair.json").write_text(json.dumps(per_pair, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
