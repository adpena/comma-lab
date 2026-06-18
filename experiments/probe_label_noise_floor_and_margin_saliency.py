# SPDX-License-Identifier: MIT
"""LABEL-NOISE FLOOR + margin-saliency map — Yousfi #1 decisive $0 measurement (task #141).

Gates the long-train sub-0.15 d_seg thesis. The bc20 basin's d_seg=0.00260
residual flips sit at SegNet GT-frame top-2 margin median ~0.137. Yousfi (the
detector's designer) argues some of those flips are the SegNet's OWN label
ambiguity (comma10k labelers disagree on the lane-paint edge -> 64% road<->lane)
-> UNWINNABLE by the decoder. The winnable d_seg target =
0.00260 - (flips at pixels where the SegNet's OWN GT-frame top-2 margin < tau).
If a large fraction is label-noise, sub-0.15 d_seg (target 0.000322) may be
UNREACHABLE by the decoder -> the FP-shrink RATE lever becomes load-bearing.

MEASURES (real frozen SegNet, real basin parse-back, real GT, CPU):
  1. GT-frame top-2 margin map g_margin(p) (the detector's own confidence on GT).
  2. Current flips (argmax(SegNet(basin recon)) != argmax(SegNet(GT))).
  3. LABEL-NOISE FLOOR: for tau in {0.05, 0.1, 0.137, 0.2, 0.3}, the fraction of
     current flips at pixels where g_margin < tau (detector barely sure on GT).
  4. WINNABLE d_seg = current_d_seg * (1 - ambiguous_flip_fraction) per tau;
     mapped to S = 100*d_seg and re-checked vs the sub-0.15 target d_seg<0.000322.
  5. MARGIN-SALIENCY map |d(margin)/d(input)| via autograd through the SegNet (the
     detector-informed cost = Z8 P18 saliency = the unified-lever asset); its
     magnitude distribution + concentration at the boundary band.

AUTHORITY: [contest-CPU advisory] / [macOS-CPU advisory] -- NON-PROMOTABLE.
G3 proved local CPU ~ contest-CPU. Pointer UNMOVED. The only authoritative
d_seg is upstream/evaluate.py on a byte-closed archive. CPU-only by design
(MPS owns the live train + would 2x-corrupt the SegNet per CLAUDE.md).

NO FAKE: real frozen SegNet (load_frozen_distortion_net(device='cpu')), real
basin (parse-back of best/best_archive.bin -- the contest-visible bytes), real
GT via frame_utils.yuv420_to_rgb. Reuses the PROVEN detector-cost-B plumbing
(no duplication) + the new tac.margin_saliency_map producer helper.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Reuse the proven real-input plumbing (no duplication per CLAUDE.md).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_yousfi_detector_cost_blindspot_b import (
    BASIN_DIR,
    _decode_gt_pairs,
    _decoded_to_camera,
    _load_basin_decoder,
    _measure_pair,
)

# Canonical sub-0.15 arithmetic (feedback_small_basis_rate_headroom_..._20260616).
# S = 100*d_seg + sqrt(10*d_pose) + 25*B/37_545_489.
# bc20 floor = 0.1178 (rate 0.05935 + pose 0.05845) -> sub-0.15 needs
# 100*d_seg < 0.0322 -> d_seg < 0.000322.
SUB015_DSEG_TARGET = 0.000322
S_FLOOR_RATE_PLUS_POSE = 0.1178  # bc20 rate+pose floor (d_seg=0 limit)
DSEG_TO_S = 100.0  # S contribution per unit d_seg
# The flip-margin median anchor cited by Yousfi (the basin's residual flips).
TAU_SWEEP = (0.05, 0.10, 0.137, 0.20, 0.30)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin-dir", default=BASIN_DIR)
    ap.add_argument("--n-pairs", type=int, default=96,
                    help="pairs for the label-noise-floor measurement (real GT)")
    ap.add_argument("--n-saliency-pairs", type=int, default=4,
                    help="pairs for the autograd margin-saliency map characterization")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU threads (be a good citizen; sister CPU probe is running)")
    ap.add_argument(
        "--out-json",
        default=".omx/research/label_noise_floor_and_margin_saliency_20260618.json",
    )
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, int(args.threads)))
    t0 = time.time()

    print(f"[probe] loading basin decoder parse-back from {args.basin_dir}/best/best_archive.bin")
    dec, latents, meta, arch_bytes = _load_basin_decoder(args.basin_dir)
    n = min(args.n_pairs, latents.shape[0])
    print(f"[probe] basin: base_ch={meta['base_channels']} latent_dim={meta['latent_dim']} "
          f"n_pairs_ckpt={latents.shape[0]} arch_bytes={arch_bytes}")

    from tac.score_aware_loop.targets import load_frozen_distortion_net

    print("[probe] loading REAL frozen SegNet (CPU AUTHORITY -- NO MPS)")
    segnet = load_frozen_distortion_net(device="cpu").segnet

    print(f"[probe] decoding {n} GT pairs (yuv420_to_rgb) + measuring real argmax-flips ...")
    gt_pairs = _decode_gt_pairs(n)
    n = min(n, len(gt_pairs))

    flips_list: list[np.ndarray] = []        # (H, W) bool per pair
    gt_margin_list: list[np.ndarray] = []     # (H, W) GT-frame top-2 margin per pair
    for i in range(n):
        f = _measure_pair(dec, latents, segnet, gt_pairs[i], i)
        flips_list.append(f.flip)
        gt_margin_list.append(f.gt_margin)
        if (i + 1) % 16 == 0:
            print(f"  ... {i + 1}/{n} pairs")

    flips = np.stack(flips_list)              # (n, H, W) bool
    gt_margin = np.stack(gt_margin_list)      # (n, H, W) float64
    n_, H, W = flips.shape

    # ── 1-2: the maps + current d_seg ───────────────────────────────────────
    d_seg = float(flips.mean())              # the basin's d_seg on these n pairs (pixel rate)
    total_flips = int(flips.sum())
    flip_margins = gt_margin[flips]          # GT margin AT the flip pixels
    all_margins = gt_margin.ravel()

    margin_stats = {
        "gt_margin_mean_all_pixels": float(all_margins.mean()),
        "gt_margin_median_all_pixels": float(np.median(all_margins)),
        "flip_gt_margin_mean": float(flip_margins.mean()) if total_flips else 0.0,
        "flip_gt_margin_median": float(np.median(flip_margins)) if total_flips else 0.0,
        "flip_gt_margin_q25": float(np.quantile(flip_margins, 0.25)) if total_flips else 0.0,
        "flip_gt_margin_q75": float(np.quantile(flip_margins, 0.75)) if total_flips else 0.0,
    }

    # ── 3: LABEL-NOISE FLOOR sweep ──────────────────────────────────────────
    # ambiguous_flip_fraction(tau) = frac of current flips at pixels where the
    # detector's OWN GT-frame margin < tau (it is barely sure on the ground
    # truth itself -> the flip is the detector's label ambiguity, not the
    # decoder's fault -> UNWINNABLE by RGB).
    floor_curve = []
    for tau in TAU_SWEEP:
        ambiguous = float((flip_margins < tau).mean()) if total_flips else 0.0
        winnable_d_seg = d_seg * (1.0 - ambiguous)
        winnable_S_contrib = DSEG_TO_S * winnable_d_seg
        floor_curve.append({
            "tau": float(tau),
            "ambiguous_flip_fraction": ambiguous,
            "winnable_d_seg": winnable_d_seg,
            "winnable_d_seg_S_contribution": winnable_S_contrib,
            "winnable_d_seg_vs_sub015_target_ratio": (
                winnable_d_seg / SUB015_DSEG_TARGET if SUB015_DSEG_TARGET > 0 else float("inf")
            ),
            "sub015_reachable_by_decoder": bool(winnable_d_seg <= SUB015_DSEG_TARGET),
            "implied_floor_S_rate_pose_plus_winnable_dseg": (
                S_FLOOR_RATE_PLUS_POSE + winnable_S_contrib
            ),
        })

    # The headline anchor: tau = 0.137 (Yousfi's cited flip-margin median).
    headline = next(r for r in floor_curve if abs(r["tau"] - 0.137) < 1e-9)

    # ── 5: MARGIN-SALIENCY map characterization (autograd) ──────────────────
    from tac.margin_saliency_map import (
        compute_margin_saliency_map,
        saliency_boundary_concentration,
    )

    print(f"[probe] computing margin-saliency (autograd d_margin/d_input) on "
          f"{args.n_saliency_pairs} pairs ...")
    sal_global_stats: list[dict] = []
    sal_flip_stats: list[dict] = []
    sal_hist_accum: list[np.ndarray] = []
    for k in range(min(args.n_saliency_pairs, n)):
        with torch.inference_mode():
            native_f1 = dec(latents[k : k + 1])[0, 1]  # (3, 384, 512)
        cam_f1 = _decoded_to_camera(native_f1.unsqueeze(0))[0]  # (3, 874, 1164)

        # Global saliency (sum of all per-pixel margins).
        msal = compute_margin_saliency_map(segnet, cam_f1)
        conc = saliency_boundary_concentration(msal.saliency, msal.margin, boundary_margin=0.5)
        sal_global_stats.append(conc)
        sal_hist_accum.append(msal.saliency.reshape(-1).cpu().numpy())

        # Flip-targeted saliency (margin summed over THIS pair's flip pixels):
        # "what input change repairs the flips". Aligns saliency mass with the
        # actual d_seg-relevant pixels.
        flip_mask = torch.from_numpy(flips_list[k]).bool()
        if flip_mask.any():
            msal_f = compute_margin_saliency_map(segnet, cam_f1, flip_pixel_mask=flip_mask)
            conc_f = saliency_boundary_concentration(
                msal_f.saliency, msal_f.margin, boundary_margin=0.5
            )
            sal_flip_stats.append(conc_f)

    def _avg(dicts: list[dict], key: str) -> float:
        vals = [d[key] for d in dicts if np.isfinite(d[key])]
        return float(np.mean(vals)) if vals else 0.0

    sal_hist = np.concatenate(sal_hist_accum) if sal_hist_accum else np.zeros(1)
    saliency_summary = {
        "n_saliency_pairs": len(sal_global_stats),
        "global_map": {
            "mean_saliency": _avg(sal_global_stats, "saliency_mean"),
            "max_saliency": _avg(sal_global_stats, "saliency_max"),
            "gini_concentration": _avg(sal_global_stats, "saliency_gini"),
            "frac_saliency_mass_in_boundary_lt_0p5": _avg(
                sal_global_stats, "frac_saliency_mass_in_boundary"
            ),
            "frac_pixels_boundary_lt_0p5": _avg(sal_global_stats, "frac_pixels_boundary"),
            "boundary_over_interior_saliency_ratio": _avg(
                sal_global_stats, "boundary_over_interior_ratio"
            ),
            "saliency_percentiles": {
                "p50": float(np.percentile(sal_hist, 50)),
                "p90": float(np.percentile(sal_hist, 90)),
                "p99": float(np.percentile(sal_hist, 99)),
            },
        },
        "flip_targeted_map": {
            "n_pairs_with_flips": len(sal_flip_stats),
            "frac_saliency_mass_in_boundary_lt_0p5": _avg(
                sal_flip_stats, "frac_saliency_mass_in_boundary"
            ),
            "boundary_over_interior_saliency_ratio": _avg(
                sal_flip_stats, "boundary_over_interior_ratio"
            ),
        },
        "interpretation": (
            "boundary_over_interior_ratio >> 1 confirms the detector's own "
            "gradient concentrates at low-margin (boundary) pixels -> the "
            "saliency map aligns with where the flips live -> it is a valid "
            "detector-informed cost for the weighted margin-hinge + FP-shrink "
            "allocation. The COMPLEMENT (low saliency) is the certified rate-shed "
            "band (must also be d_pose-checked before shedding, per Yousfi audit)."
        ),
    }

    # ── verdict ──────────────────────────────────────────────────────────────
    verdict = _verdict(headline, floor_curve, d_seg)

    report = {
        "probe": "label_noise_floor_and_margin_saliency",
        "task": "#141 (Yousfi council check-in decisive $0 measurement)",
        "authority": "[contest-CPU advisory] / [macOS-CPU advisory] -- NON-PROMOTABLE",
        "score_claim": False,
        "promotable": False,
        "pointer_unmoved": True,
        "basin_dir": args.basin_dir,
        "basin_meta": meta,
        "basin_archive_bytes": arch_bytes,
        "n_pairs_measured": n_,
        "decision_grid_hw": [H, W],
        "wall_clock_s": round(time.time() - t0, 2),
        "current_d_seg_pixelrate_advisory": d_seg,
        "current_d_seg_S_contribution": DSEG_TO_S * d_seg,
        "total_flip_pixels": total_flips,
        "sub015_dseg_target": SUB015_DSEG_TARGET,
        "s_floor_rate_plus_pose": S_FLOOR_RATE_PLUS_POSE,
        "gt_margin_stats": margin_stats,
        "label_noise_floor_curve": floor_curve,
        "headline_tau_0p137": headline,
        "margin_saliency": saliency_summary,
        "verdict": verdict,
    }

    print("\n" + "=" * 78)
    print("LABEL-NOISE FLOOR + MARGIN-SALIENCY  [contest-CPU advisory] NON-PROMOTABLE")
    print("=" * 78)
    print(json.dumps(report, indent=2))

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[wrote] {out}", file=sys.stderr)
    print(f"\nVERDICT: {verdict['headline']}", file=sys.stderr)
    return 0


def _verdict(headline: dict, floor_curve: list[dict], d_seg: float) -> dict:
    """Plain-language verdict: is sub-0.15 d_seg reachable, or is there a
    label-noise floor above the 0.000322 target?"""
    amb = headline["ambiguous_flip_fraction"]
    winnable = headline["winnable_d_seg"]
    reachable = headline["sub015_reachable_by_decoder"]
    ratio = headline["winnable_d_seg_vs_sub015_target_ratio"]

    if amb >= 0.5:
        floor_class = "HIGH"
        msg = (
            f"HIGH label-noise floor: at tau=0.137, {amb:.1%} of the basin's "
            f"residual flips are at pixels where the SegNet is barely sure on "
            f"the GROUND TRUTH itself -- UNWINNABLE by the decoder. "
        )
    elif amb >= 0.2:
        floor_class = "MODERATE"
        msg = (
            f"MODERATE label-noise floor: {amb:.1%} of flips are detector-own-"
            f"ambiguous at tau=0.137. "
        )
    else:
        floor_class = "LOW"
        msg = (
            f"LOW label-noise floor: only {amb:.1%} of flips are detector-own-"
            f"ambiguous at tau=0.137 -- the d_seg residual is mostly the "
            f"decoder's to win. "
        )

    if reachable:
        msg += (
            f"Even the WINNABLE d_seg ({winnable:.6f}) is at/below the sub-0.15 "
            f"target ({SUB015_DSEG_TARGET}); the d_seg half of the path is "
            f"well-posed -- the decoder CAN reach sub-0.15 d_seg in principle "
            f"(it must still close the remaining {winnable/SUB015_DSEG_TARGET:.1f}x "
            f"the WINNABLE flips, but no label-noise wall blocks it)."
        )
        routing = "d_seg_path_well_posed_long_train_thesis_intact"
    else:
        msg += (
            f"The WINNABLE d_seg ({winnable:.6f}) is {ratio:.1f}x ABOVE the "
            f"sub-0.15 target ({SUB015_DSEG_TARGET}). "
        )
        if floor_class == "HIGH":
            msg += (
                "Sub-0.15 d_seg is UNREACHABLE by the decoder -- a label-noise "
                "floor sits above the target. RE-ROUTE the campaign to the "
                "FP-shrink RATE lever (load-bearing)."
            )
            routing = "label_noise_floor_blocks_dseg_reroute_to_rate_lever"
        else:
            msg += (
                "Sub-0.15 d_seg requires winning nearly ALL winnable flips "
                "(aggressive); the FP-shrink RATE lever de-risks the path and "
                "should run in parallel."
            )
            routing = "dseg_path_hard_but_open_run_rate_lever_in_parallel"

    return {
        "headline": msg,
        "label_noise_floor_class": floor_class,
        "ambiguous_flip_fraction_at_tau_0p137": amb,
        "winnable_d_seg_at_tau_0p137": winnable,
        "winnable_d_seg_vs_target_ratio": ratio,
        "sub015_dseg_reachable_by_decoder": reachable,
        "campaign_routing": routing,
    }


if __name__ == "__main__":
    raise SystemExit(main())
