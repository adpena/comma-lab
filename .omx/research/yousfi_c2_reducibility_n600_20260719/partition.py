#!/usr/bin/env python3
"""Yousfi reachable-vs-floor partition of the c2 witness residual (correct-vehicle).

Levelset-native replacement for the drifted torch_vehicle
tools/measure_dseg_reducibility_gt_margin.py (which fail-closed on c2 because it
builds driver._new_vendored_decoder / expects taper_channels — a different
decoder family). This consumes the levelset-native
witness_per_stage_annulus_attribution.py maps_<ckpt>.npz (witness realized argmax
+ witness-render margin toward GT) crossed against the GT cache (gt argmax +
GT-image SegNet margin).

Two orthogonal margins:
  gt_image_margin  (gt_n600.margins)   -> SegNet confidence on the ORIGINAL frame.
                                          HIGH at a flip = robust GT label the
                                          witness DROPPED = REDUCIBLE (headroom).
                                          near-0 = SegNet self-tie = AT-FLOOR.
  witness_margin   (maps.gt_margin)    -> how close the witness got to the GT
                                          class. small-negative = a CHEAP carrier
                                          nudge flips it; very-negative = needs
                                          capacity.

Reports CURVES over threshold (no single cherry-picked tau), per-class, and the
carrier-reachable ΔS headroom vs the pointer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    maps_p = out_dir / "maps_c2best.npz"
    gt_p = Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    if not maps_p.exists():
        print(f"maps npz not ready yet: {maps_p}", file=sys.stderr)
        return 3

    m = np.load(maps_p, allow_pickle=True)
    wit_argmax = m["argmax"].astype(np.int64)          # (V,H,W) witness realized argmax
    wit_margin = m["gt_margin"].astype(np.float32)     # (V,H,W) witness-render margin toward GT
    g = np.load(gt_p, allow_pickle=True)
    gt_argmax = g["lstars"].astype(np.int64)           # (600,H,W)
    gt_margin = g["margins"].astype(np.float32)        # (600,H,W) GT-image SegNet top1-top2

    V = wit_argmax.shape[0]
    gt_argmax = gt_argmax[:V]
    gt_margin = gt_margin[:V]
    assert wit_argmax.shape == gt_argmax.shape, (wit_argmax.shape, gt_argmax.shape)

    total_px = wit_argmax.size
    flip = wit_argmax != gt_argmax                     # (V,H,W) bool
    n_flip = int(flip.sum())
    d_seg = n_flip / total_px                          # argmax disagreement rate
    S_seg = 100.0 * d_seg

    gm_at_flip = gt_margin[flip]                       # GT-image margin at each flip px
    wm_at_flip = wit_margin[flip]                      # witness-render margin at each flip px
    gt_at_flip = gt_argmax[flip]                       # true class the witness dropped

    def dS_if_fixed(mask_of_flips: np.ndarray) -> float:
        """ΔS (negative = improvement) if this subset of flips were corrected."""
        return -100.0 * int(mask_of_flips.sum()) / total_px

    # ---- Curve 1: AT-FLOOR vs REDUCIBLE over GT-image-margin threshold tau ----
    # flips at gt_image_margin > tau are on ROBUST GT labels (reducible); the rest
    # are near SegNet's own decision boundary (label-noise floor).
    taus = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    reducible_curve = []
    for tau in taus:
        red = gm_at_flip > tau
        reducible_curve.append({
            "gt_margin_tau": tau,
            "reducible_flip_frac": float(red.mean()) if n_flip else 0.0,
            "reducible_flip_count": int(red.sum()),
            "dS_headroom_if_all_fixed": dS_if_fixed(red),
        })

    # ---- Curve 2: cheap-carrier band over witness-render-margin ----
    # flips at -band < witness_margin < 0 are one small nudge from correct.
    bands = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    cheap_curve = []
    for b in bands:
        cheap = (wm_at_flip > -b) & (wm_at_flip < 0.0)
        cheap_curve.append({
            "witness_margin_band": b,
            "cheap_reachable_frac": float(cheap.mean()) if n_flip else 0.0,
            "cheap_reachable_count": int(cheap.sum()),
            "dS_if_all_cheap_fixed": dS_if_fixed(cheap),
        })

    # ---- Per-class flip mass + per-class reducible@tau=0.5 ----
    per_class = {}
    for c, name in CLASS_NAMES.items():
        sel = gt_at_flip == c
        cnt = int(sel.sum())
        red_c = int(((gm_at_flip > 0.5) & sel).sum())
        per_class[name] = {
            "flip_count": cnt,
            "flip_mass_frac": (cnt / n_flip) if n_flip else 0.0,
            "reducible_at_tau0p5_count": red_c,
            "reducible_at_tau0p5_frac_of_class": (red_c / cnt) if cnt else 0.0,
        }

    # ---- GT-image-margin distribution at flips (percentiles) ----
    pct = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    gm_pcts = {str(p): float(np.percentile(gm_at_flip, p)) for p in pct} if n_flip else {}
    wm_pcts = {str(p): float(np.percentile(wm_at_flip, p)) for p in pct} if n_flip else {}

    # ---- Route each partition to the hybrid algebraic FORM that solves it ----
    # Frame: MDL inverse-solve (campaign_is_a_constrained_mdl_inverse_solve_20260718).
    # A flip = a pixel whose render-feature is OUTSIDE the correct winner-cell cone
    # {f:(W_c-W_c')f > 0}. This is FEASIBILITY (land in the cone with margin slack),
    # NOT fidelity. GT-image margin = the target cone WIDTH; witness-render margin =
    # how far outside the cone we currently sit.
    form_routing = {
        "cheap_carrier_head_QP": (
            "flips with witness-render margin in (-band,0): already just outside a "
            "WIDE cone -> the exact rank-4 head QP / a 1-nudge carrier lands them "
            "(tropical/max-plus is EXACT here, affine head). Cheapest bytes."
        ),
        "capacity_exact_lattice_solve_plus_repair": (
            "flips at healthy GT-image margin BUT very-negative witness margin: the "
            "witness DROPPED a robust feature (target cone wide, we're far out). "
            "NOT descent-dominated -- within the active linear region scorer.A is "
            "AFFINE, so the winner-cell preimage is an exact polytope and the "
            "uint8-lattice point is an EXACT bounded Diophantine solve per 2x2 RGB "
            "block (factor-2 #532: n6 measured 3,538,944 blocks FEASIBLE_EXACT, all "
            "520 clip-round Seg failures recovered). Only the curved-SiLU/region "
            "coupling (quotient T, #531) needs the Cole-Hopf-initialized short "
            "full-loss repair (alternative_forms_conv_wall_20260718). NOT a from-scratch crawl."
        ),
        "at_floor_skip": (
            "flips at near-zero GT-image margin: SegNet self-tie, the cone is a "
            "SLIVER. Rate-dominated / label-noise -> DO NOT spend bytes "
            "(opportunity_pools: 'unreachable' = rate-dominated)."
        ),
    }
    result = {
        "vehicle": "levelset_witness_c2 (correct-vehicle; NOT torch_vehicle reducibility tool)",
        "n_pairs": V,
        "d_seg": d_seg,
        "S_seg_contribution": S_seg,
        "n_flip_px": n_flip,
        "total_px": total_px,
        "reducible_over_gt_margin_tau": reducible_curve,
        "cheap_carrier_over_witness_margin_band": cheap_curve,
        "per_class_flip_mass": per_class,
        "gt_image_margin_pctiles_at_flip": gm_pcts,
        "witness_render_margin_pctiles_at_flip": wm_pcts,
        "form_routing_hybrid_formulations": form_routing,
        "NON_ADDITIVE_POOLS_CAVEAT": (
            "per_class_flip_mass is POOL-structured (Road-Lane = pool A competes). "
            "Same-pool levers COMPETE -> do NOT sum the per-pool dS headrooms as if "
            "additive (opportunity_pools_non_additive_rate_distortion_reachable). The "
            "3-axis KKT waterfill (#536) is the correct joint, NOT a sum."
        ),
        "note": (
            "REDUCIBLE = flip at healthy GT-image margin (wide target cone; a "
            "better decoder/carrier lands it). AT-FLOOR = flip near SegNet self-tie "
            "(sliver cone; rate-dominated). CHEAP = witness-render margin in "
            "(-band,0), one head-QP nudge from feasible. dS headroom is vs the seg "
            "term; pointer is 0.19108 [contest-CPU]. Feeds #536 (KKT) + #543 (receiver)."
        ),
    }
    out_json = out_dir / "reducibility_partition.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
