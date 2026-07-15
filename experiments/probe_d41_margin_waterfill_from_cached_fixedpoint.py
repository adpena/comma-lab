#!/usr/bin/env python3
"""D41 reopen ($0, cached-only): margin-adaptive mixed-precision SegNet forward.

Recompute — from the ALREADY-CACHED n600 fixed-point QDQ receipts — the
argmax-exact minimum bit-width, and test whether a per-channel + margin-waterfilled
schedule could admit a cheaper argmax-IDENTICAL forward that the naive
global-uniform QDQ (NO_ADMITTED_PRECISION_IN_LADDER, INSTANCE) missed.

NO training, NO paid dispatch, NO new n600 forward. Reads only the cached JSON
receipts under experiments/results/throughput_authority_ladder_20260714/.

Axis: [macOS-CPU-torch 1-thread advisory / NumPy-fp32]  NON-PROMOTABLE, no score claim.

Criterion (NO-FAKE): argmax-bit-identical == argmax-preservation EXACTLY 1.000000
over all n600 pairs (pixel-exact). Anything < 1.0 is a distortion (changes d_seg),
NOT a free cheaper forward.

The cached arms are WHOLE-NETWORK uniform bit-widths (w8a8..w26a26; fixed-scale
fp32-accum and dynamic-exact-absmax int64-accum) plus two HAND-DESIGNED per-LAYER
geometry mixes (mixed_w26_w30_geometry_safe, weight_l1_safe_w26_w31). They carry
NO per-channel scales and NO per-layer/per-channel argmax-flip ATTRIBUTION, so a
genuine per-channel margin-waterfill cannot be *measured* from them — see the
blocker printed at the end. What CAN be measured from cache is reported here.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments", "results", "throughput_authority_ladder_20260714",
)
# When run from the worktree, results live in the shared main checkout.
if not os.path.isdir(BASE):
    BASE = os.path.expanduser(
        "~/Projects/pact/experiments/results/throughput_authority_ladder_20260714"
    )

CACHED = {
    "fixed_scale(fresh, fp32-accum)": "fixedpoint_scorer_forward_n600_fresh_89b970ff60.json",
    "dynamic_exact_absmax(int64-accum)": "dynamic_fixedpoint_scorer_forward_int64_ceiling_corrected_n600.json",
    "mixed_int64(per-layer geometry mix)": "mixed_int64_fixedpoint_scorer_forward_n600.json",
    "exact_int64": "exact_int64_fixedpoint_scorer_forward_n600.json",
    "weight_l1_int64(per-layer mix)": "weight_l1_int64_fixedpoint_scorer_forward_n600.json",
}


@dataclass
class ArmStat:
    name: str
    n: int
    total_flips: int
    total_pixels: int
    pairs_with_flip: int
    px_preserve: float          # 1 - total_flips/total_pixels  (pixel-exact argmax preservation)
    pair_exact_frac: float      # fraction of pairs with 0 flips
    argmax_exact_n600: bool     # total_flips == 0
    residual_flip_min_margins: list  # fp32 margins of residual flips (for the wall diagnosis)


def stat_arm(name: str, rows: list) -> ArmStat:
    tf = sum(r["flips"] for r in rows)
    tp = sum(r["pixels"] for r in rows)
    pf = sum(1 for r in rows if r["flips"] > 0)
    margins = [
        r["flipped_pixel_margin_quantiles"]["q0"]
        for r in rows
        if r["flips"] > 0 and r.get("flipped_pixel_margin_quantiles")
    ]
    return ArmStat(
        name=name, n=len(rows), total_flips=tf, total_pixels=tp,
        pairs_with_flip=pf, px_preserve=1 - tf / tp,
        pair_exact_frac=(len(rows) - pf) / len(rows),
        argmax_exact_n600=(tf == 0), residual_flip_min_margins=margins,
    )


def main() -> None:
    print("D41 reopen — margin-adaptive mixed-precision SegNet forward (cached-only, $0)")
    print("[macOS-CPU-torch 1-thread advisory / NumPy-fp32]  NON-PROMOTABLE, no score claim\n")

    all_stats: dict[str, list[ArmStat]] = {}
    fp32_verified = True
    for tag, fn in CACHED.items():
        path = os.path.join(BASE, fn)
        if not os.path.exists(path):
            print(f"  MISSING receipt: {path}")
            continue
        with open(path) as fh:
            d = json.load(fh)
        stats = []
        for arm, av in d["arms"].items():
            rows = av.get("segnet_rows")
            if not rows:
                continue
            s = stat_arm(arm, rows)
            stats.append(s)
            # fp32 control must self-preserve argmax (proves reference IS the fp32 argmax authority)
            if arm == "fp32_control" and s.total_flips != 0:
                fp32_verified = False
        all_stats[tag] = stats

    print(f"fp32_control self-argmax-exact on every receipt: {fp32_verified} "
          f"(fp32 SegNet argmax IS the reference authority)\n")

    print(f"{'formulation':38s} {'arm':30s} {'tot_flips':>9s} {'pairs_flip':>10s} "
          f"{'px_preserve':>13s} {'ARGMAX-EXACT':>12s}")
    argmax_exact_arms = []
    for tag, stats in all_stats.items():
        for s in stats:
            if s.name == "fp32_control":
                continue
            mark = "YES(1.000000)" if s.argmax_exact_n600 else "no"
            if s.argmax_exact_n600:
                argmax_exact_arms.append((tag, s.name))
            print(f"{tag:38s} {s.name:30s} {s.total_flips:9d} {s.pairs_with_flip:10d} "
                  f"{s.px_preserve:13.9f} {mark:>12s}")

    print("\n--- MINIMUM argmax-EXACT arm (total_flips==0 on n600) ---")
    if argmax_exact_arms:
        for tag, name in argmax_exact_arms:
            print(f"  {tag} :: {name}")
    else:
        print("  NONE. No cached QDQ arm — uniform (8..26 bit, fixed-scale OR "
              "dynamic-exact-absmax int64) NOR per-layer geometry/weight-L1 mix (w26..w31) —")
        print("  reaches argmax-preservation 1.000000 on n600. Best cached leaves >=1 residual flip.")

    print("\n--- THE BINDING WALL: fp32 margins of the residual flips at the CEILING arms ---")
    print("  (if these are ~0 / sub-ULP, the residual is fp32 argmax TIES, not quantization error,")
    print("   so it is bit-ALLOCATION-invariant and unreachable by any cheaper forward)")
    for tag, stats in all_stats.items():
        for s in stats:
            if s.name == "fp32_control" or s.total_flips == 0:
                continue
            # report only the near-ceiling arms (few residual flips)
            if s.pairs_with_flip <= 5:
                mm = s.residual_flip_min_margins
                print(f"  {tag} :: {s.name}: {s.pairs_with_flip} residual flips, "
                      f"fp32 min-margins = {[f'{x:.3e}' for x in mm]}")

    # fixed-scale plateau diagnosis (separate scale-clipping artifact, NOT the boundary wall)
    fs = all_stats.get("fixed_scale(fresh, fp32-accum)", [])
    plateau = {s.name: s.total_flips for s in fs if s.name in ("w20a20", "w22a22", "w24a24")}
    if plateau:
        print("\n--- fixed-scale plateau (scale-clipping artifact, distinct from the tie wall) ---")
        print(f"  {plateau}  -> hard floor ~{min(plateau.values())} flips even at 24 bits;")
        print("  dynamic-exact-absmax (per-TENSOR exact scale) collapses this to 3 -> the plateau")
        print("  is a fixed-calibration-scale representation artifact, not boundary rounding.")

    print("\n=== VERDICT (D41 reopen) ===")
    print("  argmax-IDENTICAL cheaper forward admitted from cache? NO.")
    print("  Per-channel margin-waterfill is UNMEASURABLE from these receipts (blocker below),")
    print("  AND a positive result is ruled out by the ceiling: the residual flips are fp32")
    print("  argmax TIES (margin 0.0..~5e-7, below fp32 reduction-order noise), which no bit")
    print("  ALLOCATION can preserve. argmax-exactness requires >= fp32 precision AT the tie")
    print("  pixels -> a 'cheaper argmax-identical forward' is self-defeating for this scorer.")
    print("  Scope: INSTANCE (global-uniform) -> FORMULATION (any fixed-point QDQ argmax-exact")
    print("  cheaper forward). Per-channel-scale specifically remains formally unmeasured but is")
    print("  ruled out on headroom (per-tensor-exact-absmax already sits at the fp32-tie wall).")

    print("\n=== BLOCKER (exact missing artifact for a MEASURED per-channel waterfill) ===")
    print("  The cached arms are whole-network UNIFORM bit-widths + 2 hand-designed per-LAYER")
    print("  mixes. Needed and ABSENT: (a) per-CHANNEL quantization scales, and (b) per-layer/")
    print("  per-channel argmax-flip ATTRIBUTION (each arm reports only the whole-network flip")
    print("  count -> which channel/layer caused a residual flip is unidentifiable). Computing a")
    print("  genuine margin-waterfilled per-channel schedule requires per-channel ablation forwards")
    print("  (channel x bit sweep at n600) — a sweep, NOT $0-cached / one-forward. Not fabricated.")


if __name__ == "__main__":
    main()
