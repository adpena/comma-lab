#!/usr/bin/env python
"""ddm_et1 -- pool the paired eta receipt and do the S-arithmetic against every named consumer.

Reads whatever `ddm_et1_eta_on_priced_band.py` has written so far (it checkpoints after every
pair), so a partial run is harvestable without loss.  Reports the DENOMINATOR on every line per
the m50 vacuity law: a pooled number whose n is not printed is not admissible.

Pooled eta is the ratio of SUMS (total flips removed / total flips described), not the mean of
per-pair ratios -- the two differ whenever pairs carry unequal flip mass, and the ratio-of-sums
is the one that maps to score.  Both are printed.

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE.  score_claim=false.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RATE_PER_BYTE = 25.0 / 37_545_489.0
LIVE_BEST_S = 0.7910689
PR130_FLOOR_S = 0.172141
GAP_S = LIVE_BEST_S - PR130_FLOOR_S
S_PER_FLIP = 100.0 / (600 * 384 * 512)
TOTAL_FLIPS_N600 = 508640
SEG_LEG_S = TOTAL_FLIPS_N600 * S_PER_FLIP
BASE_D_POSE = 0.0025513987495742437
DS_DDPOSE = 5.0 / math.sqrt(10.0 * BASE_D_POSE)

# n600 band facts from Job 0 (et1_band_recalibration.json)
BANDS_N600 = {
    "gp1":     {"bytes": 331824, "capture": 0.83334, "label": "honest (shipped seed, L1 SE)"},
    "sq1":     {"bytes": 369414, "capture": 0.86701, "label": "sq1 SE (shipped seed)"},
    "A3_gp1":  {"bytes": 367873, "capture": 0.97264, "label": "gp1 PRICED A3 (ORACLE seed, L1)"},
}

GP1_F1_THRESHOLD = 0.583          # gp1's published falsifier line, quoted not re-derived


def breakeven_eta(band_key: str) -> float:
    """The eta at which a row is exactly worth doing: rate == eta * gross.

    DERIVED from the band's own measured bytes and capture (never a hardcoded constant --
    the provenance-ladder rule).  Reproduces gp1's F1 = 0.583 when handed gp1's PRICED band,
    which is the internal-consistency check that licenses the other bands' bars.
    """
    b = BANDS_N600[band_key]
    return (b["bytes"] * RATE_PER_BYTE) / (b["capture"] * SEG_LEG_S)


HONEST_BREAKEVEN = breakeven_eta("gp1")


def mean_sd(xs: list) -> tuple:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pooled(rows: list, tag: str) -> dict:
    ok = [r for r in rows if r.get(f"{tag}_eta_net") is not None]
    if not ok:
        return {"n": 0}
    num = sum(r["flips_before"] - r[f"{tag}_flips_after"] for r in ok)
    den = sum(r[f"{tag}_n_described"] for r in ok)
    per = [r[f"{tag}_eta_net"] for r in ok]
    m, sd = mean_sd(per)
    pin = [r.get(f"{tag}_cap_pinned") for r in ok if r.get(f"{tag}_cap_pinned") is not None]
    out = {
        "n": len(ok), "eta_pooled": num / den if den else None,
        "eta_per_pair_mean": m, "eta_per_pair_sd": sd,
        "eta_min": min(per), "eta_max": max(per),
        "flips_removed": num, "flips_described": den,
        "pairs_above_gp1_F1": sum(1 for x in per if x > GP1_F1_THRESHOLD),
        "pairs_above_honest_breakeven": sum(1 for x in per if x > HONEST_BREAKEVEN),
    }
    if pin:
        out["cap_pinned_frac"] = sum(1 for x in pin if x) / len(pin)
        out["cap_pinned_n"] = f"{sum(1 for x in pin if x)}/{len(pin)}"
    dp = [(r["d_pose_before"], r[f"{tag}_d_pose_after"]) for r in ok
          if r.get(f"{tag}_d_pose_after") is not None]
    if dp:
        b = sum(x for x, _ in dp) / len(dp)
        a = sum(y for _, y in dp) / len(dp)
        out["d_pose_before_mean"] = b
        out["d_pose_after_mean"] = a
        out["d_pose_ratio"] = a / b if b else None
        out["pairs_d_pose_improved"] = sum(1 for x, y in dp if y < x)
        out["d_pose_n"] = len(dp)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = json.loads(args.receipt.read_text())
    rows = d["rows"]
    n = len(rows)
    print(f"=== ddm_et1 pooled  n={n}/32 pairs  (DENOMINATOR stated per m50) ===")
    print(f"axis: {d['axis']}   score_claim={d['score_claim']}")
    print(f"controls: C2 {sum(r['C2_lstar_matches_cache'] for r in rows)}/{n}  "
          f"C3 {sum(r['C3_lgt_matches_cache'] for r in rows)}/{n}")

    # measured band geometry on THESE pairs (subset), vs the n600 population
    print(f"\n--- band geometry on these {n} pairs ---")
    for nm in ("gp1", "sq1", "gp1snap"):
        px = sum(r[f"band_{nm}_px"] for r in rows)
        cap = sum(r[f"band_{nm}_capture"] * r["flips_before"] for r in rows) / \
            sum(r["flips_before"] for r in rows)
        by = sum(r[f"band_{nm}_addr_bits"] + r[f"band_{nm}_pay_bits"] for r in rows) / 8.0
        print(f"  {nm:8s} px {px:9d}  capture {100*cap:6.3f}%  bytes(subset) {by:9.0f}")
    snap = sum(r["band_gp1snap_px"] for r in rows) / sum(r["band_gp1_px"] for r in rows)
    snapb = (sum(r["band_gp1snap_addr_bits"] + r["band_gp1snap_pay_bits"] for r in rows) /
             sum(r["band_gp1_addr_bits"] + r["band_gp1_pay_bits"] for r in rows))
    print(f"  SNAP TAX (sq1 2.8 IOU, discharged): px {snap:.4f}x  BYTES {snapb:.4f}x")

    res = {}
    print("\n--- eta by arm (all on the PRICED L1 SE; matched to sq1's pairs) ---")
    arms = [("a_truth_gp1", "(a) truth paint  [control]", "-3.7640"),
            ("b_solved_gp1", "(b) solved unconstrained  ", "+0.7895"),
            ("c_null_gp1", "(c) solved + Q3 pose-null ", "+0.5406")]
    if any("x_extended_eta_net" in r for r in rows):
        arms.append(("x_extended", "(x) EXTENDED headroom probe", "n/a"))
    for tag, lbl, sq1v in arms:
        s = pooled(rows, tag)
        res[tag] = s
        if not s["n"]:
            continue
        print(f"\n{lbl}   sq1 on ITS band: {sq1v}")
        print(f"  eta_pooled          {s['eta_pooled']:+.4f}   (n={s['n']}, "
              f"{s['flips_removed']} removed / {s['flips_described']} described)")
        print(f"  per-pair mean+-sd    {s['eta_per_pair_mean']:+.4f} +- {s['eta_per_pair_sd']:.4f}"
              f"   min {s['eta_min']:+.4f} max {s['eta_max']:+.4f}")
        print(f"  pairs > {GP1_F1_THRESHOLD:.5f} (gp1 F1)      "
              f"{s['pairs_above_gp1_F1']}/{s['n']}")
        print(f"  pairs > {HONEST_BREAKEVEN:.5f} (HONEST BE) "
              f"{s['pairs_above_honest_breakeven']}/{s['n']}")
        if "cap_pinned_n" in s:
            print(f"  CAP-PINNED (sm1 #935)       {s['cap_pinned_n']} = "
                  f"{100*s['cap_pinned_frac']:.1f}%")
        if "d_pose_ratio" in s:
            print(f"  d_pose {s['d_pose_before_mean']:.6f} -> {s['d_pose_after_mean']:.6f}"
                  f"  = {s['d_pose_ratio']:.4f}x   improved on "
                  f"{s['pairs_d_pose_improved']}/{s['d_pose_n']}")

    print(f"\n--- S-ARITHMETIC on the honest band (gap {GAP_S:.7f}; "
          f"seg leg {SEG_LEG_S:.6f}) ---")
    print("SEG-ONLY net S = rate - eta*gross;  NEGATIVE = score goes DOWN = worth doing.")
    print("POSE IS DELIBERATELY *NOT* FOLDED INTO net S.  sq1 1.6/2.6 measured this 32-pair")
    print("subset at 0.2692x of population on d_pose (gp1: the pose axis is 4.6x skewed vs")
    print("1.05x for seg), so a subset mean d_pose CANNOT be converted into a population dS.")
    print("The pose column below is a SUBSET-SCOPED ratio -- a gate, never a price.")
    print(f"{'row':44s} {'eta':>8s} {'rate':>8s} {'gross':>8s} {'segnet S':>9s} "
          f"{'%gap':>7s} {'d_pose x [subset]':>18s}")
    for tag, lbl, _ in arms:
        s = res.get(tag) or {}
        if not s.get("n"):
            continue
        b = BANDS_N600["gp1"]
        rate = b["bytes"] * RATE_PER_BYTE
        gross = b["capture"] * SEG_LEG_S
        eta = s["eta_pooled"]
        net = rate - eta * gross
        pr = f"{s['d_pose_ratio']:.4f}x" if "d_pose_ratio" in s else "n/a"
        print(f"{lbl + ' on ' + b['label']:44s} {eta:+8.4f} {rate:8.5f} {gross:8.5f} "
              f"{net:+9.5f} {100*(-net)/GAP_S:+7.2f} {pr:>18s}")
    print(f"\nbreak-even eta (DERIVED, honest band) = rate/gross = {HONEST_BREAKEVEN:.5f}"
          f"   [gp1's own PRICED band gives {breakeven_eta('A3_gp1'):.5f} vs its published "
          f"F1 = {GP1_F1_THRESHOLD} -- the consistency check]")

    if args.out:
        args.out.write_text(json.dumps(
            {"schema": "ddm_et1_aggregate.v1", "n_pairs": n,
             "axis": d["axis"], "score_claim": False, "promotion_eligible": False,
             "pointer": "0.1910828242 [contest-CPU] UNMOVED",
             "denominators": {"gap_S": GAP_S, "seg_leg_S": SEG_LEG_S,
                              "dS_ddpose_current": DS_DDPOSE,
                              "rate_per_byte": RATE_PER_BYTE},
             "bands_n600": BANDS_N600, "arms": res,
             "snap_tax_px": snap, "snap_tax_bytes": snapb}, indent=1))
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
