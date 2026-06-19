#!/usr/bin/env python3
"""Analyze the amortized-NCA capacity-break sweep -> the decisive d_seg(params) vs 29.3*params^-0.71 curve.

Reads the daemon's gate_state.json (resumable; can run on a PARTIAL sweep) and emits the capacity-break
table + the fitted-exponent verdict. NO new training; pure analysis of measured rows. This lets the verdict
be computed the moment enough converged points exist, without waiting for every config.

$0, no GPU. `[contest-CPU advisory]` NON-PROMOTABLE; pointer UNMOVED 0.19110.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

POWERLAW_A = 29.3
POWERLAW_K = 0.71
FRONTIER_DSEG = 0.00056
FRONTIER_S = 0.19110
GREEN_DSEG = 0.0012
HELD_POSE = 0.00034


def powerlaw(p):
    return POWERLAW_A * (max(p, 1.0) ** (-POWERLAW_K))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(REPO / "experiments/results/nca_amortized_capacity_break_main/gate_state.json"))
    ap.add_argument("--converged-dseg-max", type=float, default=0.3,
                    help="rows with best_converged avg d_seg above this are treated as non-converged (excluded from fit)")
    args = ap.parse_args()

    sp = Path(args.state)
    if not sp.exists():
        print(f"[state not written yet] {sp} — first config still running")
        return
    state = json.loads(sp.read_text())
    rows = state.get("rows", {})
    if not rows:
        print("[no rows yet]")
        return

    print(f"{'config':>10} {'rule_pc':>9} {'conv':>6} {'avg_dseg':>10} {'xfront':>8} "
          f"{'powerlaw':>9} {'ratio':>7} {'beats':>6} {'rate':>8} {'S':>8}")
    pts = []
    for key in sorted(rows, key=lambda k: rows[k]["rule_param_count"]):
        r = rows[key]
        pc = r["rule_param_count"]
        d = r["best_converged_avg_realized_dseg"]
        conv = f"{r['n_converged_restarts']}/{r['n_restarts']}"
        pl = powerlaw(pc)
        ratio = d / pl if pl > 0 else float("nan")
        beats = d < pl
        rate = r["rate_amortized"]
        s = r["S_projected_amortized"]
        converged = r["n_converged_restarts"] > 0 and d < args.converged_dseg_max
        flag = "" if converged else "  (NOT-CONV)"
        print(f"{key:>10} {pc:>9d} {conv:>6} {d:>10.5f} {d/FRONTIER_DSEG:>7.1f}x "
              f"{pl:>9.5f} {ratio:>7.2f} {str(beats):>6} {rate:>8.5f} {s:>8.3f}{flag}")
        if converged:
            pts.append((pc, d, s, rate))

    print(f"\n[converged points for fit]: {len(pts)}")
    if len(pts) >= 2:
        import numpy as np

        xs = np.log([p[0] for p in pts])
        ys = np.log([p[1] for p in pts])
        slope, intercept = np.polyfit(xs, ys, 1)
        k = -slope
        A = math.exp(intercept)
        print(f"[fitted] our d_seg(params) = {A:.3g} * params^-{k:.3f}")
        print(f"[powerlaw wall]            d_seg(params) = {POWERLAW_A} * params^-{POWERLAW_K}")
        print(f"[exponent]  ours k={k:.3f} vs wall k={POWERLAW_K}  -> "
              f"{'STEEPER (iteration helps)' if k > POWERLAW_K + 0.10 else 'SAME/SHALLOWER (no capacity break)'}")
        # where would our fit cross GREEN (d_seg=0.0012)?
        if k > 0:
            p_green = (A / GREEN_DSEG) ** (1.0 / k)
            print(f"[extrapolation] our fit reaches GREEN d_seg={GREEN_DSEG} at ~{p_green:,.0f} params")
    elif len(pts) == 1:
        pc, d, s, rate = pts[0]
        print(f"[single converged point] params={pc} d_seg={d:.5f} ({d/powerlaw(pc):.2f}x powerlaw) S={s:.3f}")
        print("  (need >=2 converged points to fit the exponent)")

    best_s = min((p[2] for p in pts), default=float("inf"))
    best_d = min((p[1] for p in pts), default=float("inf"))
    print(f"\n[best converged] d_seg={best_d:.5f} ({best_d/FRONTIER_DSEG:.1f}x frontier)  S={best_s:.4f} (frontier {FRONTIER_S})")
    overall_conv = sum(r["n_converged_restarts"] for r in rows.values()) / max(1, sum(r["n_restarts"] for r in rows.values()))
    print(f"[overall convergence rate] {overall_conv:.2f} ({sum(r['n_converged_restarts'] for r in rows.values())}/"
          f"{sum(r['n_restarts'] for r in rows.values())} restarts)")
    if best_s < FRONTIER_S:
        print("[fork] -> GREEN candidate (S < frontier); byte-close + paired CPU/CUDA next")
    elif best_s < 0.19 and overall_conv >= 0.5:
        print("[fork] -> AMBER (reliable + shared-rule holds, S in [0.15,0.19))")
    else:
        print("[fork] -> trends RED (caps above sub-0.15 / does not beat frontier)")


if __name__ == "__main__":
    main()
