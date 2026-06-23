#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Math-optimal joint decoder solver — the "solve math-optimal everywhere" entry point.

Runs ``tac.optimization.math_optimal_joint_solver.solve_math_optimal_joint`` and prints
the math-optimal joint config (C*, T*, Q*, E*), the predicted S*, the MEASURED
achievable-S lower bound (the real T_floor over the surface), the PHYSICAL achievable
floor (the existence-proof construction), the training-time Pareto, and the gating
sensitivities. Writes a JSON report.

NO GPU, NO training, NO score claim — every number is [advisory] / [macOS-CPU advisory]
NON-PROMOTABLE. The exact pointer stays pointer-only (0.19110). The OUTPUT is a CONFIG
RECOMMENDATION that feeds the next training run.

Usage:
  .venv/bin/python tools/math_optimal_joint_solve.py
  .venv/bin/python tools/math_optimal_joint_solve.py --hold-delta 0.0001 \
      --json reports/math_optimal_joint_solve.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from tac.optimization.math_optimal_joint_solver import (
    min_training_budget_for_threshold,
    solve_math_optimal_joint,
)


def _cfg_line(c) -> str:
    e = "inf" if c.epochs == float("inf") else f"{c.epochs:g}"
    return (
        f"C={c.base_ch} ({c.decoder_params}p)  T={c.taper_dseg_multiplier:g}  "
        f"Q=int{c.qat_nbits}/fl{c.qat_frac_low_precision:g}  E={e}  "
        f"-> d_seg={c.d_seg:.5g} d_pose={c.d_pose:.4g} bytes={c.archive_bytes} S={c.S:.5f}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-chs", type=int, nargs="+", default=[16, 20, 24, 28, 32, 36, 40])
    ap.add_argument("--qat-nbits", type=int, nargs="+", default=[4, 5, 6, 8])
    ap.add_argument("--frac-low", type=float, nargs="+", default=[0.0, 0.70, 1.0])
    ap.add_argument(
        "--epochs", type=float, nargs="+", default=[2325.0, 10000.0, 30000.0, 1e6]
    )
    ap.add_argument(
        "--taper-mults",
        type=float,
        nargs="+",
        default=[1.0],
        help="taper d_seg multipliers (1.0=vendored; <1=boundary-band realloc)",
    )
    ap.add_argument(
        "--hold-delta",
        type=float,
        default=0.0003,
        help="QAT d_seg-hold spill (the distortion-hold ASSUMPTION)",
    )
    ap.add_argument("--json", type=str, default="reports/math_optimal_joint_solve.json")
    args = ap.parse_args(argv)

    res = solve_math_optimal_joint(
        base_chs=tuple(args.base_chs),
        taper_dseg_multipliers=tuple(args.taper_mults),
        qat_nbits_options=tuple(args.qat_nbits),
        qat_frac_low_options=tuple(args.frac_low),
        epochs_options=tuple(args.epochs),
        qat_d_seg_hold_delta=args.hold_delta,
    )
    o = res.optimum

    W = 96
    print("=" * W)
    print("MATH-OPTIMAL JOINT DECODER SOLVE  [advisory] NON-PROMOTABLE  (pointer 0.19110 unmoved)")
    print("=" * W)
    print("MATH-OPTIMAL CONFIG (C*, T*, Q*, E*):")
    print(f"  {_cfg_line(o)}")
    print(f"  d_seg evidence: {o.d_seg_evidence}")
    print()
    print("ACHIEVABLE-S LOWER BOUND (surface-model T_floor, power-law d_seg):")
    print(f"  S_LB = {res.achievable_S_lower_bound:.5f}  @ {_cfg_line(res.achievable_S_lower_bound_config)}")
    print()
    print("PHYSICAL ACHIEVABLE FLOOR (existence-proof: perfect-384 d_seg + converged pose + small bytes):")
    pf = res.physical_floor_config
    print(f"  S_floor = {res.physical_floor_S:.5f}  @ {_cfg_line(pf)}")
    print("  (this REPLACES the loose analytic 0.118 floor)")
    print()
    print(f"  frontier S = {res.frontier_S:.5f}  |  sub-0.15 = THE goal  |  sub-0.19 = floor of acceptable")
    print()

    # Min training budget to cross each threshold at the optimum (C,T,Q).
    print("MIN TRAINING BUDGET (effective epochs) to cross S thresholds at optimum (C,T,Q):")
    for thr in (0.19, 0.17, 0.15):
        b = min_training_budget_for_threshold(
            thr,
            base_ch=o.base_ch,
            taper_dseg_multiplier=o.taper_dseg_multiplier,
            qat_nbits=o.qat_nbits,
            qat_frac_low_precision=o.qat_frac_low_precision,
            qat_d_seg_hold_delta=args.hold_delta,
            surfaces=res.surfaces,
        )
        if b is None:
            txt = "UNREACHABLE at this (C,T,Q) (converged S above threshold)"
        elif b == float("inf"):
            txt = "needs the asymptote (unbounded budget)"
        else:
            txt = f"~{round(b)} epochs"
        print(f"  sub-{thr:.2f}: {txt}")
    print()

    print("TRAINING-TIME PARETO @ optimum (C,T,Q)  [S vs effective epochs]:")
    print(f"  {'epochs':>10} {'d_seg':>10} {'S':>9}")
    for p in res.training_pareto:
        e = "inf" if p.epochs == float("inf") else f"{p.epochs:g}"
        print(f"  {e:>10} {p.d_seg:>10.5g} {p.S:>9.5f}")
    print()

    print("GATING SENSITIVITIES (what's gated + the value that flips the optimum):")
    for k, v in res.gating.items():
        print(f"  [{k}]")
        print(f"      {v}")
    print()

    print("NOTES:")
    for n in res.notes:
        print(f"  · {n}")
    print("=" * W)

    out = {
        "tool": "math_optimal_joint_solve",
        "authority": "[contest-CPU advisory] / [macOS-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotable": False,
        "frontier_pointer_moved": False,
        "params": {
            "base_chs": args.base_chs,
            "qat_nbits": args.qat_nbits,
            "frac_low": args.frac_low,
            "epochs": args.epochs,
            "taper_mults": args.taper_mults,
            "hold_delta": args.hold_delta,
        },
        "optimum": dataclasses.asdict(o),
        "achievable_S_lower_bound": res.achievable_S_lower_bound,
        "achievable_S_lower_bound_config": dataclasses.asdict(res.achievable_S_lower_bound_config),
        "achievable_S_existence_proof": dataclasses.asdict(res.achievable_S_existence_proof),
        "physical_floor_S": res.physical_floor_S,
        "physical_floor_config": dataclasses.asdict(res.physical_floor_config),
        "frontier_S": res.frontier_S,
        "training_pareto": [dataclasses.asdict(p) for p in res.training_pareto],
        "gating": res.gating,
        "surfaces": dataclasses.asdict(res.surfaces),
        "notes": res.notes,
    }
    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"JSON -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
