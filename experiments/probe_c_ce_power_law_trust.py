# SPDX-License-Identifier: MIT
"""PROBE C / AUDIT 3: CE power-law trust on the d_seg trajectory.

Fits d_seg ~= A * ep^(-p) to the trajectory eval rows and reports whether the
"flattening tail" is real power-law decay or a shadow-lag floor. Also fits on a
late-window-only subset to detect a lag-induced floor (a true power law has a
window-stable exponent; an EMA-shadow floor shows the exponent collapsing toward
0 in the late window as the shadow saturates below the live weights).

$0, READ-ONLY, NON-PROMOTABLE [contest-CPU advisory]. NumPy-only; no scorer load.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_eval_rows(result_dir: Path):
    rows = []
    p = result_dir / "torch_vehicle_trajectory.jsonl"
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r.get("d_seg") is not None and r.get("evaluated"):
                rows.append((int(r["global_epoch"]), float(r["d_seg"]),
                             float(r["d_pose"]) if r.get("d_pose") is not None else None))
    return sorted(rows)


def powerlaw_fit(eps, ys):
    """Least-squares fit log(y) = log(A) - p*log(ep). Returns (A, p, r2)."""
    xs = [math.log(e) for e in eps]
    ly = [math.log(y) for y in ys]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ly) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ly))
    slope = sxy / sxx
    inter = my - slope * mx
    p = -slope
    A = math.exp(inter)
    ss_res = sum((y - (inter + slope * x)) ** 2 for x, y in zip(xs, ly))
    ss_tot = sum((y - my) ** 2 for y in ly)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return A, p, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="basin", choices=["basin", "armb"])
    ap.add_argument("--warmup-skip", type=int, default=40,
                    help="skip the init-collapse epochs (ep<=this) before fitting")
    args = ap.parse_args()
    dirs = {
        "basin": REPO / "experiments/results/torch_vehicle_full_mps_basin_bc20_n600",
        "armb": REPO / "experiments/results/bindall_arm_b_canonical50k_n600",
    }
    rows = load_eval_rows(dirs[args.arm])
    fit_rows = [(e, ds) for (e, ds, _) in rows if e > args.warmup_skip]
    eps = [e for e, _ in fit_rows]
    ys = [d for _, d in fit_rows]

    A, p, r2 = powerlaw_fit(eps, ys)

    # window-stability check: fit early-half vs late-half exponents.
    half = len(fit_rows) // 2
    Ae, pe, r2e = powerlaw_fit([e for e, _ in fit_rows[:half]],
                               [d for _, d in fit_rows[:half]])
    Al, pl, r2l = powerlaw_fit([e for e, _ in fit_rows[half:]],
                               [d for _, d in fit_rows[half:]])

    # canonical reference fit from MEMORY: d_seg ~= 0.0367 * ep^-0.351
    ref_A, ref_p = 0.0367, 0.351

    print(json.dumps({
        "arm": args.arm,
        "n_eval_rows": len(rows),
        "n_fit_rows": len(fit_rows),
        "epoch_range": [eps[0], eps[-1]],
        "fit_full": {"A": A, "p": p, "r2": r2},
        "fit_early_half": {"A": Ae, "p": pe, "r2": r2e, "ep_range": [fit_rows[0][0], fit_rows[half-1][0]]},
        "fit_late_half": {"A": Al, "p": pl, "r2": r2l, "ep_range": [fit_rows[half][0], fit_rows[-1][0]]},
        "exponent_window_stability": {
            "early_p": pe, "late_p": pl, "delta": pl - pe,
            "interpretation": ("late_p collapsing toward 0 => shadow-floor / "
                               "true-plateau signal; stable p => clean power law"),
        },
        "canonical_reference": {"A": ref_A, "p": ref_p,
                                "note": "MEMORY d_seg~=0.0367*ep^-0.351 (capstone-era CE)"},
        "predicted_d_seg_at_50k_full_fit": A * (50000 ** -p),
        "predicted_d_seg_at_50k_canonical": ref_A * (50000 ** -ref_p),
        "sub015_d_seg_target": 0.000322,
        "authority": "[contest-CPU advisory] NON-PROMOTABLE",
    }, indent=2))


if __name__ == "__main__":
    main()
