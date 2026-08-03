# SPDX-License-Identifier: MIT
"""ddm_hg1 PROBE 4 -- close F2, the falsifier this arm registered against its OWN framing.

F2 as registered in the memo: "interior-stratum residual scales with INTERFACE LENGTH; Lane's
scales with AREA.  REFUTED if Lane's flip count is better explained by interface length than by
area -- the annihilation framing is wrong."

This is the cheapest test available and it attacks the arm's own §6 stratification, so it runs
before the memo is committed rather than being left as an owed item.

Two competing scaling models for per-class ERASURE flips (flips OUT of class c):
    M_perimeter : flips_c  proportional to  perimeter_c   (boundary-nudge regime)
    M_area      : flips_c  proportional to  area_c        (annihilation regime)
Scored by coefficient of variation of the implied constant across the 5 classes -- the model
whose "constant" is actually constant wins.  Lane is then checked for OUTLIER status under the
winning model; the framing survives only if Lane deviates.

Also resolves a second-order question the depth profile raised: the Road|Lane annulus is NOT
balanced (a 2-px annulus on the Lane side is truncated by Lane's own ~1.7-px width), so the
per-pixel RATE asymmetry and the FLIP-MASS asymmetry are different numbers.  Both are reported.

Inputs: sx1's n600 GT separatrix geometry (edge_len, class_area) + the witness directed flip
counts (96 frames).  Vehicle caveat travels with every flip-derived number.
$0, scorer-free.  [macOS-numpy advisory . NON-PROMOTABLE]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CLASSES = ["Road", "Lane", "Undrivable", "Movable", "MyCar"]
IDX = {n: i for i, n in enumerate(CLASSES)}


def main() -> int:
    geo = json.loads((REPO / ".omx/research/ddm_sx1_separatrix_geometry_n600.json").read_text())
    q1 = json.loads((REPO / "experiments/results/t5_probe_waveB_20260708/q1_signed_asymmetry.json").read_text())
    # sx1 spells class 2 "Undriv"; the q1 artifact spells it "Undrivable". Normalise to CLASSES.
    norm = {"Undriv": "Undrivable", "Mova": "Movable"}
    area = {norm.get(k, k): v for k, v in geo["class_area"].items()}
    assert set(area) == set(CLASSES), sorted(area)
    # undirected crack length -> per-class perimeter
    per = dict.fromkeys(CLASSES, 0)
    for pair, L in geo["edge_len"].items():
        a, b = (norm.get(t, t) for t in pair.split("|"))
        per[a] += L
        per[b] += L

    flips_out = dict.fromkeys(CLASSES, 0)
    pop_out = dict.fromkeys(CLASSES, 0)
    for row in q1["pairs"].values():
        flips_out[row["class_from"]] += row["n_flips"]
        pop_out[row["class_from"]] += row["n_pixels"]

    rows = []
    for c in CLASSES:
        rows.append({"class": c, "area": area[c], "perimeter": per[c],
                     "mean_width_2A_over_P": 2.0 * area[c] / per[c],
                     "flips_out": flips_out[c],
                     "flips_per_perimeter": flips_out[c] / per[c],
                     "flips_per_area": flips_out[c] / area[c]})

    def cv(vals):
        v = np.asarray(vals, dtype=float)
        return float(v.std() / v.mean())

    cv_per = cv([r["flips_per_perimeter"] for r in rows])
    cv_area = cv([r["flips_per_area"] for r in rows])
    lane_fp = rows[IDX["Lane"]]["flips_per_perimeter"]
    others_fp = [r["flips_per_perimeter"] for r in rows if r["class"] != "Lane"]
    lane_z = (lane_fp - np.mean(others_fp)) / np.std(others_fp)

    # Road|Lane: rate asymmetry vs flip-mass asymmetry (they differ because the annulus is unbalanced)
    rl, lr = q1["pairs"]["0->1"], q1["pairs"]["1->0"]
    edge = {
        "rate_Road_to_Lane": rl["flip_rate"], "rate_Lane_to_Road": lr["flip_rate"],
        "rate_asymmetry": lr["flip_rate"] / rl["flip_rate"],
        "annulus_pop_Road_side": rl["n_pixels"], "annulus_pop_Lane_side": lr["n_pixels"],
        "annulus_population_imbalance": rl["n_pixels"] / lr["n_pixels"],
        "flips_Road_to_Lane": rl["n_flips"], "flips_Lane_to_Road": lr["n_flips"],
        "flip_mass_asymmetry": lr["n_flips"] / rl["n_flips"],
        "Lane_to_Road_share_of_edge": lr["n_flips"] / (lr["n_flips"] + rl["n_flips"]),
    }

    out = {
        "probe": "hg1_probe4_f2_scaling_law",
        "advisory": "[macOS-numpy advisory . NON-PROMOTABLE] pointer 0.1910828242 UNMOVED",
        "geometry_source": "ddm_sx1_separatrix_geometry_n600.json (GT n600)",
        "flip_source": "q1_signed_asymmetry.json (WITNESS vehicle, 96 frames -- caveat travels)",
        "rows": rows,
        "cv_flips_per_perimeter": cv_per,
        "cv_flips_per_area": cv_area,
        "winning_model": "perimeter" if cv_per < cv_area else "area",
        "lane_z_under_winning_model": float(lane_z),
        "f2_verdict": {
            "registered": "REFUTED if Lane's flip count is better explained by interface length than by area",
            "refuted": bool(cv_per < cv_area and abs(lane_z) < 2.0),
        },
        "road_lane_edge": edge,
    }
    dst = REPO / ".omx/research/ddm_hg1_f2_scaling_law.json"
    dst.write_text(json.dumps(out, indent=1))

    print("=== per-class erasure scaling (flips OUT of class) ===")
    print(f"  {'class':>12} {'area':>10} {'perim':>9} {'width':>7} {'flips':>7} "
          f"{'flips/perim':>12} {'flips/area':>11}")
    for r in rows:
        print(f"  {r['class']:>12} {r['area']:>10} {r['perimeter']:>9} "
              f"{r['mean_width_2A_over_P']:>7.2f} {r['flips_out']:>7} "
              f"{r['flips_per_perimeter']:>12.5f} {r['flips_per_area']:>11.2e}")
    print(f"\n  CV(flips/perimeter) = {cv_per:.4f}   CV(flips/area) = {cv_area:.4f}"
          f"   -> winning model: {out['winning_model']}")
    print(f"  Lane z-score under winning model (vs other 4): {lane_z:+.3f}")
    print(f"  F2 REFUTED = {out['f2_verdict']['refuted']}")
    print("\n=== Road|Lane : rate asymmetry is NOT flip-mass asymmetry ===")
    for k, v in edge.items():
        print(f"  {k:>32}: {v:.4f}" if isinstance(v, float) else f"  {k:>32}: {v}")
    print(f"\n[done] {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
