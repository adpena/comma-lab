# SPDX-License-Identifier: MIT
"""ddm_hg1 PROBE 3 -- (a) make the TRUNCATION claim exact, (b) test the registered prediction P1.

(a) TRUNCATION.  Probe 2 showed every directed side runs to the depth cap EXCEPT the Lane
    sides.  Quantify it without eyeballing: for each side report the depth at which the
    surviving population falls below 1% / 0.1% of its depth-1 population, and the integral of
    the margin profile over the surviving support (the total barrier to ANNIHILATING the class
    from that side, as opposed to merely nudging its boundary one pixel).

(b) P1 (registered in probe 2 BEFORE the answer was seen): the witness-measured directed flip
    rate should be rank-predicted by the source side's profile.  Four candidate predictors,
    reported together so the winner is not cherry-picked:
       margin_d1        -- cost DENSITY of the first flip (the "shallower side is cheaper" story)
       mean_depth       -- EXTENT of the source class (the "less to defend" story)
       shell_fraction   -- fraction of the source class already one flip from the separatrix
       barrier_integral -- sum of margin over surviving support (density x extent)
    REFUTED if |Spearman| < 0.5 for all four.

Flip rates are the 8 MAJOR directed sides from the 2026-07-08 witness artifact
(experiments/results/t5_probe_waveB_20260708/q1_signed_asymmetry.json, 96 witness frames).
CAVEAT THAT TRAVELS WITH THE NUMBER: those rates are WITNESS-vehicle, 96 frames; the profiles
are frozen-scorer GT, n600, vehicle-independent.  A confirmed P1 is therefore a HYPOTHESIS for
cx1/TR1, not a transfer.

$0, scorer-free, cached artifacts only.  [macOS-numpy advisory . NON-PROMOTABLE]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def spearman(a: list[float], b: list[float]) -> float:
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float))).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main() -> int:
    prof = json.loads((REPO / ".omx/research/ddm_hg1_signed_depth_profile_n600.json").read_text())
    q1 = json.loads((REPO / "experiments/results/t5_probe_waveB_20260708/q1_signed_asymmetry.json").read_text())
    out: dict = {"probe": "hg1_probe3_truncation_and_prediction",
                 "advisory": "[macOS-numpy advisory . NON-PROMOTABLE] pointer 0.1910828242 UNMOVED",
                 "profile_source": "ddm_hg1_signed_depth_profile_n600.json (GT, n600, vehicle-independent)",
                 "flip_rate_source": "q1_signed_asymmetry.json (WITNESS vehicle, 96 frames -- caveat travels)"}

    # ---- (a) truncation ledger -----------------------------------------------------------
    trunc = {}
    for key, v in prof["directed_profiles"].items():
        p = [e for e in v["profile"] if e]
        if not p or p[0]["n"] == 0:
            continue
        n1 = p[0]["n"]
        d_1pct = next((e["depth"] for e in p if e["n"] < 0.01 * n1), None)
        d_01pct = next((e["depth"] for e in p if e["n"] < 0.001 * n1), None)
        # barrier: integrate margin over depths while population is >= 1% of depth-1
        barrier = 0.0
        for e in p:
            if e["n"] < 0.01 * n1:
                break
            barrier += e["mean_margin"]
        trunc[key] = {
            "from": v["from"], "to": v["to"], "n_pixels": v["n_pixels"],
            "n_at_depth1": n1,
            "depth_where_pop_below_1pct": d_1pct,
            "depth_where_pop_below_0.1pct": d_01pct,
            "barrier_integral_to_1pct": barrier,
            "margin_d1": p[0]["mean_margin"],
        }
    out["truncation_ledger"] = trunc

    # ---- (b) P1 --------------------------------------------------------------------------
    geom = prof["class_geometry"]
    rows = []
    for name, row in q1["pairs"].items():
        if not row["major"]:
            continue
        if name not in trunc:
            continue
        t = trunc[name]
        g = geom[row["class_from"]]
        rows.append({
            "side": f"{name} {row['class_from']}->{row['class_to']}",
            "flip_rate": row["flip_rate"],
            "margin_d1": t["margin_d1"],
            "mean_depth": g["mean_depth"],
            "shell_fraction": g["shell_fraction"],
            "barrier_integral": t["barrier_integral_to_1pct"],
        })
    preds = ["margin_d1", "mean_depth", "shell_fraction", "barrier_integral"]
    y = [r["flip_rate"] for r in rows]
    out["p1_rows"] = rows
    out["p1_spearman_vs_flip_rate"] = {p: spearman([r[p] for r in rows], y) for p in preds}
    out["p1_n_sides"] = len(rows)
    best = max(preds, key=lambda p: abs(out["p1_spearman_vs_flip_rate"][p]))
    out["p1_verdict"] = {
        "best_predictor": best,
        "best_abs_spearman": abs(out["p1_spearman_vs_flip_rate"][best]),
        "registered_refutation_band": "|rs| < 0.5 for ALL four => P1 REFUTED",
        "refuted": all(abs(v) < 0.5 for v in out["p1_spearman_vs_flip_rate"].values()),
    }

    # ---- asymmetry ratio table (measured, both directions available) ---------------------
    asym = {}
    for (i, j) in [(0, 1), (0, 2), (0, 3), (0, 4), (2, 3)]:
        a, b = f"{i}->{j}", f"{j}->{i}"
        if a in q1["pairs"] and b in q1["pairs"]:
            ra, rb = q1["pairs"][a]["flip_rate"], q1["pairs"][b]["flip_rate"]
            asym[f"{q1['pairs'][a]['class_from']}<->{q1['pairs'][a]['class_to']}"] = {
                f"rate_{a}": ra, f"rate_{b}": rb,
                "asymmetry_ratio_max_over_min": max(ra, rb) / min(ra, rb) if min(ra, rb) > 0 else None,
                "erosion_favours": (a if ra > rb else b),
            }
    out["flip_rate_asymmetry_ratios"] = asym

    dst = REPO / ".omx/research/ddm_hg1_truncation_and_prediction.json"
    dst.write_text(json.dumps(out, indent=1))

    print("=== (a) TRUNCATION LEDGER (GT n600, vehicle-independent) ===")
    print(f"  {'side':>22} {'n@d1':>9} {'d<1%':>6} {'d<0.1%':>7} {'barrier':>9} {'m(d1)':>7}")
    for k, t in sorted(trunc.items(), key=lambda kv: -kv[1]["n_pixels"]):
        if t["n_pixels"] < 100000:
            continue
        print(f"  {k+' '+t['from'][:4]+'->'+t['to'][:4]:>22} {t['n_at_depth1']:>9} "
              f"{t['depth_where_pop_below_1pct']!s:>6} {t['depth_where_pop_below_0.1pct']!s:>7} "
              f"{t['barrier_integral_to_1pct']:>9.2f} {t['margin_d1']:>7.3f}")
    print("\n=== (b) P1 : Spearman(predictor, witness flip rate), 8 major sides ===")
    for p, v in out["p1_spearman_vs_flip_rate"].items():
        print(f"  {p:>18}: rs = {v:+.4f}")
    print(f"  VERDICT: best={out['p1_verdict']['best_predictor']} "
          f"|rs|={out['p1_verdict']['best_abs_spearman']:.4f} refuted={out['p1_verdict']['refuted']}")
    print("\n=== flip-rate asymmetry ratios (witness, 96 frames) ===")
    for k, v in asym.items():
        print(f"  {k:>22}: ratio={v['asymmetry_ratio_max_over_min']:6.2f}x  erosion favours {v['erosion_favours']}")
    print(f"\n[done] {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
