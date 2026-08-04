#!/usr/bin/env python
"""ddm_sq1 aggregation -- per-stage / per-class / per-edge, with every denominator named.

Emits the numbers §2 of the memo quotes.  No scorer forwards; pure arithmetic over the two
receipts.  Typed outcome vocabulary per the operator steer 2026-08-03:
ETA_HIGH_ROW_FIREABLE | ETA_LOW_DEBT_NAMED(stage, cure, bound-retained).  Never ROW_DEAD.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CN = ["Road", "Lane", "Undrivable", "Movable", "MyCar"]
LIVE_BEST_S = 0.826496209256714
FLOOR_S = 0.172141
GAP_S = LIVE_BEST_S - FLOOR_S
BASE_D_POSE = 0.0025513987495742437
POP_MEAN_FLIPS = 847.7333333333333
A3_BYTES = 367_523
A3_GROSS_S = 0.41938
RATE_DEN = 37_545_489


def s_of_bytes(b: float) -> float:
    return 25.0 * b / RATE_DEN


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: ddm_sq1_aggregate.py <v0_receipt.json> <v1_receipt.json> <out.json>")
        return 2
    v0_obj = json.loads(Path(sys.argv[1]).read_text())
    v1_obj = json.loads(Path(sys.argv[2]).read_text())
    v0 = v0_obj["rows"]
    v1 = v1_obj["rows"]
    out: dict = {"schema": "ddm_sq1_aggregate.v1", "score_claim": False,
                 "pointer": "0.1910828242 [contest-CPU] UNMOVED",
                 "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE"}

    fb = np.array([r["flips_before"] for r in v0], float)
    print(f"n_v0={len(v0)}  n_v1={len(v1)}")
    print(f"CONTROLS  C2 {all(r['C2_lstar_matches_cache'] for r in v0)}  "
          f"C3 {all(r['C3_lgt_matches_cache'] for r in v0)}  "
          f"C4 {all(r['P0_fullpaste_flips_after'] == 0 for r in v0)}")
    print(f"m88  subset mean flips {fb.mean():.4f} vs population {POP_MEAN_FLIPS:.4f} "
          f"ratio {fb.mean()/POP_MEAN_FLIPS:.6f}")
    out["controls"] = {"C2": bool(all(r["C2_lstar_matches_cache"] for r in v0)),
                       "C3": bool(all(r["C3_lgt_matches_cache"] for r in v0)),
                       "C4_full_paste_eta1": bool(all(r["P0_fullpaste_flips_after"] == 0 for r in v0)),
                       "m88_ratio": float(fb.mean() / POP_MEAN_FLIPS)}

    # ---- S0 address ------------------------------------------------------------------------
    print("\n--- S0 ADDRESS (gp1 F4: decoder's ACTUAL L*; <90% capture is the falsifier) ---")
    s0 = {}
    for r_ in (1, 2, 3, 5, 8, 13, 21, 34, 55):
        nd = np.array([x[f"L0_r{r_}_n_described"] for x in v0], float)
        cap = nd.sum() / fb.sum()
        fr = np.mean([x[f"band_r{r_}_frac"] for x in v0])
        fa = np.array([x[f"L0_r{r_}_flips_after"] for x in v0], float)
        eta = (fb - fa).sum() / nd.sum()
        s0[r_] = {"capture": float(cap), "band_frac": float(fr), "eta_v0_truthpaste": float(eta)}
        print(f"  r={r_:3d} band {fr*100:5.1f}%  capture {cap:.4f}  eta_v0 {eta:+.4f}")
    out["S0_address_and_v0_locality_curve"] = s0

    # ---- S1/S2/S3 exactness ------------------------------------------------------------------
    eb = np.array([r["S123_max_abs_err_on_band"] for r in v1])
    eo = np.array([r["S123_max_abs_err_off_band"] for r in v1])
    print(f"\n--- S1 paint / S2 R-D / S3 uint8 EXACTNESS over {len(v1)} pairs ---")
    print(f"  max abs err ON band  {eb.max():.6g}   OFF band {eo.max():.6g}"
          f"   -> {'EXACT (zero loss)' if eb.max()==0 and eo.max()==0 else 'LOSSY'}")
    out["S123_exactness"] = {"max_abs_err_on_band": float(eb.max()),
                             "max_abs_err_off_band": float(eo.max()),
                             "verdict": "EXACT" if eb.max() == 0 and eo.max() == 0 else "LOSSY"}

    # ---- S4 residual: v0 vs cured ------------------------------------------------------------
    f1 = np.array([r["flips_before"] for r in v1], float)
    nd1 = np.array([r["described_in_band"] for r in v1], float)
    print(f"\n--- S4 ARGMAX residual, r=1 band, n={len(v1)} (F1 threshold eta<=0.583) ---")
    s4 = {}
    for tag, name in (("S4_truthpaint", "v0 truth paint"), ("S4_solvedpaint", "v1 SOLVED paint")):
        fa = np.array([r[f"{tag}_flips_after"] for r in v1], float)
        fx = np.array([r[f"{tag}_fixed"] for r in v1], float)
        ip = np.array([r[f"{tag}_introduced"] for r in v1], float)
        eta = (f1 - fa).sum() / nd1.sum()
        per = (f1 - fa) / nd1
        dp = np.array([r[f"{tag}_d_pose_after"] for r in v1], float)
        s4[tag] = {"eta_net_pooled": float(eta), "eta_per_pair_mean": float(per.mean()),
                   "eta_per_pair_sd": float(per.std(ddof=1)) if len(per) > 1 else None,
                   "pairs_above_F1": int((per > 0.583).sum()), "n_pairs": len(per),
                   "fixed": float(fx.sum()), "introduced": float(ip.sum()),
                   "d_pose_after_mean": float(dp.mean())}
        print(f"  {name:16s} eta_net {eta:+.4f}  per-pair mean {per.mean():+.4f} "
              f"sd {per.std(ddof=1) if len(per)>1 else float('nan'):.4f}  "
              f"pairs>0.583 {(per>0.583).sum()}/{len(per)}  "
              f"fixed {fx.sum():.0f} introduced {ip.sum():.0f}  d_pose {dp.mean():.6f}")
    out["S4_residual"] = s4

    # ---- cap-artifact census ----------------------------------------------------------------
    solver = v1_obj.get("solver", {})
    requested_steps = int(solver.get("steps", -1))
    explicit = [r for r in v1 if "solved_stop_reason" in r]
    if explicit:
        reason_counts = {}
        best_at_cap = 0
        for r in explicit:
            reason = str(r["solved_stop_reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if int(r.get("solved_best_step", -1)) == requested_steps:
                best_at_cap += 1
        cap_census = {
            "n_pairs": len(v1),
            "explicit_stop_reason_rows": len(explicit),
            "requested_steps": requested_steps,
            "reason_counts": reason_counts,
            "best_at_requested_cap": best_at_cap,
        }
    else:
        # Back-compat for the original n32 receipt: no explicit stop reason existed, so the
        # only honest reading is "best landed at the cap", not "converged".
        tags = [str(r.get("solved_start_tag", "")) for r in v1]
        best_at_cap = sum(t.endswith(f"@{requested_steps}") for t in tags)
        cap_census = {
            "n_pairs": len(v1),
            "explicit_stop_reason_rows": 0,
            "requested_steps": requested_steps,
            "reason_counts": {},
            "best_at_requested_cap": best_at_cap,
            "legacy_inference": (
                "receipt has no stop_reason; tag-at-cap is cap-artifact evidence only"
            ),
        }
    print("\n--- cap-artifact census (sq1 solved-paint loop) ---")
    print(f"  denominator n={cap_census['n_pairs']} explicit_stop_reason_rows="
          f"{cap_census['explicit_stop_reason_rows']} requested_steps="
          f"{cap_census['requested_steps']} best_at_cap={cap_census['best_at_requested_cap']}")
    if cap_census["reason_counts"]:
        for reason, count in sorted(cap_census["reason_counts"].items()):
            print(f"  {reason:36s} {count:3d}/{cap_census['n_pairs']}")
    out["cap_artifact_census"] = cap_census

    trajectory_explicit = [r for r in v1 if "solved_trajectory_stop_reason" in r]
    if trajectory_explicit:
        trajectory_counts = {}
        trajectory_bound = 0
        for r in trajectory_explicit:
            reason = str(r["solved_trajectory_stop_reason"])
            trajectory_counts[reason] = trajectory_counts.get(reason, 0) + 1
            if reason == "safety_bound_REPORTED":
                trajectory_bound += 1
        trajectory_census = {
            "n_pairs": len(v1),
            "explicit_trajectory_stop_rows": len(trajectory_explicit),
            "reason_counts": trajectory_counts,
            "safety_bound_reported": trajectory_bound,
            "law_ref": trajectory_explicit[0]["solved_trajectory_stop"].get("law_ref"),
        }
        print("\n--- trajectory-derived stop census (canonical tj1 law) ---")
        print(f"  denominator n={trajectory_census['n_pairs']} "
              f"explicit_trajectory_stop_rows={len(trajectory_explicit)}")
        for reason, count in sorted(trajectory_counts.items()):
            print(f"  {reason:36s} {count:3d}/{trajectory_census['n_pairs']}")
        out["trajectory_stop_census"] = trajectory_census

    # ---- per-EDGE (pc2: never per class alone) ------------------------------------------------
    C0 = np.sum([np.array(r["C_before"]) for r in v1], axis=0)
    print("\n--- per-EDGE flips (gt -> rendered), pooled; pc2 hub law ---")
    # One key set for EVERY tag (a per-tag key set would make the comparison table ragged and
    # KeyError on any edge that only one rung touches).
    tags = ("S4_truthpaint", "S4_solvedpaint")
    Cs = {t: np.sum([np.array(r[f"{t}_C_after"]) for r in v1], axis=0) for t in tags}
    keys = [f"{CN[i]}->{CN[j]}" for i in range(5) for j in range(5)
            if i != j and (C0[i, j] > 0 or any(Cs[t][i, j] > 0 for t in tags))]
    edge = {t: {} for t in tags}
    for i in range(5):
        for j in range(5):
            k = f"{CN[i]}->{CN[j]}"
            if k not in keys:
                continue
            for t in tags:
                edge[t][k] = {"before": int(C0[i, j]), "after": int(Cs[t][i, j]),
                              "delta": int(Cs[t][i, j] - C0[i, j])}
    print(f"  total flips before {int(C0.sum() - np.trace(C0))}")
    for k in sorted(keys, key=lambda s: -edge[tags[0]][s]["before"])[:10]:
        a, b = edge["S4_truthpaint"][k], edge["S4_solvedpaint"][k]
        print(f"  {k:24s} before {a['before']:7d} | v0 {a['after']:7d} ({a['delta']:+7d}) "
              f"| SOLVED {b['after']:7d} ({b['delta']:+7d})")
    out["per_edge"] = edge

    # ---- re-price gp1 A3 at the MEASURED eta -------------------------------------------------
    print("\n--- gp1 A3 re-priced at measured eta (bound RETAINED, never a row kill) ---")
    rate = s_of_bytes(A3_BYTES)
    rep = {"A3_bytes": A3_BYTES, "A3_rate_S": rate, "A3_gross_bound_S": A3_GROSS_S,
           "A3_net_bound_S": rate - A3_GROSS_S, "gap_S": GAP_S}
    for tag in ("S4_truthpaint", "S4_solvedpaint"):
        e = s4[tag]["eta_net_pooled"]
        net = rate - e * A3_GROSS_S
        rep[tag] = {"eta": e, "realized_gross_S": e * A3_GROSS_S, "net_S": net,
                    "pct_of_gap": 100.0 * (-net) / GAP_S}
        print(f"  {tag:16s} eta {e:+.4f} -> gross {e*A3_GROSS_S:+.5f}  net {net:+.5f} S "
              f"({-net/GAP_S*100:+.2f}% of gap)")
    out["A3_repriced"] = rep

    # ---- pose collateral (scoped; NEVER extrapolated -- different population) ------------------
    # Baseline d_pose must be averaged over EXACTLY the pairs the v1 rungs were measured on,
    # or the "x baseline" ratio compares two different pair sets (the apples-to-apples rule).
    v1_pairs = {int(r["pair"]) for r in v1}
    dpb = np.array([r["d_pose_before"] for r in v0 if int(r["pair"]) in v1_pairs], float)
    print(f"\n--- POSE collateral (subset mean d_pose {dpb.mean():.8f} vs population "
          f"{BASE_D_POSE:.8f}, ratio {dpb.mean()/BASE_D_POSE:.4f}: DIFFERENT population, "
          f"reported not extrapolated) ---")
    out["pose"] = {"subset_mean_before": float(dpb.mean()), "population_mean": BASE_D_POSE,
                   "subset_vs_population_ratio": float(dpb.mean() / BASE_D_POSE)}
    for tag in ("S4_truthpaint", "S4_solvedpaint"):
        dp = np.array([r[f"{tag}_d_pose_after"] for r in v1], float)
        out["pose"][tag] = {"mean_after": float(dp.mean()),
                            "x_baseline": float(dp.mean() / dpb.mean())}
        print(f"  {tag:16s} d_pose {dp.mean():.6f}  = {dp.mean()/dpb.mean():.1f}x subset baseline")

    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {sys.argv[3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
