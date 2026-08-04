#!/usr/bin/env python
"""ddm_js1 aggregator -- pools the staging arms and does the S-arithmetic.

DENOMINATOR CORRECTION THIS UNIT OWNS.  et1 §8 quotes dS/dd_pose = 31.3026.  That is the
PRE-pu2 operating point: pu2's own receipt records baseline d_pose_mean 0.0025514 with
pose_contribution 0.1597, and 5/0.159731 = 31.3026 exactly.  pu2 then LOWERED d_pose to
0.00154517 (its win was pure pose -- independently confirmed by et1's C4 control).  At the
CURRENT live-best operating point the pose term is 0.124306 and

    dS/dd_pose = 5 / sqrt(10*d_pose) = 5 / 0.124306 = 40.2234

which is 28.5% HIGHER than the inherited constant.  Pose damage is MORE expensive than the
block16 row was priced against, because pu2 already harvested the cheap pose and moved us up
the sqrt curve where the derivative is steeper.  m66/qd1: a delta without its baseline is
unanchored, and baselines move.  Both values are reported so the correction is auditable.

POSE IS NEVER FOLDED INTO A POPULATION ΔS.  sq1 §1.6 / m96: the pose axis is 2.5-4.2x skewed
on non-population subsets and et1 measured this stratified selection at 0.2692x of population
on d_pose.  So pose is reported as a SUBSET-SCOPED GATE plus a break-even ratio, never as a
population score delta.  Mean-of-RATIOS is additionally the wrong statistic for a population
effect (a tiny-d_pose pair at 3.65x moves less absolute mass than a large-d_pose pair at
1.1x), so the absolute Δd_pose is what is pooled.

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE.  score_claim=false.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

# --- denominators, all n600 unless stated --------------------------------------------------
S_LIVE_BEST = 0.7910689
BYTES_LIVE_BEST = 353_805
S_FLOOR_PR130 = 0.172141
GAP = S_LIVE_BEST - S_FLOOR_PR130                      # 0.6189279
SEG_TERM = 0.431179                                    # 508,640 flips
RATE_TERM = 25.0 * BYTES_LIVE_BEST / 37_545_489.0
POSE_TERM = S_LIVE_BEST - SEG_TERM - RATE_TERM         # 0.124306
D_POSE_POP = (POSE_TERM ** 2) / 10.0                   # 0.00154517
DS_DDPOSE_NOW = 5.0 / POSE_TERM                        # 40.2234  <-- CORRECTED
DS_DDPOSE_ET1 = 31.3026                                # inherited, pre-pu2 (superseded)

# block16 regional phase field, re-solved on our vehicle (et1 §4 arm B, n600)
GROSS_S_BLOCK16 = 0.18039
BYTES_BLOCK16 = 46_247
RATE_COST_BLOCK16 = 25.0 * BYTES_BLOCK16 / 37_545_489.0
BREAKEVEN_ETA = RATE_COST_BLOCK16 / GROSS_S_BLOCK16


def pool(vals: list[float]) -> tuple[float, float, int]:
    v = [x for x in vals if x is not None]
    if not v:
        return float("nan"), float("nan"), 0
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), len(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # MERGE by pair, never take-first: a pair may be measured in several receipts, each carrying
    # a DIFFERENT arm (the n32 run carries cprime/poseonly, the cheapdct runs carry the carriage
    # arms).  Deduping by pair id and keeping the first row silently DROPS every arm measured in
    # a later receipt -- a defect this aggregator shipped with and which hid the entire k=4
    # carriage result on its first run.
    by_pair: dict[int, dict] = {}
    for r in args.receipts:
        if not r.exists():
            continue
        for row in json.loads(r.read_text()).get("rows", []):
            p = int(row["pair"])
            if p in by_pair:
                for k, v in row.items():
                    by_pair[p].setdefault(k, v)
            else:
                by_pair[p] = dict(row)
    rows = [by_pair[p] for p in sorted(by_pair)]
    if not rows:
        raise SystemExit("no rows")

    # ---- controls -------------------------------------------------------------------------
    ctl = {
        "C2_lstar_matches_cache": all(r.get("C2_lstar_matches_cache") for r in rows),
        "C3_lgt_matches_cache": all(r.get("C3_lgt_matches_cache") for r in rows),
        "frame0_is_seg_free_MEASURED": all(
            r.get("control_frame0_is_seg_free") for r in rows),
        "n_pairs": len(rows),
    }

    out: dict = {
        "schema": "ddm_js1_aggregate.v1",
        "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False,
        "verdict_scope_default": "FORMULATION",
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "own_vehicle_frontier": f"S = {S_LIVE_BEST} @ {BYTES_LIVE_BEST} B [macOS-CPU advisory]",
        "denominators": {
            "gap_S": GAP, "seg_term": SEG_TERM, "rate_term": RATE_TERM,
            "pose_term": POSE_TERM, "d_pose_population": D_POSE_POP,
            "dS_dd_pose_CORRECTED_current_operating_point": DS_DDPOSE_NOW,
            "dS_dd_pose_et1_inherited_PRE_PU2_SUPERSEDED": DS_DDPOSE_ET1,
            "correction_pct": 100.0 * (DS_DDPOSE_NOW / DS_DDPOSE_ET1 - 1.0),
            "gross_S_block16": GROSS_S_BLOCK16, "bytes_block16": BYTES_BLOCK16,
            "rate_cost_block16": RATE_COST_BLOCK16, "breakeven_eta": BREAKEVEN_ETA,
        },
        "controls": ctl,
        "pose_scope_warning": (
            "subset-scoped gate ONLY (m96/sq1 1.6): et1 measured this stratified selection at "
            "0.2692x of population on d_pose; pose deltas here are NOT folded into net S"),
        "arms": {},
    }

    # ---- stage 1 (shared by every arm) -----------------------------------------------------
    s1 = [r["stage1"] for r in rows if "stage1" in r]
    eta_m, eta_sd, eta_n = pool([x["eta_realized"] for x in s1])
    dseg_S = -eta_m * GROSS_S_BLOCK16
    net_seg_rate = dseg_S + RATE_COST_BLOCK16
    # absolute pose mass moved by stage 1, on this subset
    dp_abs = [x["d_pose_after"] - r["d_pose_before"]
              for r, x in zip(rows, s1, strict=True)]
    dpa_m, dpa_sd, _ = pool(dp_abs)
    ratio_m, _, _ = pool([x["d_pose_ratio"] for x in s1])
    out["arms"]["stage1_unconstrained_seg"] = {
        "eta_mean": eta_m, "eta_sd": eta_sd, "n": eta_n,
        "cap_pinned_frac": sum(1 for x in s1 if x["cap_pinned"]) / max(len(s1), 1),
        "clears_breakeven": bool(eta_m > BREAKEVEN_ETA),
        "eta_over_breakeven": eta_m / BREAKEVEN_ETA,
        "delta_S_seg": dseg_S, "rate_cost": RATE_COST_BLOCK16,
        "net_S_seg_plus_rate": net_seg_rate,
        "pct_of_gap_seg_plus_rate": -100.0 * net_seg_rate / GAP,
        "d_pose_ratio_mean_SUBSET": ratio_m,
        "d_pose_abs_delta_mean_SUBSET": dpa_m,
        "d_pose_abs_delta_sd_SUBSET": dpa_sd,
        "pose_breakeven_abs_delta": -net_seg_rate / DS_DDPOSE_NOW,
        "pose_breakeven_ratio_vs_population": (
            1.0 + (-net_seg_rate / DS_DDPOSE_NOW) / D_POSE_POP),
    }

    # ---- stage-2 arms ----------------------------------------------------------------------
    for key, label in (("arm_cprime", "C_PRIME_frame0_pose_repair"),
                       ("arm_ccell", "C_cellconstrained_frame1_repair")):
        a = [(r, r[key]) for r in rows if key in r]
        if not a:
            continue
        eta2_m, eta2_sd, n2 = pool([x["eta_realized"] for _, x in a])
        rep_m, rep_sd, _ = pool([x["repair_fraction_of_damage"] for _, x in a])
        rat_m, _, _ = pool([x["d_pose_ratio_vs_before"] for _, x in a])
        gap_m, gap_sd, _ = pool([x["realization_gap"] for _, x in a])
        dp2 = [x["d_pose_verified_from_camera"] - r["d_pose_before"] for r, x in a]
        dp2_m, dp2_sd, _ = pool(dp2)
        dseg2 = -eta2_m * GROSS_S_BLOCK16
        net2 = dseg2 + RATE_COST_BLOCK16
        out["arms"][label] = {
            "n": n2,
            "seg_exactly_preserved_ALL": all(x["seg_exactly_preserved"] for _, x in a),
            "eta_mean": eta2_m, "eta_sd": eta2_sd,
            "eta_retained_vs_stage1": (eta2_m / eta_m) if eta_m else None,
            "repair_fraction_of_pose_damage_mean": rep_m,
            "repair_fraction_sd": rep_sd,
            "d_pose_ratio_mean_SUBSET": rat_m,
            "d_pose_abs_delta_mean_SUBSET": dp2_m,
            "d_pose_abs_delta_sd_SUBSET": dp2_sd,
            "realization_gap_mean": gap_m, "realization_gap_sd": gap_sd,
            "delta_S_seg": dseg2, "rate_cost_seg_field_only": RATE_COST_BLOCK16,
            "net_S_seg_plus_rate": net2,
            "pct_of_gap_seg_plus_rate": -100.0 * net2 / GAP,
            "pose_breakeven_abs_delta": -net2 / DS_DDPOSE_NOW,
            "pose_breakeven_ratio_vs_population": (
                1.0 + (-net2 / DS_DDPOSE_NOW) / D_POSE_POP),
            "frame0_stream_bytes": "OPEN -- NOT PRICED (see memo); this net excludes it",
        }

    # ---- pose-only CONTROL (isolates unharvested frame_0 headroom from the staging) ---------
    po = [(r, r["arm_poseonly_control"]) for r in rows if "arm_poseonly_control" in r]
    if po:
        rat_po, sd_po, n_po = pool([x["d_pose_ratio_vs_before"] for _, x in po])
        out["arms"]["POSEONLY_control_no_seg_solve"] = {
            "n": n_po, "d_pose_ratio_mean_SUBSET": rat_po, "sd": sd_po,
            "seg_unchanged_ALL": all(x["seg_unchanged_vs_shipped"] for _, x in po),
            "interpretation": (
                "pu2 solved frame_0 on only 6 pairs, so a ratio <1.0 here is headroom the "
                "STAGING did not create.  staging-attributable pose = cprime_ratio - this."),
        }

    # ---- cheap generic-basis carriage arms --------------------------------------------------
    for key in sorted({k for r in rows for k in r if k.startswith("arm_cprime_cheap_dct")}):
        a = [(r, r[key]) for r in rows if key in r]
        rat, sd, n2 = pool([x["d_pose_ratio_vs_before"] for _, x in a])
        vs1, _, _ = pool([x["d_pose_ratio_vs_stage1_damage"] for _, x in a])
        b = a[0][1]
        out["arms"][key] = {
            "n": n2, "k": b["k"],
            "d_pose_ratio_mean_SUBSET": rat, "sd": sd,
            "d_pose_ratio_vs_stage1_damage_mean": vs1,
            "solved_to_all_zero_frac": sum(1 for _, x in a if x["solved_to_all_zero"]) / len(a),
            "counted_bytes_per_pair": b["counted_bytes_per_pair"],
            "counted_bytes_n600": b["counted_bytes_n600"],
            "rate_cost_S_n600": b["rate_cost_S_n600"],
            "note": ("SOLVED WITHIN the generic DCT basis (free under rule 118), not projected "
                     "onto it -- p3v2: the free win is BASIS-ADVERSARIAL"),
        }

    out["per_pair"] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print(f"=== ddm_js1 aggregate  n={len(rows)} ===")
    print(f"breakeven eta {BREAKEVEN_ETA:.5f} | dS/dd_pose CORRECTED {DS_DDPOSE_NOW:.4f} "
          f"(et1 inherited {DS_DDPOSE_ET1} = pre-pu2, +{100*(DS_DDPOSE_NOW/DS_DDPOSE_ET1-1):.1f}%)")
    print(f"controls: {ctl}")
    for k, v in out["arms"].items():
        print(f"\n-- {k}")
        for kk, vv in v.items():
            print(f"     {kk}: {vv}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
