#!/usr/bin/env python
"""ddm_ob1 aggregate -- one ladder, one set of denominators, one break-even.

Scorer-free. Consumes the four ob1 receipts and emits the single table the memo quotes.
Axis: [macOS-CPU advisory] NON-PROMOTABLE. score_claim=False.
pointer 0.1910828242 [contest-CPU] UNMOVED.
"""
from __future__ import annotations

import json
import os

OUT_DIR = "/Volumes/VertigoDataTier/pact/ddm_ob1_20260803"
RATE_PER_BYTE = 25.0 / 37_545_489.0
LIVE_BEST_S = 0.7910689
LIVE_BEST_BYTES = 353_805
GAP = 0.6189279
ETA_PN = 0.5406       # sq1 n=32, pose-neutral (P7 yuv6-null)
ETA_UN = 0.7895       # sq1 n=32, unconstrained (pose-catastrophic)
GP1_STUDENT_CEILING_B = 106_954
GP1_A2_NET_S = -0.039


def load(name):
    with open(os.path.join(OUT_DIR, name)) as f:
        return json.load(f)


def main() -> int:
    A = load("ob1_band_reprice_n600.json")
    B = load("ob1_ordering_ceiling_n600.json")
    C = load("ob1_legal_address_n600.json")
    D = load("ob1_margin_oracle_n600.json")

    flips = A["totals"]["flips_ALL"]
    N = A["totals"]["field_pixels"]
    gross = B["totals"]["gross_dS_at_eta1_full_capture"]

    def hard(ls, dn, r):
        return next(x for x in A["rows"] if x["label_source"] == ls
                and x["dilation"] == dn and x["dilate_r"] == r)

    def soft(src, afeat, pfeat):
        return next(x for x in src["rows"] if x["address_feature"] == afeat
                and x["payload_feature"] == pfeat)

    gp1_pub = hard("gp1_proxy", "gp1_L1", 1)
    actual_hard = hard("actual_Lstar", "gp1_L1", 1)
    sq1_hard = hard("actual_Lstar", "sq1_aniso", 1)
    lstar_soft = soft(B, "d x own x nbr", "H(gt|own,nbr)")
    legal_soft = soft(C, "d_edge x row x grad", "H(gt|X_legal) LEGAL")
    margin_row = next(x for x in D["rows"] if x["address_feature"].startswith("frozen margin ("))
    margin_d_row = next(x for x in D["rows"] if "d(L* boundary)" in x["address_feature"])

    ladder = []

    def add(name, legal, byts, capture, note):
        rate = byts * RATE_PER_BYTE
        ladder.append({
            "rung": name, "receiver_legal": legal, "bytes": byts, "capture": capture,
            "rate_cost_S": rate,
            "net_S_at_eta_pose_neutral": rate - ETA_PN * capture * gross,
            "net_S_at_eta_unconstrained": rate - ETA_UN * capture * gross,
            "break_even_eta": rate / (capture * gross),
            "pct_of_gap_at_eta_pn": -(rate - ETA_PN * capture * gross) / GAP * 100.0,
            "note": note,
        })

    add("gp1 A3 AS PUBLISHED (proxy label field, gp1 dilation, r=1)", False,
        gp1_pub["ALL_total_bytes"], gp1_pub["capture_rate_ALL"],
        "reproduces gp1's 367,523 B / 0.972639 capture; label field agrees with GT "
        f"{A['proxy_field_agreement']['agree_with_GT_frac']:.4%} -- effectively a GT band")
    add("sq1's band (actual L*, sq1 ANISOTROPIC dilation, r=1)", False,
        sq1_hard["ALL_total_bytes"], sq1_hard["capture_rate_ALL"],
        "reproduces sq1's n=32 capture 0.8668 at n600; 2.20x gp1's band area")
    add("CORRECTED hard band (actual L*, gp1 dilation, r=1)", False,
        actual_hard["ALL_total_bytes"], actual_hard["capture_rate_ALL"],
        "gp1's own convention on the decoder's real label field -- F4 fires (<90%)")
    add("SOFT model on L* (d x own-class x edge)", False,
        lstar_soft["total_bytes"], 1.0,
        "optimal coder, full capture; still needs the 73 MB SegNet to compute L*")
    add("SOFT model, RECEIVER-LEGAL (decoded RGB only)", True,
        legal_soft["total_bytes"], 1.0,
        "the honest free floor: no label field, no scorer weights")
    add("ORACLE: frozen SegNet margin (illegal ceiling)", False,
        margin_row["total_bytes_with_Lstar_payload"], 1.0,
        "payload charged at the ILLEGAL L*-conditioned 0.2633 b/flip -- every benefit given")
    add("ORACLE: frozen margin x d(L* boundary) (illegal ceiling)", False,
        margin_d_row["total_bytes_with_Lstar_payload"], 1.0,
        "the tightest measured address; the absolute ceiling for any student")

    # ---- the student break-even, from the LEGAL floor -------------------------------------
    legal_B = legal_soft["total_bytes"]
    budget_B = ETA_PN * gross / RATE_PER_BYTE                 # bytes affordable at measured eta
    need_saving_B = legal_B - budget_B                        # to reach net = 0
    need_for_a2_B = legal_B - (GP1_A2_NET_S + ETA_PN * gross) / RATE_PER_BYTE
    oracle_B = margin_d_row["total_bytes_with_Lstar_payload"]
    max_saving_B = legal_B - oracle_B                         # ceiling: legal -> oracle

    student = {
        "legal_floor_bytes": legal_B,
        "affordable_bytes_at_measured_eta_0.5406": budget_B,
        "saving_needed_to_break_even_B": need_saving_B,
        "saving_needed_to_match_gp1_A2_-0.039S_B": need_for_a2_B,
        "oracle_ceiling_bytes": oracle_B,
        "max_saving_any_student_can_deliver_B": max_saving_B,
        "break_even_is_reachable": bool(max_saving_B > need_saving_B),
        "fraction_of_the_oracle_gap_a_student_must_capture_to_break_even":
            need_saving_B / max_saving_B if max_saving_B > 0 else None,
        "fraction_of_the_oracle_gap_needed_to_match_A2":
            need_for_a2_B / max_saving_B if max_saving_B > 0 else None,
        "gp1_published_student_ceiling_B": GP1_STUDENT_CEILING_B,
        "true_student_value_over_gp1_x": max_saving_B / GP1_STUDENT_CEILING_B,
        "bits_per_field_px_of_extra_MI_to_break_even": need_saving_B * 8 / N,
        "bits_per_flip_of_extra_MI_to_break_even": need_saving_B * 8 / flips,
    }

    # ---- eta sensitivity: does the realizer work AWAY from the boundary? -------------------
    cap_r1 = actual_hard["capture_rate_ALL"]
    sens = []
    for name, eff in (
        ("eta 0.5406 uniform over the whole field", ETA_PN),
        (f"eta 0.5406 only within r=1 ({cap_r1:.4f} of flips), 0 beyond", ETA_PN * cap_r1),
        ("eta 0.5406 within r=1, HALF beyond", ETA_PN * (cap_r1 + 0.5 * (1 - cap_r1))),
    ):
        for rung, byts in (("LEGAL floor", legal_B), ("L* soft (illegal)",
                                                      lstar_soft["total_bytes"])):
            net = byts * RATE_PER_BYTE - eff * gross
            sens.append({"eta_assumption": name, "rung": rung, "effective_eta": eff,
                         "net_S": net, "live": bool(net < 0),
                         "pct_of_gap": -net / GAP * 100.0})

    out = {
        "schema": "ddm_ob1_aggregate.v1",
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "baseline": {"live_best_S": LIVE_BEST_S, "live_best_bytes": LIVE_BEST_BYTES,
                     "gap_to_PR130_bar": GAP,
                     "one_pct_of_gap_S": GAP / 100.0,
                     "one_pct_of_gap_bytes": (GAP / 100.0) / RATE_PER_BYTE,
                     "flips": flips, "gross_dS_full_capture_eta1": gross},
        "eta_used": {"pose_neutral_n32": ETA_PN, "unconstrained_n32": ETA_UN,
                     "scope": "sq1 n=32, measured on sq1's ANISOTROPIC r=1 band with the "
                              "solved-paint realizer; transfer to any other address object "
                              "is UNMEASURED and is this unit's largest carried assumption"},
        "controls": {
            "d_seg_reproduced": A["totals"]["d_seg_reproduced"],
            "payload_H_gt_given_rendered": A["totals"]["payload_H_gt_given_rendered_bits_per_flip"],
            "gp1_A3_reproduced_bytes": gp1_pub["ALL_total_bytes"],
            "gp1_A3_published_bytes": 367_523,
            "segnet_argmax_matches_cache_all_pairs":
                D["positive_controls"]["C1_argmax_matches_cx1_cache_all_pairs"],
            "segnet_argmax_failing_pairs": D["positive_controls"]["C1_failing_pairs"],
        },
        "ladder": ladder,
        "student_break_even": student,
        "eta_sensitivity": sens,
    }
    path = os.path.join(OUT_DIR, "ob1_aggregate.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    print(f"wrote {path}\n")
    print("THE LADDER  (n600; net at sq1's measured pose-neutral eta 0.5406)")
    print(f"{'rung':<58s} {'legal':<6s} {'bytes':>9s} {'cap':>7s} {'net S':>9s} "
          f"{'%gap':>7s} {'be_eta':>7s}")
    for r in ladder:
        print(f"{r['rung']:<58s} {r['receiver_legal']!s:<6s} {r['bytes']:9,.0f} "
              f"{r['capture']:7.4f} {r['net_S_at_eta_pose_neutral']:+9.5f} "
              f"{r['pct_of_gap_at_eta_pn']:+7.2f} {r['break_even_eta']:7.4f}")
    s = student
    print("\nSTUDENT BREAK-EVEN (from the LEGAL floor, which is the only honest start)")
    print(f"  legal floor                              {s['legal_floor_bytes']:>10,.0f} B")
    print(f"  affordable at measured eta 0.5406        {s['affordable_bytes_at_measured_eta_0.5406']:>10,.0f} B")
    print(f"  -> must save to break even               {s['saving_needed_to_break_even_B']:>10,.0f} B")
    print(f"  -> must save to match gp1's A2 -0.039 S  {s['saving_needed_to_match_gp1_A2_-0.039S_B']:>10,.0f} B")
    print(f"  oracle ceiling (frozen margin)           {s['oracle_ceiling_bytes']:>10,.0f} B")
    print(f"  MAX any student can ever save            {s['max_saving_any_student_can_deliver_B']:>10,.0f} B")
    print(f"  break-even reachable at all?             {s['break_even_is_reachable']}")
    if s["fraction_of_the_oracle_gap_a_student_must_capture_to_break_even"] is not None:
        print(f"  fraction of the oracle gap needed:  break-even "
              f"{s['fraction_of_the_oracle_gap_a_student_must_capture_to_break_even']:.1%}"
              f"   match-A2 {s['fraction_of_the_oracle_gap_needed_to_match_A2']:.1%}")
    print(f"  gp1 published the student ceiling as {GP1_STUDENT_CEILING_B:,} B; true value is "
          f"{s['true_student_value_over_gp1_x']:.2f}x that")
    print("\nETA SENSITIVITY (the largest carried assumption)")
    for x in sens:
        print(f"  {x['eta_assumption']:<52s} {x['rung']:<18s} eff_eta {x['effective_eta']:.4f}  "
              f"net {x['net_S']:+.5f}  {'LIVE' if x['live'] else 'DEAD'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
