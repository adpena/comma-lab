#!/usr/bin/env python3
"""ddm_pn2 -- does realizing a seg edit INSIDE PoseNet's null space remove the pose tax?

THE QUESTION THIS PAYS
----------------------
Every seg-edit candidate this campaign refused died on POSE, not on seg: qs4 (seg -1.441e-5 real,
pose +2.396e-4), sf1 (pose-dominated), rt2's own de-blur (seg -16 flips at alpha=0.25, pose
x2.27 -> +0.0237 S net).  rt2 then measured that PoseNet's null space is EXACTLY 6 of 12 DOF per
2x2 block (50% of the scorer-resolution RGB field) and that projecting onto it attenuates the
pose cost 44x at matched perturbation.  Q3 (2026-07-31) had already measured the same kernel from
the other side: a frame_1 yuv6-null perturbation reaches SegNet at amplitude 6.0/255 while
PoseNet sees 5.7e-14 -- EXACTLY zero, a linear kernel, at any amplitude.

Q3 left two limits open.  (1) reaching SegNet's input is not moving its argmax; (2) realizability
at camera-res uint8 was UNMEASURED.  rt2 closed (2) with an exact preimage.  This module answers
(1) -- the SEG REACH -- and it answers it with the arm Q3's own law demands.

Q3, verbatim: "a reader holding only the dimension count would design the Q3 probe ISOTROPIC --
the generic control our own standing law forbids -- and a null result would close a FAMILY off a
rung-1 design.  Perturb along rung-3-ranked directions; isotropic is the CONTROL arm, never the
treatment."  So:

  CONTROL   (undirected): rt2's de-blur ladder, pose-null projected, confined to rt1's ring-0
                          support.  `ddm_rt2_deblur_ladder.py --support ... --pose-null`.
  TREATMENT (directed)  : rt1's eta-gate solver, same ring-0 support, run in BOTH modes on the
                          SAME seeded-random pairs -- `--mode null` (projected) vs `--mode free`
                          (unprojected).  That matched A/B is the seg-reach measurement, and it
                          is the one thing neither rt1 nor rt2 ran: rt1's support ladder was
                          unconstrained and its gate was projected, but on DIFFERENT pairs.

WHAT THIS MODULE DOES
---------------------
Aggregation and arithmetic only -- it runs no scorer.  It joins the two eta-gate row sets on
`pair`, reports the matched seg/pose deltas with a paired sign test, and prices the joint dS at
n600 scale under both channel framings (rt1 describe-everything, sr1 waterfill).

THE POSE-INSTRUMENT OFFSET, HANDLED EXPLICITLY
----------------------------------------------
The advisory CPU pose instrument does not read the contest level: rt1's 12 pairs carry mean
d_pose 1.30107e-04 against hv1's contest-CUDA n600 aggregate 6.885643e-06 -- a factor 18.90,
which reproduces rn1's independently measured ~18.2x instrument discrepancy.  So an ABSOLUTE
dS_pose computed off the advisory base overstates the contest cost by ~sqrt(18.9) = 4.3x.  Both
conversions are emitted, labelled, and neither is presented as the other:

  dS_pose[contest-scaled] = (sqrt(ratio) - 1) * sqrt(10 * d_pose_n600_contest)   <- DERIVED
  dS_pose[advisory-absolute] = sqrt(10*after) - sqrt(10*before)                  <- what the
                                                                                    instrument
                                                                                    literally read
Ratios survive a multiplicative offset; absolutes do not.  Only the ratio is carried forward.

axis: [macOS-CPU advisory] -- NEVER a score.  score_claim=false.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# exact contest arithmetic, reused not re-derived (rt1 s4 / sr1 s4 pins)
S_PER_FLIP = 100.0 / 117_964_800          # 8.477105034722222e-07
S_PER_BYTE = 25.0 / 37_545_489            # 6.658589531221714e-07
SEG_MARKET_BAR_B_PER_FLIP = S_PER_FLIP / S_PER_BYTE   # 1.273108 B per scored flip

# hv1 ep0634 frontier [contest-CUDA T4 n600]
BASE_S = 0.15959729295498598
GAP_TO_015 = 0.15 - BASE_S                # -0.0095973
D_POSE_N600_CONTEST = 6.885642960696714e-06   # sr1 SR1_A1POSE.json hv1_d_pose_reference
POSE_CONTRIB = math.sqrt(10.0 * D_POSE_N600_CONTEST)   # 0.00829797

# the object being priced (rt1 s3.3 / s5, sr1 s4) -- retained figures, not re-derived
BAND_PX = 2_551_464
BAND_FLIPS = 34_666
RT1_CHANNEL_B = 33_235                    # 32,270 real CABAC (M7) + 965 target class
RT1_M7_BITS_PER_FLIP = 7.447066289736341
RT1_IID_BITS_PER_FLIP = 7.635608377084175
SR1_WF_FLIPS = 6_512                      # waterfill selection at eta 0.6235, 41 cells
SR1_WF_BYTES = 4276.171156196069          # IDEAL conditional entropy, NOT a real coder
ETA_BAR_RT1 = 0.753                       # rt1's describe-everything break-even
ETA_BAR_SR1_GUARDED = 0.3871              # sr1's waterfill supplier margin, >=500-px guard


def load_rows(path: Path) -> dict[int, dict]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {int(r["pair"]): r for r in rows}


def pooled_eta(rows: list[dict]) -> float:
    d = sum(r["n_described_ring0"] for r in rows)
    return sum(r["flips_before"] - r["flips_after"] for r in rows) / d if d else float("nan")


def pose_ratio(rows: list[dict]) -> float:
    """scorer convention: ratio of MEANS of d_pose, never a mean of per-pair ratios (rt1 s6.2b)."""
    b = sum(r["d_pose_before"] for r in rows) / len(rows)
    a = sum(r["d_pose_after"] for r in rows) / len(rows)
    return a / b if b else float("nan")


def sign_test_p(n_pos: int, n: int) -> float:
    """Exact two-sided binomial sign test against p=0.5 (no scipy dependency)."""
    if n == 0:
        return float("nan")
    k = min(n_pos, n - n_pos)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def channel_dS(eta: float, flips_described: float, channel_bytes: float,
               pose_ratio_val: float) -> dict:
    seg = -eta * flips_described * S_PER_FLIP
    rate = channel_bytes * S_PER_BYTE
    pose = (math.sqrt(pose_ratio_val) - 1.0) * POSE_CONTRIB
    net = seg + rate + pose
    realized = eta * flips_described
    return {
        "eta": eta,
        "flips_described": flips_described,
        "flips_realized": realized,
        "channel_bytes": channel_bytes,
        "bytes_per_realized_flip": channel_bytes / realized if realized else float("inf"),
        "seg_market_bar_B_per_flip": SEG_MARKET_BAR_B_PER_FLIP,
        "dS_seg": seg,
        "dS_rate": rate,
        "dS_pose_contest_scaled": pose,
        "dS_net": net,
        "share_of_gap_closed": (-net / -GAP_TO_015) if net < 0 else 0.0,
        "verdict": "SUPPLIER" if net < 0 else "NON-SUPPLIER",
    }


def run(args: argparse.Namespace) -> int:
    null_rows = load_rows(args.null_rows)
    free_rows = load_rows(args.free_rows)
    shared = sorted(set(null_rows) & set(free_rows))
    if not shared:
        raise SystemExit("no matched pairs between the two row sets -- the A/B is not matched")

    n_list = [null_rows[p] for p in shared]
    f_list = [free_rows[p] for p in shared]
    eta_n, eta_f = pooled_eta(n_list), pooled_eta(f_list)
    pr_n, pr_f = pose_ratio(n_list), pose_ratio(f_list)
    d_eta = [null_rows[p]["eta_net"] - free_rows[p]["eta_net"] for p in shared]
    n_pos = sum(1 for x in d_eta if x > 0)

    # THREE-WAY DECOMPOSITION.  `null` mode snaps the edit support to whole 2x2 blocks
    # (snap_tax ~1.77x) as well as projecting, so null-vs-free confounds the PROJECTION with a
    # SUPPORT-SIZE change that rt1 s6.1 measured at ~0.65 eta on its own.  The `free + snapped`
    # arm splits the two: free -> free_snap is the SNAP alone, free_snap -> null is the
    # PROJECTION alone.
    decomposition = None
    if args.snap_rows is not None and args.snap_rows.exists():
        snap_rows = load_rows(args.snap_rows)
        tri = sorted(set(shared) & set(snap_rows))
        if tri:
            s_list = [snap_rows[p] for p in tri]
            n3 = [null_rows[p] for p in tri]
            f3 = [free_rows[p] for p in tri]
            e_f, e_s, e_n = pooled_eta(f3), pooled_eta(s_list), pooled_eta(n3)
            p_f, p_s, p_n = pose_ratio(f3), pose_ratio(s_list), pose_ratio(n3)
            total = e_n - e_f
            decomposition = {
                "pairs": tri, "n": len(tri),
                "pooled_eta": {"free": e_f, "free_snapped": e_s, "null_projected": e_n},
                "eta_attributable_to_snap": e_s - e_f,
                "eta_attributable_to_projection": e_n - e_s,
                "eta_total": total,
                "projection_share_of_eta_gain": ((e_n - e_s) / total) if total else float("nan"),
                "pose_ratio": {"free": p_f, "free_snapped": p_s, "null_projected": p_n},
                "pose_reduction_from_snap": (p_f / p_s) if p_s else float("nan"),
                "pose_reduction_from_projection": (p_s / p_n) if p_n else float("nan"),
                "note": ("free -> free_snapped isolates the 2x2 support snap; free_snapped -> "
                         "null_projected isolates the pose-null projection at matched support"),
            }

    matched = {
        "pairs": shared,
        "n_matched": len(shared),
        "three_way_decomposition": decomposition,
        "pooled_eta_null": eta_n,
        "pooled_eta_free": eta_f,
        "pooled_eta_delta_null_minus_free": eta_n - eta_f,
        "per_pair_delta_eta": {str(p): d for p, d in zip(shared, d_eta, strict=True)},
        "pairs_where_projection_helped_seg": n_pos,
        "sign_test_two_sided_p": sign_test_p(n_pos, len(shared)),
        "pose_ratio_null_scorer_convention": pr_n,
        "pose_ratio_free_scorer_convention": pr_f,
        "pose_ratio_free_over_null": pr_f / pr_n if pr_n else float("nan"),
        "note": ("eta_net = (flips_before - flips_after) / n_described_ring0, whole-frame "
                 "accounted; pose ratio is the ratio of MEANS per rt1 s6.2b"),
    }

    # full-scale arithmetic at the MEASURED projected eta
    full = {
        "rt1_describe_everything": channel_dS(eta_n, BAND_FLIPS, RT1_CHANNEL_B, pr_n),
        "sr1_waterfill_ideal_entropy": channel_dS(eta_n, SR1_WF_FLIPS, SR1_WF_BYTES, pr_n),
        "sr1_waterfill_pose_neutral": channel_dS(eta_n, SR1_WF_FLIPS, SR1_WF_BYTES, 1.0),
    }
    # how much worse than sr1's IDEAL entropy a REAL coder may be and still break even
    realized = eta_n * SR1_WF_FLIPS
    breakeven_bytes = realized * SEG_MARKET_BAR_B_PER_FLIP
    full["rate_headroom"] = {
        "sr1_ideal_bytes": SR1_WF_BYTES,
        "breakeven_bytes_at_measured_eta": breakeven_bytes,
        "real_coder_headroom_frac": breakeven_bytes / SR1_WF_BYTES - 1.0,
        "sr1_ideal_bits_per_described_flip": SR1_WF_BYTES * 8 / SR1_WF_FLIPS,
        "rt1_M7_real_bits_per_flip_full_band": RT1_M7_BITS_PER_FLIP,
        "rt1_iid_bits_per_flip_full_band": RT1_IID_BITS_PER_FLIP,
        "rt1_M7_gain_over_iid_frac": 1.0 - RT1_M7_BITS_PER_FLIP / RT1_IID_BITS_PER_FLIP,
        "STATUS": ("sr1's waterfill bytes are an IDEAL conditional-entropy ceiling with a 148 B "
                   "model cost, NOT a real coder.  rt1's M7 CABAC is the only REAL coder measured "
                   "on this object and it beat i.i.d. by 2.47% on the FULL band.  Coding the "
                   "waterfilled (denser, 41-cell) support with a real coder is UNMEASURED and is "
                   "the named owed row."),
        "bar_note": f"seg market bar {SEG_MARKET_BAR_B_PER_FLIP:.6f} B per scored flip",
    }

    out = {
        "schema": "ddm_pn2_posenull_seg_channel.v1",
        "arm": "ddm_pn2",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "pointer_moved": False,
        "own_vehicle_frontier": {"S": BASE_S, "gap_to_0p15": GAP_TO_015,
                                 "d_pose_n600_contest": D_POSE_N600_CONTEST},
        "verdict_scope": ("INSTANCE on the hv1 ep0634 vehicle at the measured n; the matched A/B "
                          "is a SCOPE reduction (seeded-random pairs, never a prefix, m96)"),
        "matched_ab": matched,
        "full_scale_arithmetic": full,
        "exchange": {"S_per_flip": S_PER_FLIP, "S_per_byte": S_PER_BYTE,
                     "seg_market_bar_B_per_flip": SEG_MARKET_BAR_B_PER_FLIP},
        "bars": {"rt1_describe_everything": ETA_BAR_RT1,
                 "sr1_waterfill_guarded_supplier_margin": ETA_BAR_SR1_GUARDED},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nwrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    R1 = Path("/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--null-rows", type=Path,
                    default=R1 / "eta_gate_null" / "ETA_GATE_ROWS.jsonl")
    ap.add_argument("--free-rows", type=Path,
                    default=Path("/Volumes/APDataStore/pact/ddm_pn2/eta_gate_free_n12"
                                 "/ETA_GATE_ROWS.jsonl"))
    ap.add_argument("--snap-rows", type=Path,
                    default=Path("/Volumes/APDataStore/pact/ddm_pn2/eta_gate_free_snapped_n12"
                                 "/ETA_GATE_ROWS.jsonl"),
                    help="free-mode rows run WITH the 2x2 support snap; enables the three-way "
                         "decomposition that separates the snap from the projection")
    ap.add_argument("--out", type=Path,
                    default=Path("/Volumes/APDataStore/pact/ddm_pn2/PN2_VERDICT.json"))
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
