"""ddm_rc4 - fold the MEASURED seg amplification A(u) into the exact rate ladder.

Emits the full three-component pointer arithmetic for every measured threshold.
The charter's pure-rate byte equivalence is deliberately NOT used: token drop
moves the decoded frame_1 field, so seg (and pose) move with it.
"""

from __future__ import annotations

import json
from pathlib import Path

STORE = Path("/Volumes/APDataStore/pact/ddm_rc4_rung4_token_drop_20260816")

S_BASE = 0.15959729295498598
SEG_TERM = 0.029611
POSE_TERM = 0.0082945765
S_PER_BYTE = 25.0 / 37_545_489
S_PER_SEG_FLIP = 100.0 / (600 * 384 * 512)
POSE_MARGINAL = 5.0 / POSE_TERM  # linearisation at the AUTHORITY baseline only
D_POSE_AUTH = POSE_TERM**2 / 10.0
TARGET = 0.15


def d_pose_headroom_exact(headroom_S: float) -> float:
    """Largest ABSOLUTE d_pose rise a given S headroom can absorb.

    Exact inverse of dS = sqrt(10*(d0+x)) - sqrt(10*d0); never the linear
    marginal, and never evaluated at a floor-inflated baseline (ddm_pi2).
    """
    import math

    if headroom_S <= 0:
        return 0.0
    return (headroom_S + math.sqrt(10.0 * D_POSE_AUTH)) ** 2 / 10.0 - D_POSE_AUTH


def main() -> int:
    ladder = json.loads((STORE / "DROP_LADDER.json").read_text())["ladder"]
    amp = json.loads((STORE / "AMPLIFICATION.json").read_text())
    by_u = {round(r["u"], 6): r for r in ladder}

    pose = None
    pose_path = STORE / "POSE_LEG.json"
    if pose_path.exists():
        pose = {round(r["u"], 6): r for r in json.loads(pose_path.read_text())["results"]}

    rows = []
    for m in amp["results"]:
        u = round(m["u"], 6)
        lad = by_u[u]
        A = m["A_net_seg_flips_per_token_flip"]
        d_rate = lad["delta_S_rate"]
        d_seg = A * lad["token_flips"] * S_PER_SEG_FLIP
        row = {
            "u": u,
            "p_max_threshold": lad["p_max_threshold"],
            "bytes_saved_n600_exact": lad["bytes_saved"],
            "token_flips_n600_exact": lad["token_flips"],
            "A_measured_advisory": A,
            "sample_pairs": m["sample_pairs"],
            "sample_B_beneficial": m["beneficial_B"],
            "sample_H_harmful": m["harmful_H"],
            "sample_W_wrong_to_wrong": m["wrong_to_wrong_W"],
            "delta_S_rate": d_rate,
            "delta_S_seg": d_seg,
            "net_S_rate_plus_seg": d_rate + d_seg,
            "S_after_rate_plus_seg": S_BASE + d_rate + d_seg,
            "pose_headroom_S": -(d_rate + d_seg),
            "pose_headroom_d_pose_exact": d_pose_headroom_exact(-(d_rate + d_seg)),
            "pose_headroom_multiple_of_base": d_pose_headroom_exact(-(d_rate + d_seg))
            / D_POSE_AUTH,
        }
        if pose and u in pose:
            p = pose[u]
            row["delta_d_pose_absolute_advisory"] = p["delta_d_pose_absolute"]
            row["delta_S_pose_at_authority_baseline"] = p[
                "delta_S_pose_at_authority_baseline"
            ]
            row["net_S_all_three"] = (
                d_rate + d_seg + p["delta_S_pose_at_authority_baseline"]
            )
            row["S_after_all_three"] = S_BASE + row["net_S_all_three"]
        rows.append(row)

    best = min(rows, key=lambda r: r["net_S_rate_plus_seg"])
    out = {
        "arm": "ddm_rc4",
        "stage": "joint_verdict",
        "base": {"S": S_BASE, "archive_bytes": 182_759, "gap_to_target": S_BASE - TARGET},
        "axis": "rate EXACT; seg [macOS-CPU advisory, stratified-random n=120]; "
                "pose = ABSOLUTE advisory delta priced at the AUTHORITY baseline "
                "(ddm_pi2 additive-floor rule; ratios forbidden)",
        "score_claim": False,
        "promotable": False,
        "rows": rows,
        "best_measured_rung": best,
        "crosses_sub_0_15_alone": bool(best["S_after_rate_plus_seg"] < TARGET),
    }
    (STORE / "JOINT_VERDICT.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(f"base S {S_BASE:.17f}  gap to 0.15 {S_BASE-TARGET:.9f}\n")
    hdr = f"{'p_max>=':>13}{'bytes':>9}{'tokflip':>9}{'A':>8}{'dS_rate':>12}{'dS_seg':>12}{'net':>12}{'S_after':>11}"
    print(hdr)
    for r in rows:
        print(f"{r['p_max_threshold']:>13.9f}{r['bytes_saved_n600_exact']:>9.0f}"
              f"{r['token_flips_n600_exact']:>9,}{r['A_measured_advisory']:>8.4f}"
              f"{r['delta_S_rate']:>12.3e}{r['delta_S_seg']:>12.3e}"
              f"{r['net_S_rate_plus_seg']:>12.3e}{r['S_after_rate_plus_seg']:>11.7f}")
    print(f"\nBEST measured rung: p_max>={best['p_max_threshold']:.9f}  "
          f"{best['bytes_saved_n600_exact']:.0f} B  net {best['net_S_rate_plus_seg']:.4e}  "
          f"S -> {best['S_after_rate_plus_seg']:.7f}")
    print(f"pose headroom at that rung: d_pose may rise by "
          f"{best['pose_headroom_d_pose_exact']:.3e} absolute "
          f"({best['pose_headroom_multiple_of_base']:.2f}x base) before the rung goes flat")
    print(f"crosses sub-0.15 alone: {out['crosses_sub_0_15_alone']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
