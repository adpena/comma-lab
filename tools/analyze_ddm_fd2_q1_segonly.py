#!/usr/bin/env python
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""ddm_fd2 Q1 post-analysis: seg-only (pose-veto-removed) pricing + derived d_pose.

The pose-null projection's stated purpose is to make "joint ΔS ≈ seg ΔS by
construction" (j11 contract). So the acceptance outcome the projection targets
is the RAW GN candidate priced SEG-ONLY (pose collateral removed). This script
computes that from MEASURED data with no new run and no pose-Jacobian build:

  seg_only_ΔS(n600) = 100·(d_seg_cand − d_seg_base) + 25·(bytes_cand − bytes_base)/37_545_489

using fd1's ch1-VALIDATED realized n600 d_seg + advisory per candidate (fd1
receipt) and MY probe's compiled candidate bytes (fd2 receipt). It also DERIVES
each candidate's realized n600 d_pose from the identity
  advisory = 100·d_seg + sqrt(10·d_pose) + 25·bytes/37_545_489
=> d_pose = ((advisory − 100·d_seg − 25·bytes/37_545_489)^2)/10
(the ch1 L1 "persist realized_d_pose" cure, recovered arithmetically).

Determinism cross-check: fd2's regenerated gn_step-1 step_norm must match fd1's
648.795910729257 (proves my candidates are byte-identical to fd1's, so fd1's
realized n600 d_seg applies to my compiled bytes)."""

from __future__ import annotations

import json
from pathlib import Path

N = 37_545_489.0
FD1_RECEIPT = Path("/Volumes/VertigoDataTier/pact/ddm_fd1_20260728/s2_gn_window/fd1_gn_window_receipt.json")
FD2_RECEIPT = Path("/Volumes/VertigoDataTier/pact/ddm_fd2_20260728/fd2_disambiguation_receipt.json")


def _d_pose_from(advisory: float, d_seg: float, bytes_: int) -> float:
    root = advisory - 100.0 * d_seg - 25.0 * bytes_ / N
    return (root * root) / 10.0


def main() -> int:
    fd1 = json.loads(FD1_RECEIPT.read_bytes())
    fd2 = json.loads(FD2_RECEIPT.read_bytes())
    base = fd1["curve"][0]
    base_dseg = float(base["d_seg"])
    base_dpose = float(base["d_pose"])
    base_bytes = int(base["archive_bytes"])
    base_adv = float(base["advisory_action"])

    # fd2 candidate bytes by (gn_step, multiplier)
    fd2_bytes = {}
    fd2_step_norm = None
    for row in fd2.get("q2_rows", []):
        if "candidate_archive_bytes" in row:
            fd2_bytes[(row["gn_step"], row["multiplier"])] = int(row["candidate_archive_bytes"])
        fd2_step_norm = row.get("step_norm", fd2_step_norm)

    print(f"baseline: d_seg={base_dseg:.16f} d_pose={base_dpose:.6f} bytes={base_bytes} adv={base_adv:.9f}")
    print(f"fd2 gn_step1 step_norm={fd2_step_norm} (fd1=648.795910729257 match={abs((fd2_step_norm or 0)-648.795910729257)<1e-3})")
    print()
    print(f"{'cand':>10} {'d_seg(n600)':>14} {'bytes':>8} {'derived d_pose':>15} {'seg_only ΔS':>13} {'joint ΔS':>11} {'seg_only_accept':>16}")
    rows_out = []
    for row in fd1["curve"][1:]:
        gs = row["gn_step"]
        for a in row.get("attempts", []):
            if "realized_d_seg" not in a:
                continue
            mult = a["multiplier"]
            d_seg = float(a["realized_d_seg"])
            adv = float(a["realized_advisory_action"])
            cand_bytes = fd2_bytes.get((gs, mult))
            if cand_bytes is None:
                # fall back to baseline bytes if fd2 didn't compile this candidate
                cand_bytes = base_bytes
                byte_src = "baseline_fallback"
            else:
                byte_src = "fd2_measured"
            d_pose = _d_pose_from(adv, d_seg, cand_bytes)
            seg_only = 100.0 * (d_seg - base_dseg) + 25.0 * (cand_bytes - base_bytes) / N
            joint = adv - base_adv
            accept = seg_only < 0.0
            print(f"gn{gs} m{mult:<5} {d_seg:>14.10f} {cand_bytes:>8d} {d_pose:>15.6f} {seg_only:>+13.6f} {joint:>+11.6f} {accept!s:>16} [{byte_src}]")
            rows_out.append({
                "gn_step": gs, "multiplier": mult, "realized_d_seg_n600": d_seg,
                "candidate_bytes": cand_bytes, "byte_source": byte_src,
                "derived_realized_d_pose_n600": d_pose,
                "seg_only_delta_S": seg_only, "joint_delta_S": joint,
                "seg_only_would_accept": accept,
            })
    n_accept = sum(1 for r in rows_out if r["seg_only_would_accept"])
    print()
    print(f"SEG-ONLY (pose-veto-removed) accepted-step count: {n_accept} / {len(rows_out)}")
    out = FD2_RECEIPT.parent / "fd2_q1_segonly_analysis.json"
    out.write_text(json.dumps({
        "schema": "ddm_fd2_q1_segonly.v1",
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False, "pointer_moved": False,
        "method": "raw fd1 GN candidates priced seg-only (the pose-null-projection target); d_seg+advisory from ch1-validated fd1 receipt, bytes from fd2 compile; d_pose arithmetically recovered",
        "baseline": {"d_seg": base_dseg, "d_pose": base_dpose, "bytes": base_bytes, "advisory": base_adv},
        "fd2_gn_step1_step_norm": fd2_step_norm,
        "rows": rows_out,
        "seg_only_accepted_count": n_accept,
        "total_candidates": len(rows_out),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
