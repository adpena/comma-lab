#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DDM-FC1 STAGE 5 -- COMPOSE the MEASURED stream bytes into the S arithmetic (fork decision).

NO byte-closed evaluate.py row is fired: the near-solved d_seg (1.52e-4) requires the REALIZATION
operator (label -> pixel flip) which is UNBUILT (the master-thesis crux); a copy-base byte-close
would only reproduce the rate-wall regime (worse than r6cal). Per the charter fork, this NAMES the
binding stream with its MEASURED bytes and lays out the exact gap arithmetic under layered best-cases.

Distortion legs:
  d_seg  = near-solved 1.52e-4 (byte FLOOR of the correction stream MEASURED here; realization UNBUILT)
  d_pose = banked R1 dxi 0.001610 (settled, stored-target sidecar ~875 B; contrib 0.127)
Rate legs (all MEASURED n600, real coders):
  frame_0 carrier  = stage3 WebP-Q1 (2,695,020 B) .. the #1 binding stream
  support geometry = stage2 LZMA packbits (421,366 B); contour best-case ~142,220 B (NAMED, unbuilt)
  labels           = stage2 constriction (41,392 B) + table 828 B  [the STAGE-1 tier-mover, REAL]
  pose sidecar     = ~875 B

`[macOS-CPU advisory]` -- arithmetic over MEASURED bytes; no score claim; pointer UNMOVED 0.19108.
"""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

N_REF = 37_545_489
D_SEG_NEARSOLVED = 1.52e-4          # near-solved point (charter/A15) -> 100x = 0.0152
D_POSE_BANKED = 0.001610            # R1 dxi banked -> contrib sqrt(10*.) = 0.127
D_POSE_SOLVED = 1.02e-4             # solved pose (exists only at 291 MB solve) -> contrib 0.0319
POSE_SIDECAR_BYTES = 875
CONTOUR_BEST_CASE_SUPPORT = 142_220  # r2s named ~100-200 KB contour target (UNBUILT)
BAR = 0.172
AIM = 0.15


def S(d_seg: float, d_pose: float, total_bytes: int) -> dict:
    seg = 100.0 * d_seg
    pose = sqrt(10.0 * d_pose)
    rate = 25.0 * total_bytes / N_REF
    return {"S": seg + pose + rate, "seg_term": seg, "pose_term": pose, "rate_term": rate, "bytes": total_bytes}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage2", type=Path, required=True)
    ap.add_argument("--stage3", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    s2 = json.loads(args.stage2.read_text())
    s3 = json.loads(args.stage3.read_text())

    labels_b = s2["labels"]["label_coded_bytes"] + s2["labels"]["model_table_bytes_lzma"]
    support_b = s2["support"]["support_coded_bytes_lzma"]
    frame0_q1 = next(r for r in s3["crush_curve"] if r["webp_quality"] == 1)["total_bytes"]

    correction_measured = labels_b + support_b
    correction_contour = labels_b + CONTOUR_BEST_CASE_SUPPORT

    scenarios = {
        "A_full_measured_streams": S(
            D_SEG_NEARSOLVED, D_POSE_BANKED, frame0_q1 + correction_measured + POSE_SIDECAR_BYTES),
        "B_frame0_free_correction_measured": S(
            D_SEG_NEARSOLVED, D_POSE_BANKED, correction_measured + POSE_SIDECAR_BYTES),
        "C_frame0_free_correction_contour_bestcase": S(
            D_SEG_NEARSOLVED, D_POSE_BANKED, correction_contour + POSE_SIDECAR_BYTES),
        "D_distortion_floor_banked_pose_zero_rate": S(
            D_SEG_NEARSOLVED, D_POSE_BANKED, 0),
        "E_distortion_floor_SOLVED_pose_zero_rate": S(
            D_SEG_NEARSOLVED, D_POSE_SOLVED, 0),
    }

    binding = "frame_0 carrier (WebP-Q1 2,695,020 B; rate_term 1.79) -> then support geometry (421 KB)"
    out = {
        "schema": "ddm_fc1_stage5_compose.v1",
        "evidence_axis": "[macOS-CPU advisory] arithmetic over MEASURED n600 stream bytes; NO byte-closed evaluate.py row (near-solved d_seg needs UNBUILT realization); pointer UNMOVED 0.19108",
        "measured_stream_bytes": {
            "frame0_webp_q1": frame0_q1,
            "support_geometry_lzma": support_b,
            "labels_constriction_plus_table": labels_b,
            "pose_sidecar_est": POSE_SIDECAR_BYTES,
            "contour_support_bestcase_UNBUILT": CONTOUR_BEST_CASE_SUPPORT,
        },
        "scenarios": scenarios,
        "bar_0p172": BAR,
        "aim_0p15": AIM,
        "binding_stream": binding,
        "fork": "S(full measured) = %.3f >> 0.35 -> FORK-3: name binding stream + gap arithmetic (typed verdict, FAMILY scope)" % scenarios["A_full_measured_streams"]["S"],
        "gap_arithmetic": {
            "full_measured_S": scenarios["A_full_measured_streams"]["S"],
            "gap_to_bar": scenarios["A_full_measured_streams"]["S"] - BAR,
            "even_free_frame0_S": scenarios["B_frame0_free_correction_measured"]["S"],
            "even_free_frame0_contour_S": scenarios["C_frame0_free_correction_contour_bestcase"]["S"],
            "distortion_floor_banked_pose": scenarios["D_distortion_floor_banked_pose_zero_rate"]["S"],
            "distortion_floor_solved_pose": scenarios["E_distortion_floor_SOLVED_pose_zero_rate"]["S"],
            "note": "banked pose 0.127 + near-solved seg 0.0152 = 0.142 distortion floor leaves only 0.030 rate budget (45 KB) for ALL streams; the correction stream alone is 463 KB (contour best-case 184 KB); frame_0 is 2.70 MB. Sub-bar is UNREACHABLE on the copy-PREDICT task-lossy-correction codec without (a) frame amortization (banned INR lineage) and (b) SOLVED pose (291 MB solve).",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
