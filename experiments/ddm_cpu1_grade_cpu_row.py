#!/usr/bin/env python
"""ddm_cpu1 — grade the harvested contest-CPU row against every frame that binds.

Reads the harvested ``contest_auth_eval.json`` for the jg5 CPU-axis dispatch and
emits, in one place: the falsifier checks, the score arithmetic, the axis delta
against the ``[contest-CUDA T4]`` row on the SAME bytes, the GT-lineage
attribution measured by ``ddm_cpu1_gt_lineage_attribution.py``, and the
wall-clock verdict in all three published frames.

Every number is read or derived from receipts. Nothing is typed in as a result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# The jg5 [contest-CUDA T4] row on the SAME archive bytes. Read from the pointer
# mirror rather than typed: see --cuda-receipt.
ARCHIVE_SHA = "f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e"
ARCHIVE_BYTES = 180625
RUNTIME_TREE_SHA = "2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b"
PR135_BAR = 0.162  # best public PR at freeze; the declaration bar in the plan


def score(seg: float, pose: float, rate_unscaled: float) -> float:
    return 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * rate_unscaled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-receipt", type=Path, required=True)
    ap.add_argument(
        "--cpu-provenance",
        type=Path,
        required=True,
        help="the harvested provenance.json: it, NOT the inner receipt, carries archive_sha256 "
        "and inflate_runtime_manifest.runtime_tree_sha256",
    )
    ap.add_argument("--cuda-receipt", type=Path, required=True)
    ap.add_argument("--attribution", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cpu = json.loads(args.cpu_receipt.read_text())
    prov = json.loads(args.cpu_provenance.read_text())
    cuda = json.loads(args.cuda_receipt.read_text())
    attr = json.loads(args.attribution.read_text())

    # ---- FALSIFIERS, pre-registered in the seal ---------------------------
    falsifiers: dict[str, dict] = {}
    got_sha = str(prov.get("archive_sha256") or cpu.get("archive_sha256") or "")
    falsifiers["archive_sha256_is_jg5"] = {
        "expected": ARCHIVE_SHA, "got": got_sha, "pass": got_sha == ARCHIVE_SHA,
        "note": "PRIMARY. A receipt naming other bytes voids the row entirely.",
    }
    got_bytes = int(cpu.get("archive_size_bytes") or 0)
    falsifiers["archive_bytes_is_180625"] = {
        "expected": ARCHIVE_BYTES, "got": got_bytes, "pass": got_bytes == ARCHIVE_BYTES,
    }
    grade = str(cpu.get("evidence_grade") or "")
    falsifiers["evidence_grade_is_contest_cpu"] = {
        "expected": "contest-CPU", "got": grade, "pass": "contest-cpu" in grade.lower(),
    }
    tree = str((prov.get("inflate_runtime_manifest") or {}).get("runtime_tree_sha256") or "")
    falsifiers["runtime_tree_matches_shipped"] = {
        "expected": RUNTIME_TREE_SHA, "got": tree, "pass": tree == RUNTIME_TREE_SHA,
        "note": "Confirms the CPU worker evaluated the SAME tree the T4 row did.",
    }

    # ---- the row -----------------------------------------------------------
    cpu_seg = float(cpu["avg_segnet_dist"])
    cpu_pose = float(cpu["avg_posenet_dist"])
    rate = float(cpu["rate_unscaled"])
    cpu_S = score(cpu_seg, cpu_pose, rate)

    cuda_seg = float(cuda["avg_segnet_dist"])
    cuda_pose = float(cuda["avg_posenet_dist"])
    # The pointer's anchor MIRROR is compact and omits rate. Rate is a pure
    # function of the archive bytes, which are identical on both axes by
    # construction, so deriving it is exact -- not a substitution.
    cuda_rate = float(cuda.get("rate_unscaled") or rate)
    cuda_S = score(cuda_seg, cuda_pose, cuda_rate)

    def legs(seg: float, pose: float, r: float) -> dict:
        return {"seg": 100.0 * seg, "pose": math.sqrt(10.0 * pose), "rate": 25.0 * r}

    cpu_legs, cuda_legs = legs(cpu_seg, cpu_pose, rate), legs(cuda_seg, cuda_pose, cuda_rate)
    dS = cpu_S - cuda_S

    # ---- wall clock, all three published frames ----------------------------
    inflate = float(cpu.get("inflate_elapsed_seconds") or 0.0)
    evaluate = float(cpu.get("evaluate_elapsed_seconds") or 0.0)
    charged = inflate + evaluate
    from tac.contest_budget import JOB_WALL_SECONDS, residual_window

    w = residual_window("contest_cpu")
    cold, warm = float(w.narrow_end_seconds), float(w.wide_end_seconds)
    # Frame B: the packet's published reading -- the residual band corrected by
    # replacing the ESTIMATED evaluate with the MEASURED one.
    est_lo, est_hi = 120.0, 180.0
    frame_b = (cold + (est_lo - evaluate), warm + (est_hi - evaluate))

    walls = {
        "inflate_seconds": inflate,
        "evaluate_seconds": evaluate,
        "charged_seconds": charged,
        "job_wall_seconds": JOB_WALL_SECONDS,
        "headroom_in_job_wall_seconds": JOB_WALL_SECONDS - charged,
        "frame_A_canonical": {
            "band": [cold, warm], "value": charged,
            "verdict": "PASS" if charged <= cold else ("WARN" if charged <= warm else "REFUSE"),
        },
        "frame_B_evaluate_corrected": {
            "band": list(frame_b), "value": inflate,
            "verdict": "PASS" if inflate <= frame_b[0] else ("WARN" if inflate <= frame_b[1] else "REFUSE"),
        },
        "frame_C_absolute_job_wall": {
            "band": [0.0, float(JOB_WALL_SECONDS)], "value": charged,
            "verdict": "PASS" if charged <= JOB_WALL_SECONDS else "REFUSE",
        },
    }

    # ---- attribution -------------------------------------------------------
    lineage_dS = score(attr["d_seg_vs_PYAV_gt"], attr["d_pose_vs_PYAV_gt"], rate) - score(
        attr["d_seg_vs_DALI_gt"], attr["d_pose_vs_DALI_gt"], rate
    )

    out = {
        "schema": "ddm_cpu1_cpu_row_grade.v1",
        "axis": "[contest-CPU]",
        "archive_sha256": got_sha,
        "archive_size_bytes": got_bytes,
        "falsifiers": falsifiers,
        "all_falsifiers_pass": all(f["pass"] for f in falsifiers.values()),
        "row": {
            "canonical_score": cpu_S,
            "canonical_score_in_receipt": cpu.get("canonical_score"),
            "avg_segnet_dist": cpu_seg,
            "avg_posenet_dist": cpu_pose,
            "rate_unscaled": rate,
            "contributions": cpu_legs,
        },
        "cuda_row_same_bytes": {
            "canonical_score": cuda_S,
            "avg_segnet_dist": cuda_seg,
            "avg_posenet_dist": cuda_pose,
            "contributions": cuda_legs,
        },
        "axis_delta_cpu_minus_cuda": {
            "total": dS,
            "seg": cpu_legs["seg"] - cuda_legs["seg"],
            "pose": cpu_legs["pose"] - cuda_legs["pose"],
            "rate": cpu_legs["rate"] - cuda_legs["rate"],
        },
        "gt_lineage_attribution": {
            "lineage_only_dS": lineage_dS,
            "share_of_axis_delta": (lineage_dS / dS) if dS else None,
            "residual_non_lineage_dS": dS - lineage_dS,
            "C_pose_gt_table_mse": attr["C_pose_gt_table_mse"],
            "gt_argmax_lineage_disagreement_rate": attr["gt_argmax_lineage_disagreement_rate"],
            "axis_of_attribution": attr["axis"],
        },
        "walls": walls,
        "decision_inputs": {
            "pr135_bar": PR135_BAR,
            "cpu_score_at_or_above_bar": cpu_S >= PR135_BAR,
            "cuda_score": cuda_S,
            "cpu_minus_bar": cpu_S - PR135_BAR,
        },
        "score_claim": True,
        "promotable": False,
        "note": (
            "Dual-axis completion on bytes whose CUDA authority row already exists. "
            "The CPU row is an authority row for the CPU axis; it does not replace the "
            "CUDA row and does not move the effective frontier."
        ),
    }
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True))

    print(f"FALSIFIERS: {'ALL PASS' if out['all_falsifiers_pass'] else 'FAILED'}")
    for k, v in falsifiers.items():
        print(f"  [{'ok' if v['pass'] else 'FAIL'}] {k}")
    print(f"\n[contest-CPU] S = {cpu_S:.8f}   (seg {cpu_legs['seg']:.6f} + pose {cpu_legs['pose']:.6f} + rate {cpu_legs['rate']:.6f})")
    print(f"[contest-CUDA] S = {cuda_S:.8f}   (seg {cuda_legs['seg']:.6f} + pose {cuda_legs['pose']:.6f} + rate {cuda_legs['rate']:.6f})")
    print(f"dS(CPU - CUDA) = {dS:+.8f}")
    share = f"{100 * lineage_dS / dS:.2f}%" if dS else "n/a (dS == 0)"
    print(f"  GT lineage explains {lineage_dS:+.8f} ({share}), residual {dS - lineage_dS:+.2e}")
    print(f"\nWALL: inflate {inflate:.1f}s + evaluate {evaluate:.1f}s = {charged:.1f}s of {JOB_WALL_SECONDS}s")
    for name, fr in (("A canonical", walls["frame_A_canonical"]),
                     ("B evaluate-corrected", walls["frame_B_evaluate_corrected"]),
                     ("C absolute job wall", walls["frame_C_absolute_job_wall"])):
        print(f"  frame {name}: {fr['verdict']} — {fr['value']:.1f}s vs [{fr['band'][0]:.1f}, {fr['band'][1]:.1f}]")
    print(f"\nPR135 bar {PR135_BAR}: CPU {'ABOVE' if cpu_S >= PR135_BAR else 'BELOW'} by {abs(cpu_S - PR135_BAR):.5f}")
    print(f"\nWROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
