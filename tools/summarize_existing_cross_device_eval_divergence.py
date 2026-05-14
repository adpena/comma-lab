#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stream 1 — summarize existing cross-device eval divergence.

Operator directive 2026-05-13 AGGRESSIVE LOCAL HARDWARE SWEEP Stream 1.

Compares contest-CPU canonical scores vs macOS-CPU advisory + MPS-research-signal
scores for a fixed set of archives. Identifies substrates with high cross-device
divergence as PRE-DISPATCH-BUG-SUSPECT — divergence > 0.05 is unusual and
predicts cloud-dispatch issues.

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 + "Apples-to-apples
evidence discipline".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_eval(path: Path) -> dict | None:
    try:
        body = json.loads(path.read_text())
        return body
    except Exception:
        return None


def main(out_path: str) -> int:
    diag_rows = []

    # A1 archive — 3 axes
    a1_contest_cpu_path = REPO_ROOT / "submissions" / "a1" / "contest_auth_eval.cpu.json"
    a1_contest_cuda_path = REPO_ROOT / "submissions" / "a1" / "contest_auth_eval.cuda.json"
    a1_macos_cpu_path = None
    a1_mps_path = None

    run_dir = REPO_ROOT / "experiments" / "results" / "lane_local_hardware_aggressive_sweep_20260513_20260513T212901Z"
    if (run_dir / "cpu_a1" / "contest_auth_eval.json").exists():
        a1_macos_cpu_path = run_dir / "cpu_a1" / "contest_auth_eval.json"
    if (run_dir / "mps_a1" / "contest_auth_eval.json").exists():
        a1_mps_path = run_dir / "mps_a1" / "contest_auth_eval.json"

    a1_axes = {
        "[contest-CPU]": (load_eval(a1_contest_cpu_path) if a1_contest_cpu_path.exists() else None),
        "[contest-CUDA]": (load_eval(a1_contest_cuda_path) if a1_contest_cuda_path.exists() else None),
        "[macOS-CPU advisory]": (load_eval(a1_macos_cpu_path) if a1_macos_cpu_path else None),
        "[MPS-research-signal]": (load_eval(a1_mps_path) if a1_mps_path else None),
    }
    a1_row = {
        "archive_sha256": "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5",
        "archive_bytes": 178262,
        "substrate": "A1 (PR101 fine-tuned)",
        "axes": {},
    }
    for axis, body in a1_axes.items():
        if body is None:
            a1_row["axes"][axis] = None
            continue
        score = body.get("canonical_score") or body.get("canonical_score_recomputed") or body.get("score_recomputed_from_components")
        a1_row["axes"][axis] = {
            "canonical_score": score,
            "avg_posenet_dist": body.get("avg_posenet_dist"),
            "avg_segnet_dist": body.get("avg_segnet_dist"),
            "evaluate_elapsed_seconds": body.get("evaluate_elapsed_seconds"),
            "inflate_elapsed_seconds": body.get("inflate_elapsed_seconds"),
        }
    diag_rows.append(a1_row)

    # PR106 sidecar — 3 axes (contest-CUDA reference, our MPS, no macOS-CPU full eval yet)
    pr106_contest_cuda_path = REPO_ROOT / "submissions" / "pr106_latent_sidecar_r2" / "contest_auth_eval.json"
    pr106_mps_path = None
    if (run_dir / "mps_pr106" / "contest_auth_eval.json").exists():
        pr106_mps_path = run_dir / "mps_pr106" / "contest_auth_eval.json"

    pr106_axes = {
        "[contest-CUDA]": (load_eval(pr106_contest_cuda_path) if pr106_contest_cuda_path.exists() else None),
        "[MPS-research-signal]": (load_eval(pr106_mps_path) if pr106_mps_path else None),
    }
    pr106_row = {
        "archive_sha256": "7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f",
        "archive_bytes": 186822,
        "substrate": "PR106 latent_sidecar_r2 (HNeRV-PR106 + latent delta sidecar)",
        "axes": {},
    }
    for axis, body in pr106_axes.items():
        if body is None:
            pr106_row["axes"][axis] = None
            continue
        score = body.get("canonical_score") or body.get("canonical_score_recomputed") or body.get("score_recomputed_from_components")
        pr106_row["axes"][axis] = {
            "canonical_score": score,
            "avg_posenet_dist": body.get("avg_posenet_dist"),
            "avg_segnet_dist": body.get("avg_segnet_dist"),
            "evaluate_elapsed_seconds": body.get("evaluate_elapsed_seconds"),
            "inflate_elapsed_seconds": body.get("inflate_elapsed_seconds"),
        }
    diag_rows.append(pr106_row)

    # Compute divergences per row
    bug_flags = []
    for row in diag_rows:
        axes = row["axes"]
        contest_cpu = axes.get("[contest-CPU]")
        contest_cuda = axes.get("[contest-CUDA]")
        macos_cpu = axes.get("[macOS-CPU advisory]")
        mps = axes.get("[MPS-research-signal]")
        ref = contest_cpu or contest_cuda
        if ref is None:
            row["divergence_analysis"] = {"verdict": "no_reference_axis"}
            continue
        ref_score = ref["canonical_score"]
        ref_label = "[contest-CPU]" if contest_cpu else "[contest-CUDA]"
        deltas = {}
        if macos_cpu:
            deltas["macos_cpu_vs_ref"] = macos_cpu["canonical_score"] - ref_score
        if mps:
            deltas["mps_vs_ref"] = mps["canonical_score"] - ref_score
        if contest_cuda and contest_cpu:
            deltas["contest_cuda_vs_cpu"] = contest_cuda["canonical_score"] - contest_cpu["canonical_score"]
        max_abs_delta = max((abs(d) for d in deltas.values()), default=0.0)
        verdict = "ok_no_divergence"
        if max_abs_delta > 0.05:
            verdict = "PRE_DISPATCH_BUG_SUSPECT_HIGH_DIVERGENCE"
            bug_flags.append({
                "substrate": row["substrate"],
                "max_abs_delta": max_abs_delta,
                "deltas": deltas,
            })
        elif max_abs_delta > 0.01:
            verdict = "MODERATE_DIVERGENCE_INVESTIGATE"
            bug_flags.append({
                "substrate": row["substrate"],
                "max_abs_delta": max_abs_delta,
                "deltas": deltas,
            })
        row["divergence_analysis"] = {
            "reference_axis": ref_label,
            "reference_score": ref_score,
            "deltas": deltas,
            "max_abs_delta": max_abs_delta,
            "verdict": verdict,
        }

    out = {
        "schema": "cross_device_divergence_diagnostic_v1",
        "lane_id": "lane_local_hardware_aggressive_sweep_20260513",
        "evidence_grade": "macOS-CPU+MPS-research-signal",
        "evidence_tag": "[macOS-CPU advisory] + [MPS-research-signal]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ranking_only": True,
        "diagnostic_rows": diag_rows,
        "pre_dispatch_bug_suspects": bug_flags,
        "n_substrates_checked": len(diag_rows),
        "n_bug_suspects": len(bug_flags),
        "interpretation": (
            "Cross-device divergence > 0.05 between any pair of axes "
            "(macOS-CPU vs contest-CPU/CUDA, or MPS vs contest-CPU/CUDA) "
            "predicts cloud-dispatch issues. Investigate before $0.50+ GPU spend."
        ),
        "method_notes": [
            "A1 (HNeRV PR101 substrate): tests scorer numerical stability since "
            "decoder output is identical bytes on CPU/MPS (deterministic torch).",
            "PR106 (HNeRV PR106 + sidecar): inflate.py FORBIDS MPS device, so "
            "the inflated frames came from CPU; MPS only affected SegNet/PoseNet "
            "scorer forwards in evaluate.py.",
            "Divergence has TWO components: (1) decoder/inflate-side numerical "
            "drift (CPU vs MPS-renderer-output bytes differ); (2) scorer-side "
            "numerical drift (SegNet/PoseNet forwards under different precision).",
        ],
    }

    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    for row in diag_rows:
        print()
        print(f"Substrate: {row['substrate']}")
        print(f"  archive_bytes: {row['archive_bytes']}")
        for axis, axis_data in row["axes"].items():
            if axis_data is None:
                print(f"  {axis}: <not measured>")
            else:
                sc = axis_data.get('canonical_score')
                pose = axis_data.get('avg_posenet_dist')
                seg = axis_data.get('avg_segnet_dist')
                es = axis_data.get('evaluate_elapsed_seconds')
                sc_s = f"{sc:.6f}" if sc is not None else "NA"
                pose_s = f"{pose:.2e}" if pose is not None else "NA"
                seg_s = f"{seg:.2e}" if seg is not None else "NA"
                es_s = f"{es:.1f}" if es is not None else "NA"
                print(f"  {axis}: score={sc_s}  pose={pose_s}  seg={seg_s}  eval_sec={es_s}")
        da = row.get("divergence_analysis", {})
        if "max_abs_delta" in da:
            print(f"  --> max_abs_delta vs {da['reference_axis']} = {da['max_abs_delta']:.6f}")
            print(f"  --> verdict: {da['verdict']}")
    print()
    print(f"PRE-DISPATCH BUG SUSPECTS: {len(bug_flags)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: summarize_existing_cross_device_eval_divergence.py <out_path.json>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
