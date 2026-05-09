#!/usr/bin/env python3
"""Convert M5 Max sweep results into typed atoms for solver consumption.

Reads ``results.jsonl`` from a ``tools/sweep_m5max_hnerv_cluster.py`` run and
emits ``atoms.jsonl`` consumable by:
- meta-Lagrangian solver
- cathedral autopilot's GHA promotion queue
- continual-learning posterior updates
- Pareto frontier expansion

Each atom row carries the typed fields required by the meta-Lagrangian
solver per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable, plus
the M5 Max-specific calibration tag and ε band.

Schema (per row):
    {
      "schema_version": 1,
      "atom_kind": "m5max_sweep_result",
      "candidate_id": str,
      "archive_sha256": str,
      "archive_size_bytes": int,
      "architecture_class": str,
      "macos_cpu_calibrated": float | null,
      "macos_cpu_calibrated_tag": "[macOS-CPU calibrated]" | "[macOS-CPU advisory only]",
      "epsilon_band_low": float | null,
      "epsilon_band_high": float | null,
      "predicted_contest_cpu_gha": float | null,
      "predicted_contest_cpu_gha_tag": "[predicted; macos_x86_64_calibration_pr107]"
                                        | null,
      "ready_for_gha_dispatch": bool,
      "ready_for_cuda_dispatch": null,  // requires sibling-B mechanism verdict
      "promotion_verdict": str,
      "drift_flag": str,
      "evidence_grade": "macOS-CPU-calibrated" | "macOS-CPU-advisory" | "eval_failed",
      "score_claim": false,                           // M5 Max is NEVER authoritative
      "score_claim_promote_with_gha": bool,
      "promotion_eligible": false,                    // shippable archives need GHA
      "rank_or_kill_eligible": false,                 // per kill-as-last-resort
      "next_unblock_action": str,
      "source_results_jsonl": str,
      "atom_emitted_utc": str
    }

Usage:
    .venv/bin/python tools/m5max_sweep_to_atoms.py \\
        --results-jsonl experiments/results/m5max_sweep_<ts>/results.jsonl \\
        --output-jsonl experiments/results/m5max_sweep_atoms_<ts>/atoms.jsonl
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds")


def _row_to_atom(row: dict, source_path: str) -> dict:
    score = row.get("macos_cpu_score")
    tag = row.get("macos_cpu_calibrated_tag", "[macOS-CPU advisory only]")
    is_calibrated = tag == "[macOS-CPU calibrated]"

    if score is None:
        evidence_grade = "eval_failed"
    elif is_calibrated:
        evidence_grade = "macOS-CPU-calibrated"
    else:
        evidence_grade = "macOS-CPU-advisory"

    verdict = row.get("promotion_verdict", "LOG_ONLY")
    ready_for_gha = (verdict == "AUTO_PROMOTE_GHA"
                     or (verdict == "OPERATOR_DECISION" and is_calibrated))
    score_claim_promote_with_gha = ready_for_gha

    if verdict == "AUTO_PROMOTE_GHA":
        next_action = (
            "dispatch_GHA_cpu_eval_via_dispatch_cpu_eval_via_github_actions_py"
        )
    elif verdict == "OPERATOR_DECISION":
        next_action = "operator_review_silver_band_proximity"
    elif verdict == "EVAL_FAILED":
        next_action = "investigate_eval_failure_then_resweep"
    else:
        next_action = "log_only_no_dispatch"

    predicted_tag = ("[predicted; macos_x86_64_calibration_pr107]"
                     if is_calibrated and score is not None else None)

    return {
        "schema_version": 1,
        "atom_kind": "m5max_sweep_result",
        "candidate_id": row.get("candidate_id"),
        "archive_path": row.get("archive_path"),
        "archive_sha256": row.get("archive_sha256"),
        "archive_size_bytes": row.get("archive_size_bytes"),
        "architecture_class": row.get("architecture_class"),
        "macos_cpu_calibrated": score,
        "macos_cpu_calibrated_tag": tag,
        "macos_cpu_avg_segnet_dist": row.get("macos_cpu_avg_segnet_dist"),
        "macos_cpu_avg_posenet_dist": row.get("macos_cpu_avg_posenet_dist"),
        "compression_rate": row.get("compression_rate"),
        "n_samples": row.get("n_samples"),
        "epsilon_band_low": row.get("epsilon_band_low"),
        "epsilon_band_high": row.get("epsilon_band_high"),
        "predicted_contest_cpu_gha": row.get("predicted_contest_cpu_gha"),
        "predicted_contest_cpu_gha_tag": predicted_tag,
        "ready_for_gha_dispatch": ready_for_gha,
        "ready_for_cuda_dispatch": None,
        "ready_for_cuda_dispatch_blocked_reason":
            "requires_sibling_B_cuda_cpu_drift_discriminator_mechanism_verdict",
        "promotion_verdict": verdict,
        "drift_flag": row.get("drift_flag", "NA"),
        "evidence_grade": evidence_grade,
        "score_claim": False,
        "score_claim_promote_with_gha": score_claim_promote_with_gha,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "rank_or_kill_eligible_blocked_reason":
            "shippable_archives_require_contest_cpu_gha_anchor_per_dual_eval_mandate",
        "ready_for_exact_eval_dispatch": False,
        "next_unblock_action": next_action,
        "source_results_jsonl": source_path,
        "atom_emitted_utc": _now_iso(),
        "calibration_policy_ref":
            "feedback_macos_x86_64_epsilon_calibrated_tag_20260508",
        "domain_atom_id_ref": "domain_exploit_hardware_3",
        "lane_id_ref": "lane_m5max_parallel_sweep_substrate",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-jsonl", type=Path, required=True,
                        help="Input results.jsonl from sweep_m5max_hnerv_cluster.py")
    parser.add_argument("--output-jsonl", type=Path, required=True,
                        help="Output atoms.jsonl path under experiments/results/")
    args = parser.parse_args()

    src = args.results_jsonl.resolve()
    if not src.exists():
        raise SystemExit(f"--results-jsonl does not exist: {src}")
    # Check both raw input and resolved path; macOS symlinks /tmp → /private/tmp
    # so a check on the resolved path alone misses operator-typed /tmp/...
    raw_dst = args.output_jsonl
    dst = raw_dst.resolve()
    if (str(raw_dst).startswith("/tmp") or str(dst).startswith("/tmp")
            or str(dst).startswith("/private/tmp")):
        raise SystemExit("/tmp paths are FORBIDDEN per CLAUDE.md "
                         "(transient-evidence-trap)")
    dst.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(src) as fin, open(dst, "w") as fout:
        for ln in fin:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            atom = _row_to_atom(row, str(src))
            fout.write(json.dumps(atom) + "\n")
            n += 1
    print(f"[atoms] wrote {n} atom(s) → {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
