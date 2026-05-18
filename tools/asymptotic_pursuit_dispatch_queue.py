#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ASYMPTOTIC PURSUIT ordered dispatch queue + operator-attention budget rollup.

Per CLAUDE.md "Council hierarchy: 4-tier protocol" operator-attention budget
section. Reads the canonical readiness assessment from
``tools/asymptotic_pursuit_candidate_readiness_assessment.py`` + emits an
ordered queue with:

  * Per-candidate dispatch sequence (smoke → 100ep → full eval → paired axis
    per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA")
  * Cost-band rollup ($-low to $-high) per Catalog #270
  * Operator-attention budget per tier (T2 ≤3/day; T3 ≤3/week)

Sister of ``tools/asymptotic_pursuit_candidate_readiness_assessment.py``.
Lane: lane_asymptotic_pursuit_substrate_class_shift_q4_pivot_top_priority_20260517.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
# Add tools/ to sys.path so we can import sibling tool by name.
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from asymptotic_pursuit_candidate_readiness_assessment import (  # noqa: E402
    CANONICAL_CANDIDATES,
    ReadinessAssessment,
    assess_candidates,
)


def build_dispatch_sequence(
    assessment: ReadinessAssessment,
) -> list[dict[str, Any]]:
    """Build an ordered dispatch sequence for each candidate.

    Per Catalog #167 smoke-before-full pattern: every candidate must
    smoke-first ($1) before full ($N), then paired-axis CPU eval (~$0.10) for
    contest-CPU verification.
    """
    sequence: list[dict[str, Any]] = []
    for sid in assessment.ranked_by_ev_per_dollar:
        c = next(x for x in assessment.candidates if x.substrate_id == sid)
        smoke_cost = 1.0  # Canonical $1 smoke ceiling
        full_cost = c.estimated_dispatch_cost_usd
        paired_cost = 0.10
        total_cost = smoke_cost + full_cost + paired_cost
        sequence.append(
            {
                "substrate_id": sid,
                "readiness_verdict": c.readiness_verdict,
                "horizon_class": c.horizon_class,
                "predicted_delta_s_band": [c.predicted_delta_s_band_low, c.predicted_delta_s_band_high],
                "stages": [
                    {
                        "stage": "smoke_100ep",
                        "estimated_cost_usd": smoke_cost,
                        "gpu": c.min_smoke_gpu,
                        "wall_clock_seconds": min(c.estimated_dispatch_wall_clock_seconds, 600),
                        "rationale": "Catalog #167 smoke-before-full pattern; refuses full dispatch on smoke failure",
                    },
                    {
                        "stage": "full_eval_paired_cpu_cuda",
                        "estimated_cost_usd": full_cost,
                        "gpu": c.gpu_class,
                        "wall_clock_seconds": c.estimated_dispatch_wall_clock_seconds,
                        "rationale": "Catalog #226 + CLAUDE.md 'BOTH CPU AND CUDA' canonical paired auth-eval",
                    },
                    {
                        "stage": "paired_cpu_axis_verification",
                        "estimated_cost_usd": paired_cost,
                        "gpu": "CPU",
                        "wall_clock_seconds": 3600,
                        "rationale": "Catalog #316 frontier-scan drift detection on contest-CPU axis",
                    },
                ],
                "total_estimated_cost_usd": round(total_cost, 3),
                "blocking_issues": list(c.blocking_issues),
            }
        )
    return sequence


def compute_cost_band_rollup(sequence: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-tier cost rollup per CLAUDE.md 'Production-hardened dispatch optimization protocol'."""
    ready = [s for s in sequence if s["readiness_verdict"] == "READY"]
    needs_fix = [s for s in sequence if s["readiness_verdict"] == "NEEDS_FIX"]
    defer = [s for s in sequence if s["readiness_verdict"] == "DEFER"]
    return {
        "ready_total_cost_usd_if_dispatched": round(
            sum(s["total_estimated_cost_usd"] for s in ready), 3
        ),
        "needs_fix_total_cost_usd_if_unblocked_and_dispatched": round(
            sum(s["total_estimated_cost_usd"] for s in needs_fix), 3
        ),
        "defer_total_cost_usd_if_phase_2_council_completes": round(
            sum(s["total_estimated_cost_usd"] for s in defer), 3
        ),
        "ready_count": len(ready),
        "needs_fix_count": len(needs_fix),
        "defer_count": len(defer),
        "total_count": len(sequence),
    }


def compute_operator_attention_budget(
    sequence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per CLAUDE.md 'Council hierarchy 4-tier protocol' operator-attention
    budget per tier. Asymptotic-pursuit substrates typically need:
      - T2 sextet council deliberation per substrate before unblock
      - T3 grand council only if cross-cutting (e.g., Phase 2 lift)
    """
    # Per-substrate T2 deliberation: 1 per substrate; aggregate
    needs_fix_t2 = sum(1 for s in sequence if s["readiness_verdict"] == "NEEDS_FIX")
    defer_t2 = sum(1 for s in sequence if s["readiness_verdict"] == "DEFER")
    return {
        "t1_working_group_unbounded": "OK",
        "t2_sextet_council_per_substrate_to_unblock": needs_fix_t2 + defer_t2,
        "t2_per_30_day_budget": 90,
        "t2_within_budget": (needs_fix_t2 + defer_t2) <= 90,
        "t3_grand_council_for_phase_2_cross_cutting_unblock": defer_t2,
        "t3_per_30_day_budget": 13,
        "t3_within_budget": defer_t2 <= 13,
        "t4_symposium_only_for_strategic_pivots": 0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true", help="Emit JSON output")
    p.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Comma-separated candidate IDs",
    )
    p.add_argument("--repo-root", type=Path, default=None, help="Override repo root.")
    args = p.parse_args(argv)

    candidates = (
        tuple(c.strip() for c in args.candidates.split(",")) if args.candidates else None
    )
    assessment = assess_candidates(candidates, repo_root=args.repo_root)
    sequence = build_dispatch_sequence(assessment)
    rollup = compute_cost_band_rollup(sequence)
    budget = compute_operator_attention_budget(sequence)

    payload = {
        "dispatch_sequence": sequence,
        "cost_band_rollup": rollup,
        "operator_attention_budget": budget,
        "top_1_substrate": assessment.top_1_substrate,
        "top_1_readiness_verdict": assessment.top_1_readiness_verdict,
        "top_2_substrate_for_stage_2_stacking": assessment.top_2_substrate,
        "assessment_utc": assessment.assessment_utc,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "predicted",
        "provenance_kind": "PREDICTED_FROM_MODEL",
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"=== ASYMPTOTIC PURSUIT dispatch queue ===")
        print(f"  TOP-1: {assessment.top_1_substrate} ({assessment.top_1_readiness_verdict})")
        print(f"  TOP-2 (Stage 2 stacking): {assessment.top_2_substrate}")
        print()
        print(f"=== Cost-band rollup ===")
        for k, v in rollup.items():
            print(f"  {k}: {v}")
        print()
        print(f"=== Operator-attention budget ===")
        for k, v in budget.items():
            print(f"  {k}: {v}")
        print()
        print(f"=== Per-substrate dispatch sequence ===")
        for s in sequence:
            print(f"\n  {s['substrate_id']} [{s['readiness_verdict']}] horizon={s['horizon_class']}")
            print(f"    predicted_ΔS={s['predicted_delta_s_band']}")
            print(f"    total_cost=${s['total_estimated_cost_usd']}")
            if s["blocking_issues"]:
                print(f"    blocking: {len(s['blocking_issues'])} issue(s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
