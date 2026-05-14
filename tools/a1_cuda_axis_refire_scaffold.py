#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""A1 CUDA-axis refire scaffold ($0 GPU, planning only).

Per CLAUDE.md "Frontier target" + `reports/phase_a_pareto_20260508.md`:

    A1 score-gradient has paired anchors on archive SHA
    87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5
    (178,262 B):
       [contest-CPU]  0.19284757743677347   (GHA Linux x86_64; sub-gold)
       [contest-CUDA] 0.22635202347843950   (Modal T4; above gold)

    CUDA-CPU gap = 0.03350. Per the axis advisor in phase_a:
       CUDA-internal priority: pose (weight 119.87 vs SegNet 100)
       CPU-leaderboard priority: seg (weight 85.47 vs pose 53.39)

    "Current next move is score-domain validation/early stopping that
     changes the runtime-consumed packet, not another blind A1 refire or
     selector in the same basin."

This scaffold ($0 GPU) prepares the next dispatch:

  * documents the CUDA-axis basin and why it is gap-wide on the pose axis;
  * predicts dispatch cost via tac.cost_band_calibration.predict;
  * emits a paired smoke-before-full plan (Catalog #167);
  * surfaces operator decision DEC-A1-CUDA-DISPATCH with cost, expected
    outcome, and reactivation criteria.

It does NOT modify experiments/train_score_gradient_pr101_finetune.py.
The actual inner-loop CUDA validation hook is documented as a planning
proposal in the emitted dispatch packet; the trainer modification is the
follow-on subagent's job once the operator approves DEC-A1-CUDA-DISPATCH.

Sister tooling:
    experiments/train_score_gradient_pr101_finetune.py   (A1 trainer)
    experiments/modal_phase_a1_score_gradient_pr101.py   (Modal dispatcher)
    experiments/contest_auth_eval.py                     (CUDA scorer)
    tools/run_modal_smoke_before_full.py                 (Catalog #167)
    tools/operator_authorize.py                          (canonical entry)

Output:
    --json-out (optional) emits the dispatch packet JSON
    stdout shows the operator-facing summary

NO GPU SPEND. NO ARCHIVE BUILD. NO AUTH-EVAL. Planning surface only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tac.cost_band_calibration import predict


# ---------------------------------------------------------------------------
# Empirical anchors (frozen at landing-time per CLAUDE.md "Apples-to-apples
# evidence discipline").
# ---------------------------------------------------------------------------

A1_ANCHOR_ARCHIVE_SHA256 = (
    "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5"
)
A1_ANCHOR_ARCHIVE_BYTES = 178_262

# Paired exact-CUDA / exact-CPU anchors from phase_a_pareto_20260508.md:
A1_ANCHOR_CPU_SCORE = 0.19284757743677347   # [contest-CPU; GHA Linux x86_64]
A1_ANCHOR_CUDA_SCORE = 0.22635202347843950  # [contest-CUDA; Modal T4]
A1_CUDA_CPU_GAP = A1_ANCHOR_CUDA_SCORE - A1_ANCHOR_CPU_SCORE  # +0.03350

# Axis advisor marginals at the A1 operating point (phase_a section "Solver
# planning targets"; chain-rule planner marginals):
AXIS_ADVISOR = {
    "operating_point": {
        "d_seg_cuda": 6.88e-04,
        "d_pose_cuda": 1.74e-04,
        "archive_bytes": A1_ANCHOR_ARCHIVE_BYTES,
    },
    "cuda_internal": {
        "priority_axis": "pose",
        "seg_marginal": 100.00,
        "pose_marginal": 119.87,
    },
    "cpu_leaderboard": {
        "priority_axis": "seg",
        "seg_marginal": 85.47,
        "pose_marginal": 53.39,
    },
}


# ---------------------------------------------------------------------------
# Prior-attempt ledger (frozen; do NOT auto-refire any of these basins).
# ---------------------------------------------------------------------------

# Every row is a measured-config retirement — they are NOT killed (per
# CLAUDE.md "KILL is LAST RESORT"); reactivation criteria are documented
# inline.
PRIOR_RETIRED_BASINS = [
    {
        "basin_id": "a1_latent_aligned_constrained_refire",
        "archive_bytes": 178_262,
        "archive_sha256": A1_ANCHOR_ARCHIVE_SHA256,
        "contest_cuda_score": 0.22635202347843950,
        "contest_cpu_score": 0.19284757743677347,
        "retirement_reason": (
            "current A1 anchor — paired CPU positive + CUDA above gold; "
            "selection/early-stopping in this basin would only reproduce the "
            "same archive"
        ),
        "evidence": "phase_a_pareto_20260508.md",
    },
    {
        "basin_id": "a1_long_lr1e6_modal_refire",
        "archive_bytes": 178_276,
        "advisory_cpu_score": 0.19359165212500000,
        "retirement_reason": (
            "longer fine-tuning at lr=1e-6 regressed CPU; CUDA eval blocked "
            "by Modal temp-workdir bug at the time"
        ),
        "evidence": "phase_a_pareto_20260508.md",
    },
    {
        "basin_id": "a1_guarded_kl05_pixel_l1_002_refire",
        "archive_bytes": 178_279,
        "contest_cuda_score": 0.22655968711150934,
        "advisory_cpu_score": 0.19309483549345535,
        "retirement_reason": (
            "stronger KL=0.5 + pixel_l1=0.02 regularization regressed BOTH "
            "axes vs the A1 anchor; auxiliary losses are not the lever"
        ),
        "evidence": "phase_a1_best_proxy_checkpoint_selection_20260509_codex.md",
    },
    {
        "basin_id": "a1_best_proxy_checkpoint_selection",
        "archive_bytes": 178_262,
        "archive_sha256": A1_ANCHOR_ARCHIVE_SHA256,
        "contest_cuda_score": 0.22635202347843950,
        "retirement_reason": (
            "best-proxy selection produced same archive SHA / same CUDA "
            "score as the latent-aligned constrained refire (duplicate "
            "confirmation, not a new anchor)"
        ),
        "evidence": "phase_a1_best_proxy_modal_dispatch_20260509_codex.md",
    },
]


# ---------------------------------------------------------------------------
# Proposed inner-loop CUDA-validation pattern (planning surface only).
# ---------------------------------------------------------------------------

INNER_LOOP_PROPOSAL = {
    "name": "periodic_cuda_validation_early_stop",
    "summary": (
        "Every K epochs, the trainer builds a byte-closed archive from the "
        "current EMA shadow and runs the full contest_auth_eval CUDA path "
        "against it. Track best-CUDA checkpoint separately from "
        "best-proxy and final-EMA. Early-stop on M consecutive no-improvement "
        "CUDA windows."
    ),
    "why_score_domain": (
        "Per CLAUDE.md 'Forbidden weight-domain saliency on score-gradient "
        "substrate' (Catalog #123): the A1 substrate is score-gradient-"
        "trained; weight-domain proxies are anti-correlated with score "
        "saliency. The training-loss weighted_proxy already used by "
        "checkpoint_best_proxy is a score-domain quantity, but it is a "
        "CPU/proxy-axis quantity. The new signal is the EXACT [contest-CUDA] "
        "score on a real byte-closed archive — the same axis the operator "
        "promotion criterion targets."
    ),
    "why_pose_targeted": (
        "The CUDA-internal axis advisor priority is pose (weight 119.87 vs "
        "SegNet 100). The CPU-leaderboard advisor priority is seg (weight "
        "85.47 vs pose 53.39). The A1 anchor's CUDA pose component dominates "
        "the gap. Adding the EXACT-CUDA score as the trainer's selection "
        "signal aligns the optimizer's search direction with the dominant "
        "gap axis."
    ),
    "reactivation_criterion_per_phase_a_memo": (
        "produces archive with DIFFERENT SHA than 87ec7ca5... AND new "
        "[contest-CUDA] anchor that closes ≥0.01 of the 0.0335 gap"
    ),
    "design_blockers_for_followup_subagent": [
        "AUTH-EVAL-COST: every in-loop CUDA validation invokes the full "
        "inflate.sh + evaluate.py path (~10-15 min on T4); for K=10, "
        "60-epoch run = 6 in-loop evals = ~60-90 min of T4 inflate/eval "
        "wall-clock on TOP of the 40-epoch training (~$2-4 extra). "
        "Mitigation: validate only on the first 4 of 600 test videos as a "
        "smoke; full-600 only on the early-stop window's final checkpoint.",
        "ARCHIVE-BUILD-COST: each in-loop validation requires the byte-"
        "closed archive build (~10-30s on T4 per the existing post-train "
        "step). K=10 → 6 builds; negligible vs eval cost.",
        "GRADIENT-DISCONNECT: contest_auth_eval is a SCORING surface, "
        "not a differentiable surrogate. The optimizer cannot backprop "
        "through it; the signal is a SELECTION/EARLY-STOP signal, like "
        "best_proxy is today. No optimizer step receives a CUDA-aware "
        "gradient (which is correct per CLAUDE.md non-negotiables — "
        "differentiable scorer-preprocess discipline applies to the "
        "TRAINING loss, not the SELECTION signal).",
        "BASIN-DIVERGENCE-EVIDENCE: the proposal assumes the CUDA-pose-"
        "dominated gap exists because the trainer's CPU-proxy gradient "
        "selects a different basin than the CUDA-pose-dominated basin. "
        "If the gap is instead a contest_auth_eval pose-component "
        "numerical drift (e.g. PyAV vs DALI ground-truth decode, "
        "PoseNet kernel CUDA-CPU drift), then no amount of in-loop CUDA "
        "validation will close it because the basin IS the same — only "
        "the measurement differs. The 2x2 decoder/network diagnostic "
        "(CLAUDE.md 'Apples-to-apples evidence discipline') should run "
        "FIRST to rule this out; estimated $0.30 Modal T4 smoke.",
    ],
    "estimated_trainer_loc_change": "~80-120 LOC in train_score_gradient_pr101_finetune.py",
}


# ---------------------------------------------------------------------------
# Cost-band prediction helper.
# ---------------------------------------------------------------------------

def predict_dispatch_costs() -> dict[str, Any]:
    """Predict cost bands for the four canonical A1 dispatch options.

    Per phase_a1_best_proxy_modal_dispatch memo, the prior A1 dispatch was
    Modal T4 / 40 epochs at estimated $2.36. The in-loop CUDA-validation
    pattern adds ~60-90 min of inflate/eval wall-clock; new estimate is
    Modal T4 / 60-80 epochs (longer to amortize the in-loop validation).
    """
    options = [
        # Primary option: extend A1's existing Modal T4 basin with in-loop val.
        ("modal", "T4", 60, "Primary: Modal T4 + 60ep + in-loop CUDA val (~K=10)"),
        ("modal", "T4", 40, "Replay: Modal T4 + 40ep (same as bestproxy basin)"),
        # Lightning is the high-throughput parallel-sweep option.
        ("lightning", "T4", 60, "Parallel: Lightning T4 + 60ep (free pool)"),
        # A10G/A100 dramatically reduce wall-clock per epoch.
        ("modal", "A10G", 60, "Faster: Modal A10G + 60ep"),
        ("modal", "A100", 60, "Fastest: Modal A100 + 60ep"),
    ]
    out: list[dict[str, Any]] = []
    for platform, gpu, epochs, label in options:
        p = predict(platform, gpu, epochs)
        out.append({
            "label": label,
            "platform": platform,
            "gpu": gpu,
            "epochs": epochs,
            "p10_cost_usd": p.p10_cost_usd,
            "p50_cost_usd": p.p50_cost_usd,
            "p90_cost_usd": p.p90_cost_usd,
            "n_anchors": p.n_anchors,
            "confidence_tag": p.confidence_tag,
        })
    return {"options": out}


def smoke_before_full_plan(primary: dict[str, Any]) -> dict[str, Any]:
    """Per Catalog #167 smoke-before-full pattern."""
    # 100-epoch Modal T4 smoke at ~$0.30 (per substrate_balle_renderer recipe)
    # OR cheaper for A1: 5-epoch smoke at proxy-only, no in-loop CUDA val.
    smoke = predict("modal", "T4", 100)
    return {
        "phase": "smoke",
        "platform": "modal",
        "gpu": "T4",
        "epochs": 5,
        "in_loop_cuda_val": False,
        "expected_outcome": "rc=0 + checkpoint_ema.pt present + proxy < 1.0",
        "predicted_p50_cost_usd": min(smoke.p50_cost_usd, 0.30),
        "predicted_p90_cost_usd": min(smoke.p90_cost_usd, 0.50),
        "purpose": (
            "validate the in-loop CUDA-val hook fires without crashing the "
            "trainer; confirms training + archive-build path is intact "
            "before paying for the full dispatch"
        ),
        "next_phase": "full" if smoke.p50_cost_usd >= 0 else "abort",
        "primary_dispatch_predicted_p50_usd": primary["p50_cost_usd"],
    }


# ---------------------------------------------------------------------------
# Operator decision packet.
# ---------------------------------------------------------------------------

def build_operator_decision_packet() -> dict[str, Any]:
    cost_bands = predict_dispatch_costs()
    primary = cost_bands["options"][0]
    smoke = smoke_before_full_plan(primary)
    return {
        "decision_id": "DEC-A1-CUDA-DISPATCH",
        "decision_class": "GPU dispatch ($1-3 Lightning T4 / $2-4 Modal T4)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_grade": "[prediction; cost-band hand-calibrated fallback]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "current_anchor": {
            "archive_sha256": A1_ANCHOR_ARCHIVE_SHA256,
            "archive_bytes": A1_ANCHOR_ARCHIVE_BYTES,
            "contest_cpu_score": A1_ANCHOR_CPU_SCORE,
            "contest_cuda_score": A1_ANCHOR_CUDA_SCORE,
            "cuda_cpu_gap": A1_CUDA_CPU_GAP,
        },
        "axis_advisor": AXIS_ADVISOR,
        "prior_retired_basins": PRIOR_RETIRED_BASINS,
        "inner_loop_proposal": INNER_LOOP_PROPOSAL,
        "predicted_cost_bands": cost_bands,
        "smoke_before_full_plan": smoke,
        "expected_outcome": {
            "best_case": (
                "in-loop CUDA-val selects a checkpoint that produces a NEW "
                "archive SHA with [contest-CUDA] ≤ 0.216 (closes ≥0.01 of "
                "the 0.0335 gap); paired [contest-CPU] regression ≤ 0.001 "
                "acceptable since CUDA is the dominant gap axis"
            ),
            "median_case": (
                "in-loop CUDA-val selects a checkpoint that produces a NEW "
                "archive SHA, but the new CUDA score is within ±0.005 of "
                "0.226 (basin is the same shape; the 0.0335 gap is "
                "measurement-driven, not basin-driven)"
            ),
            "worst_case": (
                "in-loop CUDA-val never finds a checkpoint better than the "
                "current A1 anchor (all candidates produce same/worse CUDA "
                "score); $2-4 GPU spent; gap remains 0.0335; learning is "
                "that the gap requires a different attack (e.g. PR101 → "
                "PR103 medal-delta replay on A1 substrate, lane "
                "lane_a1_inflate_time_bias_correction_sweep)"
            ),
        },
        "reactivation_criteria": [
            "new archive SHA != " + A1_ANCHOR_ARCHIVE_SHA256[:16] + "...",
            "OR new [contest-CUDA] anchor ≤ 0.216 (closes ≥0.01 of gap)",
            "OR 2x2 decoder/network diagnostic resolves the gap as "
            "measurement-only (in which case this lane retires and a "
            "PyAV/DALI normalization lane reactivates)",
        ],
        "do_not_auto_dispatch": True,
        "operator_approval_required": True,
        "cross_refs": [
            "reports/phase_a_pareto_20260508.md",
            ".omx/research/phase_a1_best_proxy_checkpoint_selection_20260509_codex.md",
            ".omx/research/phase_a1_best_proxy_modal_dispatch_20260509_codex.md",
            "CLAUDE.md 'Apples-to-apples evidence discipline'",
            "CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA'",
            "Catalog #123 (no weight-domain saliency on score-gradient substrate)",
            "Catalog #167 (smoke-before-full pattern)",
        ],
    }


# ---------------------------------------------------------------------------
# CLI surface.
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to emit the operator decision packet as JSON.",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Print only the operator-facing summary (suppress full packet).",
    )
    args = parser.parse_args(argv)

    packet = build_operator_decision_packet()

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(packet, indent=2))
        print(f"[wrote] {args.json_out}")

    print()
    print("=" * 78)
    print(" A1 CUDA-AXIS REFIRE — OPERATOR DECISION: DEC-A1-CUDA-DISPATCH")
    print("=" * 78)
    print()
    print(f"  Current A1 anchor (archive SHA {A1_ANCHOR_ARCHIVE_SHA256[:16]}…):")
    print(f"    archive_bytes        = {A1_ANCHOR_ARCHIVE_BYTES:,}")
    print(f"    [contest-CPU]        = {A1_ANCHOR_CPU_SCORE:.10f}  (sub-gold)")
    print(f"    [contest-CUDA]       = {A1_ANCHOR_CUDA_SCORE:.10f}  (above gold)")
    print(f"    CUDA-CPU gap         = +{A1_CUDA_CPU_GAP:.5f}")
    print()
    print("  Axis advisor (planner marginals @ A1 operating point):")
    print(f"    CUDA-internal: priority POSE  "
          f"(pose={AXIS_ADVISOR['cuda_internal']['pose_marginal']:.2f}, "
          f"seg={AXIS_ADVISOR['cuda_internal']['seg_marginal']:.2f})")
    print(f"    CPU-leaderboard: priority SEG "
          f"(pose={AXIS_ADVISOR['cpu_leaderboard']['pose_marginal']:.2f}, "
          f"seg={AXIS_ADVISOR['cpu_leaderboard']['seg_marginal']:.2f})")
    print()
    print("  Proposed inner-loop signal:")
    print(f"    {INNER_LOOP_PROPOSAL['name']}")
    print("    Every K epochs, build archive + run contest_auth_eval CUDA on the")
    print("    EMA shadow. Track best-CUDA checkpoint. Early-stop on M no-improve.")
    print()
    print("  Predicted cost bands (cost_band_calibration.predict):")
    for opt in packet["predicted_cost_bands"]["options"]:
        print(f"    {opt['label']}")
        print(f"      p10=${opt['p10_cost_usd']:.2f} "
              f"p50=${opt['p50_cost_usd']:.2f} "
              f"p90=${opt['p90_cost_usd']:.2f} "
              f"({opt['confidence_tag']}, N={opt['n_anchors']})")
    print()
    print("  Smoke-before-full (Catalog #167):")
    smoke = packet["smoke_before_full_plan"]
    print(f"    smoke   = {smoke['platform']}/{smoke['gpu']} × {smoke['epochs']}ep, "
          f"~${smoke['predicted_p50_cost_usd']:.2f} (p50) / "
          f"${smoke['predicted_p90_cost_usd']:.2f} (p90)")
    print(f"    purpose = {smoke['purpose']}")
    print()
    print("  Reactivation criteria:")
    for crit in packet["reactivation_criteria"]:
        print(f"    • {crit}")
    print()
    print("  Operator decision required: DO NOT AUTO-DISPATCH.")
    print("  Follow-up subagent task (NOT this scaffold): land the in-loop CUDA-val")
    print("  hook in experiments/train_score_gradient_pr101_finetune.py per the")
    print("  blockers documented in INNER_LOOP_PROPOSAL.design_blockers_for_followup_subagent")
    print("=" * 78)
    print()

    if not args.brief:
        print("Full packet (use --brief to suppress):")
        print(json.dumps(packet, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
