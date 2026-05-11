#!/usr/bin/env bash
# Phase 3 (joint scorer-renderer-codec) — PRE-STAGED, multi-precondition gate.
#
# Per CLAUDE.md "Phase 3 dispatch — non-negotiable" + Catalog #134
# (check_phase3_dispatch_gate_fail_closed): Phase3DispatchGate refuses
# construction unless ALL 8 preconditions met OR unsafe_test_only=True AND
# unsafe_test_only_path_audit_waived=True (production paths cannot use the
# escape hatch).
#
# Multi-week ($600-$1200) sustained spend; operator approval REQUIRED AT
# EACH WEEK GATE.
#
# Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

set -euo pipefail

LANE_ID="lane_phase3_joint_scorer_renderer_codec"
PROVIDER="vastai_or_modal"
PREDICTED_DELTA_SCORE="0.115-0.130 [predicted; Phase 3 joint training under Tishby IB Lagrangian]"
COST_BAND_USD_TOTAL_LOW="600"
COST_BAND_USD_TOTAL_HIGH="1200"
COST_BAND_USD_WEEK_GATE="200"   # operator approval at each week boundary

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Phase 3 dispatch (PRE-STAGED-BUT-BLOCKED on Phase 2 completion)"
echo "    Lane: ${LANE_ID}"
echo "    Predicted: ${PREDICTED_DELTA_SCORE}"
echo "    Total cost band: \$${COST_BAND_USD_TOTAL_LOW}-\$${COST_BAND_USD_TOTAL_HIGH}"
echo "    Per-week cap: \$${COST_BAND_USD_WEEK_GATE}"
echo

if [[ "${STAGED_PHASE_DISPATCH_OPERATOR_APPROVED:-0}" != "1" ]]; then
  cat <<EOF
[REFUSE] STAGED_PHASE_DISPATCH_OPERATOR_APPROVED != 1

This script is PRE-STAGED-BUT-BLOCKED on Phase 2 completion. The
Phase3DispatchGate (src/tac/phase3/joint_scorer_renderer_codec.py, Catalog
#134) refuses construction unless ALL 8 preconditions are met:

  (1) phase2_anchor_verified: bool  — Phase 2 [contest-CUDA] anchor verified
  (2) phase2_anchor_score: float    — must be ≤ 0.142 per Phase 3 entry criterion
  (3) phase2_anchor_evidence_path: str — committed repo-relative path
  (4) distillation_gap_estimate: float — must be ≤ 0.03 (T10 dispatch lands this)
  (5) distillation_gap_evidence_path: str — committed repo-relative path
  (6) operator_approved_gpu_budget_usd: float — must be ≥ 600
  (7) aaf68f37_verdict_clean: bool   — adversarial-review verdict CLEAN
  (8) aaf68f37_verdict_evidence_path: str
  (9) phase3_council_deliberation_path: str
      ✓ ALREADY EXISTS: .omx/research/fields_medal_grand_council_all_phases_design_deliberate_implement_20260509.md

Multi-week burn schedule (per dispatch readiness manifest §"$600-$1200 burn schedule"):
  W1  $20-$50    T10 + T15 + T17 + T18 (Phase 2 closeout / Probes)
  W2  $80-$120   Joint scorer-renderer-codec end-to-end SMOKE (Phase 2 ≤0.155 verified)
  W3  $200-$400  Joint training full-data (Phase 2 ≤0.142 verified)
  W4  $200-$400  Distillation refinement + EMA convergence (W3 ≤0.130)
  W5  $100-$280  Floor-saturation polish (W4 ≤0.125)

OPERATOR APPROVAL REQUIRED AT EACH WEEK BOUNDARY. Pause-on-no-improvement.

To dispatch (post-Phase-2-saturation):
  STAGED_PHASE_DISPATCH_OPERATOR_APPROVED=1 \\
  STAGED_PHASE_DISPATCH_LANE=${LANE_ID} \\
  STAGED_PHASE_DISPATCH_BUDGET_USD=${COST_BAND_USD_WEEK_GATE} \\
  STAGED_PHASE_DISPATCH_PHASE3_WEEK=W2 \\
  STAGED_PHASE_DISPATCH_PHASE2_ANCHOR_PATH=experiments/results/<phase2-anchor-dir>/dual_eval_adjudicated.json \\
  STAGED_PHASE_DISPATCH_PHASE2_ANCHOR_SCORE=<float ≤ 0.142> \\
  STAGED_PHASE_DISPATCH_T10_DISTILLATION_GAP=<float ≤ 0.03> \\
  STAGED_PHASE_DISPATCH_T10_DISTILLATION_EVIDENCE_PATH=experiments/results/<t10-anchor-dir>/distillation_gap_estimate.json \\
  STAGED_PHASE_DISPATCH_PHASE3_OPERATOR_AUTH_MEMO=.omx/research/operator_authorizations/phase_b_auth_phase3_<YYYYMMDD>.md \\
  bash $0

Required: place the Phase 3 authorization memo at the canonical path
.omx/research/operator_authorizations/phase_b_auth_phase3_<YYYYMMDD>.md
containing the line:
  operator_phase_b_authorization=true
(per CLAUDE.md Catalog #150).
EOF
  exit 0
fi

# All-preconditions sanity check.
if [[ -z "${STAGED_PHASE_DISPATCH_PHASE2_ANCHOR_PATH:-}" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_PHASE2_ANCHOR_PATH not set." >&2
  exit 3
fi
if [[ ! -f "${STAGED_PHASE_DISPATCH_PHASE2_ANCHOR_PATH}" ]]; then
  echo "[REFUSE] Phase 2 anchor path does not exist: ${STAGED_PHASE_DISPATCH_PHASE2_ANCHOR_PATH}" >&2
  exit 3
fi
if [[ -z "${STAGED_PHASE_DISPATCH_PHASE2_ANCHOR_SCORE:-}" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_PHASE2_ANCHOR_SCORE not set." >&2
  exit 3
fi
# Use awk because bash floats compare poorly.
if ! awk "BEGIN {exit !(${STAGED_PHASE_DISPATCH_PHASE2_ANCHOR_SCORE} <= 0.142)}"; then
  echo "[REFUSE] Phase 2 anchor score ${STAGED_PHASE_DISPATCH_PHASE2_ANCHOR_SCORE} > 0.142 target." >&2
  exit 3
fi
if [[ -z "${STAGED_PHASE_DISPATCH_T10_DISTILLATION_GAP:-}" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_T10_DISTILLATION_GAP not set." >&2
  exit 3
fi
if ! awk "BEGIN {exit !(${STAGED_PHASE_DISPATCH_T10_DISTILLATION_GAP} <= 0.03)}"; then
  echo "[REFUSE] T10 distillation gap ${STAGED_PHASE_DISPATCH_T10_DISTILLATION_GAP} > 0.03 target." >&2
  exit 3
fi
if [[ -z "${STAGED_PHASE_DISPATCH_T10_DISTILLATION_EVIDENCE_PATH:-}" || ! -f "${STAGED_PHASE_DISPATCH_T10_DISTILLATION_EVIDENCE_PATH}" ]]; then
  echo "[REFUSE] T10 distillation evidence path missing/not-a-file." >&2
  exit 3
fi
if [[ -z "${STAGED_PHASE_DISPATCH_PHASE3_OPERATOR_AUTH_MEMO:-}" || ! -f "${STAGED_PHASE_DISPATCH_PHASE3_OPERATOR_AUTH_MEMO}" ]]; then
  echo "[REFUSE] Phase 3 operator auth memo missing." >&2
  exit 3
fi
if [[ "${STAGED_PHASE_DISPATCH_BUDGET_USD}" -gt "${COST_BAND_USD_WEEK_GATE}" ]]; then
  echo "[REFUSE] Per-week cap \$${COST_BAND_USD_WEEK_GATE} exceeded; operator must re-approve." >&2
  exit 3
fi

UTC_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="experiments/results/staged_phase3_${STAGED_PHASE_DISPATCH_PHASE3_WEEK:-W?}_${UTC_TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "==> Provider job not created by this pre-stage script"
echo "    No lane claim opened. Phase 3 is still command-not-wired; a claim is"
echo "    valid only after a real provider job/call id exists."

echo "[REFUSE] Phase 3 dispatch command not yet wired in this script."
echo "    Operator: instantiate the Phase3DispatchGate via"
echo "      .venv/bin/python -m tac.phase3 \\"
echo "        --phase2-anchor-verified \\"
echo "        --phase2-anchor-score ${STAGED_PHASE_DISPATCH_PHASE2_ANCHOR_SCORE} \\"
echo "        --phase2-anchor-evidence-path ${STAGED_PHASE_DISPATCH_PHASE2_ANCHOR_PATH} \\"
echo "        --distillation-gap-estimate ${STAGED_PHASE_DISPATCH_T10_DISTILLATION_GAP} \\"
echo "        --distillation-gap-evidence-path ${STAGED_PHASE_DISPATCH_T10_DISTILLATION_EVIDENCE_PATH} \\"
echo "        --operator-approved-gpu-budget-usd ${STAGED_PHASE_DISPATCH_BUDGET_USD} \\"
echo "        --phase3-council-deliberation-path .omx/research/fields_medal_grand_council_all_phases_design_deliberate_implement_20260509.md \\"
echo "        --phase-b-auth-memo ${STAGED_PHASE_DISPATCH_PHASE3_OPERATOR_AUTH_MEMO} \\"
echo "        --output-dir ${OUTPUT_DIR}"
exit 3
