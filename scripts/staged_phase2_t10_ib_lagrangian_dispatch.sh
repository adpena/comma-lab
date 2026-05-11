#!/usr/bin/env bash
# Phase 2 T10 (IB-Lagrangian co-trained aux scorer) — PRE-STAGED, BLOCKED.
#
# T10 is the Phase 3 prerequisite per Phase3DispatchGate (Catalog #134):
# requires distillation_gap_estimate ≤ 0.03. T10 must dispatch BEFORE Phase 3
# and produces the distillation gap evidence that gates Phase 3.
#
# BLOCKER: T10 trainer does NOT YET exist. This script is STAGED-BUT-BLOCKED.
#
# Per CLAUDE.md HNeRV parity discipline (Catalog #124) + "Tishby IB
# Lagrangian" non-negotiable. Aux scorer co-distilled via Hinton T=2.0.
#
# Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

set -euo pipefail

LANE_ID="lane_t10_ib_lagrangian_aux_scorer_phase2_preregistered"
PROVIDER="modal"
PREDICTED_DELTA_SCORE="distillation_gap < 0.03 [Phase 3 prerequisite; no direct score Δ]"
COST_BAND_USD="40"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Phase 2 T10 dispatch (PRE-STAGED-BUT-BLOCKED)"
echo "    Lane: ${LANE_ID}"
echo "    Predicted: ${PREDICTED_DELTA_SCORE}"
echo "    Cost band: \$${COST_BAND_USD}"
echo

if [[ "${STAGED_PHASE_DISPATCH_OPERATOR_APPROVED:-0}" != "1" ]]; then
  cat <<EOF
[REFUSE] STAGED_PHASE_DISPATCH_OPERATOR_APPROVED != 1

This script is PRE-STAGED-BUT-BLOCKED. Required:
  (1) Build experiments/train_t10_ib_lagrangian_aux_scorer.py (~300 LOC)
      with Tishby IB Lagrangian L = I(X;T) − β·I(T;Y) + Hinton T=2.0 distill
  (2) Approve \$${COST_BAND_USD} dispatch budget
  (3) Recognize T10 is a Phase 3 prerequisite, not a score-band candidate

To dispatch (post-trainer-build):
  STAGED_PHASE_DISPATCH_OPERATOR_APPROVED=1 \\
  STAGED_PHASE_DISPATCH_LANE=${LANE_ID} \\
  STAGED_PHASE_DISPATCH_BUDGET_USD=${COST_BAND_USD} \\
  STAGED_PHASE_DISPATCH_T10_TRAINER_BUILT=1 \\
  bash $0
EOF
  exit 0
fi

if [[ "${STAGED_PHASE_DISPATCH_T10_TRAINER_BUILT:-0}" != "1" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_T10_TRAINER_BUILT != 1; trainer not yet built." >&2
  exit 3
fi

echo "[REFUSE] T10 trainer reported-built; operator must wire dispatch command in this script."
exit 3
