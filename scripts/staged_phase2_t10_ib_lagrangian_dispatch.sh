#!/usr/bin/env bash
# Phase 2 T10 (IB-Lagrangian co-trained aux scorer) — STAGED, operator-gated.
#
# T10 trainer EXISTS as of 2026-05-11:
#   experiments/train_t10_ib_lagrangian_aux_scorer.py (~440 LOC)
#   tests/test_train_t10_ib_lagrangian_aux_scorer.py (18 tests, 18/18 PASS)
#   Lane: lane_t10_ib_lagrangian_aux_scorer_phase2_preregistered (L1)
#
# T10 is the Phase 3 prerequisite per Phase3DispatchGate (Catalog #134):
# requires distillation_gap_estimate ≤ 0.03.  T10 must dispatch BEFORE
# Phase 3 and produces the distillation_gap_estimate.json artifact that
# gates Phase 3.
#
# Per CLAUDE.md HNeRV parity discipline + Hinton 2014 §3.  Aux scorer
# co-distilled via Hinton T=2.0; output is EMA shadow + distillation gap.
#
# Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

set -euo pipefail

LANE_ID="lane_t10_ib_lagrangian_aux_scorer_phase2_preregistered"
PROVIDER="modal_or_lightning_or_vastai"
PREDICTED_DELTA_SCORE="N/A — Phase 3 prerequisite (gates Phase3DispatchGate distillation_gap_estimate ≤ 0.03 per Catalog #134); no direct score Δ on its own"
COST_BAND_USD="40"
TRAINER_PATH="experiments/train_t10_ib_lagrangian_aux_scorer.py"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Phase 2 T10 dispatch (STAGED)"
echo "    Lane: ${LANE_ID}"
echo "    Trainer: ${TRAINER_PATH}"
echo "    Predicted: ${PREDICTED_DELTA_SCORE}"
echo "    Cost band: \$${COST_BAND_USD}"
echo

if [[ "${STAGED_PHASE_DISPATCH_OPERATOR_APPROVED:-0}" != "1" ]]; then
  cat <<EOF
[REFUSE] STAGED_PHASE_DISPATCH_OPERATOR_APPROVED != 1

This script is STAGED. The T10 trainer is BUILT (commit landing 2026-05-11):
  ${TRAINER_PATH}

Required operator decisions:

  (1) Approve \$${COST_BAND_USD} dispatch budget on a contest-CUDA provider
      (T10 dispatch is part of Phase 2 envelope \$223-303; T10 alone is \$40).
  (2) Recognize T10 is a Phase 3 prerequisite, not a score-band candidate.
      T10's deliverable is distillation_gap_estimate.json with gap ≤ 0.03.
  (3) Open dispatch claim via tools/claim_lane_dispatch.py first.

To smoke-test locally first (free, CPU, ~4s, no contest scorer):
  .venv/bin/python ${TRAINER_PATH} \\
      --output-dir experiments/results/t10_local_smoke \\
      --device cpu --smoke \\
      --epochs 1 --n-batches 2

To dispatch when ready:
  STAGED_PHASE_DISPATCH_OPERATOR_APPROVED=1 \\
  STAGED_PHASE_DISPATCH_LANE=${LANE_ID} \\
  STAGED_PHASE_DISPATCH_BUDGET_USD=${COST_BAND_USD} \\
  STAGED_PHASE_DISPATCH_T10_TRAINER_BUILT=1 \\
  bash $0
EOF
  exit 0
fi

if [[ "${STAGED_PHASE_DISPATCH_T10_TRAINER_BUILT:-0}" != "1" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_T10_TRAINER_BUILT != 1." >&2
  exit 3
fi

if [[ "${STAGED_PHASE_DISPATCH_BUDGET_USD:-0}" -gt "${COST_BAND_USD}" ]]; then
  echo "[REFUSE] Budget cap exceeded (\$${STAGED_PHASE_DISPATCH_BUDGET_USD} > \$${COST_BAND_USD})." >&2
  exit 3
fi

UTC_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="experiments/results/staged_phase2_t10_${UTC_TIMESTAMP}"

cat <<EOF
==> Canonical training command (operator wires to provider):

  .venv/bin/python ${TRAINER_PATH} \\
      --output-dir ${OUTPUT_DIR} \\
      --device cuda \\
      --epochs 6 \\
      --batch-size 8 \\
      --learning-rate 1e-4 \\
      --distill-temperature 2.0 \\
      --lambda-gt 0.5 \\
      --ema-decay 0.997

The trainer wires the contest SegNet + PoseNet forward (via
``tac.scorer.load_differentiable_scorers``) when not in --smoke mode.
The dispatch script body in this scaffold currently REFUSES the non-smoke
path with a structured operator-gated message — the operator is
expected to wire the real dataloader + contest-scorer forward at
dispatch time (since the real frame loading requires upstream/videos/0.mkv
on the GPU host).

Output to monitor:
  ${OUTPUT_DIR}/distillation_gap_estimate.json
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                Phase 3 prereq artifact
                                                must have gap ≤ 0.03 to
                                                unblock Phase3DispatchGate.

Lane claim:
  .venv/bin/python tools/claim_lane_dispatch.py claim \\
      --lane-id ${LANE_ID} \\
      --instance-or-job-id <provider_job_id> \\
      --status active_dispatching
EOF
exit 0
