#!/usr/bin/env bash
# Phase 2 T6 (Ballé + UNIWARD cross-paradigm) dispatch — STAGED, operator-approval-gated.
#
# T6 trainer EXISTS as of 2026-05-11:
#   experiments/train_t6_balle_uniward_cross_paradigm.py (~520 LOC)
#   tests/test_train_t6_balle_uniward_cross_paradigm.py (20 tests, 20/20 PASS)
#   Lane: lane_t6_balle_uniward_cross_paradigm_phase2_preregistered (L1)
#
# Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" non-
# negotiable + Catalog #124 (representation-lane archive grammar at design
# time): T6 declares all 8 archive-grammar fields in its docstring.
#
# Operator decision required:
#   1. Approve $80 dispatch budget (Phase 2 envelope $223-303 covers T6+T10+...)
#   2. Provide a contest-CUDA provider (Modal T4 / Lightning / Vast.ai 4090)
#   3. Provide --canonical-a1-relpath path to A1 EMA shadow checkpoint
#
# Predicted Δ score:
#   -0.018 ± 0.008 [predicted; T1+T18 STAR + UNIWARD-budget Ballé cross-paradigm]
#
# NOT a score claim until contest-CUDA anchor lands.
#
# Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

set -euo pipefail

LANE_ID="lane_t6_balle_uniward_cross_paradigm_phase2_preregistered"
PROVIDER="modal_or_lightning_or_vastai"
PREDICTED_DELTA_SCORE="-0.018 ± 0.008 [predicted; T1+T18 STAR + UNIWARD-budget Ballé cross-paradigm]"
COST_BAND_USD="80"
TRAINER_PATH="experiments/train_t6_balle_uniward_cross_paradigm.py"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Phase 2 T6 dispatch (STAGED)"
echo "    Lane: ${LANE_ID}"
echo "    Trainer: ${TRAINER_PATH}"
echo "    Predicted Δ score: ${PREDICTED_DELTA_SCORE}"
echo "    Cost band: \$${COST_BAND_USD}"
echo

# OPERATOR APPROVAL GATE.
if [[ "${STAGED_PHASE_DISPATCH_OPERATOR_APPROVED:-0}" != "1" ]]; then
  cat <<EOF
[REFUSE] STAGED_PHASE_DISPATCH_OPERATOR_APPROVED != 1

This script is STAGED. The T6 trainer is BUILT (commit landing 2026-05-11):
  ${TRAINER_PATH}

Required operator decisions:

  (1) Approve \$${COST_BAND_USD} dispatch budget on a contest-CUDA provider
      (Phase 2 envelope per ITEM 2 in NOT-YET memo; envelope is \$223-303
      total for T6+T10+T15+T17+T18 combined).
  (2) Choose provider: Modal T4 (\$0.59/hr), Lightning (\$0.45/hr), or
      Vast.ai 4090 (\$0.25/hr).
  (3) Provide --canonical-a1-relpath to A1 EMA shadow checkpoint
      (default: experiments/results/A1_canonical).
  (4) Open dispatch claim via tools/claim_lane_dispatch.py first (per
      CLAUDE.md "Cross-agent dispatch coordination").

To smoke-test locally first (free, CPU, ~10s, no archive):
  .venv/bin/python ${TRAINER_PATH} \\
      --output-dir experiments/results/t6_local_smoke \\
      --device cpu --smoke --allow-missing-canonical-a1 \\
      --epochs 1

To dispatch when ready (operator wires actual provider invocation):
  STAGED_PHASE_DISPATCH_OPERATOR_APPROVED=1 \\
  STAGED_PHASE_DISPATCH_LANE=${LANE_ID} \\
  STAGED_PHASE_DISPATCH_BUDGET_USD=${COST_BAND_USD} \\
  STAGED_PHASE_DISPATCH_T6_TRAINER_BUILT=1 \\
  bash $0

This script then prints (does NOT execute) the canonical training command
that the operator wires to a provider invocation.
EOF
  exit 0
fi

if [[ "${STAGED_PHASE_DISPATCH_T6_TRAINER_BUILT:-0}" != "1" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_T6_TRAINER_BUILT != 1." >&2
  exit 3
fi

if [[ "${STAGED_PHASE_DISPATCH_BUDGET_USD:-0}" -gt "${COST_BAND_USD}" ]]; then
  echo "[REFUSE] Budget cap exceeded (\$${STAGED_PHASE_DISPATCH_BUDGET_USD} > \$${COST_BAND_USD})." >&2
  exit 3
fi

UTC_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="experiments/results/staged_phase2_t6_${UTC_TIMESTAMP}"

cat <<EOF
==> Canonical training command (operator wires to provider):

  .venv/bin/python ${TRAINER_PATH} \\
      --output-dir ${OUTPUT_DIR} \\
      --device cuda \\
      --epochs 200 \\
      --batch-size 8 \\
      --learning-rate 1e-4 \\
      --rate-target-bytes 80000 \\
      --seg-target 7e-4 \\
      --pose-target 1.7e-4 \\
      --t6-flat-floor 0.5 \\
      --t6-textured-ceiling 1.5 \\
      --canonical-a1-relpath experiments/results/A1_canonical

Lane claim:
  .venv/bin/python tools/claim_lane_dispatch.py claim \\
      --lane-id ${LANE_ID} \\
      --instance-or-job-id <provider_job_id> \\
      --status active_dispatching

Provider invocation is NOT wired in this pre-stage script (per pre-stage G
manifest: stays staged-but-blocked on operator approval per dispatch).
EOF
exit 0
