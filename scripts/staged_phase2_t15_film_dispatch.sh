#!/usr/bin/env bash
# Phase 2 T15 (Time-varying FiLM) — PRE-STAGED, wiring-gap.
#
# T15 module scaffold exists at src/tac/film_time_varying.py (12.9KB, 21
# tests passing per pre-design pass). T15 trainer needs derivation from T1
# with ~50 LOC modulator wiring. Per pre-design pass §3 NN-1 monitoring
# requirement: T15 gradient-flow regression test MUST pass before dispatch.
#
# Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

set -euo pipefail

LANE_ID="lane_t15_time_varying_film_phase2_preregistered"
PROVIDER="modal"
PREDICTED_DELTA_SCORE="-0.005 ± 0.003 [predicted; Berger pose; ρ_pose=0.85 fallback per pre-design pass]"
COST_BAND_USD="28"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Phase 2 T15 dispatch (PRE-STAGED-WITH-WIRING-GAP)"
echo "    Lane: ${LANE_ID}"
echo "    Predicted: ${PREDICTED_DELTA_SCORE}"
echo "    Cost band: \$${COST_BAND_USD}"
echo

if [[ "${STAGED_PHASE_DISPATCH_OPERATOR_APPROVED:-0}" != "1" ]]; then
  cat <<EOF
[REFUSE] STAGED_PHASE_DISPATCH_OPERATOR_APPROVED != 1

This script is PRE-STAGED-WITH-WIRING-GAP. Required:
  (1) Derive experiments/train_t15_film_t1_clone.py from T1 trainer
      (+~50 LOC for FiLM modulator activation toggle + EMA wrapping)
  (2) Verify NN-1 gradient-flow regression test passes:
      .venv/bin/pytest src/tac/tests/test_film_time_varying.py -v
      (assert modulator L2 movement > 1e-3 over 5 Adam steps)
  (3) Approve \$${COST_BAND_USD} dispatch budget
  (4) Run Probe T15-A (\$4) smoke first

To dispatch (post-wiring):
  STAGED_PHASE_DISPATCH_OPERATOR_APPROVED=1 \\
  STAGED_PHASE_DISPATCH_LANE=${LANE_ID} \\
  STAGED_PHASE_DISPATCH_BUDGET_USD=${COST_BAND_USD} \\
  STAGED_PHASE_DISPATCH_T15_WIRED=1 \\
  STAGED_PHASE_DISPATCH_T15_NN1_PASSED=1 \\
  bash $0
EOF
  exit 0
fi

if [[ "${STAGED_PHASE_DISPATCH_T15_WIRED:-0}" != "1" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_T15_WIRED != 1; trainer not yet wired." >&2
  exit 3
fi

if [[ "${STAGED_PHASE_DISPATCH_T15_NN1_PASSED:-0}" != "1" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_T15_NN1_PASSED != 1; NN-1 gradient-flow regression must pass first." >&2
  echo "    Run: .venv/bin/pytest src/tac/tests/test_film_time_varying.py -v" >&2
  exit 3
fi

echo "[REFUSE] T15 wiring + NN-1 passed; operator must wire dispatch command in this script."
exit 3
