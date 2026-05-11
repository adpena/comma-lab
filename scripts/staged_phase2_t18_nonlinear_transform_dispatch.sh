#!/usr/bin/env bash
# Phase 2 T18 (Ballé nonlinear transform coding) — PRE-STAGED, wiring-gap + HARD GATE.
#
# T18 module scaffold exists at src/tac/balle_nonlinear_transform.py
# (11.9KB, 19 tests passing per pre-design pass). T18 trainer needs ~280
# LOC bolt-on derivation. HARD GATE: Probe T18-B must demonstrate net-byte-
# savings > 0; if HARD GATE FAILS, T18 is DEFERRED-PENDING-PHASE-3-
# DISTILLATION ($8 spent on T18-B; $35 saved on conditional inclusion).
#
# NN-3 monitoring requirement: per-100-step invertibility check
#   ||z_e - invert(forward(z_e))||² < 0.5
# with halt-on-breach.
#
# Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

set -euo pipefail

LANE_ID="lane_t18_balle_nonlinear_transform_phase2_preregistered"
PROVIDER="modal"
PREDICTED_DELTA_SCORE="-0.003 ± 0.002 [predicted; He-Zheng 2024 BD-rate; conditional on T18-B HARD GATE]"
COST_BAND_USD_CONDITIONAL="43"
COST_BAND_USD_PROBE_ONLY="8"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Phase 2 T18 dispatch (PRE-STAGED-WITH-WIRING-GAP + HARD GATE)"
echo "    Lane: ${LANE_ID}"
echo "    Predicted: ${PREDICTED_DELTA_SCORE}"
echo "    Cost band (conditional): \$${COST_BAND_USD_CONDITIONAL}"
echo "    Cost band (HARD GATE fails): \$${COST_BAND_USD_PROBE_ONLY}"
echo

if [[ "${STAGED_PHASE_DISPATCH_OPERATOR_APPROVED:-0}" != "1" ]]; then
  cat <<EOF
[REFUSE] STAGED_PHASE_DISPATCH_OPERATOR_APPROVED != 1

This script is PRE-STAGED-WITH-WIRING-GAP + HARD GATE. Required:
  (1) Derive experiments/train_t18_nonlinear_transform_t1_clone.py from T1
      trainer with nonlinear transform coding wiring:
        - GELU activation (matches scaffold + He-Zheng 2024 canon)
        - latent_dim shrunk 192 → 64 (MacKay MDL critique per pre-design pass)
        - 3 hidden layers (= 4-layer MLP) per He-Zheng 2024 §3.2 sweet spot
        - Mixed precision: FP16 train, FP4 ship
        - NN-3 sustained invertibility monitor (per-100-step halt-on-breach):
            assert ||z_e − invert(forward(z_e))||² < 0.5
  (2) Probe T18-A (\$4 GPU) latent-dim/depth sweep: confirm winner in
      {64, 128} × {2, 3}
  (3) **HARD GATE** Probe T18-B (\$4 GPU smoke): demonstrate
      **net byte savings > 0** vs T1 baseline. If FAILS, T18 is DEFERRED.
  (4) Approve \$${COST_BAND_USD_CONDITIONAL} budget if T18-B passes,
      \$${COST_BAND_USD_PROBE_ONLY} if it fails.

To dispatch the PROBE T18-B HARD GATE first:
  STAGED_PHASE_DISPATCH_OPERATOR_APPROVED=1 \\
  STAGED_PHASE_DISPATCH_LANE=${LANE_ID} \\
  STAGED_PHASE_DISPATCH_BUDGET_USD=${COST_BAND_USD_PROBE_ONLY} \\
  STAGED_PHASE_DISPATCH_T18_PROBE_B_ONLY=1 \\
  bash $0

To dispatch FULL T18 (HARD GATE must already have passed):
  STAGED_PHASE_DISPATCH_OPERATOR_APPROVED=1 \\
  STAGED_PHASE_DISPATCH_LANE=${LANE_ID} \\
  STAGED_PHASE_DISPATCH_BUDGET_USD=${COST_BAND_USD_CONDITIONAL} \\
  STAGED_PHASE_DISPATCH_T18_WIRED=1 \\
  STAGED_PHASE_DISPATCH_T18_PROBE_A_PASSED=1 \\
  STAGED_PHASE_DISPATCH_T18_PROBE_B_PASSED=1 \\
  bash $0
EOF
  exit 0
fi

if [[ "${STAGED_PHASE_DISPATCH_T18_PROBE_B_ONLY:-0}" == "1" ]]; then
  echo "[STAGED] PROBE T18-B HARD GATE only; \$${STAGED_PHASE_DISPATCH_BUDGET_USD} budget."
  echo "[REFUSE] Probe T18-B dispatch command not yet wired in this script; operator must wire."
  exit 3
fi

if [[ "${STAGED_PHASE_DISPATCH_T18_WIRED:-0}" != "1" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_T18_WIRED != 1; trainer not yet wired." >&2
  exit 3
fi
if [[ "${STAGED_PHASE_DISPATCH_T18_PROBE_A_PASSED:-0}" != "1" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_T18_PROBE_A_PASSED != 1; run Probe T18-A first." >&2
  exit 3
fi
if [[ "${STAGED_PHASE_DISPATCH_T18_PROBE_B_PASSED:-0}" != "1" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_T18_PROBE_B_PASSED != 1; HARD GATE not yet cleared." >&2
  exit 3
fi

echo "[REFUSE] T18 wiring + HARD GATE passed; operator must wire dispatch command in this script."
exit 3
