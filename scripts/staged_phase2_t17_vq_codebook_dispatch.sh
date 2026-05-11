#!/usr/bin/env bash
# Phase 2 T17 (Shared VQ-VAE codebook) — PRE-STAGED, wiring-gap.
#
# T17 module scaffold exists at src/tac/shared_vq_codebook.py (15.1KB, 24
# tests passing per pre-design pass). T17 trainer needs derivation from T1
# with ~300 LOC VQ-VAE wiring + persistent EMA (van den Oord §3.2). NN-2
# monitoring requirement: per-epoch perplexity ≥ 102 (= 0.4·256) gate with
# dead-entry re-initialization.
#
# Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

set -euo pipefail

LANE_ID="lane_t17_shared_vq_codebook_phase2_preregistered"
PROVIDER="modal"
PREDICTED_DELTA_SCORE="-0.006 ± 0.003 [predicted; van den Oord BD-rate; 256-entry × 64-dim @ FP16]"
COST_BAND_USD="32"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Phase 2 T17 dispatch (PRE-STAGED-WITH-WIRING-GAP)"
echo "    Lane: ${LANE_ID}"
echo "    Predicted: ${PREDICTED_DELTA_SCORE}"
echo "    Cost band: \$${COST_BAND_USD}"
echo

if [[ "${STAGED_PHASE_DISPATCH_OPERATOR_APPROVED:-0}" != "1" ]]; then
  cat <<EOF
[REFUSE] STAGED_PHASE_DISPATCH_OPERATOR_APPROVED != 1

This script is PRE-STAGED-WITH-WIRING-GAP. Required:
  (1) Derive experiments/train_t17_shared_vq_t1_clone.py from T1 trainer
      with VQ-VAE codebook integration:
        - 256-entry codebook × 64-dim @ FP16 (32,768 bytes)
        - Persistent EMA codebook (van den Oord §3.2; decay 0.99 per
          CLAUDE.md "EMA — non-negotiable" exception clause)
        - Commitment loss β=0.25 (van den Oord canonical)
        - NN-2 perplexity gate: assert perplexity >= 0.4 * num_entries
          per epoch; re-init dead entries on breach
  (2) Probe T17-A (\$5 CPU) k-means analysis: confirm natural cluster count
      ∈ [128, 512] on A1 encoder outputs
  (3) Probe T17-B (\$2 GPU smoke) persistent-EMA collapse: verify codebook
      does NOT collapse on Comma's continuous-features substrate
  (4) Approve \$${COST_BAND_USD} dispatch budget

To dispatch (post-wiring):
  STAGED_PHASE_DISPATCH_OPERATOR_APPROVED=1 \\
  STAGED_PHASE_DISPATCH_LANE=${LANE_ID} \\
  STAGED_PHASE_DISPATCH_BUDGET_USD=${COST_BAND_USD} \\
  STAGED_PHASE_DISPATCH_T17_WIRED=1 \\
  STAGED_PHASE_DISPATCH_T17_PROBE_A_PASSED=1 \\
  STAGED_PHASE_DISPATCH_T17_PROBE_B_PASSED=1 \\
  bash $0
EOF
  exit 0
fi

if [[ "${STAGED_PHASE_DISPATCH_T17_WIRED:-0}" != "1" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_T17_WIRED != 1; trainer not yet wired." >&2
  exit 3
fi
if [[ "${STAGED_PHASE_DISPATCH_T17_PROBE_A_PASSED:-0}" != "1" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_T17_PROBE_A_PASSED != 1; run Probe T17-A first." >&2
  exit 3
fi
if [[ "${STAGED_PHASE_DISPATCH_T17_PROBE_B_PASSED:-0}" != "1" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_T17_PROBE_B_PASSED != 1; run Probe T17-B first." >&2
  exit 3
fi

echo "[REFUSE] T17 wiring + probes passed; operator must wire dispatch command in this script."
exit 3
