#!/usr/bin/env bash
# Phase 2 T6 (Ballé + UNIWARD cross-paradigm) dispatch — PRE-STAGED, operator-approval-gated.
#
# BLOCKER: T6 trainer does NOT YET exist. This script is STAGED-BUT-BLOCKED;
# it refuses to dispatch until BOTH (a) a T6 trainer is built AND
# (b) operator approval lands.
#
# Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" non-
# negotiable + Catalog #124 (representation-lane archive grammar at design
# time): T6 must declare all 8 archive-grammar fields before dispatch. The
# Phase 2 manifest documents these as TBD pending trainer build.
#
# Operator decision required (per manifest):
#   1. Approve T6 trainer build (~250 LOC bolt-on per HNeRV parity ≤350 LOC)
#   2. Approve $80 dispatch budget on landing
#
# Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

set -euo pipefail

LANE_ID="lane_t6_balle_uniward_cross_paradigm_phase2_preregistered"
PROVIDER="modal"
PREDICTED_DELTA_SCORE="-0.018 ± 0.008 [predicted; T1+T18 STAR + UNIWARD-budget Ballé cross-paradigm]"
COST_BAND_USD="80"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Phase 2 T6 dispatch (PRE-STAGED-BUT-BLOCKED)"
echo "    Lane: ${LANE_ID}"
echo "    Predicted Δ score: ${PREDICTED_DELTA_SCORE}"
echo "    Cost band: \$${COST_BAND_USD}"
echo

# OPERATOR APPROVAL GATE.
if [[ "${STAGED_PHASE_DISPATCH_OPERATOR_APPROVED:-0}" != "1" ]]; then
  cat <<EOF
[REFUSE] STAGED_PHASE_DISPATCH_OPERATOR_APPROVED != 1

This script is PRE-STAGED-BUT-BLOCKED. Two operator decisions required:

  (1) Approve T6 trainer construction (~250 LOC; HNeRV parity ≤350 LOC budget).
      The trainer must derive from experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py
      with UNIWARD-budget weighting added to the cross-paradigm Lagrangian.
      Required 8 archive-grammar fields (per Catalog #124):
        archive_grammar:       extends T1's 3-member with UNIWARD sidecar mode-1
        parser_section_manifest: 3 T1 members + uniward_budget metadata
        inflate_runtime_loc_budget: +10 LOC over T1
        runtime_dep_closure:   T1 + (no new deps)
        export_format:         phase1_contest_contract with UNIWARD extension
        score_aware_loss:      T1's Lagrangian + UNIWARD-weighted
        bolt_on_loc_budget:    ~250 LOC
        no_op_detector_planned: Phase 1 packet compiler byte-mutation smoke

  (2) Approve \$${COST_BAND_USD} dispatch budget once (1) lands.

To dispatch (post-trainer-build):
  STAGED_PHASE_DISPATCH_OPERATOR_APPROVED=1 \\
  STAGED_PHASE_DISPATCH_LANE=${LANE_ID} \\
  STAGED_PHASE_DISPATCH_BUDGET_USD=${COST_BAND_USD} \\
  STAGED_PHASE_DISPATCH_T6_TRAINER_BUILT=1 \\
  bash $0
EOF
  exit 0
fi

if [[ "${STAGED_PHASE_DISPATCH_T6_TRAINER_BUILT:-0}" != "1" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_T6_TRAINER_BUILT != 1; T6 trainer not yet built." >&2
  echo "    Build experiments/train_t6_balle_uniward_cross_paradigm.py first." >&2
  exit 3
fi

echo "[REFUSE] T6 trainer is REPORTED-BUILT but this script's blocker logic does not yet"
echo "    have a corresponding trainer to call. Operator must update this script after"
echo "    landing the trainer to wire the dispatch command. Until then, this is a no-op gate."
exit 3
