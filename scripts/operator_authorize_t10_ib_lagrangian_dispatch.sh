#!/bin/bash
# Operator authorize: T10 IB Lagrangian aux scorer GPU dispatch ($40 Modal T4).
#
# Per WW's "T10 unique unlock" analysis 2026-05-11: T10 (IB Lagrangian
# co-trained aux scorer) is the canonical unblock for W's DEFERRED criteria
# #1-#4 (Phase 3 joint-scorer-renderer-codec readiness gates). Phase 3 dispatch
# requires T10 anchor empirical evidence on the [contest-CUDA] axis.
#
# Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
# CONTEST-COMPLIANT HARDWARE": this dispatch produces a CUDA-axis anchor;
# the CPU pair is a separate dispatch.
#
# Per CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION": claims the lane via
# `tools/claim_lane_dispatch.py claim` BEFORE the network call to Modal.
#
# Per CLAUDE.md "GPU budget" + the operator $20 envelope: $40 EXCEEDS the
# $20 envelope; this script requires fresh operator approval via the
# inline confirmation prompt + AskUserQuestion in the calling session.
#
# Cost: ~$40 on Modal T4 (8 hours @ ~$5/hr; conservative)
# Predicted Δ: Phase 3 readiness gate — direct score Δ unknown until
#              empirical eval; opens 4 DEFERRED lanes that themselves
#              produce predicted -0.005-0.030 sub-floor.
# Risk: T10 IB Lagrangian aux scorer is novel architecture; first dispatch
#       carries empirical-validation risk (council 2026-05-09 verdict B-4).
#
# Usage: bash scripts/operator_authorize_t10_ib_lagrangian_dispatch.sh
#
# Lane: lane_t10_ib_lagrangian_aux_scorer (must exist in lane_registry.json)
# Cross-ref: project_operator_decision_dashboard_20260511.md B-4 / B-9
#            feedback_grand_council_fields_medal_phase2_floor_refinement_20260509.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LANE_ID="lane_t10_ib_lagrangian_aux_scorer"
EXPECTED_COST_USD="40.00"
PLATFORM="${T10_PLATFORM:-modal}"

cat <<EOF

=== T10 IB Lagrangian dispatch operator confirmation ===

lane_id:                 ${LANE_ID}
platform:                ${PLATFORM} (Modal T4 default; override with T10_PLATFORM=)
expected cost:           \$${EXPECTED_COST_USD} USD
predicted Δ:             Phase 3 readiness gate (unblocks 4 DEFERRED lanes)
                         Direct score Δ unknown until empirical eval.
risk:                    Novel architecture; empirical-validation risk on
                         first dispatch (council verdict B-4 2026-05-09).
envelope status:         EXCEEDS \$20 cumulative envelope.

dependencies:
  - tools/claim_lane_dispatch.py (lane-claim helper present)
  - scripts/remote_lane_t10_ib_lagrangian_aux_scorer.sh (remote driver)
  - .omx/state/active_lane_dispatch_claims.md (claim ledger)

PRE-FLIGHT CHECK — verify these before authorizing:
  [ ] T10 trainer module exists under src/tac (or experiments/)
  [ ] Phase 2 anchor saturation evidence per W's DEFERRED criteria #1-#4
  [ ] Fresh operator approval (this exceeds the standing \$20 envelope)

EOF
read -r -p "Proceed with T10 dispatch (~\$40 Modal T4)? [y/N] " confirm
case "$confirm" in
    [yY]|[yY][eE][sS])
        ;;
    *)
        echo "[t10-dispatch] aborted — no dispatch fired"
        exit 0
        ;;
esac

# Lane-claim coordination (per CLAUDE.md CROSS-AGENT DISPATCH COORDINATION).
INSTANCE_JOB_ID="t10_ib_lagrangian_$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python tools/claim_lane_dispatch.py claim \
    --lane-id "$LANE_ID" \
    --platform "$PLATFORM" \
    --instance-job-id "$INSTANCE_JOB_ID" \
    --agent "claude:operator_authorize_t10_ib_lagrangian_dispatch" \
    --status "active_dispatch" \
    --notes "operator-authorized T10 IB Lagrangian dispatch via scripts/operator_authorize_t10_ib_lagrangian_dispatch.sh; estimated cost ~\$${EXPECTED_COST_USD}"

REMOTE_DRIVER="scripts/remote_lane_t10_ib_lagrangian_aux_scorer.sh"
if [ ! -f "$REMOTE_DRIVER" ]; then
    cat >&2 <<EOM

[t10-dispatch] WARN: canonical remote driver missing at ${REMOTE_DRIVER}.

The lane is claimed but no remote driver script exists yet. Options:
  1. Build the remote driver script (sister of scripts/remote_lane_t1_balle_endtoend.sh)
     before re-running this authorize script.
  2. Manually dispatch via Modal/Vast.ai using the lane-claim ledger as
     coordination.

Per CLAUDE.md "Operator gates must be wired and used", this exit-with-warning
is the safe default. Lane claim is still active and will block sibling
dispatchers per the 24h TTL.
EOM
    exit 3
fi

# Delegate to the canonical remote driver. The driver verifies the lane claim
# matches its expected INSTANCE_JOB_ID + handles closure.
T10_DISPATCH_INSTANCE_JOB_ID="$INSTANCE_JOB_ID" \
    DISPATCH_PLATFORM="$PLATFORM" \
    bash "$REMOTE_DRIVER"

echo "[t10-dispatch] complete; review active dispatch claims ledger:"
echo "  .omx/state/active_lane_dispatch_claims.md"
