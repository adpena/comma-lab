#!/bin/bash
# Operator authorize: T10 IB Lagrangian aux scorer GPU dispatch (~$40 Modal T4).
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
# Per CLAUDE.md "GPU budget" + the operator $20 envelope: ~$40 EXCEEDS the
# $20 envelope; this script requires fresh operator approval via the
# inline confirmation prompt + AskUserQuestion in the calling session.
#
# W/I/A I-2 backport (2026-05-12, decision I-2): the hand-derived $40 literal
# is replaced with a runtime call to the cost-band posterior at
# `.omx/state/cost_band_posterior.jsonl`. At this scale (8 h Modal T4) the
# posterior is COLD-START with no matching anchors yet — predict() returns
# the hand_calibrated_fallback band centered around $40. The OD-CB-1
# wire-in pattern (commit 5eb355aa + 0666720a) makes the cost source
# explicit AND keeps the envelope check honest as anchors accumulate.
#
# Cost band: see runtime call below; hand-derived fallback ~$40 Modal T4
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
PLATFORM="${T10_PLATFORM:-modal}"

# W/I/A I-2 fix (2026-05-12): replace stale $40 literal with runtime call
# to the cost-band posterior. T10 is 8 h on Modal T4; the posterior is
# cold-start for this bucket so predict() emits a hand_calibrated_fallback
# band. As empirical anchors land via T10 completions, the band will tighten.
MODAL_GPU="${MODAL_GPU:-T4}"
PLATFORM_KEY="modal"
GPU_CLASS="${MODAL_GPU}"
T10_EPOCHS_LONG=8000   # 8 hour-scale long-train bucket; matches posterior tag
COST_BAND_TEXT=$(.venv/bin/python -c "
from tac.cost_band_calibration import predict
p = predict('${PLATFORM_KEY}', '${GPU_CLASS}', ${T10_EPOCHS_LONG}, all_flags_on=True)
print(f'\${p.p10_cost_usd:.2f}/\${p.p50_cost_usd:.2f}/\${p.p90_cost_usd:.2f}'
      f'  (N={p.n_anchors}, {p.confidence_tag})')
" 2>/dev/null || echo "predict() unavailable — hand-calibrated fallback \$40.00")
EXPECTED_COST_USD=$(.venv/bin/python -c "
from tac.cost_band_calibration import predict
p = predict('${PLATFORM_KEY}', '${GPU_CLASS}', ${T10_EPOCHS_LONG}, all_flags_on=True)
# Cold-start bucket (no matching anchors AND no hand-stub for 8000-epoch
# long-train) falls back to the $40 literal so the operator banner stays
# informative while the posterior accumulates anchors.
print(f'{p.p50_cost_usd:.2f}' if p.p50_cost_usd > 0 else '40.00')
" 2>/dev/null || echo "40.00")
CONFIDENCE_TAG=$(.venv/bin/python -c "
from tac.cost_band_calibration import predict
print(predict('${PLATFORM_KEY}', '${GPU_CLASS}', ${T10_EPOCHS_LONG}, all_flags_on=True).confidence_tag)
" 2>/dev/null || echo "hand_calibrated_fallback")

cat <<EOF

=== T10 IB Lagrangian dispatch operator confirmation ===

lane_id:                 ${LANE_ID}
platform:                ${PLATFORM} (Modal ${MODAL_GPU} default; override T10_PLATFORM / MODAL_GPU)
cost band p10/p50/p90:   ${COST_BAND_TEXT}
                         Source: tac.cost_band_calibration.predict()
                         Posterior: .omx/state/cost_band_posterior.jsonl
                         Confidence: ${CONFIDENCE_TAG}
expected p50 cost:       \$${EXPECTED_COST_USD} USD
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
