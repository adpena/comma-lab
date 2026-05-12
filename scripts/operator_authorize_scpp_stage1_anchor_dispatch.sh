#!/bin/bash
# Operator authorize: SC++ Stage 1 anchor smoke dispatch (Modal T4).
#
# Per VV+QQ substrate composition matrix analysis 2026-05-11: SC++ substrate
# (block-FP weight self-compression per Selfcomp/szabolcs-cs) is predicted to
# dominate the S_floor low end. Stage 1 is the smoke-anchor dispatch (3 epochs)
# that produces the first [contest-CUDA] empirical anchor for the SC++
# architecture class.
#
# Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lesson 7:
# substrate engineering may exceed the 350-LOC bolt-on budget (tagged
# lane_class=substrate_engineering); this lane is in that category.
#
# Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
# CONTEST-COMPLIANT HARDWARE": this dispatch produces a CUDA-axis anchor; CPU
# paired eval is a separate dispatch.
#
# Per CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION": claims the lane via
# tools/claim_lane_dispatch.py BEFORE Modal launch.
#
# Cost: ~$3 Modal T4 (3 epochs smoke; conservative)
# Predicted Δ: -0.005 to -0.010 [predicted; SC++ block-FP per HNeRV parity
#              discipline lesson 7]
# Risk: Stage 1 smoke is the first empirical SC++ anchor; falsification risk
#       per the SCPP family (see B-10 in operator decision dashboard).
#
# Usage: bash scripts/operator_authorize_scpp_stage1_anchor_dispatch.sh
#
# Lane: lane_scpp_stage1_smoke_anchor (must be registered if not yet)
# Cross-ref: project_operator_decision_dashboard_20260511.md B-10

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LANE_ID="lane_scpp_stage1_smoke_anchor"
EXPECTED_COST_USD="3.00"
PLATFORM="modal"

cat <<EOF

=== SC++ Stage 1 anchor dispatch operator confirmation ===

lane_id:                 ${LANE_ID}
platform:                ${PLATFORM} (Modal T4)
expected cost:           \$${EXPECTED_COST_USD} USD (3-epoch smoke)
predicted Δ:             -0.005 to -0.010 [predicted; SC++ block-FP;
                         HNeRV parity discipline lesson 7]
risk:                    First empirical SC++ anchor — falsification risk
                         per the SCPP family (operator dashboard B-10).
                         Substrate engineering lane (>350 LOC budget OK).
envelope status:         FITS \$5/individual cap with ~1.67× headroom.

pre-flight check:
  - SC++ trainer module present (src/tac/selfcomp_block_fp.py or similar)
  - scripts/remote_lane_scpp_stage1.sh remote driver (build if missing)
  - tools/claim_lane_dispatch.py present
  - .omx/state/active_lane_dispatch_claims.md present

EOF
read -r -p "Proceed with SC++ Stage 1 anchor dispatch (~\$${EXPECTED_COST_USD} Modal T4)? [y/N] " confirm
case "$confirm" in
    [yY]|[yY][eE][sS])
        ;;
    *)
        echo "[scpp-stage1] aborted — no dispatch fired"
        exit 0
        ;;
esac

INSTANCE_JOB_ID="scpp_stage1_$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python tools/claim_lane_dispatch.py claim \
    --lane-id "$LANE_ID" \
    --platform "$PLATFORM" \
    --instance-job-id "$INSTANCE_JOB_ID" \
    --agent "claude:operator_authorize_scpp_stage1_anchor_dispatch" \
    --status "active_dispatch" \
    --notes "operator-authorized SC++ Stage 1 anchor dispatch via scripts/operator_authorize_scpp_stage1_anchor_dispatch.sh; estimated cost ~\$${EXPECTED_COST_USD}"

REMOTE_DRIVER="scripts/remote_lane_scpp_stage1.sh"
if [ ! -f "$REMOTE_DRIVER" ]; then
    cat >&2 <<EOM

[scpp-stage1] WARN: canonical remote driver missing at ${REMOTE_DRIVER}.

The lane is claimed but no remote driver script exists yet. Build the remote
driver (sister of scripts/remote_lane_t1_balle_endtoend.sh) before re-running
this authorize script.

Per CLAUDE.md "Operator gates must be wired and used", this exit-with-warning
is the safe default. Lane claim is still active (24h TTL).
EOM
    exit 3
fi

# Delegate to canonical Modal launcher (modal_train_lane.py); the remote
# script runs INSIDE the Modal container which mounts /workspace/pact.
# Per CLAUDE.md "Remote code parity" + Modal `.spawn()` harvest-or-lose,
# this writes modal_metadata.json with the call_id for harvest within 24h.
ENV_OVERRIDES="SCPP_DISPATCH_INSTANCE_JOB_ID=${INSTANCE_JOB_ID}"
ENV_OVERRIDES="${ENV_OVERRIDES},SCPP_ALLOW_SCORE_DOMAIN_TRAINING=1"
ENV_OVERRIDES="${ENV_OVERRIDES},SCPP_STAGE_1_EPOCHS=3"
ENV_OVERRIDES="${ENV_OVERRIDES},SCPP_RUN_CONTEST_CUDA_AUTH_EVAL=1"
ENV_OVERRIDES="${ENV_OVERRIDES},LOCAL_CUDA_WORKER=1"

.venv/bin/modal run --detach experiments/modal_train_lane.py \
    --lane-script "$REMOTE_DRIVER" \
    --label "${INSTANCE_JOB_ID}" \
    --gpu T4 \
    --timeout-hours 3.0 \
    --env-overrides "${ENV_OVERRIDES}"

echo "[scpp-stage1] complete; review active dispatch claims ledger:"
echo "  .omx/state/active_lane_dispatch_claims.md"
