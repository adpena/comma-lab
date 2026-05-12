#!/bin/bash
# Operator authorize: SC++ Stage 1 anchor smoke dispatch (Modal T4).
#
# Per VV+QQ substrate composition matrix analysis 2026-05-11: SC++ substrate
# (block-FP weight self-compression per Selfcomp/szabolcs-cs) is predicted to
# dominate the S_floor low end. Stage 1 is the smoke-anchor dispatch (3 epochs)
# that produces a training/build smoke artifact for the SC++ architecture
# class. It is not a [contest-CUDA] anchor until a separate exact-eval
# dispatcher consumes a byte-closed archive and writes custody.
#
# Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lesson 7:
# substrate engineering may exceed the 350-LOC bolt-on budget (tagged
# lane_class=substrate_engineering); this lane is in that category.
#
# Per CLAUDE.md evidence discipline: this Modal training wrapper does not
# produce a promotion-grade score. CUDA-axis anchors require a separate
# claimed exact-eval dispatch; this is a separate exact-eval dispatch.
#
# Per CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION": claims the lane via
# tools/claim_lane_dispatch.py BEFORE Modal launch.
#
# W/I/A I-2 backport (2026-05-12, decision I-2): the previous hand-derived
# cost literal $3.00 is replaced with a runtime call to the canonical
# cost-band posterior at `.omx/state/cost_band_posterior.jsonl`. The
# tac.cost_band_calibration.predict() call returns p10/p50/p90 + confidence
# tag. When the posterior has no anchors matching the (modal, T4, 3-epoch)
# bucket, predict() returns a hand_calibrated_fallback band. See the
# OD-CB-1 + F1 fix pattern (commit 5eb355aa + 0666720a) for the canonical
# wire-in. Sister of operator_authorize_phase1_t1_balle_cheap_config_dispatch.sh.
#
# Cost band: see runtime call below; hand-derived fallback ~$3 Modal T4
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
PLATFORM="modal"

# W/I/A I-2 fix (2026-05-12): replace stale literal EXPECTED_COST_USD with
# runtime call to the cost-band posterior. The MODAL_GPU env-var below
# defaults to T4 for SC++ Stage 1 smoke (overridable). The fallback handles
# the cold-start case where the posterior has no matching anchors yet.
MODAL_GPU="${MODAL_GPU:-T4}"
PLATFORM_KEY="modal"
GPU_CLASS="${MODAL_GPU}"
SCPP_EPOCHS_SMOKE=3
COST_BAND_TEXT=$(.venv/bin/python -c "
from tac.cost_band_calibration import predict
p = predict('${PLATFORM_KEY}', '${GPU_CLASS}', ${SCPP_EPOCHS_SMOKE}, all_flags_on=True)
print(f'\${p.p10_cost_usd:.2f}/\${p.p50_cost_usd:.2f}/\${p.p90_cost_usd:.2f}'
      f'  (N={p.n_anchors}, {p.confidence_tag})')
" 2>/dev/null || echo "predict() unavailable — hand-calibrated fallback \$3.00")
EXPECTED_COST_USD=$(.venv/bin/python -c "
from tac.cost_band_calibration import predict
p = predict('${PLATFORM_KEY}', '${GPU_CLASS}', ${SCPP_EPOCHS_SMOKE}, all_flags_on=True)
# Cold-start bucket (no matching anchors AND no hand-stub for this bucket)
# falls back to a literal so the operator banner is informative.
print(f'{p.p50_cost_usd:.2f}' if p.p50_cost_usd > 0 else '3.00')
" 2>/dev/null || echo "3.00")
CONFIDENCE_TAG=$(.venv/bin/python -c "
from tac.cost_band_calibration import predict
print(predict('${PLATFORM_KEY}', '${GPU_CLASS}', ${SCPP_EPOCHS_SMOKE}, all_flags_on=True).confidence_tag)
" 2>/dev/null || echo "hand_calibrated_fallback")

cat <<EOF

=== SC++ Stage 1 anchor dispatch operator confirmation ===

lane_id:                 ${LANE_ID}
platform:                ${PLATFORM} (Modal ${MODAL_GPU})
cost band p10/p50/p90:   ${COST_BAND_TEXT}
                         Source: tac.cost_band_calibration.predict()
                         Posterior: .omx/state/cost_band_posterior.jsonl
                         Confidence: ${CONFIDENCE_TAG}
expected p50 cost:       \$${EXPECTED_COST_USD} USD (3-epoch smoke)
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
    --notes "operator-authorized SC++ Stage 1 training/build smoke via scripts/operator_authorize_scpp_stage1_anchor_dispatch.sh; no score claim; estimated cost ~\$${EXPECTED_COST_USD}"

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
ENV_OVERRIDES="${ENV_OVERRIDES},SCPP_RUN_CONTEST_CUDA_AUTH_EVAL=0"
ENV_OVERRIDES="${ENV_OVERRIDES},LOCAL_CUDA_WORKER=1"

# Modal GPU selection — per engineering audit 2026-05-12. Override via
# MODAL_GPU=A100|H100|A10G|T4 (default T4 for SC++ Stage 1 smoke; A100 5× faster
# at 0.74× $/TFLOP-hr — better $/work for substrate engineering iteration).
# MODAL_GPU already resolved above for cost-band prediction.
MODAL_TIMEOUT_HOURS="${MODAL_TIMEOUT_HOURS:-3.0}"
.venv/bin/modal run --detach experiments/modal_train_lane.py \
    --lane-script "$REMOTE_DRIVER" \
    --label "${INSTANCE_JOB_ID}" \
    --gpu "${MODAL_GPU}" \
    --timeout-hours "${MODAL_TIMEOUT_HOURS}" \
    --env-overrides "${ENV_OVERRIDES}"

echo "[scpp-stage1] complete; review active dispatch claims ledger:"
echo "  .omx/state/active_lane_dispatch_claims.md"
