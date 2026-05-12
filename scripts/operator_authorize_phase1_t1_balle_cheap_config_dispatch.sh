#!/bin/bash
# Operator authorize: Phase 1 T1 Ballé end-to-end trainer (T13+T19+T20+T22 all-on)
# cheap-config dispatch.
#
# Per TT cost refinement landing 2026-05-11
# (feedback_autopilot_end_to_end_hf_refresh_phase1_cost_refinement_landed_20260511.md):
#   - Modal T4: 3000 epochs, T13+T19+T20+T22 all-on, ~$0.59 cost band (\$0.75 cap).
#   - Vast.ai 4090: 3000 epochs, T13+T19+T20+T22 all-on, ~$0.06 cost band (\$0.10 cap).
#
# Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
# CONTEST-COMPLIANT HARDWARE": produces a CUDA-axis anchor on Tesla T4 or 4090;
# CPU paired eval is a separate dispatch (delegated post-CUDA harvest).
#
# Predicted Δ: -0.012 ± 0.007 (TT cost refinement; per-flag contribution
#              accounting). Fits ≤$5/individual cap with ~6.6× headroom on
#              Modal T4 (~50× headroom on Vast.ai 4090).
# Risk: Phase 1 trainer is now contest-compliant per Catalog #146; archive
#       grammar 8/8 declared per Catalog #124; runtime emission verified.
#
# Usage: PHASE1_PLATFORM=modal bash scripts/operator_authorize_phase1_t1_balle_cheap_config_dispatch.sh
#        PHASE1_PLATFORM=vastai bash scripts/operator_authorize_phase1_t1_balle_cheap_config_dispatch.sh
#
# Lane: lane_t1_balle_128k_endtoend (Phase 1 trainer; per canonical
#       scripts/remote_lane_t1_balle_endtoend.sh)
# Cross-ref: project_operator_decision_dashboard_20260511.md A-1 / A-2
#            feedback_autopilot_end_to_end_hf_refresh_phase1_cost_refinement_landed_20260511.md
#            feedback_phase1_trainer_write_runtime_fix_landed_20260509.md (Catalog #146)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LANE_ID="t1_balle_128k_endtoend"
PLATFORM="${PHASE1_PLATFORM:-modal}"
case "$PLATFORM" in
    modal)
        EXPECTED_COST_USD="0.75"
        EXPECTED_COST_BAND_USD="0.59"
        HARDWARE="Modal T4 (16 GB VRAM)"
        ;;
    vastai|vast)
        EXPECTED_COST_USD="0.10"
        EXPECTED_COST_BAND_USD="0.06"
        HARDWARE="Vast.ai RTX 4090 (24 GB VRAM)"
        ;;
    *)
        echo "[phase1-t1-cheap-config] FATAL: PHASE1_PLATFORM=${PLATFORM} not supported (modal|vastai)" >&2
        exit 1
        ;;
esac

cat <<EOF

=== Phase 1 T1 Ballé cheap-config dispatch operator confirmation ===

lane_id:                 ${LANE_ID}
platform:                ${PLATFORM} (${HARDWARE})
expected cost band:      \$${EXPECTED_COST_BAND_USD} USD (cap: \$${EXPECTED_COST_USD})
config:                  T13+T19+T20+T22 all-on, 3000 epochs, batch_size=16
predicted Δ:             -0.012 ± 0.007 [predicted; TT cost refinement]
risk:                    Phase 1 trainer is contest-compliant per Catalog #146
                         (write_runtime fix); archive grammar 8/8 declared
                         per Catalog #124. Empirical-validation risk is LOW.
envelope status:         FITS \$5/individual cap with ~6.6× (Modal) /
                         ~50× (Vast.ai) headroom.

pre-flight check:
  - scripts/remote_lane_t1_balle_endtoend.sh present
  - tools/claim_lane_dispatch.py present
  - .omx/state/active_lane_dispatch_claims.md present

EOF
read -r -p "Proceed with Phase 1 T1 Ballé dispatch (~\$${EXPECTED_COST_BAND_USD} ${PLATFORM})? [y/N] " confirm
case "$confirm" in
    [yY]|[yY][eE][sS])
        ;;
    *)
        echo "[phase1-t1-cheap-config] aborted — no dispatch fired"
        exit 0
        ;;
esac

if [ ! -f "scripts/remote_lane_t1_balle_endtoend.sh" ]; then
    echo "[phase1-t1-cheap-config] FATAL: canonical remote driver missing at scripts/remote_lane_t1_balle_endtoend.sh" >&2
    exit 1
fi

INSTANCE_JOB_ID="t1_balle_cheap_config_$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python tools/claim_lane_dispatch.py claim \
    --lane-id "$LANE_ID" \
    --platform "$PLATFORM" \
    --instance-job-id "$INSTANCE_JOB_ID" \
    --agent "claude:operator_authorize_phase1_t1_balle_cheap_config_dispatch" \
    --status "active_dispatch" \
    --notes "operator-authorized Phase 1 T1 Ballé T13+T19+T20+T22 all-on dispatch via scripts/operator_authorize_phase1_t1_balle_cheap_config_dispatch.sh; expected cost \$${EXPECTED_COST_BAND_USD} on ${PLATFORM}"

# Delegate to canonical Modal launcher (modal_train_lane.py) which mounts
# the workspace at /workspace/pact + runs the remote-side script inside the
# Modal container. Per CLAUDE.md "Remote code parity" + Modal `.spawn()`
# harvest-or-lose discipline, this writes modal_metadata.json with the
# call_id; harvest via tools/harvest_modal_calls.py within 24h.
ENV_OVERRIDES="T1_DISPATCH_INSTANCE_JOB_ID=${INSTANCE_JOB_ID}"
ENV_OVERRIDES="${ENV_OVERRIDES},T1_ALLOW_SCORE_DOMAIN_TRAINING=1"
ENV_OVERRIDES="${ENV_OVERRIDES},T1_ENABLE_T13_SQRT_N_BUDGET=1"
ENV_OVERRIDES="${ENV_OVERRIDES},T1_ENABLE_T19_ADAPTIVE_RHO=1"
ENV_OVERRIDES="${ENV_OVERRIDES},T1_ENABLE_T20_KL_POSE_DISTILL=1"
ENV_OVERRIDES="${ENV_OVERRIDES},T1_ENABLE_T22_TEMPORAL_CONSISTENCY=1"
ENV_OVERRIDES="${ENV_OVERRIDES},LOCAL_CUDA_WORKER=1"

case "$PLATFORM" in
    modal)
        # Cheap-config Modal T4 ~$0.59 cost band (~1h on T4). modal_train_lane.py
        # uses .spawn() for detached runs — harvest via tools/harvest_modal_calls.py.
        .venv/bin/modal run --detach experiments/modal_train_lane.py \
            --lane-script scripts/remote_lane_t1_balle_endtoend.sh \
            --label "${INSTANCE_JOB_ID}" \
            --gpu T4 \
            --timeout-hours 2.0 \
            --env-overrides "${ENV_OVERRIDES}"
        ;;
    vastai|vast)
        # Vast.ai 4090 path: delegate to canonical launch script.
        LAUNCHER=""
        [ -f "scripts/launch_lane_on_vastai.py" ] && LAUNCHER="scripts/launch_lane_on_vastai.py"
        [ -f "tools/launch_lane_on_vastai.py" ] && LAUNCHER="tools/launch_lane_on_vastai.py"
        if [ -z "$LAUNCHER" ]; then
            echo "[phase1-t1-cheap-config] FATAL: Vast.ai launcher missing; pick PHASE1_PLATFORM=modal" >&2
            exit 5
        fi
        .venv/bin/python "$LAUNCHER" \
            --lane-script scripts/remote_lane_t1_balle_endtoend.sh \
            --label "${INSTANCE_JOB_ID}" \
            --gpu RTX_4090 \
            --timeout-hours 2.0 \
            --env-overrides "${ENV_OVERRIDES}"
        ;;
esac

echo "[phase1-t1-cheap-config] complete; review active dispatch claims ledger:"
echo "  .omx/state/active_lane_dispatch_claims.md"
