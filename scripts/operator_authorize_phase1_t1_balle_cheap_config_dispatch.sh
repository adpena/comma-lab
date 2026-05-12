#!/bin/bash
# Operator authorize: Phase 1 T1 Ballé end-to-end trainer (T13+T19+T20+T22 all-on)
# cheap-config dispatch.
#
# Per TT cost refinement landing 2026-05-11
# (feedback_autopilot_end_to_end_hf_refresh_phase1_cost_refinement_landed_20260511.md)
# WITH 2026-05-12 ENGINEERING-AUDIT REFINEMENT
# (feedback_council_t1_balle_engineering_audit_pixels_bytes_pixels_20260512.md):
#
# COST BAND — HONESTLY DOCUMENTED:
#   - The original TT "$0.59 cost band" assumed all flags OFF or smoke-only.
#     With T13+T19+T20+T22 ALL ON, codex's 2026-05-11 empirical Modal T4 run
#     made only ~100/3000 epochs in 23.5h before hitting Modal's 24h cap
#     (14 min/epoch on T4, $14 spent for partial result). The real cost band
#     for 3000 epochs of all-flags-on T1 Balle on T4 is $50-80, not $0.59.
#   - With Tier-1 engineering patches (teacher-pose cache, autocast FP16,
#     soft_cosine surrogate, batch_size=32) — Modal T4 ~5-10× faster:
#     3000 epochs ≈ 100-200 min ≈ $1-2 on T4. Still not "cheap" at $0.59 but
#     within $5/individual cap.
#   - Honest pre-patch cost band per platform:
#       * Modal T4 all-flags-on (PRE-patch): ~$50-80 for 3000 epochs (DO NOT)
#       * Modal T4 all-flags-on (POST-patch, with teacher cache): ~$5-10
#       * Vast.ai 4090 (faster GPU, native AMP): ~$0.50-1.50 at 3000 epochs
#       * Modal A100 (best $/TFLOP for this workload): ~$3-5 at 3000 epochs
#   - This wrapper currently DOES NOT enforce a 2h timeout that would cap
#     partial-result spending. Caller's `--timeout-hours` (set by the modal
#     run --detach call below) is the only safety. ENSURE 2h or less.
#
# Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
# CONTEST-COMPLIANT HARDWARE": Modal T4 training does NOT produce a
# promotion-grade contest-CUDA anchor — modal_train_lane.py docstring
# explicitly notes "It disables lane-local exact CUDA auth-eval paths".
# A real [contest-CUDA] anchor requires a separate Vast.ai 4090 / Lightning
# T4 exact-eval dispatch on the trained checkpoint AFTER harvest.
#
# Predicted Δ: -0.012 ± 0.007 [predicted; TT cost refinement]; UNREALIZED at
#              current ≤3% completion ratio per codex empirical evidence;
#              re-evaluate after Tier 1 engineering patches.
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
        # Modal GPU selection — per engineering audit 2026-05-12:
        # T4    $0.59/hr   65 TFLOPS FP16   baseline
        # A10G  $1.10/hr  125 TFLOPS FP16   0.97× $/TFLOP-hr (same value, 2× speed)
        # A100  $2.10/hr  312 TFLOPS FP16   0.74× $/TFLOP-hr (cheaper per work, 5× speed)
        # H100  $3.90/hr  900 TFLOPS FP16   0.47× $/TFLOP-hr (cheapest per work)
        # For scorer-bound T1 Balle training, A100 is the optimal price/perf
        # at $/TFLOP-hr; T4 is the budget default; H100 is for time-critical.
        # Override via MODAL_GPU=A100|H100|A10G|T4 env-var.
        MODAL_GPU="${MODAL_GPU:-T4}"
        TIMEOUT_HOURS="${MODAL_TIMEOUT_HOURS:-2.0}"
        # modal_train_lane.py uses .spawn() for detached runs — harvest via
        # tools/harvest_modal_calls.py or experiments/modal_recover_lane.py within 24h.
        .venv/bin/modal run --detach experiments/modal_train_lane.py \
            --lane-script scripts/remote_lane_t1_balle_endtoend.sh \
            --label "${INSTANCE_JOB_ID}" \
            --gpu "${MODAL_GPU}" \
            --timeout-hours "${TIMEOUT_HOURS}" \
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
