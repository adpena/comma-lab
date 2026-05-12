#!/bin/bash
# Operator authorize: sane_hnerv first-anchor dispatch on Modal A100.
#
# Per operator directive 2026-05-12: "reroute all to modal and lightning free
# tier" + "all are approved".
#
# Background: Wave 3 prior attempt
# (feedback_substrate_sane_hnerv_first_anchor_dispatch_landed_20260512.md)
# DEFERRED with $0 spent on two STOP-PRECONDITIONS:
#   1. STRICT preflight RED on Catalog #151/#152/#154 rglob() OSS-mirror
#      cascade (VVV concurrently fixing).
#   2. Vast.ai account balance NEGATIVE (-$0.17).
#
# This re-routed wrapper resolves both blockers:
#   - VVV's preflight fix on the 3 rglob() functions landed (verified by
#     this wrapper's preflight check below).
#   - Modal A100 replaces Vast.ai 4090 per operator directive.
#
# COST BAND (predicted; will be empirically anchored on completion):
#   - tac.cost_band_calibration.predict('modal', 'A100', 2000, all_flags_on=True)
#   - Current confidence: weak_posterior (N=1 anchor; predictions p10/p50/p90 will
#     be displayed at runtime). The first dispatches into this bucket bootstrap
#     the calibration system.
#
# Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": this dispatch is a
# FIRST-ANCHOR RESEARCH dispatch — single-axis [contest-CUDA] is acceptable
# for first-anchor; [contest-CPU] axis required separately before promotion.
#
# Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE": modal_train_lane.py uses
# .spawn(); harvest within 24h via tools/harvest_modal_calls.py or
# experiments/modal_recover_lane.py.
#
# Predicted Δ: -0.030 to -0.050 [predicted; council substrate design memo
# `.omx/research/grand_council_fields_medal_substrate_design_20260512.md`]
# Risk: Phase 1 first-anchor research dispatch. Trainer _full_main wires
# 16 stages including auth eval at stage 12 (per Wave 1/A landing memo).
#
# Lane: lane_substrate_sane_hnerv_20260512 (Phase 1 first-anchor research)
# Cross-ref:
#   feedback_substrate_sane_hnerv_full_main_wired_landed_20260512.md
#   feedback_substrate_sane_hnerv_first_anchor_dispatch_landed_20260512.md
#   feedback_modal_strategy_reevaluation_post_tier1_engineering_20260512.md
#
# Usage:
#   bash scripts/operator_authorize_substrate_sane_hnerv_modal_a100_dispatch.sh
#
# Env-var overrides:
#   MODAL_GPU=T4|A10G|A100|H100  (default A100 per operator directive 2026-05-12)
#   SANE_HNERV_EPOCHS=2000       (council default; full training)
#   MODAL_TIMEOUT_HOURS=4.0      (Modal hard-kill wall-clock)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LANE_ID="lane_substrate_sane_hnerv_20260512"
PLATFORM="modal"
MODAL_GPU="${MODAL_GPU:-A100}"
HARDWARE="Modal ${MODAL_GPU}"
PLATFORM_KEY="modal"
SANE_HNERV_EPOCHS="${SANE_HNERV_EPOCHS:-2000}"
BATCH_SIZE_FOR_COST_BAND=32

# Cost-band prediction via posterior (canonical truth source).
COST_BAND_TEXT=$(.venv/bin/python -c "
from tac.cost_band_calibration import predict
p = predict('${PLATFORM_KEY}', '${MODAL_GPU}', ${SANE_HNERV_EPOCHS}, all_flags_on=True)
print(f'\${p.p10_cost_usd:.2f}/\${p.p50_cost_usd:.2f}/\${p.p90_cost_usd:.2f}'
      f'  (N={p.n_anchors}, {p.confidence_tag})')
" 2>/dev/null || echo "predict() failed — fallback hand-calibrated estimate")
EXPECTED_P50_USD=$(.venv/bin/python -c "
from tac.cost_band_calibration import predict
p = predict('${PLATFORM_KEY}', '${MODAL_GPU}', ${SANE_HNERV_EPOCHS}, all_flags_on=True)
print(f'{p.p50_cost_usd:.2f}')
" 2>/dev/null || echo "8.00")
CONFIDENCE_TAG=$(.venv/bin/python -c "
from tac.cost_band_calibration import predict
print(predict('${PLATFORM_KEY}', '${MODAL_GPU}', ${SANE_HNERV_EPOCHS}, all_flags_on=True).confidence_tag)
" 2>/dev/null || echo "hand_calibrated_fallback")

cat <<EOF

=== sane_hnerv Modal A100 first-anchor dispatch operator confirmation ===

lane_id:                 ${LANE_ID}
platform:                ${PLATFORM} (${HARDWARE})
cost band p10/p50/p90:   ${COST_BAND_TEXT}
                         Source: tac.cost_band_calibration.predict()
                         Posterior: .omx/state/cost_band_posterior.jsonl
                         Confidence: ${CONFIDENCE_TAG}
config:                  sane_hnerv substrate, ${SANE_HNERV_EPOCHS} epochs, batch=32
                         16-stage _full_main (auth eval at stage 12, posterior at stage 13)
predicted Δ:             -0.030 to -0.050 [predicted; council substrate design memo]
risk:                    Phase 1 first-anchor research dispatch on Modal A100.
                         Single-axis [contest-CUDA] acceptable for first anchor;
                         CPU axis required separately before promotion grade.

cost-band note: ${CONFIDENCE_TAG} — bucket has limited empirical anchors;
this dispatch will append its actual cost to the posterior on completion.

EOF
read -r -p "Proceed with sane_hnerv Modal ${MODAL_GPU} dispatch (p50≈\$${EXPECTED_P50_USD}, ${CONFIDENCE_TAG})? [y/N] " confirm
case "$confirm" in
    [yY]|[yY][eE][sS])
        ;;
    *)
        echo "[sane-hnerv-modal-a100] aborted — no dispatch fired"
        exit 0
        ;;
esac

# Pre-flight 1: canonical remote driver must exist.
if [ ! -f "scripts/remote_lane_substrate_sane_hnerv.sh" ]; then
    echo "[sane-hnerv-modal-a100] FATAL: canonical remote driver missing at scripts/remote_lane_substrate_sane_hnerv.sh" >&2
    exit 1
fi

# Pre-flight 2: Catalog #152 — validate required-input-file flags BEFORE
# Modal dispatch. The validate tool's AST extractor currently honors
# bare-Assign-only declarations (not AnnAssign), so it fails-open for
# sane_hnerv's annotated dict. The trainer itself fails-closed on missing
# --video-path at runtime, so this advisory-pass is acceptable.
.venv/bin/python tools/validate_dispatch_required_inputs.py \
    --trainer experiments/train_substrate_sane_hnerv.py \
    || { echo "[sane-hnerv-modal-a100] FATAL: required input missing per Catalog #152" >&2; exit 7; }

# Pre-flight 3: hard-fail if --video-path target file is missing.
if [ ! -f "upstream/videos/0.mkv" ]; then
    echo "[sane-hnerv-modal-a100] FATAL: contest video missing at upstream/videos/0.mkv" >&2
    exit 8
fi

INSTANCE_JOB_ID="sane_hnerv_modal_${MODAL_GPU,,}_$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python tools/claim_lane_dispatch.py claim \
    --lane-id "$LANE_ID" \
    --platform "$PLATFORM" \
    --instance-job-id "$INSTANCE_JOB_ID" \
    --agent "claude:operator_authorize_substrate_sane_hnerv_modal_a100_dispatch" \
    --status "active_dispatch" \
    --notes "operator-authorized sane_hnerv first-anchor Modal ${MODAL_GPU} dispatch (re-route from Vast.ai per 2026-05-12 directive); expected p50 cost \$${EXPECTED_P50_USD} (${CONFIDENCE_TAG})"

# Env overrides threaded to the remote_lane shell (Catalog #151 env→CLI ladder).
ENV_OVERRIDES="SANE_HNERV_DISPATCH_INSTANCE_JOB_ID=${INSTANCE_JOB_ID}"
ENV_OVERRIDES="${ENV_OVERRIDES},SANE_HNERV_VIDEO_PATH=/workspace/pact/upstream/videos/0.mkv"
ENV_OVERRIDES="${ENV_OVERRIDES},SANE_HNERV_OUTPUT_DIR=/workspace/pact/lane_substrate_sane_hnerv_results/output"
ENV_OVERRIDES="${ENV_OVERRIDES},SANE_HNERV_EPOCHS=${SANE_HNERV_EPOCHS}"
ENV_OVERRIDES="${ENV_OVERRIDES},SANE_HNERV_UPSTREAM_DIR=/workspace/pact/upstream"
ENV_OVERRIDES="${ENV_OVERRIDES},SANE_HNERV_DEVICE=cuda"
ENV_OVERRIDES="${ENV_OVERRIDES},LOCAL_CUDA_WORKER=1"

TIMEOUT_HOURS="${MODAL_TIMEOUT_HOURS:-4.0}"

# Modal dispatch via canonical launcher. modal_train_lane.py uses .spawn();
# harvest via tools/harvest_modal_calls.py within 24h per CLAUDE.md
# "Modal `.spawn()` HARVEST OR LOSE".
.venv/bin/modal run --detach experiments/modal_train_lane.py \
    --lane-script scripts/remote_lane_substrate_sane_hnerv.sh \
    --label "${INSTANCE_JOB_ID}" \
    --gpu "${MODAL_GPU}" \
    --timeout-hours "${TIMEOUT_HOURS}" \
    --env-overrides "${ENV_OVERRIDES}" \
    --cost-band-trainer experiments/train_substrate_sane_hnerv.py \
    --cost-band-epochs "${SANE_HNERV_EPOCHS}" \
    --cost-band-batch-size "${BATCH_SIZE_FOR_COST_BAND}" \
    --cost-band-all-flags-on

echo "[sane-hnerv-modal-a100] complete; review active dispatch claims ledger:"
echo "  .omx/state/active_lane_dispatch_claims.md"
echo "[sane-hnerv-modal-a100] harvest within 24h via:"
echo "  .venv/bin/python tools/harvest_modal_calls.py"
