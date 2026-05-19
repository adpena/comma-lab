#!/bin/bash
# Remote lane script: Z7-Mamba-2 selective-state-space substrate trainer.
#
# Trainer: experiments/train_substrate_time_traveler_l5_z7_mamba2.py
# Lane: lane_z7_as_mamba_2_full_landing_20260518
# Recipe: .omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml
# Design memo: .omx/research/z7_mamba2_substrate_design_memo_20260518.md
#
# This driver supports both smoke and full modes via Z7_MAMBA2_TRAINER_MODE.
# The default is the smoke path; full mode requires per-substrate symposium
# PROCEED-unconditional per Catalog #325 + Wave-N+1 council approval per
# Z7 parent symposium Revision #6.
#
# Sister to scripts/remote_lane_substrate_time_traveler_l5_z7_lstm_predictive_coding.sh
# (Z7-LSTM/GRU canonical driver template).
#
# Catalog #244 canonical NVML/CUDA env block (DALI_DISABLE_NVML, CUBLAS_WORKSPACE_CONFIG,
# PYTORCH_CUDA_ALLOC_CONF), Catalog #163 sentinel for bootstrap source, Catalog #326
# Z7_MAMBA2_TRAINER_MODE env-var consumption with explicit precedence, Catalog #152
# multi-candidate Modal-aware path resolution for required input files.

set -euo pipefail

# Catalog #244 canonical NVML/CUDA env exports — required for Modal/Vast.ai
# substrate dispatch per CLAUDE.md "Production-hardened dispatch optimization
# protocol" Tier 2 hardware correctness.
# Source: src/tac/deploy/modal/runtime.py canonical constants
# Reference incident: D1 NVML 999 crash 2026-05-15 (fc-01KRKABYAC9C6MA161NKSGH9PY)
# Standing directive: feedback_consolidate_everything_into_meta_layer_or_canonical_helpers_standing_directive_20260515.md
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
DEFAULT_LANE_ID="lane_z7_as_mamba_2_full_landing_20260518"
LANE_ID="${Z7_MAMBA2_LANE_ID:-${PACT_DISPATCH_LANE_ID:-$DEFAULT_LANE_ID}}"
TAG="${TAG:-substrate_time_traveler_l5_z7_mamba2}"
RECIPE_PATH="${Z7_MAMBA2_RECIPE_PATH:-.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
DISPATCH_INSTANCE_JOB_ID="${Z7_MAMBA2_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
CLAIM_VERIFIED=0
CLAIM_VERIFICATION_RC=0

# Multi-candidate workspace resolution per Catalog #152 Modal-aware extension.
# In Modal runtime the workspace lives at /workspace/pact (mounted code) AND
# /tmp/pact (writable copy); Vast.ai uses $WORKSPACE; local uses $PWD.
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_time_traveler_l5_z7_mamba2_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${Z7_MAMBA2_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$Z7_MAMBA2_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
Z7_MAMBA2_OUTPUT_DIR="${Z7_MAMBA2_OUTPUT_DIR:-$OUTPUT_DIR}"
PROVENANCE="$LOG_DIR/provenance.json"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Resolve Python binary before dispatch-claim verification and trainer launch.
if [ -z "$PYBIN" ]; then
    if [ -x "$WORKSPACE/.venv/bin/python" ]; then
        PYBIN="$WORKSPACE/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYBIN="$(command -v python3)"
    else
        PYBIN="$(command -v python)"
    fi
fi

CLAIM_PYTHON="$PYBIN"

append_terminal_claim() {
    local rc="$1"
    if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
        echo "[lane-z7-mamba2] WARN: missing dispatch job id; cannot append terminal dispatch claim" >&2
        return 0
    fi
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        echo "[lane-z7-mamba2] WARN: claim helper missing; cannot append terminal dispatch claim" >&2
        return 0
    fi
    local status
    if [ "$CLAIM_VERIFIED" != "1" ]; then
        local verify_rc="$CLAIM_VERIFICATION_RC"
        if [ "$verify_rc" = "0" ]; then
            verify_rc="$rc"
        fi
        status="failed_z7_mamba2_claim_verification_rc_${verify_rc}"
    elif [ "$rc" -eq 0 ]; then
        status="completed_z7_mamba2_remote_driver_no_score_claim"
    else
        status="failed_z7_mamba2_remote_driver_rc_${rc}"
    fi
    local stats_path="$OUTPUT_DIR/z7_mamba2_full_main_export_stats.json"
    if [ "${SMOKE_ONLY:-0}" = "1" ]; then
        stats_path="$OUTPUT_DIR/z7_mamba2_scaffold_smoke_stats.json"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --force \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "remote_lane_substrate_time_traveler_l5_z7_mamba_2" \
        --status "$status" \
        --notes "remote_driver_terminal rc=$rc score_claim=false mode=${Z7_MAMBA2_TRAINER_MODE:-unknown} stats_path=$stats_path output_dir=$OUTPUT_DIR" \
        >> "$LOG_DIR/run.log" 2>&1 || {
        echo "[lane-z7-mamba2] WARN: failed to append terminal dispatch claim status=$status" >&2
    }
}

cleanup() {
    local rc="$?"
    append_terminal_claim "$rc"
    exit "$rc"
}
trap cleanup EXIT

if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    echo "[lane-z7-mamba2] FATAL: Z7_MAMBA2_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required" >&2
    exit 21
fi
if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
    echo "[lane-z7-mamba2] FATAL: claim helper missing at $WORKSPACE/tools/claim_lane_dispatch.py" >&2
    CLAIM_VERIFICATION_RC=21
    exit 21
fi
if [ ! -f "$DISPATCH_CLAIMS_PATH" ]; then
    echo "[lane-z7-mamba2] FATAL: dispatch-claims ledger missing at $DISPATCH_CLAIMS_PATH" >&2
    CLAIM_VERIFICATION_RC=21
    exit 21
fi
CLAIM_SUMMARY_JSON="$LOG_DIR/dispatch_claim_summary.json"
set +e
"$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" summary \
    --claims-path "$DISPATCH_CLAIMS_PATH" \
    --live-only \
    --format json > "$CLAIM_SUMMARY_JSON"
SUMMARY_RC=$?
set -e
if [ "$SUMMARY_RC" -ne 0 ]; then
    echo "[lane-z7-mamba2] FATAL: claim summary failed for $DISPATCH_CLAIMS_PATH" >&2
    CLAIM_VERIFICATION_RC="$SUMMARY_RC"
    exit 21
fi
set +e
"$CLAIM_PYTHON" - "$CLAIM_SUMMARY_JSON" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID" <<'PY'
import json
import sys

summary_path, lane_id, job_id = sys.argv[1:4]
with open(summary_path, encoding="utf-8") as fh:
    payload = json.load(fh)
for row in payload.get("active", []):
    if row.get("lane_id") == lane_id and row.get("instance_job_id") == job_id:
        raise SystemExit(0)
print(f"missing active claim lane={lane_id} job={job_id}", file=sys.stderr)
raise SystemExit(1)
PY
CLAIM_RC=$?
set -e
if [ "$CLAIM_RC" -ne 0 ]; then
    echo "[lane-z7-mamba2] FATAL: no active dispatch claim for lane=$LANE_ID job=$DISPATCH_INSTANCE_JOB_ID" >&2
    CLAIM_VERIFICATION_RC="$CLAIM_RC"
    exit 21
fi
CLAIM_VERIFIED=1
echo "[lane-z7-mamba2] stage_0_dispatch_claim_verified lane=$LANE_ID job=$DISPATCH_INSTANCE_JOB_ID" | tee -a "$LOG_DIR/run.log"

# Catalog #326 driver-mode-discipline: support multi-key mode resolution with
# explicit precedence Z7_MAMBA2_TRAINER_MODE > SMOKE_ONLY. Default to
# `timing_smoke` (non-smoke real-pair decode but tiny config), which is cheap
# enough to validate the integration without the full $20-30 dispatch budget.
Z7_MAMBA2_TRAINER_MODE="${Z7_MAMBA2_TRAINER_MODE:-timing_smoke}"
SMOKE_ONLY="${SMOKE_ONLY:-}"
case "$Z7_MAMBA2_TRAINER_MODE" in
    smoke|SMOKE|Smoke)
        SMOKE_ONLY="1"
        ;;
    timing_smoke|TIMING_SMOKE|Timing_smoke|TimingSmoke|timing)
        SMOKE_ONLY="0"
        ;;
    full|FULL|Full)
        SMOKE_ONLY="0"
        ;;
    *)
        echo "[lane-z7-mamba2] FATAL: invalid Z7_MAMBA2_TRAINER_MODE=$Z7_MAMBA2_TRAINER_MODE; expected smoke|timing_smoke|full" >&2
        exit 29
        ;;
esac

if [ "$Z7_MAMBA2_TRAINER_MODE" = "full" ]; then
    RECIPE_ABS="$WORKSPACE/$RECIPE_PATH"
    if [ ! -f "$RECIPE_ABS" ]; then
        echo "[lane-z7-mamba2] FATAL: full mode requires checked-in recipe at $RECIPE_ABS" >&2
        exit 31
    fi
    if ! grep -Eq '^[[:space:]]*research_only:[[:space:]]*false([[:space:]]*#.*)?$' "$RECIPE_ABS"; then
        echo "[lane-z7-mamba2] FATAL: full mode refused because recipe is still research_only" >&2
        exit 31
    fi
    if ! grep -Eq '^[[:space:]]*dispatch_enabled:[[:space:]]*true([[:space:]]*#.*)?$' "$RECIPE_ABS"; then
        echo "[lane-z7-mamba2] FATAL: full mode refused because recipe dispatch_enabled is not true" >&2
        exit 31
    fi
fi

# Required-input file resolution per Catalog #152 (Wave 1+Wave 2 multi-candidate)
# Search order: $WORKSPACE (Vast.ai) -> /workspace/pact (Modal mount) -> /tmp/pact (Modal writable)
resolve_required_input_modal_aware() {
    local relpath="$1"
    local description="${2:-input}"
    local candidates=(
        "$WORKSPACE/$relpath"
        "/workspace/pact/$relpath"
        "/tmp/pact/$relpath"
    )
    for candidate in "${candidates[@]}"; do
        if [ -f "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    echo "[lane-z7-mamba2] FATAL: $description not found at any candidate location:" >&2
    for candidate in "${candidates[@]}"; do
        echo "  - $candidate" >&2
    done
    return 1
}

if [ "$SMOKE_ONLY" != "1" ]; then
    Z7_MAMBA2_VIDEO_PATH="${Z7_MAMBA2_VIDEO_PATH:-$(resolve_required_input_modal_aware upstream/videos/0.mkv 'contest video upstream/videos/0.mkv')}"
    Z7_MAMBA2_UPSTREAM_DIR="${Z7_MAMBA2_UPSTREAM_DIR:-$(dirname "$(dirname "$Z7_MAMBA2_VIDEO_PATH")")}"
else
    Z7_MAMBA2_VIDEO_PATH="${Z7_MAMBA2_VIDEO_PATH:-}"
    Z7_MAMBA2_UPSTREAM_DIR="${Z7_MAMBA2_UPSTREAM_DIR:-$WORKSPACE/upstream}"
fi

# Mode-specific defaults
Z7_MAMBA2_EPOCHS="${Z7_MAMBA2_EPOCHS:-}"
Z7_MAMBA2_BATCH_SIZE="${Z7_MAMBA2_BATCH_SIZE:-}"
Z7_MAMBA2_LR="${Z7_MAMBA2_LR:-}"
Z7_MAMBA2_LR_WARMUP_STEPS="${Z7_MAMBA2_LR_WARMUP_STEPS:-}"
Z7_MAMBA2_GRAD_CLIP_NORM="${Z7_MAMBA2_GRAD_CLIP_NORM:-}"
Z7_MAMBA2_DEVICE="${Z7_MAMBA2_DEVICE:-cuda}"
Z7_MAMBA2_LOSS_MODE="${Z7_MAMBA2_LOSS_MODE:-score_aware}"
Z7_MAMBA2_LATENT_DIM="${Z7_MAMBA2_LATENT_DIM:-}"
Z7_MAMBA2_EGO_MOTION_DIM="${Z7_MAMBA2_EGO_MOTION_DIM:-}"
Z7_MAMBA2_D_MODEL="${Z7_MAMBA2_D_MODEL:-}"
Z7_MAMBA2_D_STATE="${Z7_MAMBA2_D_STATE:-}"
Z7_MAMBA2_EXPAND="${Z7_MAMBA2_EXPAND:-}"
Z7_MAMBA2_BACKEND="${Z7_MAMBA2_BACKEND:-reference_torch}"
Z7_MAMBA2_MAX_PAIRS="${Z7_MAMBA2_MAX_PAIRS:-}"
Z7_MAMBA2_DECODER_EMBED_DIM="${Z7_MAMBA2_DECODER_EMBED_DIM:-}"
Z7_MAMBA2_DECODER_CHANNELS="${Z7_MAMBA2_DECODER_CHANNELS:-}"
Z7_MAMBA2_DECODER_UPSAMPLE_BLOCKS="${Z7_MAMBA2_DECODER_UPSAMPLE_BLOCKS:-}"
Z7_MAMBA2_DECODER_INITIAL_GRID_H="${Z7_MAMBA2_DECODER_INITIAL_GRID_H:-}"
Z7_MAMBA2_DECODER_INITIAL_GRID_W="${Z7_MAMBA2_DECODER_INITIAL_GRID_W:-}"
Z7_MAMBA2_OUTPUT_HEIGHT="${Z7_MAMBA2_OUTPUT_HEIGHT:-}"
Z7_MAMBA2_OUTPUT_WIDTH="${Z7_MAMBA2_OUTPUT_WIDTH:-}"

if [ "$Z7_MAMBA2_TRAINER_MODE" = "smoke" ]; then
    # Smoke: tiny config; architecture sanity check only.
    Z7_MAMBA2_EPOCHS="${Z7_MAMBA2_EPOCHS:-1}"
    Z7_MAMBA2_BATCH_SIZE="${Z7_MAMBA2_BATCH_SIZE:-600}"
    Z7_MAMBA2_LR="${Z7_MAMBA2_LR:-5e-4}"
    Z7_MAMBA2_LR_WARMUP_STEPS="${Z7_MAMBA2_LR_WARMUP_STEPS:-0}"
    Z7_MAMBA2_GRAD_CLIP_NORM="${Z7_MAMBA2_GRAD_CLIP_NORM:-1.0}"
    Z7_MAMBA2_LATENT_DIM="${Z7_MAMBA2_LATENT_DIM:-24}"
    Z7_MAMBA2_EGO_MOTION_DIM="${Z7_MAMBA2_EGO_MOTION_DIM:-8}"
    Z7_MAMBA2_D_MODEL="${Z7_MAMBA2_D_MODEL:-64}"
    Z7_MAMBA2_D_STATE="${Z7_MAMBA2_D_STATE:-16}"
    Z7_MAMBA2_EXPAND="${Z7_MAMBA2_EXPAND:-2}"
    Z7_MAMBA2_MAX_PAIRS="${Z7_MAMBA2_MAX_PAIRS:-1}"
    Z7_MAMBA2_DECODER_EMBED_DIM="${Z7_MAMBA2_DECODER_EMBED_DIM:-32}"
    Z7_MAMBA2_DECODER_CHANNELS="${Z7_MAMBA2_DECODER_CHANNELS:-32,24,16,12}"
    Z7_MAMBA2_DECODER_UPSAMPLE_BLOCKS="${Z7_MAMBA2_DECODER_UPSAMPLE_BLOCKS:-4}"
    Z7_MAMBA2_DECODER_INITIAL_GRID_H="${Z7_MAMBA2_DECODER_INITIAL_GRID_H:-24}"
    Z7_MAMBA2_DECODER_INITIAL_GRID_W="${Z7_MAMBA2_DECODER_INITIAL_GRID_W:-32}"
    Z7_MAMBA2_OUTPUT_HEIGHT="${Z7_MAMBA2_OUTPUT_HEIGHT:-384}"
    Z7_MAMBA2_OUTPUT_WIDTH="${Z7_MAMBA2_OUTPUT_WIDTH:-512}"
elif [ "$Z7_MAMBA2_TRAINER_MODE" = "full" ]; then
    # Full: 100ep on 600 pairs at contest resolution. Requires per-substrate
    # symposium PROCEED-unconditional per Catalog #325 + Wave-N+1 council.
    Z7_MAMBA2_EPOCHS="${Z7_MAMBA2_EPOCHS:-100}"
    Z7_MAMBA2_BATCH_SIZE="${Z7_MAMBA2_BATCH_SIZE:-600}"
    Z7_MAMBA2_LR="${Z7_MAMBA2_LR:-5e-4}"
    Z7_MAMBA2_LR_WARMUP_STEPS="${Z7_MAMBA2_LR_WARMUP_STEPS:-10}"
    Z7_MAMBA2_GRAD_CLIP_NORM="${Z7_MAMBA2_GRAD_CLIP_NORM:-1.0}"
    Z7_MAMBA2_LATENT_DIM="${Z7_MAMBA2_LATENT_DIM:-24}"
    Z7_MAMBA2_EGO_MOTION_DIM="${Z7_MAMBA2_EGO_MOTION_DIM:-8}"
    Z7_MAMBA2_D_MODEL="${Z7_MAMBA2_D_MODEL:-64}"
    Z7_MAMBA2_D_STATE="${Z7_MAMBA2_D_STATE:-16}"
    Z7_MAMBA2_EXPAND="${Z7_MAMBA2_EXPAND:-2}"
    Z7_MAMBA2_MAX_PAIRS="${Z7_MAMBA2_MAX_PAIRS:-600}"
    Z7_MAMBA2_DECODER_EMBED_DIM="${Z7_MAMBA2_DECODER_EMBED_DIM:-32}"
    Z7_MAMBA2_DECODER_CHANNELS="${Z7_MAMBA2_DECODER_CHANNELS:-32,24,16,12}"
    Z7_MAMBA2_DECODER_UPSAMPLE_BLOCKS="${Z7_MAMBA2_DECODER_UPSAMPLE_BLOCKS:-4}"
    Z7_MAMBA2_DECODER_INITIAL_GRID_H="${Z7_MAMBA2_DECODER_INITIAL_GRID_H:-24}"
    Z7_MAMBA2_DECODER_INITIAL_GRID_W="${Z7_MAMBA2_DECODER_INITIAL_GRID_W:-32}"
    Z7_MAMBA2_OUTPUT_HEIGHT="${Z7_MAMBA2_OUTPUT_HEIGHT:-384}"
    Z7_MAMBA2_OUTPUT_WIDTH="${Z7_MAMBA2_OUTPUT_WIDTH:-512}"
else
    # Timing smoke: real-pair decode + tiny config for end-to-end timing measurement.
    Z7_MAMBA2_EPOCHS="${Z7_MAMBA2_EPOCHS:-1}"
    Z7_MAMBA2_BATCH_SIZE="${Z7_MAMBA2_BATCH_SIZE:-2}"
    Z7_MAMBA2_LR="${Z7_MAMBA2_LR:-1e-3}"
    Z7_MAMBA2_LR_WARMUP_STEPS="${Z7_MAMBA2_LR_WARMUP_STEPS:-0}"
    Z7_MAMBA2_GRAD_CLIP_NORM="${Z7_MAMBA2_GRAD_CLIP_NORM:-1.0}"
    Z7_MAMBA2_LATENT_DIM="${Z7_MAMBA2_LATENT_DIM:-8}"
    Z7_MAMBA2_EGO_MOTION_DIM="${Z7_MAMBA2_EGO_MOTION_DIM:-4}"
    Z7_MAMBA2_D_MODEL="${Z7_MAMBA2_D_MODEL:-16}"
    Z7_MAMBA2_D_STATE="${Z7_MAMBA2_D_STATE:-8}"
    Z7_MAMBA2_EXPAND="${Z7_MAMBA2_EXPAND:-2}"
    Z7_MAMBA2_MAX_PAIRS="${Z7_MAMBA2_MAX_PAIRS:-2}"
    Z7_MAMBA2_DECODER_EMBED_DIM="${Z7_MAMBA2_DECODER_EMBED_DIM:-8}"
    Z7_MAMBA2_DECODER_CHANNELS="${Z7_MAMBA2_DECODER_CHANNELS:-8,8,4,4}"
    Z7_MAMBA2_DECODER_UPSAMPLE_BLOCKS="${Z7_MAMBA2_DECODER_UPSAMPLE_BLOCKS:-2}"
    Z7_MAMBA2_DECODER_INITIAL_GRID_H="${Z7_MAMBA2_DECODER_INITIAL_GRID_H:-4}"
    Z7_MAMBA2_DECODER_INITIAL_GRID_W="${Z7_MAMBA2_DECODER_INITIAL_GRID_W:-4}"
    Z7_MAMBA2_OUTPUT_HEIGHT="${Z7_MAMBA2_OUTPUT_HEIGHT:-16}"
    Z7_MAMBA2_OUTPUT_WIDTH="${Z7_MAMBA2_OUTPUT_WIDTH:-16}"
fi

Z7_MAMBA2_EGO_SOURCE="${Z7_MAMBA2_EGO_SOURCE:-frame_delta_proxy}"
Z7_MAMBA2_IDENTITY_PREDICTOR="${Z7_MAMBA2_IDENTITY_PREDICTOR:-false}"
Z7_MAMBA2_STATEFUL="${Z7_MAMBA2_STATEFUL:-true}"
Z7_MAMBA2_BETA_IB="${Z7_MAMBA2_BETA_IB:-1.0}"
Z7_MAMBA2_NOISE_STD="${Z7_MAMBA2_NOISE_STD:-0.0}"
Z7_MAMBA2_ALPHA_RATE="${Z7_MAMBA2_ALPHA_RATE:-25.0}"
Z7_MAMBA2_BETA_SEG="${Z7_MAMBA2_BETA_SEG:-100.0}"
Z7_MAMBA2_INFLATE_VERIFY="${Z7_MAMBA2_INFLATE_VERIFY:-true}"
Z7_MAMBA2_EMIT_STATIC_CONTROL="${Z7_MAMBA2_EMIT_STATIC_CONTROL:-true}"
DEVICE_TYPE="$Z7_MAMBA2_DEVICE"
MPS_RESEARCH_SIGNAL_ONLY=false
CONTEST_AUTHORITY_TRAINING_DEVICE=false
INFLATE_VERIFY_DEVICE="$Z7_MAMBA2_DEVICE"
if [ "$Z7_MAMBA2_DEVICE" = "mps" ]; then
    MPS_RESEARCH_SIGNAL_ONLY=true
    INFLATE_VERIFY_DEVICE=cpu
elif [ "$Z7_MAMBA2_DEVICE" = "cuda" ]; then
    CONTEST_AUTHORITY_TRAINING_DEVICE=true
fi

# Provenance manifest per Catalog L "remote scripts write provenance"
cat > "$PROVENANCE" <<EOF
{
  "schema": "z7_mamba2_remote_lane_provenance_v1",
  "lane_id": "$LANE_ID",
  "tag": "$TAG",
  "trainer": "experiments/train_substrate_time_traveler_l5_z7_mamba2.py",
  "recipe": "$RECIPE_PATH",
  "trainer_mode": "$Z7_MAMBA2_TRAINER_MODE",
  "smoke_only": "$SMOKE_ONLY",
  "device": "$Z7_MAMBA2_DEVICE",
  "evidence_tag": "[z7-mamba2-remote-driver-no-score-claim]",
  "device_runtime_contract": {
    "training_device": "$Z7_MAMBA2_DEVICE",
    "device_type": "$DEVICE_TYPE",
    "mps_research_signal_only": $MPS_RESEARCH_SIGNAL_ONLY,
    "contest_authority_training_device": $CONTEST_AUTHORITY_TRAINING_DEVICE,
    "inflate_verify_device": "$INFLATE_VERIFY_DEVICE",
    "pytorch_enable_mps_fallback": "${PYTORCH_ENABLE_MPS_FALLBACK:-<unset>}",
    "score_claim": false,
    "promotion_eligible": false,
    "ready_for_exact_eval_dispatch": false,
    "ready_for_paid_dispatch": false,
    "rank_or_kill_eligible": false
  },
  "epochs": "$Z7_MAMBA2_EPOCHS",
  "batch_size": "$Z7_MAMBA2_BATCH_SIZE",
  "lr_warmup_steps": "$Z7_MAMBA2_LR_WARMUP_STEPS",
  "grad_clip_norm": "$Z7_MAMBA2_GRAD_CLIP_NORM",
  "max_pairs": "$Z7_MAMBA2_MAX_PAIRS",
  "latent_dim": "$Z7_MAMBA2_LATENT_DIM",
  "ego_motion_dim": "$Z7_MAMBA2_EGO_MOTION_DIM",
  "mamba2_d_model": "$Z7_MAMBA2_D_MODEL",
  "mamba2_d_state": "$Z7_MAMBA2_D_STATE",
  "mamba2_expand": "$Z7_MAMBA2_EXPAND",
  "mamba2_backend": "$Z7_MAMBA2_BACKEND",
  "stateful": "$Z7_MAMBA2_STATEFUL",
  "identity_predictor": "$Z7_MAMBA2_IDENTITY_PREDICTOR",
  "loss_mode": "$Z7_MAMBA2_LOSS_MODE",
  "beta_ib": "$Z7_MAMBA2_BETA_IB",
  "alpha_rate": "$Z7_MAMBA2_ALPHA_RATE",
  "beta_seg": "$Z7_MAMBA2_BETA_SEG",
  "video_path": "$Z7_MAMBA2_VIDEO_PATH",
  "upstream_dir": "$Z7_MAMBA2_UPSTREAM_DIR",
  "output_dir": "$OUTPUT_DIR",
  "log_dir": "$LOG_DIR",
  "dispatch_platform": "$DISPATCH_PLATFORM",
  "dispatch_instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "dispatch_claim_verified": "$CLAIM_VERIFIED",
  "dali_disable_nvml": "$DALI_DISABLE_NVML",
  "cublas_workspace_config": "$CUBLAS_WORKSPACE_CONFIG",
  "pytorch_cuda_alloc_conf": "$PYTORCH_CUDA_ALLOC_CONF",
  "started_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "score_claim": false,
  "promotion_eligible": false,
  "ready_for_exact_eval_dispatch": false,
  "ready_for_paid_dispatch": false,
  "rank_or_kill_eligible": false,
  "result_review_blockers": [
    "z7_mamba2_full_train_packet_not_paired_exact_eval_validated",
    "wave_n_plus_1_council_required_per_z7_mamba2_symposium",
    "per_substrate_symposium_evidence_required_per_catalog_325"
  ]
}
EOF
echo "[lane-z7-mamba2] provenance written: $PROVENANCE"

# Build trainer args based on mode
if [ "$SMOKE_ONLY" = "1" ]; then
    TRAINER_ARGS=(
        "--smoke"
        "--device" "$Z7_MAMBA2_DEVICE"
        "--mamba2-d-model" "$Z7_MAMBA2_D_MODEL"
        "--mamba2-d-state" "$Z7_MAMBA2_D_STATE"
        "--mamba2-expand" "$Z7_MAMBA2_EXPAND"
        "--mamba2-backend" "$Z7_MAMBA2_BACKEND"
        "--ego-motion-dim" "$Z7_MAMBA2_EGO_MOTION_DIM"
        "--output-dir" "$OUTPUT_DIR"
    )
else
    TRAINER_ARGS=(
        "--video-path" "$Z7_MAMBA2_VIDEO_PATH"
        "--output-dir" "$OUTPUT_DIR"
        "--epochs" "$Z7_MAMBA2_EPOCHS"
        "--batch-size" "$Z7_MAMBA2_BATCH_SIZE"
        "--lr" "$Z7_MAMBA2_LR"
        "--lr-warmup-steps" "$Z7_MAMBA2_LR_WARMUP_STEPS"
        "--grad-clip-norm" "$Z7_MAMBA2_GRAD_CLIP_NORM"
        "--latent-dim" "$Z7_MAMBA2_LATENT_DIM"
        "--ego-motion-dim" "$Z7_MAMBA2_EGO_MOTION_DIM"
        "--mamba2-d-model" "$Z7_MAMBA2_D_MODEL"
        "--mamba2-d-state" "$Z7_MAMBA2_D_STATE"
        "--mamba2-expand" "$Z7_MAMBA2_EXPAND"
        "--mamba2-backend" "$Z7_MAMBA2_BACKEND"
        "--ego-source" "$Z7_MAMBA2_EGO_SOURCE"
        "--identity-predictor" "$Z7_MAMBA2_IDENTITY_PREDICTOR"
        "--stateful" "$Z7_MAMBA2_STATEFUL"
        "--beta-ib" "$Z7_MAMBA2_BETA_IB"
        "--max-pairs" "$Z7_MAMBA2_MAX_PAIRS"
        "--decoder-embed-dim" "$Z7_MAMBA2_DECODER_EMBED_DIM"
        "--decoder-channels" "$Z7_MAMBA2_DECODER_CHANNELS"
        "--decoder-num-upsample-blocks" "$Z7_MAMBA2_DECODER_UPSAMPLE_BLOCKS"
        "--decoder-initial-grid-h" "$Z7_MAMBA2_DECODER_INITIAL_GRID_H"
        "--decoder-initial-grid-w" "$Z7_MAMBA2_DECODER_INITIAL_GRID_W"
        "--output-height" "$Z7_MAMBA2_OUTPUT_HEIGHT"
        "--output-width" "$Z7_MAMBA2_OUTPUT_WIDTH"
        "--inflate-verify" "$Z7_MAMBA2_INFLATE_VERIFY"
        "--emit-static-control" "$Z7_MAMBA2_EMIT_STATIC_CONTROL"
        "--loss-mode" "$Z7_MAMBA2_LOSS_MODE"
        "--noise-std" "$Z7_MAMBA2_NOISE_STD"
        "--upstream-dir" "$Z7_MAMBA2_UPSTREAM_DIR"
        "--alpha-rate" "$Z7_MAMBA2_ALPHA_RATE"
        "--beta-seg" "$Z7_MAMBA2_BETA_SEG"
        "--device" "$Z7_MAMBA2_DEVICE"
    )
fi

# PYTHONPATH per README canonical entry point
export PYTHONPATH="${PYTHONPATH:-${WORKSPACE}/src:${WORKSPACE}/upstream:${WORKSPACE}}"

echo "[lane-z7-mamba2] mode=$Z7_MAMBA2_TRAINER_MODE device=$Z7_MAMBA2_DEVICE epochs=$Z7_MAMBA2_EPOCHS max_pairs=$Z7_MAMBA2_MAX_PAIRS"
echo "[lane-z7-mamba2] PYBIN=$PYBIN PYTHONPATH=$PYTHONPATH"
echo "[lane-z7-mamba2] trainer args: ${TRAINER_ARGS[*]}"

# Run the trainer
TRAINER_LOG="$LOG_DIR/trainer_$(date -u +%Y%m%dT%H%M%SZ).log"
cd "$WORKSPACE"
set +e
"$PYBIN" -u experiments/train_substrate_time_traveler_l5_z7_mamba2.py "${TRAINER_ARGS[@]}" 2>&1 | tee "$TRAINER_LOG"
TRAINER_RC="${PIPESTATUS[0]}"
set -e

if [ "$TRAINER_RC" -ne 0 ]; then
    echo "[lane-z7-mamba2] FATAL: trainer rc=$TRAINER_RC" >&2
    exit "$TRAINER_RC"
fi

if [ "$SMOKE_ONLY" = "1" ]; then
    STATS_PATH="$OUTPUT_DIR/z7_mamba2_scaffold_smoke_stats.json"
else
    STATS_PATH="$OUTPUT_DIR/z7_mamba2_full_main_export_stats.json"
fi
if [ ! -f "$STATS_PATH" ]; then
    echo "[lane-z7-mamba2] FATAL: expected stats missing at $STATS_PATH" >&2
    exit 30
fi
"$PYBIN" - "$STATS_PATH" <<'PY'
import json
import sys

stats_path = sys.argv[1]
with open(stats_path, encoding="utf-8") as fh:
    stats = json.load(fh)
bad = []
for key in (
    "score_claim",
    "promotion_eligible",
    "ready_for_exact_eval_dispatch",
    "ready_for_paid_dispatch",
    "rank_or_kill_eligible",
):
    if stats.get(key) is not False:
        bad.append(f"{key}={stats.get(key)!r}")
contract = stats.get("device_runtime_contract")
if not isinstance(contract, dict):
    bad.append("device_runtime_contract missing")
else:
    for key in (
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "ready_for_paid_dispatch",
        "rank_or_kill_eligible",
    ):
        if contract.get(key) is not False:
            bad.append(f"device_runtime_contract.{key}={contract.get(key)!r}")
static = stats.get("static_capacity_control")
if isinstance(static, dict):
    for key in (
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "ready_for_paid_dispatch",
    ):
        if static.get(key) not in (None, False):
            bad.append(f"static_capacity_control.{key}={static.get(key)!r}")
if bad:
    print(
        f"Z7-Mamba-2 stats authority flags must stay false: {', '.join(bad)}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

echo "[lane-z7-mamba2] DONE [no-auth-eval-pending-wave-N+1-council per Catalog #325]"
echo "[lane-z7-mamba2] artifacts under: $OUTPUT_DIR"
exit 0
