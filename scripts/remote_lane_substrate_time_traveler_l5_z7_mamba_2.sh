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

# Multi-candidate workspace resolution per Catalog #152 Modal-aware extension.
# In Modal runtime the workspace lives at /workspace/pact (mounted code) AND
# /tmp/pact (writable copy); Vast.ai uses $WORKSPACE; local uses $PWD.
if [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -n "${DISPATCH_INSTANCE_JOB_ID:-}" ]; then
    DEFAULT_OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    DEFAULT_OUTPUT_DIR="$WORKSPACE/lane_substrate_time_traveler_l5_z7_mamba2_results/output"
fi
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_time_traveler_l5_z7_mamba2_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
Z7_MAMBA2_OUTPUT_DIR="${Z7_MAMBA2_OUTPUT_DIR:-$OUTPUT_DIR}"
PROVENANCE="$LOG_DIR/provenance.json"

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

Z7_MAMBA2_VIDEO_PATH="${Z7_MAMBA2_VIDEO_PATH:-$(resolve_required_input_modal_aware upstream/videos/0.mkv 'contest video upstream/videos/0.mkv')}"
Z7_MAMBA2_UPSTREAM_DIR="${Z7_MAMBA2_UPSTREAM_DIR:-$(dirname "$(dirname "$Z7_MAMBA2_VIDEO_PATH")")}"

# Mode-specific defaults
Z7_MAMBA2_EPOCHS="${Z7_MAMBA2_EPOCHS:-}"
Z7_MAMBA2_BATCH_SIZE="${Z7_MAMBA2_BATCH_SIZE:-}"
Z7_MAMBA2_LR="${Z7_MAMBA2_LR:-}"
Z7_MAMBA2_DEVICE="${Z7_MAMBA2_DEVICE:-cuda}"
Z7_MAMBA2_LOSS_MODE="${Z7_MAMBA2_LOSS_MODE:-score_aware}"
Z7_MAMBA2_LATENT_DIM="${Z7_MAMBA2_LATENT_DIM:-}"
Z7_MAMBA2_EGO_MOTION_DIM="${Z7_MAMBA2_EGO_MOTION_DIM:-}"
Z7_MAMBA2_D_MODEL="${Z7_MAMBA2_D_MODEL:-}"
Z7_MAMBA2_D_STATE="${Z7_MAMBA2_D_STATE:-}"
Z7_MAMBA2_EXPAND="${Z7_MAMBA2_EXPAND:-}"
Z7_MAMBA2_BACKEND="${Z7_MAMBA2_BACKEND:-auto}"
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
    Z7_MAMBA2_BATCH_SIZE="${Z7_MAMBA2_BATCH_SIZE:-4}"
    Z7_MAMBA2_LR="${Z7_MAMBA2_LR:-5e-4}"
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
    Z7_MAMBA2_BATCH_SIZE="${Z7_MAMBA2_BATCH_SIZE:-4}"
    Z7_MAMBA2_LR="${Z7_MAMBA2_LR:-5e-4}"
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
    Z7_MAMBA2_BATCH_SIZE="${Z7_MAMBA2_BATCH_SIZE:-1}"
    Z7_MAMBA2_LR="${Z7_MAMBA2_LR:-1e-3}"
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

Z7_MAMBA2_EGO_SOURCE="${Z7_MAMBA2_EGO_SOURCE:-posenet_projection}"
Z7_MAMBA2_IDENTITY_PREDICTOR="${Z7_MAMBA2_IDENTITY_PREDICTOR:-false}"
Z7_MAMBA2_STATEFUL="${Z7_MAMBA2_STATEFUL:-true}"
Z7_MAMBA2_BETA_IB="${Z7_MAMBA2_BETA_IB:-1.0}"
Z7_MAMBA2_NOISE_STD="${Z7_MAMBA2_NOISE_STD:-0.0}"
Z7_MAMBA2_ALPHA_RATE="${Z7_MAMBA2_ALPHA_RATE:-25.0}"
Z7_MAMBA2_BETA_SEG="${Z7_MAMBA2_BETA_SEG:-100.0}"
Z7_MAMBA2_INFLATE_VERIFY="${Z7_MAMBA2_INFLATE_VERIFY:-true}"
Z7_MAMBA2_EMIT_STATIC_CONTROL="${Z7_MAMBA2_EMIT_STATIC_CONTROL:-true}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

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
  "epochs": "$Z7_MAMBA2_EPOCHS",
  "batch_size": "$Z7_MAMBA2_BATCH_SIZE",
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
  "dali_disable_nvml": "$DALI_DISABLE_NVML",
  "cublas_workspace_config": "$CUBLAS_WORKSPACE_CONFIG",
  "pytorch_cuda_alloc_conf": "$PYTORCH_CUDA_ALLOC_CONF",
  "started_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "score_claim": false,
  "promotion_eligible": false,
  "ready_for_paid_dispatch": false,
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

# Resolve Python binary
if [ -z "$PYBIN" ]; then
    if [ -x "$WORKSPACE/.venv/bin/python" ]; then
        PYBIN="$WORKSPACE/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYBIN="$(command -v python3)"
    else
        PYBIN="$(command -v python)"
    fi
fi

# PYTHONPATH per README canonical entry point
export PYTHONPATH="${PYTHONPATH:-${WORKSPACE}/src:${WORKSPACE}/upstream:${WORKSPACE}}"

echo "[lane-z7-mamba2] mode=$Z7_MAMBA2_TRAINER_MODE device=$Z7_MAMBA2_DEVICE epochs=$Z7_MAMBA2_EPOCHS max_pairs=$Z7_MAMBA2_MAX_PAIRS"
echo "[lane-z7-mamba2] PYBIN=$PYBIN PYTHONPATH=$PYTHONPATH"
echo "[lane-z7-mamba2] trainer args: ${TRAINER_ARGS[*]}"

# Run the trainer
TRAINER_LOG="$LOG_DIR/trainer_$(date -u +%Y%m%dT%H%M%SZ).log"
cd "$WORKSPACE"
"$PYBIN" -u experiments/train_substrate_time_traveler_l5_z7_mamba2.py "${TRAINER_ARGS[@]}" 2>&1 | tee "$TRAINER_LOG"
TRAINER_RC="${PIPESTATUS[0]}"

if [ "$TRAINER_RC" -ne 0 ]; then
    echo "[lane-z7-mamba2] FATAL: trainer rc=$TRAINER_RC" >&2
    exit "$TRAINER_RC"
fi

echo "[lane-z7-mamba2] DONE [no-auth-eval-pending-wave-N+1-council per Catalog #325]"
echo "[lane-z7-mamba2] artifacts under: $OUTPUT_DIR"
exit 0
