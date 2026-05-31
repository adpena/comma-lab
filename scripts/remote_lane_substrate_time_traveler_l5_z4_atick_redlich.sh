#!/bin/bash
# Remote lane script: Z4 Atick-Redlich L1 SCAFFOLD research-only driver.
#
# Lane: lane_z4_atick_redlich_l1_scaffold_resume_20260530
# Recipe: .omx/operator_authorize_recipes/substrate_time_traveler_l5_z4_atick_redlich_mlx_local.yaml
# Substrate: src/tac/substrates/time_traveler_l5_z4/
#
# Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
# non-negotiable: the recipe is `research_only: true` + `dispatch_enabled:
# false`. THIS DRIVER will refuse to fire any actual training until the L2
# promotion contract per Catalog #233 4-gate is satisfied AND a trainer
# (experiments/train_substrate_time_traveler_l5_z4_atick_redlich.py) is
# authored.
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline"
# non-negotiable: this script DELEGATES bootstrap to
# scripts/remote_archive_only_eval.sh's bootstrap_runtime_deps() function
# rather than re-implementing uv install / ffmpeg install / torch CUDA pin.
#
# Per Catalog #163 sister discipline: REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1
# is prepended to the source line so the sourced script's main flow does
# NOT run.
#
# Per Catalog #189: optional-array expansion guard ``${ARR[@]+"${ARR[@]}"}``
# for set -u compatibility on macOS bash 3.2.
#
# Per Catalog #244: canonical NVML 3-export block immediately after
# set -euo pipefail (DALI_DISABLE_NVML + CUBLAS_WORKSPACE_CONFIG +
# PYTORCH_CUDA_ALLOC_CONF) — auto-emitted via the canonical
# tac.deploy.modal.runtime constants per the META-CONSOLIDATION standing
# directive.
#
# Per Catalog #204: Modal-aware OUTPUT_DIR resolution under
# /modal_results/${INSTANCE_JOB_ID}/output when MODAL_RUNTIME=1 so
# contest_auth_eval.py accepts durable provider custody.
#
# Per Catalog #326: this driver consumes Z4ATR_TRAINER_MODE env var
# (smoke|full) with explicit precedence over SMOKE_ONLY; default = "smoke"
# so misconfigured dispatches DEFAULT to the cheap smoke path.

set -euo pipefail

# Catalog #244 canonical NVML 3-export block (commit 611495f26 +
# tac.deploy.modal.runtime canonical constants).
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_z4_atick_redlich_l1_scaffold_resume_20260530"
TAG="${TAG:-substrate_time_traveler_l5_z4_atick_redlich}"

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_z4_atick_redlich_results}"

# Catalog #204 Modal-aware OUTPUT_DIR resolution canonical 3-branch
# (Z4ATR_OUTPUT_DIR env override > Modal runtime durable volume > LOG_DIR
# default). contest_auth_eval.py refuses temp-storage evidence per
# CLAUDE.md "Forbidden /tmp paths in any persisted artifact"
# non-negotiable.
if [ -n "${Z4ATR_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$Z4ATR_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ] && [ -n "${INSTANCE_JOB_ID:-}" ]; then
    OUTPUT_DIR="/modal_results/${INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
Z4ATR_VIDEO_PATH="${Z4ATR_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
Z4ATR_OUTPUT_DIR="${Z4ATR_OUTPUT_DIR:-$OUTPUT_DIR}"
Z4ATR_EPOCHS="${Z4ATR_EPOCHS:-200}"
Z4ATR_BATCH_SIZE="${Z4ATR_BATCH_SIZE:-4}"
Z4ATR_LR="${Z4ATR_LR:-5e-4}"
Z4ATR_LATENT_DIM="${Z4ATR_LATENT_DIM:-32}"
Z4ATR_EMBED_DIM="${Z4ATR_EMBED_DIM:-48}"
Z4ATR_NUM_UPSAMPLE_BLOCKS="${Z4ATR_NUM_UPSAMPLE_BLOCKS:-5}"
Z4ATR_BETA_SEG="${Z4ATR_BETA_SEG:-100.0}"
Z4ATR_GAMMA_POSE="${Z4ATR_GAMMA_POSE:-3.1622776601683795}"
Z4ATR_DELTA_COOP_RECEIVER="${Z4ATR_DELTA_COOP_RECEIVER:-0.05}"
Z4ATR_BETA_ATICK_REDLICH="${Z4ATR_BETA_ATICK_REDLICH:-0.5}"
Z4ATR_DEVICE="${Z4ATR_DEVICE:-cuda}"
Z4ATR_UPSTREAM_DIR="${Z4ATR_UPSTREAM_DIR:-$WORKSPACE/upstream}"
Z4ATR_ENABLE_AUTOCAST_FP16="${Z4ATR_ENABLE_AUTOCAST_FP16:-true}"
Z4ATR_ENABLE_TF32="${Z4ATR_ENABLE_TF32:-true}"
Z4ATR_ENABLE_TORCH_COMPILE="${Z4ATR_ENABLE_TORCH_COMPILE:-true}"
Z4ATR_ENABLE_GT_SCORER_CACHE="${Z4ATR_ENABLE_GT_SCORER_CACHE:-true}"

# Catalog #326: trainer mode env var with explicit precedence over
# SMOKE_ONLY; default = "smoke" so misconfigured dispatches DEFAULT to
# the cheap smoke path. Set Z4ATR_TRAINER_MODE=full in recipe
# env_overrides to opt into the full training run.
Z4ATR_TRAINER_MODE="${Z4ATR_TRAINER_MODE:-smoke}"
SMOKE_ONLY="${SMOKE_ONLY:-1}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Bootstrap canonical helper per CLAUDE.md "Forbidden re-implementing
# remote bootstrap inline" non-negotiable + Catalog #163 sentinel
# (REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 stops the sourced main flow).
REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"

# Stage 0: L1 SCAFFOLD RESEARCH-ONLY refusal — fail-closed BEFORE any
# paid GPU spend. The recipe carries `research_only: true` +
# `dispatch_enabled: false` so the canonical operator_authorize.py
# refuses dispatch upstream of this driver; this is a defense-in-depth
# guard at the driver surface.
RESEARCH_ONLY_BYPASS="${Z4ATR_RESEARCH_ONLY_BYPASS_OK:-}"
if [ -z "$RESEARCH_ONLY_BYPASS" ]; then
    echo "[Z4ATR DRIVER] research_only=true at L1 SCAFFOLD per CLAUDE.md" >&2
    echo "[Z4ATR DRIVER] dispatch_enabled=false in recipe; refusing to fire" >&2
    echo "[Z4ATR DRIVER] L2 promotion contract per Catalog #233 4-gate:" >&2
    echo "  1. smoke green (Modal T4 100ep)" >&2
    echo "  2. Tier-C density measurement post-training" >&2
    echo "  3. 100ep auth-eval anchor (byte-deterministic archive)" >&2
    echo "  4. custody validated per Catalog #127 (paired CUDA + CPU)" >&2
    echo "[Z4ATR DRIVER] Author trainer first:" >&2
    echo "  experiments/train_substrate_time_traveler_l5_z4_atick_redlich.py" >&2
    echo "[Z4ATR DRIVER] Then operator-routable via:" >&2
    echo "  Z4ATR_RESEARCH_ONLY_BYPASS_OK=1 + operator approval" >&2
    exit 11
fi

# Heartbeat every 5 min per CLAUDE.md "Remote code parity" non-negotiable.
HEARTBEAT_FILE="${HEARTBEAT_FILE:-$WORKSPACE/.omx/tmp/heartbeat_${TAG}.log}"
mkdir -p "$(dirname "$HEARTBEAT_FILE")"
(
    while true; do
        echo "$(date -u +%FT%TZ) heartbeat $TAG" >> "$HEARTBEAT_FILE"
        sleep 300
    done
) &
HEARTBEAT_PID=$!
trap 'kill $HEARTBEAT_PID 2>/dev/null || true' EXIT

echo "[Z4ATR DRIVER] L2 promotion bypass acknowledged; proceeding"
echo "[Z4ATR DRIVER] mode=$Z4ATR_TRAINER_MODE smoke_only=$SMOKE_ONLY"
echo "[Z4ATR DRIVER] output=$OUTPUT_DIR"
echo "[Z4ATR DRIVER] video=$Z4ATR_VIDEO_PATH"

# When the trainer lands, the canonical invocation will be:
#   $PYBIN experiments/train_substrate_time_traveler_l5_z4_atick_redlich.py \
#       --video-path "$Z4ATR_VIDEO_PATH" \
#       --output-dir "$OUTPUT_DIR" \
#       --epochs "$Z4ATR_EPOCHS" \
#       --batch-size "$Z4ATR_BATCH_SIZE" \
#       --lr "$Z4ATR_LR" \
#       --latent-dim "$Z4ATR_LATENT_DIM" \
#       --embed-dim "$Z4ATR_EMBED_DIM" \
#       --num-upsample-blocks "$Z4ATR_NUM_UPSAMPLE_BLOCKS" \
#       --beta-seg "$Z4ATR_BETA_SEG" \
#       --gamma-pose "$Z4ATR_GAMMA_POSE" \
#       --delta-coop-receiver "$Z4ATR_DELTA_COOP_RECEIVER" \
#       --beta-atick-redlich "$Z4ATR_BETA_ATICK_REDLICH" \
#       --device "$Z4ATR_DEVICE" \
#       --upstream-dir "$Z4ATR_UPSTREAM_DIR" \
#       ${Z4ATR_ENABLE_AUTOCAST_FP16:+--enable-autocast-fp16} \
#       ${Z4ATR_ENABLE_TF32:+--enable-tf32} \
#       ${Z4ATR_ENABLE_TORCH_COMPILE:+--enable-torch-compile} \
#       ${Z4ATR_ENABLE_GT_SCORER_CACHE:+--enable-gt-scorer-cache} \
#       $( [ "$Z4ATR_TRAINER_MODE" = "smoke" ] || [ "$SMOKE_ONLY" = "1" ] && echo "--smoke" )

echo "[Z4ATR DRIVER] trainer not yet authored — refusing to fire blank stage"
exit 12
