#!/bin/bash
# Remote lane script: substrate Z6 Time-Traveler L5 predictive-coding smoke + full dispatch.
#
# Trainer: experiments/train_substrate_time_traveler_l5_z6.py
# Lane: lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516
# Recipe: .omx/operator_authorize_recipes/substrate_time_traveler_l5_z6_modal_t4_dispatch.yaml
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Per Catalog #163 the sentinel ``REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1`` is
# prepended to the source line so the sourced script's main flow does NOT
# run.
#
# Per Catalog #189 every optional-array expansion is guarded as
# ``${ARR[@]+"${ARR[@]}"}`` so the script tolerates `set -u` on macOS bash 3.2.
#
# Per Catalog #244 the canonical 3-export NVML/CUDA env block is emitted
# IMMEDIATELY after `set -euo pipefail` so DALI does not crash with
# `nvml error (999)` and CUBLAS produces deterministic results.
#
# Design refs:
#   - .omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md
#   - lane_z5_predictive_coding_world_model_step3_20260514 (sister Z5 L1 scaffold pattern)
#   - Rao & Ballard (1999) "Predictive coding in the visual cortex"
#   - Atick-Redlich (1990) cooperative-receiver theorem
#   - Perez et al. (2017) FiLM modulation
#
# Score-tagging: smoke/no-scorer artifacts are explicitly logged as
# score_claim=false and never as [contest-CUDA]. A [contest-CUDA] marker is
# allowed only when stats.json proves a valid contest_cuda score claim
# (auth_eval_score_claim_valid=true AND auth_eval_score_axis=contest_cuda).
# Per Catalog #204 the output is written to
# /modal_results/${DISPATCH_INSTANCE_JOB_ID}/output for durable provider
# custody when MODAL_RUNTIME=1.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
set -euo pipefail

# Catalog #244 canonical 3-export NVML/CUDA env block. Emitted IMMEDIATELY
# after `set -euo pipefail` per the canonical helper at
# tac.deploy.modal.runtime so DALI does not crash with `nvml error (999)`
# inside fn.experimental.inputs.video and so CUBLAS produces deterministic
# matmul outputs. Sister D1/D4/Z3/Z4/Z5 substrate drivers carry the same
# block; commit 611495f26 (the canonical anchor).
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516"
TAG="${TAG:-substrate_time_traveler_l5_z6}"

# Catalog #204: when running on Modal default output to durable provider volume.
if [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -n "${DISPATCH_INSTANCE_JOB_ID:-}" ]; then
    DEFAULT_OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    DEFAULT_OUTPUT_DIR="$WORKSPACE/lane_substrate_time_traveler_l5_z6_results/output"
fi
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_time_traveler_l5_z6_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
Z6_VIDEO_PATH="${Z6_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
Z6_OUTPUT_DIR="${Z6_OUTPUT_DIR:-$OUTPUT_DIR}"
Z6_EPOCHS="${Z6_EPOCHS:-300}"
Z6_BATCH_SIZE="${Z6_BATCH_SIZE:-4}"
Z6_LR="${Z6_LR:-5e-4}"
Z6_LAMBDA_RESIDUAL_ENTROPY="${Z6_LAMBDA_RESIDUAL_ENTROPY:-1.0}"
Z6_PREDICTOR_KERNEL_SIZE="${Z6_PREDICTOR_KERNEL_SIZE:-3}"
Z6_PREDICTOR_EGO_MOTION_DIM="${Z6_PREDICTOR_EGO_MOTION_DIM:-8}"
Z6_IDENTITY_PREDICTOR="${Z6_IDENTITY_PREDICTOR:-false}"
Z6_DEVICE="${Z6_DEVICE:-cuda}"
Z6_UPSTREAM_DIR="${Z6_UPSTREAM_DIR:-$WORKSPACE/upstream}"
Z6_ENABLE_AUTOCAST_FP16="${Z6_ENABLE_AUTOCAST_FP16:-false}"

# Smoke vs full ladder: SMOKE_ONLY=1 forces --smoke (default for first-anchor v1).
SMOKE_ONLY="${SMOKE_ONLY:-1}"

DISPATCH_INSTANCE_JOB_ID="${Z6_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${Z6_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""
CLAIM_VERIFIED=0

log() { echo "[lane-z6-pcwm] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: Z6_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
    exit 21
fi

CLAIM_PYTHON="${PYBIN:-}"
if [ -z "$CLAIM_PYTHON" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
    CLAIM_PYTHON="$WORKSPACE/.venv/bin/python"
fi
if [ -z "$CLAIM_PYTHON" ]; then
    CLAIM_PYTHON="python3"
fi

append_terminal_claim() {
    local rc="$1"
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        log "WARN: claim helper missing; cannot append terminal dispatch claim"
        return 0
    fi
    local status
    if [ "$rc" -eq 0 ]; then
        status="completed_z6_pcwm_remote_driver"
    elif [ "${CLAIM_VERIFIED:-0}" != "1" ]; then
        status="failed_z6_pcwm_claim_verification_rc_${rc}"
    else
        status="failed_z6_pcwm_remote_driver_rc_${rc}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --force \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "remote_lane_substrate_time_traveler_l5_z6" \
        --status "$status" \
        --notes "remote_driver_terminal rc=$rc output_dir=$Z6_OUTPUT_DIR smoke=$SMOKE_ONLY identity_predictor=$Z6_IDENTITY_PREDICTOR predictor_kernel=$Z6_PREDICTOR_KERNEL_SIZE" \
        >> "$LOG_DIR/run.log" 2>&1 || {
        log "WARN: failed to append terminal dispatch claim status=$status"
    }
}

cleanup() {
    local rc="$?"
    if [ -n "$HEARTBEAT_PID" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    append_terminal_claim "$rc"
    exit "$rc"
}
trap cleanup EXIT

# Stage 1: bootstrap remote runtime deps via canonical sourced helper.
# Per Catalog #163 prepend REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 so the
# sourced script's main flow does NOT run.
if [ -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "stage_1_bootstrap_via_canonical_sourced_helper"
    # shellcheck disable=SC1091
    REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"
    bootstrap_runtime_deps || {
        log "FATAL: bootstrap_runtime_deps failed; refusing dispatch"
        exit 22
    }
else
    log "WARN: canonical bootstrap script missing; assuming runtime deps present"
fi

# Stage 2: heartbeat (every 5 min).
(
    while true; do
        date -u +%FT%TZ > "$LOG_DIR/heartbeat.log"
        sleep 300
    done
) &
HEARTBEAT_PID=$!

# Stage 3: write provenance.
cat > "$PROVENANCE" <<EOF
{
  "lane_id": "$LANE_ID",
  "tag": "$TAG",
  "trainer": "experiments/train_substrate_time_traveler_l5_z6.py",
  "recipe": ".omx/operator_authorize_recipes/substrate_time_traveler_l5_z6_modal_t4_dispatch.yaml",
  "video_path": "$Z6_VIDEO_PATH",
  "output_dir": "$Z6_OUTPUT_DIR",
  "epochs": "$Z6_EPOCHS",
  "batch_size": "$Z6_BATCH_SIZE",
  "lr": "$Z6_LR",
  "device": "$Z6_DEVICE",
  "lambda_residual_entropy": "$Z6_LAMBDA_RESIDUAL_ENTROPY",
  "predictor_kernel_size": "$Z6_PREDICTOR_KERNEL_SIZE",
  "predictor_ego_motion_dim": "$Z6_PREDICTOR_EGO_MOTION_DIM",
  "identity_predictor": "$Z6_IDENTITY_PREDICTOR",
  "enable_autocast_fp16": "$Z6_ENABLE_AUTOCAST_FP16",
  "smoke_only": "$SMOKE_ONLY",
  "dispatch_instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "started_at_utc": "$(date -u +%FT%TZ)"
}
EOF

# Stage 4: invoke trainer.
log "stage_4_trainer_begin epochs=$Z6_EPOCHS lambda_res=$Z6_LAMBDA_RESIDUAL_ENTROPY kernel=$Z6_PREDICTOR_KERNEL_SIZE identity=$Z6_IDENTITY_PREDICTOR smoke=$SMOKE_ONLY"
TRAINER_PY="$WORKSPACE/experiments/train_substrate_time_traveler_l5_z6.py"
if [ ! -f "$TRAINER_PY" ]; then
    log "FATAL: trainer missing at $TRAINER_PY"
    exit 23
fi

PYBIN_RESOLVED="$CLAIM_PYTHON"

SMOKE_FLAG_ARGS=()
if [ "$SMOKE_ONLY" = "1" ]; then
    SMOKE_FLAG_ARGS+=(--smoke)
fi

IDENTITY_PREDICTOR_ARGS=()
case "$Z6_IDENTITY_PREDICTOR" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        IDENTITY_PREDICTOR_ARGS+=(--identity-predictor)
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off|"")
        ;;
    *)
        log "FATAL: invalid Z6_IDENTITY_PREDICTOR=$Z6_IDENTITY_PREDICTOR; expected 0/1/true/false"
        exit 24
        ;;
esac

AUTOCAST_FLAG_ARGS=()
case "$Z6_ENABLE_AUTOCAST_FP16" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        AUTOCAST_FLAG_ARGS+=(--enable-autocast-fp16)
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off|"")
        ;;
    *)
        log "FATAL: invalid Z6_ENABLE_AUTOCAST_FP16=$Z6_ENABLE_AUTOCAST_FP16; expected 0/1/true/false"
        exit 25
        ;;
esac

"$PYBIN_RESOLVED" "$TRAINER_PY" \
    --video-path "$Z6_VIDEO_PATH" \
    --output-dir "$Z6_OUTPUT_DIR" \
    --epochs "$Z6_EPOCHS" \
    --batch-size "$Z6_BATCH_SIZE" \
    --lr "$Z6_LR" \
    --device "$Z6_DEVICE" \
    --upstream-dir "$Z6_UPSTREAM_DIR" \
    --lambda-residual-entropy "$Z6_LAMBDA_RESIDUAL_ENTROPY" \
    --predictor-kernel-size "$Z6_PREDICTOR_KERNEL_SIZE" \
    --predictor-ego-motion-dim "$Z6_PREDICTOR_EGO_MOTION_DIM" \
    ${SMOKE_FLAG_ARGS[@]+"${SMOKE_FLAG_ARGS[@]}"} \
    ${IDENTITY_PREDICTOR_ARGS[@]+"${IDENTITY_PREDICTOR_ARGS[@]}"} \
    ${AUTOCAST_FLAG_ARGS[@]+"${AUTOCAST_FLAG_ARGS[@]}"} \
    2>&1 | tee -a "$LOG_DIR/trainer.log"

# Stage 5: emit completion marker (operator + autopilot consume).
EVIDENCE_STATUS="$("$PYBIN_RESOLVED" - "$Z6_OUTPUT_DIR/stats.json" <<'PY'
import json
import sys
from pathlib import Path

stats_path = Path(sys.argv[1])
marker = "[training-artifact-no-score-claim]"
score_claim = "score_claim=false"
if stats_path.is_file():
    try:
        stats = json.loads(stats_path.read_text())
    except json.JSONDecodeError:
        stats = {}
    if (
        stats.get("auth_eval_score_claim_valid") is True
        and stats.get("auth_eval_score_axis") == "contest_cuda"
    ):
        marker = "[contest-CUDA]"
        score_claim = "score_claim=true"
    elif stats.get("evidence_grade"):
        marker = f"[{stats['evidence_grade']}]"
print(f"{marker} {score_claim}")
PY
)"
EVIDENCE_MARKER="${EVIDENCE_STATUS%% *}"
SCORE_CLAIM_FLAG="${EVIDENCE_STATUS#* }"
log "LANE_Z6_PCWM_DONE ${EVIDENCE_MARKER} output_dir=$Z6_OUTPUT_DIR smoke=$SMOKE_ONLY identity_predictor=$Z6_IDENTITY_PREDICTOR ${SCORE_CLAIM_FLAG}"
echo "LANE_Z6_PCWM_DONE ${EVIDENCE_MARKER} $LANE_ID ${SCORE_CLAIM_FLAG} identity_predictor=$Z6_IDENTITY_PREDICTOR $(date -u +%FT%TZ)" >> "$LOG_DIR/completion.log"
