#!/bin/bash
# Remote lane script: substrate Z4 cooperative-receiver loss smoke + full dispatch.
#
# Trainer: experiments/train_substrate_z4_cooperative_receiver_loss.py
# Lane: lane_z4_cooperative_receiver_loss_step2_20260514
# Recipe: .omx/operator_authorize_recipes/substrate_z4_cooperative_receiver_loss_modal_t4_dispatch.yaml
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
# Design refs:
#   - .omx/research/campaign_z4_cooperative_receiver_loss_20260514.md (Time-Traveler L5 Step 2)
#   - feedback_grand_council_maximize_value_landed_20260514.md (Time-Traveler peer-seat)
#   - feedback_zen_floor_field_medal_grade_council_landed_20260514.md (zen-floor band)
#   - Atick & Redlich (1990) "Towards a theory of early visual processing"
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

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_z4_cooperative_receiver_loss_step2_20260514"
TAG="${TAG:-substrate_z4_cooperative_receiver_loss}"

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_z4_cooperative_receiver_loss_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${Z4_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$Z4_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Catalog #224: CUBLAS deterministic + DALI NVML disable for stable inflate.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
Z4_VIDEO_PATH="${Z4_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
Z4_OUTPUT_DIR="${Z4_OUTPUT_DIR:-$OUTPUT_DIR}"
Z4_EPOCHS="${Z4_EPOCHS:-200}"
Z4_BATCH_SIZE="${Z4_BATCH_SIZE:-4}"
Z4_LR="${Z4_LR:-5e-4}"
Z4_LAMBDA_PIXEL="${Z4_LAMBDA_PIXEL:-0.0}"
Z4_BETA_SEG="${Z4_BETA_SEG:-100.0}"
Z4_GAMMA_POSE="${Z4_GAMMA_POSE:-3.1622776601683795}"
Z4_DEVICE="${Z4_DEVICE:-cuda}"
Z4_UPSTREAM_DIR="${Z4_UPSTREAM_DIR:-$WORKSPACE/upstream}"
Z4_ENABLE_AUTOCAST_FP16="${Z4_ENABLE_AUTOCAST_FP16:-false}"
Z4_ENABLE_TF32="${Z4_ENABLE_TF32:-true}"
Z4_ENABLE_TORCH_COMPILE="${Z4_ENABLE_TORCH_COMPILE:-false}"
Z4_ENABLE_GT_SCORER_CACHE="${Z4_ENABLE_GT_SCORER_CACHE:-false}"
Z4_RELAX_DETERMINISM="${Z4_RELAX_DETERMINISM:-true}"

# Smoke vs full ladder: SMOKE_ONLY=1 forces --smoke (default for first-anchor v1).
SMOKE_ONLY="${SMOKE_ONLY:-1}"

DISPATCH_INSTANCE_JOB_ID="${Z4_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${Z4_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""
CLAIM_VERIFIED=0

log() { echo "[lane-z4-coopreceiver] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: Z4_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
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
        status="completed_z4_coopreceiver_remote_driver"
    elif [ "${CLAIM_VERIFIED:-0}" != "1" ]; then
        status="failed_z4_coopreceiver_claim_verification_rc_${rc}"
    else
        status="failed_z4_coopreceiver_remote_driver_rc_${rc}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --force \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "remote_lane_substrate_z4_cooperative_receiver_loss" \
        --status "$status" \
        --notes "remote_driver_terminal rc=$rc output_dir=$Z4_OUTPUT_DIR smoke=$SMOKE_ONLY lambda_pixel=$Z4_LAMBDA_PIXEL" \
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
  "trainer": "experiments/train_substrate_z4_cooperative_receiver_loss.py",
  "recipe": ".omx/operator_authorize_recipes/substrate_z4_cooperative_receiver_loss_modal_t4_dispatch.yaml",
  "video_path": "$Z4_VIDEO_PATH",
  "output_dir": "$Z4_OUTPUT_DIR",
  "epochs": "$Z4_EPOCHS",
  "batch_size": "$Z4_BATCH_SIZE",
  "lr": "$Z4_LR",
  "device": "$Z4_DEVICE",
  "lambda_pixel": "$Z4_LAMBDA_PIXEL",
  "beta_seg": "$Z4_BETA_SEG",
  "gamma_pose": "$Z4_GAMMA_POSE",
  "enable_autocast_fp16": "$Z4_ENABLE_AUTOCAST_FP16",
  "enable_tf32": "$Z4_ENABLE_TF32",
  "enable_torch_compile": "$Z4_ENABLE_TORCH_COMPILE",
  "enable_gt_scorer_cache": "$Z4_ENABLE_GT_SCORER_CACHE",
  "relax_determinism": "$Z4_RELAX_DETERMINISM",
  "smoke_only": "$SMOKE_ONLY",
  "dispatch_instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "started_at_utc": "$(date -u +%FT%TZ)"
}
EOF

# Stage 4: invoke trainer.
log "stage_4_trainer_begin epochs=$Z4_EPOCHS lambda_pixel=$Z4_LAMBDA_PIXEL smoke=$SMOKE_ONLY autocast=$Z4_ENABLE_AUTOCAST_FP16 tf32=$Z4_ENABLE_TF32 torch_compile=$Z4_ENABLE_TORCH_COMPILE gt_cache=$Z4_ENABLE_GT_SCORER_CACHE relax_determinism=$Z4_RELAX_DETERMINISM"
TRAINER_PY="$WORKSPACE/experiments/train_substrate_z4_cooperative_receiver_loss.py"
if [ ! -f "$TRAINER_PY" ]; then
    log "FATAL: trainer missing at $TRAINER_PY"
    exit 23
fi

PYBIN_RESOLVED="$CLAIM_PYTHON"

SMOKE_FLAG_ARGS=()
if [ "$SMOKE_ONLY" = "1" ]; then
    SMOKE_FLAG_ARGS+=(--smoke)
fi

AUTOCAST_FLAG_ARGS=()
case "$Z4_ENABLE_AUTOCAST_FP16" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        AUTOCAST_FLAG_ARGS+=(--enable-autocast-fp16)
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off|"")
        ;;
    *)
        log "FATAL: invalid Z4_ENABLE_AUTOCAST_FP16=$Z4_ENABLE_AUTOCAST_FP16; expected 0/1/true/false"
        exit 24
        ;;
esac

TF32_FLAG_ARGS=()
case "$Z4_ENABLE_TF32" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        TF32_FLAG_ARGS+=(--enable-tf32)
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off|"")
        ;;
    *)
        log "FATAL: invalid Z4_ENABLE_TF32=$Z4_ENABLE_TF32; expected 0/1/true/false"
        exit 25
        ;;
esac

TORCH_COMPILE_FLAG_ARGS=()
case "$Z4_ENABLE_TORCH_COMPILE" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        TORCH_COMPILE_FLAG_ARGS+=(--enable-torch-compile)
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off|"")
        ;;
    *)
        log "FATAL: invalid Z4_ENABLE_TORCH_COMPILE=$Z4_ENABLE_TORCH_COMPILE; expected 0/1/true/false"
        exit 26
        ;;
esac

GT_SCORER_CACHE_FLAG_ARGS=()
case "$Z4_ENABLE_GT_SCORER_CACHE" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        GT_SCORER_CACHE_FLAG_ARGS+=(--enable-gt-scorer-cache)
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off|"")
        ;;
    *)
        log "FATAL: invalid Z4_ENABLE_GT_SCORER_CACHE=$Z4_ENABLE_GT_SCORER_CACHE; expected 0/1/true/false"
        exit 27
        ;;
esac

RELAX_DETERMINISM_FLAG_ARGS=()
case "$Z4_RELAX_DETERMINISM" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        RELAX_DETERMINISM_FLAG_ARGS+=(--relax-determinism-for-backward)
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off|"")
        ;;
    *)
        log "FATAL: invalid Z4_RELAX_DETERMINISM=$Z4_RELAX_DETERMINISM; expected 0/1/true/false"
        exit 28
        ;;
esac

"$PYBIN_RESOLVED" "$TRAINER_PY" \
    --video-path "$Z4_VIDEO_PATH" \
    --output-dir "$Z4_OUTPUT_DIR" \
    --epochs "$Z4_EPOCHS" \
    --batch-size "$Z4_BATCH_SIZE" \
    --lr "$Z4_LR" \
    --device "$Z4_DEVICE" \
    --upstream-dir "$Z4_UPSTREAM_DIR" \
    --lambda-pixel "$Z4_LAMBDA_PIXEL" \
    --beta-seg "$Z4_BETA_SEG" \
    --gamma-pose "$Z4_GAMMA_POSE" \
    ${SMOKE_FLAG_ARGS[@]+"${SMOKE_FLAG_ARGS[@]}"} \
    ${AUTOCAST_FLAG_ARGS[@]+"${AUTOCAST_FLAG_ARGS[@]}"} \
    ${TF32_FLAG_ARGS[@]+"${TF32_FLAG_ARGS[@]}"} \
    ${TORCH_COMPILE_FLAG_ARGS[@]+"${TORCH_COMPILE_FLAG_ARGS[@]}"} \
    ${GT_SCORER_CACHE_FLAG_ARGS[@]+"${GT_SCORER_CACHE_FLAG_ARGS[@]}"} \
    ${RELAX_DETERMINISM_FLAG_ARGS[@]+"${RELAX_DETERMINISM_FLAG_ARGS[@]}"} \
    2>&1 | tee -a "$LOG_DIR/trainer.log"

# Stage 5: emit completion marker (operator + autopilot consume).
EVIDENCE_STATUS="$("$PYBIN_RESOLVED" - "$Z4_OUTPUT_DIR/stats.json" <<'PY'
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
log "LANE_Z4_COOPRECEIVER_DONE ${EVIDENCE_MARKER} output_dir=$Z4_OUTPUT_DIR smoke=$SMOKE_ONLY lambda_pixel=$Z4_LAMBDA_PIXEL ${SCORE_CLAIM_FLAG}"
echo "LANE_Z4_COOPRECEIVER_DONE ${EVIDENCE_MARKER} $LANE_ID ${SCORE_CLAIM_FLAG} lambda_pixel=$Z4_LAMBDA_PIXEL $(date -u +%FT%TZ)" >> "$LOG_DIR/completion.log"
