#!/bin/bash
# Remote lane script: Z7 GRU recurrent predictive-coding timing/full smoke.
#
# Trainer: experiments/train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py
# Lane: lane_per_substrate_symposium_z7_lstm_predictive_coding_20260517
# Recipe: .omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_lstm_predictive_coding_modal_t4_dispatch.yaml
#
# This driver exists to close the missing-driver launch-precondition gap while
# preserving the recipe's research-only / dispatch-disabled state. It requires
# an already-open dispatch claim, never opens provider work by itself, and
# terminalizes every exit. Successful artifacts are classified as no-score-claim
# unless a future exact-eval path writes explicit authority fields.
set -euo pipefail

export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
DEFAULT_LANE_ID="lane_per_substrate_symposium_z7_lstm_predictive_coding_20260517"
LANE_ID="${Z7_GRU_LANE_ID:-${PACT_DISPATCH_LANE_ID:-$DEFAULT_LANE_ID}}"
TAG="${TAG:-substrate_time_traveler_l5_z7_lstm_predictive_coding}"
RECIPE_PATH="${Z7_GRU_RECIPE_PATH:-.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_lstm_predictive_coding_modal_t4_dispatch.yaml}"

if [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -n "${DISPATCH_INSTANCE_JOB_ID:-}" ]; then
    DEFAULT_OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    DEFAULT_OUTPUT_DIR="$WORKSPACE/lane_substrate_time_traveler_l5_z7_gru_results/output"
fi
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_time_traveler_l5_z7_gru_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
Z7_GRU_OUTPUT_DIR="${Z7_GRU_OUTPUT_DIR:-$OUTPUT_DIR}"
PROVENANCE="$LOG_DIR/provenance.json"

Z7_GRU_TRAINER_MODE="${Z7_GRU_TRAINER_MODE:-timing_smoke}"
SMOKE_ONLY="${SMOKE_ONLY:-}"
case "$Z7_GRU_TRAINER_MODE" in
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
        echo "[lane-z7-gru] FATAL: invalid Z7_GRU_TRAINER_MODE=$Z7_GRU_TRAINER_MODE; expected smoke|timing_smoke|full" >&2
        exit 29
        ;;
esac

Z7_GRU_VIDEO_PATH="${Z7_GRU_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
Z7_GRU_EPOCHS="${Z7_GRU_EPOCHS:-}"
Z7_GRU_BATCH_SIZE="${Z7_GRU_BATCH_SIZE:-}"
Z7_GRU_LR="${Z7_GRU_LR:-}"
Z7_GRU_DEVICE="${Z7_GRU_DEVICE:-cuda}"
Z7_GRU_LOSS_MODE="${Z7_GRU_LOSS_MODE:-score_aware}"
Z7_GRU_CONTEXT_CONDITIONING_MODE="${Z7_GRU_CONTEXT_CONDITIONING_MODE:-none}"
Z7_GRU_CONTEXT_AFFINE_STRENGTH="${Z7_GRU_CONTEXT_AFFINE_STRENGTH:-0.125}"
Z7_GRU_LATENT_DIM="${Z7_GRU_LATENT_DIM:-}"
Z7_GRU_EGO_MOTION_DIM="${Z7_GRU_EGO_MOTION_DIM:-}"
Z7_GRU_HIDDEN_DIM="${Z7_GRU_HIDDEN_DIM:-}"
Z7_GRU_NUM_LAYERS="${Z7_GRU_NUM_LAYERS:-}"
Z7_GRU_MAX_PAIRS="${Z7_GRU_MAX_PAIRS:-}"
Z7_GRU_DECODER_EMBED_DIM="${Z7_GRU_DECODER_EMBED_DIM:-}"
Z7_GRU_DECODER_CHANNELS="${Z7_GRU_DECODER_CHANNELS:-}"
Z7_GRU_DECODER_UPSAMPLE_BLOCKS="${Z7_GRU_DECODER_UPSAMPLE_BLOCKS:-}"
Z7_GRU_DECODER_INITIAL_GRID_H="${Z7_GRU_DECODER_INITIAL_GRID_H:-}"
Z7_GRU_DECODER_INITIAL_GRID_W="${Z7_GRU_DECODER_INITIAL_GRID_W:-}"
Z7_GRU_OUTPUT_HEIGHT="${Z7_GRU_OUTPUT_HEIGHT:-}"
Z7_GRU_OUTPUT_WIDTH="${Z7_GRU_OUTPUT_WIDTH:-}"

if [ "$Z7_GRU_TRAINER_MODE" = "smoke" ]; then
    Z7_GRU_EPOCHS="${Z7_GRU_EPOCHS:-1}"
    Z7_GRU_BATCH_SIZE="${Z7_GRU_BATCH_SIZE:-4}"
    Z7_GRU_LR="${Z7_GRU_LR:-5e-4}"
    Z7_GRU_LATENT_DIM="${Z7_GRU_LATENT_DIM:-24}"
    Z7_GRU_EGO_MOTION_DIM="${Z7_GRU_EGO_MOTION_DIM:-8}"
    Z7_GRU_HIDDEN_DIM="${Z7_GRU_HIDDEN_DIM:-128}"
    Z7_GRU_NUM_LAYERS="${Z7_GRU_NUM_LAYERS:-1}"
    Z7_GRU_MAX_PAIRS="${Z7_GRU_MAX_PAIRS:-1}"
    Z7_GRU_DECODER_EMBED_DIM="${Z7_GRU_DECODER_EMBED_DIM:-32}"
    Z7_GRU_DECODER_CHANNELS="${Z7_GRU_DECODER_CHANNELS:-32,24,16,12}"
    Z7_GRU_DECODER_UPSAMPLE_BLOCKS="${Z7_GRU_DECODER_UPSAMPLE_BLOCKS:-4}"
    Z7_GRU_DECODER_INITIAL_GRID_H="${Z7_GRU_DECODER_INITIAL_GRID_H:-24}"
    Z7_GRU_DECODER_INITIAL_GRID_W="${Z7_GRU_DECODER_INITIAL_GRID_W:-32}"
    Z7_GRU_OUTPUT_HEIGHT="${Z7_GRU_OUTPUT_HEIGHT:-384}"
    Z7_GRU_OUTPUT_WIDTH="${Z7_GRU_OUTPUT_WIDTH:-512}"
elif [ "$Z7_GRU_TRAINER_MODE" = "full" ]; then
    Z7_GRU_EPOCHS="${Z7_GRU_EPOCHS:-100}"
    Z7_GRU_BATCH_SIZE="${Z7_GRU_BATCH_SIZE:-4}"
    Z7_GRU_LR="${Z7_GRU_LR:-5e-4}"
    Z7_GRU_LATENT_DIM="${Z7_GRU_LATENT_DIM:-24}"
    Z7_GRU_EGO_MOTION_DIM="${Z7_GRU_EGO_MOTION_DIM:-8}"
    Z7_GRU_HIDDEN_DIM="${Z7_GRU_HIDDEN_DIM:-128}"
    Z7_GRU_NUM_LAYERS="${Z7_GRU_NUM_LAYERS:-1}"
    Z7_GRU_MAX_PAIRS="${Z7_GRU_MAX_PAIRS:-600}"
    Z7_GRU_DECODER_EMBED_DIM="${Z7_GRU_DECODER_EMBED_DIM:-32}"
    Z7_GRU_DECODER_CHANNELS="${Z7_GRU_DECODER_CHANNELS:-32,24,16,12}"
    Z7_GRU_DECODER_UPSAMPLE_BLOCKS="${Z7_GRU_DECODER_UPSAMPLE_BLOCKS:-4}"
    Z7_GRU_DECODER_INITIAL_GRID_H="${Z7_GRU_DECODER_INITIAL_GRID_H:-24}"
    Z7_GRU_DECODER_INITIAL_GRID_W="${Z7_GRU_DECODER_INITIAL_GRID_W:-32}"
    Z7_GRU_OUTPUT_HEIGHT="${Z7_GRU_OUTPUT_HEIGHT:-384}"
    Z7_GRU_OUTPUT_WIDTH="${Z7_GRU_OUTPUT_WIDTH:-512}"
else
    Z7_GRU_EPOCHS="${Z7_GRU_EPOCHS:-1}"
    Z7_GRU_BATCH_SIZE="${Z7_GRU_BATCH_SIZE:-1}"
    Z7_GRU_LR="${Z7_GRU_LR:-1e-3}"
    Z7_GRU_LATENT_DIM="${Z7_GRU_LATENT_DIM:-6}"
    Z7_GRU_EGO_MOTION_DIM="${Z7_GRU_EGO_MOTION_DIM:-3}"
    Z7_GRU_HIDDEN_DIM="${Z7_GRU_HIDDEN_DIM:-8}"
    Z7_GRU_NUM_LAYERS="${Z7_GRU_NUM_LAYERS:-1}"
    Z7_GRU_MAX_PAIRS="${Z7_GRU_MAX_PAIRS:-1}"
    Z7_GRU_DECODER_EMBED_DIM="${Z7_GRU_DECODER_EMBED_DIM:-4}"
    Z7_GRU_DECODER_CHANNELS="${Z7_GRU_DECODER_CHANNELS:-4,4}"
    Z7_GRU_DECODER_UPSAMPLE_BLOCKS="${Z7_GRU_DECODER_UPSAMPLE_BLOCKS:-2}"
    Z7_GRU_DECODER_INITIAL_GRID_H="${Z7_GRU_DECODER_INITIAL_GRID_H:-2}"
    Z7_GRU_DECODER_INITIAL_GRID_W="${Z7_GRU_DECODER_INITIAL_GRID_W:-2}"
    Z7_GRU_OUTPUT_HEIGHT="${Z7_GRU_OUTPUT_HEIGHT:-16}"
    Z7_GRU_OUTPUT_WIDTH="${Z7_GRU_OUTPUT_WIDTH:-16}"
fi

Z7_GRU_EGO_SOURCE="${Z7_GRU_EGO_SOURCE:-posenet_projection}"
Z7_GRU_IDENTITY_PREDICTOR="${Z7_GRU_IDENTITY_PREDICTOR:-false}"
Z7_GRU_STATEFUL="${Z7_GRU_STATEFUL:-true}"
Z7_GRU_BETA_IB="${Z7_GRU_BETA_IB:-1.0}"
Z7_GRU_UPSTREAM_DIR="${Z7_GRU_UPSTREAM_DIR:-$WORKSPACE/upstream}"
Z7_GRU_ALPHA_RATE="${Z7_GRU_ALPHA_RATE:-25.0}"
Z7_GRU_BETA_SEG="${Z7_GRU_BETA_SEG:-100.0}"
Z7_GRU_GAMMA_POSE="${Z7_GRU_GAMMA_POSE:-3.1622776601683795}"
Z7_GRU_NOISE_STD="${Z7_GRU_NOISE_STD:-0.0}"
Z7_GRU_INFLATE_VERIFY="${Z7_GRU_INFLATE_VERIFY:-true}"
Z7_GRU_EMIT_STATIC_CONTROL="${Z7_GRU_EMIT_STATIC_CONTROL:-true}"

DISPATCH_INSTANCE_JOB_ID="${Z7_GRU_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${Z7_GRU_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""
CLAIM_VERIFIED=0
EVIDENCE_MARKER="[not-yet-classified]"
SCORE_CLAIM_FLAG="score_claim=unknown"

log() { echo "[lane-z7-gru] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$Z7_GRU_OUTPUT_DIR"
cd "$WORKSPACE"

if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: Z7_GRU_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
    exit 21
fi

CLAIM_PYTHON="${PYBIN:-}"
if [ -z "$CLAIM_PYTHON" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
    CLAIM_PYTHON="$WORKSPACE/.venv/bin/python"
fi
if [ -z "$CLAIM_PYTHON" ]; then
    CLAIM_PYTHON="python3"
fi

verify_active_dispatch_claim() {
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        log "FATAL: claim helper missing; cannot verify active dispatch claim"
        exit 26
    fi
    local claim_summary_json="$LOG_DIR/dispatch_claim_summary.json"
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" summary \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --live-only \
        --format json \
        > "$claim_summary_json" || {
        log "FATAL: claim summary failed; refusing remote driver startup"
        exit 26
    }
    "$CLAIM_PYTHON" - "$claim_summary_json" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID" <<'PY' || {
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
lane_id = sys.argv[2]
job_id = sys.argv[3]
payload = json.loads(summary_path.read_text(encoding="utf-8"))
for row in payload.get("active", []):
    if row.get("lane_id") == lane_id and row.get("instance_job_id") == job_id:
        raise SystemExit(0)
print(
    f"no active dispatch claim for lane_id={lane_id} instance_job_id={job_id}",
    file=sys.stderr,
)
raise SystemExit(1)
PY
        log "FATAL: no active dispatch claim for lane=$LANE_ID instance/job=$DISPATCH_INSTANCE_JOB_ID"
        exit 27
    }
    CLAIM_VERIFIED=1
    log "Stage 0 DONE: active dispatch claim verified"
}

append_terminal_claim() {
    local rc="$1"
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        log "WARN: claim helper missing; cannot append terminal dispatch claim"
        return 0
    fi
    local status
    if [ "$rc" -eq 0 ]; then
        status="completed_z7_gru_remote_driver_no_score_claim"
    elif [ "${CLAIM_VERIFIED:-0}" != "1" ]; then
        status="failed_z7_gru_claim_verification_rc_${rc}"
    else
        status="failed_z7_gru_remote_driver_rc_${rc}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --force \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "remote_lane_substrate_time_traveler_l5_z7_lstm_predictive_coding" \
        --status "$status" \
        --notes "remote_driver_terminal rc=$rc evidence_marker=${EVIDENCE_MARKER:-unknown} ${SCORE_CLAIM_FLAG:-score_claim=unknown} stats_json=$Z7_STATS_JSON output_dir=$Z7_GRU_OUTPUT_DIR mode=$Z7_GRU_TRAINER_MODE loss_mode=$Z7_GRU_LOSS_MODE context_mode=$Z7_GRU_CONTEXT_CONDITIONING_MODE max_pairs=$Z7_GRU_MAX_PAIRS emit_static_control=$Z7_GRU_EMIT_STATIC_CONTROL" \
        >> "$LOG_DIR/run.log" 2>&1 || {
        log "WARN: failed to append terminal dispatch claim status=$status"
    }
}

cleanup() {
    local rc="$?"
    if [ -n "$HEARTBEAT_PID" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
        wait "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    append_terminal_claim "$rc"
    exit "$rc"
}

if [ "$SMOKE_ONLY" = "1" ]; then
    Z7_STATS_JSON="$Z7_GRU_OUTPUT_DIR/z7_gru_scaffold_smoke_stats.json"
else
    Z7_STATS_JSON="$Z7_GRU_OUTPUT_DIR/z7_gru_prebuild_full_main_export_stats.json"
fi

trap cleanup EXIT
verify_active_dispatch_claim

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

(
    HEARTBEAT_SLEEP_PID=""
    trap 'if [ -n "$HEARTBEAT_SLEEP_PID" ]; then kill "$HEARTBEAT_SLEEP_PID" 2>/dev/null || true; fi; exit 0' TERM INT EXIT
    while true; do
        date -u +%FT%TZ > "$LOG_DIR/heartbeat.log"
        sleep 300 &
        HEARTBEAT_SLEEP_PID="$!"
        wait "$HEARTBEAT_SLEEP_PID" || true
        HEARTBEAT_SLEEP_PID=""
    done
) &
HEARTBEAT_PID=$!

cat > "$PROVENANCE" <<EOF
{
  "lane_id": "$LANE_ID",
  "tag": "$TAG",
  "trainer": "experiments/train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py",
  "recipe": "$RECIPE_PATH",
  "video_path": "$Z7_GRU_VIDEO_PATH",
  "output_dir": "$Z7_GRU_OUTPUT_DIR",
  "mode": "$Z7_GRU_TRAINER_MODE",
  "epochs": "$Z7_GRU_EPOCHS",
  "batch_size": "$Z7_GRU_BATCH_SIZE",
  "lr": "$Z7_GRU_LR",
  "device": "$Z7_GRU_DEVICE",
  "loss_mode": "$Z7_GRU_LOSS_MODE",
  "context_conditioning_mode": "$Z7_GRU_CONTEXT_CONDITIONING_MODE",
  "context_affine_strength": "$Z7_GRU_CONTEXT_AFFINE_STRENGTH",
  "latent_dim": "$Z7_GRU_LATENT_DIM",
  "ego_motion_dim": "$Z7_GRU_EGO_MOTION_DIM",
  "gru_hidden_dim": "$Z7_GRU_HIDDEN_DIM",
  "gru_num_layers": "$Z7_GRU_NUM_LAYERS",
  "max_pairs": "$Z7_GRU_MAX_PAIRS",
  "decoder_embed_dim": "$Z7_GRU_DECODER_EMBED_DIM",
  "decoder_channels": "$Z7_GRU_DECODER_CHANNELS",
  "decoder_num_upsample_blocks": "$Z7_GRU_DECODER_UPSAMPLE_BLOCKS",
  "decoder_initial_grid_h": "$Z7_GRU_DECODER_INITIAL_GRID_H",
  "decoder_initial_grid_w": "$Z7_GRU_DECODER_INITIAL_GRID_W",
  "output_height": "$Z7_GRU_OUTPUT_HEIGHT",
  "output_width": "$Z7_GRU_OUTPUT_WIDTH",
  "ego_source": "$Z7_GRU_EGO_SOURCE",
  "beta_ib": "$Z7_GRU_BETA_IB",
  "inflate_verify": "$Z7_GRU_INFLATE_VERIFY",
  "emit_static_control": "$Z7_GRU_EMIT_STATIC_CONTROL",
  "dispatch_instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "started_at_utc": "$(date -u +%FT%TZ)"
}
EOF

TRAINER_PY="$WORKSPACE/experiments/train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py"
if [ ! -f "$TRAINER_PY" ]; then
    log "FATAL: trainer missing at $TRAINER_PY"
    exit 23
fi

SMOKE_FLAG_ARGS=()
if [ "$SMOKE_ONLY" = "1" ]; then
    SMOKE_FLAG_ARGS+=(--smoke)
fi

IDENTITY_FLAG_ARGS=()
case "$Z7_GRU_IDENTITY_PREDICTOR" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        IDENTITY_FLAG_ARGS+=(--identity-predictor)
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off|"")
        ;;
    *)
        log "FATAL: invalid Z7_GRU_IDENTITY_PREDICTOR=$Z7_GRU_IDENTITY_PREDICTOR; expected 0/1/true/false"
        exit 24
        ;;
esac

STATEFUL_FLAG_ARGS=()
case "$Z7_GRU_STATEFUL" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        STATEFUL_FLAG_ARGS+=(--stateful)
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off|"")
        STATEFUL_FLAG_ARGS+=(--no-stateful)
        ;;
    *)
        log "FATAL: invalid Z7_GRU_STATEFUL=$Z7_GRU_STATEFUL; expected 0/1/true/false"
        exit 25
        ;;
esac

INFLATE_VERIFY_ARGS=()
case "$Z7_GRU_INFLATE_VERIFY" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        INFLATE_VERIFY_ARGS+=(--inflate-verify)
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off|"")
        INFLATE_VERIFY_ARGS+=(--no-inflate-verify)
        ;;
    *)
        log "FATAL: invalid Z7_GRU_INFLATE_VERIFY=$Z7_GRU_INFLATE_VERIFY; expected 0/1/true/false"
        exit 28
        ;;
esac

STATIC_CONTROL_ARGS=()
case "$Z7_GRU_EMIT_STATIC_CONTROL" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        STATIC_CONTROL_ARGS+=(--emit-static-control)
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off|"")
        STATIC_CONTROL_ARGS+=(--no-emit-static-control)
        ;;
    *)
        log "FATAL: invalid Z7_GRU_EMIT_STATIC_CONTROL=$Z7_GRU_EMIT_STATIC_CONTROL; expected 0/1/true/false"
        exit 30
        ;;
esac

if [ -f "$Z7_STATS_JSON" ]; then
    STALE_STATS_DIR="$LOG_DIR/stale_stats_quarantine"
    mkdir -p "$STALE_STATS_DIR"
    STALE_STATS_QUARANTINE="$STALE_STATS_DIR/$(basename "$Z7_STATS_JSON").before_${DISPATCH_INSTANCE_JOB_ID:-unknown}.$(date -u +%Y%m%dT%H%M%SZ).$$.json"
    mv "$Z7_STATS_JSON" "$STALE_STATS_QUARANTINE"
    log "quarantined_preexisting_stats_json path=$STALE_STATS_QUARANTINE"
fi

log "stage_4_trainer_begin mode=$Z7_GRU_TRAINER_MODE epochs=$Z7_GRU_EPOCHS max_pairs=$Z7_GRU_MAX_PAIRS loss_mode=$Z7_GRU_LOSS_MODE context_mode=$Z7_GRU_CONTEXT_CONDITIONING_MODE device=$Z7_GRU_DEVICE"
REMOTE_DRIVER_STAGE4_STARTED_UNIX="$(date +%s)"
"$CLAIM_PYTHON" "$TRAINER_PY" \
    --video-path "$Z7_GRU_VIDEO_PATH" \
    --output-dir "$Z7_GRU_OUTPUT_DIR" \
    --epochs "$Z7_GRU_EPOCHS" \
    --batch-size "$Z7_GRU_BATCH_SIZE" \
    --lr "$Z7_GRU_LR" \
    --gru-hidden-dim "$Z7_GRU_HIDDEN_DIM" \
    --gru-num-layers "$Z7_GRU_NUM_LAYERS" \
    --ego-source "$Z7_GRU_EGO_SOURCE" \
    --ego-motion-dim "$Z7_GRU_EGO_MOTION_DIM" \
    --beta-ib "$Z7_GRU_BETA_IB" \
    --latent-dim "$Z7_GRU_LATENT_DIM" \
    --max-pairs "$Z7_GRU_MAX_PAIRS" \
    --decoder-embed-dim "$Z7_GRU_DECODER_EMBED_DIM" \
    --decoder-channels "$Z7_GRU_DECODER_CHANNELS" \
    --decoder-num-upsample-blocks "$Z7_GRU_DECODER_UPSAMPLE_BLOCKS" \
    --decoder-initial-grid-h "$Z7_GRU_DECODER_INITIAL_GRID_H" \
    --decoder-initial-grid-w "$Z7_GRU_DECODER_INITIAL_GRID_W" \
    --output-height "$Z7_GRU_OUTPUT_HEIGHT" \
    --output-width "$Z7_GRU_OUTPUT_WIDTH" \
    --loss-mode "$Z7_GRU_LOSS_MODE" \
    --context-conditioning-mode "$Z7_GRU_CONTEXT_CONDITIONING_MODE" \
    --context-affine-strength "$Z7_GRU_CONTEXT_AFFINE_STRENGTH" \
    --upstream-dir "$Z7_GRU_UPSTREAM_DIR" \
    --alpha-rate "$Z7_GRU_ALPHA_RATE" \
    --beta-seg "$Z7_GRU_BETA_SEG" \
    --gamma-pose "$Z7_GRU_GAMMA_POSE" \
    --noise-std "$Z7_GRU_NOISE_STD" \
    --device "$Z7_GRU_DEVICE" \
    ${SMOKE_FLAG_ARGS[@]+"${SMOKE_FLAG_ARGS[@]}"} \
    ${IDENTITY_FLAG_ARGS[@]+"${IDENTITY_FLAG_ARGS[@]}"} \
    ${STATEFUL_FLAG_ARGS[@]+"${STATEFUL_FLAG_ARGS[@]}"} \
    ${INFLATE_VERIFY_ARGS[@]+"${INFLATE_VERIFY_ARGS[@]}"} \
    ${STATIC_CONTROL_ARGS[@]+"${STATIC_CONTROL_ARGS[@]}"} \
    2>&1 | tee -a "$LOG_DIR/trainer.log"

EVIDENCE_STATUS="$("$CLAIM_PYTHON" - "$Z7_STATS_JSON" "$REMOTE_DRIVER_STAGE4_STARTED_UNIX" <<'PY'
import json
import sys
from pathlib import Path

stats_path = Path(sys.argv[1])
stage4_started_unix = float(sys.argv[2])
marker = "[training-artifact-no-score-claim]"
score_claim = "score_claim=false"
if not stats_path.is_file():
    print(f"missing required stats JSON at {stats_path}", file=sys.stderr)
    raise SystemExit(31)
stats_mtime = stats_path.stat().st_mtime
if stats_mtime < stage4_started_unix:
    print(
        f"stale stats JSON at {stats_path}: "
        f"mtime={stats_mtime:.6f} < stage4_started_unix={stage4_started_unix:.6f}",
        file=sys.stderr,
    )
    raise SystemExit(33)
try:
    stats = json.loads(stats_path.read_text())
except json.JSONDecodeError as exc:
    print(f"malformed stats JSON at {stats_path}: {exc}", file=sys.stderr)
    raise SystemExit(32)
if stats.get("score_claim") is not False:
    print("Z7 remote driver refuses stats without score_claim=false", file=sys.stderr)
    raise SystemExit(34)
if stats.get("promotion_eligible") is not False:
    print("Z7 remote driver refuses stats without promotion_eligible=false", file=sys.stderr)
    raise SystemExit(35)
if stats.get("ready_for_paid_dispatch") is not False:
    print("Z7 remote driver refuses stats without ready_for_paid_dispatch=false", file=sys.stderr)
    raise SystemExit(36)
if stats.get("evidence_grade"):
    marker = f"[{stats['evidence_grade']}]"
print(f"{marker} {score_claim}")
PY
)"
EVIDENCE_MARKER="${EVIDENCE_STATUS%% *}"
SCORE_CLAIM_FLAG="${EVIDENCE_STATUS#* }"
log "LANE_Z7_GRU_DONE ${EVIDENCE_MARKER} output_dir=$Z7_GRU_OUTPUT_DIR mode=$Z7_GRU_TRAINER_MODE max_pairs=$Z7_GRU_MAX_PAIRS ${SCORE_CLAIM_FLAG}"
echo "LANE_Z7_GRU_DONE ${EVIDENCE_MARKER} $LANE_ID ${SCORE_CLAIM_FLAG} mode=$Z7_GRU_TRAINER_MODE $(date -u +%FT%TZ)" >> "$LOG_DIR/completion.log"
