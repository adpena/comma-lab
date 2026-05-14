#!/bin/bash
# Remote lane script: substrate C1 world-model + foveation (C1WMFV1) smoke dispatch.
#
# Trainer: experiments/train_substrate_c1_world_model_foveation.py
# Lane: lane_c1_world_model_foveation_campaign_l1_scaffold_20260514
# Recipe: .omx/operator_authorize_recipes/substrate_c1_world_model_foveation_modal_t4_smoke_dispatch.yaml
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
# Design refs:
#   - .omx/research/campaign_c1_world_model_foveation_20260514.md
#   - C1 fair probe v2/adversarial review: identity/no-world-model is the
#     executable default; GRU/LSTM/Transformer require explicit opt-in.
#   - Atick-Redlich 1990 foveation remains the default C1 signal.
#
# V1 SCOPE: smoke dispatch validates substrate plumbing (~$1 Modal T4 100ep).
# Full multi-stage training requires Phase 3 council approval; trainer's
# _full_main raises NotImplementedError before council.
#
# Score-tagging: this L1 driver emits [smoke-no-scorer] training artifacts
# only. It makes no score claim and never launches full training.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_c1_world_model_foveation_campaign_l1_scaffold_20260514"
TAG="${TAG:-substrate_c1_world_model_foveation}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_c1_world_model_foveation_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
C1_WORLD_MODEL_FOVEATION_VIDEO_PATH="${C1_WORLD_MODEL_FOVEATION_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
C1_WORLD_MODEL_FOVEATION_OUTPUT_DIR="${C1_WORLD_MODEL_FOVEATION_OUTPUT_DIR:-$OUTPUT_DIR}"
C1_WORLD_MODEL_FOVEATION_EPOCHS="${C1_WORLD_MODEL_FOVEATION_EPOCHS:-100}"
C1_WORLD_MODEL_FOVEATION_UPSTREAM_DIR="${C1_WORLD_MODEL_FOVEATION_UPSTREAM_DIR:-$WORKSPACE/upstream}"
C1_WORLD_MODEL_FOVEATION_DEVICE="${C1_WORLD_MODEL_FOVEATION_DEVICE:-cuda}"
C1_WORLD_MODEL_FOVEATION_RECURRENCE_MODE="${C1_WORLD_MODEL_FOVEATION_RECURRENCE_MODE:-identity_no_world_model}"
C1_WORLD_MODEL_FOVEATION_FOVEATION_STRATEGY="${C1_WORLD_MODEL_FOVEATION_FOVEATION_STRATEGY:-ego_motion_radial}"
C1_WORLD_MODEL_FOVEATION_LATENT_DIM="${C1_WORLD_MODEL_FOVEATION_LATENT_DIM:-64}"
# Modal durable output path per Catalog #204
if [ -n "${MODAL_RUNTIME:-}" ] && [ -n "${DISPATCH_INSTANCE_JOB_ID:-}" ]; then
    C1_WORLD_MODEL_FOVEATION_OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
fi

DISPATCH_INSTANCE_JOB_ID="${C1_WORLD_MODEL_FOVEATION_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${C1_WORLD_MODEL_FOVEATION_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""
CLAIM_VERIFIED=0

log() { echo "[lane-c1-wmf] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification (Catalog #157 / Cross-agent dispatch coordination).
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: C1_WORLD_MODEL_FOVEATION_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required"
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
        status="completed_c1_wmf_remote_driver"
    elif [ "${CLAIM_VERIFIED:-0}" != "1" ]; then
        status="failed_c1_wmf_claim_verification_rc_${rc}"
    else
        status="failed_c1_wmf_remote_driver_rc_${rc}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --force \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "remote_lane_substrate_c1_world_model_foveation" \
        --status "$status" \
        --notes "remote_driver_terminal rc=$rc output_dir=$C1_WORLD_MODEL_FOVEATION_OUTPUT_DIR" \
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

# Stage 3: write provenance (Catalog #166 source-parity sentinel anchor).
cat > "$PROVENANCE" <<EOF
{
  "lane_id": "$LANE_ID",
  "tag": "$TAG",
  "trainer": "experiments/train_substrate_c1_world_model_foveation.py",
  "recipe": ".omx/operator_authorize_recipes/substrate_c1_world_model_foveation_modal_t4_smoke_dispatch.yaml",
  "video_path": "$C1_WORLD_MODEL_FOVEATION_VIDEO_PATH",
  "output_dir": "$C1_WORLD_MODEL_FOVEATION_OUTPUT_DIR",
  "epochs": "$C1_WORLD_MODEL_FOVEATION_EPOCHS",
  "device": "$C1_WORLD_MODEL_FOVEATION_DEVICE",
  "recurrence_mode": "$C1_WORLD_MODEL_FOVEATION_RECURRENCE_MODE",
  "foveation_strategy": "$C1_WORLD_MODEL_FOVEATION_FOVEATION_STRATEGY",
  "latent_dim": "$C1_WORLD_MODEL_FOVEATION_LATENT_DIM",
  "dispatch_instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "started_at_utc": "$(date -u +%FT%TZ)"
}
EOF

# Stage 4: invoke trainer (SMOKE-only path per V1 scope).
log "stage_4_trainer_begin recurrence=$C1_WORLD_MODEL_FOVEATION_RECURRENCE_MODE foveation=$C1_WORLD_MODEL_FOVEATION_FOVEATION_STRATEGY epochs=$C1_WORLD_MODEL_FOVEATION_EPOCHS"
TRAINER_PY="$WORKSPACE/experiments/train_substrate_c1_world_model_foveation.py"
if [ ! -f "$TRAINER_PY" ]; then
    log "FATAL: trainer missing at $TRAINER_PY"
    exit 23
fi

PYBIN_RESOLVED="$CLAIM_PYTHON"

"$PYBIN_RESOLVED" "$TRAINER_PY" \
    --video-path "$C1_WORLD_MODEL_FOVEATION_VIDEO_PATH" \
    --output-dir "$C1_WORLD_MODEL_FOVEATION_OUTPUT_DIR" \
    --epochs "$C1_WORLD_MODEL_FOVEATION_EPOCHS" \
    --upstream-dir "$C1_WORLD_MODEL_FOVEATION_UPSTREAM_DIR" \
    --device "$C1_WORLD_MODEL_FOVEATION_DEVICE" \
    --recurrence-mode "$C1_WORLD_MODEL_FOVEATION_RECURRENCE_MODE" \
    --foveation-strategy "$C1_WORLD_MODEL_FOVEATION_FOVEATION_STRATEGY" \
    --latent-dim "$C1_WORLD_MODEL_FOVEATION_LATENT_DIM" \
    --smoke \
    2>&1 | tee -a "$LOG_DIR/trainer.log"

# Stage 5: emit completion marker.
log "LANE_C1_WMF_DONE [smoke-no-scorer] output_dir=$C1_WORLD_MODEL_FOVEATION_OUTPUT_DIR"
cat >> "$LOG_DIR/completion.log" <<EOF
LANE_C1_WMF_DONE [smoke-no-scorer] $LANE_ID $(date -u +%FT%TZ)
EOF
