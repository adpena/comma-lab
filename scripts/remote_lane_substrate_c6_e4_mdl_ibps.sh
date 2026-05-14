#!/bin/bash
# Remote lane script: substrate C6 MDL-IBPS (IBPS1) first-anchor dispatch.
#
# Trainer: experiments/train_substrate_c6_e4_mdl_ibps.py
# Lane: lane_c6_e4_mdl_ibps_substrate_20260514
# Recipe: .omx/operator_authorize_recipes/substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml
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
#   - .omx/research/campaign_lane_c6_e4_mdl_ibps_substrate_20260514.md
#   - .omx/research/adjusted_theoretical_floor_v3_post_pr106_falsification_20260513.md
#   - Tishby & Zaslavsky 2015 IB; Rissanen 1978 MDL; Alemi et al. 2017 VIB
#
# Score-tagging: any score this script produces is tagged [contest-CUDA] in
# the completion-log line (LANE_C6_MDL_IBPS_DONE marker).
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_c6_e4_mdl_ibps_substrate_20260514"
TAG="${TAG:-substrate_c6_e4_mdl_ibps}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_c6_e4_mdl_ibps_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
C6_E4_MDL_IBPS_VIDEO_PATH="${C6_E4_MDL_IBPS_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
C6_E4_MDL_IBPS_OUTPUT_DIR="${C6_E4_MDL_IBPS_OUTPUT_DIR:-$OUTPUT_DIR}"
C6_E4_MDL_IBPS_EPOCHS="${C6_E4_MDL_IBPS_EPOCHS:-200}"
C6_E4_MDL_IBPS_BATCH_SIZE="${C6_E4_MDL_IBPS_BATCH_SIZE:-4}"
C6_E4_MDL_IBPS_LR="${C6_E4_MDL_IBPS_LR:-5e-4}"
C6_E4_MDL_IBPS_LATENT_DIM="${C6_E4_MDL_IBPS_LATENT_DIM:-24}"
C6_E4_MDL_IBPS_BETA_IB="${C6_E4_MDL_IBPS_BETA_IB:-0.01}"
C6_E4_MDL_IBPS_UPSTREAM_DIR="${C6_E4_MDL_IBPS_UPSTREAM_DIR:-$WORKSPACE/upstream}"
C6_E4_MDL_IBPS_DEVICE="${C6_E4_MDL_IBPS_DEVICE:-cuda}"

DISPATCH_INSTANCE_JOB_ID="${C6_E4_MDL_IBPS_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${C6_E4_MDL_IBPS_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""
CLAIM_VERIFIED=0

log() { echo "[lane-c6-mdl-ibps] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: C6_E4_MDL_IBPS_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
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
        status="completed_c6_mdl_ibps_remote_driver"
    elif [ "${CLAIM_VERIFIED:-0}" != "1" ]; then
        status="failed_c6_mdl_ibps_claim_verification_rc_${rc}"
    else
        status="failed_c6_mdl_ibps_remote_driver_rc_${rc}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --force \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "remote_lane_substrate_c6_e4_mdl_ibps" \
        --status "$status" \
        --notes "remote_driver_terminal rc=$rc output_dir=$C6_E4_MDL_IBPS_OUTPUT_DIR" \
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
  "trainer": "experiments/train_substrate_c6_e4_mdl_ibps.py",
  "recipe": ".omx/operator_authorize_recipes/substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml",
  "video_path": "$C6_E4_MDL_IBPS_VIDEO_PATH",
  "output_dir": "$C6_E4_MDL_IBPS_OUTPUT_DIR",
  "epochs": "$C6_E4_MDL_IBPS_EPOCHS",
  "batch_size": "$C6_E4_MDL_IBPS_BATCH_SIZE",
  "lr": "$C6_E4_MDL_IBPS_LR",
  "latent_dim": "$C6_E4_MDL_IBPS_LATENT_DIM",
  "beta_ib": "$C6_E4_MDL_IBPS_BETA_IB",
  "device": "$C6_E4_MDL_IBPS_DEVICE",
  "dispatch_instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "started_at_utc": "$(date -u +%FT%TZ)"
}
EOF

# Stage 4: invoke trainer.
log "stage_4_trainer_begin epochs=$C6_E4_MDL_IBPS_EPOCHS latent_dim=$C6_E4_MDL_IBPS_LATENT_DIM beta_ib=$C6_E4_MDL_IBPS_BETA_IB"
TRAINER_PY="$WORKSPACE/experiments/train_substrate_c6_e4_mdl_ibps.py"
if [ ! -f "$TRAINER_PY" ]; then
    log "FATAL: trainer missing at $TRAINER_PY"
    exit 23
fi

PYBIN_RESOLVED="$CLAIM_PYTHON"

"$PYBIN_RESOLVED" "$TRAINER_PY" \
    --video-path "$C6_E4_MDL_IBPS_VIDEO_PATH" \
    --output-dir "$C6_E4_MDL_IBPS_OUTPUT_DIR" \
    --epochs "$C6_E4_MDL_IBPS_EPOCHS" \
    --batch-size "$C6_E4_MDL_IBPS_BATCH_SIZE" \
    --lr "$C6_E4_MDL_IBPS_LR" \
    --latent-dim "$C6_E4_MDL_IBPS_LATENT_DIM" \
    --beta-ib "$C6_E4_MDL_IBPS_BETA_IB" \
    --upstream-dir "$C6_E4_MDL_IBPS_UPSTREAM_DIR" \
    --device "$C6_E4_MDL_IBPS_DEVICE" \
    2>&1 | tee -a "$LOG_DIR/trainer.log"

# Stage 5: emit completion marker (operator + autopilot consume).
log "LANE_C6_MDL_IBPS_DONE [contest-CUDA] output_dir=$C6_E4_MDL_IBPS_OUTPUT_DIR"
echo "LANE_C6_MDL_IBPS_DONE [contest-CUDA] $LANE_ID $(date -u +%FT%TZ)" >> "$LOG_DIR/completion.log"
