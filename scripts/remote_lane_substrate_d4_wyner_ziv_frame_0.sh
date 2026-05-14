#!/bin/bash
# Remote lane script: substrate D4 Wyner-Ziv frame-0 (WZF01) first-anchor dispatch.
#
# Trainer: experiments/train_substrate_d4_wyner_ziv_frame_0.py
# Lane: lane_d4_wyner_ziv_frame_0_substrate_20260514
# Recipe: .omx/operator_authorize_recipes/substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.yaml
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
#   - .omx/research/deep_math_geometry_manifolds_synthesis_20260514.md §3.5 + §6 D4
#   - Wyner & Ziv 1976; Slepian & Wolf 1973; Lucas-Kanade 1981; Rodrigues 1840
#
# Score-tagging: any score this script produces is tagged [contest-CUDA] in
# the completion-log line (LANE_D4_WZF0_DONE marker).
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_d4_wyner_ziv_frame_0_substrate_20260514"
TAG="${TAG:-substrate_d4_wyner_ziv_frame_0}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_d4_wyner_ziv_frame_0_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
D4_WYNER_ZIV_FRAME_0_VIDEO_PATH="${D4_WYNER_ZIV_FRAME_0_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
D4_WYNER_ZIV_FRAME_0_OUTPUT_DIR="${D4_WYNER_ZIV_FRAME_0_OUTPUT_DIR:-$OUTPUT_DIR}"
D4_WYNER_ZIV_FRAME_0_EPOCHS="${D4_WYNER_ZIV_FRAME_0_EPOCHS:-2000}"
D4_WYNER_ZIV_FRAME_0_UPSTREAM_DIR="${D4_WYNER_ZIV_FRAME_0_UPSTREAM_DIR:-$WORKSPACE/upstream}"
D4_WYNER_ZIV_FRAME_0_DEVICE="${D4_WYNER_ZIV_FRAME_0_DEVICE:-cuda}"
D4_WYNER_ZIV_FRAME_0_MOTION_MODE="${D4_WYNER_ZIV_FRAME_0_MOTION_MODE:-se3_parametric}"
D4_WYNER_ZIV_FRAME_0_RESIDUAL_COARSE_H="${D4_WYNER_ZIV_FRAME_0_RESIDUAL_COARSE_H:-48}"
D4_WYNER_ZIV_FRAME_0_RESIDUAL_COARSE_W="${D4_WYNER_ZIV_FRAME_0_RESIDUAL_COARSE_W:-64}"
D4_WYNER_ZIV_FRAME_0_BASE_ARCHIVE_PATH="${D4_WYNER_ZIV_FRAME_0_BASE_ARCHIVE_PATH:-}"
D4_WYNER_ZIV_FRAME_0_MAX_PAIRS="${D4_WYNER_ZIV_FRAME_0_MAX_PAIRS:-}"

DISPATCH_INSTANCE_JOB_ID="${D4_WYNER_ZIV_FRAME_0_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${D4_WYNER_ZIV_FRAME_0_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""
CLAIM_VERIFIED=0

log() { echo "[lane-d4-wzf0] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: D4_WYNER_ZIV_FRAME_0_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
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
        status="completed_d4_wzf0_remote_driver"
    elif [ "${CLAIM_VERIFIED:-0}" != "1" ]; then
        status="failed_d4_wzf0_claim_verification_rc_${rc}"
    else
        status="failed_d4_wzf0_remote_driver_rc_${rc}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --force \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "remote_lane_substrate_d4_wyner_ziv_frame_0" \
        --status "$status" \
        --notes "remote_driver_terminal rc=$rc output_dir=$D4_WYNER_ZIV_FRAME_0_OUTPUT_DIR" \
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
  "trainer": "experiments/train_substrate_d4_wyner_ziv_frame_0.py",
  "recipe": ".omx/operator_authorize_recipes/substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.yaml",
  "video_path": "$D4_WYNER_ZIV_FRAME_0_VIDEO_PATH",
  "output_dir": "$D4_WYNER_ZIV_FRAME_0_OUTPUT_DIR",
  "epochs": "$D4_WYNER_ZIV_FRAME_0_EPOCHS",
  "device": "$D4_WYNER_ZIV_FRAME_0_DEVICE",
  "motion_mode": "$D4_WYNER_ZIV_FRAME_0_MOTION_MODE",
  "residual_coarse_h": "$D4_WYNER_ZIV_FRAME_0_RESIDUAL_COARSE_H",
  "residual_coarse_w": "$D4_WYNER_ZIV_FRAME_0_RESIDUAL_COARSE_W",
  "base_archive_path": "$D4_WYNER_ZIV_FRAME_0_BASE_ARCHIVE_PATH",
  "dispatch_instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "started_at_utc": "$(date -u +%FT%TZ)"
}
EOF

# Stage 4: invoke trainer.
log "stage_4_trainer_begin motion_mode=$D4_WYNER_ZIV_FRAME_0_MOTION_MODE epochs=$D4_WYNER_ZIV_FRAME_0_EPOCHS"
TRAINER_PY="$WORKSPACE/experiments/train_substrate_d4_wyner_ziv_frame_0.py"
if [ ! -f "$TRAINER_PY" ]; then
    log "FATAL: trainer missing at $TRAINER_PY"
    exit 23
fi

PYBIN_RESOLVED="$CLAIM_PYTHON"

MAX_PAIRS_ARGS=()
if [ -n "${D4_WYNER_ZIV_FRAME_0_MAX_PAIRS:-}" ]; then
    MAX_PAIRS_ARGS+=(--max-pairs "$D4_WYNER_ZIV_FRAME_0_MAX_PAIRS")
fi

"$PYBIN_RESOLVED" "$TRAINER_PY" \
    --video-path "$D4_WYNER_ZIV_FRAME_0_VIDEO_PATH" \
    --output-dir "$D4_WYNER_ZIV_FRAME_0_OUTPUT_DIR" \
    --epochs "$D4_WYNER_ZIV_FRAME_0_EPOCHS" \
    --upstream-dir "$D4_WYNER_ZIV_FRAME_0_UPSTREAM_DIR" \
    --device "$D4_WYNER_ZIV_FRAME_0_DEVICE" \
    --motion-mode "$D4_WYNER_ZIV_FRAME_0_MOTION_MODE" \
    --residual-coarse-h "$D4_WYNER_ZIV_FRAME_0_RESIDUAL_COARSE_H" \
    --residual-coarse-w "$D4_WYNER_ZIV_FRAME_0_RESIDUAL_COARSE_W" \
    --base-archive-path "$D4_WYNER_ZIV_FRAME_0_BASE_ARCHIVE_PATH" \
    ${MAX_PAIRS_ARGS[@]+"${MAX_PAIRS_ARGS[@]}"} \
    2>&1 | tee -a "$LOG_DIR/trainer.log"

# Stage 5: emit completion marker (operator + autopilot consume).
log "LANE_D4_WZF0_DONE [contest-CUDA] output_dir=$D4_WYNER_ZIV_FRAME_0_OUTPUT_DIR"
echo "LANE_D4_WZF0_DONE [contest-CUDA] $LANE_ID $(date -u +%FT%TZ)" >> "$LOG_DIR/completion.log"
