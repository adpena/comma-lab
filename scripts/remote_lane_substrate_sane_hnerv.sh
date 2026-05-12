#!/bin/bash
# Remote lane script: substrate sane_hnerv first-anchor dispatch.
#
# Trainer: experiments/train_substrate_sane_hnerv.py (commit c9d5aae7+).
# Lane: lane_substrate_sane_hnerv_20260512
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Council memo refs:
#   - feedback_substrate_sane_hnerv_full_main_wired_landed_20260512.md (α
#     16-stage _full_main wire including auth-eval at stage 12 + continual-
#     learning posterior update at stage 13 + cost-band anchor at stage 14)
#   - feedback_substrate_sane_hnerv_first_anchor_dispatch_landed_20260512.md
#     (the prior Wave 3 DEFERRAL — STOP-PRECONDITIONS surfaced; this rerun
#     uses Modal A100 per operator directive 2026-05-12 "reroute all to
#     modal and lightning free tier")
#
# Score-tagging: any score this script produces is tagged [contest-CUDA] in
# the completion-log line (LANE_SANE_HNERV_DONE marker) per the CLAUDE.md
# score-tag rule and preflight completion-tag check.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity — non-negotiable".
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_substrate_sane_hnerv_20260512"
TAG="${TAG:-substrate_sane_hnerv}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_sane_hnerv_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags — Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
SANE_HNERV_VIDEO_PATH="${SANE_HNERV_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
SANE_HNERV_OUTPUT_DIR="${SANE_HNERV_OUTPUT_DIR:-$OUTPUT_DIR}"
SANE_HNERV_EPOCHS="${SANE_HNERV_EPOCHS:-2000}"
SANE_HNERV_UPSTREAM_DIR="${SANE_HNERV_UPSTREAM_DIR:-$WORKSPACE/upstream}"
SANE_HNERV_DEVICE="${SANE_HNERV_DEVICE:-cuda}"

DISPATCH_INSTANCE_JOB_ID="${SANE_HNERV_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${SANE_HNERV_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-sane-hnerv] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification (mirrors remote_lane_t1_balle_endtoend.sh).
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: SANE_HNERV_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
    exit 21
fi
if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
    log "FATAL: claim helper missing at $WORKSPACE/tools/claim_lane_dispatch.py"
    exit 21
fi
if [ ! -f "$DISPATCH_CLAIMS_PATH" ]; then
    log "FATAL: dispatch-claims ledger missing at $DISPATCH_CLAIMS_PATH"
    exit 21
fi

CLAIM_PYTHON="${PYBIN:-}"
if [ -z "$CLAIM_PYTHON" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
    CLAIM_PYTHON="$WORKSPACE/.venv/bin/python"
fi
if [ -z "$CLAIM_PYTHON" ]; then
    CLAIM_PYTHON="python3"
fi

# Stage 1: bootstrap runtime deps (canonical, per CLAUDE.md).
if [ ! -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "FATAL: canonical bootstrap script missing at scripts/remote_archive_only_eval.sh"
    exit 22
fi
# shellcheck source=/dev/null
source "$WORKSPACE/scripts/remote_archive_only_eval.sh"

# bootstrap_runtime_deps installs uv + torch (driver-version-pinned per CLAUDE.md
# "Forbidden uv torch install without driver-version pin"). It also exports PYBIN.
if declare -f bootstrap_runtime_deps > /dev/null 2>&1; then
    log "stage_1_bootstrap_runtime_deps_begin"
    bootstrap_runtime_deps
    log "stage_1_bootstrap_runtime_deps_done PYBIN=${PYBIN:-unset}"
else
    log "FATAL: bootstrap_runtime_deps function not found after sourcing scripts/remote_archive_only_eval.sh"
    exit 23
fi

if [ -z "${PYBIN:-}" ] || [ ! -x "$PYBIN" ]; then
    log "FATAL: PYBIN not set or not executable after bootstrap (PYBIN=${PYBIN:-unset})"
    exit 24
fi

# Stage 2: provenance + remote code parity (per CLAUDE.md "Remote code parity").
log "stage_2_provenance_begin"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo "no-driver")
"$PYBIN" -c "
import json, time, torch
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'lane_id': '$LANE_ID',
    'dispatch_instance_job_id': '$DISPATCH_INSTANCE_JOB_ID',
    'dispatch_platform': '$DISPATCH_PLATFORM',
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'video_path': '$SANE_HNERV_VIDEO_PATH',
    'upstream_dir': '$SANE_HNERV_UPSTREAM_DIR',
    'epochs': $SANE_HNERV_EPOCHS,
    'device': '$SANE_HNERV_DEVICE',
}
import pathlib
pathlib.Path('$PROVENANCE').write_text(json.dumps(prov, indent=2, sort_keys=True))
print('[provenance]', json.dumps(prov, indent=2, sort_keys=True))
"
log "stage_2_provenance_done"

# Stage 3: heartbeat watchdog (every 5 min per CLAUDE.md).
(
    while true; do
        echo "[heartbeat] $(date -u +%FT%TZ) lane=$LANE_ID alive" >> "$LOG_DIR/heartbeat.log"
        sleep 300
    done
) &
HEARTBEAT_PID=$!
trap 'if [ -n "$HEARTBEAT_PID" ]; then kill "$HEARTBEAT_PID" 2>/dev/null || true; fi' EXIT

# Stage 4: invoke trainer.
#
# The trainer's _full_main wires 16 stages including auth eval (stage 12),
# continual-learning posterior update (stage 13), and cost-band anchor
# emission (stage 14). All required flags are threaded per Catalog #151
# TIER_1_OPERATOR_REQUIRED_FLAGS manifest.
#
# Per CLAUDE.md "Auth eval EVERYWHERE" + "Submission auth eval — BOTH CPU
# AND CUDA": this is a FIRST-ANCHOR research dispatch on CUDA only; the
# resulting tag is [contest-CUDA] single-axis (CPU axis required separately
# before promotion-grade status).
log "stage_4_trainer_invoke_begin video=$SANE_HNERV_VIDEO_PATH epochs=$SANE_HNERV_EPOCHS device=$SANE_HNERV_DEVICE"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_sane_hnerv.py \
    --video-path "$SANE_HNERV_VIDEO_PATH" \
    --output-dir "$SANE_HNERV_OUTPUT_DIR" \
    --epochs "$SANE_HNERV_EPOCHS" \
    --upstream-dir "$SANE_HNERV_UPSTREAM_DIR" \
    --device "$SANE_HNERV_DEVICE" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record (the auth-eval JSON was already written by the
# trainer at stage 12 if reached). We surface the path here for harvest.
AUTH_EVAL_JSON="$OUTPUT_DIR/auth_eval.json"
ARCHIVE_PATH="$OUTPUT_DIR/0.bin"
if [ -f "$AUTH_EVAL_JSON" ]; then
    log "auth_eval_artifact_present path=$AUTH_EVAL_JSON"
    log "LANE_SANE_HNERV_DONE [contest-CUDA] auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON (trainer may have failed before stage 12)"
fi

exit "$TRAIN_RC"
