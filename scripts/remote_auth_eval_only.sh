#!/bin/bash
# Single-purpose remote bootstrap: AUTH EVAL ONLY for a pre-built archive.
#
# Use case (2026-04-27): Lane A's Texas instance (35691284) ran 3.4h of pose
# TTO successfully and produced experiments/results/lane_a_landed/archive_lane_a.zip
# (694045 bytes), then the eval crashed with CUDA_ERROR_NO_DEVICE because
# NVDEC was missing on that host. We pulled the artifacts; we now need ONLY
# the auth eval on a fresh NVDEC-verified host. NO pose TTO, NO rebuild —
# the EXACT archive bytes must be evaluated to compare against the 2.29 baseline.
#
# Stages (mirrors remote_lane_a_pose_tto.sh patterns):
#   0. NVDEC probe (REFUSE to spend GPU on a host that can't run evaluate.py)
#   1. Pre-flight: required artifacts present
#   2. contest_auth_eval.py on the pinned archive bytes
#   3. RESULT_JSON capture + completion marker
#
# Heartbeat: 60s file write to $LOG_DIR/heartbeat.log (mirrors other scripts).
#
# Usage on remote (after rsync of archive to $WORKSPACE/auth_eval_input/archive.zip):
#   bash scripts/remote_auth_eval_only.sh
#
# Total wall time on a healthy 4090 with NVDEC: ~10-15 min (inflate ~5min,
# evaluate ~5-8 min). If it runs longer than 25 min, something is wrong.
set -euo pipefail

WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
ARCHIVE_INPUT="$WORKSPACE/auth_eval_input/archive.zip"
LOG_DIR="$WORKSPACE/auth_eval_results"
mkdir -p "$LOG_DIR"

# Source canonical env (PATH, PYTHONPATH, CUBLAS_WORKSPACE_CONFIG, FFMPEG_BIN)
source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

log() { echo "[auth-eval] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Heartbeat: 5-min interval (mirror canonical remote scripts; auth eval is
# short so 60s would be noise, 5min is enough resolution for a 15-min run).
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] auth_eval_only gpu=$GPU" >> "$LOG_DIR/heartbeat.log"
    sleep 300
  done ) &
HB_PID=$!
trap "kill $HB_PID 2>/dev/null || true" EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend. The Lane A pose-TTO host
# (35691284, Texas) ran 3.4h of pose TTO successfully then crashed at
# upstream/evaluate.py with CUDA_ERROR_NO_DEVICE because NVDEC was missing
# on that host. The probe catches the bad-host case in 5 seconds. Reference:
# feedback_vastai_nvdec_host_variation memory entry.
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — refusing to spend GPU on a host that"
    log "       cannot run upstream/evaluate.py. Destroy this Vast.ai"
    log "       instance and pick a different host."
    exit 2
}

# Stage 1: pre-flight — verify the EXACT archive bytes are in place.
# This is a 694045-byte file with sha256 a9921cd3... — anything else means
# rsync went to the wrong path or got truncated.
log "=== Stage 1: artifact preflight ==="
if [ ! -f "$ARCHIVE_INPUT" ]; then
    log "FATAL: archive not found at $ARCHIVE_INPUT"
    log "       rsync experiments/results/lane_a_landed/archive_lane_a.zip"
    log "       to $ARCHIVE_INPUT before invoking this script."
    exit 1
fi
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE_INPUT")
ARCHIVE_SHA=$("$PYBIN" -c "
import hashlib
with open('$ARCHIVE_INPUT', 'rb') as f:
    print(hashlib.sha256(f.read()).hexdigest())
")
log "  archive: $ARCHIVE_INPUT"
log "  bytes:   $ARCHIVE_BYTES (expected 694045)"
log "  sha256:  $ARCHIVE_SHA"
if [ "$ARCHIVE_BYTES" != "694045" ]; then
    log "FATAL: archive bytes mismatch — refusing to ship a score for the"
    log "       wrong archive. Expected 694045, got $ARCHIVE_BYTES."
    exit 1
fi
EXPECTED_SHA="a9921cd3b974ff0a7c37b39e7af22d9b75802f1219fc46aecb6eb8eaa7a08e84"
if [ "$ARCHIVE_SHA" != "$EXPECTED_SHA" ]; then
    log "FATAL: archive sha256 mismatch."
    log "       Expected: $EXPECTED_SHA"
    log "       Got:      $ARCHIVE_SHA"
    exit 1
fi
log "  archive bytes + sha verified — same EXACT artifact as Lane A pose TTO output."

# Verify upstream + inflate.sh present on this host.
for f in submissions/robust_current/inflate.sh \
         upstream/evaluate.py \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { log "FATAL: missing $f"; exit 1; }
done
log "  inflate.sh + upstream evaluate + scorers + GT video all present."

# Stage 2: contest_auth_eval.py on the pinned archive bytes.
# Same flags as remote_lane_a_pose_tto.sh:90-98 (Stage 4 there). The eval
# tool computes its own provenance, so we don't duplicate it here.
log "=== Stage 2: contest_auth_eval.py on Lane A archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE_INPUT" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log"

# Stage 3: RESULT_JSON capture + completion marker.
# contest_auth_eval.py prints "RESULT_JSON: {...}" to stdout on success.
log "=== Stage 3: RESULT_JSON capture ==="
if grep -q '^RESULT_JSON:' "$LOG_DIR/auth_eval.log"; then
    grep '^RESULT_JSON:' "$LOG_DIR/auth_eval.log" | tee "$LOG_DIR/result_json.txt"
    log "  RESULT_JSON captured to $LOG_DIR/result_json.txt"
else
    log "FATAL: no RESULT_JSON line in auth_eval.log — eval did not complete cleanly."
    exit 3
fi

# Also copy the full provenance.json + contest_auth_eval.json from the
# work dir into the LOG_DIR so a single rsync brings everything home.
if [ -f "$LOG_DIR/eval_work/contest_auth_eval.json" ]; then
    cp "$LOG_DIR/eval_work/contest_auth_eval.json" "$LOG_DIR/contest_auth_eval.json"
    log "  contest_auth_eval.json copied to $LOG_DIR/contest_auth_eval.json"
fi
if [ -f "$LOG_DIR/eval_work/provenance.json" ]; then
    cp "$LOG_DIR/eval_work/provenance.json" "$LOG_DIR/provenance.json"
    log "  provenance.json copied to $LOG_DIR/provenance.json"
fi

log "=== AUTH_EVAL_DONE ==="
echo "=== AUTH_EVAL_DONE ==="
