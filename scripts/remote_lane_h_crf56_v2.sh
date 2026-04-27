#!/bin/bash
# Lane H CRF56 v2: rebuild masks at CRF=56 ON CUDA HOST (not local CPU),
# reuse Lane A's optimized poses (the baseline poses that scored 2.29),
# reuse the baseline renderer, then auth-eval the resulting archive.
#
# WHY: a previous Lane H run built CRF56 masks on local CPU and shipped them
# to the host. Score regressed 1.15 -> 3.20. Hypothesis: CPU vs CUDA AV1
# encoder produces a different byte distribution, breaking the renderer's
# motion module which is sensitive to mask boundary noise. This v2 builds
# masks on the host's CUDA so the renderer sees the same distribution it
# was trained on (deterministic per device + seed).
#
# Single variable: CRF only (56 instead of baseline 50). All other inputs
# match the verified 2.29 baseline. Pose TTO is INTENTIONALLY SKIPPED.
#
# Reference: feedback_dead_flag_wiring_pattern (--crf verified via grep
# add_argument experiments/build_baseline_archive.py: line 86, default=50).
set -euo pipefail
WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_h_crf56_v2_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-h-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Stage 0: NVDEC probe BEFORE any GPU spend (saves $3-4 per bad host).
log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed (exit $?). Refusing to spend GPU on a"
    log "       host that cannot run upstream/evaluate.py at the end."
    log "       Destroy this Vast.ai instance and pick a different host."
    exit 2
}

# Pre-flight: required artifacts present.
# Lane H reuses Lane A's optimized poses (scored 1.15) — those poses must
# have been pushed to the host in advance (rsync uploads them to
# /workspace/pact/lane_a_poses_for_h/optimized_poses.pt before this runs).
LANE_A_POSES="$WORKSPACE/lane_a_poses_for_h/optimized_poses.pt"
for f in submissions/baseline_dilated_h64_0_90/renderer.bin \
         "$LANE_A_POSES" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 1: rebuild masks at CRF=56 ON CUDA (single variable test) ==="
log "   Inputs: baseline renderer + LANE A poses + GT video"
log "   Encoder: CUDA AV1 via mask_codec (deterministic per device+seed)"
"$PYBIN" experiments/build_baseline_archive.py \
    --renderer submissions/baseline_dilated_h64_0_90/renderer.bin \
    --poses "$LANE_A_POSES" \
    --gt-video upstream/videos/0.mkv \
    --device cuda --crf 56 \
    --output "$LOG_DIR/archive_crf56.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -10

# Validate: archive built and reasonable size
[ -f "$LOG_DIR/archive_crf56.zip" ] || { echo "FATAL: build_baseline_archive didn't produce archive_crf56.zip"; exit 2; }
ARCH_BYTES=$(stat -c '%s' "$LOG_DIR/archive_crf56.zip")
log "  archive_crf56.zip = ${ARCH_BYTES} bytes"
if [ "$ARCH_BYTES" -lt 100000 ] || [ "$ARCH_BYTES" -gt 1000000 ]; then
    log "FATAL: archive size $ARCH_BYTES bytes outside sanity range [100KB, 1MB]"
    exit 2
fi

# Stage 2: SKIP pose TTO entirely. Lane H tests masks-only effect.
log "=== Stage 2: SKIPPED (Lane H reuses baseline poses, no TTO) ==="

# Stage 3: archive_crf56.zip is already the final archive
# (build_baseline_archive includes renderer + masks + poses inline).
# No re-bundle needed; just symlink for clarity in Stage 4.
ARCHIVE="$LOG_DIR/archive_crf56.zip"
log "=== Stage 3: archive ready at $ARCHIVE (${ARCH_BYTES} bytes) ==="

log "=== Stage 4: contest_auth_eval on Lane H CRF56 archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20

log "=== LANE_H_V2_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
