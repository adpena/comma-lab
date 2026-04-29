#!/bin/bash
# Lane PS — per-class SegNet weighting on auxiliary KL distill loss.
# Anchored on Lane A (1.15 [contest-CUDA]). The hypothesis is that
# Lane A's averaged SegNet distortion (0.0046) hides per-class
# imbalance: cheap classes (road, sky) are at floor while costly
# classes (lane mark, vehicle) carry most of the residual error.
# Per-class weights `1,5,5,1,1` boost classes 1 (lane) + 2 (vehicle)
# in the auxiliary KL distill loss only — pose TTO's primary SegNet
# hinge loss is left untouched so the per-pixel argmax-flip signal
# that drives pose convergence is unaffected. Per memory
# `project_research_survey_20260420` Lane PS is a research-grade
# technique that has never been implemented in this codebase.
#
# Predicted band: [1.05, 1.20] [contest-CUDA]. Could BEAT Lane A 1.15
# if the per-class hypothesis holds.
#
# Anchor difference vs Lane A:
#   - renderer:  experiments/results/lane_a_landed/iter_0/renderer.bin
#                (Lane A's renderer, NOT baseline_dilated_h64_0_90)
#   - poses:     experiments/results/lane_a_landed/optimized_poses.pt
#                (warm-start from Lane A's pose TTO output, then
#                 re-TTO with --segnet-class-weights enabled)
#   - masks:     experiments/results/lane_a_landed/extracted/masks.mkv
#   - delta:     --segnet-class-weights "1,5,5,1,1"
#                --kl-distill-weight 1.0
#                (the per-class weighting only takes effect on the
#                 auxiliary KL distill path, so kl_distill_weight > 0
#                 is mandatory for the technique to fire)
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234

LOG_DIR="$WORKSPACE/lane_ps_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-ps] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md canonical pipeline standard +
# memory feedback_canonical_remote_bootstraps): every remote run must
# emit provenance.json so a fresh agent can reconstruct the experiment.
PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
"$PYBIN" -c "
import json, time, torch
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'lane_script': 'scripts/remote_lane_ps_per_class_segnet.sh',
    'lane_name': 'lane_ps_per_class_segnet_weighting',
    'anchor_renderer': 'experiments/results/lane_a_landed/iter_0/renderer.bin',
    'anchor_poses': 'experiments/results/lane_a_landed/optimized_poses.pt',
    'anchor_masks': 'experiments/results/lane_a_landed/extracted/masks.mkv',
    'anchor_score_baseline': 1.15,
    'predicted_band': [1.05, 1.20],
    'delta_from_lane_a': 'segnet_class_weights_1_5_5_1_1_on_kl_distill',
    'segnet_class_weights': '1,5,5,1,1',
    'kl_distill_weight': 1.0,
    'kl_distill_temperature': 2.0,
    'output_dir': '$LOG_DIR',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=PS gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0 (2026-04-27): NVDEC probe BEFORE any GPU spend. Catches bad-host
# case in 5 seconds. Reference: feedback_vastai_nvdec_host_variation.
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

# Pre-flight: anchor on Lane A (1.15 [contest-CUDA]).
ANCHOR_RENDERER="experiments/results/lane_a_landed/iter_0/renderer.bin"
ANCHOR_POSES="experiments/results/lane_a_landed/optimized_poses.pt"
ANCHOR_MASKS="experiments/results/lane_a_landed/extracted/masks.mkv"
for f in "$ANCHOR_RENDERER" \
         "$ANCHOR_POSES" \
         "$ANCHOR_MASKS" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
log "  anchor_renderer: $ANCHOR_RENDERER"
log "  anchor_poses:    $ANCHOR_POSES (warm-start for per-class TTO)"
log "  anchor_masks:    $ANCHOR_MASKS"

log "=== Stage 1: stage Lane A masks (no rebuild — anchor on Lane A's exact bytes) ==="
mkdir -p "$LOG_DIR/extracted"
cp "$ANCHOR_MASKS" "$LOG_DIR/extracted/masks.mkv"
log "  staged $(stat -c '%s' "$LOG_DIR/extracted/masks.mkv") bytes of Lane A masks"

log "=== Stage 2: pose TTO with --segnet-class-weights '1,5,5,1,1' (Lane PS) ==="
log "   --gt-poses-path = $ANCHOR_POSES (warm-start)"
log "   --kl-distill-weight 1.0 (mandatory — Lane PS only fires on KL distill path)"
log "   --kl-distill-temperature 2.0 (Hinton 2015 / Quantizr default)"
log "   --segnet-class-weights '1,5,5,1,1' (boost lane + vehicle classes)"
log "   --eval-roundtrip + --posetto-noise-std=0.5 (Fridrich C1 fixes)"
"$PYBIN" -u experiments/optimize_poses.py \
    --checkpoint "$ANCHOR_RENDERER" \
    --masks "$LOG_DIR/extracted/masks.mkv" \
    --gt-poses-path "$ANCHOR_POSES" \
    --device cuda \
    --steps 500 \
    --batch-pairs 8 \
    --eval-roundtrip \
    --posetto-noise-std 0.5 \
    --kl-distill-weight 1.0 \
    --kl-distill-temperature 2.0 \
    --segnet-class-weights "1,5,5,1,1" \
    --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/optimize_poses.log" | tail -30
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

# Validate: did optimize_poses produce the file?
[ -f "$LOG_DIR/optimized_poses.bin" ] || { echo "FATAL: optimize_poses didn't produce optimized_poses.bin"; exit 2; }
log "  produced optimized_poses.bin ($(stat -c '%s' "$LOG_DIR/optimized_poses.bin") bytes)"
[ -f "$LOG_DIR/optimized_poses.pt" ] || { echo "FATAL: optimize_poses didn't produce optimized_poses.pt"; exit 2; }

# Sanity check: the lane-ps banner must have appeared in the log,
# proving the per-class weights tensor was actually parsed and active.
# A silent fall-back to uniform weighting (e.g., due to an empty CSV
# parsed as None) would burn $0.20 of contest_auth_eval to discover.
grep -q "lane-ps" "$LOG_DIR/optimize_poses.log" || {
    echo "FATAL: optimize_poses.py did NOT log the [lane-ps] banner — "
    echo "       per-class SegNet weights are NOT active. Aborting before"
    echo "       wasting GPU on a uniform-weighting run misnamed as Lane PS." >&2
    exit 2
}
log "  verified [lane-ps] banner present in optimize_poses.log"

log "=== Stage 3: build NEW archive (Lane A renderer + Lane A masks + Lane PS poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$ANCHOR_RENDERER" "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
cp "$LOG_DIR/optimized_poses.pt" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_ps.zip"
"$PYBIN" -c "
import zipfile, os
src = '$LOG_DIR/iter_0'
dst = '$ARCHIVE'
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in ('renderer.bin', 'masks.mkv', 'optimized_poses.pt'):
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing {p}'
        z.write(p, arcname=n)
print(f'archive {dst}: {os.path.getsize(dst)} bytes')
"

log "=== Stage 4: contest_auth_eval on Lane PS archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -15
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

log "=== LANE_PS_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
