#!/bin/bash
# Lane PS-V2: LEARNABLE per-class SegNet weights via Lagrangian on
# per-class distortion equalisation.
#
# Council 2026-04-27: Lane PS-V1 hard-coded "1,5,5,1,1" (boost classes 1
# = lane mark and 2 = vehicle by 5×). The ratio is heuristic — the
# scoring formula is per-pixel argmax disagreement averaged across
# classes, so the optimal weighting depends on the per-class error
# distribution which shifts during training.
#
# Lane PS-V2 makes the per-class weights LEARNABLE. The parameterisation
# is softmax (sum=1, all positive) over a 5-vector of raw logits. The
# Lagrangian objective drives per-class distortion variance toward zero
# — the optimiser spends its budget on the bottleneck classes.
#
# Math:
#   weights_c = softmax(raw_c)
#   loss = sum_c weights_c * loss_c + λ_var * Var(weights_c * distortion_c)
#
# At the optimum every class contributes equally — Pareto-optimal under
# the score formula.
#
# Predicted band: [1.02, 1.18] [contest-CUDA] vs Lane PS-V1's [1.05, 1.20]
# (council: tighter floor because the variance-equalisation penalty is a
# smoother gradient signal than the hard 1,5,5,1,1 jump; same ceiling
# because the underlying mechanism — per-class re-weighting of the KL
# distill auxiliary — is the same).
#
# Pipeline (mirrors Lane PS-V1 with one V2 substitution at Stage 2):
#   Stage 0 — NVDEC probe.
#   Stage 1 — stage Lane A masks (no rebuild).
#   Stage 2 — pose TTO with --learnable-segnet-class-weights (Lane PS-V2).
#   Stage 3 — Build archive (Lane A renderer + Lane A masks + V2 poses).
#   Stage 4 — contest_auth_eval [contest-CUDA].
#
# Anchored artifacts (Lane A's 1.15 frontier):
#   * renderer.bin: experiments/results/lane_a_landed/iter_0/renderer.bin
#   * masks.mkv:    experiments/results/lane_a_landed/extracted/masks.mkv
#   * poses.pt:     experiments/results/lane_a_landed/optimized_poses.pt (warm-start)
#
# Council preflight (CLAUDE.md non-negotiable: NEVER invent CLI flags).
# Verified 2026-04-27 against argparse:
#   * optimize_poses.py: --checkpoint --masks --gt-poses-path --device
#     --steps --batch-pairs --eval-roundtrip --posetto-noise-std
#     --kl-distill-weight --kl-distill-temperature --segnet-class-weights
#     --output-dir --learnable-segnet-class-weights
#     --learnable-segnet-class-weights-lr
#     --learnable-segnet-class-weights-var-lambda
#   * contest_auth_eval.py: --archive --inflate-sh --upstream-dir --device
#     --keep-work-dir --work-dir
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234

LOG_DIR="$WORKSPACE/lane_ps_v2_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-ps-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_ps_v2_learnable_class_weights.sh',
    'lane_name': 'lane_ps_v2_learnable_segnet_class_weights',
    'anchor_renderer': 'experiments/results/lane_a_landed/iter_0/renderer.bin',
    'anchor_poses': 'experiments/results/lane_a_landed/optimized_poses.pt',
    'anchor_masks': 'experiments/results/lane_a_landed/extracted/masks.mkv',
    'anchor_score_baseline': 1.15,
    'predicted_band': [1.02, 1.18],
    'rationale': 'Learnable softmax-parameterised per-class weights with Lagrangian variance equalisation; replaces Lane PS-V1 hard-coded 1,5,5,1,1 CSV.',
    'delta_from_lane_a': 'learnable_segnet_class_weights_v2',
    'segnet_class_weights_warm_start': '1,5,5,1,1',
    'learnable_segnet_class_weights': True,
    'learnable_segnet_class_weights_lr': 1e-2,
    'learnable_segnet_class_weights_var_lambda': 1.0,
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
    echo "[$(date -u +%FT%TZ)] lane=PS-V2 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend.
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
log "  anchor_poses:    $ANCHOR_POSES (warm-start for V2 TTO)"
log "  anchor_masks:    $ANCHOR_MASKS"

log "=== Stage 1: stage Lane A masks (no rebuild — anchor on Lane A's exact bytes) ==="
mkdir -p "$LOG_DIR/extracted"
cp "$ANCHOR_MASKS" "$LOG_DIR/extracted/masks.mkv"
log "  staged $(stat -c '%s' "$LOG_DIR/extracted/masks.mkv") bytes of Lane A masks"

log "=== Stage 2: pose TTO with --learnable-segnet-class-weights (Lane PS-V2) ==="
log "   --gt-poses-path = $ANCHOR_POSES (warm-start)"
log "   --kl-distill-weight 1.0 (mandatory — Lane PS only fires on KL distill path)"
log "   --kl-distill-temperature 2.0 (Hinton 2015 / Quantizr default)"
log "   --segnet-class-weights '1,5,5,1,1' (warm-start for the LEARNABLE module)"
log "   --learnable-segnet-class-weights (V2: softmax + Lagrangian var equalisation)"
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
    --learnable-segnet-class-weights \
    --learnable-segnet-class-weights-lr 1e-2 \
    --learnable-segnet-class-weights-var-lambda 1.0 \
    --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/optimize_poses.log" | tail -30
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPESTATUS[0]}" >&2; exit "${PIPESTATUS[0]}"
    fi

# Validate: did optimize_poses produce the file?
[ -f "$LOG_DIR/optimized_poses.bin" ] || { echo "FATAL: optimize_poses didn't produce optimized_poses.bin"; exit 2; }
log "  produced optimized_poses.bin ($(stat -c '%s' "$LOG_DIR/optimized_poses.bin") bytes)"
[ -f "$LOG_DIR/optimized_poses.pt" ] || { echo "FATAL: optimize_poses didn't produce optimized_poses.pt"; exit 2; }

# Sanity check: the lane-ps-v2 banner must have appeared in the log,
# proving the LEARNABLE per-class weights tensor was actually parsed and
# active. A silent fall-back to V1 (uniform or static 1,5,5,1,1 only)
# would burn $0.20 of contest_auth_eval to discover.
grep -q "lane-ps-v2" "$LOG_DIR/optimize_poses.log" || {
    echo "FATAL: optimize_poses.py did NOT log the [lane-ps-v2] banner — "
    echo "       LEARNABLE per-class SegNet weights are NOT active. Aborting"
    echo "       before wasting GPU on a V1-equivalent run misnamed as Lane PS-V2." >&2
    exit 2
}
log "  verified [lane-ps-v2] banner present in optimize_poses.log"

log "=== Stage 3: build NEW archive (Lane A renderer + Lane A masks + Lane PS-V2 poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$ANCHOR_RENDERER" "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
cp "$LOG_DIR/optimized_poses.pt" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_ps_v2.zip"
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

log "=== Stage 4: contest_auth_eval on Lane PS-V2 archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -15
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPESTATUS[0]}" >&2; exit "${PIPESTATUS[0]}"
    fi

log "=== LANE_PS_V2_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
