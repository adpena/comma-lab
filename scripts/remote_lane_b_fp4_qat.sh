#!/bin/bash
# Lane F-V2 (formerly Lane B retry): FP4 QAT of Lane A's renderer.
# Anchored on Lane A 1.15 [contest-CUDA] (NOT baseline 2.29) per
# findings.md "Lane F regression — bugged or dead?" (2026-04-27).
# Three council-mandated fixes vs the original Lane B:
#   1. Threads --poses Lane A's optimized_poses.pt (Bug 1: silent zero-pose).
#   2. --fp4-epochs 500 (Bug 2: was 50 = 5% of canonical recipe).
#   3. Anchor on Lane A's renderer + poses (Bug 3: wrong baseline).
# Predicted: [1.05, 1.30] [contest-CUDA] — could beat Lane A 1.15.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234

LOG_DIR="$WORKSPACE/lane_f_v2_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-f-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_b_fp4_qat.sh',
    'lane_name': 'lane_f_v2_fp4_qat_on_lane_a',
    'anchor_renderer': 'experiments/results/lane_a_landed/iter_0/renderer.bin',
    'anchor_poses': 'experiments/results/lane_a_landed/optimized_poses.pt',
    'anchor_score_baseline': 1.15,
    'predicted_band': [1.05, 1.30],
    'output_dir': '$LOG_DIR',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=F-V2 gpu=$GPU" >> "$HEARTBEAT"
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

# Pre-flight: anchor on Lane A (1.15 [contest-CUDA]), NOT baseline 2.29.
# Lane A artifacts are committed to the repo at experiments/results/lane_a_landed/.
ANCHOR_RENDERER="experiments/results/lane_a_landed/iter_0/renderer.bin"
ANCHOR_POSES="experiments/results/lane_a_landed/optimized_poses.pt"
ANCHOR_MASKS="experiments/results/lane_a_landed/extracted/masks.mkv"
for f in "$ANCHOR_RENDERER" \
         "$ANCHOR_POSES" \
         "$ANCHOR_MASKS" \
         upstream/videos/0.mkv \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
log "  anchor_renderer: $ANCHOR_RENDERER"
log "  anchor_poses:    $ANCHOR_POSES"
log "  anchor_masks:    $ANCHOR_MASKS"

log "=== Stage 1: stage Lane A masks (no rebuild — we anchor on Lane A's exact bytes) ==="
mkdir -p "$LOG_DIR/extracted"
cp "$ANCHOR_MASKS" "$LOG_DIR/extracted/masks.mkv"
log "  staged $(stat -c '%s' "$LOG_DIR/extracted/masks.mkv") bytes of Lane A masks"

log "=== Stage 2: FP4 QAT fine-tune on Lane A renderer ==="
log "  input: $ANCHOR_RENDERER (FP32 ASYM, 290KB)"
log "  poses: $ANCHOR_POSES (Lane A optimized_poses.pt)"
log "  output: $LOG_DIR/qat/renderer_fp4.bin"
log "  fp4_epochs: 500 (Bug 2 fix: was 50 in original Lane B)"
"$PYBIN" -u experiments/qat_finetune.py \
    --checkpoint "$ANCHOR_RENDERER" \
    --poses "$ANCHOR_POSES" \
    --upstream upstream \
    --output-dir "$LOG_DIR/qat" \
    --device cuda \
    --base-ch 36 --mid-ch 60 --pose-dim 6 --motion-hidden 32 --depth 1 --embed-dim 6 \
    --use-zoom-flow --padding-mode zeros \
    --skip-int8-warmup \
    --fp4-epochs 500 \
    --lr 5e-5 \
    --batch-size 4 2>&1 | tee "$LOG_DIR/qat.log" | tail -30
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPESTATUS[0]}" >&2; exit "${PIPESTATUS[0]}"
    fi

# qat_finetune saves renderer_fp4.bin in output-dir per its convention
FP4_BIN=$(find "$LOG_DIR/qat" -name "renderer_fp4.bin" -o -name "*_fp4*.bin" 2>/dev/null | head -1)
[ -n "$FP4_BIN" ] && [ -f "$FP4_BIN" ] || { echo "FATAL: qat_finetune didn't produce FP4 binary"; ls -la "$LOG_DIR/qat/"; exit 2; }
FP4_SIZE=$(stat -c '%s' "$FP4_BIN")
log "  FP4 binary: $FP4_BIN ($FP4_SIZE bytes vs FP32 290KB original)"

log "=== Stage 3: build NEW archive (FP4 renderer + Lane A masks + Lane A poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$FP4_BIN" "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
cp "$ANCHOR_POSES" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_f_v2.zip"
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

log "=== Stage 4: contest_auth_eval on Lane F-V2 archive ==="
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

log "=== LANE_F_V2_DONE ==="
