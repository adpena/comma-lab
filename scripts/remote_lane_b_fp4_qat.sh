#!/bin/bash
# Lane B: FP4 QAT of the dilated-h64 renderer (~290KB → ~165KB).
# Predicted: 2.29 → 2.18 (rate -0.108). Single variable: renderer bytes only.
# Distortion expected unchanged (FP4 noise is imperceptible in inference).
set -euo pipefail
WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234

LOG_DIR="$WORKSPACE/lane_b_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-b] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'output_dir': '$LOG_DIR',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=B gpu=$GPU" >> "$HEARTBEAT"
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

# Pre-flight
for f in submissions/baseline_dilated_h64_0_90/renderer.bin \
         submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
         upstream/videos/0.mkv \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 1: rebuild full-res masks (so this lane is fully reproducible from-scratch) ==="
"$PYBIN" experiments/build_baseline_archive.py \
    --device cuda --crf 50 \
    --output "$LOG_DIR/archive_baseline_seed.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -5
mkdir -p "$LOG_DIR/extracted"
cd "$LOG_DIR/extracted" && unzip -o "$LOG_DIR/archive_baseline_seed.zip" 2>&1 | tail -3
cd "$WORKSPACE"

log "=== Stage 2: FP4 QAT fine-tune ==="
log "  input: submissions/baseline_dilated_h64_0_90/renderer.bin (FP32 ASYM)"
log "  output: $LOG_DIR/qat/renderer_fp4.bin"
"$PYBIN" -u experiments/qat_finetune.py \
    --checkpoint submissions/baseline_dilated_h64_0_90/renderer.bin \
    --upstream upstream \
    --output-dir "$LOG_DIR/qat" \
    --device cuda \
    --base-ch 36 --mid-ch 60 --pose-dim 6 --motion-hidden 32 --depth 1 --embed-dim 6 \
    --use-zoom-flow --padding-mode zeros \
    --skip-int8-warmup \
    --fp4-epochs 50 \
    --lr 5e-5 \
    --batch-size 4 2>&1 | tee "$LOG_DIR/qat.log" | tail -30

# qat_finetune saves renderer_fp4.bin in output-dir per its convention
FP4_BIN=$(find "$LOG_DIR/qat" -name "renderer_fp4.bin" -o -name "*_fp4*.bin" 2>/dev/null | head -1)
[ -n "$FP4_BIN" ] && [ -f "$FP4_BIN" ] || { echo "FATAL: qat_finetune didn't produce FP4 binary"; ls -la "$LOG_DIR/qat/"; exit 2; }
FP4_SIZE=$(stat -c '%s' "$FP4_BIN")
log "  FP4 binary: $FP4_BIN ($FP4_SIZE bytes vs FP32 290KB original)"

log "=== Stage 3: build NEW archive (FP4 renderer + same masks + same poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$FP4_BIN" "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
cp submissions/baseline_dilated_h64_0_90/optimized_poses.pt "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_b.zip"
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

log "=== Stage 4: contest_auth_eval on Lane B archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -15

log "=== LANE_B_DONE ==="
