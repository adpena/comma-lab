#!/bin/bash
# Lane G V2: Pose TTO warm-started from VERIFIED baseline poses, with the
# Quantizr-validated SegNet KL-distillation auxiliary loss stacked on top
# of the standard scorer loss. Predicted: -0.05 to -0.15 distortion vs
# Lane A's 1.15 baseline.
#
# V2 DELTA from V1: --kl-distill-weight 1.0 → 0.01. V1's 1.0 dominated the
# loss ~14000× over the scorer hinge — the renderer optimized for KL fidelity
# instead of scorer score, defeating the purpose. 0.01 keeps KL as an
# AUXILIARY signal in the same OOM as the hinge term. Plus codex round-6 fix
# #2 (commit a03ff214) closed the KL-roundtrip distribution-mismatch bug.
# Same warm-start poses, same 600 pairs × 500 steps, same eval_roundtrip + noise.
#
# Per CLAUDE.md non-negotiable (NEVER invent CLI flags): the flag names
# `--kl-distill-weight` / `--kl-distill-temperature` were verified by
# argparse-grep on experiments/optimize_poses.py L159-171 (commit aed52ead).
# Loss is `kl_distill_segnet_only` from tac.losses (per-step log will show
# `kl=<float>` when --kl-distill-weight > 0).
set -euo pipefail
WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_g_v2_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-g] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_g_kldistill_pose_tto.sh',
    'output_dir': '$LOG_DIR',
    'kl_distill_weight': 0.01,
    'kl_distill_temperature': 2.0,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=G gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend. --ensure-dali so a fresh
# container that hasn't run remote_setup_full.sh installs DALI rather
# than spuriously failing on a missing import. Per
# feedback_vastai_nvdec_host_variation: same 4090 image / driver, different
# NVDEC exposure between hosts. The probe catches the bad-host case in 5
# seconds and saves $3-4 of pose-TTO GPU dollars per catch.
log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed -- refusing to spend GPU on a host that"
    log "       cannot run upstream/evaluate.py at the end. Destroy this"
    log "       Vast.ai instance and pick a different host."
    exit 2
}

# Pre-flight: required artifacts present
for f in submissions/baseline_dilated_h64_0_90/renderer.bin \
         submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
         submissions/baseline_dilated_h64_0_90/optimized_poses.bin \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 1: rebuild full-res masks (same as 2.29 baseline) ==="
"$PYBIN" experiments/build_baseline_archive.py \
    --device cuda --crf 50 \
    --output "$LOG_DIR/archive_baseline_seed.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -5

# Extract just masks for optimize_poses input
mkdir -p "$LOG_DIR/extracted"
cd "$LOG_DIR/extracted" && unzip -o "$LOG_DIR/archive_baseline_seed.zip" 2>&1 | tail -3
cd "$WORKSPACE"

log "=== Stage 2: pose TTO (Lane A flow) + KL-distill SegNet auxiliary ==="
log "   --gt-poses-path = submissions/baseline_dilated_h64_0_90/optimized_poses.pt"
log "   --eval-roundtrip + --posetto-noise-std=0.5 (Fridrich C1 fixes)"
log "   --kl-distill-weight=0.01  --kl-distill-temperature=2.0  (LANE G V2 delta)"
# Determinism: pin seeds + cublas + python hash (CLAUDE.md non-negotiable).
export PYTHONHASHSEED=1234
# 2026-04-27 launch finding: KL distill auxiliary loss adds a SECOND SegNet
# forward+backward pass per step (kl_distill_segnet_only computes fs_logits
# WITH grad in addition to the standard scorer pass). On RTX 4090 24GB this
# OOMs at --batch-pairs 8 (Lane A's setting) — peak alloc 23GB. Halving to
# 4 fits in ~13GB. Total work (600 pairs × 500 steps = 300k pair-steps) is
# unchanged; wall time roughly doubles vs Lane A.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PYBIN" -u experiments/optimize_poses.py \
    --checkpoint submissions/baseline_dilated_h64_0_90/renderer.bin \
    --masks "$LOG_DIR/extracted/masks.mkv" \
    --gt-poses-path submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
    --device cuda \
    --steps 500 \
    --batch-pairs 4 \
    --eval-roundtrip \
    --posetto-noise-std 0.5 \
    --kl-distill-weight 0.01 \
    --kl-distill-temperature 2.0 \
    --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/optimize_poses.log" | tail -30

# Validate: did optimize_poses produce the file?
[ -f "$LOG_DIR/optimized_poses.bin" ] || { echo "FATAL: optimize_poses didn't produce optimized_poses.bin"; exit 2; }
log "  produced optimized_poses.bin ($(stat -c '%s' "$LOG_DIR/optimized_poses.bin") bytes)"

log "=== Stage 3: build NEW archive (renderer + masks + NEW poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp submissions/baseline_dilated_h64_0_90/renderer.bin "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
[ -f "$LOG_DIR/optimized_poses.pt" ] && cp "$LOG_DIR/optimized_poses.pt" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_g_v2.zip"
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

log "=== Stage 4: contest_auth_eval on Lane G archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -15

log "=== LANE_G_DONE -- see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
