#!/bin/bash
# Lane OS-A: openpilot supercombo seeded pose TTO.
#
# Mirrors scripts/remote_lane_a_pose_tto.sh exactly with one delta:
#   Stage 1.5 runs experiments/seed_poses_from_openpilot.py to derive a
#   compress-time-only warm-start pose tensor from openpilot's supercombo
#   (~30 MB ONNX, NOT bundled in archive). Stage 2 then passes that seed
#   into optimize_poses.py via --seed-poses-path instead of --gt-poses-path.
#
# Predicted band: [1.05, 1.15] [contest-CUDA] anchored on Lane A (1.15).
# Reasoning: a physically-grounded warm-start from openpilot's lane/path
# model should converge to the same or better local minimum than the
# baseline-pose warm-start, with potential for additional gain because
# supercombo's pose head is independent of PoseNet's learned embedding
# (different inductive bias).
#
# Strict-scorer-rule (CLAUDE.md non-negotiable): supercombo is loaded ONLY
# at compress time. The archive contains exactly the same files as Lane A
# (renderer.bin + masks.mkv + optimized_poses.pt). Inflate path is
# unchanged — supercombo never runs at inflate.
#
# References:
# - memory project_openpilot_seeding_demo (the design intent)
# - memory project_openpilot_lane_forcing (lane forcing follow-up = Lane OS-B)
# - memory feedback_canonical_remote_bootstraps (provenance + heartbeat layout)
# - memory feedback_zip_dep_bootstrap_trap (set -euo pipefail + python zipfile)
# - memory feedback_vastai_nvdec_host_variation (Stage 0 NVDEC probe)
# - memory feedback_dead_flag_wiring_pattern (every CLI flag verified vs argparse)

set -euo pipefail
WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_os_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-os] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md canonical pipeline standard +
# memory feedback_canonical_remote_bootstraps).
PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
SUPERCOMBO_PATH=${SUPERCOMBO_PATH:-/workspace/openpilot/models/supercombo.onnx}
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
    'lane_script': 'scripts/remote_lane_os_supercombo_seed_tto.sh',
    'output_dir': '$LOG_DIR',
    'anchor_score_baseline': 1.15,
    'anchor_lane': 'lane_a',
    'predicted_band': [1.05, 1.15],
    'delta_from_lane_a': 'supercombo_seeded_warm_start_replaces_baseline_poses',
    'supercombo_path': '$SUPERCOMBO_PATH',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=OS gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend (memory:
# feedback_vastai_nvdec_host_variation).
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — refusing to spend GPU on a host that"
    log "       cannot run upstream/evaluate.py at the end. Destroy this"
    log "       Vast.ai instance and pick a different host."
    exit 2
}

# Pre-flight: required artifacts present
for f in submissions/baseline_dilated_h64_0_90/renderer.bin \
         submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 1: rebuild full-res masks (same as 2.29 / Lane A baseline) ==="
"$PYBIN" experiments/build_baseline_archive.py \
    --device cuda --crf 50 \
    --output "$LOG_DIR/archive_baseline_seed.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -5

# Extract just masks for optimize_poses + supercombo seed.
mkdir -p "$LOG_DIR/extracted"
cd "$LOG_DIR/extracted" && "$PYBIN" -c "
import zipfile, sys
src = '$LOG_DIR/archive_baseline_seed.zip'
with zipfile.ZipFile(src) as z:
    z.extractall()
print('extracted:', sorted([n.filename for n in z.infolist()]))
" 2>&1 | tail -3
cd "$WORKSPACE"

log "=== Stage 1.5: derive seed poses from openpilot supercombo ==="
log "   --supercombo-path=$SUPERCOMBO_PATH"
log "   --baseline-poses=submissions/baseline_dilated_h64_0_90/optimized_poses.pt"
log "   (compress-time only; supercombo NOT in archive)"
SEED_POSES="$LOG_DIR/seed_poses.pt"
# --allow-fallback ensures: if supercombo can't be loaded (e.g. driver
# mismatch on this Vast.ai host) we degrade gracefully to the masks-only
# analytical pose path (lane_mark_pose). Better to ship Lane M-equivalent
# poses than to fail the whole run.
"$PYBIN" -u experiments/seed_poses_from_openpilot.py \
    --supercombo-path "$SUPERCOMBO_PATH" \
    --video upstream/videos/0.mkv \
    --output "$SEED_POSES" \
    --baseline-poses submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
    --device cuda \
    --n-frames 1200 \
    --masks "$LOG_DIR/extracted/masks.mkv" \
    --allow-fallback 2>&1 | tee "$LOG_DIR/seed.log" | tail -15

[ -f "$SEED_POSES" ] || { echo "FATAL: seed_poses.pt not produced" >&2; exit 2; }
log "  produced $SEED_POSES ($(stat -c '%s' "$SEED_POSES") bytes)"

log "=== Stage 2: pose TTO with WARM-START from openpilot seed ==="
log "   --seed-poses-path=$SEED_POSES"
log "   --eval-roundtrip + --posetto-noise-std=0.5 (Fridrich C1 fixes)"
# Determinism: pin seeds + cublas + python hash (CLAUDE.md non-negotiable).
export PYTHONHASHSEED=1234
"$PYBIN" -u experiments/optimize_poses.py \
    --checkpoint submissions/baseline_dilated_h64_0_90/renderer.bin \
    --masks "$LOG_DIR/extracted/masks.mkv" \
    --seed-poses-path "$SEED_POSES" \
    --device cuda \
    --steps 500 \
    --batch-pairs 8 \
    --eval-roundtrip \
    --posetto-noise-std 0.5 \
    --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/optimize_poses.log" | tail -30

# Validate: did optimize_poses produce the file?
[ -f "$LOG_DIR/optimized_poses.bin" ] || { echo "FATAL: optimize_poses didn't produce optimized_poses.bin"; exit 2; }
log "  produced optimized_poses.bin ($(stat -c '%s' "$LOG_DIR/optimized_poses.bin") bytes)"

log "=== Stage 3: build NEW archive (renderer + masks + NEW poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp submissions/baseline_dilated_h64_0_90/renderer.bin "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
# Use the .pt format (matches the canonical archive structure)
[ -f "$LOG_DIR/optimized_poses.pt" ] && cp "$LOG_DIR/optimized_poses.pt" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_os.zip"
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

log "=== Stage 4: contest_auth_eval on Lane OS archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -15

log "=== LANE_OS_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
