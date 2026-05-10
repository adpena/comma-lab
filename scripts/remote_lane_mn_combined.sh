#!/bin/bash
# Lane M + Lane N (combined) — radial-zoom 1-DOF poses + Fridrich L∞ pose penalty.
#
# Lane M (--pose-mode radial-zoom): per memory project_posenet_rank1_discovery,
#   PoseNet's Jacobian is rank ≈ 1.008 with 99.8% variance in dim 0 — a scalar
#   radial zoom is the information-theoretic minimum. Predicted: -0.05 distortion
#   + ~free pose bytes (1 scalar vs 6).
#
# Lane N (--linf-pose-weight 1.0): per memory project_fridrich_inverse_steganalysis
#   Principle 3 ("spread small errors, don't concentrate large ones"). Soft L∞-ball
#   penalty on (pose - baseline) delta. Predicted: -0.02 variance reduction.
#
# Combined target: ~1.05-1.10 vs Lane A's 1.15. Both flags committed in 904eeb05;
# both default-off so the combined run is the only thing changing.
#
# Mirrors scripts/remote_lane_a_pose_tto.sh: Stage 0 NVDEC probe, Stage 1 mask
# rebuild on host CUDA, Stage 2 pose TTO, Stage 3 archive build, Stage 4
# contest_auth_eval. Three deltas vs Lane A: LOG_DIR, optimize_poses flags, label.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_mn_combined_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-mn] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'predicted_band': [1.05, 1.15],  # added 2026-04-27 per preflight check 31
    'lane_script': 'scripts/remote_lane_mn_combined.sh',
    'output_dir': '$LOG_DIR',
    'lane_m_pose_mode': 'radial-zoom',
    'lane_n_linf_pose_weight': 1.0,
    'lane_n_linf_pose_budget': 0.05,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=MN gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0 (2026-04-27): NVDEC probe BEFORE any GPU spend. The Texas instance
# (35691284) ran 3.4h of pose TTO successfully then crashed at upstream/
# evaluate.py with CUDA_ERROR_NO_DEVICE because NVDEC was missing on that
# host. The probe catches the bad-host case in 5 seconds. Reference:
# feedback_vastai_nvdec_host_variation memory entry. --ensure-dali is the
# strengthened mode (probe DALI MIXED video op, not just nvidia-smi).
log "=== Stage 0: NVDEC probe (with --ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed — refusing to spend GPU on a host that"
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
set +e
"$PYBIN" experiments/build_baseline_archive.py \
    --device cuda --crf 50 \
    --output "$LOG_DIR/archive_baseline_seed.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -5
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

# Extract just masks for optimize_poses input
mkdir -p "$LOG_DIR/extracted"
cd "$LOG_DIR/extracted" && unzip -o "$LOG_DIR/archive_baseline_seed.zip" 2>&1 | tail -3
cd "$WORKSPACE"

log "=== Stage 2: pose TTO with WARM-START + Lane M (radial-zoom) + Lane N (L∞ penalty) ==="
log "   --gt-poses-path = submissions/baseline_dilated_h64_0_90/optimized_poses.pt"
log "   --pose-mode radial-zoom (Lane M, 1-DOF)"
log "   --linf-pose-weight 1.0 --linf-pose-budget 0.05 (Lane N)"
log "   --eval-roundtrip + --posetto-noise-std=0.5 (Fridrich C1 fixes)"
# Determinism: pin seeds + cublas + python hash (CLAUDE.md non-negotiable).
export PYTHONHASHSEED=1234
set +e
"$PYBIN" -u experiments/optimize_poses.py \
    --checkpoint submissions/baseline_dilated_h64_0_90/renderer.bin \
    --masks "$LOG_DIR/extracted/masks.mkv" \
    --gt-poses-path submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
    --device cuda \
    --steps 500 \
    --batch-pairs 8 \
    --eval-roundtrip \
    --posetto-noise-std 0.5 \
    --pose-mode radial-zoom \
    --linf-pose-weight 1.0 \
    --linf-pose-budget 0.05 \
    --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/optimize_poses.log" | tail -30
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

# Validate: did optimize_poses produce the file?
[ -f "$LOG_DIR/optimized_poses.bin" ] || { echo "FATAL: optimize_poses didn't produce optimized_poses.bin"; exit 2; }
log "  produced optimized_poses.bin ($(stat -c '%s' "$LOG_DIR/optimized_poses.bin") bytes)"

log "=== Stage 3: build NEW archive (renderer + masks + NEW poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp submissions/baseline_dilated_h64_0_90/renderer.bin "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
# Use the .pt format (matches the canonical archive structure)
[ -f "$LOG_DIR/optimized_poses.pt" ] && cp "$LOG_DIR/optimized_poses.pt" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_mn.zip"
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

log "=== Stage 4: contest_auth_eval on Lane M+N combined archive ==="
rm -rf "$LOG_DIR/eval_work"
set +e
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -15
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

log "=== LANE_MN_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON [contest-CUDA] ==="
