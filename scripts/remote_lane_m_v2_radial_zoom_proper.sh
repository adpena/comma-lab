#!/bin/bash
# Lane M-V2 (radial-zoom pose engineering audit retry — proper 6-DOF padding).
# Anchored on Lane A (1.15 [contest-CUDA]), warm-started from Lane A's
# optimized_poses.pt. The previous Lane M-V1 scored 2.35 because the
# saved (N, 1) tensor was ZERO-padded back to (N, 6) at the inflate
# side — but PoseNet's auxiliary 5 dims encode the rank-1 information
# the renderer was trained on, and zero-padding them destroys the
# per-pair signal (V1 PoseNet ~16x worse than baseline).
#
# V2 fix (committed in experiments/optimize_poses.py): when
# --pose-mode radial-zoom is set, the save-block composes a (N, 6)
# tensor where dim 0 is the optimized scalar and dims 1-5 are the
# FROZEN baseline values (from --gt-poses-path). The saved file is
# now consumable by inflate without any pose_mode-aware adapter — it
# IS the canonical 6-DOF pose tensor.
#
# Predicted band: [1.10, 1.30] [contest-CUDA] — could BEAT Lane A 1.15
# if the rank-1 hypothesis (project_posenet_rank1_discovery: 99.8%
# variance in dim 0) holds when properly engineered. The headroom is
# narrow but real: by freezing dims 1-5 at baseline values and only
# letting dim 0 (the radial-zoom from Focus-of-Expansion) move, the
# optimizer cannot drift the auxiliary signal that PoseNet detects.
#
# Anchor difference vs Lane A:
#   - renderer: experiments/results/lane_a_landed/iter_0/renderer.bin
#               (Lane A's renderer, NOT baseline_dilated_h64_0_90)
#   - poses:    experiments/results/lane_a_landed/optimized_poses.pt
#               (warm-start from Lane A's pose TTO output, then re-TTO
#                with the radial-zoom-only optimizable)
#   - masks:    experiments/results/lane_a_landed/extracted/masks.mkv
#               (rebuilt to extracted/ via baseline pipeline if missing)
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234

LOG_DIR="$WORKSPACE/lane_m_v2_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-m-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_m_v2_radial_zoom_proper.sh',
    'lane_name': 'lane_m_v2_radial_zoom_proper_padding',
    'anchor_renderer': 'experiments/results/lane_a_landed/iter_0/renderer.bin',
    'anchor_poses': 'experiments/results/lane_a_landed/optimized_poses.pt',
    'anchor_masks': 'experiments/results/lane_a_landed/extracted/masks.mkv',
    'anchor_score_baseline': 1.15,
    'predicted_band': [1.10, 1.30],
    'delta_from_v1': 'save_n6_with_frozen_baseline_dims15_not_zero_pad',
    'pose_mode': 'radial-zoom',
    'output_dir': '$LOG_DIR',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=M-V2 gpu=$GPU" >> "$HEARTBEAT"
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
log "  anchor_poses:    $ANCHOR_POSES (warm-start for radial-zoom TTO)"
log "  anchor_masks:    $ANCHOR_MASKS"

log "=== Stage 1: stage Lane A masks (no rebuild — anchor on Lane A's exact bytes) ==="
mkdir -p "$LOG_DIR/extracted"
cp "$ANCHOR_MASKS" "$LOG_DIR/extracted/masks.mkv"
log "  staged $(stat -c '%s' "$LOG_DIR/extracted/masks.mkv") bytes of Lane A masks"

log "=== Stage 2: pose TTO with --pose-mode radial-zoom (1-DOF optimizable) ==="
log "   --gt-poses-path = $ANCHOR_POSES (warm-start AND frozen-pad source for dims 1-5)"
log "   --pose-mode radial-zoom (V2: optimizable is (N, 1), saved as (N, 6) padded)"
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
    --pose-mode radial-zoom \
    --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/optimize_poses.log" | tail -30
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPESTATUS[0]}" >&2; exit "${PIPESTATUS[0]}"
    fi

# Validate: did optimize_poses produce the file?
[ -f "$LOG_DIR/optimized_poses.bin" ] || { echo "FATAL: optimize_poses didn't produce optimized_poses.bin"; exit 2; }
log "  produced optimized_poses.bin ($(stat -c '%s' "$LOG_DIR/optimized_poses.bin") bytes)"
[ -f "$LOG_DIR/optimized_poses.pt" ] || { echo "FATAL: optimize_poses didn't produce optimized_poses.pt"; exit 2; }

# V2 critical sanity check: the saved tensor MUST be (N, 6). If it is
# (N, 1) we've regressed to the V1 zero-padding bug. Fail loud here so
# the operator does not waste $0.20 on a contest_auth_eval that would
# crash with a shape mismatch downstream.
"$PYBIN" -c "
import torch, sys
p = torch.load('$LOG_DIR/optimized_poses.pt', map_location='cpu', weights_only=True)
shape = tuple(p.shape)
if shape[1] != 6:
    print(f'FATAL: Lane M-V2 expects (N, 6) saved poses, got {shape}', file=sys.stderr)
    print('       The V1 bug saved (N, 1) and zero-padded at inflate.', file=sys.stderr)
    print('       Verify experiments/optimize_poses.py save block was patched.', file=sys.stderr)
    sys.exit(2)
print(f'[lane-m-v2-sanity] saved optimized_poses.pt shape: {shape} OK')
"

log "=== Stage 3: build NEW archive (Lane A renderer + Lane A masks + V2 poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$ANCHOR_RENDERER" "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
cp "$LOG_DIR/optimized_poses.pt" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_m_v2.zip"
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

log "=== Stage 4: contest_auth_eval on Lane M-V2 archive ==="
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

log "=== LANE_M_V2_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
