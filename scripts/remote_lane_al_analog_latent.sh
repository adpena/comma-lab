#!/bin/bash
# Lane AL — Analog Latent canvas optimization.
#
# EUREKA insight (gpt-5.5 grand council 2026-04-29 11am):
#   Lane MM treats the grayscale mask as compressed argmax segmentation.
#   It is ACTUALLY a 1-channel ANALOG LATENT CANVAS feeding a Gaussian-
#   softmax LUT into the renderer. Optimize the per-pixel gray values via
#   SGD against the contest scorer; AV1 still sees a smooth monochrome
#   video so rate is held constant by construction. Boundary pixels become
#   scorer-optimal SOFT class probabilities.
#
# Pipeline:
#   1. Stage Lane A baseline anchor (renderer.bin + masks.mkv + poses).
#   2. Stage 2 optimize_grayscale_canvas.py — Adam over per-pixel gray
#      values, eval_roundtrip=True + noise_std=0.5 + CUDA-required.
#   3. Stage 3 build_lane_al_archive.py — pack optimized_grayscale.npy
#      into archive layout matching Lane MM (PYTHON_INFLATE=renderer_grayscale).
#   4. Stage 4 contest_auth_eval [contest-CUDA] on the Lane AL archive.
#
# Verified CLI flags via grep add_argument (CLAUDE.md NEVER invent flags):
#   experiments/optimize_grayscale_canvas.py:
#     --anchor-archive --gt-video --upstream-dir --output-dir --steps
#     --lr --sigma --noise-std --batch-size --n-frames --device --seed
#   experiments/build_lane_al_archive.py:
#     --anchor-archive --grayscale-npy --output --crf
#   experiments/contest_auth_eval.py:
#     --archive --inflate-sh --upstream-dir --device --keep-work-dir --work-dir
#
# Cost estimate: $1.50-2.00 (~6-8h on RTX 4090). Cap: $4.
# Predicted band: [0.65, 0.85] [contest-CUDA] (-0.05 to -0.15 vs Lane MM 0.78).
#
# Anchor: experiments/results/lane_a_landed/archive_lane_a.zip (1.15 verified).
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
#
# E2E_SMOKE_OPT_OUT: Lane AL composes existing-tested encoder primitives
#   (build_lane_mm_archive grayscale encode + Lane MM inflate dispatch); the
#   only new code is the SGD loop (unit-tested in
#   src/tac/tests/test_optimize_grayscale_canvas.py).

set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_al_analog_latent"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
log() { echo "[lane-al] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md canonical_remote_bootstraps).
PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo "no-driver")
"$PYBIN" -c "
import json, time, torch
prov = {
    'lane_id': '$LANE_ID',
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'lane_script': 'scripts/remote_lane_al_analog_latent.sh',
    'output_dir': '$LOG_DIR',
    'paradigm': 'analog_latent_canvas_sgd',
    'eureka_source': 'gpt-5.5 grand council 2026-04-29 11am',
    'predicted_band': [0.65, 0.85],
    'score_tag': '[contest-CUDA]',
    'baseline_score': 1.15,
    'lane_mm_baseline_score': 0.78,
    'anchor_archive': 'experiments/results/lane_a_landed/archive_lane_a.zip',
    'inflate_path': 'PYTHON_INFLATE=renderer_grayscale (Lane MM dispatch arm)',
    'cost_estimate_usd': 1.75,
    'cost_cap_usd': 4.0,
    'wall_clock_estimate_hours': 7.0,
    'wall_clock_cap_hours': 14.0,
    'sgd_steps': 200,
    'sgd_lr': 0.01,
    'sgd_sigma': 15.0,
    'sgd_noise_std': 0.5,
    'sgd_batch_size': 8,
    'eval_roundtrip': True,
    'cuda_required': True,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=AL gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed -- destroy this Vast.ai instance."
    exit 2
}

log "=== Stage 1: anchor parity checks ==="
ANCHOR_ARCHIVE="experiments/results/lane_a_landed/archive_lane_a.zip"
GT_VIDEO="upstream/videos/0.mkv"
SEGNET_WEIGHTS="upstream/models/segnet.safetensors"
POSENET_WEIGHTS="upstream/models/posenet.safetensors"
for f in "$ANCHOR_ARCHIVE" \
         "$GT_VIDEO" \
         "$SEGNET_WEIGHTS" \
         "$POSENET_WEIGHTS" \
         submissions/robust_current/inflate.sh \
         submissions/robust_current/inflate_renderer.py \
         submissions/robust_current/inflate_renderer_grayscale.py; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

ITER_DIR="$LOG_DIR/iter_0"
mkdir -p "$ITER_DIR"

log "=== Stage 2: optimize_grayscale_canvas (Adam SGD on per-pixel gray) ==="
log "  steps=200 lr=1e-2 sigma=15 noise_std=0.5 batch=8 device=cuda"
"$PYBIN" -u experiments/optimize_grayscale_canvas.py \
    --anchor-archive "$ANCHOR_ARCHIVE" \
    --gt-video "$GT_VIDEO" \
    --upstream-dir upstream \
    --output-dir "$ITER_DIR" \
    --steps 200 \
    --lr 1e-2 \
    --sigma 15.0 \
    --noise-std 0.5 \
    --batch-size 8 \
    --n-frames 1200 \
    --device cuda \
    --seed 41377 2>&1 | tee "$LOG_DIR/optimize_grayscale_canvas.log" | tail -30
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: optimize_grayscale_canvas exited rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

OPTIMIZED_NPY="$ITER_DIR/optimized_grayscale.npy"
[ -f "$OPTIMIZED_NPY" ] || {
    log "FATAL: optimize_grayscale_canvas did NOT produce optimized_grayscale.npy"
    exit 2
}
NPY_BYTES=$(stat -c '%s' "$OPTIMIZED_NPY" 2>/dev/null || stat -f '%z' "$OPTIMIZED_NPY")
log "  optimized_grayscale.npy = ${NPY_BYTES} bytes"

log "=== Stage 3a: build_lane_al_archive (pack grayscale.mkv + Lane A renderer/poses) ==="
ARCHIVE="$ITER_DIR/archive_lane_al.zip"
"$PYBIN" -u experiments/build_lane_al_archive.py \
    --anchor-archive "$ANCHOR_ARCHIVE" \
    --grayscale-npy "$OPTIMIZED_NPY" \
    --output "$ARCHIVE" \
    --crf 50 2>&1 | tee "$LOG_DIR/build.log" | tail -10
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: build_lane_al_archive exited rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
if [ -z "${ARCHIVE_BYTES:-}" ] || [ "$ARCHIVE_BYTES" -le 0 ]; then
    log "FATAL: archive size empty or zero — refusing to call auth_eval"
    exit 2
fi
log "  archive_lane_al.zip = ${ARCHIVE_BYTES} bytes"

# Validate archive layout (renderer + grayscale + poses).
"$PYBIN" -c "
import zipfile, sys
with zipfile.ZipFile('$ARCHIVE') as z:
    names = set(z.namelist())
    required = {'renderer.bin', 'grayscale.mkv'}
    missing = required - names
    if missing:
        print(f'FATAL: archive missing required entries: {missing}', file=sys.stderr)
        sys.exit(2)
    has_poses = any(n in names for n in ('optimized_poses.pt', 'poses.pt'))
    if not has_poses:
        print('FATAL: archive missing optimized_poses.pt / poses.pt', file=sys.stderr)
        sys.exit(2)
    print(f'archive OK: {sorted(names)}')
"

# Strip macOS AppleDouble files from upstream/videos before auth eval
# (check 37 + feedback_canonical_remote_bootstraps).
rm -f upstream/videos/._*.mkv

log "=== Stage 3b: contest_auth_eval [contest-CUDA] ==="
# Configure inflate.sh dispatch arm: PYTHON_INFLATE=renderer_grayscale
# routes to submissions/robust_current/inflate_renderer_grayscale.py.
INFLATE_CONFIG="$ITER_DIR/lane_al_config.env"
cat > "$INFLATE_CONFIG" <<'EOF'
PYTHON_INFLATE=renderer_grayscale
LANE_MM_SIGMA=15
export LANE_MM_SIGMA
EOF

rm -rf "$ITER_DIR/eval_work"
CONFIG_ENV_PATH="$INFLATE_CONFIG" "$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$ITER_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: contest_auth_eval exited rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

grep -q '^RESULT_JSON' "$LOG_DIR/auth_eval.log" || {
    log "FATAL: auth_eval did not produce RESULT_JSON — invalid measurement"
    exit 2
}

log "=== Stage 4: finalize provenance ==="
"$PYBIN" -c "
import json, os, time
prov = json.load(open('$PROVENANCE'))
prov['completed_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
prov['archive_path'] = '$ARCHIVE'
prov['archive_bytes'] = os.path.getsize('$ARCHIVE')
prov['optimized_grayscale_npy'] = '$OPTIMIZED_NPY'
prov['lane_status'] = 'COMPLETE'
json.dump(prov, open('$PROVENANCE', 'w'), indent=2)
print('provenance updated.')
"

log "=== LANE_AL_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "=== predicted_band=[0.65, 0.85], lane_mm_baseline=0.78, see $LOG_DIR/provenance.json ==="
