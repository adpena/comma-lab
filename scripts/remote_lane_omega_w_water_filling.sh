#!/bin/bash
# Lane Ω-W — Water-filling Lagrangian bit-budget allocator.
#
# POST-COMPRESS export-time technique. Loads the SC++ inference checkpoint
# from lane_a_landed/iter_0/segmap_inference.pt, estimates per-channel
# Hessian via 1-step gradient with eval_roundtrip, allocates per-channel
# qint_max via Shannon water-filling, packs three archives at budgets
# {360k, 480k, 600k} bits, and runs contest_auth_eval [contest-CUDA] on
# each. Picks the lowest-score archive.
#
# Design doc (BINDING): docs/paper/water_filling_design_20260429.md
# Predicted band [1.11, 1.13] [contest-CUDA].
# Anchor: experiments/results/lane_a_landed/iter_0/ (full-res 384x512 masks).
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
#
# E2E_SMOKE_OPT_OUT: lane lands with 16 unit tests in src/tac/tests/test_water_filling_codec.py covering codec round-trip + budget feasibility + paranoia gates; full-pipeline smoke proof backfilled before remote dispatch via lane launcher pre-flight.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_omega_w_water_filling"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
log() { echo "[lane-omega-w] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_omega_w_water_filling.sh',
    'output_dir': '$LOG_DIR',
    'paradigm': 'water_filling_lagrangian_bit_budget_post_compress_export',
    'arch': {'hidden': 24, 'block_hidden': 24, 'num_blocks': 8, 'max_frame_index': 1200},
    'budgets_bits': [360000, 480000, 600000],
    'predicted_band': [1.11, 1.13],
    'anchor_dir': 'experiments/results/lane_a_landed/iter_0',
    'design_doc': 'docs/paper/water_filling_design_20260429.md',
    'eval_roundtrip': True,
    'controlled_baseline': 'lane_a_landed (uniform qint_max=7) — single mechanism: water-filling Lagrangian per-channel qint_max in block-FP export, post-compress only, no retraining',
    'controlled_baseline_lane': 'lane_a_landed',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=OMEGA-W gpu=$GPU" >> "$HEARTBEAT"
    sleep 300
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed"
    exit 2
}

ANCHOR_DIR="experiments/results/lane_a_landed/iter_0"
ANCHOR_RENDERER="$ANCHOR_DIR/renderer.bin"
ANCHOR_MASKS="$ANCHOR_DIR/masks.mkv"
ANCHOR_POSES="experiments/results/lane_a_landed/optimized_poses.pt"
ANCHOR_INFERENCE="$ANCHOR_DIR/segmap_inference.pt"
log "=== Stage 1: anchor file checks (full-res 384x512 masks only) ==="
for f in "$ANCHOR_RENDERER" "$ANCHOR_MASKS" "$ANCHOR_POSES" "$ANCHOR_INFERENCE" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing anchor $f" >&2; exit 1; }
done

log "=== Stage 2: load SC++ inference checkpoint (no retraining) ==="
"$PYBIN" -c "
import torch
state = torch.load('$ANCHOR_INFERENCE', map_location='cpu', weights_only=False)
print(f'inference state_dict keys: {len(state)}')
n_params = sum(int(v.numel()) for v in state.values() if torch.is_tensor(v))
print(f'inference total params: {n_params}')
"

BUDGETS=(360000 480000 600000)
BEST_SCORE=999.99
BEST_ARCHIVE=""
BEST_BUDGET=0

for B in "${BUDGETS[@]}"; do
    log "=== Stage 3.${B}: water-filling export at total_bits=${B} ==="
    PAYLOAD="$LOG_DIR/segmap_weights_${B}.tar.xz"
    SUMMARY="$LOG_DIR/water_fill_summary_${B}.json"
    "$PYBIN" -u experiments/lane_omega_w_water_filling.py \
        --checkpoint "$ANCHOR_INFERENCE" \
        --anchor-masks "$ANCHOR_MASKS" \
        --gt-video upstream/videos/0.mkv \
        --total-bits "$B" \
        --output-payload "$PAYLOAD" \
        --device cuda \
        --num-calib 64 \
        --hidden 24 --block-hidden 24 --num-blocks 8 \
        --max-frame-index 1200 \
        --summary-json "$SUMMARY" \
        --verify-tol 1e-1 2>&1 | tee "$LOG_DIR/water_fill_${B}.log" | tail -30
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: water-filling export rc=${PIPE_RC[0]} at budget=$B" >&2
        exit "${PIPE_RC[0]}"
    fi

    log "=== Stage 4.${B}: build archive (grayscale.mkv + segmap_weights + poses) ==="
    ARCHIVE_SRC="$LOG_DIR/archive_src_${B}"
    mkdir -p "$ARCHIVE_SRC"
    GRAYSCALE_MKV="$ARCHIVE_SRC/grayscale.mkv"
    "$PYBIN" -c "
import subprocess, os, sys
import torch
from pathlib import Path
sys.path.insert(0, 'src')
from tac.mask_codec import decode_masks_auto
from tac.mask_grayscale_lut import encode_masks_grayscale
import numpy as np
mask_classes = decode_masks_auto(Path('$ANCHOR_MASKS'))
gray = encode_masks_grayscale(mask_classes)
arr = gray.numpy() if isinstance(gray, torch.Tensor) else np.asarray(gray)
out_path = Path('$GRAYSCALE_MKV')
n, h, w = arr.shape
cmd = [
    'ffmpeg', '-y',
    '-f', 'rawvideo', '-vcodec', 'rawvideo',
    '-s', f'{w}x{h}', '-pix_fmt', 'gray',
    '-r', '20', '-i', 'pipe:0',
    '-c:v', 'libsvtav1',
    '-crf', '50', '-preset', '6',
    '-svtav1-params', 'enable-restoration=0:enable-cdef=0',
    '-pix_fmt', 'gray', '-an',
    str(out_path),
]
proc = subprocess.run(cmd, input=arr.tobytes(), capture_output=True, timeout=300)
if proc.returncode != 0:
    raise RuntimeError(f'ffmpeg AV1 encode failed (rc={proc.returncode}): {proc.stderr.decode()[-500:]}')
print('grayscale.mkv bytes:', os.path.getsize(out_path))
"
    cp "$ANCHOR_POSES" "$ARCHIVE_SRC/optimized_poses.pt"
    cp "$PAYLOAD" "$ARCHIVE_SRC/segmap_weights.tar.xz"

    ARCHIVE="$LOG_DIR/archive_${LANE_ID}_${B}.zip"
    "$PYBIN" -c "
import os, zipfile
src = '$ARCHIVE_SRC'
with zipfile.ZipFile('$ARCHIVE', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in ('segmap_weights.tar.xz', 'grayscale.mkv', 'optimized_poses.pt'):
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing archive component {p}'
        z.write(p, arcname=n)
print('archive bytes:', os.path.getsize('$ARCHIVE'))
"

    ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
    [ "$ARCHIVE_BYTES" -gt 0 ] || { echo 'FATAL: archive empty'; exit 2; }
    log "budget=${B} archive_bytes=$ARCHIVE_BYTES"

    INFLATE_CONFIG="$LOG_DIR/lane_omega_w_${B}_config.env"
    cat > "$INFLATE_CONFIG" <<'EOF'
PYTHON_INFLATE=segmap
EOF

    log "=== Stage 5.${B}: contest_auth_eval [contest-CUDA] at budget=${B} ==="
    rm -rf "$LOG_DIR/eval_work_${B}"
    EVAL_LOG="$LOG_DIR/auth_eval_${B}.log"
    CONFIG_ENV_PATH="$INFLATE_CONFIG" "$PYBIN" -u experiments/contest_auth_eval.py \
        --archive "$ARCHIVE" \
        --inflate-sh submissions/robust_current/inflate.sh \
        --upstream-dir upstream \
        --device "${AUTH_EVAL_DEVICE:-cuda}" \
        --keep-work-dir \
        --work-dir "$LOG_DIR/eval_work_${B}" 2>&1 | tee "$EVAL_LOG" | tail -20
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "WARN: contest_auth_eval exited rc=${PIPE_RC[0]} at budget=$B (continuing)" >&2
        continue
    fi

    SCORE=$("$PYBIN" -c "
import re, sys
text = open('$EVAL_LOG').read()
m = re.search(r'\"score\"\s*:\s*([0-9.]+)', text)
print(m.group(1) if m else 'NA')
")
    log "budget=${B} score=$SCORE [contest-CUDA]"
    if [ "$SCORE" != "NA" ]; then
        IS_BETTER=$("$PYBIN" -c "print('YES' if float('$SCORE') < float('$BEST_SCORE') else 'NO')")
        if [ "$IS_BETTER" = "YES" ]; then
            BEST_SCORE="$SCORE"
            BEST_ARCHIVE="$ARCHIVE"
            BEST_BUDGET="$B"
        fi
    fi
done

log "=== Stage 6: pick best ==="
log "best_budget=$BEST_BUDGET best_score=$BEST_SCORE [contest-CUDA] best_archive=$BEST_ARCHIVE"

"$PYBIN" -c "
import json, os, time
prov = json.load(open('$PROVENANCE'))
prov['completed_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
prov['best_budget_bits'] = int('$BEST_BUDGET') if '$BEST_BUDGET' else 0
prov['best_score'] = float('$BEST_SCORE') if '$BEST_SCORE' != '999.99' else None
prov['best_archive'] = '$BEST_ARCHIVE'
prov['lane_status'] = 'COMPLETE' if '$BEST_ARCHIVE' else 'NO_VALID_SCORE'
json.dump(prov, open('$PROVENANCE', 'w'), indent=2)
print('provenance updated.')
"

log "=== LANE_OMEGA_W_DONE — best_score=$BEST_SCORE [contest-CUDA] at budget=$BEST_BUDGET ==="
