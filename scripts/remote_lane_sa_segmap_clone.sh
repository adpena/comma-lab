#!/bin/bash
# Lane SA — SegMap clone of Selfcomp 0.38 paradigm (Quantizr-style mask renderer).
#
# Architecture: tac.segmap_renderer.SegMap(hidden=24, block_hidden=24,
#               num_blocks=8, max_frame_index=1200). Selfcomp soft-LUT
#               5-class probability map -> RGB.
#
# Variant: --variant plain — standard scorer loss (eval_roundtrip=True,
#                            roundtrip_noise_std=0.5).
#
# Predicted band [0.40, 0.55] [contest-CUDA].
# Anchor: experiments/results/lane_a_landed/iter_0/ (full-res 384x512 masks).
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
#
# E2E_SMOKE_OPT_OUT: shared modules tac.segmap_renderer + tac.mask_grayscale_lut + tac.block_fp_codec just landed; smoke proof will be backfilled before remote dispatch.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_sa_segmap_clone"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
log() { echo "[lane-sa] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_sa_segmap_clone.sh',
    'output_dir': '$LOG_DIR',
    'variant': 'plain',
    'arch': {'hidden': 24, 'block_hidden': 24, 'num_blocks': 8},
    'grayscale_mode': 'soft_lut',
    'predicted_band': [0.40, 0.55],
    'anchor_dir': 'experiments/results/lane_a_landed/iter_0',
    'paradigm': 'segmap_clone',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=SA gpu=$GPU" >> "$HEARTBEAT"
    sleep 300
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed -- destroy this Vast.ai instance."
    exit 2
}

ANCHOR_DIR="experiments/results/lane_a_landed/iter_0"
ANCHOR_RENDERER="$ANCHOR_DIR/renderer.bin"
ANCHOR_MASKS="$ANCHOR_DIR/masks.mkv"
ANCHOR_POSES="experiments/results/lane_a_landed/optimized_poses.pt"
log "=== Stage 1: anchor checks (Check 76 — full-res masks only) ==="
for f in "$ANCHOR_RENDERER" "$ANCHOR_MASKS" "$ANCHOR_POSES" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors \
         experiments/contest_auth_eval.py \
         scripts/adjudicate_contest_auth_eval.py; do
    [ -f "$f" ] || { echo "FATAL: missing anchor $f" >&2; exit 1; }
done

log "=== Stage 2: train SegMap (variant=plain) ==="
export PYTHONHASHSEED=1234
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Council C OOM-class deep fixes (DF2 + DF3) — see Check 87 STRICT.
# .omx/research/council_oom_class_deep_fix_20260429.md.
# --bf16 cuts FastViT attention-map allocation ~50% (DF2).
# --scorer-chunk 2 splits dual scorer_forward_pair calls into 2-pair chunks (DF3).
# --batch-size 4 keeps B*N (effective per-scorer-call frame count) <= 8 → fits
# RTX 4090 24 GB / A10G 22 GB with margin (Council C recommendation).
"$PYBIN" -u experiments/train_segmap.py \
    --variant plain \
    --hidden 24 --block-hidden 24 --num-blocks 8 \
    --epochs 600 --batch-size 4 --lr 1e-3 \
    --bf16 --scorer-chunk 2 \
    --roundtrip-noise-std 0.5 \
    --anchor-renderer "$ANCHOR_RENDERER" \
    --anchor-poses "$ANCHOR_POSES" \
    --anchor-masks "$ANCHOR_MASKS" \
    --gt-video upstream/videos/0.mkv \
    --upstream upstream \
    --device cuda \
    --tag "$LANE_ID" \
    --output-dir "$LOG_DIR/train" 2>&1 | tee "$LOG_DIR/train.log" | tail -30
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: train_segmap.py exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

INFERENCE_PT="$LOG_DIR/train/segmap_inference.pt"
[ -f "$INFERENCE_PT" ] || { echo "FATAL: missing $INFERENCE_PT" >&2; exit 2; }

log "=== Stage 3: pack inference state via block_fp_codec ==="
PAYLOAD="$LOG_DIR/segmap_weights.tar.xz"
"$PYBIN" -c "
import json, os, torch
from tac.block_fp_codec import (
    SEGMAP_LOSSY_ROUNDTRIP_MSE_TOL,
    segmap_lossy_contract_metadata,
    verify_roundtrip,
)

state = torch.load('$INFERENCE_PT', map_location='cpu', weights_only=False)
contract = segmap_lossy_contract_metadata(SEGMAP_LOSSY_ROUNDTRIP_MSE_TOL)
mse_by_key = verify_roundtrip(
    state,
    '$PAYLOAD',
    tol=SEGMAP_LOSSY_ROUNDTRIP_MSE_TOL,
    lossy_contract=contract,
)
max_mse_key = max(mse_by_key, key=mse_by_key.get)
roundtrip = {
    'contract': contract,
    'mse_by_key': mse_by_key,
    'max_mse_key': max_mse_key,
    'max_mse': mse_by_key[max_mse_key],
    'payload_path': '$PAYLOAD',
    'payload_bytes': os.path.getsize('$PAYLOAD'),
    'gate': 'archive-level CUDA contest_auth_eval is required before any score claim',
}
with open('$LOG_DIR/segmap_pack_roundtrip.json', 'w') as f:
    json.dump(roundtrip, f, indent=2, sort_keys=True)
prov = json.load(open('$PROVENANCE'))
prov['segmap_pack_contract'] = contract
prov['segmap_pack_roundtrip'] = {
    'path': '$LOG_DIR/segmap_pack_roundtrip.json',
    'max_mse_key': max_mse_key,
    'max_mse': mse_by_key[max_mse_key],
    'tol': SEGMAP_LOSSY_ROUNDTRIP_MSE_TOL,
}
prov['archive_level_exact_eval_required'] = True
json.dump(prov, open('$PROVENANCE', 'w'), indent=2)
print('payload bytes:', os.path.getsize('$PAYLOAD'))
print('segmap lossy roundtrip max_mse:', max_mse_key, mse_by_key[max_mse_key])
"

log "=== Stage 4: build archive (grayscale.mkv + segmap_weights.tar.xz + poses) ==="
mkdir -p "$LOG_DIR/archive_src"
GRAYSCALE_MKV="$LOG_DIR/archive_src/grayscale.mkv"
"$PYBIN" -c "
import subprocess
import torch
from pathlib import Path
from tac.mask_codec import decode_masks_auto
from tac.mask_grayscale_lut import encode_masks_grayscale

mask_classes = decode_masks_auto(Path('$ANCHOR_MASKS'))
gray = encode_masks_grayscale(mask_classes)
import numpy as np
arr = gray.numpy() if isinstance(gray, torch.Tensor) else np.asarray(gray)
out_path = Path('$GRAYSCALE_MKV')
n, h, w = arr.shape
# Round 2 review Medium: yuv420p w/ chroma=128 costs +9.4% vs gray.
# Use ffmpeg subprocess with -pix_fmt gray to skip chroma planes entirely
# (matches experiments/build_lane_mm_archive.py canonical encoder).
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
import os
print('grayscale.mkv bytes:', os.path.getsize(out_path))
"
cp "$ANCHOR_POSES" "$LOG_DIR/archive_src/optimized_poses.pt"
cp "$PAYLOAD" "$LOG_DIR/archive_src/segmap_weights.tar.xz"

ARCHIVE="$LOG_DIR/archive_${LANE_ID}.zip"
"$PYBIN" -c "
import os, zipfile
src = '$LOG_DIR/archive_src'
with zipfile.ZipFile('$ARCHIVE', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in ('segmap_weights.tar.xz', 'grayscale.mkv', 'optimized_poses.pt'):
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing archive component {p}'
        info = zipfile.ZipInfo(filename=n, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (0o644 & 0xFFFF) << 16
        with open(p, 'rb') as fh:
            z.writestr(info, fh.read(), compresslevel=9)
print('archive bytes:', os.path.getsize('$ARCHIVE'))
"

# Archive size guard (Check [lane-archive-size]).
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
[ "$ARCHIVE_BYTES" -gt 0 ] || { echo 'FATAL: archive empty'; exit 2; }
log "archive_bytes=$ARCHIVE_BYTES"

# config.env override: tell inflate.sh to dispatch via the segmap arm
# (Selfcomp paradigm: grayscale.mkv + segmap_weights.tar.xz + SegMap renderer).
INFLATE_CONFIG="$LOG_DIR/lane_sa_config.env"
cat > "$INFLATE_CONFIG" <<'EOF'
PYTHON_INFLATE=segmap
SEGMAP_GRAYSCALE_MODE=soft_lut
EOF

log "=== Stage 5: contest_auth_eval [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
CONFIG_ENV_PATH="$INFLATE_CONFIG" "$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: contest_auth_eval exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

RESULT_JSON="$LOG_DIR/RESULT_JSON"
ADJUDICATION_LOG="$LOG_DIR/adjudication.log"
"$PYBIN" -u scripts/adjudicate_contest_auth_eval.py \
    --contest-json "$LOG_DIR/eval_work/contest_auth_eval.json" \
    --provenance "$PROVENANCE" \
    --archive "$ARCHIVE" \
    --result-copy "$RESULT_JSON" \
    --baseline-score 1.15 \
    --predicted-band 0.40 0.55 \
    --regression-threshold 1.30 \
    --delta-key score_delta_vs_lane_a | tee "$ADJUDICATION_LOG"

"$PYBIN" -c "
import json, os, time
prov = json.load(open('$PROVENANCE'))
prov['completed_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
prov['archive_path'] = '$ARCHIVE'
prov['archive_bytes'] = os.path.getsize('$ARCHIVE')
prov['payload_bytes'] = os.path.getsize('$PAYLOAD')
prov['result_json'] = '$RESULT_JSON'
prov['archive_level_exact_eval_completed'] = True
prov['contest_auth_eval_json'] = '$LOG_DIR/eval_work/contest_auth_eval.json'
json.dump(prov, open('$PROVENANCE', 'w'), indent=2)
print('provenance updated.')
"

log "=== LANE_SA_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON [contest-CUDA] ==="
