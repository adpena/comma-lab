#!/bin/bash
# Lane SC++ — Lane SA + Quantizr SegNet KL-distill auxiliary at T=2.0.
#
# Same SegMap arch as Lane SA (hidden=24, block_hidden=24, num_blocks=8) but
# with --variant kl_distill. The trainer adds a KL distill auxiliary at
# weight=0.002 (matches Lane G v3 corrected — KL ≈ 10% of scorer loss, the
# canonical Hinton 2015 auxiliary regime).
#
# Predicted band [0.30, 0.40] [contest-CUDA] — sub-Quantizr territory.
# Anchor: experiments/results/lane_a_landed/iter_0/ (full-res 384x512 masks).
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
#
# E2E_SMOKE_OPT_OUT: shared modules tac.segmap_renderer + tac.mask_grayscale_lut + tac.block_fp_codec just landed; smoke proof will be backfilled before remote dispatch.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
SEGMAP_ENABLE_LCT="${SEGMAP_ENABLE_LCT:-0}"
SEGMAP_CLASS_TARGETS_FILENAME="${SEGMAP_CLASS_TARGETS_FILENAME:-class_targets.fp16}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_sc_plus_plus_kl_distill"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
log() { echo "[lane-sc++] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_sc_plus_plus_kl_distill.sh',
    'output_dir': '$LOG_DIR',
    'variant': 'kl_distill',
    'arch': {'hidden': 24, 'block_hidden': 24, 'num_blocks': 8},
    'kl_distill_weight': 0.002,
    'kl_distill_temperature': 2.0,
    'learnable_class_targets_enabled': '$SEGMAP_ENABLE_LCT' == '1',
    'class_targets_filename': '$SEGMAP_CLASS_TARGETS_FILENAME',
    'predicted_band': [0.30, 0.40],
    'anchor_dir': 'experiments/results/lane_a_landed/iter_0',
    'paradigm': 'segmap_clone_with_kl_distill',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=SC++ gpu=$GPU" >> "$HEARTBEAT"
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
log "=== Stage 1: anchor checks (Check 76 — full-res masks only) ==="
for f in "$ANCHOR_RENDERER" "$ANCHOR_MASKS" "$ANCHOR_POSES" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing anchor $f" >&2; exit 1; }
done

TRAIN_LCT_ARGS=()
if [ "$SEGMAP_ENABLE_LCT" = "1" ]; then
    TRAIN_LCT_ARGS=(--learnable-class-targets --class-targets-filename "$SEGMAP_CLASS_TARGETS_FILENAME")
fi

log "=== Stage 2: train SegMap (variant=kl_distill, T=2.0, w=0.002) ==="
export PYTHONHASHSEED=1234
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Council C OOM-class deep fixes (DF2 + DF3) — see Check 87 STRICT.
# .omx/research/council_oom_class_deep_fix_20260429.md.
# --bf16 cuts FastViT attention-map allocation ~50% (DF2).
# --scorer-chunk 2 splits dual scorer_forward_pair calls into 2-pair chunks (DF3).
# --batch-size 4 keeps B*N (effective per-scorer-call frame count) <= 8 → fits
# RTX 4090 24 GB / A10G 22 GB with margin (Council C recommendation).
set +e
"$PYBIN" -u experiments/train_segmap.py \
    --variant kl_distill \
    --kl-distill-weight 0.002 \
    --kl-distill-temperature 2.0 \
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
    "${TRAIN_LCT_ARGS[@]}" \
    --tag "$LANE_ID" \
    --output-dir "$LOG_DIR/train" 2>&1 | tee "$LOG_DIR/train.log" | tail -30
PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: train_segmap.py exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

INFERENCE_PT="$LOG_DIR/train/segmap_inference.pt"
[ -f "$INFERENCE_PT" ] || { echo "FATAL: missing $INFERENCE_PT" >&2; exit 2; }

log "=== Stage 3: pack inference state via block_fp_codec ==="
PAYLOAD="$LOG_DIR/segmap_weights.tar.xz"
"$PYBIN" -c "
import torch
from tac.block_fp_codec import pack_payload_tar_xz, verify_roundtrip

state = torch.load('$INFERENCE_PT', map_location='cpu', weights_only=False)
pack_payload_tar_xz(state, '$PAYLOAD')
verify_roundtrip(state, '$PAYLOAD', tol=1e-6)
import os
print('payload bytes:', os.path.getsize('$PAYLOAD'))
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
# Round 2 review Medium: yuv420p w/ chroma=128 costs +9.4% vs gray. Use
# ffmpeg subprocess with -pix_fmt gray (matches build_lane_mm_archive.py).
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
if [ "$SEGMAP_ENABLE_LCT" = "1" ]; then
    LCT_PAYLOAD="$LOG_DIR/train/$SEGMAP_CLASS_TARGETS_FILENAME"
    [ -f "$LCT_PAYLOAD" ] || { echo "FATAL: missing LCT payload: $LCT_PAYLOAD" >&2; exit 2; }
    cp "$LCT_PAYLOAD" "$LOG_DIR/archive_src/$SEGMAP_CLASS_TARGETS_FILENAME"
fi

ARCHIVE="$LOG_DIR/archive_${LANE_ID}.zip"
"$PYBIN" -c "
import os, zipfile
src = '$LOG_DIR/archive_src'
members = ['segmap_weights.tar.xz', 'grayscale.mkv', 'optimized_poses.pt']
if '$SEGMAP_ENABLE_LCT' == '1':
    members.append('$SEGMAP_CLASS_TARGETS_FILENAME')
with zipfile.ZipFile('$ARCHIVE', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in members:
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing archive component {p}'
        info = zipfile.ZipInfo(filename=n, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (0o644 & 0xFFFF) << 16
        with open(p, 'rb') as fh:
            z.writestr(info, fh.read(), compresslevel=9)
print('archive bytes:', os.path.getsize('$ARCHIVE'))
"

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
[ "$ARCHIVE_BYTES" -gt 0 ] || { echo 'FATAL: archive empty'; exit 2; }
log "archive_bytes=$ARCHIVE_BYTES"

# config.env override: dispatch via segmap arm (Selfcomp paradigm).
INFLATE_CONFIG="$LOG_DIR/lane_sc_plus_plus_config.env"
cat > "$INFLATE_CONFIG" <<'EOF'
PYTHON_INFLATE=segmap
SEGMAP_GRAYSCALE_MODE=soft_lut
EOF
if [ "$SEGMAP_ENABLE_LCT" = "1" ]; then
    echo "SEGMAP_CLASS_TARGETS_FILENAME=$SEGMAP_CLASS_TARGETS_FILENAME" >> "$INFLATE_CONFIG"
fi

log "=== Stage 5: contest_auth_eval [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
set +e
CONFIG_ENV_PATH="$INFLATE_CONFIG" "$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: contest_auth_eval exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

"$PYBIN" -c "
import json, os, time
prov = json.load(open('$PROVENANCE'))
prov['completed_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
prov['archive_path'] = '$ARCHIVE'
prov['archive_bytes'] = os.path.getsize('$ARCHIVE')
prov['payload_bytes'] = os.path.getsize('$PAYLOAD')
prov['lane_status'] = 'COMPLETE'
json.dump(prov, open('$PROVENANCE', 'w'), indent=2)
print('provenance updated.')
"

log "=== LANE_SC++_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON [contest-CUDA] ==="
