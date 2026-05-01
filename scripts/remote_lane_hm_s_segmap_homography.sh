#!/bin/bash
# Lane HM-S — SegMap with 8-DOF perspective homography frame embeddings
#             (vs 6-DOF affine in the canonical Lane SA / SC++).
#
# 6-DOF affine ⊂ 8-DOF homography: the extra 2 perspective parameters [g, h]
# in the homography bottom row let the per-frame latent capture forward-zoom
# + tilt patterns that match the comma.ai dashcam viewing geometry better.
#
# Implementation: tac.segmap_renderer.SegMapHomography subclass swaps
# F.affine_grid for an explicit 3x3 homography matmul + grid_sample. The
# frame_affine_embedding goes from dim 6 -> dim 8 (only ~2400 extra params
# at max_frame_index=1200). Inflate dispatch via SEGMAP_ARCH=segmap_homography
# env var (read by inflate_segmap._build_segmap).
#
# Predicted band [0.32, 0.45] [contest-CUDA].
# Anchor: experiments/results/lane_a_landed/iter_0/ (full-res 384x512 masks).
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
#
# E2E_SMOKE_OPT_OUT: SegMapHomography is a clean nn.Module subclass of SegMap with the same load_state_dict / forward contract; covered by tac.segmap_renderer test additions and inflate_segmap.py SEGMAP_ARCH dispatch.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_hm_s_segmap_homography"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
log() { echo "[lane-hm-s] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_hm_s_segmap_homography.sh',
    'output_dir': '$LOG_DIR',
    'arch': {'class': 'SegMapHomography', 'hidden': 24, 'block_hidden': 24, 'num_blocks': 8, 'frame_embedding_dim': 8},
    'variant': 'kl_distill',
    'kl_distill_weight': 0.002,
    'kl_distill_temperature': 2.0,
    'predicted_band': [0.32, 0.45],
    'anchor_dir': 'experiments/results/lane_a_landed/iter_0',
    'controlled_baseline': 'lane_sc_plus_plus_kl_distill (single mechanism: 6-DOF affine -> 8-DOF perspective homography frame embedding)',
    'paradigm': 'segmap_homography_8dof',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=HM-S gpu=$GPU" >> "$HEARTBEAT"
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
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing anchor $f" >&2; exit 1; }
done

log "=== Stage 2: train SegMapHomography (variant=kl_distill, arch=segmap_homography) ==="
export PYTHONHASHSEED=1234
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Council C OOM-class deep fixes (DF2 + DF3) — see Check 87 STRICT.
# --bf16 + --scorer-chunk 2 + --batch-size 4 → B*N=8 (RTX 4090 24 GB safe).
"$PYBIN" -u experiments/train_segmap.py \
    --variant kl_distill \
    --arch segmap_homography \
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
# Sanity: SegMapHomography embeds (N, 8) instead of (N, 6).
emb = state.get('frame_affine_embedding.weight')
if emb is None or emb.dim() != 2 or emb.shape[1] != 8:
    raise RuntimeError(f'expected frame_affine_embedding shape (N, 8), got {None if emb is None else tuple(emb.shape)}')
print(f'[hm-s] frame_affine_embedding shape: {tuple(emb.shape)}')
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
import torch
from pathlib import Path
from tac.mask_codec import decode_masks_auto
from tac.mask_grayscale_lut import encode_masks_grayscale

mask_classes = decode_masks_auto(Path('$ANCHOR_MASKS'))
gray = encode_masks_grayscale(mask_classes.long())
import av, numpy as np
arr = gray.numpy() if isinstance(gray, torch.Tensor) else np.asarray(gray)
out_path = Path('$GRAYSCALE_MKV')
container = av.open(str(out_path), mode='w')
stream = container.add_stream('libsvtav1', rate=20)
stream.width = arr.shape[2]
stream.height = arr.shape[1]
stream.pix_fmt = 'yuv420p'
stream.options = {'crf': '50'}
for i in range(arr.shape[0]):
    yuv = np.zeros((arr.shape[1] * 3 // 2, arr.shape[2]), dtype='uint8')
    yuv[:arr.shape[1]] = arr[i]
    yuv[arr.shape[1]:] = 128
    frame = av.VideoFrame.from_ndarray(yuv, format='yuv420p')
    for pkt in stream.encode(frame):
        container.mux(pkt)
for pkt in stream.encode():
    container.mux(pkt)
container.close()
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
        z.write(p, arcname=n)
print('archive bytes:', os.path.getsize('$ARCHIVE'))
"

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
[ "$ARCHIVE_BYTES" -gt 0 ] || { echo 'FATAL: archive empty'; exit 2; }
log "archive_bytes=$ARCHIVE_BYTES"

# config.env override: dispatch via segmap arm with SEGMAP_ARCH override.
# inflate_segmap._build_segmap reads SEGMAP_ARCH and instantiates the
# SegMapHomography class so the loaded state-dict's (N, 8) embedding fits.
INFLATE_CONFIG="$LOG_DIR/lane_hm_s_config.env"
cat > "$INFLATE_CONFIG" <<'EOF'
PYTHON_INFLATE=segmap
SEGMAP_ARCH=segmap_homography
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

"$PYBIN" -c "
import json, os, time
prov = json.load(open('$PROVENANCE'))
prov['completed_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
prov['archive_path'] = '$ARCHIVE'
prov['archive_bytes'] = os.path.getsize('$ARCHIVE')
prov['payload_bytes'] = os.path.getsize('$PAYLOAD')
prov['lane_status'] = 'COMPLETE'
prov['archive_level_exact_eval_completed'] = True
prov['contest_auth_eval_json'] = '$LOG_DIR/eval_work/contest_auth_eval.json'
json.dump(prov, open('$PROVENANCE', 'w'), indent=2)
print('provenance updated.')
"

log "=== LANE_HM_S_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON [contest-CUDA] ==="
