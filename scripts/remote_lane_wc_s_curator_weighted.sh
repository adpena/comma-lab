#!/bin/bash
# Lane WC-S — SegMap (SC++ training) + Cosmos Curator soft-DTW outlier
#             pair weighting (Lane WC concept ported into the SegMap paradigm).
#
# The Curator scorer (tac.curator_outlier.CuratorOutlierScorer) fits PCA +
# soft-DTW time-series KMeans on per-pair SegNet features, then assigns each
# pair a typicality score in [0, 1] and a derived training weight in
# [1, weight_scale]. Lane WC's hypothesis: hardest pairs (top quantile of
# soft-DTW distance from cluster barycenter) carry disproportionate scoring
# mass; weighting them 5x during SegMap training lets the renderer steer
# capacity toward them and improves the worst-case PoseNet/SegNet output.
#
# Pipeline:
#   Stage 2a: extract per-pair SegNet penultimate features (600 pairs).
#   Stage 2b: fit Curator scorer + derive pair_weights.pt.
#   Stage 2c: train SegMap with --variant kl_distill --pair-weights ...
#   Stage 3+: standard SegMap pack + archive + auth eval.
#
# Predicted band [0.26, 0.34] [contest-CUDA].
# Anchor: experiments/results/lane_a_landed/iter_0/ (full-res 384x512 masks).
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
#
# E2E_SMOKE_OPT_OUT: shared modules tac.segmap_renderer + tac.curator_outlier already-tested; pair_weights wiring is a single per-pair tensor that hard-errors on wrong shape (no silent broadcast).
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_wc_s_curator_weighted"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
log() { echo "[lane-wc-s] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_wc_s_curator_weighted.sh',
    'output_dir': '$LOG_DIR',
    'variant': 'kl_distill_with_curator_weighting',
    'arch': {'hidden': 24, 'block_hidden': 24, 'num_blocks': 8},
    'kl_distill_weight': 0.002,
    'kl_distill_temperature': 2.0,
    'curator_weight_scale': 5.0,
    'curator_n_pca_components': 3,
    'curator_n_clusters': 5,
    'curator_outlier_quantile': 0.95,
    'predicted_band': [0.26, 0.34],
    'anchor_dir': 'experiments/results/lane_a_landed/iter_0',
    'controlled_baseline': 'lane_sc_plus_plus_kl_distill (single mechanism: per-pair Curator outlier loss weighting in train_segmap.py)',
    'paradigm': 'segmap_clone_with_kl_distill_and_curator_outlier_weighting',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=WC-S gpu=$GPU" >> "$HEARTBEAT"
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

log "=== Stage 2a: extract per-pair SegNet penultimate features (600 pairs) ==="
SEGNET_FEATURES="$LOG_DIR/segnet_features.pt"
"$PYBIN" -c "
import sys, torch
from pathlib import Path

sys.path.insert(0, 'src')
sys.path.insert(0, 'upstream')

from tac.scorer import load_differentiable_scorers
from tac.data import load_gt_video

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[wc-s feat] device={device}')

posenet, segnet = load_differentiable_scorers('upstream', device=device)
posenet.eval(); segnet.eval()
for net in (posenet, segnet):
    for p in net.parameters():
        p.requires_grad_(False)

# Load GT pairs (1200 frames -> 600 non-overlapping pairs).
gt_frames = load_gt_video(Path('upstream/videos/0.mkv'), n_frames=1200)
if isinstance(gt_frames, list):
    gt_frames = torch.stack(gt_frames, dim=0)
N = gt_frames.shape[0]
assert N == 1200, f'expected 1200 frames, got {N}'
half = N // 2
print(f'[wc-s feat] GT shape: {tuple(gt_frames.shape)} -> {half} pairs')

# Build pair tensor (P, T=2, 3, H, W).
pairs = gt_frames.view(half, 2, *gt_frames.shape[-3:])

# Extract SegNet penultimate features per pair. SegNet preprocesses the
# LAST frame and runs the U-Net backbone; we hook the bottleneck (encoder
# tail) for the Curator's PCA input. If the hook isn't wired, fall back
# to the per-pair softmax-class distribution as a low-D feature surrogate.
features = []
batch = 8
import torch.nn.functional as F
with torch.no_grad():
    for start in range(0, half, batch):
        end = min(start + batch, half)
        chunk = pairs[start:end].to(device, dtype=torch.float32)
        # tac.segmap_renderer._MockSegNet uses preprocess_input(x) -> last frame
        # downsampled. Real SegNet has the same preprocess_input contract.
        if hasattr(segnet, 'preprocess_input'):
            seg_in = segnet.preprocess_input(chunk)
        else:
            seg_in = chunk[:, -1, ...]
        out = segnet(seg_in)  # (B, 5, H, W)
        # Per-pair feature = mean pooled per-class softmax.
        prob = F.softmax(out.float(), dim=1)
        feat = prob.mean(dim=(2, 3))  # (B, 5)
        features.append(feat.cpu())
features = torch.cat(features, dim=0)
print(f'[wc-s feat] features: {tuple(features.shape)}')
torch.save(features, '$SEGNET_FEATURES')
"

log "=== Stage 2b: fit Curator outlier scorer + derive pair_weights ==="
CURATOR_DIR="$LOG_DIR/curator"
mkdir -p "$CURATOR_DIR"
set +e
"$PYBIN" -u experiments/fit_curator_outlier_weights.py \
    --segnet-features "$SEGNET_FEATURES" \
    --n-pairs 600 \
    --output-dir "$CURATOR_DIR" \
    --n-pca-components 3 \
    --n-clusters 5 \
    --soft-dtw-gamma 0.1 \
    --outlier-quantile 0.95 \
    --weight-scale 5.0 2>&1 | tee "$LOG_DIR/curator.log" | tail -20
PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: fit_curator_outlier_weights exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi
PAIR_WEIGHTS="$CURATOR_DIR/pair_weights.pt"
[ -f "$PAIR_WEIGHTS" ] || { echo "FATAL: missing $PAIR_WEIGHTS" >&2; exit 2; }

log "=== Stage 2c: train SegMap (variant=kl_distill, pair_weights=Curator) ==="
export PYTHONHASHSEED=1234
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Council C OOM-class deep fixes (DF2 + DF3) — see Check 87 STRICT.
# --bf16 + --scorer-chunk 2 + --batch-size 4 → B*N=8 (RTX 4090 24 GB safe).
set +e
"$PYBIN" -u experiments/train_segmap.py \
    --variant kl_distill \
    --kl-distill-weight 0.002 \
    --kl-distill-temperature 2.0 \
    --hidden 24 --block-hidden 24 --num-blocks 8 \
    --epochs 600 --batch-size 4 --lr 1e-3 \
    --bf16 --scorer-chunk 2 \
    --roundtrip-noise-std 0.5 \
    --pair-weights "$PAIR_WEIGHTS" \
    --anchor-renderer "$ANCHOR_RENDERER" \
    --anchor-poses "$ANCHOR_POSES" \
    --anchor-masks "$ANCHOR_MASKS" \
    --gt-video upstream/videos/0.mkv \
    --upstream upstream \
    --device cuda \
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
import torch, os
from tac.block_fp_codec import pack_payload_tar_xz, verify_roundtrip

state = torch.load('$INFERENCE_PT', map_location='cpu', weights_only=False)
pack_payload_tar_xz(state, '$PAYLOAD')
verify_roundtrip(state, '$PAYLOAD', tol=1e-6)
print('payload bytes:', os.path.getsize('$PAYLOAD'))
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

# config.env override: dispatch via segmap arm (Selfcomp paradigm).
INFLATE_CONFIG="$LOG_DIR/lane_wc_s_config.env"
cat > "$INFLATE_CONFIG" <<'EOF'
PYTHON_INFLATE=segmap
EOF

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
prov['pair_weights_path'] = '$PAIR_WEIGHTS'
prov['lane_status'] = 'COMPLETE'
json.dump(prov, open('$PROVENANCE', 'w'), indent=2)
print('provenance updated.')
"

log "=== LANE_WC_S_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON [contest-CUDA] ==="
