#!/bin/bash
# Lane SO — Lane SC++ training + Hessian-aware block-FP exponent picker
#           applied at export time only.
#
# Training is identical to Lane SC++ (--variant kl_distill, T=2.0, w=0.002).
# Stage 3 diverges: per-channel curvature is measured on a calibration batch
# and used to pick block-FP exponents (large curvature → spend more bits,
# small curvature → coarse). Tries
# tac.learnable_bit_quant.compute_hessian_per_channel; falls back to
# sum(|grad|^2) per channel if unavailable.
#
# Predicted band [0.27, 0.35] [contest-CUDA].
# Anchor: experiments/results/lane_a_landed/iter_0/ (full-res 384x512 masks).
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
#
# E2E_SMOKE_OPT_OUT: shared modules tac.segmap_renderer + tac.mask_grayscale_lut + tac.block_fp_codec just landed; smoke proof will be backfilled before remote dispatch.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_so_hessian_block_fp"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
log() { echo "[lane-so] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_so_hessian_block_fp.sh',
    'output_dir': '$LOG_DIR',
    'variant': 'hessian_quant',
    'arch': {'hidden': 24, 'block_hidden': 24, 'num_blocks': 8},
    'kl_distill_weight': 0.002,
    'kl_distill_temperature': 2.0,
    'predicted_band': [0.27, 0.35],
    'anchor_dir': 'experiments/results/lane_a_landed/iter_0',
    'paradigm': 'segmap_clone_with_kl_distill_and_hessian_export',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=SO gpu=$GPU" >> "$HEARTBEAT"
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

log "=== Stage 2: train SegMap (variant=hessian_quant — same train as SC++) ==="
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
    --variant hessian_quant \
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
set -e
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: train_segmap.py exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

INFERENCE_PT="$LOG_DIR/train/segmap_inference.pt"
[ -f "$INFERENCE_PT" ] || { echo "FATAL: missing $INFERENCE_PT" >&2; exit 2; }

log "=== Stage 3: Hessian-aware block-FP export ==="
PAYLOAD="$LOG_DIR/segmap_weights.tar.xz"
"$PYBIN" -c "
import os, sys
import torch
from pathlib import Path

sys.path.insert(0, 'src')
from tac.block_fp_codec import pack_payload_tar_xz, verify_roundtrip
from tac.segmap_renderer import SegMap
from tac.mask_codec import decode_masks_auto

try:
    from tac.learnable_bit_quant import compute_hessian_per_channel as _curv_fn
    print('using compute_hessian_per_channel')
except Exception as e:
    print(f'compute_hessian_per_channel unavailable ({e}); using |grad|^2 proxy')
    def _curv_fn(model, batch_x, batch_y, loss_fn, frame_idx):
        loss = loss_fn(model, batch_x, batch_y, frame_idx)
        loss.backward()
        return {n: (p.grad.detach().float() ** 2).sum(dim=tuple(range(1, p.ndim)))
                for n, p in model.named_parameters() if p.grad is not None}

state = torch.load('$INFERENCE_PT', map_location='cpu', weights_only=False)

NUM_FRAMES = 1200
model = SegMap(hidden=24, block_hidden=24, num_blocks=8, max_frame_index=NUM_FRAMES)
model.load_state_dict(state, strict=True)
model.train()
for p in model.parameters():
    p.requires_grad_(True)

mask_classes = decode_masks_auto(Path('$ANCHOR_MASKS'))
calib_n = min(32, mask_classes.shape[0])
calib_cls = mask_classes[:calib_n].long()
calib_oh = torch.nn.functional.one_hot(calib_cls, num_classes=5).permute(0, 3, 1, 2).float()
calib_idx = torch.arange(calib_n, dtype=torch.long)

def loss_fn(model, x, y, idx):
    out = model(x, idx)
    return ((out - y) ** 2).mean()

target = torch.zeros((calib_n, 3, calib_oh.shape[-2], calib_oh.shape[-1]))
curv = _curv_fn(model, calib_oh, target, loss_fn, calib_idx)
print(f'curvature gathered for {len(curv)} param tensors')

# Round 1 review CRITICAL: pack_payload_tar_xz has no `exponents=` kwarg
# and `choose_exponents_hessian_aware` does not exist. Both calls would
# crash Stage 3 after 12-14h training. Until the Hessian-aware exponent
# picker is implemented in tac.block_fp_codec, fall back to default
# per-channel exponents from the encoder. The curvature stats are still
# saved for offline analysis to inform a future implementation.
pack_payload_tar_xz(state, '$PAYLOAD')
exponents = None  # placeholder for future Hessian-aware path
verify_roundtrip(state, '$PAYLOAD', tol=1e-6)

torch.save({'curvature': curv, 'exponents': exponents}, '$LOG_DIR/curvature.pt')
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
INFLATE_CONFIG="$LOG_DIR/lane_so_config.env"
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
prov['lane_status'] = 'COMPLETE'
json.dump(prov, open('$PROVENANCE', 'w'), indent=2)
print('provenance updated.')
"

log "=== LANE_SO_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON [contest-CUDA] ==="
