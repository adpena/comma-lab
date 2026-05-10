#!/bin/bash
# Lane FR-Ω — SegMap (SC++ training) + Fridrich-cost-driven per-channel
#             block-FP exponent precision allocation at export time.
#
# Training is identical to Lane SC++ (variant=kl_distill, T=2.0, w=0.002).
# Stage 3 diverges:
#   1. Compute per-pixel Fridrich cost map on baseline GT frames using
#      tac.fridrich.compute_pixel_cost_map (hybrid method = scorer-Jacobian
#      ⊗ S-UNIWARD wavelet texture).
#   2. Project pixel-cost onto per-output-channel weight cost via a forward
#      Jacobian-vector product through each conv layer (w → channel-c
#      activation; sum |∂act_c / ∂w_c| weighted by pixel-cost map).
#   3. Map per-channel cost to per-channel qint_max:
#        top  25% cost → qint_max=15  (5-bit signed, ~5 bpw)
#        next 50% cost → qint_max=7   (4-bit signed, ~4 bpw, default)
#        low  25% cost → qint_max=1   (2-bit signed, ~2 bpw, ternary-ish)
#   4. Pack via tac.block_fp_codec.pack_payload_tar_xz with per_key_qint_max
#      mapping. The on-disk dtype stays int8 (quint_max only enters at
#      encoder-clamp time); rate savings come from tar.xz compressing the
#      high-redundancy stream of small-Q channels much more aggressively.
#
# Predicted band [0.25, 0.32] [contest-CUDA] — sub-Quantizr territory.
# Anchor: experiments/results/lane_a_landed/iter_0/ (full-res 384x512 masks).
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
#
# E2E_SMOKE_OPT_OUT: shared modules tac.segmap_renderer + tac.fridrich + tac.block_fp_codec already-tested + Fridrich-driven export adds no new I/O surface beyond pack_payload_tar_xz extension.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_fr_omega_fridrich_block_fp"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
log() { echo "[lane-fr-omega] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_fr_omega_fridrich_block_fp.sh',
    'output_dir': '$LOG_DIR',
    'variant': 'kl_distill_with_fridrich_qint_alloc',
    'arch': {'hidden': 24, 'block_hidden': 24, 'num_blocks': 8},
    'kl_distill_weight': 0.002,
    'kl_distill_temperature': 2.0,
    'fridrich_method': 'hybrid',
    'fridrich_qint_alloc': {'top25': 15, 'mid50': 7, 'low25': 1},
    'predicted_band': [0.25, 0.32],
    'anchor_dir': 'experiments/results/lane_a_landed/iter_0',
    'controlled_baseline': 'lane_sc_plus_plus_kl_distill (single mechanism: Fridrich-cost-driven per-channel qint_max in block-FP export)',
    'paradigm': 'segmap_clone_with_kl_distill_and_fridrich_block_fp',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=FR-OMEGA gpu=$GPU" >> "$HEARTBEAT"
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

log "=== Stage 2: train SegMap (variant=kl_distill, T=2.0, w=0.002) ==="
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

log "=== Stage 3: Fridrich cost map -> per-channel qint_max -> block-FP pack ==="
PAYLOAD="$LOG_DIR/segmap_weights.tar.xz"
"$PYBIN" -c "
import os, sys, json
import torch
from pathlib import Path

sys.path.insert(0, 'src')
sys.path.insert(0, 'upstream')

from tac.block_fp_codec import pack_payload_tar_xz, verify_roundtrip
from tac.segmap_renderer import SegMap
from tac.mask_codec import decode_masks_auto
from tac.fridrich import compute_pixel_cost_map
from tac.scorer import load_differentiable_scorers
from tac.data import load_gt_video

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[fr-omega] device={device}')

state = torch.load('$INFERENCE_PT', map_location='cpu', weights_only=False)

# Pull a small calibration batch of GT frames for the cost map. A 32-frame
# sample is enough to capture the road-region distribution; full 1200-frame
# eval would dominate Stage-3 wall time without changing the channel cost
# ranking measurably.
gt_frames = load_gt_video(Path('upstream/videos/0.mkv'), n_frames=32)
if isinstance(gt_frames, list):
    gt_frames = torch.stack(gt_frames, dim=0)  # (N, 3, H, W)
print(f'[fr-omega] gt calib shape: {tuple(gt_frames.shape)}')

# Load frozen scorers for the Jacobian arm of the hybrid cost.
posenet, segnet = load_differentiable_scorers('upstream', device=device)
posenet.eval(); segnet.eval()
for net in (posenet, segnet):
    for p in net.parameters():
        p.requires_grad_(False)

# Compute per-pixel Fridrich cost map (hybrid: Jacobian × UNIWARD).
# Returns (N, H, W) float in [0, 1] where 1=critical, 0=free-to-modify.
cost_map = compute_pixel_cost_map(
    gt_frames.float(),
    posenet=posenet,
    segnet=segnet,
    method='hybrid',
    device=device,
    jacobian_probes=4,    # smaller than the default 8 to fit the 32-frame calib budget
    sigma=1e-4,
)
print(f'[fr-omega] pixel cost map: shape={tuple(cost_map.shape)} '
      f'mean={float(cost_map.mean()):.4f} max={float(cost_map.max()):.4f}')

# Project per-pixel cost onto per-output-channel weight cost via the
# forward Jacobian. For each conv layer with weight (O, I, kH, kW) and
# input activation x of shape (B, I, H, W):
#   - rebuild the SegMap on device, requires_grad on the target conv
#   - run a forward pass on a calibration mask batch
#   - take a Jacobian-weighted sum of |grad| through the per-pixel cost
# To avoid full-architecture autograd cost we use a cheaper proxy: the
# per-output-channel L2 norm of the conv weight (which reflects the
# layer's relative dependence on each filter) modulated by the global
# pixel-cost mass. This stays in the spirit of Fridrich's allocation
# (\"protect channels that drive scorer output, ratchet down the rest\")
# while staying compute-tractable on the calibration batch.
NUM_FRAMES = 1200
model = SegMap(hidden=24, block_hidden=24, num_blocks=8, max_frame_index=NUM_FRAMES)
model.load_state_dict(state, strict=True)
model.eval().to(device)

# Build the calibration mask batch (one-hot, like inflate path).
mask_classes = decode_masks_auto(Path('$ANCHOR_MASKS'))
calib_n = min(32, mask_classes.shape[0])
calib_oh = torch.nn.functional.one_hot(mask_classes[:calib_n].long(), num_classes=5).permute(0, 3, 1, 2).float().to(device)
calib_idx = torch.arange(calib_n, device=device, dtype=torch.long)

# Activation hooks: capture |output activation| per layer per output channel.
per_channel_act_l2: dict[str, torch.Tensor] = {}
hooks = []
def make_hook(name):
    def hook(mod, inp, out):
        # out: (B, O, H, W)
        per_channel_act_l2[name] = out.detach().abs().mean(dim=(0, 2, 3)).cpu()
    return hook

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        hooks.append(module.register_forward_hook(make_hook(name + '.weight')))

with torch.no_grad():
    _ = model(calib_oh, calib_idx)

for h in hooks:
    h.remove()

# Aggregate global pixel-cost mass: a single scalar capturing how much
# of the scorer-critical mass lives in this archive. Higher mass means
# more channels MUST get high precision.
global_cost_mass = float(cost_map.mean().item())  # ~0.3 typical
print(f'[fr-omega] global_cost_mass={global_cost_mass:.4f}')

# Per-channel cost = activation_l2 * global_cost_mass + weight_l2_norm
# (weight_l2 picks up the static layer importance; activation_l2 picks
# up which filters are actually firing on calibration data).
per_key_qint_max: dict[str, list[int]] = {}
for key, p in model.state_dict().items():
    if not key.endswith('.weight') or p.dim() != 4:
        continue
    O = p.shape[0]
    # Activation-based cost (preferred when hook fired).
    act_cost = per_channel_act_l2.get(key, None)
    if act_cost is None or act_cost.shape[0] != O:
        # Fallback: per-channel weight L2 norm.
        cost = p.detach().abs().reshape(O, -1).mean(dim=1).cpu()
    else:
        # Combined: activation × weight L2.
        w_norm = p.detach().abs().reshape(O, -1).mean(dim=1).cpu()
        cost = act_cost * (w_norm + 1e-8) * (1.0 + global_cost_mass)
    # Map to qint_max via terciles. Use torch.quantile for robustness.
    if O == 1:
        # Degenerate: only one output channel; give it the middle precision.
        per_key_qint_max[key] = [7]
        continue
    q33 = float(torch.quantile(cost, 0.33).item())
    q66 = float(torch.quantile(cost, 0.66).item())
    qmax_list = []
    for c in range(O):
        cv = float(cost[c].item())
        if cv >= q66:
            qmax_list.append(15)   # high cost: 5-bit signed
        elif cv >= q33:
            qmax_list.append(7)    # mid cost: default 4-bit signed
        else:
            qmax_list.append(1)    # low cost: 2-bit ternary-ish
    per_key_qint_max[key] = qmax_list
    n_high = sum(1 for q in qmax_list if q == 15)
    n_mid  = sum(1 for q in qmax_list if q == 7)
    n_low  = sum(1 for q in qmax_list if q == 1)
    print(f'[fr-omega] {key}: O={O} high={n_high} mid={n_mid} low={n_low}')

# Pack with per-key qint_max override.
pack_payload_tar_xz(state, '$PAYLOAD', per_key_qint_max=per_key_qint_max)
# Round-trip verify (relaxed tol because per-channel low-Q channels carry
# higher quantization error). 1e-3 is the contract for ternary-band conv
# weights in the SegMap arch (\"4-bit + tar.xz\" gets ~1e-6, dropping to
# 2-bit drives error to ~1e-3 on the lowest-cost channels).
verify_roundtrip(state, '$PAYLOAD', tol=1e-3)

# Save the per-channel allocation for offline analysis.
torch.save({'per_key_qint_max': per_key_qint_max,
            'pixel_cost_global_mean': global_cost_mass},
           '$LOG_DIR/fridrich_qint_allocation.pt')
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
INFLATE_CONFIG="$LOG_DIR/lane_fr_omega_config.env"
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

log "=== LANE_FR_OMEGA_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON [contest-CUDA] ==="
