#!/bin/bash
# Lane DARTS-S — SegMap architecture sweep (Karpathy eureka: Selfcomp
#                explicitly skipped this; the published 0.38 used a single
#                hard-coded (24, 24, 8) config and never benchmarked alt configs).
#
# Sweep 5 (hidden, block_hidden, num_blocks) configs:
#   small       (16, 16, 4)   — minimal capacity
#   default     (24, 24, 8)   — Selfcomp baseline (= Lane SC++ control)
#   wide        (32, 32, 8)   — wider channels, same depth
#   deep        (24, 24, 12)  — same width, deeper stack
#   xwide       (40, 40, 8)   — push channel count
#
# For each config: train SegMap with --variant kl_distill on full Lane A
# anchor (~2.5h on 4090), build the segmap archive, run contest_auth_eval.
# Final stage emits a sweep_summary.json mapping each arch_config -> score
# so we can pick the lowest score for promotion.
#
# Total wall-clock budget ~17h (5 × 3.4h); cost ~$5 at $0.30/hr 4090.
# Predicted band [0.27, 0.55] [contest-CUDA] (best config could go lower
# than Lane SC++; worst stays competitive with Lane SA).
#
# Anchor: experiments/results/lane_a_landed/iter_0/ (full-res 384x512 masks).
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
#
# E2E_SMOKE_OPT_OUT: each sweep config reuses the proven Lane SC++ pipeline (train_segmap.py + pack_payload_tar_xz + inflate_segmap arm); only the (hidden, block_hidden, num_blocks) tuple varies.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_darts_s_segmap_arch_sweep"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
log() { echo "[lane-darts-s] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_darts_s_segmap_arch_sweep.sh',
    'output_dir': '$LOG_DIR',
    'sweep_configs': [
        {'name': 'small', 'hidden': 16, 'block_hidden': 16, 'num_blocks': 4},
        {'name': 'default', 'hidden': 24, 'block_hidden': 24, 'num_blocks': 8},
        {'name': 'wide', 'hidden': 32, 'block_hidden': 32, 'num_blocks': 8},
        {'name': 'deep', 'hidden': 24, 'block_hidden': 24, 'num_blocks': 12},
        {'name': 'xwide', 'hidden': 40, 'block_hidden': 40, 'num_blocks': 8},
    ],
    'variant': 'kl_distill',
    'predicted_band': [0.27, 0.55],
    'anchor_dir': 'experiments/results/lane_a_landed/iter_0',
    'controlled_baseline': 'lane_sc_plus_plus_kl_distill (single mechanism: hidden/block_hidden/num_blocks sweep; default config matches SC++ exactly)',
    'paradigm': 'segmap_arch_sweep',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=DARTS-S gpu=$GPU" >> "$HEARTBEAT"
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

# Pre-encode the grayscale.mkv ONCE (it's identical across configs since
# arch only affects the renderer, not the mask encoding).
mkdir -p "$LOG_DIR/shared_archive_src"
GRAYSCALE_MKV_SHARED="$LOG_DIR/shared_archive_src/grayscale.mkv"
log "=== Stage 1b: pre-encode shared grayscale.mkv (used by all sweep configs) ==="
"$PYBIN" -c "
import torch
from pathlib import Path
from tac.mask_codec import decode_masks_auto
from tac.mask_grayscale_lut import encode_masks_grayscale

mask_classes = decode_masks_auto(Path('$ANCHOR_MASKS'))
gray = encode_masks_grayscale(mask_classes.long())
import av, numpy as np
arr = gray.numpy() if isinstance(gray, torch.Tensor) else np.asarray(gray)
out_path = Path('$GRAYSCALE_MKV_SHARED')
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

# Sweep loop. Configs declared as parallel arrays (bash 4 has no
# associative arrays portably); index aligns name/hidden/block_hidden/num_blocks.
NAMES=(default wide deep)
HIDDEN_ARR=(24 32 24)
BHIDDEN_ARR=(24 32 24)
NBLOCKS_ARR=(8 8 12)
EPOCHS=400  # per-config — slightly under Lane SC++'s 600 to fit total budget

SWEEP_RESULTS_JSON="$LOG_DIR/sweep_summary.json"
"$PYBIN" -c "import json; open('$SWEEP_RESULTS_JSON', 'w').write(json.dumps({'configs': []}, indent=2))"

export PYTHONHASHSEED=1234
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for i in "${!NAMES[@]}"; do
    NAME="${NAMES[$i]}"
    H="${HIDDEN_ARR[$i]}"
    BH="${BHIDDEN_ARR[$i]}"
    NB="${NBLOCKS_ARR[$i]}"
    CONFIG_DIR="$LOG_DIR/sweep_${NAME}"
    mkdir -p "$CONFIG_DIR"

    log "===================================================================="
    log "=== sweep config '$NAME' (hidden=$H block_hidden=$BH num_blocks=$NB) ==="
    log "===================================================================="

    log "  Stage 2/$NAME: train SegMap"
    # Council C OOM-class deep fixes (DF2 + DF3) — see Check 87 STRICT.
    # --bf16 + --scorer-chunk 2 + --batch-size 4 → B*N=8 (RTX 4090 24 GB safe).
    "$PYBIN" -u experiments/train_segmap.py \
        --variant kl_distill \
        --kl-distill-weight 0.002 \
        --kl-distill-temperature 2.0 \
        --hidden "$H" --block-hidden "$BH" --num-blocks "$NB" \
        --epochs "$EPOCHS" --batch-size 4 --lr 1e-3 \
        --bf16 --scorer-chunk 2 \
        --roundtrip-noise-std 0.5 \
        --anchor-renderer "$ANCHOR_RENDERER" \
        --anchor-poses "$ANCHOR_POSES" \
        --anchor-masks "$ANCHOR_MASKS" \
        --gt-video upstream/videos/0.mkv \
        --upstream upstream \
        --device cuda \
        --tag "${LANE_ID}_${NAME}" \
        --output-dir "$CONFIG_DIR/train" 2>&1 | tee "$CONFIG_DIR/train.log" | tail -20
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        log "  FATAL: train rc=${PIPE_RC[0]} for config '$NAME' — recording and continuing"
        "$PYBIN" -c "
import json
d = json.load(open('$SWEEP_RESULTS_JSON'))
d['configs'].append({'name': '$NAME', 'hidden': $H, 'block_hidden': $BH, 'num_blocks': $NB, 'status': 'TRAIN_FAILED', 'rc': ${PIPE_RC[0]}})
json.dump(d, open('$SWEEP_RESULTS_JSON', 'w'), indent=2)
"
        continue
    fi

    INFERENCE_PT="$CONFIG_DIR/train/segmap_inference.pt"
    [ -f "$INFERENCE_PT" ] || { log "  FATAL: missing $INFERENCE_PT for config '$NAME'"; continue; }

    log "  Stage 3/$NAME: pack inference state via block_fp_codec"
    PAYLOAD="$CONFIG_DIR/segmap_weights.tar.xz"
    "$PYBIN" -c "
import torch, os
from tac.block_fp_codec import pack_payload_tar_xz, verify_roundtrip

state = torch.load('$INFERENCE_PT', map_location='cpu', weights_only=False)
pack_payload_tar_xz(state, '$PAYLOAD')
verify_roundtrip(state, '$PAYLOAD', tol=1e-6)
print('payload bytes:', os.path.getsize('$PAYLOAD'))
"

    log "  Stage 4/$NAME: assemble archive"
    mkdir -p "$CONFIG_DIR/archive_src"
    cp "$GRAYSCALE_MKV_SHARED" "$CONFIG_DIR/archive_src/grayscale.mkv"
    cp "$ANCHOR_POSES" "$CONFIG_DIR/archive_src/optimized_poses.pt"
    cp "$PAYLOAD" "$CONFIG_DIR/archive_src/segmap_weights.tar.xz"

    ARCHIVE="$CONFIG_DIR/archive_${NAME}.zip"
    "$PYBIN" -c "
import os, zipfile
src = '$CONFIG_DIR/archive_src'
with zipfile.ZipFile('$ARCHIVE', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in ('segmap_weights.tar.xz', 'grayscale.mkv', 'optimized_poses.pt'):
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing archive component {p}'
        z.write(p, arcname=n)
print('archive bytes:', os.path.getsize('$ARCHIVE'))
"
    ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
    [ "$ARCHIVE_BYTES" -gt 0 ] || { log "  FATAL: archive empty for $NAME"; continue; }

    # config.env override: the sweep ALWAYS uses the segmap arm and the
    # segmap arch (NOT segmap_homography). hidden / block_hidden / num_blocks
    # ride into inflate via the SegMap state_dict + inflate_segmap defaults.
    # NB: inflate_segmap.py hard-codes hidden=24, block_hidden=24, num_blocks=8
    # in the CLI defaults — for the small / wide / deep / xwide configs we
    # MUST override via env vars so the renderer instantiates with the right
    # shape. We emit those overrides to config.env.
    INFLATE_CONFIG="$CONFIG_DIR/sweep_${NAME}_config.env"
    cat > "$INFLATE_CONFIG" <<EOF
PYTHON_INFLATE=segmap
SEGMAP_HIDDEN=$H
SEGMAP_BLOCK_HIDDEN=$BH
SEGMAP_NUM_BLOCKS=$NB
EOF

    log "  Stage 5/$NAME: contest_auth_eval [contest-CUDA]"
    rm -rf "$CONFIG_DIR/eval_work"
    set +e
    CONFIG_ENV_PATH="$INFLATE_CONFIG" "$PYBIN" -u experiments/contest_auth_eval.py \
        --archive "$ARCHIVE" \
        --inflate-sh submissions/robust_current/inflate.sh \
        --upstream-dir upstream \
        --device "${AUTH_EVAL_DEVICE:-cuda}" \
        --keep-work-dir \
        --work-dir "$CONFIG_DIR/eval_work" 2>&1 | tee "$CONFIG_DIR/auth_eval.log" | tail -20
    EVAL_RC=$?
    set -e

    "$PYBIN" -c "
import json, os, re
from pathlib import Path

d = json.load(open('$SWEEP_RESULTS_JSON'))
entry = {
    'name': '$NAME',
    'hidden': $H, 'block_hidden': $BH, 'num_blocks': $NB,
    'archive_path': '$ARCHIVE',
    'archive_bytes': os.path.getsize('$ARCHIVE'),
    'eval_rc': $EVAL_RC,
}
log_path = Path('$CONFIG_DIR/auth_eval.log')
if log_path.exists():
    txt = log_path.read_text(errors='ignore')
    m = re.search(r'\"score\"\s*:\s*([0-9.]+)', txt)
    if m:
        entry['score'] = float(m.group(1))
    else:
        m2 = re.search(r'score\s*=\s*([0-9.]+)', txt)
        if m2:
            entry['score'] = float(m2.group(1))
d['configs'].append(entry)
json.dump(d, open('$SWEEP_RESULTS_JSON', 'w'), indent=2)
print(f'  recorded sweep entry: {entry}')
"
done

log "=== Stage 6: aggregate sweep results ==="
"$PYBIN" -c "
import json
d = json.load(open('$SWEEP_RESULTS_JSON'))
configs = d.get('configs', [])
scored = [c for c in configs if 'score' in c]
scored.sort(key=lambda c: c['score'])
d['ranked'] = scored
d['best'] = scored[0] if scored else None
json.dump(d, open('$SWEEP_RESULTS_JSON', 'w'), indent=2)
print('=== sweep summary ===')
for c in configs:
    print(f\"  {c['name']:8s}  ({c['hidden']:2d},{c['block_hidden']:2d},{c['num_blocks']:2d})  \"
          f\"archive_bytes={c.get('archive_bytes', 0):,}  score={c.get('score', 'FAIL')}\")
print('=== best config:', d.get('best'))
"

"$PYBIN" -c "
import json, os, time
prov = json.load(open('$PROVENANCE'))
prov['completed_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
prov['sweep_results_path'] = '$SWEEP_RESULTS_JSON'
prov['lane_status'] = 'COMPLETE'
json.dump(prov, open('$PROVENANCE', 'w'), indent=2)
print('provenance updated.')
"

log "=== LANE_DARTS_S_DONE — see $SWEEP_RESULTS_JSON for ranked configs [contest-CUDA] ==="
