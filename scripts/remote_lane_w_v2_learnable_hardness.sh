#!/bin/bash
# Lane W-V2: LEARNABLE per-pair loss weights via Lagrangian dual ascent.
#
# Council 2026-04-27: Lane W-V1 used a hard top-K + uniform weight
# (K=30, weight=5.0) heuristic to up-weight the hardest pairs. Per the
# user's water-fill insight (memory: project_arbitrary_vs_learnable_taxonomy)
# this is the same anti-pattern as Lane Ω-V1's water-fill: a closed-form
# heuristic where a learnable parameter would be mathematically optimal.
#
# Lane W-V2 replaces top-K with a CONTINUOUS per-pair learnable weight
# vector. Math:
#
#   weights_i = softplus(raw_i)
#   loss      = sum_i weights_i * pair_loss_i
#             + λ · (sum_i weights_i - N_pairs)²    # rate Lagrangian
#
# The profiler's contribution-proportional output (--mode continuous) is
# the WARM-START. The continuous parameter then adapts during training:
# softplus is smooth → SGD converges where hard top-K cannot (top-K is
# piecewise constant; gradient zero almost everywhere).
#
# Predicted band: [0.85, 1.05] [contest-CUDA] vs Lane W-V1's [0.85, 1.10]
# (council Yousfi+Fridrich+Hotz: tighter because Lagrangian convergence is
# more reliable than the static top-K cliff, and the continuous parameter
# can re-allocate as training shifts the per-pair contribution rankings).
#
# Pipeline:
#   Stage 0 — NVDEC probe (catches bad-host in 5s).
#   Stage 1 — profile_pair_sensitivity --mode continuous → warm-start tensor.
#   Stage 2 — train_renderer.py SC + --learnable-pair-weights (dual ascent).
#   Stage 3 — SCv1 export of best fp32 checkpoint.
#   Stage 4 — Build archive (V2 renderer + Lane A masks + Lane A poses).
#   Stage 5 — contest_auth_eval [contest-CUDA].
#
# Anchored artifacts (Lane A's 1.15 frontier):
#   * renderer.bin: experiments/results/lane_a_landed/iter_0/renderer.bin
#   * masks.mkv:    experiments/results/lane_a_landed/iter_0/masks.mkv
#   * poses.pt:     experiments/results/lane_a_landed/iter_0/optimized_poses.pt
#
# Council preflight (CLAUDE.md non-negotiable: NEVER invent CLI flags).
# Verified 2026-04-27 against argparse:
#   * profile_pair_sensitivity.py: --checkpoint --poses --masks-mkv
#     --video-mkv --output --top-k --hard-weight --device --upstream
#     --batch-size --mode --continuous-normalize --no-continuous-normalize
#   * train_renderer.py: --tag --output-dir --device --resume-from
#     --use-self-compress-codec --pair-loss-weights --epochs --lr
#     --no-auth-eval-on-best --learnable-pair-weights
#     --learnable-pair-weights-lr --learnable-pair-weights-rate-lambda
#   * contest_auth_eval.py: --archive --inflate-sh --upstream-dir --device
#     --keep-work-dir --work-dir
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234

LOG_DIR="$WORKSPACE/lane_w_v2_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-w-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md feedback_canonical_remote_bootstraps).
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
    'lane_script': 'scripts/remote_lane_w_v2_learnable_hardness.sh',
    'lane_name': 'lane_w_v2_learnable_pair_weights',
    'anchor_renderer': 'experiments/results/lane_a_landed/iter_0/renderer.bin',
    'anchor_poses': 'experiments/results/lane_a_landed/iter_0/optimized_poses.pt',
    'anchor_masks': 'experiments/results/lane_a_landed/iter_0/masks.mkv',
    'anchor_score_baseline': 1.15,
    'predicted_band': [0.85, 1.05],
    'rationale': 'Continuous per-pair learnable weight with Lagrangian rate penalty; warm-started from continuous-mode profiler. Replaces Lane W-V1 hard top-K + uniform-weight heuristic.',
    'output_dir': '$LOG_DIR',
    'profile_mode': 'continuous',
    'top_k': 30,
    'hard_weight': 5.0,
    'sc_codec': True,
    'learnable_pair_weights': True,
    'learnable_pair_weights_lr': 1e-3,
    'learnable_pair_weights_rate_lambda': 1e-4,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=W-V2 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# ── Stage 0: NVDEC probe ──
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

# ── Stage 0b: anchor sanity ──
ANCHOR_RENDERER="experiments/results/lane_a_landed/iter_0/renderer.bin"
ANCHOR_POSES="experiments/results/lane_a_landed/iter_0/optimized_poses.pt"
ANCHOR_MASKS="experiments/results/lane_a_landed/iter_0/masks.mkv"
GT_VIDEO="upstream/videos/0.mkv"
for f in "$ANCHOR_RENDERER" "$ANCHOR_POSES" "$ANCHOR_MASKS" "$GT_VIDEO" \
         upstream/models/posenet.safetensors upstream/models/segnet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
log "  anchor_renderer: $ANCHOR_RENDERER ($(stat -c '%s' "$ANCHOR_RENDERER") bytes)"
log "  anchor_poses:    $ANCHOR_POSES   ($(stat -c '%s' "$ANCHOR_POSES") bytes)"
log "  anchor_masks:    $ANCHOR_MASKS   ($(stat -c '%s' "$ANCHOR_MASKS") bytes)"

# ── Stage 1: per-pair sensitivity profile in CONTINUOUS mode ──
log "=== Stage 1: profile_pair_sensitivity --mode continuous (Lane W-V2) ==="
PAIR_WEIGHTS="$LOG_DIR/pair_weights_continuous.pt"
"$PYBIN" -u experiments/profile_pair_sensitivity.py \
    --checkpoint "$ANCHOR_RENDERER" \
    --poses "$ANCHOR_POSES" \
    --masks-mkv "$ANCHOR_MASKS" \
    --video-mkv "$GT_VIDEO" \
    --upstream upstream \
    --output "$PAIR_WEIGHTS" \
    --top-k 30 \
    --hard-weight 5.0 \
    --mode continuous \
    --device cuda 2>&1 | tee "$LOG_DIR/profile.log" | tail -40
[ -f "$PAIR_WEIGHTS" ] || { echo "FATAL: profile_pair_sensitivity didn't produce $PAIR_WEIGHTS"; exit 2; }
log "  pair_weights: $PAIR_WEIGHTS ($(stat -c '%s' "$PAIR_WEIGHTS") bytes)"

# ── Stage 2: SC training + --learnable-pair-weights (Lane W-V2 core) ──
log "=== Stage 2: train_renderer.py SC + LearnablePairWeights ==="
TRAIN_OUT="$LOG_DIR/train"
mkdir -p "$TRAIN_OUT"
"$PYBIN" -u -m tac.experiments.train_renderer \
    --tag lane_w_v2_learnable \
    --output-dir "$TRAIN_OUT" \
    --device cuda \
    --resume-from "$ANCHOR_RENDERER" \
    --use-self-compress-codec \
    --pair-loss-weights "$PAIR_WEIGHTS" \
    --learnable-pair-weights \
    --learnable-pair-weights-lr 1e-3 \
    --learnable-pair-weights-rate-lambda 1e-4 \
    --epochs 500 \
    --lr 5e-5 \
    --no-auth-eval-on-best 2>&1 | tee "$LOG_DIR/train.log" | tail -40

BEST_FP32=$(ls -t "$TRAIN_OUT"/renderer_*_best_fp32.pt 2>/dev/null | head -1)
[ -n "$BEST_FP32" ] && [ -f "$BEST_FP32" ] || {
    echo "FATAL: training produced no renderer_*_best_fp32.pt in $TRAIN_OUT"
    ls -la "$TRAIN_OUT/" || true
    exit 2
}
log "  best fp32 checkpoint: $BEST_FP32 ($(stat -c '%s' "$BEST_FP32") bytes)"

# ── Stage 3: SCv1 export of best checkpoint ──
log "=== Stage 3: SCv1 export ==="
SCV1_BIN="$LOG_DIR/iter_0/renderer.bin"
mkdir -p "$LOG_DIR/iter_0"
"$PYBIN" -c "
import sys
sys.path.insert(0, 'src')
from pathlib import Path
import torch
from tac.renderer import AsymmetricPairGenerator
from tac.self_compress import swap_renderer_convs_with_self_compress
from tac.renderer_export import export_self_compressed_renderer

ckpt = '$BEST_FP32'
out = Path('$SCV1_BIN')
state = torch.load(ckpt, map_location='cpu', weights_only=False)
sd = state.get('model_state_dict') or state.get('state_dict') or state
meta = state.get('__meta__', {}) or {}

model = AsymmetricPairGenerator(
    num_classes=meta.get('num_classes', 5),
    embed_dim=meta.get('embed_dim', 6),
    base_ch=meta.get('base_ch', 36),
    mid_ch=meta.get('mid_ch', 60),
    motion_hidden=meta.get('motion_hidden', 32),
    depth=meta.get('depth', 1),
    pose_dim=meta.get('pose_dim', 0),
    use_dsconv=meta.get('use_dsconv', False),
    padding_mode=meta.get('padding_mode', 'zeros'),
    use_dilation=meta.get('use_dilation', False),
    use_zoom_flow=meta.get('use_zoom_flow', False),
)
diag = swap_renderer_convs_with_self_compress(model, init_bits=8.0)
print(f'swapped {len(diag[\"swapped\"])} layers, protected {len(diag[\"protected\"])}')
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'load_state_dict: missing={len(missing)} unexpected={len(unexpected)}')
n = export_self_compressed_renderer(model, out, use_lzma=True)
print(f'SCv1 exported {n} bytes to {out}')
" 2>&1 | tee "$LOG_DIR/export.log" | tail -15
[ -f "$SCV1_BIN" ] || { echo "FATAL: SCv1 export produced no $SCV1_BIN"; exit 2; }
log "  SCv1 binary: $SCV1_BIN ($(stat -c '%s' "$SCV1_BIN") bytes)"

# ── Stage 4: build archive (V2 renderer + Lane A masks + Lane A poses) ──
log "=== Stage 4: build archive ==="
cp "$ANCHOR_MASKS" "$LOG_DIR/iter_0/masks.mkv"
cp "$ANCHOR_POSES" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_w_v2.zip"
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
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE")
[ -n "$ARCHIVE_BYTES" ] && [ "$ARCHIVE_BYTES" -gt 0 ] || {
    echo "FATAL: archive build failed (zero bytes)"
    exit 2
}
log "  archive: $ARCHIVE ($ARCHIVE_BYTES bytes)"

# ── Stage 5: contest_auth_eval (the ONLY trustworthy score) ──
log "=== Stage 5: contest_auth_eval ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20

# Run record (canonical; consumed by remote_pull + monitor scripts).
"$PYBIN" -c "
import json, os, time
rec = {
    'lane': 'W-V2',
    'finished_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'archive': '$ARCHIVE',
    'archive_bytes': $ARCHIVE_BYTES,
    'pair_weights': '$PAIR_WEIGHTS',
    'best_fp32_checkpoint': '$BEST_FP32',
    'scv1_binary': '$SCV1_BIN',
    'predicted_band': [0.85, 1.05],
    'anchor_score': 1.15,
}
with open('$LOG_DIR/run_record.json', 'w') as f:
    json.dump(rec, f, indent=2)
print('run_record:', json.dumps(rec))
"

log "=== LANE_W_V2_DONE ==="
