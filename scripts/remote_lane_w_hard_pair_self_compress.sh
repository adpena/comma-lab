#!/bin/bash
# Lane W: Hard-Pair-Weighted Self-Compression.
#
# Anchors on Lane A (1.15 [contest-CUDA]) — the current frontier — and
# combines two memory-grounded ideas the council just synthesised:
#
#   1. feedback_overfit_is_the_goal + feedback_posenet_tracking — the
#      contest is a single 1200-frame video. Average distortion hides
#      the heavy tail; per-pair tracking is the only correct view.
#
#   2. project_council_shower_thoughts + feedback_curriculum_must_use_full_score
#      — a hard-pair curriculum that uses the FULL score formula
#      (100*seg + sqrt(10*pose) + 25*rate) lets the optimiser spend
#      gradient where it actually moves the score.
#
# Combined with Lane S (Self-Compression, commit 0396228c): we pre-compute
# per-pair PoseNet+SegNet contribution using Lane A's renderer + poses,
# identify the top-K hardest pairs, and weight those pairs 5x in the SC
# training loss. Per-pair scaling multiplies BOTH the scorer loss AND the
# Lagrangian rate penalty, so the per-channel learnable bit-depth
# allocation is steered to protect the worst-pair channels.
#
# Predicted band (council-set): [0.85, 1.10] [contest-CUDA] — beats Lane A
# 1.15 if the heavy-tail hypothesis is correct.
#
# Bug-class compliance (CLAUDE.md non-negotiables):
#   - set -euo pipefail (NOT -uo): required by feedback_zip_dep_bootstrap_trap
#   - python zipfile (NOT shell zip binary): required by same memory
#   - --device cuda everywhere (NO MPS fallback): feedback_mps_cuda_drift_critical
#   - NVDEC probe BEFORE GPU spend: feedback_vastai_nvdec_host_variation
#   - provenance.json + heartbeat.log + run_record.json: feedback_canonical_remote_bootstraps
#   - argparse-grep verified flags only: feedback_dead_flag_wiring_pattern
#   - container python /opt/conda/bin/python (NOT venv): feedback_canonical_remote_bootstraps

set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234

LOG_DIR="$WORKSPACE/lane_w_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-w] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat
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
    'lane_script': 'scripts/remote_lane_w_hard_pair_self_compress.sh',
    'lane_name': 'lane_w_hard_pair',
    'anchor_renderer': 'experiments/results/lane_a_landed/iter_0/renderer.bin',
    'anchor_poses': 'experiments/results/lane_a_landed/iter_0/optimized_poses.pt',
    'anchor_masks': 'experiments/results/lane_a_landed/iter_0/masks.mkv',
    'anchor_score_baseline': 1.15,
    'predicted_band': [0.85, 1.10],
    'output_dir': '$LOG_DIR',
    'top_k': 30,
    'hard_weight': 5.0,
    'sc_codec': True,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=W gpu=$GPU" >> "$HEARTBEAT"
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

# ── Stage 1: per-pair sensitivity profile (Lane A frontier) ──
log "=== Stage 1: profile_pair_sensitivity (Lane A renderer + poses + masks) ==="
PAIR_WEIGHTS="$LOG_DIR/pair_weights.pt"
"$PYBIN" -u experiments/profile_pair_sensitivity.py \
    --checkpoint "$ANCHOR_RENDERER" \
    --poses "$ANCHOR_POSES" \
    --masks-mkv "$ANCHOR_MASKS" \
    --video-mkv "$GT_VIDEO" \
    --upstream upstream \
    --output "$PAIR_WEIGHTS" \
    --top-k 30 \
    --hard-weight 5.0 \
    --device cuda 2>&1 | tee "$LOG_DIR/profile.log" | tail -40
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi
[ -f "$PAIR_WEIGHTS" ] || { echo "FATAL: profile_pair_sensitivity didn't produce $PAIR_WEIGHTS"; exit 2; }
log "  pair_weights: $PAIR_WEIGHTS ($(stat -c '%s' "$PAIR_WEIGHTS") bytes)"

# ── Stage 2: SC training resumed from Lane A renderer ──
# Argparse-verified flags (every one grepped against train_renderer.py):
#   --use-self-compress-codec      (line 326)
#   --pair-loss-weights            (NEW Lane W flag, this commit)
#   --resume-from                  (line 420)
#   --epochs                       (line 186)
#   --lr                           (line 187)
#   --no-auth-eval-on-best         (line 458)
#   --tag --output-dir --device    (lines 441/442/444)
#   --upstream-dir is NOT a flag — train_renderer uses --auth-eval-upstream-dir
#
# CLAUDE.md "Auth eval EVERYWHERE" non-negotiable: train_renderer ends with
# auth eval on best (default-true). We DISABLE it here because we run our
# own contest_auth_eval at Stage 5 against the BUILT archive (which has the
# correct rate term). The default in-script eval would build a wrong
# archive (renderer-only or train-time poses), per feedback_phantom_baseline_pattern.
log "=== Stage 2: train_renderer.py SC + per-pair weighted ==="
TRAIN_OUT="$LOG_DIR/train"
mkdir -p "$TRAIN_OUT"
# --resume-from removed: ANCHOR_RENDERER is the quantized .bin (ASYM magic),
# not a float checkpoint. train_renderer rejects quantized binaries (Round 11
# Finding 1). Train from scratch with self-compress codec + per-pair weights.
"$PYBIN" -u -m tac.experiments.train_renderer \
    --tag lane_w_hard_pair \
    --output-dir "$TRAIN_OUT" \
    --device cuda \
    --use-self-compress-codec \
    --pair-loss-weights "$PAIR_WEIGHTS" \
    --epochs 500 \
    --lr 5e-5 \
    --no-auth-eval-on-best 2>&1 | tee "$LOG_DIR/train.log" | tail -40
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

# Train script saves renderer_<tag>_best_fp32.pt (full state_dict — preserves
# the SC per-channel weights) AND a renderer_<tag>_best_fp4.pt (which would
# CORRUPT SC weights — DO NOT use for SCv1 export). Use the fp32 .pt.
BEST_FP32=$(ls -t "$TRAIN_OUT"/renderer_*_best_fp32.pt 2>/dev/null | head -1)
[ -n "$BEST_FP32" ] && [ -f "$BEST_FP32" ] || {
    echo "FATAL: training produced no renderer_*_best_fp32.pt in $TRAIN_OUT"
    ls -la "$TRAIN_OUT/" || true
    exit 2
}
log "  best fp32 checkpoint: $BEST_FP32 ($(stat -c '%s' "$BEST_FP32") bytes)"

# ── Stage 3: SCv1 export of best checkpoint ──
# Uses tac.renderer_export.export_self_compressed_renderer which writes
# the SCv1 magic-byte format. Reconstructs the model from the fp32 .pt
# `__meta__` arch dict, swaps in SelfCompressingConv2d, loads state_dict,
# then packs at the LEARNED per-channel bit-depth.
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
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi
[ -f "$SCV1_BIN" ] || { echo "FATAL: SCv1 export produced no $SCV1_BIN"; exit 2; }
log "  SCv1 binary: $SCV1_BIN ($(stat -c '%s' "$SCV1_BIN") bytes)"

# ── Stage 4: build archive (SCv1 renderer + Lane A masks + Lane A poses) ──
log "=== Stage 4: build archive ==="
cp "$ANCHOR_MASKS" "$LOG_DIR/iter_0/masks.mkv"
cp "$ANCHOR_POSES" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_w.zip"
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
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

# Run record (canonical; consumed by remote_pull + monitor scripts).
"$PYBIN" -c "
import json, os, time
rec = {
    'lane': 'W',
    'finished_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'archive': '$ARCHIVE',
    'archive_bytes': $ARCHIVE_BYTES,
    'pair_weights': '$PAIR_WEIGHTS',
    'best_fp32_checkpoint': '$BEST_FP32',
    'scv1_binary': '$SCV1_BIN',
    'predicted_band': [0.85, 1.10],
    'anchor_score': 1.15,
}
with open('$LOG_DIR/run_record.json', 'w') as f:
    json.dump(rec, f, indent=2)
print('run_record:', json.dumps(rec))
"

log "=== LANE_W_DONE ==="
