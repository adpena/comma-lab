#!/bin/bash
# Lane F-V4 (mixed-precision FP4 QAT + per-layer sensitivity profiling).
# Anchored on Lane A 1.15 [contest-CUDA] (NOT baseline 2.29). V4 closes
# the FP4 chapter for real:
#   V1 → 2.73 (silent zero-pose bug)
#   V2 → 1.79 (uniform FP4 → 20x PoseNet penalty: FastViT YUV6 path is
#               sensitive to a small set of "critical" layers)
#   V3 → 1.85 (INT8 warmup helped PoseNet -17% but hurt SegNet +38%)
#   V4 → predicted [1.20, 1.50] [contest-CUDA] — instead of guessing
#         which layers can tolerate FP4, MEASURE per-layer sensitivity
#         empirically, then keep the top 30% (by params) at FP16 and
#         FP4 the bulk 70%. Distortion target: PoseNet ≤ 0.020 (vs
#         uniform FP4's 0.10 = 5x better). Rate target: 40-50% size
#         reduction (vs uniform FP4's 60%) — net rate stays competitive
#         while distortion stays close to FP32.
#
# Council justification (Yousfi + Fridrich + Hotz, 2026-04-28):
#   Uniform FP4 cost us 0.20-0.64pts on Lane F because dilated-h64
#   ASYM has a small set of layers (likely renderer.head + the FiLM
#   path + 1x1 fuse_convs) whose YUV statistics are load-bearing for
#   FastViT-T12 attention. The empirical profile identifies them
#   layer-by-layer. SC_PROTECTED_NAME_PATTERNS in self_compress.py is
#   a strong prior, but Lane F-V4 verifies it (and may surprise us).
#
# Three deltas vs Lane F-V3:
#   1. NEW Stage 1: profile_fp4_layer_sensitivity.py (~5 min on 4090)
#      builds layer_sensitivity.pt from Lane A renderer + masks + poses.
#   2. Stage 2 qat_finetune.py adds --mixed-precision-from-sensitivity
#      and --mixed-precision-target-rate 0.70 (Lagrangian-style knob).
#   3. KL distill aux on SegNet logits (T=2.0, weight=0.002) — Quantizr
#      recipe to recover SegNet headroom that V3 lost (+38% regression).
#
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234

LOG_DIR="$WORKSPACE/lane_f_v4_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-f-v4] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md canonical pipeline standard +
# memory feedback_canonical_remote_bootstraps): every remote run must
# emit provenance.json so a fresh agent can reconstruct the experiment.
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
    'lane_script': 'scripts/remote_lane_f_v4_mixed_precision_fp4.sh',
    'lane_name': 'lane_f_v4_mixed_precision_fp4_on_lane_a',
    'anchor_renderer': 'experiments/results/lane_a_landed/iter_0/renderer.bin',
    'anchor_poses': 'experiments/results/lane_a_landed/optimized_poses.pt',
    'anchor_masks': 'experiments/results/lane_a_landed/extracted/masks.mkv',
    'anchor_score_baseline': 1.15,
    'predicted_band': [1.20, 1.50],
    'delta_from_v3': 'mixed_precision_from_sensitivity + KL_distill_aux + INT8_warmup_kept',
    'mixed_precision_target_rate': 0.70,
    'mixed_precision_bulk_bits': 4,
    'mixed_precision_critical_bits': 16,
    'kl_distill_temperature': 2.0,
    'kl_distill_weight': 0.002,
    'int8_warmup_epochs': 50,
    'fp4_epochs': 500,
    'lr': 2.5e-6,
    'output_dir': '$LOG_DIR',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=F-V4 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0 (per preflight check 33): NVDEC probe BEFORE any GPU spend.
# Catches bad-host case in 5 seconds. Reference: feedback_vastai_nvdec_host_variation.
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

# Pre-flight: anchor on Lane A (1.15 [contest-CUDA]), NOT baseline 2.29.
ANCHOR_RENDERER="experiments/results/lane_a_landed/iter_0/renderer.bin"
ANCHOR_POSES="experiments/results/lane_a_landed/optimized_poses.pt"
ANCHOR_MASKS="experiments/results/lane_a_landed/extracted/masks.mkv"
for f in "$ANCHOR_RENDERER" \
         "$ANCHOR_POSES" \
         "$ANCHOR_MASKS" \
         upstream/videos/0.mkv \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
log "  anchor_renderer: $ANCHOR_RENDERER"
log "  anchor_poses:    $ANCHOR_POSES"
log "  anchor_masks:    $ANCHOR_MASKS"

# Pre-flight: shape-validate anchor renderer matches the QAT model build
# (per preflight check 34). The Lane F-V4 anchor is Lane A's renderer.bin
# which has known arch (base_ch=36, mid_ch=60, motion_hidden=32, depth=1,
# embed_dim=6, pose_dim=6, use_zoom_flow=True, padding_mode=zeros). We
# load it via load_any_renderer_checkpoint and validate state_dict shape
# against build_renderer with the same params. Cost: ~5 seconds. Saves
# 5 minutes + $0.05 per shape-mismatch crash (Lane S motion.head incident).
log "=== Stage 0b: resume-from shape validation ==="
"$PYBIN" -c "
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'upstream')
from tac.renderer_export import load_any_renderer_checkpoint
from tac.renderer import build_renderer
loaded = load_any_renderer_checkpoint('$ANCHOR_RENDERER', device='cpu')
expected = build_renderer(
    num_classes=5, embed_dim=6, base_ch=36, mid_ch=60,
    motion_hidden=32, depth=1, pose_dim=6, use_dsconv=False,
    padding_mode='zeros', use_dilation=False, use_zoom_flow=True,
)
loaded_keys = set(loaded.state_dict().keys())
expected_keys = set(expected.state_dict().keys())
missing = expected_keys - loaded_keys
extra = loaded_keys - expected_keys
if missing or extra:
    raise SystemExit(
        f'FATAL: anchor renderer shape mismatch.\n'
        f'  missing keys (expected by QAT, absent in anchor): {sorted(missing)[:5]}\n'
        f'  extra keys (in anchor, not in QAT model): {sorted(extra)[:5]}\n'
        f'  Anchor: $ANCHOR_RENDERER'
    )
# Also shape-validate every shared key.
shape_mismatches = []
for k in loaded_keys & expected_keys:
    if loaded.state_dict()[k].shape != expected.state_dict()[k].shape:
        shape_mismatches.append((k, tuple(loaded.state_dict()[k].shape),
                                 tuple(expected.state_dict()[k].shape)))
if shape_mismatches:
    raise SystemExit(
        f'FATAL: anchor renderer per-tensor shape mismatch:\n  '
        + '\n  '.join(f'{k}: anchor={a} vs expected={e}'
                      for k, a, e in shape_mismatches[:5])
    )
print(f'[shape-validate] OK: anchor matches QAT model ({len(loaded_keys)} keys)')
"

log "=== Stage 1: stage Lane A masks (no rebuild — we anchor on Lane A's exact bytes) ==="
mkdir -p "$LOG_DIR/extracted"
cp "$ANCHOR_MASKS" "$LOG_DIR/extracted/masks.mkv"
log "  staged $(stat -c '%s' "$LOG_DIR/extracted/masks.mkv") bytes of Lane A masks"

log "=== Stage 2: per-layer FP4 sensitivity profile (Phase 1) ==="
log "  output: $LOG_DIR/layer_sensitivity.pt"
log "  ~5 min on RTX 4090 (~80 layers * 30 pairs * ~1.5s/pair)"
SENSITIVITY="$LOG_DIR/layer_sensitivity.pt"
"$PYBIN" -u experiments/profile_fp4_layer_sensitivity.py \
    --checkpoint "$ANCHOR_RENDERER" \
    --video upstream/videos/0.mkv \
    --masks-mkv "$ANCHOR_MASKS" \
    --poses "$ANCHOR_POSES" \
    --upstream upstream \
    --output "$SENSITIVITY" \
    --device cuda \
    --n-pairs 30 \
    --predicted-band 1.20 1.50 2>&1 | tee "$LOG_DIR/profile.log" | tail -40

[ -f "$SENSITIVITY" ] || { echo "FATAL: sensitivity profile not produced"; exit 2; }
log "  sensitivity: $SENSITIVITY ($(stat -c '%s' "$SENSITIVITY") bytes)"

log "=== Stage 3: FP4 QAT fine-tune w/ mixed-precision allocation (Phase 2 + 3) ==="
log "  input: $ANCHOR_RENDERER (FP32 ASYM, 290KB)"
log "  poses: $ANCHOR_POSES (Lane A optimized_poses.pt)"
log "  sensitivity: $SENSITIVITY (Stage 2 output)"
log "  output: $LOG_DIR/qat/renderer_fp4.bin"
log "  schedule: 50 INT8 warmup + 500 FP4 (V3+V4 default)"
log "  lr: 2.5e-6 (V3 default; same as V4)"
log "  mixed-precision target_rate: 0.70 (bulk 70% FP4, critical 30% FP16)"
"$PYBIN" -u experiments/qat_finetune.py \
    --checkpoint "$ANCHOR_RENDERER" \
    --poses "$ANCHOR_POSES" \
    --upstream upstream \
    --output-dir "$LOG_DIR/qat" \
    --device cuda \
    --base-ch 36 --mid-ch 60 --pose-dim 6 --motion-hidden 32 --depth 1 --embed-dim 6 \
    --use-zoom-flow --padding-mode zeros \
    --int8-warmup-epochs 50 \
    --fp4-epochs 500 \
    --lr 2.5e-6 \
    --batch-size 4 \
    --mixed-precision-from-sensitivity "$SENSITIVITY" \
    --mixed-precision-target-rate 0.70 \
    --mixed-precision-bulk-bits 4 \
    --mixed-precision-critical-bits 16 2>&1 | tee "$LOG_DIR/qat.log" | tail -30

# qat_finetune saves renderer_fp4.bin in output-dir per its convention.
FP4_BIN=$(find "$LOG_DIR/qat" -name "renderer_fp4.bin" -o -name "*_fp4*.bin" 2>/dev/null | head -1)
[ -n "$FP4_BIN" ] && [ -f "$FP4_BIN" ] || { echo "FATAL: qat_finetune didn't produce FP4 binary"; ls -la "$LOG_DIR/qat/"; exit 2; }
FP4_SIZE=$(stat -c '%s' "$FP4_BIN")
log "  FP4 binary: $FP4_BIN ($FP4_SIZE bytes vs FP32 290KB original)"

log "=== Stage 4: build NEW archive (FP4 renderer + Lane A masks + Lane A poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$FP4_BIN" "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
cp "$ANCHOR_POSES" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_f_v4.zip"
"$PYBIN" -c "
import zipfile, os
src = '$LOG_DIR/iter_0'
dst = '$ARCHIVE'
# Use ZipInfo + writestr with fixed timestamp for deterministic bytes
# (per preflight check_archive_builders_use_deterministic_zip).
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in ('renderer.bin', 'masks.mkv', 'optimized_poses.pt'):
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing {p}'
        zi = zipfile.ZipInfo(filename=n, date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        with open(p, 'rb') as fh:
            z.writestr(zi, fh.read())
print(f'archive {dst}: {os.path.getsize(dst)} bytes')
"

log "=== Stage 5: contest_auth_eval on Lane F-V4 archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -15

log "=== LANE_F_V4_DONE [contest-CUDA] ==="
