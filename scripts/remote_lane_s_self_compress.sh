#!/bin/bash
# Lane S: Self-Compressing renderer (Szabolcs 2301.13142) anchored on Lane A.
#
# Council 2026-04-27: Lane A landed at 1.15 [contest-CUDA] (pose 0.005 + seg
# 0.0046 + rate 0.0185 — distortion is essentially at the floor). The
# remaining wedge is rate (60% of the score). FP4-QAT (Lane F) tried to
# attack rate by uniform 4-bit weights but PoseNet exploded +58% (memory:
# project_lane_f_fp4_qat_regression_20260427). Lane S attacks rate via
# per-channel learnable bit-depth (init 8.0, target 2.5) WHILE protecting
# the PoseNet-sensitive layers (renderer.head, motion.head, FiLM linears,
# fuse_conv stay FP32 per SC_PROTECTED_NAME_PATTERNS). Predicted: SC
# preserves Lane A's distortion floor AND drops the renderer payload from
# ~290KB → ~16-20KB → score [0.85, 1.20] [contest-CUDA].
#
# Pipeline strategy:
#   1. Stage 0 NVDEC probe (5s sanity)
#   2. Stage 1 — load Lane A's ASYM renderer.bin → vanilla AsymmetricPairGenerator
#      → swap_renderer_convs_with_self_compress (copies weights into SC layers)
#      → save as lane_a_sc_init.pt. This warm-starts the SC training so the
#      Lagrangian penalty has good FP32 weights to compress, NOT random init.
#   3. Stage 2 — train_renderer.py --profile self_compress_renderer_full
#      with Lane A arch overrides (use_zoom_flow=False, mask_half_sim_prob=0.0)
#      and a SHORT fine-tune schedule (500 ep total: 150ep FP32 warmup +
#      350ep with Lagrangian rate ramp). LR 5e-5 to preserve Lane A quality.
#   4. Stage 3 — SCv1 export of best fp32 checkpoint to renderer_sc.bin.
#   5. Stage 4 — build archive (renderer_sc.bin + Lane A masks + Lane A poses).
#   6. Stage 5 — contest_auth_eval [contest-CUDA].
#
# Why short schedule:
#   We are NOT training from scratch (1980 ep). We resume FP32 weights from
#   Lane A and only need to (a) anneal the bit-depth and (b) fine-tune the
#   compressed weights to recover any quantization drift. 500 ep fine-tune
#   is the same budget Lane B used for its FP4-QAT short tune. Cost: ~1.5h
#   on 4090 = $0.40.
#
# Council preflight (CLAUDE.md non-negotiable: NEVER invent CLI flags):
#   * --use-self-compress-codec      ✓ argparse line 326
#   * --self-compress-init-bits      ✓ argparse line 329
#   * --self-compress-target-bits    ✓ argparse line 332
#   * --self-compress-lambda-start   ✓ argparse line 335
#   * --self-compress-lambda-end     ✓ argparse line 337
#   * --self-compress-lambda-ramp-start-frac ✓ argparse line 339
#   * --resume-from                  ✓ argparse line 420
#   * --no-auth-eval-on-best         ✓ argparse line 458 (auto-disabled by
#                                      train_renderer for SC mode anyway)
#   * --phase[1-5]-epochs            ✓ argparse lines 204-212
#   * --phase[1-5]-lr                ✓ argparse lines 214-222
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
# 2026-04-27 lesson (Lane D): train_renderer.py defaults its scorer-weight
# search to $REPO/workspace/upstream/.../models. The canonical Vast.ai
# layout puts the scorers at /workspace/pact/upstream/models. Force it.
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_s_results"
mkdir -p "$LOG_DIR"
TAG="lane_s_self_compress"

log() { echo "[lane-s] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md canonical pipeline standard +
# memory feedback_canonical_remote_bootstraps).
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
    'lane_script': 'scripts/remote_lane_s_self_compress.sh',
    'lane_name': 'lane_s_self_compress_on_lane_a',
    'profile': 'self_compress_renderer_full',
    'anchor_renderer': 'experiments/results/lane_a_landed/iter_0/renderer.bin',
    'anchor_poses': 'experiments/results/lane_a_landed/iter_0/optimized_poses.pt',
    'anchor_masks': 'experiments/results/lane_a_landed/iter_0/masks.mkv',
    'anchor_score_baseline': 1.15,
    'predicted_band': [0.85, 1.20],
    'rationale': 'SC per-channel bit-depth (init 8 → target 2.5) preserves Lane A distortion floor while collapsing rate from 0.0185 → ~0.005-0.007 via 290KB → 16-20KB renderer.',
    'sc_init_bits': 8.0,
    'sc_target_bits': 2.5,
    'sc_lambda_end': 1.0,
    'sc_lambda_ramp_start_frac': 0.3,
    'total_epochs': 500,
    'lr': 5e-5,
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=S gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0 (2026-04-27): NVDEC probe BEFORE any GPU spend. Catches bad-host
# in 5 seconds. Reference: feedback_vastai_nvdec_host_variation.
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

# Pre-flight: anchor on Lane A (1.15 [contest-CUDA]). Lane A artifacts
# committed to the repo at experiments/results/lane_a_landed/.
ANCHOR_RENDERER="experiments/results/lane_a_landed/iter_0/renderer.bin"
ANCHOR_POSES="experiments/results/lane_a_landed/iter_0/optimized_poses.pt"
ANCHOR_MASKS="experiments/results/lane_a_landed/iter_0/masks.mkv"
for f in "$ANCHOR_RENDERER" \
         "$ANCHOR_POSES" \
         "$ANCHOR_MASKS" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
log "  anchor_renderer: $ANCHOR_RENDERER ($(stat -c '%s' "$ANCHOR_RENDERER") bytes, ASYM FP32)"
log "  anchor_poses:    $ANCHOR_POSES"
log "  anchor_masks:    $ANCHOR_MASKS ($(stat -c '%s' "$ANCHOR_MASKS") bytes)"

# Pre-flight: profile validation (catches missing SC fields BEFORE GPU burn).
log "=== Pre-flight: profile validation ==="
"$PYBIN" -c "
import sys
sys.path.insert(0, 'src')
from tac.profiles import PROFILES
assert 'self_compress_renderer_full' in PROFILES, \
    'profile self_compress_renderer_full not registered'
p = PROFILES['self_compress_renderer_full']
for k in ('use_self_compress_codec', 'self_compress_init_bits',
          'self_compress_target_bits', 'self_compress_lambda_end',
          'self_compress_lambda_ramp_start_frac'):
    assert k in p, f'profile missing key: {k}'
assert p['use_self_compress_codec'] is True
assert p['self_compress_init_bits'] >= 6.0, 'init bits too low — would collapse Lane A'
print(f'PROFILE OK: init={p[\"self_compress_init_bits\"]} '
      f'target={p[\"self_compress_target_bits\"]} '
      f'lambda_end={p[\"self_compress_lambda_end\"]} '
      f'ramp_start={p[\"self_compress_lambda_ramp_start_frac\"]}')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Pre-flight: dead-flag-wiring guard — every CLI flag in the train_renderer
# invocation below must exist in train_renderer.py's argparse. Catches the
# class of bug CLAUDE.md's "NEVER invent CLI flags" rule was created for.
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
sys.path.insert(0, 'src')
script = open('scripts/remote_lane_s_self_compress.sh').read()
tr_src = open('src/tac/experiments/train_renderer.py').read()
real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', tr_src))
m = re.search(r'src/tac/experiments/train_renderer\.py(.*?)(?=\n# Stage 3:|\n\s*BEST_FP32=|\Z)',
              script, re.DOTALL)
assert m, 'could not locate train_renderer.py invocation in script'
used = set(re.findall(r'\B--([a-z][a-z0-9-]+)', m.group(0)))
invented = used - real
if invented:
    print(f'INVENTED FLAGS: {sorted(invented)} not in train_renderer argparse',
          file=sys.stderr); sys.exit(3)
print(f'OK: {len(used)} flags all real')
" 2>&1 | tee -a "$LOG_DIR/run.log"

log "=== Stage 1: warm-start SC init from Lane A's FP32 ASYM renderer ==="
# Build the same arch that Lane A's renderer.bin holds, copy its weights
# into a SelfCompressingConv2d-swapped model, save as a .pt that
# train_renderer's --resume-from can ingest. This is the cleanest way to
# bridge "Lane A's vanilla ASYM weights" → "SC-swapped layer keys" without
# touching the resume-from torch.load() path inside train_renderer.
RESUME_INIT_PT="$LOG_DIR/lane_a_sc_init.pt"
"$PYBIN" -c "
import sys, torch
sys.path.insert(0, 'src')
from tac.renderer_export import load_any_renderer_checkpoint
from tac.self_compress import (
    swap_renderer_convs_with_self_compress,
    list_self_compress_layers,
    renderer_average_bits_per_weight,
)
src_bin = '$ANCHOR_RENDERER'
out_pt = '$RESUME_INIT_PT'
print(f'[stage1] loading Lane A renderer: {src_bin}')
model = load_any_renderer_checkpoint(src_bin, device='cpu')
print(f'[stage1] loaded class={type(model).__name__}')
diag = swap_renderer_convs_with_self_compress(model, init_bits=8.0)
print(f'[stage1] SC swap: swapped={len(diag[\"swapped\"])} '
      f'protected={len(diag[\"protected\"])} skipped={len(diag[\"skipped\"])}')
print(f'[stage1] swapped layers (first 6): {diag[\"swapped\"][:6]}')
print(f'[stage1] protected layers: {diag[\"protected\"]}')
sc_layers = list_self_compress_layers(model)
print(f'[stage1] {len(sc_layers)} SC layers; init bits/weight = '
      f'{renderer_average_bits_per_weight(model):.2f} (expect 8.0)')
# Save as the format train_renderer.resume_from expects: a dict with
# either 'model' or 'model_state_dict' key. EMA shadow + optimizer +
# scheduler will be reinitialized fresh (this is a warm-start, not a resume).
torch.save({'model_state_dict': model.state_dict()}, out_pt)
import os
print(f'[stage1] wrote {out_pt}: {os.path.getsize(out_pt)} bytes')
" 2>&1 | tee -a "$LOG_DIR/run.log"
[ -f "$RESUME_INIT_PT" ] || { echo "FATAL: stage 1 didn't produce $RESUME_INIT_PT"; exit 2; }

log "=== Stage 2: SC fine-tune (500 ep) anchored on Lane A weights ==="
log "  profile:  self_compress_renderer_full (overridden phase epochs + LR + arch)"
log "  schedule: 150ep FP32 warmup (lambda=0) + 350ep with rate ramp → 500 total"
log "  lambda:   0 → 1.0 starting at 30% of training (epoch 150)"
log "  lr:       5e-5 across all phases (preserves Lane A quality)"
log "  arch:     base_ch=36 mid_ch=60 motion_hidden=32 embed_dim=6 depth=1"
log "            pose_dim=6 use_zoom_flow=False (matches Lane A's ASYM bin)"
log "  resume:   $RESUME_INIT_PT (Lane A weights pre-loaded into SC layers)"
log "  estimated wall clock on 4090: ~1.5h (\$0.40 at \$0.25/hr)"

# Smoke-kill metadata sidecar for external watchdog
cat > "$LOG_DIR/kill_targets.json" <<'EOF'
{
  "phase1_end_epoch": 150,
  "phase1_pixel_l1_max": 5.0,
  "phase2_smoke_epoch": 250,
  "phase2_smoke_scorer_max": 3.0,
  "phase4_end_epoch": 500,
  "phase4_end_scorer_max": 1.50,
  "comment": "Lane S kill targets — anchored on Lane A 1.15. If the SC fine-tune ends >1.50 it has degraded vs Lane A and should be aborted/recommitted with a higher target_bits."
}
EOF

# train_renderer.py invocation. We use --no-auth-eval-on-best because:
#   (a) SCv1 export needs a separate Python invocation with the trained model
#       in memory (Stage 3 below), not the FP4A path that auth_eval expects.
#   (b) train_renderer auto-disables auth-eval-on-best for SC mode anyway
#       (line 1098-1107 of train_renderer.py) — passing the flag explicitly
#       documents the intent and avoids a confusing WARN line in the log.
# Phase epoch overrides: collapse the profile's 1980 ep schedule to 500 ep
# (150+250+50+50+0). This matches what we want for a fine-tune-from-Lane-A
# rather than train-from-scratch.
"$PYBIN" -u src/tac/experiments/train_renderer.py \
    --profile self_compress_renderer_full \
    --tag "$TAG" \
    --device cuda \
    --video upstream/videos/0.mkv \
    --output-dir "$LOG_DIR/train" \
    --resume-from "$RESUME_INIT_PT" \
    --use-self-compress-codec \
    --self-compress-init-bits 8.0 \
    --self-compress-target-bits 2.5 \
    --self-compress-lambda-start 0.0 \
    --self-compress-lambda-end 1.0 \
    --self-compress-lambda-ramp-start-frac 0.3 \
    --phase1-epochs 150 \
    --phase2-epochs 250 \
    --phase3-epochs 50 \
    --phase4-epochs 50 \
    --phase5-epochs 0 \
    --phase1-lr 5e-5 \
    --phase2-lr 5e-5 \
    --phase3-lr 5e-5 \
    --phase4-lr 5e-5 \
    --phase5-lr 5e-5 \
    --no-auth-eval-on-best \
    2>&1 | tee "$LOG_DIR/train.log" | grep -E "^\[(train|eval|masks|phase|best|arch|stage1|sc)\]|epoch|Phase|scorer|bits/weight" | tail -200

# Stage 3: SCv1 export.
# train_renderer saves renderer_<tag>_best_fp32.pt with the SC-swapped
# state_dict (since SC swap happens at model build time). We rebuild the
# same arch + apply SC swap + load the fp32 weights + run
# export_self_compressed_renderer to produce the SCv1 binary.
BEST_FP32="$LOG_DIR/train/renderer_${TAG}_best_fp32.pt"
[ -f "$BEST_FP32" ] || {
    echo "FATAL: train_renderer didn't produce ${BEST_FP32}" >&2
    ls -la "$LOG_DIR/train/" >&2
    exit 2
}
log "  best fp32 checkpoint: $BEST_FP32 ($(stat -c '%s' "$BEST_FP32") bytes)"

log "=== Stage 3: SCv1 export of best fp32 → renderer_sc.bin ==="
mkdir -p "$LOG_DIR/qat"
SC_BIN="$LOG_DIR/qat/renderer_sc.bin"
"$PYBIN" -c "
import sys, torch
sys.path.insert(0, 'src')
from tac.renderer import build_renderer
from tac.self_compress import (
    swap_renderer_convs_with_self_compress,
    list_self_compress_layers,
    renderer_average_bits_per_weight,
)
from tac.renderer_export import export_self_compressed_renderer

ckpt_path = '$BEST_FP32'
out_bin = '$SC_BIN'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
meta = ckpt.get('__meta__', {}) or {}
print(f'[stage3] loaded fp32 checkpoint, meta keys: {sorted(meta.keys())[:8]}')

# Rebuild the EXACT arch the training run used. Defaults match Lane A.
def m(key, default):
    return meta.get(key, default)

model = build_renderer(
    embed_dim=m('embed_dim', 6),
    base_ch=m('base_ch', 36),
    mid_ch=m('mid_ch', 60),
    motion_hidden=m('motion_hidden', 32),
    depth=m('depth', 1),
    pose_dim=m('pose_dim', 6),
    use_dsconv=m('use_dsconv', False),
    padding_mode=m('padding_mode', 'zeros'),
    use_dilation=m('use_dilation', False),
    use_zoom_flow=m('use_zoom_flow', False),
)
diag = swap_renderer_convs_with_self_compress(model, init_bits=8.0)
print(f'[stage3] SC swap: {len(diag[\"swapped\"])} swapped, '
      f'{len(diag[\"protected\"])} protected')

missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    raise RuntimeError(
        f'SC load mismatch: missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}'
    )
print(f'[stage3] loaded fp32 weights into SC-swapped model')

avg_bits = renderer_average_bits_per_weight(model)
print(f'[stage3] learned bits/weight (mean): {avg_bits:.3f} (target 2.5)')

n_bytes = export_self_compressed_renderer(model, out_bin, use_lzma=True)
print(f'[stage3] WROTE {out_bin}: {n_bytes} bytes (vs Lane A 290KB)')
" 2>&1 | tee -a "$LOG_DIR/run.log"
[ -f "$SC_BIN" ] || { echo "FATAL: SCv1 export failed — no $SC_BIN" >&2; exit 2; }
SC_SIZE=$(stat -c '%s' "$SC_BIN")
log "  SCv1 binary: $SC_BIN ($SC_SIZE bytes)"

log "=== Stage 4: build NEW archive (SCv1 renderer + Lane A masks + Lane A poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$SC_BIN" "$LOG_DIR/iter_0/renderer.bin"
cp "$ANCHOR_MASKS" "$LOG_DIR/iter_0/masks.mkv"
cp "$ANCHOR_POSES" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_s.zip"
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
[ "$ARCHIVE_BYTES" -gt 0 ] || { echo "FATAL: archive empty" >&2; exit 2; }
log "  archive: $ARCHIVE ($ARCHIVE_BYTES bytes)"

log "=== Stage 5: contest_auth_eval on Lane S archive [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20

# Sanity-check the auth eval actually emitted a RESULT_JSON line — without
# this guard, an auth-eval crash would still let Stage 5 exit 0.
if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval log has no RESULT_JSON line — eval crashed."
    exit 4
fi

log "=== LANE_S_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
