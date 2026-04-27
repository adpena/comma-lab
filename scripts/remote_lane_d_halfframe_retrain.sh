#!/bin/bash
# Lane D: dilated-h64 retrain for HALF-FRAME masks (Quantizr paradigm).
# Council 2026-04-27 (5/0). Implements feedback_half_frame_breaks_posenet.md fix.
#
# Background:
#   The verified 0.9001 baseline renderer has MotionPredictor (e_t1 - e_t).abs()
#   diff features. When fed half-frame masks at inflate (warp-expanded via
#   RadialZoomWarp.warp_inverse_masks) the diff feature collapses → PoseNet
#   distortion explodes 2,600x (28.7 vs 0.011 baseline). Score = 17.55 instead
#   of the predicted ~0.71.
#
# Fix:
#   Retrain a renderer with the SAME ASYM arch family but with:
#     (a) use_zoom_flow=True — motion outputs gate+residual (4ch); flow comes
#         from RadialZoomWarp at inflate (matches the half-frame paradigm).
#     (b) mask_half_sim_prob=0.5 — 50% of training batches replace mask_t with
#         inverse_warp(mask_t1) so the motion module learns BOTH distributions
#         (independently-SegNet-extracted AND warp-reconstructed even-frame masks).
#
# Predicted score landing zone (standalone):
#   - PoseNet: 28.7 → 0.05-0.10 (10x worse than baseline 0.011 acceptable)
#   - SegNet: ~0.003 (already excellent at half-frame)
#   - Rate: ~0.14-0.26 (half-frame archive savings of 0.20-0.32)
#   - Total: 0.55-0.75
# Stacked with Lane A pose TTO + Lane C δ: 0.40-0.55 (BEAT QUANTIZR's 0.33 zone).
#
# Cost: 4090 @ $0.25/hr × 5h = $1.25.
#
# Smoke kill protocol (runs are aborted if these targets are missed):
#   * Phase 1 end (ep400, ~1h): pixel L1 < 12 (kill if higher)
#   * Phase 2 ep200 (~2h): scorer < 8.0
#   * Phase 2 ep800 (~3.5h): scorer < 3.0  -- must beat the broken 17.55 by 5x+
#   * Phase 4 end (ep1880, ~5h): scorer < 1.5  -- target 0.55-0.75
#
# Reproducibility:
#   * seed=42 (pinned in profile)
#   * deterministic=True → CUBLAS_WORKSPACE_CONFIG=:4096:8, cudnn.deterministic=True
#   * PYTHONHASHSEED=1234 (set below for python's hash randomisation)
#   * Same RTX 4090 SKU + PyTorch version → bit-exact checkpoints across re-runs
#
# Codex R-Lane-D-Issue3 (2026-04-27): the previous version had THREE
# deployment-blocking bugs that argparse/glob/header-magic would have caught
# only after $1.25 of 4090 burn:
#   (a) train_renderer.py requires --tag (required=True) — script omitted it.
#   (b) --auth-eval-on-best defaults TRUE; without --auth-eval-masks/--poses
#       train_renderer hard-fails. Stage 4 below builds a real archive and
#       runs contest_auth_eval.py separately, so we explicitly disable the
#       built-in auth-eval here.
#   (c) train_renderer writes renderer_<tag>_best_fp4.pt (a torch.save dict
#       of FP4-packed scales/indices) — NOT a renderer.bin with FP4A magic.
#       The previous glob `*_fp4.bin` matched zero files; if it had matched,
#       cp'ing it to renderer.bin would still fail at inflate (no FP4A magic).
#       We now use the renderer_<tag>_best_fp32.pt + tac.renderer_export →
#       export_asymmetric_checkpoint_fp4 to produce a real .bin. Same path
#       pipeline.py:step_export uses, so we get bit-identical bytes to
#       the canonical chain.
set -euo pipefail
WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
# 2026-04-27 fix: train_renderer.py defaults to
# $REPO/workspace/upstream/comma_video_compression_challenge/models for scorer
# weights. On the canonical Vast.ai layout the scorers live at
# /workspace/pact/upstream/models, so override TAC_UPSTREAM_DIR — same as
# remote_train_bootstrap.sh and remote_pose_tto_bootstrap.sh do.
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_d_results"
mkdir -p "$LOG_DIR"
TAG="lane_d_halfframe"

log() { echo "[lane-d] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Stage 0 (2026-04-27): NVDEC probe BEFORE any GPU spend. Catches the
# bad-host case in 5 seconds. Reference: feedback_vastai_nvdec_host_variation.
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

# Pre-flight: required artifacts present
for f in upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors \
         submissions/baseline_dilated_h64_0_90/optimized_poses.pt; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

# Pre-flight: profile loads + passes preflight (catches mask_half_sim_prob/use_zoom_flow
# inconsistency before we burn 5h of GPU on a misconfigured run).
log "=== Pre-flight: profile validation ==="
"$PYBIN" -c "
import sys
sys.path.insert(0, 'src')
from tac.profiles import PROFILES
from tac.preflight import preflight_profiles
assert 'dilated_h64_half_frame' in PROFILES, 'profile dilated_h64_half_frame not registered'
violations = preflight_profiles(strict=False, verbose=False)
ours = [v for v in violations if 'dilated_h64_half_frame' in v]
if ours:
    print(f'PREFLIGHT FAIL: {ours}', file=sys.stderr); sys.exit(2)
p = PROFILES['dilated_h64_half_frame']
print(f'PROFILE OK: base_ch={p[\"base_ch\"]} mid_ch={p[\"mid_ch\"]} '
      f'motion_hidden={p[\"motion_hidden\"]} use_zoom_flow={p[\"use_zoom_flow\"]} '
      f'mask_half_sim_prob={p[\"mask_half_sim_prob\"]} '
      f'total_epochs={sum(p[f\"phase{i}_epochs\"] for i in range(1,6))}')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Pre-flight: dead-flag-wiring guard — every CLI flag in the train_renderer
# invocation below must exist in train_renderer.py's argparse. Catches the
# class of bug CLAUDE.md's "NEVER invent CLI flags" rule was created for
# (the 2026-04-26 dead-auth-eval-masks incident burned multiple chains).
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
sys.path.insert(0, 'src')
script = open('scripts/remote_lane_d_halfframe_retrain.sh').read()
tr_src = open('src/tac/experiments/train_renderer.py').read()
real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', tr_src))
m = re.search(r'src/tac/experiments/train_renderer\.py(.*?)(?=\n\s*BEST_FP32=|\Z)',
              script, re.DOTALL)
assert m, 'could not locate train_renderer.py invocation in script'
used = set(re.findall(r'\B--([a-z][a-z0-9-]+)', m.group(0)))
invented = used - real
if invented:
    print(f'INVENTED FLAGS: {sorted(invented)} not in train_renderer argparse',
          file=sys.stderr); sys.exit(3)
print(f'OK: {len(used)} flags all real')
" 2>&1 | tee -a "$LOG_DIR/run.log"

log "=== Stage 1: full-frame training (NB: training uses full-frame masks; the"
log "    --mask-half-sim-prob=0.5 setting injects warp-reconstructed even-frame"
log "    masks into 50% of batches — no separate half-frame mask precompute needed) ==="
log "  profile: dilated_h64_half_frame"
log "  tag:     $TAG"
log "  schedule: 400ep P1 + 1080ep P2 + 200ep P3 + 200ep P4 + 100ep P5 = 1980 epochs"
log "  estimated wall clock on 4090: ~5h (\$1.25 at \$0.25/hr)"

# Smoke-kill metadata sidecar so the watchdog can read targets externally
cat > "$LOG_DIR/kill_targets.json" <<'EOF'
{
  "phase1_end_epoch": 400,
  "phase1_pixel_l1_max": 12.0,
  "phase2_smoke_epoch": 600,
  "phase2_smoke_scorer_max": 8.0,
  "phase2_mid_epoch": 1200,
  "phase2_mid_scorer_max": 3.0,
  "phase4_end_epoch": 1880,
  "phase4_end_scorer_max": 1.5,
  "comment": "Hard kill targets per scripts/remote_lane_d_halfframe_retrain.sh header. The watchdog should tail eval logs and SIGTERM the train job when any threshold is exceeded."
}
EOF

# Codex R-Lane-D-Issue3: --no-auth-eval-on-best because Stage 4 below builds
# the real archive (renderer + masks + poses + zoom_scalars) and runs
# contest_auth_eval.py against it. train_renderer's built-in auth-eval would
# need both --auth-eval-masks AND --auth-eval-poses (which don't exist yet at
# this point in the chain — masks haven't been encoded, poses haven't been
# TTO-optimized).
"$PYBIN" -u src/tac/experiments/train_renderer.py \
    --profile dilated_h64_half_frame \
    --tag "$TAG" \
    --device cuda \
    --video upstream/videos/0.mkv \
    --output-dir "$LOG_DIR/train" \
    --use-qat \
    --no-auth-eval-on-best \
    2>&1 | tee "$LOG_DIR/train.log" | grep -E "^\[(train|eval|masks|phase|best|arch)\]|epoch|Phase|scorer" | tail -200

# Codex R-Lane-D-Issue3: train_renderer writes the FP4-packed checkpoint as
# `renderer_<tag>_best_fp4.pt` (torch.save of a quantize_fp4() dict — keys
# look like `weight.packed`, `weight.scales`, `weight.shape`). It is NOT an
# FP4A .bin. To get a renderer.bin the inflate side can read, we use the
# fp32 .pt (which has `model_state_dict` + `__meta__`) and feed it to
# tac.renderer_export.export_asymmetric_checkpoint_fp4 — same path
# pipeline.py:step_export uses, so we get the canonical FP4A binary.
BEST_FP32="$LOG_DIR/train/renderer_${TAG}_best_fp32.pt"
[ -f "$BEST_FP32" ] || {
    echo "FATAL: train_renderer didn't produce ${BEST_FP32}" >&2
    ls -la "$LOG_DIR/train/" >&2
    exit 2
}
log "  best fp32 checkpoint: $BEST_FP32 ($(stat -c '%s' "$BEST_FP32") bytes)"

log "=== Stage 1b: export FP4A renderer.bin from fp32 best ==="
mkdir -p "$LOG_DIR/iter_0"
"$PYBIN" -c "
import sys, torch
sys.path.insert(0, 'src')
from tac.renderer import build_renderer
from tac.renderer_export import export_asymmetric_checkpoint_fp4
ckpt_path = '$BEST_FP32'
out_bin = '$LOG_DIR/iter_0/renderer.bin'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state = ckpt.get('model_state_dict', ckpt)
meta = ckpt.get('__meta__', {}) or {}
fp4_codebook = meta.get('fp4_codebook', 'residual')
fp4_robust_scale = bool(meta.get('fp4_robust_scale', True))
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
    use_zoom_flow=m('use_zoom_flow', True),
)
missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    raise RuntimeError(
        f'shape mismatch refusing to ship: missing={list(missing)[:3]} '
        f'unexpected={list(unexpected)[:3]}'
    )
nbytes = export_asymmetric_checkpoint_fp4(
    model, out_bin, codebook_name=fp4_codebook, robust_scale=fp4_robust_scale,
)
print(f'WROTE {out_bin}: {nbytes} bytes (codebook={fp4_codebook}, robust={fp4_robust_scale})')
" 2>&1 | tee -a "$LOG_DIR/run.log"
[ -f "$LOG_DIR/iter_0/renderer.bin" ] || { echo "FATAL: FP4A export failed" >&2; exit 2; }

log "=== Stage 2: build half-frame archive (renderer + 600 odd masks + poses + zoom_scalars) ==="
# Build half-frame masks (600 odd-frame masks only) — Quantizr paradigm
"$PYBIN" experiments/build_baseline_archive.py \
    --device cuda --crf 50 --half-frame \
    --output "$LOG_DIR/archive_halfframe_seed.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -5
mkdir -p "$LOG_DIR/extracted"
cd "$LOG_DIR/extracted" && unzip -o "$LOG_DIR/archive_halfframe_seed.zip" 2>&1 | tail -5
cd "$WORKSPACE"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"

# Pose TTO using the new renderer (poses are renderer-specific). Use the
# baseline poses as warm-start — they're a good prior but the new renderer
# may want different poses.
log "=== Stage 3: pose TTO with the new half-frame renderer ==="
"$PYBIN" -u experiments/optimize_poses.py \
    --checkpoint "$LOG_DIR/iter_0/renderer.bin" \
    --masks "$LOG_DIR/iter_0/masks.mkv" \
    --gt-poses-path submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
    --device cuda \
    --steps 500 \
    --batch-pairs 8 \
    --eval-roundtrip \
    --posetto-noise-std 0.5 \
    --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/optimize_poses.log" | tail -30

[ -f "$LOG_DIR/optimized_poses.pt" ] || { echo "FATAL: optimize_poses didn't produce optimized_poses.pt"; exit 3; }
cp "$LOG_DIR/optimized_poses.pt" "$LOG_DIR/iter_0/optimized_poses.pt"

# zoom_scalars are required for the inflate-side warp expansion (use_zoom_flow=True)
ZOOM_SCALARS=$(find "$LOG_DIR/train" -name "zoom_scalars.pt" 2>/dev/null | head -1)
if [ -n "$ZOOM_SCALARS" ] && [ -f "$ZOOM_SCALARS" ]; then
    cp "$ZOOM_SCALARS" "$LOG_DIR/iter_0/zoom_scalars.pt"
    log "  bundling zoom_scalars.pt ($(stat -c '%s' "$ZOOM_SCALARS") bytes)"
else
    log "  WARN: no zoom_scalars.pt produced by training — inflate will use identity zoom (degraded)"
fi

ARCHIVE="$LOG_DIR/archive_lane_d.zip"
"$PYBIN" -c "
import zipfile, os
src = '$LOG_DIR/iter_0'
dst = '$ARCHIVE'
files = ['renderer.bin', 'masks.mkv', 'optimized_poses.pt']
if os.path.isfile(os.path.join(src, 'zoom_scalars.pt')):
    files.append('zoom_scalars.pt')
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in files:
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing {p}'
        z.write(p, arcname=n)
print(f'archive {dst}: {os.path.getsize(dst)} bytes ({len(files)} files)')
"

log "=== Stage 4: contest_auth_eval on Lane D archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20

log "=== LANE_D_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
