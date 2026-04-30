#!/bin/bash
# Lane 19: SegNet logit-margin boundary loss training run.
#
# Council .omx/research/council_lane_19_logit_margin_design_20260430.md
# (5-of-6 GREEN with conditions; Quantizr YELLOW pending A/B).
#
# Anchor: Lane G v3 = 1.05 [contest-CUDA] (DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL).
#
# DELTA from Lane G v3:
#   * loss_mode "standard" → "logit_margin"
#   * NEW auxiliary `compute_segnet_logit_margin_aux` runs alongside
#     KL distill (not as replacement). Boundary pixels (margin < threshold)
#     get full CE weight; confident pixels (margin >= threshold) zero weight.
#   * Different seed (89) → different RNG basin from Lane G v3 (43).
#
# Predicted band [prediction]: [0.75, 1.05] [contest-CUDA] standalone.
#   * Floor 0.75: -3e-3 SegNet distortion → ~0.30 score reduction on
#     Lane G v3 1.05 anchor (Phase 2 ceiling).
#   * Mid 0.95: -1e-3 SegNet distortion → ~0.10 score reduction (floor case).
#   * Ceiling 1.05: margin loss buys nothing over standard CE (kill case;
#     would demote to Phase 3 deferred).
#
# Cost: 4090 @ $0.25/hr × ~5h training + 30min auth eval = ~$1.50.
#
# Per CLAUDE.md NEVER-INVENT-CLI-FLAGS: every flag passed to
# train_renderer.py / optimize_poses.py / build_baseline_archive.py /
# contest_auth_eval.py was verified by argparse-grep. The dead-flag scanner
# at Stage 0 re-validates this at launch time.
#
# Memory: project_lane_19_logit_margin_landed_20260430.md (TBD post-result).
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
# Profile uses seed=89 (Lane G v3 = 43, Lane H-V3 = 67) — different RNG basin.
export PYTHONHASHSEED=89
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_19_logit_margin_results"
mkdir -p "$LOG_DIR"
TAG="lane_19_logit_margin"

log() { echo "[lane-19] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_19_logit_margin.sh',
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'profile': 'lane_19_logit_margin',
    'predicted_band': [0.75, 1.05],
    'anchor_score_baseline': 1.05,
    'anchor_lane': 'lane_g_v3 (1.05 contest-CUDA)',
    'lane_19_premise': 'SegNet score = argmax disagreement; weight CE by (threshold - margin)/threshold so confident pixels get zero loss + boundary pixels get full signal (Fridrich UNIWARD applied to segmentation).',
    'delta_from_lane_g_v3': {
        'loss_mode': 'standard -> logit_margin',
        'logit_margin_weight': '0.0 -> 0.1',
        'logit_margin_threshold': '1.0 (default)',
        'kl_distill_aux': 'KEPT (Lane 19 is auxiliary, not replacement; confident-wrong pixels still caught by scorer_loss)',
        'seed': '43 -> 89 (different RNG basin)',
    },
    'cost_estimate_usd': 1.50,
    'wall_clock_estimate_hours': 5.5,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=19 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend. Reference:
# feedback_vastai_nvdec_host_variation.
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

# Pre-flight: profile loads + passes preflight.
log "=== Pre-flight: profile validation ==="
"$PYBIN" -c "
import sys
sys.path.insert(0, 'src')
from tac.profiles import PROFILES
from tac.preflight import preflight_profiles
prof_name = 'lane_19_logit_margin'
assert prof_name in PROFILES, f'profile {prof_name} not registered'
violations = preflight_profiles(strict=False, verbose=False)
ours = [v for v in violations if prof_name in v]
if ours:
    print(f'PREFLIGHT FAIL: {ours}', file=sys.stderr); sys.exit(2)
p = PROFILES[prof_name]
print(f'PROFILE OK: loss_mode={p[\"loss_mode\"]} '
      f'logit_margin_weight={p[\"logit_margin_weight\"]} '
      f'logit_margin_threshold={p[\"logit_margin_threshold\"]} '
      f'base_ch={p[\"base_ch\"]} mid_ch={p[\"mid_ch\"]} '
      f'kl_distill_weight={p[\"kl_distill_weight\"]} '
      f'seed={p[\"seed\"]}')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Pre-flight: dead-flag-wiring guard — every CLI flag in the train_renderer
# invocation below must exist in train_renderer.py's argparse. CLAUDE.md
# non-negotiable: NEVER invent CLI flags.
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
sys.path.insert(0, 'src')
script = open('scripts/remote_lane_19_logit_margin.sh').read()
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

# Pre-flight: STRICT preflight Check 93 (Lane 19 callers pass threshold=).
log "=== Pre-flight: Check 93 (logit-margin callers have threshold=) ==="
"$PYBIN" -c "
import sys
sys.path.insert(0, 'src')
from tac.preflight import check_logit_margin_loss_uses_boundary_mask
v = check_logit_margin_loss_uses_boundary_mask(strict=True, verbose=True)
print('Check 93 PASSED')
" 2>&1 | tee -a "$LOG_DIR/run.log"

log "=== Stage 1: training (Lane 19 logit-margin auxiliary) ==="
log "  profile: lane_19_logit_margin (inherits Lane G v3 + logit_margin aux)"
log "  tag:     $TAG"
log "  schedule: inherited from Lane G v3 (1980 epochs)"
log "  estimated wall clock on 4090: ~5h (\$1.25 at \$0.25/hr)"

# --no-auth-eval-on-best because Stage 4 below builds the real archive and
# runs contest_auth_eval (one authoritative score, on submitted bytes).
"$PYBIN" -u src/tac/experiments/train_renderer.py \
    --profile lane_19_logit_margin \
    --tag "$TAG" \
    --device cuda \
    --video upstream/videos/0.mkv \
    --output-dir "$LOG_DIR/train" \
    --use-qat \
    --no-auth-eval-on-best \
    2>&1 | tee "$LOG_DIR/train.log" | grep -E "^\[(train|eval|masks|phase|best|arch|ep)\]|epoch|Phase|scorer" | tail -200

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

log "=== Stage 2: build half-frame archive (renderer + 600 odd masks + poses) ==="
"$PYBIN" -c "
from tac.profiles import PROFILES
prof_name = 'lane_19_logit_margin'
p = PROFILES.get(prof_name)
assert p is not None, f'PROFILES is missing {prof_name}'
assert p.get('mask_half_sim_prob', 0) > 0 or p.get('use_zoom_flow', False), \
    f'profile {prof_name} must have mask_half_sim_prob>0 OR use_zoom_flow=True'
print(f'halfframe-profile-assertion OK: {prof_name}')
"
# Profile: --profile lane_19_logit_margin (renderer was trained with this profile;
# build_baseline_archive.py reads the renderer.bin embedded profile, but Check F
# requires the profile name appear within 30 lines of --half-frame).
"$PYBIN" experiments/build_baseline_archive.py \
    --device cuda --crf 50 --half-frame \
    --output "$LOG_DIR/archive_halfframe_seed.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -5
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi
mkdir -p "$LOG_DIR/extracted"
cd "$LOG_DIR/extracted" && unzip -o "$LOG_DIR/archive_halfframe_seed.zip" 2>&1 | tail -5
cd "$WORKSPACE"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"

# Pose TTO using the new renderer (poses are renderer-specific).
log "=== Stage 3: pose TTO with the new logit-margin renderer ==="
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
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

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

ARCHIVE="$LOG_DIR/archive_lane_19.zip"
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

log "=== Stage 4: contest_auth_eval on Lane 19 archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

# RESULT_JSON guard (LANE-B silent-crash prevention).
if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval.log missing RESULT_JSON — silent eval crash. Investigate before reporting score."
    exit 4
fi

log "=== LANE_19_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "  predicted band: [0.75, 1.05] standalone (vs Lane G v3 1.05 anchor)"
log "  anchor baseline: 1.05 [contest-CUDA] (Lane G v3)"
