#!/bin/bash
# Lane K: DSConv Quantizr-killer (88K params from-scratch retrain, FULL-frame).
# Council brief 2026-04-27. Predicted band [0.85, 1.10].
#
# Background:
#   Lane A holds the 1.15 [contest-CUDA] frontier with the dilated-h64 baseline
#   arch (288K params, ~290KB FP32 renderer.bin, total archive 694KB, rate
#   contribution 0.46). The remaining wedge is rate. Quantizr's 0.33 leader
#   uses 88K params + DSConv (depthwise-separable convolutions) + FiLM
#   conditioning. Lane K's bet: train a NEW renderer FROM SCRATCH at
#   Quantizr-class capacity so the FP4 byte count drops from 180KB to ~50KB
#   without the half-frame paradigm risk that broke Lane D (memory:
#   feedback_half_frame_breaks_posenet, score 17.55).
#
#   Anchor strategy: ship the new renderer alongside Lane A's verified
#   masks.mkv + optimized_poses.pt. The ONLY archive delta is the renderer
#   byte count; if Lane K's renderer is ≤ Lane A's distortion, the rate
#   savings translate directly to score improvement.
#
# Architecture (verified empirically: 88,996 params = exact Quantizr match):
#   * base_ch=24, mid_ch=32, motion_hidden=16, embed_dim=6, depth=1, pose_dim=6
#   * use_dsconv=True, use_zoom_flow=False (PairGenerator path, full-frame)
#   * padding_mode='zeros', use_dilation=False (no confounds)
#
# Predicted score landing zone [0.85, 1.10]:
#   - SegNet: ~0.005 (DSConv + capacity reduction may cost 0.001-0.002)
#   - PoseNet: ~0.04-0.10 (smaller arch + FiLM-from-epoch-0 → comparable to
#     dilated-h64 baseline 0.247 PoseNet because FiLM gives per-pair pose
#     conditioning the baseline lacks)
#   - Rate: ~0.20-0.28 (renderer drops from 290KB → ~50KB FP4 → -0.18 rate)
#   - Standalone score: 0.85-1.10. If 0.95 we BEAT Lane A's 1.15 by 0.20.
#
# Cost: 4090 @ $0.25/hr × ~12h = ~$3-4. Hard cap: $4 ($24 Vast.ai budget).
#
# Smoke kill protocol (the watchdog should SIGTERM if any threshold missed):
#   * Phase 1 end (ep600, ~3h): pixel L1 < 12 (Quantizr-class arch should
#     converge similarly to dilated-h64 baseline which plateaus 5-7)
#   * Phase 2 ep600 (~5h): scorer < 8.0 (kill if higher)
#   * Phase 2 ep1200 (~7h): scorer < 3.0 (must be on track for sub-1.5)
#   * Phase 4 end (ep2900, ~11h): scorer < 1.5 (target 0.85-1.10 standalone)
#
# Reproducibility:
#   * seed=1234 (pinned in profile, matches build_baseline_archive.py default)
#   * deterministic=True → CUBLAS_WORKSPACE_CONFIG=:4096:8, cudnn.deterministic
#   * PYTHONHASHSEED=1234 (set below for python's hash randomisation)
#   * Same RTX 4090 SKU + PyTorch version → bit-exact checkpoints
#
# Lane K specifically guards against the failure modes catalogued in CLAUDE.md:
#   * --device cuda REQUIRED (no MPS fallback — drift 23x on PoseNet)
#   * --tag passed (train_renderer.py argparse requires=True)
#   * --no-auth-eval-on-best (Stage 4 builds the real archive + runs
#     contest_auth_eval.py; the built-in path needs masks/poses that don't
#     exist yet at training time)
#   * Python zipfile (NOT shell `zip` — PyTorch container has no zip binary,
#     memory: feedback_zip_dep_bootstrap_trap)
#   * NVDEC probe Stage 0 (memory: feedback_vastai_nvdec_host_variation)
#   * provenance.json + heartbeat.log (memory: feedback_canonical_remote_bootstraps)
#   * Pre-flight argparse dead-flag scan (CLAUDE.md non-negotiable: NEVER invent
#     CLI flags, memory: feedback_dead_flag_wiring_pattern)

set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
# train_renderer.py defaults to $REPO/workspace/upstream/comma_video_compression_challenge/models
# for scorer weights. On the canonical Vast.ai layout the scorers live at
# /workspace/pact/upstream/models, so override TAC_UPSTREAM_DIR — same as
# remote_train_bootstrap.sh / remote_pose_tto_bootstrap.sh / Lane D do.
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_k_results"
mkdir -p "$LOG_DIR"
TAG="lane_k_dsconv_quantizr_killer"

log() { echo "[lane-k] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_k_dsconv_quantizr_killer.sh',
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'profile': 'dsconv_quantizr_killer',
    'predicted_band': [0.85, 1.10],
    'anchor_score_baseline': 1.15,
    'anchor_lane': 'experiments/results/lane_a_landed',
    'arch_param_count': 88996,
    'arch_summary': 'base_ch=24, mid_ch=32, motion_hidden=16, embed_dim=6, depth=1, pose_dim=6, use_dsconv=True, use_zoom_flow=False',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=K gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend. Catches the bad-host case in 5s.
# Reference: feedback_vastai_nvdec_host_variation. Same 4090 image, different
# hosts, same driver, different NVDEC outcome — Oregon worked, California
# failed with CUDA_ERROR_NO_DEVICE from DALI's video MIXED operator.
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

# Pre-flight: required artifacts present. Lane K anchors masks + poses on
# Lane A's verified 1.15 [contest-CUDA] artifacts so the only delta is the
# renderer byte count. If these are missing, the build stage will fail
# AFTER 12h of training — fail fast here.
log "=== Pre-flight: required artifacts ==="
for f in upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors \
         experiments/results/lane_a_landed/iter_0/masks.mkv \
         experiments/results/lane_a_landed/iter_0/optimized_poses.pt; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

# Pre-flight: profile loads + passes preflight. Catches profile-key typos and
# eval_roundtrip=False before we burn 12h of GPU on a misconfigured run.
log "=== Pre-flight: profile validation ==="
"$PYBIN" -c "
import sys
sys.path.insert(0, 'src')
from tac.profiles import PROFILES
from tac.preflight import preflight_profiles
from tac.renderer import build_renderer

assert 'dsconv_quantizr_killer' in PROFILES, 'profile dsconv_quantizr_killer not registered'
violations = preflight_profiles(strict=False, verbose=False)
ours = [v for v in violations if 'dsconv_quantizr_killer' in v]
if ours:
    print(f'PREFLIGHT FAIL: {ours}', file=sys.stderr); sys.exit(2)

p = PROFILES['dsconv_quantizr_killer']
m = build_renderer(
    num_classes=5,
    embed_dim=p['embed_dim'],
    base_ch=p['base_ch'],
    mid_ch=p['mid_ch'],
    motion_hidden=p['motion_hidden'],
    depth=p['depth'],
    blend_mode='scalar',
    noise_mode='deterministic',
    motion_type='learned_cnn',
    use_zoom_flow=p['use_zoom_flow'],
    use_dsconv=p['use_dsconv'],
    padding_mode=p['padding_mode'],
    use_dilation=p['use_dilation'],
    pose_dim=p['pose_dim'],
)
n = sum(pp.numel() for pp in m.parameters())
assert 80_000 <= n <= 100_000, f'param count {n} outside Quantizr-class band [80K, 100K]'
print(f'PROFILE OK: base_ch={p[\"base_ch\"]} mid_ch={p[\"mid_ch\"]} '
      f'motion_hidden={p[\"motion_hidden\"]} use_dsconv={p[\"use_dsconv\"]} '
      f'use_zoom_flow={p[\"use_zoom_flow\"]} pose_dim={p[\"pose_dim\"]} '
      f'params={n:,} '
      f'total_epochs={sum(p[f\"phase{i}_epochs\"] for i in range(1,6))}')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Pre-flight: dead-flag-wiring guard — every CLI flag in the train_renderer
# invocation below must exist in train_renderer.py's argparse. CLAUDE.md
# non-negotiable: NEVER invent CLI flags. Memory: feedback_dead_flag_wiring_pattern
# (the 2026-04-26 dead --auth-eval-masks burn).
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
sys.path.insert(0, 'src')
script = open('scripts/remote_lane_k_dsconv_quantizr_killer.sh').read()
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

log "=== Stage 1: full-frame from-scratch training (88K params, DSConv, FiLM) ==="
log "  profile: dsconv_quantizr_killer"
log "  tag:     $TAG"
log "  schedule: 600ep P1 + 1500ep P2 + 400ep P3 + 400ep P4 + 100ep P5 = 3000 epochs"
log "  estimated wall clock on 4090: ~12h (\$3-4 at \$0.25/hr)"

# Smoke-kill metadata sidecar so the watchdog can read targets externally.
cat > "$LOG_DIR/kill_targets.json" <<'EOF'
{
  "phase1_end_epoch": 600,
  "phase1_pixel_l1_max": 12.0,
  "phase2_smoke_epoch": 1200,
  "phase2_smoke_scorer_max": 8.0,
  "phase2_mid_epoch": 2100,
  "phase2_mid_scorer_max": 3.0,
  "phase4_end_epoch": 2900,
  "phase4_end_scorer_max": 1.5,
  "predicted_band": [0.85, 1.10],
  "anchor_score_baseline": 1.15,
  "anchor_lane": "experiments/results/lane_a_landed",
  "comment": "Hard kill targets per scripts/remote_lane_k_dsconv_quantizr_killer.sh header. The watchdog should tail eval logs and SIGTERM the train job when any threshold is exceeded."
}
EOF

# --no-auth-eval-on-best: Stage 4 below builds the real archive (renderer +
# Lane A masks + Lane A poses) and runs contest_auth_eval.py against it.
# The built-in --auth-eval-on-best path needs --auth-eval-masks AND
# --auth-eval-poses (which would be Lane A's anyway, but we keep the auth
# eval out of the training process for clean log separation).
"$PYBIN" -u src/tac/experiments/train_renderer.py \
    --profile dsconv_quantizr_killer \
    --tag "$TAG" \
    --device cuda \
    --video upstream/videos/0.mkv \
    --output-dir "$LOG_DIR/train" \
    --use-qat \
    --fp4-codebook residual \
    --fp4-robust-scale \
    --fp4-stochastic \
    --no-auth-eval-on-best \
    2>&1 | tee "$LOG_DIR/train.log" | grep -E "^\[(train|eval|masks|phase|best|arch)\]|epoch|Phase|scorer" | tail -200

# train_renderer writes the FP4-packed checkpoint as `renderer_<tag>_best_fp4.pt`
# (torch.save of a quantize_fp4() dict). It is NOT an FP4A .bin. To get a
# renderer.bin the inflate side can read, we use the fp32 .pt (which has
# `model_state_dict` + `__meta__`) and feed it to
# tac.renderer_export.export_asymmetric_checkpoint_fp4 — same path
# pipeline.py:step_export uses. Bit-identical bytes to the canonical chain.
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
    base_ch=m('base_ch', 24),
    mid_ch=m('mid_ch', 32),
    motion_hidden=m('motion_hidden', 16),
    depth=m('depth', 1),
    pose_dim=m('pose_dim', 6),
    use_dsconv=m('use_dsconv', True),
    padding_mode=m('padding_mode', 'zeros'),
    use_dilation=m('use_dilation', False),
    use_zoom_flow=m('use_zoom_flow', False),
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

log "=== Stage 2: bundle Lane A masks + poses into the iter_0 staging dir ==="
# Lane K anchors on Lane A's verified 1.15 [contest-CUDA] artifacts. Anchoring
# minimises the change surface — the ONLY delta from Lane A is the renderer
# byte count. No re-encoding of masks (which would change AV1 bytes), no
# re-running pose TTO (which would change pose-bin bytes).
cp experiments/results/lane_a_landed/iter_0/masks.mkv \
   "$LOG_DIR/iter_0/masks.mkv"
cp experiments/results/lane_a_landed/iter_0/optimized_poses.pt \
   "$LOG_DIR/iter_0/optimized_poses.pt"
log "  bundled masks.mkv ($(stat -c '%s' "$LOG_DIR/iter_0/masks.mkv") bytes)"
log "  bundled optimized_poses.pt ($(stat -c '%s' "$LOG_DIR/iter_0/optimized_poses.pt") bytes)"
log "  renderer.bin ($(stat -c '%s' "$LOG_DIR/iter_0/renderer.bin") bytes)"

log "=== Stage 3: build Lane K archive (Python zipfile, deterministic) ==="
# Python zipfile (NOT shell `zip`) — PyTorch container has no `zip` binary.
# Memory: feedback_zip_dep_bootstrap_trap (LANE-B 2026-04-26 burned 6.5h+$2
# discovering this).
ARCHIVE="$LOG_DIR/archive_lane_k.zip"
"$PYBIN" -c "
import zipfile, os
src = '$LOG_DIR/iter_0'
dst = '$ARCHIVE'
files = ['renderer.bin', 'masks.mkv', 'optimized_poses.pt']
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in files:
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing {p}'
        z.write(p, arcname=n)
sz = os.path.getsize(dst)
print(f'archive {dst}: {sz} bytes ({len(files)} files)')
# Sanity: refuse to ship an archive whose renderer.bin alone is >150KB
# (target ~50KB FP4 from 88K params; if it's 150KB+ something's wrong).
rb = os.path.getsize(os.path.join(src, 'renderer.bin'))
if rb > 150_000:
    raise SystemExit(f'renderer.bin {rb} bytes > 150KB — Lane K target ~50KB; investigate')
print(f'renderer.bin within target band: {rb} bytes (~{rb//1024}KB)')
"

log "=== Stage 4: contest_auth_eval on Lane K archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPESTATUS[0]}" >&2; exit "${PIPESTATUS[0]}"
    fi

log "=== LANE_K_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "  predicted band: [0.85, 1.10]"
log "  anchor (Lane A): 1.15 [contest-CUDA]"
log "  if score < 1.15 we BEAT the frontier; if < 0.95 we beat by 0.20+"
