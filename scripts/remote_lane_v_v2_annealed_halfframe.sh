#!/bin/bash
# Lane V-V2: Quantizr-replica (88K params, DSConv + FiLM + KL distill) with
# ANNEALED mask_half_sim_prob (0.0 → 1.0 over training).
#
# V1 oversight: Lane V's mask_half_sim_prob=1.0 from epoch 0 is a cold-start.
# The motion module's randomly-initialised weights had to learn the warp-
# expansion distribution from the very first batch, with no preview of the
# easier full-frame baseline. The cold start was a council bet that paid
# off (predicted band [0.50, 1.10]) but was high-variance.
#
# V2 fix: ANNEAL the warp probability over training:
#   * Warmup full-frame (epoch 0..900, 30%): mask_half_sim_prob = 0.0
#     The motion module learns from the easy distribution first — strong
#     initialisation that survives the transition.
#   * Linear ramp (epoch 900..2100, 40%): 0.0 → 1.0
#     The renderer adapts to the warp-expansion distribution gradually.
#   * Lane V endpoint (epoch 2100..3000, 30%): 1.0 (always-on warp)
#     Same final convergence target as Lane V-V1.
#
# Single variable vs Lane V-V1: --profile quantizr_replica_88k_halfframe_annealed
# (replaces --profile quantizr_replica_88k_halfframe). Everything else
# identical to remote_lane_v_quantizr_replica_88k_halfframe.sh.
#
# Predicted band: [0.45, 1.05] [contest-CUDA].
#   * Slightly TIGHTER than V1's [0.50, 1.10] because annealing reduces the
#     early-epoch optimisation variance (the warmup phase establishes a
#     strong initialisation).
#   * Floor 0.45: optimal annealing finds a better local minimum than V1's
#     cold-start.
#   * Ceiling 1.05: annealing buys nothing — same convergence as V1.
#
# Cost / wall-clock identical to V1 (~12h on 4090, $4-5 end-to-end with
# pose-TTO + auth-eval).
#
# ── Bash safety ─────────────────────────────────────────────────────────
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
# Lane V-V2 uses seed=1235 (V1 uses 1234) — match the profile.
export PYTHONHASHSEED=1235
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_v_v2_results"
mkdir -p "$LOG_DIR"
TAG="lane_v_v2_quantizr_replica_88k_halfframe_annealed"

log() { echo "[lane-v-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_v_v2_annealed_halfframe.sh',
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'profile': 'quantizr_replica_88k_halfframe_annealed',
    'predicted_band': [0.45, 1.05],
    'anchor_score_baseline': 1.15,
    'anchor_lane': 'lane_a (1.15 contest-CUDA)',
    'lane_v_v2_premise': 'mask_half_sim_prob ANNEALED 0->1 (vs V1 1.0 from epoch 0)',
    'annealing_schedule': 'warmup [0..30%], ramp [30%..70%], endpoint [70%..100%]',
    'cost_estimate_usd': 4.5,
    'wall_clock_estimate_hours': 13.0,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=V-V2 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend.
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

# Pre-flight: required artifacts present.
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
prof_name = 'quantizr_replica_88k_halfframe_annealed'
assert prof_name in PROFILES, f'profile {prof_name} not registered'
violations = preflight_profiles(strict=False, verbose=False)
ours = [v for v in violations if prof_name in v]
if ours:
    print(f'PREFLIGHT FAIL: {ours}', file=sys.stderr); sys.exit(2)
p = PROFILES[prof_name]
sched = p['mask_half_sim_prob_anneal']
print(f'PROFILE OK: base_ch={p[\"base_ch\"]} mid_ch={p[\"mid_ch\"]} '
      f'motion_hidden={p[\"motion_hidden\"]} use_zoom_flow={p[\"use_zoom_flow\"]} '
      f'mask_half_sim_prob={p[\"mask_half_sim_prob\"]} (endpoint, annealed) '
      f'anneal_schedule={sched} '
      f'kl_distill_weight={p[\"kl_distill_weight\"]} '
      f'pose_dim={p[\"pose_dim\"]} use_dsconv={p[\"use_dsconv\"]} '
      f'total_epochs={sum(p[f\"phase{i}_epochs\"] for i in range(1,6))} '
      f'seed={p[\"seed\"]}')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Pre-flight: param count smoke (88K-class).
log "=== Pre-flight: arch smoke (param count target ~88K) ==="
"$PYBIN" -c "
import sys
sys.path.insert(0, 'src')
from tac.profiles import PROFILES
from tac.renderer import build_renderer
p = PROFILES['quantizr_replica_88k_halfframe_annealed']
m = build_renderer(
    num_classes=5, embed_dim=p['embed_dim'], base_ch=p['base_ch'],
    mid_ch=p['mid_ch'], motion_hidden=p['motion_hidden'], depth=p['depth'],
    pose_dim=p['pose_dim'], use_dsconv=p['use_dsconv'],
    padding_mode=p['padding_mode'], use_dilation=p['use_dilation'],
    use_zoom_flow=p['use_zoom_flow'],
)
total = sum(pp.numel() for pp in m.parameters())
assert 80_000 <= total <= 100_000, f'param count {total} outside 88K-class budget'
print(f'ARCH OK: {total} params ({total/1000:.1f}K)')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Pre-flight: dead-flag-wiring guard.
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
sys.path.insert(0, 'src')
script = open('scripts/remote_lane_v_v2_annealed_halfframe.sh').read()
tr_src = open('src/tac/experiments/train_renderer.py').read()
real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', tr_src))
m = re.search(
    r'src/tac/experiments/train_renderer\.py(.*?)(?=\n\s*BEST_FP32=|\Z)',
    script, re.DOTALL,
)
assert m, 'could not locate train_renderer.py invocation in script'
used = set(re.findall(r'\B--([a-z][a-z0-9-]+)', m.group(0)))
invented = used - real
if invented:
    print(f'INVENTED FLAGS: {sorted(invented)} not in train_renderer argparse',
          file=sys.stderr); sys.exit(3)
print(f'OK: {len(used)} flags all real')
" 2>&1 | tee -a "$LOG_DIR/run.log"

log "=== Stage 1: from-scratch training (annealed mask_half_sim_prob 0.0 → 1.0) ==="
log "    Schedule: warmup full-frame for 30%, linear ramp 30%->70%, half-frame 70%->100%"
log "    NB: epoch 0..900 trains on full-frame, easy distribution → strong init"
log "         epoch 900..2100 ramps to half-frame → smooth transition"
log "         epoch 2100..3000 trains on half-frame → endpoint = Lane V-V1"
log "  profile: quantizr_replica_88k_halfframe_annealed"
log "  tag:     $TAG"
log "  schedule: 600ep P1 + 1500ep P2 + 400ep P3 + 400ep P4 + 100ep P5 = 3000 epochs"
log "  estimated wall clock on 4090: ~12h (\$3.00 at \$0.25/hr; total budget \$4-5 with TTO+eval)"

# Smoke-kill metadata sidecar (mirrors V1).
cat > "$LOG_DIR/kill_targets.json" <<'EOF'
{
  "phase1_end_epoch": 600,
  "phase1_pixel_l1_max": 14.0,
  "phase2_smoke_epoch": 1200,
  "phase2_smoke_scorer_max": 10.0,
  "phase2_end_epoch": 2100,
  "phase2_end_scorer_max": 4.0,
  "phase4_end_epoch": 2900,
  "phase4_end_scorer_max": 2.0,
  "comment": "Hard kill targets per scripts/remote_lane_v_v2_annealed_halfframe.sh header. The watchdog should tail eval logs and SIGTERM the train job when any threshold is exceeded."
}
EOF

# We do NOT pass --mask-half-sim-prob-schedule explicitly — the profile
# carries the schedule via mask_half_sim_prob_anneal and train_renderer.py
# reads it directly. Passing the CLI flag would be redundant + the dead-
# flag scanner would flag any drift.
"$PYBIN" -u src/tac/experiments/train_renderer.py \
    --profile quantizr_replica_88k_halfframe_annealed \
    --tag "$TAG" \
    --device cuda \
    --video upstream/videos/0.mkv \
    --output-dir "$LOG_DIR/train" \
    --use-qat \
    --no-auth-eval-on-best \
    2>&1 | tee "$LOG_DIR/train.log" | grep -E "^\[(train|eval|masks|phase|best|arch)\]|epoch|Phase|scorer" | tail -200

# train_renderer writes the FP4-packed checkpoint as
# `renderer_<tag>_best_fp4.pt`. To get a renderer.bin the inflate side
# can read, we use the fp32 .pt + tac.renderer_export.export_asymmetric_checkpoint_fp4.
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
# Half-frame archive REQUIRES a renderer trained with mask_half_sim_prob > 0 OR
# use_zoom_flow=True. Our annealed profile satisfies BOTH (mask_half_sim_prob
# = 1.0 endpoint + use_zoom_flow=True). The assertion below confirms.
"$PYBIN" -c "
from tac.profiles import PROFILES
p = PROFILES.get('quantizr_replica_88k_halfframe_annealed')
assert p is not None, 'PROFILES is missing quantizr_replica_88k_halfframe_annealed'
assert p.get('mask_half_sim_prob', 0) > 0 or p.get('use_zoom_flow', False), \
    'profile quantizr_replica_88k_halfframe_annealed must have mask_half_sim_prob>0 OR use_zoom_flow=True'
print('halfframe-profile-assertion OK: quantizr_replica_88k_halfframe_annealed')
"
# NOTE: --profile quantizr_replica_88k_halfframe_annealed used in Stage 1
# train above (asserted by previous python check). build_baseline_archive
# itself doesn't take --profile, just packages the trained renderer.
set +e
"$PYBIN" experiments/build_baseline_archive.py \
    --device cuda --crf 50 --half-frame \
    --output "$LOG_DIR/archive_halfframe_seed.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -5
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi
mkdir -p "$LOG_DIR/extracted"
cd "$LOG_DIR/extracted" && unzip -o "$LOG_DIR/archive_halfframe_seed.zip" 2>&1 | tail -5
cd "$WORKSPACE"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"

log "=== Stage 3: pose TTO with the new annealed half-frame renderer ==="
set +e
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
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

[ -f "$LOG_DIR/optimized_poses.pt" ] || { echo "FATAL: optimize_poses didn't produce optimized_poses.pt"; exit 3; }
cp "$LOG_DIR/optimized_poses.pt" "$LOG_DIR/iter_0/optimized_poses.pt"

# zoom_scalars are required for the inflate-side warp expansion (use_zoom_flow=True).
ZOOM_SCALARS=$(find "$LOG_DIR/train" -name "zoom_scalars.pt" 2>/dev/null | head -1)
if [ -n "$ZOOM_SCALARS" ] && [ -f "$ZOOM_SCALARS" ]; then
    cp "$ZOOM_SCALARS" "$LOG_DIR/iter_0/zoom_scalars.pt"
    log "  bundling zoom_scalars.pt ($(stat -c '%s' "$ZOOM_SCALARS") bytes)"
else
    log "  WARN: no zoom_scalars.pt produced by training — inflate will use identity zoom (degraded)"
fi

ARCHIVE="$LOG_DIR/archive_lane_v_v2.zip"
# Python zipfile (NOT shell `zip`) — PyTorch container has no `zip` binary
# (memory: feedback_zip_dep_bootstrap_trap).
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

log "=== Stage 4: contest_auth_eval on Lane V-V2 archive ==="
rm -rf "$LOG_DIR/eval_work"
set +e
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

# RESULT_JSON guard.
if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval.log missing RESULT_JSON — silent eval crash. Investigate before reporting score."
    exit 4
fi

log "=== LANE_V_V2_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "  predicted band: [0.45, 1.05] standalone (vs Lane V-V1 [0.50, 1.10])"
log "  anchor baseline: 1.15 [contest-CUDA] (Lane A frontier)"
