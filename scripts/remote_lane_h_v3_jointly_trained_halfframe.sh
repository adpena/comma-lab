#!/bin/bash
# Lane H-V3: half-frame revival via JOINT warp-expansion training.
#
# Forensic audit (project_killed_lanes_forensic_audit_20260428) found that
# our 4 prior half-frame attempts (Lane D V1/V2/V3, Lane V V1/V2) all failed
# for distinct, fixable reasons:
#   * Lane D RETROFIT — `mask_half_sim_prob=0.5` mid-train on a renderer
#     locked into (e_t1-e_t).abs() diff features.
#   * Lane D-V3 distribution mismatch — train endpoint 0.5 vs inflate 1.0.
#   * Lane V channel bug — DSConv 88K path crashes downstream of warp.
#
# Quantizr ships half-frame at 0.33 because they train JOINTLY from epoch 0.
# Lane H-V3 is the proper revival:
#   1. JOINT training from epoch 0 (with brief full-frame warmup for init).
#   2. Train endpoint = inflate distribution (mask_half_sim_prob=1.0).
#   3. 288K dilated-h64 arch (Lane G v3 anchor), NOT 88K DSConv.
#
# Predicted band [0.55, 0.95] [contest-CUDA].
# Cost: 4090 @ $0.25/hr × ~5h = ~$1.25 (+ TTO + auth = $1.50 total).
#
# Per CLAUDE.md non-negotiables:
#   * Stage 0 = NVDEC probe before any GPU spend.
#   * eval_roundtrip=True (inherited from Lane G v3 anchor).
#   * Stage N+1 = contest-CUDA auth eval against EXACT submission archive bytes.
#   * Heartbeat every 60s, AppleDouble cleanup after rsync, RESULT_JSON guard.
#
# CLI flag verification (per memory feedback_dead_flag_wiring_pattern):
# the in-script dead-flag scanner re-validates this at launch time.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=67
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_h_v3_results"
mkdir -p "$LOG_DIR"
TAG="lane_h_v3_joint_halfframe"

log() { echo "[lane-h-v3] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md canonical pipeline standard).
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
    'lane_script': 'scripts/remote_lane_h_v3_jointly_trained_halfframe.sh',
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'profile': 'h_v3_joint_halfframe',
    'predicted_band': [0.55, 0.95],
    'anchor_score_baseline': 1.05,
    'anchor_lane': 'lane_g_v3 (1.05 contest-CUDA)',
    'lane_h_v3_premise': 'JOINT training from epoch 0 + curriculum 0.0 -> 1.0 (5%-15% ramp) + endpoint matches inflate',
    'curriculum_schedule': 'warmup full-frame [0..5%], ramp 0->1 [5%..15%], endpoint 1.0 [15%..100%]',
    'fixes_prior_failures': {
        'lane_d_retrofit': 'JOINT from epoch 0 (NOT mid-train retrofit)',
        'lane_d_v3_mismatch': 'endpoint=1.0 matches inflate (NOT 0.5)',
        'lane_v_channel_bug': '288K dilated-h64 arch (NOT 88K DSConv)',
    },
    'cost_estimate_usd': 1.50,
    'wall_clock_estimate_hours': 5.0,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=H-V3 gpu=$GPU" >> "$HEARTBEAT"
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
prof_name = 'h_v3_joint_halfframe'
assert prof_name in PROFILES, f'profile {prof_name} not registered'
violations = preflight_profiles(strict=False, verbose=False)
ours = [v for v in violations if prof_name in v]
if ours:
    print(f'PREFLIGHT FAIL: {ours}', file=sys.stderr); sys.exit(2)
p = PROFILES[prof_name]
sched = p['mask_half_sim_prob_anneal']
assert p['mask_half_sim_prob'] == 1.0, \
    f'endpoint mismatch: mask_half_sim_prob={p[\"mask_half_sim_prob\"]} (must be 1.0 to match inflate)'
assert sched['end_value'] == 1.0, \
    f'curriculum endpoint mismatch: end_value={sched[\"end_value\"]} (must be 1.0)'
assert p['use_zoom_flow'] is True, \
    f'use_zoom_flow={p[\"use_zoom_flow\"]} required for half-frame inflate'
print(f'PROFILE OK: base_ch={p[\"base_ch\"]} mid_ch={p[\"mid_ch\"]} '
      f'motion_hidden={p[\"motion_hidden\"]} use_zoom_flow={p[\"use_zoom_flow\"]} '
      f'mask_half_sim_prob={p[\"mask_half_sim_prob\"]} (endpoint=inflate-distribution) '
      f'anneal_schedule={sched} '
      f'kl_distill_weight={p[\"kl_distill_weight\"]} '
      f'phase_lrs=[{p[\"phase1_lr\"]}, {p[\"phase2_lr\"]}, {p[\"phase3_lr\"]}, {p[\"phase4_lr\"]}, {p[\"phase5_lr\"]}] '
      f'total_epochs={sum(p[f\"phase{i}_epochs\"] for i in range(1,6))} '
      f'seed={p[\"seed\"]}')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Pre-flight: dead-flag-wiring guard.
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
sys.path.insert(0, 'src')
script = open('scripts/remote_lane_h_v3_jointly_trained_halfframe.sh').read()
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

log "=== Stage 1: JOINT half-frame training (curriculum 0.0->1.0 over first 15%) ==="
log "    Schedule: warmup full-frame [0..5%], aggressive ramp [5%..15%] (200 epochs), endpoint [15%..100%] at 1.0"
log "    Premise: train endpoint == inflate distribution (FIXES Lane D-V3 mismatch)"
log "    Arch: 288K dilated-h64 (NOT 88K DSConv — FIXES Lane V channel bug)"
log "  profile: h_v3_joint_halfframe"
log "  tag:     $TAG"
log "  schedule: 400ep P1 + 1080ep P2 + 200ep P3 + 200ep P4 + 100ep P5 = 1980 epochs"
log "  estimated wall clock on 4090: ~5h (\$1.25 at \$0.25/hr)"

# Smoke-kill metadata sidecar (matches Lane D-V3 with stricter late thresholds
# because the JOINT training should produce a strictly stronger renderer).
cat > "$LOG_DIR/kill_targets.json" <<'EOF'
{
  "phase1_end_epoch": 400,
  "phase1_pixel_l1_max": 12.0,
  "phase2_warmup_end_epoch": 100,
  "phase2_warmup_scorer_max": 50.0,
  "phase2_ramp_mid_epoch": 200,
  "phase2_ramp_mid_scorer_max": 25.0,
  "phase2_endpoint_epoch": 1500,
  "phase2_endpoint_scorer_max": 4.0,
  "phase4_end_epoch": 1880,
  "phase4_end_scorer_max": 2.0,
  "comment": "Lane H-V3 hard kill targets. Curriculum is aggressive (5%-15% ramp); endpoint scorer should track Lane G v3's eval performance closely. Late thresholds tighter than Lane D-V3 because JOINT training avoids the distribution mismatch."
}
EOF

"$PYBIN" -u src/tac/experiments/train_renderer.py \
    --profile h_v3_joint_halfframe \
    --tag "$TAG" \
    --device cuda \
    --video upstream/videos/0.mkv \
    --output-dir "$LOG_DIR/train" \
    --use-qat \
    --no-auth-eval-on-best \
    2>&1 | tee "$LOG_DIR/train.log" | grep -E "^\[(train|eval|masks|phase|best|arch|ep)\]|epoch|Phase|scorer|hf_fires" | tail -200

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
# Lane H-V3 inherits Lane G v3 dilated-h64 arch (base_ch=36, mid_ch=60).
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
# Half-frame archive REQUIRES a renderer trained with mask_half_sim_prob > 0 OR
# use_zoom_flow=True. Our Lane H-V3 profile satisfies BOTH (mask_half_sim_prob
# = 1.0 endpoint + use_zoom_flow=True). Python assertion below confirms.
"$PYBIN" -c "
from tac.profiles import PROFILES
prof_name = 'h_v3_joint_halfframe'
p = PROFILES.get(prof_name)
assert p is not None, f'PROFILES is missing {prof_name}'
assert p.get('mask_half_sim_prob', 0) > 0 or p.get('use_zoom_flow', False), \
    f'profile {prof_name} must have mask_half_sim_prob>0 OR use_zoom_flow=True'
print(f'halfframe-profile-assertion OK: {prof_name}')
"
# NOTE: --profile h_v3_joint_halfframe used in Stage 1 train above (asserted
# by previous python check). build_baseline_archive itself doesn't take
# --profile, just packages the trained renderer.
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

log "=== Stage 3: pose TTO with the new joint-trained half-frame renderer ==="
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

ZOOM_SCALARS=$(find "$LOG_DIR/train" -name "zoom_scalars.pt" 2>/dev/null | head -1)
if [ -n "$ZOOM_SCALARS" ] && [ -f "$ZOOM_SCALARS" ]; then
    cp "$ZOOM_SCALARS" "$LOG_DIR/iter_0/zoom_scalars.pt"
    log "  bundling zoom_scalars.pt ($(stat -c '%s' "$ZOOM_SCALARS") bytes)"
else
    log "  WARN: no zoom_scalars.pt produced by training — inflate will use identity zoom (degraded)"
fi

ARCHIVE="$LOG_DIR/archive_lane_h_v3.zip"
"$PYBIN" -c "
import zipfile, os
src = '$LOG_DIR/iter_0'
dst = '$ARCHIVE'
files = ['renderer.bin', 'masks.mkv', 'optimized_poses.pt']
if os.path.isfile(os.path.join(src, 'zoom_scalars.pt')):
    files.append('zoom_scalars.pt')
# Use ZipInfo + writestr with fixed timestamp for deterministic bytes
# (per preflight check_archive_builders_use_deterministic_zip).
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in files:
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing {p}'
        zi = zipfile.ZipInfo(filename=n, date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        with open(p, 'rb') as fh:
            z.writestr(zi, fh.read())
print(f'archive {dst}: {os.path.getsize(dst)} bytes ({len(files)} files)')
"
ARCHIVE_BYTES=$(stat -c%s "$ARCHIVE" 2>/dev/null || stat -f%z "$ARCHIVE")
[ "$ARCHIVE_BYTES" -gt 0 ] || { echo "FATAL: archive empty" >&2; exit 1; }
[ "$ARCHIVE_BYTES" -lt 5000000 ] || { echo "FATAL: archive >5MB ($ARCHIVE_BYTES) — composition bug" >&2; exit 1; }
log "archive size guard: $ARCHIVE_BYTES bytes (within sanity bounds)"

log "=== Stage 4: contest_auth_eval on Lane H-V3 archive [contest-CUDA] ==="
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

# RESULT_JSON guard.
if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval.log missing RESULT_JSON — silent eval crash. Investigate before reporting score."
    exit 4
fi

log "=== LANE_H_V3_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "  predicted band: [0.55, 0.95] standalone (vs Lane G v3 anchor 1.05)"
log "  cost cap: \$1.50 / 6h on RTX 4090; destroy Vast.ai instance immediately at completion."
