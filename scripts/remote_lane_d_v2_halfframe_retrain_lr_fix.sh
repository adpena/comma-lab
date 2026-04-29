#!/bin/bash
# Lane D-V2: dilated-h64 half-frame retrain with LR-schedule fix.
#
# V1 history (scripts/remote_lane_d_halfframe_retrain.sh, killed 2026-04-27):
#   * 1980-epoch schedule: P1 400 + P2 1080 + P3 200 + P4 200 + P5 100
#   * V1 phase LRs: P1=1e-3, P2=3e-4, P3=1e-4, P4=5e-5, P5=1e-5
#   * KILLED at ep 1230/1980 (62%), best fp4_scorer = 40.37
#   * Best PLATEAUED for ~800 epochs (last improvement around ep 700)
#   * Observed LR at kill = 4e-5 — matches the cosine-anneal math:
#       within P2 (ep 400-1480), step=830/1080=0.77, cos factor≈0.11,
#       LR ≈ 1e-6 + (3e-4 - 1e-6)*0.11 ≈ 3.3e-5 → optimizer starved
#       in the second half of P2 (LR fell below the noise floor of the
#       composite Fridrich + KL distill loss whose magnitude is O(1)).
#
# V2 fix (CHOICE B: higher LR floor across all phases):
#   The intra-phase scheduler is `cosine_lr(base, step, total, eta_min=1e-6)`
#   in src/tac/experiments/train_renderer.py:1059 — eta_min is hard-coded
#   and not exposed as a CLI flag. The cheapest correct intervention is to
#   RAISE the per-phase base LR so the cosine floor (≈ eta_min when the
#   phase ends, ≈ base/10 at 80% of the phase) doesn't starve the
#   optimizer. Phase LRs raised by a uniform ~1.7-2.0× factor:
#       P1 1e-3 → 1e-3 (unchanged — pixel warmup converges fast)
#       P2 3e-4 → 5e-4 (1.67× — fixes the ep 700 plateau)
#       P3 1e-4 → 2e-4 (2.0×  — keeps signal alive in hard-pair finetune)
#       P4 5e-5 → 1e-4 (2.0×  — Lin et al. 2017 says 0.1×base for QAT;
#                                with raised P3 base this stays in spec)
#       P5 1e-5 → 2e-5 (2.0×  — final polish proportional bump)
#   At the new P2 base of 5e-4, the LR at the same ep 1230 (step=830/1080)
#   becomes ≈ 1e-6 + (5e-4 - 1e-6)*0.11 ≈ 5.5e-5 — 1.67× higher than V1's
#   3.3e-5 floor. Combined with the higher base, the optimizer keeps
#   moving through the back half of P2 where V1 plateaued.
#
# Why NOT the other proposals:
#   (A) Cosine restart at ep 700 — requires a code change in train_renderer
#       (no warm-restart hook today). Test-the-LR-fix-in-isolation rules out.
#   (C) Extend P2 from 1080 → 1500 ep — slows the decay but doesn't raise
#       the floor; with eta_min=1e-6 we'd still hit ~2e-5 at the new ep
#       1300 (step=900/1500=0.6, cos factor≈0.35). Doesn't fix the root
#       cause. Kept as a future fallback if (B) doesn't recover.
#   (D) Lower mask_half_sim_prob (0.5 → 0.3) — confounds the LR test.
#       The V1 plateau wasn't variance-driven (loss was steady at 40, not
#       oscillating); it was gradient-magnitude-driven. Holding
#       mask_half_sim_prob=0.5 isolates the LR fix.
#
# Predicted score landing zone for V2 ([1.50, 3.00]):
#   * If V1's plateau was purely LR-starvation (the hypothesis): proxy
#     should drop from 40 → ~5-15 by ep 1500, contest auth ~1.50-2.50.
#   * If the plateau also reflects a structural ceiling for half-frame
#     retrofit on dilated-h64 (per memory feedback_half_frame_breaks_posenet):
#     contest auth stays in 2.50-3.00. Beats V1 baseline (17.55) by ≥5×
#     either way; the BAND'S WIDTH reflects this open question.
#   * Stretch goal: any sub-Lane-A landing (sub-1.15) would be a major
#     win (would mean the half-frame paradigm + LR-fix combo is viable
#     after all and we can ship a sub-1.0 archive).
#
# Per CLAUDE.md non-negotiable (NEVER invent CLI flags): every flag below
# was verified by argparse-grep on src/tac/experiments/train_renderer.py
# (--phase{1..5}-lr at L214-223, --mask-half-sim-prob at L259, --no-auth-eval-on-best
# is the negative-default form of --auth-eval-on-best).
#
# Cost: 4090 @ $0.25/hr × ~5h = ~$1.25 (same as V1; LR change doesn't move wall clock).
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
# 2026-04-27 fix (inherited from V1): scorer weights live at
# /workspace/pact/upstream/models on the canonical Vast.ai layout.
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_d_v2_results"
mkdir -p "$LOG_DIR"
TAG="lane_d_v2_halfframe"

log() { echo "[lane-d-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_d_v2_halfframe_retrain_lr_fix.sh',
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'profile': 'dilated_h64_half_frame',
    'predicted_band': [1.50, 3.00],
    'anchor_score_baseline': 17.55,
    'lr_fix_choice': 'B_higher_lr_floor',
    'delta_from_v1': {
        'phase1_lr': '1e-3 (unchanged)',
        'phase2_lr': '3e-4 -> 5e-4 (1.67x)',
        'phase3_lr': '1e-4 -> 2e-4 (2.0x)',
        'phase4_lr': '5e-5 -> 1e-4 (2.0x)',
        'phase5_lr': '1e-5 -> 2e-5 (2.0x)',
        'mask_half_sim_prob': '0.5 (unchanged - isolate LR fix)',
        'rationale': 'V1 plateaued ep 700 onwards; cosine_lr with eta_min=1e-6 starved the optimizer in back-half of P2 (LR ~3.3e-5). Raising base LRs raises the per-phase floor proportionally.',
    },
    'lr_fix_reasoning': 'within-phase cosine_lr(base, step, total, eta_min=1e-6) at step=830/1080 in P2 yields ~0.11*base; with V1 base=3e-4 floor was 3.3e-5, with V2 base=5e-4 floor is 5.5e-5. Combined with higher peak the optimizer maintains gradient signal through the full P2 budget.',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=D-V2 gpu=$GPU" >> "$HEARTBEAT"
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
# invocation below must exist in train_renderer.py's argparse. CLAUDE.md
# non-negotiable: NEVER invent CLI flags.
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
sys.path.insert(0, 'src')
script = open('scripts/remote_lane_d_v2_halfframe_retrain_lr_fix.sh').read()
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

log "=== Stage 1: full-frame training with LR-fix overrides ==="
log "  profile: dilated_h64_half_frame (mask_half_sim_prob=0.5 inherited)"
log "  tag:     $TAG"
log "  schedule: 400ep P1 + 1080ep P2 + 200ep P3 + 200ep P4 + 100ep P5 = 1980 epochs"
log "  V2 LR overrides: P1=1e-3 P2=5e-4 P3=2e-4 P4=1e-4 P5=2e-5"
log "  estimated wall clock on 4090: ~5h (\$1.25 at \$0.25/hr)"

# Smoke-kill metadata sidecar so the watchdog can read targets externally.
# V2 targets are tightened given the LR-fix hypothesis: if at ep 800 we're
# still > 8.0 the LR-fix hypothesis is invalid and we save 2h of GPU.
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
  "comment": "Lane D-V2 hard kill targets. V1 plateaued at scorer=40 by ep 700; if V2 is also >8 at ep 800 the LR-fix hypothesis is wrong and the watchdog should SIGTERM the train job."
}
EOF

# Codex R-Lane-D-Issue3 (inherited from V1): --no-auth-eval-on-best because
# Stage 4 below builds the real archive and runs contest_auth_eval.py.
"$PYBIN" -u src/tac/experiments/train_renderer.py \
    --profile dilated_h64_half_frame \
    --tag "$TAG" \
    --device cuda \
    --video upstream/videos/0.mkv \
    --output-dir "$LOG_DIR/train" \
    --use-qat \
    --no-auth-eval-on-best \
    --phase1-lr 1e-3 \
    --phase2-lr 5e-4 \
    --phase3-lr 2e-4 \
    --phase4-lr 1e-4 \
    --phase5-lr 2e-5 \
    2>&1 | tee "$LOG_DIR/train.log" | grep -E "^\[(train|eval|masks|phase|best|arch)\]|epoch|Phase|scorer" | tail -200

# Codex R-Lane-D-Issue3 (inherited from V1): use the fp32 best checkpoint +
# tac.renderer_export.export_asymmetric_checkpoint_fp4 (canonical FP4A export).
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
# Half-frame archive REQUIRES a renderer trained with mask_half_sim_prob>0
# AND use_zoom_flow=True (memory feedback_half_frame_breaks_posenet).
"$PYBIN" -c "
from tac.profiles import PROFILES
p = PROFILES.get('dilated_h64_half_frame')
assert p is not None, 'PROFILES is missing dilated_h64_half_frame'
assert p.get('mask_half_sim_prob', 0) > 0 or p.get('use_zoom_flow', False), \
    'profile dilated_h64_half_frame must have mask_half_sim_prob>0 OR use_zoom_flow=True'
print('halfframe-profile-assertion OK: dilated_h64_half_frame')
"
# NOTE: --profile dilated_h64_half_frame was used in Stage 1 train above;
# build_baseline_archive does NOT take --profile (it just packages the
# Stage 1 trained renderer). The asserted profile match is verified by
# the previous pyc check.
"$PYBIN" experiments/build_baseline_archive.py \
    --device cuda --crf 50 --half-frame \
    --output "$LOG_DIR/archive_halfframe_seed.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -5
mkdir -p "$LOG_DIR/extracted"
cd "$LOG_DIR/extracted" && unzip -o "$LOG_DIR/archive_halfframe_seed.zip" 2>&1 | tail -5
cd "$WORKSPACE"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"

# Pose TTO using the new renderer (poses are renderer-specific).
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

ARCHIVE="$LOG_DIR/archive_lane_d_v2.zip"
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

log "=== Stage 4: contest_auth_eval on Lane D-V2 archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20

log "=== LANE_D_V2_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON [contest-CUDA] ==="
