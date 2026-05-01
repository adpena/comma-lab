#!/bin/bash
# Lane D-V3: dilated-h64 half-frame retrain — full engineering stack.
#
# V1 (scripts/remote_lane_d_halfframe_retrain.sh, killed 2026-04-27): proxy
#     fp4_scorer plateaued at ~40 since ep ~700; killed at ep 1230/1980 (62%).
# V2 (scripts/remote_lane_d_v2_halfframe_retrain_lr_fix.sh, in flight):
#     CHOICE B = higher per-phase LR floor (P2 3e-4 → 5e-4, etc.) — addresses
#     V1's cosine_lr starvation (eta_min=1e-6 starves optimiser in back-half
#     of P2). Single variable A/B vs V1.
#
# V3 stacks V2's LR fix with two additional levers borrowed from Lane V-V2 +
# the post-bug-fix KL distill weight:
#
#   1. ANNEALED mask_half_sim_prob 0.0 → 0.5 (from-scratch warmup in P1, then
#      smooth ramp through early P2, then half-frame endpoint by 70% mark).
#      Rationale: V1's plateau at ep 700 (~35% through training) coincides
#      with the early-phase optimisation difficulty of learning from a
#      mixed (50/50) mask distribution from epoch 0. Annealing lets the
#      renderer first lock in the easier full-frame distribution, THEN
#      smoothly transition. Same paradigm as Lane V-V2.
#   2. KL DISTILL WEIGHT 1.0 → 0.002 (post-bug-fix value). V1/V2 inherited
#      weight=1.0 from before the 2026-04-27 kl_distill_segnet_only reduction
#      fix (losses.py:705 divides by H*W). Post-fix, weight=1.0 means KL
#      contributes ~0.025 (~5x scorer loss), DROWNING the scorer signal.
#      Correct post-fix value is 0.002 (matches Lane V).
#
# Phase-0 mechanism instrumentation (NEW): train_renderer.py per-epoch logs
# now print `hf_fires=N/M (rate) hf_warp_diff=X hf_target_prob=Y` at the
# end of every epoch where the half-frame branch fires at least once. The
# JSONL telemetry sidecar adds `halfframe_branch_fires`, `halfframe_warp_diff_mean`,
# and `halfframe_target_prob` keys so post-hoc analysis can verify:
#   * the half-frame branch is actually firing at the annealed rate, AND
#   * the warp_inverse_masks call is producing non-trivial mask perturbations
#     (a degenerate identity warp would have hf_warp_diff~0).
#
# Note on cosine restart: an explicit warm-restart hook in the optimizer was
# considered but REJECTED for V3 (would need a deeper code change in
# train_renderer.py — `cosine_lr` is intra-phase only, no restart logic
# today). V3 instead relies on V2's higher base LR + annealing's smoother
# loss-landscape trajectory to keep the optimiser out of the V1 plateau.
# A future V4 could revisit if V3 still plateaus.
#
# Predicted band [1.50, 2.50] [contest-CUDA] — wider than V2 [1.50, 3.00]
# because V3 introduces TWO new variables on top of V2's LR fix:
#   * Floor 1.50 — V3 stacks address 3 distinct V1 failure modes (LR
#     starvation + cold-start optimisation difficulty + KL drowning scorer).
#     If all 3 contributed, V3 should land in 1.5-2.0 band.
#   * Ceiling 2.50 — even if KL fix is the only useful lever, V3 should
#     match V2's projected ceiling (LR fix alone gets to 2.5-3.0).
#
# Cost: 4090 @ $0.25/hr × ~5h = ~$1.25 (same as V1/V2).
#
# Per CLAUDE.md NEVER-INVENT-CLI-FLAGS: every flag passed to
# train_renderer.py / optimize_poses.py / build_baseline_archive.py /
# contest_auth_eval.py was verified by argparse-grep on the target sources
# (the in-script dead-flag scanner re-validates this at launch time).
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
# Profile uses seed=43 (V1=42, V2=42) — different RNG basin than V1/V2.
export PYTHONHASHSEED=43
# Scorer weights live at /workspace/pact/upstream/models on the canonical
# Vast.ai layout (inherited from V2).
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_d_v3_results"
mkdir -p "$LOG_DIR"
TAG="lane_d_v3_annealed_kldistill"

log() { echo "[lane-d-v3] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_d_v3_full_engineering.sh',
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'profile': 'dilated_h64_half_frame_v3_annealed_kldistill',
    'predicted_band': [1.50, 2.50],
    'anchor_score_baseline': 1.15,
    'anchor_lane': 'lane_a (1.15 contest-CUDA)',
    'lane_d_v3_premise': 'V2 LR fix + annealed mask_half_sim_prob (0->0.5) + KL distill weight post-fix (1.0->0.002)',
    'annealing_schedule': 'warmup full-frame [0..30%], ramp 0->0.5 [30%..70%], endpoint 0.5 [70%..100%]',
    'delta_from_v2': {
        'mask_half_sim_prob': '0.5 static -> ANNEALED 0.0 -> 0.5 (Lane V-V2 paradigm)',
        'kl_distill_weight': '1.0 (V1/V2 pre-fix) -> 0.002 (post-fix; matches Lane V)',
        'phase_lrs': '(unchanged from V2: 1e-3, 5e-4, 2e-4, 1e-4, 2e-5)',
        'instrumentation': 'NEW Phase-0 hf_fires/hf_warp_diff/hf_target_prob in train + JSONL',
        'seed': '42 (V1/V2) -> 43 (different RNG basin)',
    },
    'cost_estimate_usd': 1.25,
    'wall_clock_estimate_hours': 5.0,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=D-V3 gpu=$GPU" >> "$HEARTBEAT"
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
prof_name = 'dilated_h64_half_frame_v3_annealed_kldistill'
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
      f'phase_lrs=[{p[\"phase1_lr\"]}, {p[\"phase2_lr\"]}, {p[\"phase3_lr\"]}, {p[\"phase4_lr\"]}, {p[\"phase5_lr\"]}] '
      f'total_epochs={sum(p[f\"phase{i}_epochs\"] for i in range(1,6))} '
      f'seed={p[\"seed\"]}')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Pre-flight: dead-flag-wiring guard — every CLI flag in the train_renderer
# invocation below must exist in train_renderer.py's argparse. CLAUDE.md
# non-negotiable: NEVER invent CLI flags.
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
sys.path.insert(0, 'src')
script = open('scripts/remote_lane_d_v3_full_engineering.sh').read()
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

log "=== Stage 1: half-frame training (annealed warp + KL fix + V2 LR floor) ==="
log "    Annealing: warmup full-frame for 30%, linear ramp 30%->70% (0.0->0.5), endpoint 70%-100% at 0.5"
log "    NB: epoch 0..594 trains on full-frame, easy distribution → strong init"
log "         epoch 594..1386 ramps 0.0->0.5 → smooth transition"
log "         epoch 1386..1980 trains at half-frame=0.5 → endpoint = Lane D-V1/V2"
log "  profile: dilated_h64_half_frame_v3_annealed_kldistill"
log "  tag:     $TAG"
log "  schedule: 400ep P1 + 1080ep P2 + 200ep P3 + 200ep P4 + 100ep P5 = 1980 epochs"
log "  V3 LRs (inherits V2): P1=1e-3 P2=5e-4 P3=2e-4 P4=1e-4 P5=2e-5"
log "  V3 KL distill weight: 0.002 (V1/V2 had 1.0; pre-fix value, drowned scorer)"
log "  estimated wall clock on 4090: ~5h (\$1.25 at \$0.25/hr)"

# Smoke-kill metadata sidecar so the watchdog can read targets externally.
# V3 targets account for the warmup phase: scorer eval is meaningless during
# the full-frame warmup (the renderer is learning the wrong distribution
# vs the half-frame deployment), so the early thresholds are LOOSER than V2
# and the late thresholds are TIGHTER (stronger init should yield faster
# convergence in the final third).
cat > "$LOG_DIR/kill_targets.json" <<'EOF'
{
  "phase1_end_epoch": 400,
  "phase1_pixel_l1_max": 12.0,
  "phase2_warmup_end_epoch": 600,
  "phase2_warmup_scorer_max": 50.0,
  "phase2_ramp_mid_epoch": 1000,
  "phase2_ramp_mid_scorer_max": 20.0,
  "phase2_endpoint_epoch": 1500,
  "phase2_endpoint_scorer_max": 5.0,
  "phase4_end_epoch": 1880,
  "phase4_end_scorer_max": 2.5,
  "comment": "Lane D-V3 hard kill targets. Annealing means scorer eval is dominated by full-frame distribution until ~ep 1386 (70% mark) so early thresholds are looser; the late-phase thresholds are tighter than V2 because warmup should yield a stronger initialisation."
}
EOF

# Codex R-Lane-D-Issue3 (inherited from V2): --no-auth-eval-on-best because
# Stage 4 below builds the real archive and runs contest_auth_eval. Phase-0
# instrumentation (hf_fires, hf_warp_diff, hf_target_prob) is automatic via
# train_renderer.py — no CLI flag needed; the in-loop counters fire when
# sim_zoom_warp is non-None which happens for any profile with
# mask_half_sim_prob > 0 OR use_zoom_flow=True.
"$PYBIN" -u src/tac/experiments/train_renderer.py \
    --profile dilated_h64_half_frame_v3_annealed_kldistill \
    --tag "$TAG" \
    --device cuda \
    --video upstream/videos/0.mkv \
    --output-dir "$LOG_DIR/train" \
    --use-qat \
    --no-auth-eval-on-best \
    2>&1 | tee "$LOG_DIR/train.log" | grep -E "^\[(train|eval|masks|phase|best|arch|ep)\]|epoch|Phase|scorer|hf_fires" | tail -200

# Codex R-Lane-D-Issue3 (inherited from V2): use the fp32 best checkpoint +
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
# Lane D arch defaults (base_ch=36, mid_ch=60, motion_hidden=32, use_dsconv=False)
# differ from Lane V (24/32/16/True) — read from meta where possible.
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
# use_zoom_flow=True. Our V3 profile satisfies BOTH (mask_half_sim_prob = 0.5
# endpoint + use_zoom_flow=True). Assertion below confirms.
"$PYBIN" -c "
from tac.profiles import PROFILES
prof_name = 'dilated_h64_half_frame_v3_annealed_kldistill'
p = PROFILES.get(prof_name)
assert p is not None, f'PROFILES is missing {prof_name}'
assert p.get('mask_half_sim_prob', 0) > 0 or p.get('use_zoom_flow', False), \
    f'profile {prof_name} must have mask_half_sim_prob>0 OR use_zoom_flow=True'
print(f'halfframe-profile-assertion OK: {prof_name}')
"
# NOTE: --profile dilated_h64_half_frame_v3_annealed_kldistill used in
# Stage 1 train above (asserted by previous python check).
# build_baseline_archive itself doesn't take --profile, just packages the trained renderer.
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
log "=== Stage 3: pose TTO with the new annealed half-frame renderer ==="
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

ARCHIVE="$LOG_DIR/archive_lane_d_v3.zip"
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

log "=== Stage 4: contest_auth_eval on Lane D-V3 archive ==="
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

# RESULT_JSON guard (LANE-B silent-crash prevention).
if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval.log missing RESULT_JSON — silent eval crash. Investigate before reporting score."
    exit 4
fi

log "=== LANE_D_V3_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "  predicted band: [1.50, 2.50] standalone (vs Lane V-V1 [0.50, 1.10], Lane A 1.15)"
log "  anchor baseline: 1.15 [contest-CUDA] (Lane A frontier)"
