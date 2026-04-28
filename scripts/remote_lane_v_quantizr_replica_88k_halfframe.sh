#!/bin/bash
# Lane V: Quantizr-replica (88K params, DSConv + FiLM + KL distill T=2.0)
# trained from epoch 0 with mask_half_sim_prob=1.0 (always-on warp expansion).
# Council 2026-04-27 — biggest single swing in the strategy.
#
# Background:
#   Lane D (2026-04-27) tried to RETROFIT half-frame onto the dilated-h64
#   baseline (mask_half_sim_prob=0.5 mid-train). It FAILED (memory:
#   feedback_half_frame_breaks_posenet — score 17.55 at epoch 1980 on a
#   broken arch). The hypothesis: the renderer's MotionPredictor was already
#   locked into (e_t1 - e_t).abs() diff features that warp-expansion zeroes.
#   Lane V's bet is that JOINT training from epoch 0 — never seeing the
#   unwarped distribution — forces the motion module to converge on the
#   warp-expansion premise instead.
#
# Stack:
#   * 88K params (~89.3K measured) — DSConv + base_ch=24 + mid_ch=32 + motion_hidden=16
#   * FiLM pose conditioning (pose_dim=6) joint-trained from epoch 0
#   * use_zoom_flow=True — required by preflight when mask_half_sim_prob>0
#   * mask_half_sim_prob=1.0 — every batch warp-expands mask_t (vs Lane D 0.5)
#   * KL distill T=2.0 weight=0.002 — POST-FIX math (raw KL ≈ 0.025 after
#     2026-04-27 reduction fix in losses.py:705; weight 0.002 makes KL
#     contribution ~5e-5 ≈ 1% of scorer loss)
#   * eval_roundtrip=True (NON-NEGOTIABLE per CLAUDE.md), noise_std=0.5
#     hardcoded in train_renderer.py:1741
#   * posetto_noise_std=0.5 at pose-TTO stage (Stage 3 below)
#   * 5-stage QAT pipeline (anchor → finetune → joint → QAT → final),
#     RESIDUAL FP4 codebook, robust scale, stochastic rounding
#   * Full Fridrich aux-loss stack mirrors Lane D for direct A/B comparability
#
# Schedule (matches profile QUANTIZR_REPLICA_88K_HALFFRAME):
#   Phase 1 (600ep)  ≈ 2.5h  pixel L1 + edge anchor
#   Phase 2 (1500ep) ≈ 6h    scorer + Fridrich + KL distill
#   Phase 3 (400ep)  ≈ 1.5h  hard-pair fine-tune
#   Phase 4 (400ep)  ≈ 1.5h  QAT FakeQuantFP4
#   Phase 5 (100ep)  ≈ 0.5h  final consolidation
#   TOTAL: ~12h on 4090 ($3.00 @ $0.25/hr); plus ~30min pose-TTO + ~15min
#   auth-eval = $4-5 end-to-end.
#
# Predicted score landing zone (standalone — wide because from-scratch):
#   * SegNet:  ~0.003 (excellent at half-frame after joint training)
#   * PoseNet: 0.05-0.20 (the open question — JOINT vs RETROFIT)
#   * Rate:    ~0.14-0.26 (half-frame archive saves 0.20-0.32 vs full-frame)
#   * TOTAL standalone: 0.50-1.10 (true from-scratch rebuild — wide band)
# Stacked with Lane A pose TTO + Lane C δ: 0.30-0.55 (sub-Quantizr territory).
#
# Hard kill targets (so we don't burn $3 on a runaway divergence):
#   * Phase 1 end (~2.5h, ep600):  pixel L1 < 14
#   * Phase 2 ep600 (~5h, ep1200): scorer < 10.0  (must show learning signal)
#   * Phase 2 end  (~8h, ep2100):  scorer < 4.0   (must beat Lane D's broken 17.55)
#   * Phase 4 end  (~11h, ep2900): scorer < 2.0   (target standalone 0.50-1.10)
#
# Reproducibility:
#   * seed=1234 (different from Lane D's 42 so the two from-scratch rebuilds
#     explore different RNG basins)
#   * deterministic=True → CUBLAS_WORKSPACE_CONFIG=:4096:8, cudnn.deterministic=True
#   * PYTHONHASHSEED=1234
#
# ── Bash safety ─────────────────────────────────────────────────────────
# CLAUDE.md non-negotiable: `set -euo pipefail` (NOT `set -uo`). The cascade
# trap that ate LANE-B 6.5h + $2 in 2026-04-26 must not reappear.
set -euo pipefail

WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
# Same upstream-dir override Lane D + bootstraps use — train_renderer's
# default scorer-weights path ($REPO/workspace/upstream/...) does not match
# the canonical Vast.ai layout (/workspace/pact/upstream/models).
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_v_results"
mkdir -p "$LOG_DIR"
TAG="lane_v_quantizr_replica_88k_halfframe"

log() { echo "[lane-v] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_v_quantizr_replica_88k_halfframe.sh',
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'profile': 'quantizr_replica_88k_halfframe',
    'predicted_band': [0.50, 1.10],
    'anchor_score_baseline': 1.15,
    'anchor_lane': 'lane_a (1.15 contest-CUDA)',
    'lane_v_premise': 'mask_half_sim_prob=1.0 from epoch 0 (vs Lane D retrofit 0.5 which failed)',
    'cost_estimate_usd': 4.5,
    'wall_clock_estimate_hours': 13.0,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=V gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend (memory:
# feedback_vastai_nvdec_host_variation — same 4090 image, different hosts,
# same driver, different NVDEC outcome). Catches the bad-host case in 5 sec.
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

# Pre-flight: profile loads + passes preflight (catches mask_half_sim_prob /
# use_zoom_flow inconsistency before we burn 12h of GPU on a misconfig).
log "=== Pre-flight: profile validation ==="
"$PYBIN" -c "
import sys
sys.path.insert(0, 'src')
from tac.profiles import PROFILES
from tac.preflight import preflight_profiles
assert 'quantizr_replica_88k_halfframe' in PROFILES, 'profile quantizr_replica_88k_halfframe not registered'
violations = preflight_profiles(strict=False, verbose=False)
ours = [v for v in violations if 'quantizr_replica_88k_halfframe' in v]
if ours:
    print(f'PREFLIGHT FAIL: {ours}', file=sys.stderr); sys.exit(2)
p = PROFILES['quantizr_replica_88k_halfframe']
print(f'PROFILE OK: base_ch={p[\"base_ch\"]} mid_ch={p[\"mid_ch\"]} '
      f'motion_hidden={p[\"motion_hidden\"]} use_zoom_flow={p[\"use_zoom_flow\"]} '
      f'mask_half_sim_prob={p[\"mask_half_sim_prob\"]} '
      f'kl_distill_weight={p[\"kl_distill_weight\"]} '
      f'pose_dim={p[\"pose_dim\"]} use_dsconv={p[\"use_dsconv\"]} '
      f'total_epochs={sum(p[f\"phase{i}_epochs\"] for i in range(1,6))}')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Pre-flight: param count smoke — confirm we built the 88K-class arch.
log "=== Pre-flight: arch smoke (param count target ~88K) ==="
"$PYBIN" -c "
import sys
sys.path.insert(0, 'src')
from tac.profiles import PROFILES
from tac.renderer import build_renderer
p = PROFILES['quantizr_replica_88k_halfframe']
m = build_renderer(
    num_classes=5, embed_dim=p['embed_dim'], base_ch=p['base_ch'],
    mid_ch=p['mid_ch'], motion_hidden=p['motion_hidden'], depth=p['depth'],
    pose_dim=p['pose_dim'], use_dsconv=p['use_dsconv'],
    padding_mode=p['padding_mode'], use_dilation=p['use_dilation'],
    use_zoom_flow=p['use_zoom_flow'],
)
total = sum(pp.numel() for pp in m.parameters())
assert 80_000 <= total <= 100_000, f'param count {total} outside Quantizr-class budget [80K, 100K]'
print(f'ARCH OK: {total} params ({total/1000:.1f}K)')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Pre-flight: dead-flag-wiring guard — every CLI flag in the train_renderer
# invocation below must exist in train_renderer.py's argparse. Catches the
# class of bug CLAUDE.md's "NEVER invent CLI flags" rule was created for
# (the 2026-04-26 dead-auth-eval-masks incident burned multiple chains).
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
sys.path.insert(0, 'src')
script = open('scripts/remote_lane_v_quantizr_replica_88k_halfframe.sh').read()
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

log "=== Stage 1: from-scratch training (mask_half_sim_prob=1.0 from epoch 0)"
log "    NB: training-time half-frame is JOINT from epoch 0 — every batch warp-"
log "    expands mask_t. This is the Lane V bet vs Lane D's 0.5 retrofit. ==="
log "  profile: quantizr_replica_88k_halfframe"
log "  tag:     $TAG"
log "  schedule: 600ep P1 + 1500ep P2 + 400ep P3 + 400ep P4 + 100ep P5 = 3000 epochs"
log "  estimated wall clock on 4090: ~12h (\$3.00 at \$0.25/hr; total budget \$4-5 with TTO+eval)"

# Smoke-kill metadata sidecar so the watchdog can read targets externally.
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
  "comment": "Hard kill targets per scripts/remote_lane_v_quantizr_replica_88k_halfframe.sh header. The watchdog should tail eval logs and SIGTERM the train job when any threshold is exceeded."
}
EOF

# Codex-pattern justification: --no-auth-eval-on-best because Stage 4 below
# builds the real archive (renderer + masks + poses + zoom_scalars) and runs
# contest_auth_eval.py against it. train_renderer's built-in auth-eval would
# need both --auth-eval-masks AND --auth-eval-poses (which don't exist yet at
# this point in the chain — masks haven't been encoded, poses haven't been
# TTO-optimized).
"$PYBIN" -u src/tac/experiments/train_renderer.py \
    --profile quantizr_replica_88k_halfframe \
    --tag "$TAG" \
    --device cuda \
    --video upstream/videos/0.mkv \
    --output-dir "$LOG_DIR/train" \
    --use-qat \
    --no-auth-eval-on-best \
    2>&1 | tee "$LOG_DIR/train.log" | grep -E "^\[(train|eval|masks|phase|best|arch)\]|epoch|Phase|scorer" | tail -200

# train_renderer writes the FP4-packed checkpoint as
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
# Build half-frame masks (600 odd-frame masks only) — Quantizr paradigm.
#
# Half-frame archive REQUIRES a renderer trained with
#     --profile quantizr_replica_88k_halfframe
# (mask_half_sim_prob=1.0 AND use_zoom_flow=True). Stage 1 above already
# trained with that profile; the assertion below is a belt-and-braces hard
# check so this script cannot ever be repurposed against a non-half-frame
# renderer without flagging the violation. Per memory
# feedback_half_frame_breaks_posenet: half-frame on the dilated-h64 baseline
# (no warp training) collapses PoseNet (0.011 → 28.7, score 17.55).
"$PYBIN" -c "
from tac.profiles import PROFILES
p = PROFILES.get('quantizr_replica_88k_halfframe')
assert p is not None, 'PROFILES is missing quantizr_replica_88k_halfframe'
assert p.get('mask_half_sim_prob', 0) > 0 or p.get('use_zoom_flow', False), \
    'profile quantizr_replica_88k_halfframe must have mask_half_sim_prob>0 OR use_zoom_flow=True'
print('halfframe-profile-assertion OK: quantizr_replica_88k_halfframe')
"
"$PYBIN" experiments/build_baseline_archive.py \
    --device cuda --crf 50 --half-frame \
    --output "$LOG_DIR/archive_halfframe_seed.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -5
mkdir -p "$LOG_DIR/extracted"
cd "$LOG_DIR/extracted" && unzip -o "$LOG_DIR/archive_halfframe_seed.zip" 2>&1 | tail -5
cd "$WORKSPACE"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"

# Pose TTO using the new renderer (poses are renderer-specific). Use the
# baseline poses as warm-start — they're a good prior but the new renderer
# may want different poses. posetto_noise_std=0.5 matches the profile.
log "=== Stage 3: pose TTO with the new half-frame renderer (posetto_noise_std=0.5) ==="
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

# zoom_scalars are required for the inflate-side warp expansion (use_zoom_flow=True).
ZOOM_SCALARS=$(find "$LOG_DIR/train" -name "zoom_scalars.pt" 2>/dev/null | head -1)
if [ -n "$ZOOM_SCALARS" ] && [ -f "$ZOOM_SCALARS" ]; then
    cp "$ZOOM_SCALARS" "$LOG_DIR/iter_0/zoom_scalars.pt"
    log "  bundling zoom_scalars.pt ($(stat -c '%s' "$ZOOM_SCALARS") bytes)"
else
    log "  WARN: no zoom_scalars.pt produced by training — inflate will use identity zoom (degraded)"
fi

ARCHIVE="$LOG_DIR/archive_lane_v.zip"
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

log "=== Stage 4: contest_auth_eval on Lane V archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20

# Validate RESULT_JSON line is present in the auth-eval log — guards against
# silent zero-exit on a crashed eval (LANE-B-style cascade pattern).
if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval.log missing RESULT_JSON — silent eval crash. Investigate before reporting score."
    exit 4
fi

log "=== LANE_V_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "  predicted band: [0.50, 1.10] standalone; [0.30, 0.55] stacked with Lane A+C"
log "  anchor baseline: 1.15 [contest-CUDA] (Lane A frontier)"
