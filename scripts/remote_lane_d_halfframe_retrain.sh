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
set -euo pipefail
WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234

LOG_DIR="$WORKSPACE/lane_d_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-d] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Pre-flight: required artifacts present
for f in upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
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

log "=== Stage 1: full-frame training (NB: training uses full-frame masks; the"
log "    --mask-half-sim-prob=0.5 setting injects warp-reconstructed even-frame"
log "    masks into 50% of batches — no separate half-frame mask precompute needed) ==="
log "  profile: dilated_h64_half_frame"
log "  schedule: 400ep P1 + 1080ep P2 + 200ep P3 + 200ep P4 + 100ep P5 = 1980 epochs"
log "  estimated wall clock on 4090: ~5h ($1.25 at \$0.25/hr)"

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

"$PYBIN" -u src/tac/experiments/train_renderer.py \
    --profile dilated_h64_half_frame \
    --device cuda \
    --video upstream/videos/0.mkv \
    --output-dir "$LOG_DIR/train" \
    --use-qat \
    2>&1 | tee "$LOG_DIR/train.log" | grep -E "^\[(train|eval|masks|phase)\]|epoch|Phase|scorer" | tail -200

# Find the best FP4 checkpoint produced by the 5-phase schedule
BEST_BIN=$(find "$LOG_DIR/train" -name "renderer_best_fp4.bin" -o -name "renderer_*_fp4.bin" 2>/dev/null | head -1)
[ -n "$BEST_BIN" ] && [ -f "$BEST_BIN" ] || {
    echo "FATAL: train_renderer didn't produce a *_fp4.bin checkpoint" >&2
    ls -la "$LOG_DIR/train/" >&2
    exit 2
}
log "  best FP4 checkpoint: $BEST_BIN ($(stat -c '%s' "$BEST_BIN") bytes)"

log "=== Stage 2: build half-frame archive (renderer + 600 odd masks + poses + zoom_scalars) ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$BEST_BIN" "$LOG_DIR/iter_0/renderer.bin"

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
