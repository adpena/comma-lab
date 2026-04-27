#!/bin/bash
# Lane A: Pose TTO warm-started from VERIFIED baseline poses (Yousfi+Fridrich's bet).
# Predicted: 2.29 → 0.85-1.10. Single variable: poses.pt only.
#
# Critical: --gt-poses-path points to the VERIFIED baseline poses (the ones that
# scored well in the 2.29 archive), NOT extract_gt_pose_targets (the LANE-B trap).
set -euo pipefail
WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_a_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-a] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Pre-flight: required artifacts present
for f in submissions/baseline_dilated_h64_0_90/renderer.bin \
         submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
         submissions/baseline_dilated_h64_0_90/optimized_poses.bin \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 1: rebuild full-res masks (same as 2.29 baseline) ==="
"$PYBIN" experiments/build_baseline_archive.py \
    --device cuda --crf 50 \
    --output "$LOG_DIR/archive_baseline_seed.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -5

# Extract just masks for optimize_poses input
mkdir -p "$LOG_DIR/extracted"
cd "$LOG_DIR/extracted" && unzip -o "$LOG_DIR/archive_baseline_seed.zip" 2>&1 | tail -3
cd "$WORKSPACE"

log "=== Stage 2: pose TTO with WARM-START from baseline poses ==="
log "   --gt-poses-path = submissions/baseline_dilated_h64_0_90/optimized_poses.pt"
log "   --eval-roundtrip + --posetto-noise-std=0.5 (Fridrich C1 fixes)"
# Determinism: pin seeds + cublas + python hash (CLAUDE.md non-negotiable).
export PYTHONHASHSEED=1234
"$PYBIN" -u experiments/optimize_poses.py \
    --checkpoint submissions/baseline_dilated_h64_0_90/renderer.bin \
    --masks "$LOG_DIR/extracted/masks.mkv" \
    --gt-poses-path submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
    --device cuda \
    --steps 500 \
    --batch-pairs 8 \
    --eval-roundtrip \
    --posetto-noise-std 0.5 \
    --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/optimize_poses.log" | tail -30

# Validate: did optimize_poses produce the file?
[ -f "$LOG_DIR/optimized_poses.bin" ] || { echo "FATAL: optimize_poses didn't produce optimized_poses.bin"; exit 2; }
log "  produced optimized_poses.bin ($(stat -c '%s' "$LOG_DIR/optimized_poses.bin") bytes)"

log "=== Stage 3: build NEW archive (renderer + masks + NEW poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp submissions/baseline_dilated_h64_0_90/renderer.bin "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
# Use the .pt format (matches the canonical archive structure)
[ -f "$LOG_DIR/optimized_poses.pt" ] && cp "$LOG_DIR/optimized_poses.pt" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_a.zip"
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

log "=== Stage 4: contest_auth_eval on Lane A archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -15

log "=== LANE_A_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
