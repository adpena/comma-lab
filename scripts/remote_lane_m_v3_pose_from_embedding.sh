#!/bin/bash
# Lane M-V3 (Path A) — pose-from-embedding distillation. Anchored on Lane A
# (1.15 [contest-CUDA]). Replaces the ~15 KB optimized_poses.pt with a
# distilled MLP (~1-2 KB FP16) + 0-byte sentinel that predicts per-pair
# 6-DOF poses at INFLATE TIME from mask features alone (zero PoseNet input).
#
# V3 differs from V1 (1-DOF radial-zoom + zero-padded dims 1-5 → 2.35) and
# V2 (1-DOF radial-zoom + frozen-baseline dims 1-5 → predicted [1.10, 1.30])
# by changing the OPTIMIZATION VARIABLE entirely:
#   * V1/V2 optimize the 6-DOF pose vector that's INPUT to the renderer.
#   * V3 distills a tiny MLP that PREDICTS that pose from mask features
#     at inflate time. The PoseNet 12-dim head output ("embedding") is a
#     COMPRESS-TIME-ONLY teacher signal; embedding-dropout ensures the MLP
#     learns the mask-only path that runs at inflate.
#
# Predicted band: [1.10, 1.18] [contest-CUDA].
#   * Floor 1.10: rate savings of ~0.0085 (15 KB → 1-2 KB) land cleanly +
#     MLP predicts dim 0 within 0.5 RMSE of Lane A's optimized poses.
#   * Ceiling 1.18: MLP underfits → PoseNet regresses by ~0.05 absolute,
#     which is +0.07 score from the noisier dim 0; net flat-to-slightly-worse.
#
# Strict-scorer-rule compliance: PoseNet IS loaded at COMPRESS time (Stage
# 2) but NOT at inflate. The archive ships the MLP + sentinel ONLY; the
# inflate dispatch in submissions/robust_current/inflate_renderer.py runs
# the MLP with zero embedding input (verified by Lane M-V3 test suite).
#
# Anchor:
#   - renderer: experiments/results/lane_a_landed/iter_0/renderer.bin
#   - target poses: experiments/results/lane_a_landed/optimized_poses.pt
#                   (supervision labels for MLP distillation)
#   - masks: experiments/results/lane_a_landed/extracted/masks.mkv
#   - GT video: upstream/videos/0.mkv (for PoseNet embedding extraction)
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234

LOG_DIR="$WORKSPACE/lane_m_v3_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-m-v3] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_m_v3_pose_from_embedding.sh',
    'lane_name': 'lane_m_v3_pose_from_embedding_path_a',
    'anchor_renderer': 'experiments/results/lane_a_landed/iter_0/renderer.bin',
    'anchor_target_poses': 'experiments/results/lane_a_landed/optimized_poses.pt',
    'anchor_masks': 'experiments/results/lane_a_landed/extracted/masks.mkv',
    'anchor_score_baseline': 1.15,
    'predicted_band': [1.10, 1.18],
    'delta_from_v2': 'mlp_predicts_pose_from_mask_features_no_pose_pt_in_archive',
    'optimization_variable': 'mlp_weights_compress_time_distillation_path_a',
    'inflate_strict_scorer_rule_compliant': True,
    'output_dir': '$LOG_DIR',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=M-V3 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend.
# Reference: feedback_vastai_nvdec_host_variation.
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

# Pre-flight: anchor on Lane A's 1.15 [contest-CUDA] artifacts.
ANCHOR_RENDERER="experiments/results/lane_a_landed/iter_0/renderer.bin"
ANCHOR_POSES="experiments/results/lane_a_landed/optimized_poses.pt"
ANCHOR_MASKS="experiments/results/lane_a_landed/extracted/masks.mkv"
for f in "$ANCHOR_RENDERER" \
         "$ANCHOR_POSES" \
         "$ANCHOR_MASKS" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
log "  anchor_renderer:      $ANCHOR_RENDERER"
log "  anchor_target_poses:  $ANCHOR_POSES (MLP supervision targets)"
log "  anchor_masks:         $ANCHOR_MASKS"

log "=== Stage 1: stage Lane A masks (no rebuild — anchor on Lane A's exact bytes) ==="
mkdir -p "$LOG_DIR/extracted"
cp "$ANCHOR_MASKS" "$LOG_DIR/extracted/masks.mkv"
log "  staged $(stat -c '%s' "$LOG_DIR/extracted/masks.mkv") bytes of Lane A masks"

log "=== Stage 2: distill pose-from-embedding MLP (compress-time PoseNet load) ==="
"$PYBIN" -u experiments/distill_pose_from_embedding.py \
    --renderer "$ANCHOR_RENDERER" \
    --target-poses "$ANCHOR_POSES" \
    --masks "$LOG_DIR/extracted/masks.mkv" \
    --gt-video upstream/videos/0.mkv \
    --upstream upstream \
    --output-dir "$LOG_DIR" \
    --device cuda \
    --epochs 200 \
    --batch-size 64 \
    --lr 3e-3 \
    --embedding-dropout-p 0.5 \
    --log-interval 20 2>&1 | tee "$LOG_DIR/distill_full.log" | tail -30

# Validate: distill produced both the MLP weights and the sentinel.
[ -f "$LOG_DIR/pose_from_embedding_v1.pt" ] || {
    echo "FATAL: distill did not produce pose_from_embedding_v1.pt"
    exit 2
}
[ -f "$LOG_DIR/pose_from_embedding_v1" ] || {
    echo "FATAL: distill did not produce sentinel pose_from_embedding_v1"
    exit 2
}
WEIGHTS_BYTES=$(stat -c '%s' "$LOG_DIR/pose_from_embedding_v1.pt")
log "  produced MLP weights: $WEIGHTS_BYTES bytes"
log "  produced sentinel:    0 bytes (verified)"

# V3 critical sanity check: the MLP weights file is small (< 30 KB FP16 ceiling)
# and the sentinel is exactly 0 bytes. If either invariant is violated the
# rate-savings premise of Lane M-V3 is broken.
"$PYBIN" -c "
import sys
from pathlib import Path
weights = Path('$LOG_DIR/pose_from_embedding_v1.pt')
sentinel = Path('$LOG_DIR/pose_from_embedding_v1')
wb = weights.stat().st_size
sb = sentinel.stat().st_size
if wb >= 30000:
    print(f'FATAL: MLP weights too large ({wb} bytes); Lane M-V3 rate-savings premise broken (target ~1-2 KB)', file=sys.stderr)
    sys.exit(2)
if sb != 0:
    print(f'FATAL: sentinel must be exactly 0 bytes, got {sb}', file=sys.stderr)
    sys.exit(2)
print(f'[lane-m-v3-sanity] MLP={wb} bytes, sentinel={sb} bytes OK')
"

log "=== Stage 3: build NEW archive (Lane A renderer + Lane A masks + MLP + sentinel) ==="
# Lane M-V3 archive contents:
#   * renderer.bin            (Lane A's, unchanged)
#   * masks.mkv               (Lane A's, unchanged)
#   * pose_from_embedding_v1.pt   (~1-2 KB FP16 MLP)
#   * pose_from_embedding_v1      (0-byte sentinel)
# NO optimized_poses.pt — that's the entire point of Lane M-V3.
mkdir -p "$LOG_DIR/iter_0"
cp "$ANCHOR_RENDERER" "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
cp "$LOG_DIR/pose_from_embedding_v1.pt" "$LOG_DIR/iter_0/pose_from_embedding_v1.pt"
cp "$LOG_DIR/pose_from_embedding_v1" "$LOG_DIR/iter_0/pose_from_embedding_v1"
ARCHIVE="$LOG_DIR/archive_lane_m_v3.zip"
"$PYBIN" -c "
import zipfile, os, sys
src = '$LOG_DIR/iter_0'
dst = '$ARCHIVE'
required = ('renderer.bin', 'masks.mkv', 'pose_from_embedding_v1.pt', 'pose_from_embedding_v1')
forbidden = ('optimized_poses.pt', 'optimized_poses.bin', 'poses.pt', 'poses.bin')
# Deterministic-zip: fixed date_time + Unix perms (codex R5-r6 #5).
DET_DATE = (2026, 4, 27, 0, 0, 0)
DET_ATTR = (0o644 & 0xFFFF) << 16
DET_SYS = 3
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in required:
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing {p}'
        info = zipfile.ZipInfo(filename=n, date_time=DET_DATE)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = DET_ATTR
        info.create_system = DET_SYS
        with open(p, 'rb') as f:
            z.writestr(info, f.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
# Sanity: NO forbidden member can have leaked in.
with zipfile.ZipFile(dst) as z:
    members = set(z.namelist())
for n in required:
    assert n in members, f'archive missing required {n}'
for n in forbidden:
    if n in members:
        print(f'FATAL: archive contains forbidden {n} — Lane M-V3 point is moot', file=sys.stderr)
        sys.exit(3)
print(f'archive {dst}: {os.path.getsize(dst)} bytes, members={sorted(members)}')
"

log "=== Stage 4: contest_auth_eval on Lane M-V3 archive ==="
# Inflate side detects the pose_from_embedding_v1 sentinel automatically
# (no env-gate needed — Lane M-V3 inflate dispatch is self-activating
# when the sentinel is present and no optimized_poses.pt is in the archive).
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20

log "=== LANE_M_V3_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "    archive: $ARCHIVE"
log "    distill_provenance: $LOG_DIR/distill_provenance.json"
log "    predicted_band: [1.10, 1.18] [contest-CUDA]"
