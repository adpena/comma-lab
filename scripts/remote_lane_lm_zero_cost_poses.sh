#!/bin/bash
# Lane LM-A — zero-archive-cost poses computed at inflate from lane-mark
# mask displacement. Anchored on Lane A (1.15 [contest-CUDA]).
#
# Strategic premise (memory project_lane_marking_speed_estimation +
# project_posenet_rank1_discovery): the video is forward-driving footage
# with MUTCD lane marks (3m × 15cm dashes). PoseNet's effective Jacobian
# rank is 1.008 — only dim 0 (scalar radial zoom from FoE) carries signal.
# Lane marks appear in EVERY frame; their inter-frame radial displacement
# encodes vehicle speed (= dim 0). Therefore dim 0 is COMPUTABLE at inflate
# from masks alone, eliminating the 15.3 KB optimized_poses.pt artifact.
#
# Lane LM-A delta vs Lane A:
#   archive: same renderer.bin + same masks.mkv, BUT optimized_poses.pt is
#   omitted and a 0-byte zero_cost_poses_v1 sentinel is written instead.
#   Inflate side reads INFLATE_ZERO_COST_POSES=1 and computes per-pair
#   6-DOF poses via tac.lane_mark_pose.compute_zero_cost_poses_from_masks
#   (pure geometric centroid math; NO scorers loaded → strict-scorer-rule
#   compliant).
#
# Predicted score band: [1.05, 1.15] [contest-CUDA].
#   * Floor 1.05: rate savings (~0.010) land cleanly + lane-mark dim 0
#     matches Lane A's optimized dim 0 closely enough that PoseNet does
#     not regress.
#   * Ceiling 1.15: analytical estimate is too noisy → PoseNet regresses
#     by exactly the rate savings → net flat vs Lane A.
#
# Strict-scorer-rule compliance: NO scorers loaded at inflate. The pose
# computation is pure geometric centroid arithmetic on the mask tensor.
# See src/tac/lane_mark_pose.py for the canonical math.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234

LOG_DIR="$WORKSPACE/lane_lm_a_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-lm-a] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_lm_zero_cost_poses.sh',
    'lane_name': 'lane_lm_a_zero_cost_poses',
    'anchor_archive': 'experiments/results/lane_a_landed/archive_lane_a.zip',
    'anchor_score_baseline': 1.15,
    'predicted_band': [1.05, 1.15],
    'delta_from_lane_a': 'omit_optimized_poses_pt_compute_at_inflate_from_lane_marks',
    'inflate_env_required': 'INFLATE_ZERO_COST_POSES=1',
    'strict_scorer_rule_compliant': True,
    'output_dir': '$LOG_DIR',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=LM-A gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0 (2026-04-27): NVDEC probe BEFORE any GPU spend. Catches bad-host
# case in 5 seconds. Reference: feedback_vastai_nvdec_host_variation.
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

# Pre-flight: anchor on Lane A's verified 1.15 [contest-CUDA] archive.
ANCHOR_ARCHIVE="experiments/results/lane_a_landed/archive_lane_a.zip"
for f in "$ANCHOR_ARCHIVE" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
log "  anchor_archive: $ANCHOR_ARCHIVE ($(stat -c '%s' "$ANCHOR_ARCHIVE") bytes)"

log "=== Stage 1: extract masks from Lane A archive (sanity check + cache) ==="
mkdir -p "$LOG_DIR/extracted"
"$PYBIN" -c "
import zipfile, sys
src = '$ANCHOR_ARCHIVE'
dst = '$LOG_DIR/extracted'
required = {'renderer.bin', 'masks.mkv', 'optimized_poses.pt'}
with zipfile.ZipFile(src) as z:
    members = set(z.namelist())
    missing = required - members
    if missing:
        print(f'FATAL: --lane-a-archive missing: {sorted(missing)}', file=sys.stderr)
        sys.exit(2)
    for n in required:
        z.extract(n, dst)
    print(f'extracted {sorted(required)} from {src}')
"
log "  staged renderer.bin: $(stat -c '%s' "$LOG_DIR/extracted/renderer.bin") bytes"
log "  staged masks.mkv:    $(stat -c '%s' "$LOG_DIR/extracted/masks.mkv") bytes"
log "  staged optimized_poses.pt: $(stat -c '%s' "$LOG_DIR/extracted/optimized_poses.pt") bytes (sanity-check only; will NOT be in output archive)"

log "=== Stage 2: compute zero-cost poses + verify correlation vs Lane A baseline ==="
# Sanity-check: lane-mark-derived dim 0 must correlate with Lane A's
# optimized dim 0 (--min-correlation 0.30). Below the gate the build
# tool aborts to save Vast.ai eval $$. The build tool ALWAYS writes the
# .provenance.json calibration report so we can read corr/rmse out-of-band.
ARCHIVE="$LOG_DIR/archive_lane_lm_a.zip"
"$PYBIN" -u experiments/build_zero_cost_pose_archive.py \
    --lane-a-archive "$ANCHOR_ARCHIVE" \
    --output "$ARCHIVE" \
    --min-correlation 0.30 \
    2>&1 | tee "$LOG_DIR/build.log" | tail -30
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPESTATUS[0]}" >&2; exit "${PIPESTATUS[0]}"
    fi

# Validate: did the builder produce the file?
[ -f "$ARCHIVE" ] || { echo "FATAL: build_zero_cost_pose_archive didn't produce $ARCHIVE"; exit 2; }
[ -f "$ARCHIVE.provenance.json" ] || { echo "FATAL: build_zero_cost_pose_archive didn't produce $ARCHIVE.provenance.json"; exit 2; }
log "  produced $ARCHIVE ($(stat -c '%s' "$ARCHIVE") bytes)"

# Critical sanity: the output archive MUST contain zero_cost_poses_v1
# sentinel and MUST NOT contain optimized_poses.pt. If those invariants
# are violated, the strict-scorer-rule compliance claim is broken.
"$PYBIN" -c "
import zipfile, sys
with zipfile.ZipFile('$ARCHIVE') as z:
    members = set(z.namelist())
if 'zero_cost_poses_v1' not in members:
    print('FATAL: archive missing zero_cost_poses_v1 sentinel', file=sys.stderr); sys.exit(3)
if 'optimized_poses.pt' in members or 'poses.pt' in members:
    print('FATAL: archive STILL contains a pose .pt — Lane LM-A point is moot', file=sys.stderr); sys.exit(3)
print(f'[lane-lm-a-sanity] archive members OK: {sorted(members)}')
"

log "=== Stage 3: build NEW archive (verified above; nothing additional to do) ==="
log "  archive_lane_lm_a.zip is the contest submission candidate"

log "=== Stage 4: contest_auth_eval on Lane LM-A archive ==="
# Inflate side requires INFLATE_ZERO_COST_POSES=1 to compute poses from
# the lane-mark sentinel. Without it, the archive falls through to
# unconditioned rendering (catastrophic). The env-gate is INTENTIONAL
# (prevents silent activation on stale archives) so we MUST set it here.
rm -rf "$LOG_DIR/eval_work"
INFLATE_ZERO_COST_POSES=1 "$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPESTATUS[0]}" >&2; exit "${PIPESTATUS[0]}"
    fi

log "=== LANE_LM_A_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "    archive: $ARCHIVE"
log "    calibration: $ARCHIVE.provenance.json"
log "    predicted_band: [1.05, 1.15] [contest-CUDA]"
