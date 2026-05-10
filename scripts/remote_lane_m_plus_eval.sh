#!/bin/bash
# Lane M+: Zero-cost poses computed at inflate from lane-mark mask
# displacement (per src/tac/lane_mark_pose.py + commit c7f0b690).
#
# Pure eval — no pose TTO. Build new archive WITHOUT optimized_poses.pt
# but WITH a zero_cost_poses_v1 sentinel, then run contest_auth_eval.py
# with INFLATE_ZERO_COST_POSES=1 so the inflate side reconstructs the
# 6-DOF poses analytically from lane marks.
#
# Predicted impact:
#   Rate: -0.005 (no pose bytes; ~7-15KB removed)
#   Distortion: UNTESTED (geometric estimate vs Lane A's TTO-converged
#   poses). Could land 1.145 (good) to 1.65+ (bad) — PoseNet very
#   sensitive to pose quality.
#
# Cheap fast test (~$0.20, 15-20 min): just an eval, no pose TTO needed.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_m_plus_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-m+] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'predicted_band': [1.10, 1.30],  # added 2026-04-27 per preflight check 31
    'lane_script': 'scripts/remote_lane_m_plus_eval.sh',
    'output_dir': '$LOG_DIR',
    'experiment': 'lane_m_plus_zero_cost_poses_eval',
    'inflate_zero_cost_poses_env': '1',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=M+ gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend (per
# feedback_vastai_nvdec_host_variation memory entry).
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed — refusing to spend GPU on a host that"
    log "       cannot run upstream/evaluate.py at the end."
    exit 2
}

# Pre-flight: required artifacts present
for f in submissions/baseline_dilated_h64_0_90/renderer.bin \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors \
         src/tac/lane_mark_pose.py; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

# Note: optimized_poses.pt is NOT required for Lane M+ build (we omit it
# from the archive entirely) but the renderer requires the same renderer
# checkpoint as Lane A baseline.

log "=== Stage 1: build archive WITH zero-cost-poses sentinel (NO poses.pt) ==="
log "   --use-zero-cost-poses → omits optimized_poses.pt, writes sentinel"
set +e
"$PYBIN" experiments/build_baseline_archive.py \
    --device cuda --crf 50 \
    --use-zero-cost-poses \
    --output "$LOG_DIR/archive_lane_m_plus.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -10
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

ARCHIVE="$LOG_DIR/archive_lane_m_plus.zip"
[ -f "$ARCHIVE" ] || { echo "FATAL: build did not produce $ARCHIVE" >&2; exit 2; }
ARCHIVE_SIZE=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
log "  archive built: $ARCHIVE ($ARCHIVE_SIZE bytes)"

# Verify the sentinel is in the archive AND optimized_poses.pt is NOT.
"$PYBIN" -c "
import zipfile, sys
z = zipfile.ZipFile('$ARCHIVE')
names = z.namelist()
print('archive members:', names)
has_sentinel = any('zero_cost_poses_v1' in n for n in names)
has_poses_pt = any(n.endswith('optimized_poses.pt') for n in names)
print(f'has zero_cost_poses_v1 sentinel: {has_sentinel}')
print(f'has optimized_poses.pt: {has_poses_pt}')
if not has_sentinel:
    print('FATAL: archive missing zero_cost_poses_v1 sentinel', file=sys.stderr)
    sys.exit(2)
if has_poses_pt:
    print('FATAL: archive still has optimized_poses.pt (defeats Lane M+)', file=sys.stderr)
    sys.exit(2)
print('OK: archive is Lane M+ shape')
"

log "=== Stage 4: contest_auth_eval with INFLATE_ZERO_COST_POSES=1 ==="
log "   inflate side will detect sentinel + call "
log "   tac.lane_mark_pose.compute_zero_cost_poses_from_masks()"
log "   look for: '[zero-cost-poses] computed N poses from lane marks'"
rm -rf "$LOG_DIR/eval_work"
# CRITICAL: env var must be exported BEFORE invoking contest_auth_eval.py
# so it inherits into the inflate.sh subprocess (contest_auth_eval.py
# does not pass an explicit env= dict to the inflate subprocess.run).
export INFLATE_ZERO_COST_POSES=1
log "   INFLATE_ZERO_COST_POSES=$INFLATE_ZERO_COST_POSES"
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

log "=== LANE_M_PLUS_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "  archive: $ARCHIVE ($ARCHIVE_SIZE bytes)"
log "  zero-cost-poses banner in eval log? grep '[zero-cost-poses]'"
grep -h "zero-cost-poses" "$LOG_DIR/auth_eval.log" 2>/dev/null | head -5 || true
log "=== LANE_M_DONE [contest-CUDA] ==="
