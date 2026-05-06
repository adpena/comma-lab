#!/bin/bash
# Lane RAFT-pose-init (PARADIGM-la-pose, RAFT-Large flow → pose warm-start).
#
# Module: src/tac/raft_pose.py + experiments/derive_poses_from_raft.py.
# Computes RAFT-Large optical flow on consecutive video frames, then projects
# road-region horizontal flow into pose dim 0 as a warm-start for pose TTO.
#
# DELTA from default Adam pose initialization:
#   * Pose TTO starts from RAFT-derived dim-0 (not zeros), giving the
#     optimizer a valid starting basin instead of cold-start. Predicted band
#     [contest-CUDA] [1.10, 1.18] vs Lane A 1.15.
#
# REGISTERED-BUT-NOT-WIRED in step_pose_tto as of 2026-05-06; the WARN guard
# (commit 77dc808a) prevents silent no-op. Operator must:
#   1. Add --init-poses flag to optimize_poses.py argparse
#   2. Wire the flag through optimize_poses.py's pose tensor construction
#   3. Then this runbook's Stage 2 can pass derived poses through
#
# Cost: T4 @ ~$0.50/hr × ~30min RAFT inference + 30min pose TTO + 30min auth
# eval = ~$0.75. RAFT-Large weights download (~200MB) is a one-time cost.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_raft_pose_init_results"
mkdir -p "$LOG_DIR"
TAG="lane_raft_pose_init"
log() { echo "[lane-raft-init] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
"$PYBIN" -c "
import json, time, torch
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'cuda_available': torch.cuda.is_available(),
    'lane_script': 'scripts/remote_lane_raft_pose_init.sh',
    'tag': '$TAG',
    'paradigm': 'la_pose_raft_init',
    'predicted_band': [1.10, 1.18],
    'lane_registry_id': 'lane_raft_pose_init',
    'cross_paradigm_wiring_status': 'WARN-guard wired in step_pose_tto (commit 77dc808a); --init-poses CLI extension required.',
    'pre_dispatch_blockers': ['optimize_poses.py needs --init-poses flag', 'RAFT-Large weights need download (one-time ~200MB)'],
    'cost_estimate_usd': 0.75,
}
with open('$PROVENANCE', 'w') as f: json.dump(prov, f, indent=2)
print(json.dumps(prov))
"

( while true; do sleep 60; echo "[$(date -u +%FT%TZ)] lane=raft-init" >> "$HEARTBEAT"; done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

log "=== Stage 0: NVDEC + RAFT module check ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || { log "FATAL: NVDEC failed"; exit 2; }
"$PYBIN" -c "
import sys; sys.path.insert(0, 'src')
from tac.raft_pose import compute_raft_flow, flow_to_pose_dim0
print('RAFT pose module importable')
try:
    from torchvision.models.optical_flow import raft_large
    print('torchvision raft_large OK')
except ImportError as e:
    print('WARN: torchvision raft_large missing:', e)
" 2>&1 | tee -a "$LOG_DIR/run.log"

log "=== Stage 1: RAFT-Large flow + pose-dim-0 derivation ==="
VIDEO="${VIDEO:-$WORKSPACE/upstream/videos/0.mkv}"
BASELINE_POSES="${BASELINE_POSES:-$WORKSPACE/experiments/results/lane_a_baseline/optimized_poses.pt}"
RAFT_OUT="$LOG_DIR/raft_derived_poses.pt"
if [ -f "$VIDEO" ] && [ -f "$BASELINE_POSES" ]; then
    "$PYBIN" -u experiments/derive_poses_from_raft.py \
        --video "$VIDEO" \
        --baseline-poses "$BASELINE_POSES" \
        --output "$RAFT_OUT" \
        --device cuda \
        2>&1 | tee -a "$LOG_DIR/run.log" || { log "FATAL: RAFT derivation failed"; exit 4; }
else
    log "WARN: VIDEO=$VIDEO or BASELINE_POSES=$BASELINE_POSES missing"
    log "STAGE 1 SKIPPED."
fi

log "=== Stage 2: pose TTO with --init-poses (BLOCKER) ==="
log "WARN: optimize_poses.py needs --init-poses flag wired before this stage."
log "WARN: Reference: optimize_poses.py:466-482 (--lora-rank pattern)."
log "STAGE 2 SKIPPED."

log "=== Stage 3: contest-CUDA auth eval (skipped) ==="
log "LANE_RAFT_POSE_INIT_DONE score=N/A [contest-CUDA] paradigm=la-pose-raft-init blocked=optimize_poses_init_poses_flag"
