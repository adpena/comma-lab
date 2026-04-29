#!/bin/bash
# Lane PD — Pose Deltas (Schmidhuber eureka, 2026-04-29).
#
# Strategy: encoder-only. Take a SegMap-paradigm OR renderer-paradigm archive,
# extract `optimized_poses.pt`, re-encode it via tac.pose_delta_codec
# (anchor + int8 deltas + per-channel scale, "pose_delta_v1" sentinel),
# rebuild the archive, run contest_auth_eval. The inflate side transparently
# decodes the new format via the existing tac.submission_archive.load_optimized_poses
# (which now branches on the pose_delta_v1 sentinel). NO inflate changes
# needed — purely additive.
#
# Predicted band: -0.04 pose rate vs upstream archive [contest-CUDA].
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
#
# E2E_SMOKE_OPT_OUT: lane PD wires through tac.pose_delta_codec.py + the new
#   load_optimized_poses sentinel branch; smoke fixture will be backfilled
#   after the first contest-CUDA score lands.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_pd_pose_deltas"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
log() { echo "[lane-pd] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# UPSTREAM_ARCHIVE: any archive containing optimized_poses.pt. The lane is
# paradigm-agnostic — works on Lane A renderer archives, Lane SA SegMap
# archives, Lane MM renderer-grayscale archives, etc.
UPSTREAM_ARCHIVE="${UPSTREAM_ARCHIVE:-$WORKSPACE/experiments/results/lane_a_landed/archive_lane_a.zip}"
# UPSTREAM_INFLATE: PYTHON_INFLATE arm the upstream archive expects
# (renderer / segmap / renderer_grayscale / postfilter / segmap_film_canvas /
# segmap_arithmetic).
UPSTREAM_INFLATE="${UPSTREAM_INFLATE:-renderer}"

PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo "no-driver")
"$PYBIN" -c "
import json, time, torch
prov = {
    'lane_id': '$LANE_ID',
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'lane_script': 'scripts/remote_lane_pd_pose_deltas.sh',
    'output_dir': '$LOG_DIR',
    'variant': 'pose_delta_v1_int8_per_channel_scale',
    'predicted_band_delta': [-0.05, -0.02],
    'paradigm': 'pose_delta_codec',
    'upstream_archive': '$UPSTREAM_ARCHIVE',
    'upstream_inflate': '$UPSTREAM_INFLATE',
    'controlled_baseline': 'UPSTREAM_ARCHIVE (encoder-only repack; only delta is fp16 absolute -> int8 deltas + per-channel scale for optimized_poses.pt)',
    'cost_estimate_usd': 2.00,
    'wall_clock_estimate_hours': 4.0,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=PD gpu=$GPU" >> "$HEARTBEAT"
    sleep 300
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed -- destroy this Vast.ai instance."
    exit 2
}

[ -f "$UPSTREAM_ARCHIVE" ] || {
    echo "FATAL: missing upstream archive: $UPSTREAM_ARCHIVE" >&2
    exit 1
}

log "=== Stage 1: extract upstream archive ==="
EXTRACT_DIR="$LOG_DIR/upstream_extracted"
mkdir -p "$EXTRACT_DIR"
"$PYBIN" -c "
import zipfile, os
with zipfile.ZipFile('$UPSTREAM_ARCHIVE', 'r') as z:
    z.extractall('$EXTRACT_DIR')
print('extracted contents:', sorted(os.listdir('$EXTRACT_DIR')))
"
UPSTREAM_POSES="$EXTRACT_DIR/optimized_poses.pt"
[ -f "$UPSTREAM_POSES" ] || { echo "FATAL: missing $UPSTREAM_POSES" >&2; exit 1; }

log "=== Stage 2: encode poses via pose_delta_codec ==="
NEW_POSES="$LOG_DIR/optimized_poses_delta.pt"
"$PYBIN" -c "
import json
from tac.pose_delta_codec import encode_pose_file
stats = encode_pose_file('$UPSTREAM_POSES', '$NEW_POSES', pose_dim=6)
print('pose-delta encode stats:', json.dumps(stats, indent=2))
if stats['savings_bytes'] <= 0:
    raise SystemExit('pose-delta encoding produced no savings; aborting')
"

log "=== Stage 3: rebuild archive ==="
NEW_ARCHIVE_SRC="$LOG_DIR/archive_src"
mkdir -p "$NEW_ARCHIVE_SRC"
"$PYBIN" -c "
import os, shutil, zipfile
src_zip = '$UPSTREAM_ARCHIVE'
dst_dir = '$NEW_ARCHIVE_SRC'
with zipfile.ZipFile(src_zip, 'r') as z:
    for name in z.namelist():
        if os.path.basename(name) == 'optimized_poses.pt':
            continue
        z.extract(name, dst_dir)
shutil.copy('$NEW_POSES', os.path.join(dst_dir, 'optimized_poses.pt'))
print('archive_src contents:', sorted(os.listdir(dst_dir)))
"
ARCHIVE="$LOG_DIR/archive_${LANE_ID}.zip"
"$PYBIN" -c "
import os, zipfile
src = '$NEW_ARCHIVE_SRC'
items = sorted(os.listdir(src))
with zipfile.ZipFile('$ARCHIVE', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in items:
        p = os.path.join(src, n)
        z.write(p, arcname=n)
print('archive bytes:', os.path.getsize('$ARCHIVE'))
"

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
[ "$ARCHIVE_BYTES" -gt 0 ] || { echo 'FATAL: archive empty'; exit 2; }
UPSTREAM_BYTES=$(stat -c '%s' "$UPSTREAM_ARCHIVE" 2>/dev/null || stat -f '%z' "$UPSTREAM_ARCHIVE")
log "archive_bytes=$ARCHIVE_BYTES (upstream was $UPSTREAM_BYTES)"

INFLATE_CONFIG="$LOG_DIR/lane_pd_config.env"
echo "PYTHON_INFLATE=$UPSTREAM_INFLATE" > "$INFLATE_CONFIG"

log "=== Stage 4: contest_auth_eval [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
CONFIG_ENV_PATH="$INFLATE_CONFIG" "$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: contest_auth_eval exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

"$PYBIN" -c "
import json, os, time
prov = json.load(open('$PROVENANCE'))
prov['completed_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
prov['archive_path'] = '$ARCHIVE'
prov['archive_bytes'] = os.path.getsize('$ARCHIVE')
prov['upstream_archive_bytes'] = os.path.getsize('$UPSTREAM_ARCHIVE')
prov['lane_status'] = 'COMPLETE'
json.dump(prov, open('$PROVENANCE', 'w'), indent=2)
print('provenance updated.')
"

log "=== LANE_PD_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON [contest-CUDA] ==="
