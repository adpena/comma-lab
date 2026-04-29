#!/bin/bash
# Lane LM-V2 — endpoint-tracking zero-archive-cost poses. Anchored on
# Lane A's 1.15 [contest-CUDA] floor.
#
# V1 oversight: Lane LM-V1 (centroid math) measured a 0.017 Pearson
# correlation vs Lane A's optimised pose dim 0 — calibration FAILED
# because lateral car drift in the lane dominated the centroid signal.
#
# V2 fix (memory project_lane_marking_speed_estimation +
# project_posenet_rank1_discovery): track lane-mark dash ENDPOINTS (top +
# bottom of the visible dashes) instead of the centroid. The bottom
# endpoint sweeps PURE radially from the FoE → image-edge as the car
# moves forward (perspective compression is a geometric necessity), so
# its radial displacement isolates forward motion from lateral drift.
#
# Plus: V2 supports per-clip RECALIBRATION via least-squares fit to the
# baseline_poses optimised dim 0 — this collapses the V1 calibration gap
# entirely (the affine map a + b * log_zoom = pose_dim0 is fit on this
# very clip's baseline, not a frozen offline constant).
#
# Single variable vs Lane LM-V1: --method endpoint (replaces the implicit
# default centroid). Everything else identical to remote_lane_lm_zero_cost_poses.sh.
#
# Predicted band: [1.08, 1.18] [contest-CUDA].
#   * Floor 1.08: V2 lands in the V1 band (only the rate saving counts).
#   * Ceiling 1.18: V2 correlation lands at 0.30+ → PoseNet does not
#     regress + rate saving lands cleanly + per-clip recalibration nudges
#     pose_dim0 toward the optimised distribution.
#
# Strict-scorer-rule compliance: identical to V1 — pure geometric mask
# arithmetic at inflate, NO scorers loaded.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234

LOG_DIR="$WORKSPACE/lane_lm_v2_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-lm-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_lm_v2_endpoint_tracking.sh',
    'lane_name': 'lane_lm_v2_endpoint_tracking',
    'anchor_archive': 'experiments/results/lane_a_landed/archive_lane_a.zip',
    'anchor_score_baseline': 1.15,
    'predicted_band': [1.08, 1.18],
    'delta_from_lane_a': 'omit_optimized_poses_pt_compute_at_inflate_via_endpoint_tracking',
    'delta_from_lane_lm_v1': 'method=endpoint (vs V1 centroid; correlation 0.017 → 0.30+)',
    'inflate_env_required': 'INFLATE_ZERO_COST_POSES=1',
    'strict_scorer_rule_compliant': True,
    'method': 'endpoint',
    'min_correlation_target': 0.30,
    'output_dir': '$LOG_DIR',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=LM-V2 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend.
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

# Pre-flight: dead-flag-wiring guard. Every CLI flag we pass to
# experiments/build_zero_cost_pose_archive.py MUST exist in its argparse.
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
sys.path.insert(0, 'src')
script = open('scripts/remote_lane_lm_v2_endpoint_tracking.sh').read()
build_src = open('experiments/build_zero_cost_pose_archive.py').read()
real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', build_src))
m = re.search(
    r'experiments/build_zero_cost_pose_archive\.py(.*?)(?=\n\s*\[ -f|\Z)',
    script, re.DOTALL,
)
assert m, 'could not locate build_zero_cost_pose_archive.py invocation in script'
used = set(re.findall(r'\B--([a-z][a-z0-9-]+)', m.group(0)))
invented = used - real
if invented:
    print(f'INVENTED FLAGS: {sorted(invented)} not in build_zero_cost_pose_archive argparse',
          file=sys.stderr); sys.exit(3)
print(f'OK: {len(used)} flags all real')
"

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

log "=== Stage 2: compute V2 endpoint-tracked poses + verify correlation ==="
# V2 raises the gate to 0.30 (V1 was effectively unable to clear that bar
# at 0.017). The build tool ALWAYS writes the .provenance.json calibration
# report so we can read corr/rmse out-of-band.
ARCHIVE="$LOG_DIR/archive_lane_lm_v2.zip"
"$PYBIN" -u experiments/build_zero_cost_pose_archive.py \
    --lane-a-archive "$ANCHOR_ARCHIVE" \
    --output "$ARCHIVE" \
    --method endpoint \
    --min-correlation 0.30 \
    2>&1 | tee "$LOG_DIR/build.log" | tail -30

# Validate: did the builder produce the file?
[ -f "$ARCHIVE" ] || { echo "FATAL: build_zero_cost_pose_archive didn't produce $ARCHIVE"; exit 2; }
[ -f "$ARCHIVE.provenance.json" ] || { echo "FATAL: build_zero_cost_pose_archive didn't produce $ARCHIVE.provenance.json"; exit 2; }
log "  produced $ARCHIVE ($(stat -c '%s' "$ARCHIVE") bytes)"

# Critical sanity: the output archive MUST contain zero_cost_poses_v1
# sentinel and MUST NOT contain optimized_poses.pt. Same invariants as V1
# — V2 only changed the COMPUTE method, not the on-disk archive shape.
"$PYBIN" -c "
import zipfile, sys
with zipfile.ZipFile('$ARCHIVE') as z:
    members = set(z.namelist())
if 'zero_cost_poses_v1' not in members:
    print('FATAL: archive missing zero_cost_poses_v1 sentinel', file=sys.stderr); sys.exit(3)
if 'optimized_poses.pt' in members or 'poses.pt' in members:
    print('FATAL: archive STILL contains a pose .pt — Lane LM-V2 point is moot', file=sys.stderr); sys.exit(3)
print(f'[lane-lm-v2-sanity] archive members OK: {sorted(members)}')
"

# Surface the calibration report so the operator can see the V2 correlation
# at a glance (the headline V2 win vs V1's 0.017).
"$PYBIN" -c "
import json
prov = json.load(open('$ARCHIVE.provenance.json'))
cal = prov.get('calibration_report', {})
print(f'[lane-lm-v2-calibration] method={prov.get(\"method\")} '
      f'correlation={cal.get(\"correlation_dim0\", float(\"nan\")):.4f} '
      f'rmse={cal.get(\"rmse_dim0\", float(\"nan\")):.4f}')
"

log "=== Stage 3: build NEW archive (verified above; nothing additional to do) ==="
log "  archive_lane_lm_v2.zip is the contest submission candidate"

log "=== Stage 4: contest_auth_eval on Lane LM-V2 archive ==="
# Inflate side requires INFLATE_ZERO_COST_POSES=1 to compute poses from
# the lane-mark sentinel. Without it the archive falls through to
# unconditioned rendering (catastrophic).
rm -rf "$LOG_DIR/eval_work"
INFLATE_ZERO_COST_POSES=1 "$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20

# RESULT_JSON guard.
if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval.log missing RESULT_JSON — silent eval crash. Investigate before reporting score."
    exit 4
fi

log "=== LANE_LM_V2_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "    archive: $ARCHIVE"
log "    calibration: $ARCHIVE.provenance.json"
log "    predicted_band: [1.08, 1.18] (V1 was [1.05, 1.15])"
