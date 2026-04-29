#!/bin/bash
# Lane STC CLEAN-SOURCE — boundary-mask coding on raw SegNet argmax (NO AV1).
#
# Replace Lane A's masks.mkv (~411KB AV1 monochrome) with masks.stcb produced
# from CLEAN SegNet argmax of upstream/videos/0.mkv. Skips the AV1 encode step
# entirely, so STC's lossless coder is not forced to re-encode AV1 quantization
# noise (which caused the regression documented in
# project_lane_stc_av1_regression_finding_20260429).
#
# Pipeline (5 stages):
#   Stage 0: NVDEC probe + STCB inflate code parity check.
#   Stage 1: anchor checks (Lane A renderer + poses, GT video, SegNet weights).
#   Stage 2: build_clean_source_stc_archive.py — runs SegNet at compress time
#            on the GT video, emits clean argmax, encodes to masks.stcb,
#            packages with renderer.bin + optimized_poses.pt from Lane A.
#   Stage 3: archive size guard (refuse to spend GPU on auth eval if STC is
#            larger than the AV1 anchor — saves $0.20/false-positive).
#   Stage 4: contest_auth_eval.py [contest-CUDA] OR [Modal-T4-CPU].
#
# Anchor: experiments/results/lane_a_landed/archive_lane_a.zip (1.15 contest-CUDA).
# Predicted band [contest-CUDA]: 1.10 - 1.13 if clean-source delivers 60-80KB
#   savings (rate -0.04 .. -0.05) AND seg/pose stay flat (lossless mask path).
# Cost cap: $0.50 (no training; SegNet forward + encode + auth eval only).
# controlled_baseline: Lane A (1.15 contest-CUDA).
#
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.

set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_stc_clean_source"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-stc-clean] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md canonical pipeline standard).
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
    'lane_script': 'scripts/remote_lane_stc_clean_source.sh',
    'output_dir': '$LOG_DIR',
    'predicted_band': [1.08, 1.13],
    'anchor_archive': 'experiments/results/lane_a_landed/archive_lane_a.zip',
    'anchor_score_baseline': 1.15,
    'controlled_baseline': 'lane_a_landed (1.15 contest-CUDA)',
    'paradigm': 'clean_source_boundary_arithmetic_coded_class_ids',
    'mask_source': 'SegNet on upstream/videos/0.mkv at compress time (NO AV1 roundtrip)',
    'inflate_path': 'PYTHON_INFLATE=renderer; masks.stcb dispatched via _resolve_mask_path',
    'hypothesis': 'clean argmax (no AV1 quantization speckle) restores STC 60-80KB savings projection',
    'risk_check_2026_04_29': 'AV1-anchor STC regressed 50x; clean source removes that cause',
    'cost_estimate_usd': 0.20,
    'cost_cap_usd': 0.50,
    'wall_clock_estimate_hours': 0.5,
    'wall_clock_cap_hours': 1.5,
    'lane_tag': '[contest-CUDA]',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ' || echo "no-gpu")
    echo "[$(date -u +%FT%TZ)] lane=STC-clean gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# ---------------------------------------------------------------------------
# Stage 0: NVDEC probe + STCB inflate code parity check.
# ---------------------------------------------------------------------------
log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed -- refusing to spend GPU on a host that"
    log "       cannot run upstream/evaluate.py at the end. Destroy this"
    log "       Vast.ai instance and pick a different host."
    exit 2
}

# Code parity: the inflate path must know how to decode masks.stcb. If the
# tarball did not include the post-2026-04-29 inflate_renderer.py changes,
# the auth eval at Stage 4 will FileNotFoundError. Catch it now (cheap)
# rather than later (expensive).
log "=== Stage 0b: STCB inflate code parity check ==="
"$PYBIN" -c "
import sys, importlib.util
sys.path.insert(0, 'submissions/robust_current')
sys.path.insert(0, 'src')
spec = importlib.util.find_spec('inflate_renderer')
src = open(spec.origin).read()
ok = ('_load_masks_from_stcb' in src
      and 'masks.stcb' in src
      and 'STCB' in src)
if not ok:
    print('FATAL: inflate_renderer.py is stale; missing STCB dispatch.', file=sys.stderr)
    print('       Re-tarball local HEAD (Check 66 tarball-anchor parity).', file=sys.stderr)
    sys.exit(2)
import importlib
mod = importlib.import_module('inflate_renderer')
assert hasattr(mod, '_load_masks_from_stcb'), '_load_masks_from_stcb missing'
print('inflate_renderer STCB parity: OK')
" || exit 2

# ---------------------------------------------------------------------------
# Stage 1: anchor + dependency checks.
# ---------------------------------------------------------------------------
ANCHOR_ARCHIVE="experiments/results/lane_a_landed/archive_lane_a.zip"
GT_VIDEO="upstream/videos/0.mkv"
SEGNET_WEIGHTS="upstream/models/segnet.safetensors"
POSENET_WEIGHTS="upstream/models/posenet.safetensors"

log "=== Stage 1: anchor + dependency checks ==="
for f in "$ANCHOR_ARCHIVE" \
         "$GT_VIDEO" \
         "$SEGNET_WEIGHTS" \
         "$POSENET_WEIGHTS" \
         submissions/robust_current/inflate.sh \
         submissions/robust_current/inflate_renderer.py \
         experiments/build_clean_source_stc_archive.py \
         experiments/contest_auth_eval.py \
         src/tac/stc_boundary_codec.py; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

# Pre-flight: argparse arity scan for the build script.
# Manually verified flags against build_clean_source_stc_archive.py argparse:
#   --anchor-archive, --gt-video, --output, --device, --boundary-fraction,
#   --batch-size, --upstream-dir, --manifest. CLAUDE.md NEVER-invent-CLI-flags rule.
log "=== Stage 1b: build script argparse keys verified (manual) ==="

# ---------------------------------------------------------------------------
# Stage 2: build clean-source STC archive.
# ---------------------------------------------------------------------------
log "=== Stage 2: build clean-source STC archive (SegNet -> argmax -> STCB) ==="
ARCHIVE="$LOG_DIR/archive_lane_stc_clean.zip"
MANIFEST="$LOG_DIR/manifest.json"
"$PYBIN" -u experiments/build_clean_source_stc_archive.py \
    --anchor-archive "$ANCHOR_ARCHIVE" \
    --gt-video "$GT_VIDEO" \
    --output "$ARCHIVE" \
    --device "${BUILD_DEVICE:-cuda}" \
    --boundary-fraction 0.05 \
    --batch-size 8 \
    --manifest "$MANIFEST" 2>&1 | tee "$LOG_DIR/build.log"
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: build_clean_source_stc_archive exited rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
if [ -z "${ARCHIVE_BYTES:-}" ] || [ "$ARCHIVE_BYTES" -le 0 ]; then
    echo "FATAL: archive size empty or zero -- refusing to call auth_eval"
    exit 2
fi
log "  clean-source archive bytes: $ARCHIVE_BYTES"

# ---------------------------------------------------------------------------
# Stage 3: archive size guard.
# ---------------------------------------------------------------------------
ANCHOR_BYTES=$(stat -c '%s' "$ANCHOR_ARCHIVE" 2>/dev/null || stat -f '%z' "$ANCHOR_ARCHIVE")
if [ "$ARCHIVE_BYTES" -gt "$ANCHOR_BYTES" ]; then
    log "WARN: clean-source STC archive ($ARCHIVE_BYTES B) is LARGER than anchor ($ANCHOR_BYTES B)."
    log "      The clean-source hypothesis ALSO failed to deliver byte savings."
    log "      Skipping auth_eval -- would only waste GPU."
    log "      LANE_STC_CLEAN_NEGATIVE [empirical]"
    log "      Manifest at $MANIFEST has byte stats for the council to review."
    exit 0
fi
log "  clean-source STC archive is SMALLER than anchor by $((ANCHOR_BYTES - ARCHIVE_BYTES)) B"

# ---------------------------------------------------------------------------
# Stage 4: contest_auth_eval [contest-CUDA] (or [Modal-T4-CPU] if explicit).
# ---------------------------------------------------------------------------
log "=== Stage 4: contest_auth_eval on clean-source Lane STC archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -30
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: contest_auth_eval exited rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

log "=== LANE_STC_CLEAN_DONE [contest-CUDA] -- see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
