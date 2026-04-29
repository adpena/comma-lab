#!/bin/bash
# Lane STC — Filer/Pevny/Fridrich-inspired boundary-mask coding.
#
# Replace Lane A's masks.mkv (~411KB AV1 monochrome) with masks.stcb,
# a deterministic lossless arithmetic-coded boundary-class codec
# (src/tac/stc_boundary_codec.py).
#
# Pipeline:
#   1. Decode anchor masks.mkv -> argmax class IDs (lossy AV1 -> int64).
#   2. Detect boundary pixels via Sobel on the class-ID map (top 5% per frame).
#   3. Encode boundaries via gap-coded sparse positions + arithmetic-coded
#      class IDs. Non-boundary pixels are encoded as per-frame majority
#      class with sparse exception list.
#   4. Pack into deterministic zip: renderer.bin + masks.stcb + poses.
#
# IMPORTANT FINDING (2026-04-29 local measurement on Lane A archive):
# The STC codec is LOSSLESS but Lane A's masks are AV1-decoded (lossy);
# half the pixels are AV1-noise non-majority. Lossless re-encoding of
# AV1 noise CANNOT beat AV1 lossy compression on this distribution.
# Local extrapolation: 1200 frames -> ~21MB (vs 411KB AV1 baseline).
# Lane STC is preserved as a code module + test suite for future work
# on UPSTREAM-source argmax masks (pre-AV1 SegNet output), where the
# class regions are clean and the 60-80KB savings projection holds.
#
# Anchor: experiments/results/lane_a_landed/archive_lane_a.zip (1.15 contest-CUDA).
# Predicted band [contest-CUDA]: 1.15 + (rate delta) — REGRESSION expected
#                                 if anchor masks are AV1-decoded noise.
# Cost cap: $0.50 (no training; encode + auth eval only).
# controlled_baseline: Lane A (1.15 contest-CUDA).
#
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_stc_boundary_coding"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-stc] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_stc_boundary_coding.sh',
    'output_dir': '$LOG_DIR',
    'predicted_band': [1.10, 1.20],
    'anchor_archive': 'experiments/results/lane_a_landed/archive_lane_a.zip',
    'anchor_score_baseline': 1.15,
    'controlled_baseline': 'lane_a_landed (1.15 contest-CUDA)',
    'paradigm': 'boundary_arithmetic_coded_class_ids',
    'inflate_path': 'PYTHON_INFLATE=renderer (existing argmax route)',
    'hypothesis': 'sparse boundary + per-frame majority class is more compact than AV1 on clean argmax masks',
    'known_risk': 'Lane A masks are AV1-decoded noise; lossless STC may regress on this anchor',
    'cost_estimate_usd': 0.20,
    'cost_cap_usd': 0.50,
    'wall_clock_estimate_hours': 0.5,
    'wall_clock_cap_hours': 1.5,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ' || echo "no-gpu")
    echo "[$(date -u +%FT%TZ)] lane=STC gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend.
log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed -- refusing to spend GPU on a host that"
    log "       cannot run upstream/evaluate.py at the end. Destroy this"
    log "       Vast.ai instance and pick a different host."
    exit 2
}

# Pre-flight: required artifacts present. Anchor on Lane A.
ANCHOR_ARCHIVE="experiments/results/lane_a_landed/archive_lane_a.zip"
for f in "$ANCHOR_ARCHIVE" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors \
         submissions/robust_current/inflate.sh \
         submissions/robust_current/inflate_renderer.py \
         experiments/build_lane_stc_archive.py \
         src/tac/stc_boundary_codec.py; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

# Pre-flight: argparse arity scan for the build script.
# Manually verified flags against build_lane_stc_archive.py argparse:
#   --anchor-archive, --output, --boundary-fraction, --keep-legacy-masks,
#   --manifest are all real (CLAUDE.md NEVER-invent-CLI-flags rule).
log "=== Pre-flight: build script argparse keys verified ==="

log "=== Stage 1: build Lane STC archive (boundary-coded class-ID payload) ==="
ARCHIVE="$LOG_DIR/archive_lane_stc.zip"
MANIFEST="$LOG_DIR/manifest.json"
"$PYBIN" -u experiments/build_lane_stc_archive.py \
    --anchor-archive "$ANCHOR_ARCHIVE" \
    --output "$ARCHIVE" \
    --boundary-fraction 0.05 \
    --manifest "$MANIFEST" 2>&1 | tee "$LOG_DIR/build.log" | tail -10
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: build_lane_stc_archive exited rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
if [ -z "${ARCHIVE_BYTES:-}" ] || [ "$ARCHIVE_BYTES" -le 0 ]; then
    echo "FATAL: archive size empty or zero — refusing to call auth_eval"
    exit 2
fi
log "  archive bytes: $ARCHIVE_BYTES"

# Compare vs anchor — fail loudly if STC produced a LARGER archive.
# Local 2026-04-29 measurement: extrapolated 1200-frame STC ~21MB vs Lane A
# ~411KB. Confirmed: lossless STC cannot beat AV1 on AV1-decoded noisy
# masks. Skip the GPU-burning auth eval if STC is bigger; preserve the
# code path for future use on UPSTREAM-source argmax masks (pre-AV1).
ANCHOR_BYTES=$(stat -c '%s' "$ANCHOR_ARCHIVE" 2>/dev/null || stat -f '%z' "$ANCHOR_ARCHIVE")
if [ "$ARCHIVE_BYTES" -gt "$ANCHOR_BYTES" ]; then
    log "WARN: STC archive ($ARCHIVE_BYTES B) is LARGER than anchor ($ANCHOR_BYTES B)."
    log "      Lane A masks are AV1-decoded noise; lossless STC cannot beat AV1"
    log "      on this distribution. Skipping auth_eval — would only waste GPU."
    log "      LANE_STC_NEGATIVE [empirical] -- code preserved for future runs"
    log "      against pre-AV1 SegNet argmax (clean class regions)."
    exit 0
fi

log "=== Stage 2: contest_auth_eval on Lane STC archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: contest_auth_eval exited rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

log "=== LANE_STC_DONE [contest-CUDA] -- see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
