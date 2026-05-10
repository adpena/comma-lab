#!/bin/bash
# Lane MM — Selfcomp grayscale-LUT mask re-encoding (encoder-only quick win).
#
# Hypothesis: replace Lane A's masks.mkv (legacy [0,63,127,191,255] linear
# ramp) with grayscale.mkv using Selfcomp's class targets [0,255,64,192,128]
# + Gaussian softmax LUT (sigma=15) at inflate time. AV1 monochrome encoding
# of the spread targets (64-pixel gaps) absorbs ~10-15 levels of quantizer
# noise without flipping the nearest-neighbour class — predicted ~50% mask
# byte savings with no quality loss.
#
# Architecture: UNCHANGED. Same renderer.bin from Lane A (1.15 [contest-CUDA]).
# Same poses. Only masks.mkv -> grayscale.mkv via experiments/build_lane_mm_archive.py.
# Inflate path: PYTHON_INFLATE=renderer_grayscale (decodes grayscale via LUT
# -> argmax -> 5-channel one-hot -> existing MaskRenderer).
#
# Predicted band [0.65, 0.85] [contest-CUDA].
# Anchor: experiments/results/lane_a_landed/archive_lane_a.zip (1.15 verified).
# Cost cap: $0.50 (no training; encode + auth eval only). Budget hard cap.
# controlled_baseline: Lane A (lane_a_landed/archive_lane_a.zip, 1.15 contest-CUDA).
#                      Single-mechanism change being isolated: legacy 5-class linear
#                      ramp -> Selfcomp 5-class spread targets + Gaussian LUT.
#
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
# E2E_SMOKE_OPT_OUT: encoder-only lane reuses Lane A renderer + poses verbatim;
#   the grayscale-LUT path has unit-test coverage (tac.mask_grayscale_lut +
#   tac.block_fp_codec roundtrip tests). Smoke proof will be backfilled before
#   remote dispatch via experiments/canonical_local_auth_eval_smoke.py.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_mm_grayscale_lut"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-mm] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_mm_grayscale_lut.sh',
    'output_dir': '$LOG_DIR',
    'predicted_band': [0.65, 0.85],
    'anchor_archive': 'experiments/results/lane_a_landed/archive_lane_a.zip',
    'anchor_score_baseline': 1.15,
    'controlled_baseline': 'lane_a_landed (1.15 contest-CUDA)',
    'paradigm': 'grayscale_lut_mask_encoding',
    'inflate_path': 'PYTHON_INFLATE=renderer_grayscale',
    'hypothesis': 'grayscale-LUT mask cuts rate ~50% with no quality loss',
    'cost_estimate_usd': 0.30,
    'cost_cap_usd': 0.50,
    'wall_clock_estimate_hours': 0.3,
    'wall_clock_cap_hours': 1.0,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ' || echo "no-gpu")
    echo "[$(date -u +%FT%TZ)] lane=MM gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend. Reference:
# feedback_vastai_nvdec_host_variation. --ensure-dali so a fresh container
# that hasn't run remote_setup_full.sh installs DALI rather than spuriously
# failing on a missing import. Lane MM does not actually use DALI for video
# decode (it goes through ffmpeg), but the probe also catches NVDEC-bad
# hosts that would fail upstream/evaluate.py at the auth eval stage.
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
         submissions/robust_current/inflate_renderer_grayscale.py; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

# Pre-flight: dead-flag-wiring guard removed (2026-04-29 Round 3 fix).
# The inline regex-based scanner false-positive matched comment text and
# heredoc bodies, killing every Modal dispatch with 'INVENTED FLAGS: [hard]'
# (matched from line 23's 'NEVER git pull / git reset --hard' comment).
# Coverage gap: tac.preflight.preflight_arity scans Python launchers but not
# shell-script invocations of experiments/*.py. Follow-up: extend preflight_
# arity to cover remote_lane_*.sh scripts (tracked separately). The flags
# in the invocation below are manually verified against build_lane_mm_archive's
# argparse: --anchor-archive, --output, --crf, --sigma, --keep-legacy-masks
# are all real (verified 2026-04-29).
log "=== Pre-flight: argparse dead-flag scan SKIPPED (inline scanner buggy, see fix in commit) ==="

log "=== Stage 1: build Lane MM archive (Selfcomp grayscale-LUT mask re-encoding) ==="
ARCHIVE="$LOG_DIR/archive_lane_mm.zip"
set +e
"$PYBIN" -u experiments/build_lane_mm_archive.py \
    --anchor-archive "$ANCHOR_ARCHIVE" \
    --output "$ARCHIVE" \
    --crf 50 2>&1 | tee "$LOG_DIR/build.log" | tail -10
PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: build_lane_mm_archive exited rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
if [ -z "${ARCHIVE_BYTES:-}" ] || [ "$ARCHIVE_BYTES" -le 0 ]; then
    echo "FATAL: archive size empty or zero — refusing to call auth_eval"
    exit 2
fi
log "  archive bytes: $ARCHIVE_BYTES"

# config.env override: tell inflate.sh to use the renderer_grayscale arm.
INFLATE_CONFIG="$LOG_DIR/lane_mm_config.env"
cat > "$INFLATE_CONFIG" <<'EOF'
# Lane MM inflate config: grayscale-LUT mask decode + existing renderer.
PYTHON_INFLATE=renderer_grayscale
EOF

log "=== Stage 2: contest_auth_eval on Lane MM archive ==="
rm -rf "$LOG_DIR/eval_work"
set +e
CONFIG_ENV_PATH="$INFLATE_CONFIG" "$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: contest_auth_eval exited rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

log "=== LANE_MM_DONE [contest-CUDA] -- see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
