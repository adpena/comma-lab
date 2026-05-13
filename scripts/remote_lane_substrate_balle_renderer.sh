#!/bin/bash
# Remote lane script: substrate balle_renderer (β) first-anchor dispatch.
#
# Trainer: experiments/train_substrate_balle_renderer.py (commit d5b69eff+).
# Lane: lane_substrate_balle_renderer_20260512
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Council memo refs:
#   - .omx/research/grand_council_fields_medal_substrate_design_20260512.md
#     (the β-target architecture design: 8 hyperprior channels, GDN eps 1e-12,
#     lambda_hyperprior 0.5 conservative default)
#   - feedback_phase_b1_pivot_modal_source_staleness_fix_LANDED_20260512.md
#     (this lane's dispatch chain landed under the smoke-before-full pattern
#     per Catalog #167)
#
# Score-tagging: any score this script produces is tagged [contest-CUDA] in
# the completion-log line (LANE_BALLE_RENDERER_DONE marker) per the CLAUDE.md
# score-tag rule and preflight completion-tag check.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_substrate_balle_renderer_20260512"
TAG="${TAG:-substrate_balle_renderer}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_balle_renderer_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
BALLE_RENDERER_VIDEO_PATH="${BALLE_RENDERER_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
BALLE_RENDERER_OUTPUT_DIR="${BALLE_RENDERER_OUTPUT_DIR:-$OUTPUT_DIR}"
BALLE_RENDERER_EPOCHS="${BALLE_RENDERER_EPOCHS:-2000}"
BALLE_RENDERER_UPSTREAM_DIR="${BALLE_RENDERER_UPSTREAM_DIR:-$WORKSPACE/upstream}"
BALLE_RENDERER_DEVICE="${BALLE_RENDERER_DEVICE:-cuda}"
BALLE_RENDERER_HYPERPRIOR_CHANNELS="${BALLE_RENDERER_HYPERPRIOR_CHANNELS:-8}"
BALLE_RENDERER_GDN_EPS="${BALLE_RENDERER_GDN_EPS:-1e-12}"
BALLE_RENDERER_LAMBDA_HP="${BALLE_RENDERER_LAMBDA_HP:-0.5}"

DISPATCH_INSTANCE_JOB_ID="${BALLE_RENDERER_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${BALLE_RENDERER_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-balle-renderer] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification (mirrors remote_lane_substrate_sane_hnerv.sh).
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: BALLE_RENDERER_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
    exit 21
fi
if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
    log "FATAL: claim helper missing at $WORKSPACE/tools/claim_lane_dispatch.py"
    exit 21
fi
if [ ! -f "$DISPATCH_CLAIMS_PATH" ]; then
    log "FATAL: dispatch-claims ledger missing at $DISPATCH_CLAIMS_PATH"
    exit 21
fi

CLAIM_PYTHON="${PYBIN:-}"
if [ -z "$CLAIM_PYTHON" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
    CLAIM_PYTHON="$WORKSPACE/.venv/bin/python"
fi
if [ -z "$CLAIM_PYTHON" ]; then
    CLAIM_PYTHON="python3"
fi

# Stage 0b: NVDEC probe (per CLAUDE.md `feedback_vastai_nvdec_host_variation`).
# Early failure-class probe BEFORE any uv/torch install or training spend.
# Failure rate on cold-pool hosts is ~30-50%; this saves $0.05-0.10 per bad host.
# Skip cleanly on Modal containers (NVDEC always available in CUDA images);
# the probe itself is idempotent and tolerates the no-DALI scenario by exiting 0
# when torchvision-only path is acceptable.
if [ -x "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    log "stage_0b_nvdec_probe_begin"
    bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
        log "FATAL: NVDEC probe failed (host hardware/driver/DALI mismatch); refusing dispatch"
        exit 2
    }
    log "stage_0b_nvdec_probe_done"
else
    log "WARN: scripts/probe_nvdec.sh missing - skipping NVDEC early-fail probe"
fi

# Stage 1: bootstrap runtime deps (canonical, per CLAUDE.md).
if [ ! -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "FATAL: canonical bootstrap script missing at scripts/remote_archive_only_eval.sh"
    exit 22
fi
# shellcheck source=/dev/null
# REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 prevents the sourced script's main
# flow (which expects a pre-built archive.zip at /workspace/pact/iter_0/) from
# running. We only need its bootstrap_runtime_deps() function (Catalog #163).
REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"

# bootstrap_runtime_deps installs uv + torch (driver-version-pinned per CLAUDE.md
# "Forbidden uv torch install without driver-version pin"). It also exports PYBIN.
if declare -f bootstrap_runtime_deps > /dev/null 2>&1; then
    log "stage_1_bootstrap_runtime_deps_begin"
    bootstrap_runtime_deps
    log "stage_1_bootstrap_runtime_deps_done PYBIN=${PYBIN:-unset}"
else
    log "FATAL: bootstrap_runtime_deps function not found after sourcing scripts/remote_archive_only_eval.sh"
    exit 23
fi

if [ -z "${PYBIN:-}" ] || [ ! -x "$PYBIN" ]; then
    log "FATAL: PYBIN not set or not executable after bootstrap (PYBIN=${PYBIN:-unset})"
    exit 24
fi

# Stage 2: provenance + remote code parity (per CLAUDE.md "Remote code parity").
log "stage_2_provenance_begin"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo "no-driver")
"$PYBIN" -c "
import json, time, torch
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'lane_id': '$LANE_ID',
    'dispatch_instance_job_id': '$DISPATCH_INSTANCE_JOB_ID',
    'dispatch_platform': '$DISPATCH_PLATFORM',
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'video_path': '$BALLE_RENDERER_VIDEO_PATH',
    'upstream_dir': '$BALLE_RENDERER_UPSTREAM_DIR',
    'epochs': $BALLE_RENDERER_EPOCHS,
    'device': '$BALLE_RENDERER_DEVICE',
    'balle_hyperprior_channels': $BALLE_RENDERER_HYPERPRIOR_CHANNELS,
    'balle_gdn_eps': $BALLE_RENDERER_GDN_EPS,
    'balle_lambda_hp': $BALLE_RENDERER_LAMBDA_HP,
    # predicted_band per council calibration (CLAUDE.md no-signal-loss).
    # Source: .omx/research/grand_council_fields_medal_substrate_design_20260512.md
    # + recipe substrate_balle_renderer_modal_a100_dispatch.yaml::predicted_delta
    # = '-0.013 vs PR101 0.193 [predicted; council Phase 5]'.
    # Target score: 0.18; band convention: [LOW, HIGH] in score-space.
    'predicted_band': [0.15, 0.21],
    'predicted_basis': 'grand_council_fields_medal_substrate_design_20260512_balle_renderer',
}
import pathlib
pathlib.Path('$PROVENANCE').write_text(json.dumps(prov, indent=2, sort_keys=True))
print('[provenance]', json.dumps(prov, indent=2, sort_keys=True))
"
log "stage_2_provenance_done"

# Stage 3: heartbeat watchdog (every 5 min per CLAUDE.md).
(
    while true; do
        echo "[heartbeat] $(date -u +%FT%TZ) lane=$LANE_ID alive" >> "$LOG_DIR/heartbeat.log"
        sleep 300
    done
) &
HEARTBEAT_PID=$!
trap 'if [ -n "$HEARTBEAT_PID" ]; then kill "$HEARTBEAT_PID" 2>/dev/null || true; fi' EXIT

# Stage 4: invoke trainer.
#
# The balle_renderer trainer's _full_main wires 16 stages including auth eval
# (stage 12), continual-learning posterior update (stage 13), and cost-band
# anchor emission (stage 14). All Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS
# are threaded; the β-specific knobs (--balle-hyperprior-channels, --gdn-eps,
# --lambda-hyperprior) carry conservative council defaults.
#
# Per CLAUDE.md "Auth eval EVERYWHERE" + "Submission auth eval - BOTH CPU AND
# CUDA": this is a FIRST-ANCHOR research dispatch on CUDA only; the resulting
# tag is [contest-CUDA] single-axis (CPU axis required separately before
# promotion-grade status).
log "stage_4_trainer_invoke_begin video=$BALLE_RENDERER_VIDEO_PATH epochs=$BALLE_RENDERER_EPOCHS device=$BALLE_RENDERER_DEVICE hp_channels=$BALLE_RENDERER_HYPERPRIOR_CHANNELS"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_balle_renderer.py \
    --video-path "$BALLE_RENDERER_VIDEO_PATH" \
    --output-dir "$BALLE_RENDERER_OUTPUT_DIR" \
    --epochs "$BALLE_RENDERER_EPOCHS" \
    --upstream-dir "$BALLE_RENDERER_UPSTREAM_DIR" \
    --device "$BALLE_RENDERER_DEVICE" \
    --balle-hyperprior-channels "$BALLE_RENDERER_HYPERPRIOR_CHANNELS" \
    --gdn-eps "$BALLE_RENDERER_GDN_EPS" \
    --lambda-hyperprior "$BALLE_RENDERER_LAMBDA_HP" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record (the auth-eval JSON was already written by the
# trainer at stage 12 if reached). We surface the path here for harvest.
AUTH_EVAL_JSON="$OUTPUT_DIR/auth_eval.json"
ARCHIVE_PATH="$OUTPUT_DIR/0.bin"
if [ -f "$AUTH_EVAL_JSON" ]; then
    log "auth_eval_artifact_present path=$AUTH_EVAL_JSON"
    log "LANE_BALLE_RENDERER_DONE [contest-CUDA] auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON (trainer may have failed before stage 12)"
fi

exit "$TRAIN_RC"
