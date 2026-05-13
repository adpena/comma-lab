#!/bin/bash
# Remote lane script: substrate self_compress_nn (delta) HIGH-target dispatch.
#
# Trainer: experiments/train_substrate_self_compress_nn.py (WAVE-1-B 2026-05-12).
# Lane: lane_substrate_self_compress_nn_20260512 (existing L0 SKETCH; this
# trainer-build dispatch wires it toward L1/L2 via the WAVE-1-B child lane
# lane_wave1_self_compress_nn_trainer_build_20260512).
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Council memo refs:
#   - feedback_wave1_self_compress_nn_trainer_build_LANDED_20260512.md
#     (the WAVE-1-B trainer + recipe + driver build memo; this dispatch is
#     the FIRST-anchor execution for the delta candidate.)
#   - .omx/research/grand_council_fields_medal_substrate_design_20260512.md
#     (the council Phase 2 substrate-design memo defining delta at L0 SKETCH
#     with explicit reactivation criteria - NO KILL per CLAUDE.md
#     "KILL is LAST RESORT".)
#   - feedback_substrate_sane_hnerv_full_main_wired_landed_20260512.md
#     (the alpha 16-stage _full_main contract this driver mirrors; identical
#     stage layout sans the delta codebook EMA-step inside the per-batch loop
#     which lives inside the trainer, not this driver.)
#
# Score-tagging: any score this script produces is tagged [contest-CUDA] in
# the completion-log line (LANE_SELF_COMPRESS_NN_DONE marker) per the
# CLAUDE.md score-tag rule and preflight completion-tag check.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_substrate_self_compress_nn_20260512"
TAG="${TAG:-substrate_self_compress_nn}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_self_compress_nn_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
SELF_COMPRESS_NN_VIDEO_PATH="${SELF_COMPRESS_NN_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
SELF_COMPRESS_NN_OUTPUT_DIR="${SELF_COMPRESS_NN_OUTPUT_DIR:-$OUTPUT_DIR}"
SELF_COMPRESS_NN_EPOCHS="${SELF_COMPRESS_NN_EPOCHS:-2000}"
SELF_COMPRESS_NN_UPSTREAM_DIR="${SELF_COMPRESS_NN_UPSTREAM_DIR:-$WORKSPACE/upstream}"
SELF_COMPRESS_NN_DEVICE="${SELF_COMPRESS_NN_DEVICE:-cuda}"

DISPATCH_INSTANCE_JOB_ID="${SELF_COMPRESS_NN_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${SELF_COMPRESS_NN_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-self-compress-nn] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification (mirrors remote_lane_substrate_sane_hnerv.sh).
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: SELF_COMPRESS_NN_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
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
# Skip cleanly on Modal containers (NVDEC always available in CUDA images);
# the probe itself is idempotent and tolerates the no-DALI scenario.
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
# running. We only need its bootstrap_runtime_deps() function. Per Catalog
# #163 (the WWW4 sentinel-required gate landed 2026-05-12); without this
# sentinel the source triggers Stage 0+ archive-only-eval main flow which
# fails with "FATAL: archive missing" before any training can start.
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
    'wave_dispatch_lane_id': 'lane_wave1_self_compress_nn_trainer_build_20260512',
    'dispatch_instance_job_id': '$DISPATCH_INSTANCE_JOB_ID',
    'dispatch_platform': '$DISPATCH_PLATFORM',
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'video_path': '$SELF_COMPRESS_NN_VIDEO_PATH',
    'upstream_dir': '$SELF_COMPRESS_NN_UPSTREAM_DIR',
    'epochs': $SELF_COMPRESS_NN_EPOCHS,
    'device': '$SELF_COMPRESS_NN_DEVICE',
    # predicted_band per council Phase 5 calibration (CLAUDE.md no-signal-loss).
    # Source: .omx/research/grand_council_fields_medal_substrate_design_20260512.md
    # + recipe substrate_self_compress_nn_modal_a100_dispatch.yaml::predicted_delta
    # = '-0.060 [predicted; council Phase 5 HIGH-target ~0.17]'.
    # Band convention: [LOW, HIGH] in score-space (lower = better contest score).
    # HIGH-target 0.17 implies ~-0.06 vs prior anchors; we band [-0.080, -0.040]
    # to surface uncertainty: SKETCH-level QAT collapse risk on the LOW side,
    # block-FP saturation on the HIGH side.
    'predicted_band': [-0.080, -0.040],
    'predicted_basis': 'grand_council_fields_medal_substrate_design_20260512_phase5_HIGH_target',
    'literature_anchor': 'Selfcomp PR #56 + Quantizr 0.33 anchor 2026-04-21 + van den Oord VQ-VAE 2017',
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
# The trainer's _full_main wires 16 stages including auth eval (stage 12),
# continual-learning posterior update (stage 13), and cost-band anchor
# emission (stage 14). All required flags are threaded per Catalog #151
# TIER_1_OPERATOR_REQUIRED_FLAGS manifest.
#
# Per CLAUDE.md "Auth eval EVERYWHERE" + "Submission auth eval - BOTH CPU
# AND CUDA": this is a FIRST-ANCHOR research dispatch on CUDA only; the
# resulting tag is [contest-CUDA] single-axis (CPU axis required separately
# before promotion-grade status).
#
# Per CLAUDE.md "EMA - non-negotiable" codebook clause: van den Oord
# codebook EMA (decay 0.99) is updated AFTER every optimizer.step inside
# the trainer's per-batch loop; the surrounding weights use the standard
# EMA decay 0.997. Both contracts are honored inside the trainer; this
# driver only invokes the trainer.
log "stage_4_trainer_invoke_begin video=$SELF_COMPRESS_NN_VIDEO_PATH epochs=$SELF_COMPRESS_NN_EPOCHS device=$SELF_COMPRESS_NN_DEVICE"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_self_compress_nn.py \
    --video-path "$SELF_COMPRESS_NN_VIDEO_PATH" \
    --output-dir "$SELF_COMPRESS_NN_OUTPUT_DIR" \
    --epochs "$SELF_COMPRESS_NN_EPOCHS" \
    --upstream-dir "$SELF_COMPRESS_NN_UPSTREAM_DIR" \
    --device "$SELF_COMPRESS_NN_DEVICE" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record (the auth-eval JSON was already written by the
# trainer at stage 12 if reached). We surface the path here for harvest.
AUTH_EVAL_JSON="$OUTPUT_DIR/contest_auth_eval_cuda.json"
ARCHIVE_PATH="$OUTPUT_DIR/0.bin"
if [ -f "$AUTH_EVAL_JSON" ]; then
    log "auth_eval_artifact_present path=$AUTH_EVAL_JSON"
    log "LANE_SELF_COMPRESS_NN_DONE [contest-CUDA] auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON (trainer may have failed before stage 12)"
fi

exit "$TRAIN_RC"
