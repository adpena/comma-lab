#!/bin/bash
# Remote lane script: substrate hybrid_renderer_residual first-anchor dispatch.
#
# Trainer: experiments/train_substrate_hybrid_renderer_residual.py (WAVE-1-C 2026-05-12).
# Lane: lane_substrate_hybrid_renderer_residual_20260512
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Per Catalog #163 the canonical sentinel REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1
# MUST be set before sourcing the bootstrap to prevent the sourced main flow
# (which expects a pre-built archive.zip) from running.
#
# COMPOSITION RISK NOTE:
#   The hybrid lane is a COMPOSITE substrate (alpha renderer + beta-style sparse
#   residual) per CLAUDE.md "Substrate vs codec composition meta-pattern".
#   The recipe carries pre_promotion_blockers requiring alpha and beta each have
#   a verified [contest-CUDA] anchor BEFORE this dispatches. The
#   operator_authorize.py wrapper enforces those blockers; this remote
#   driver assumes the wrapper has cleared them.
#
# Council memo refs:
#   - feedback_wave1_hybrid_renderer_residual_trainer_build_LANDED_20260512.md
#   - .omx/research/grand_council_fields_medal_substrate_design_20260512.md
#     (council Phase 5 prediction: ~0.17 [contest-CUDA] HIGH-target)
#   - feedback_substrate_vs_codec_composition_meta_pattern_20260508.md
#     (composition-without-verified-base anti-pattern)
#
# Score-tagging: any score this script produces is tagged [contest-CUDA] in
# the completion-log line (LANE_HYBRID_RES_DONE marker) per the CLAUDE.md
# score-tag rule and preflight completion-tag check.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
set -euo pipefail

# === Catalog #244 / D1 incident anchor (commit 611495f26): canonical Modal/CUDA env hygiene ===
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" + standing directive
# 2026-05-15 ("all possible should be pulled into the decorator or similar reusable
# and shareable tools and helpers and such"). Sister substrates D1/D4/Z3/Z4/Z5 carry
# this block; backfilled to all 31 sister drivers via Catalog #244 strict-flip wave.
# Future drivers should be auto-generated via tac.substrate_registry.driver_generator
# (which AUTO-EMITS the block from canonical constants in tac.deploy.modal.runtime).
# D1 dispatch 2026-05-15 (Modal T4 smoke) crashed at NVML 999 because the lane script
# did not export DALI_DISABLE_NVML=1 before DALI imported NVML.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_substrate_hybrid_renderer_residual_20260512"
TAG="${TAG:-substrate_hybrid_renderer_residual}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_hybrid_renderer_residual_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${HYBRID_RES_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$HYBRID_RES_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
HYBRID_RES_VIDEO_PATH="${HYBRID_RES_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
HYBRID_RES_OUTPUT_DIR="${HYBRID_RES_OUTPUT_DIR:-$OUTPUT_DIR}"
HYBRID_RES_EPOCHS="${HYBRID_RES_EPOCHS:-2000}"
HYBRID_RES_UPSTREAM_DIR="${HYBRID_RES_UPSTREAM_DIR:-$WORKSPACE/upstream}"
HYBRID_RES_DEVICE="${HYBRID_RES_DEVICE:-cuda}"

# Composition-specific env (gamma-only). Both default to "do nothing"; the
# wrapper / operator decides whether to freeze alpha or warm-start from a
# pre-trained alpha checkpoint.
HYBRID_RES_FREEZE_ALPHA="${HYBRID_RES_FREEZE_ALPHA:-}"
HYBRID_RES_ALPHA_PRETRAINED_CHECKPOINT="${HYBRID_RES_ALPHA_PRETRAINED_CHECKPOINT:-}"

DISPATCH_INSTANCE_JOB_ID="${HYBRID_RES_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${HYBRID_RES_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-hybrid-res] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0a: strip macOS AppleDouble resource forks before any auth eval path.
rm -f upstream/videos/._*.mkv

# Stage 0: dispatch claim verification (mirrors remote_lane_substrate_sane_hnerv.sh).
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: HYBRID_RES_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
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
# Per Catalog #163 (`check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap`):
# REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 prevents the sourced script's main
# flow (which expects a pre-built archive.zip at /workspace/pact/iter_0/) from
# running. We only need its bootstrap_runtime_deps() function. Without this
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
    'dispatch_instance_job_id': '$DISPATCH_INSTANCE_JOB_ID',
    'dispatch_platform': '$DISPATCH_PLATFORM',
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'video_path': '$HYBRID_RES_VIDEO_PATH',
    'upstream_dir': '$HYBRID_RES_UPSTREAM_DIR',
    'epochs': $HYBRID_RES_EPOCHS,
    'device': '$HYBRID_RES_DEVICE',
    # predicted_band per council Phase 5 substrate design memo.
    # Source: .omx/research/grand_council_fields_medal_substrate_design_20260512.md
    # + recipe substrate_hybrid_renderer_residual_modal_a100_dispatch.yaml::predicted_score_target
    # = 0.17 HIGH-target with -0.025 vs alpha standalone.
    # Band convention: [LOW, HIGH] in score-space (lower = better contest score).
    'predicted_band': [0.155, 0.185],
    'predicted_basis': 'grand_council_fields_medal_substrate_design_20260512',
    'composition_class': 'alpha_renderer_plus_beta_residual',
    'pre_promotion_blockers': [
        'sane_hnerv_first_anchor_required',
        'balle_renderer_first_anchor_required',
    ],
    'freeze_alpha_env': '$HYBRID_RES_FREEZE_ALPHA',
    'alpha_warmstart_env': '$HYBRID_RES_ALPHA_PRETRAINED_CHECKPOINT',
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
# Composition-specific flags are added to the command line conditionally
# based on env-var presence so the trainer's defaults (no freeze, no warm
# start) hold unless the operator explicitly sets them.
#
# Per CLAUDE.md "Auth eval EVERYWHERE" + "Submission auth eval - BOTH CPU
# AND CUDA": this is a FIRST-ANCHOR research dispatch on CUDA only; the
# resulting tag is [contest-CUDA] single-axis (CPU axis required separately
# before promotion-grade status).
log "stage_4_trainer_invoke_begin video=$HYBRID_RES_VIDEO_PATH epochs=$HYBRID_RES_EPOCHS device=$HYBRID_RES_DEVICE freeze_alpha=${HYBRID_RES_FREEZE_ALPHA:-no} warmstart=${HYBRID_RES_ALPHA_PRETRAINED_CHECKPOINT:-no}"

# Build the trainer command line. Composition flags are conditional.
TRAIN_CMD=("$PYBIN" experiments/train_substrate_hybrid_renderer_residual.py
    --video-path "$HYBRID_RES_VIDEO_PATH"
    --output-dir "$HYBRID_RES_OUTPUT_DIR"
    --epochs "$HYBRID_RES_EPOCHS"
    --upstream-dir "$HYBRID_RES_UPSTREAM_DIR"
    --device "$HYBRID_RES_DEVICE"
)
if [ -n "$HYBRID_RES_FREEZE_ALPHA" ]; then
    TRAIN_CMD+=(--freeze-alpha)
fi
if [ -n "$HYBRID_RES_ALPHA_PRETRAINED_CHECKPOINT" ]; then
    TRAIN_CMD+=(--alpha-pretrained-checkpoint "$HYBRID_RES_ALPHA_PRETRAINED_CHECKPOINT")
fi

TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"${TRAIN_CMD[@]}" 2>&1 | tee -a "$LOG_DIR/run.log"
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
    log "LANE_HYBRID_RES_DONE [contest-CUDA] auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON (trainer may have failed before stage 12)"
fi

exit "$TRAIN_RC"
