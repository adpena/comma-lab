#!/bin/bash
# Remote lane script: substrate sar_coherent_pose_pairs (SARC) first-anchor dispatch.
#
# Trainer: experiments/train_substrate_sar_coherent_pose_pairs.py
# Lane: lane_sar_coherent_pose_pairs_substrate_20260513
# Recipe: .omx/operator_authorize_recipes/substrate_sar_coherent_pose_pairs_modal_t4_dispatch.yaml
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Design refs:
#   - .omx/research/expert_team_signal_processing_lincoln_lab_20260513.md §2
#     (the L2 SAR coherent integration over pose pairs derivation)
#   - feedback_expert_team_signal_processing_alien_tech_landed_20260513.md
#     (rank #1 EV/$ in the alien-tech short-list, $1 dispatch cost)
#
# Score-tagging: any score this script produces is tagged [contest-CUDA] in
# the completion-log line (LANE_SARC_DONE marker) per the CLAUDE.md
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
LANE_ID="lane_sar_coherent_pose_pairs_substrate_20260513"
TAG="${TAG:-substrate_sar_coherent_pose_pairs}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_sar_coherent_pose_pairs_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
SARC_VIDEO_PATH="${SARC_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
SARC_OUTPUT_DIR="${SARC_OUTPUT_DIR:-$OUTPUT_DIR}"
SARC_EPOCHS="${SARC_EPOCHS:-2000}"
SARC_UPSTREAM_DIR="${SARC_UPSTREAM_DIR:-$WORKSPACE/upstream}"
SARC_DEVICE="${SARC_DEVICE:-cuda}"
SARC_HIDDEN_DIM="${SARC_HIDDEN_DIM:-48}"
SARC_PER_PAIR_RESIDUAL_BYTES="${SARC_PER_PAIR_RESIDUAL_BYTES:-50}"
SARC_TOPK_KEEP_FRACTION="${SARC_TOPK_KEEP_FRACTION:-0.10}"

DISPATCH_INSTANCE_JOB_ID="${SARC_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${SARC_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""
CLAIM_VERIFIED=0

log() { echo "[lane-sarc] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: SARC_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
    exit 21
fi

CLAIM_PYTHON="${PYBIN:-}"
if [ -z "$CLAIM_PYTHON" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
    CLAIM_PYTHON="$WORKSPACE/.venv/bin/python"
fi
if [ -z "$CLAIM_PYTHON" ]; then
    CLAIM_PYTHON="python3"
fi

append_terminal_claim() {
    local rc="$1"
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        log "WARN: claim helper missing; cannot append terminal dispatch claim"
        return 0
    fi
    local status
    if [ "$rc" -eq 0 ]; then
        status="completed_sarc_remote_driver"
    elif [ "${CLAIM_VERIFIED:-0}" != "1" ]; then
        status="failed_sarc_claim_verification_rc_${rc}"
    else
        status="failed_sarc_remote_driver_rc_${rc}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --force \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "remote_lane_substrate_sar_coherent_pose_pairs" \
        --status "$status" \
        --notes "remote_driver_terminal rc=$rc output_dir=$SARC_OUTPUT_DIR" \
        >> "$LOG_DIR/run.log" 2>&1 || {
        log "WARN: failed to append terminal dispatch claim status=$status"
    }
}

cleanup() {
    local rc="$?"
    if [ -n "$HEARTBEAT_PID" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    append_terminal_claim "$rc"
    exit "$rc"
}
trap cleanup EXIT

if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
    log "FATAL: claim helper missing at $WORKSPACE/tools/claim_lane_dispatch.py"
    exit 21
fi
if [ ! -f "$DISPATCH_CLAIMS_PATH" ]; then
    log "FATAL: dispatch-claims ledger missing at $DISPATCH_CLAIMS_PATH"
    exit 21
fi

CLAIM_SUMMARY_JSON="$LOG_DIR/dispatch_claim_summary.json"
"$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" summary \
    --claims-path "$DISPATCH_CLAIMS_PATH" \
    --format json > "$CLAIM_SUMMARY_JSON" || {
    log "FATAL: claim summary failed for $DISPATCH_CLAIMS_PATH"
    exit 21
}
set +e
"$CLAIM_PYTHON" - "$CLAIM_SUMMARY_JSON" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID" <<'PY'
import json
import sys
summary_path, lane_id, job_id = sys.argv[1:4]
payload = json.loads(open(summary_path, encoding="utf-8").read())
for row in payload.get("active", []):
    if row.get("lane_id") == lane_id and row.get("instance_job_id") == job_id:
        raise SystemExit(0)
print(f"missing active claim lane={lane_id} job={job_id}", file=sys.stderr)
raise SystemExit(1)
PY
CLAIM_RC=$?
set -e
if [ "$CLAIM_RC" -ne 0 ]; then
    log "FATAL: no active dispatch claim for lane=$LANE_ID job=$DISPATCH_INSTANCE_JOB_ID"
    exit 21
fi
CLAIM_VERIFIED=1
log "stage_0_dispatch_claim_verified lane=$LANE_ID job=$DISPATCH_INSTANCE_JOB_ID"

# Stage 0b: NVDEC probe (per CLAUDE.md `feedback_vastai_nvdec_host_variation`).
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
    'video_path': '$SARC_VIDEO_PATH',
    'upstream_dir': '$SARC_UPSTREAM_DIR',
    'epochs': $SARC_EPOCHS,
    'device': '$SARC_DEVICE',
    'sarc_hidden_dim': $SARC_HIDDEN_DIM,
    'sarc_per_pair_residual_bytes': $SARC_PER_PAIR_RESIDUAL_BYTES,
    'sarc_topk_keep_fraction': $SARC_TOPK_KEEP_FRACTION,
    # Predicted band per L2 ledger §2.2 first-principles math.
    # Source: .omx/research/expert_team_signal_processing_lincoln_lab_20260513.md §2
    # ΔS -0.0056 vs PR101 0.193 anchor → predicted band [0.187, 0.190]
    # [first-principles-bound, literature-prediction].
    'predicted_band': [0.187, 0.190],
    'predicted_basis': 'L2_SAR_coherent_pose_pair_integration_lincoln_lab_§2',
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

# Stage 4: invoke trainer.
#
# The SARC trainer's _full_main wires 14 stages including auth eval (stage 12),
# continual-learning posterior update (stage 13), and provenance (stage 14).
# All Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS are threaded.
#
# Per CLAUDE.md "Auth eval EVERYWHERE" + "Submission auth eval - BOTH CPU AND
# CUDA": this is a FIRST-ANCHOR research dispatch on CUDA only; the resulting
# tag is [contest-CUDA] single-axis (CPU axis required separately before
# promotion-grade status).
log "stage_4_trainer_invoke_begin video=$SARC_VIDEO_PATH epochs=$SARC_EPOCHS device=$SARC_DEVICE hidden_dim=$SARC_HIDDEN_DIM"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_sar_coherent_pose_pairs.py \
    --video-path "$SARC_VIDEO_PATH" \
    --output-dir "$SARC_OUTPUT_DIR" \
    --epochs "$SARC_EPOCHS" \
    --upstream-dir "$SARC_UPSTREAM_DIR" \
    --device "$SARC_DEVICE" \
    --hidden-dim "$SARC_HIDDEN_DIM" \
    --per-pair-residual-bytes "$SARC_PER_PAIR_RESIDUAL_BYTES" \
    --sar-topk-keep-fraction "$SARC_TOPK_KEEP_FRACTION" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record.
AUTH_EVAL_JSON="$OUTPUT_DIR/auth_eval.json"
ARCHIVE_PATH="$OUTPUT_DIR/archive.zip"
PAYLOAD_PATH="$OUTPUT_DIR/0.bin"
if [ -f "$AUTH_EVAL_JSON" ]; then
    if [ "$TRAIN_RC" -eq 0 ]; then
        log "auth_eval_artifact_present path=$AUTH_EVAL_JSON"
        log "LANE_SARC_DONE [contest-CUDA] auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH payload=$PAYLOAD_PATH rc=$TRAIN_RC"
    else
        log "auth_eval_artifact_present_but_trainer_failed path=$AUTH_EVAL_JSON rc=$TRAIN_RC; refusing [contest-CUDA] completion tag"
    fi
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH payload=$PAYLOAD_PATH (trainer may have failed before stage 12)"
    if [ "$TRAIN_RC" -eq 0 ]; then
        log "FATAL: trainer returned rc=0 but auth eval artifact is missing; refusing silent green dispatch"
        TRAIN_RC=31
    fi
fi

exit "$TRAIN_RC"
