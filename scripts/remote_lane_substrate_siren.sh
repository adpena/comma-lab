#!/bin/bash
# Remote lane script: substrate siren first-anchor dispatch.
#
# Trainer: experiments/train_substrate_siren.py (PHASE-B2-BUILD).
# Lane: lane_substrate_siren_20260512
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
# Council memo refs:
#   - feedback_phase_b2_build_3_high_target_trainers_LANDED_20260512.md
#   - .omx/research/grand_council_fields_medal_substrate_design_20260512.md
#     (council Phase 5 prediction: 0.145 [contest-CUDA])
#
# Score-tagging: completion logs report the auth-eval JSON's own score_axis /
# lane_tag. Modal A100 training is diagnostic unless the auth-eval contract
# itself says score_axis=contest_cuda and score_claim_valid=true.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity — non-negotiable".
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_substrate_siren_20260512"
TAG="${TAG:-substrate_siren}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_siren_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags — Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
SIREN_VIDEO_PATH="${SIREN_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
SIREN_OUTPUT_DIR="${SIREN_OUTPUT_DIR:-$OUTPUT_DIR}"
SIREN_EPOCHS="${SIREN_EPOCHS:-2000}"
SIREN_BATCH_SIZE="${SIREN_BATCH_SIZE:-1}"
SIREN_UPSTREAM_DIR="${SIREN_UPSTREAM_DIR:-$WORKSPACE/upstream}"
SIREN_DEVICE="${SIREN_DEVICE:-cuda}"
SIREN_DISPATCH_CONTRACT="${SIREN_DISPATCH_CONTRACT:-naked_siren_replacement}"

DISPATCH_INSTANCE_JOB_ID="${SIREN_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${SIREN_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-siren] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0a: strip macOS AppleDouble resource forks before any auth eval path.
rm -f upstream/videos/._*.mkv

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: SIREN_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
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
CLAIM_SUMMARY_JSON="$LOG_DIR/dispatch_claim_summary.json"
"$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" summary \
    --claims-path "$DISPATCH_CLAIMS_PATH" \
    --format json > "$CLAIM_SUMMARY_JSON" || {
    log "FATAL: claim summary failed for $DISPATCH_CLAIMS_PATH"
    exit 21
}
set +e
"$CLAIM_PYTHON" -c 'import json, sys
summary_path, lane_id, job_id = sys.argv[1:4]
payload = json.loads(open(summary_path, encoding="utf-8").read())
for row in payload.get("active", []):
    if row.get("lane_id") == lane_id and row.get("instance_job_id") == job_id:
        raise SystemExit(0)
print(f"missing active claim lane={lane_id} job={job_id}", file=sys.stderr)
raise SystemExit(1)' "$CLAIM_SUMMARY_JSON" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID"
CLAIM_MATCH_RC=$?
set -e
if [ "$CLAIM_MATCH_RC" -ne 0 ]; then
    log "FATAL: no active dispatch claim for lane=$LANE_ID job=$DISPATCH_INSTANCE_JOB_ID"
    exit 21
fi
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
    log "WARN: scripts/probe_nvdec.sh missing — skipping NVDEC early-fail probe"
fi

# Stage 1: bootstrap runtime deps (canonical, per CLAUDE.md).
if [ ! -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "FATAL: canonical bootstrap script missing at scripts/remote_archive_only_eval.sh"
    exit 22
fi
# shellcheck source=/dev/null
# REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 sentinel per Catalog #163.
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
"$PYBIN" - "$PROVENANCE" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID" "$DISPATCH_PLATFORM" "$GIT_HASH" "$GPU_NAME" "$DRIVER_VER" "$SIREN_VIDEO_PATH" "$SIREN_UPSTREAM_DIR" "$SIREN_EPOCHS" "$SIREN_BATCH_SIZE" "$SIREN_DEVICE" "$SIREN_DISPATCH_CONTRACT" <<'PY'
import json, sys, time, torch
(
    provenance_path,
    lane_id,
    dispatch_instance_job_id,
    dispatch_platform,
    git_hash,
    gpu_name,
    driver_version,
    video_path,
    upstream_dir,
    epochs,
    batch_size,
    device,
    dispatch_contract,
) = sys.argv[1:14]
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'lane_id': lane_id,
    'dispatch_instance_job_id': dispatch_instance_job_id,
    'dispatch_platform': dispatch_platform,
    'git_hash': git_hash,
    'gpu_name': gpu_name,
    'driver_version': driver_version,
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'video_path': video_path,
    'upstream_dir': upstream_dir,
    'epochs': int(epochs),
    'batch_size': int(batch_size),
    'device': device,
    'dispatch_contract': dispatch_contract,
    'dispatch_contracts_distinguished': [
        'naked_siren_replacement: SIREN/INR replaces the HNeRV/A1 substrate with SRV1 0.bin',
        'siren_residual_on_hnerv_a1: SIREN/INR residual sidecar over byte-verified HNeRV/A1 base',
        'hybrid_siren_domain_prior: SIREN/INR plus explicit domain-prior payload in one scored packet',
    ],
    # Literature-review prediction only, NOT score evidence.
    # Source: .omx/research/siren_literature_review_20260513.md
    # Band convention: [LOW, HIGH] in score-space (lower = better contest score).
    'predicted_band': [0.180, 0.250],
    'predicted_band_evidence_grade': 'literature_prediction_not_score_evidence',
    'predicted_basis': 'siren_literature_review_20260513',
}
import pathlib
pathlib.Path(provenance_path).write_text(json.dumps(prov, indent=2, sort_keys=True))
print('[provenance]', json.dumps(prov, indent=2, sort_keys=True))
PY
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
log "stage_4_trainer_invoke_begin video=$SIREN_VIDEO_PATH epochs=$SIREN_EPOCHS device=$SIREN_DEVICE"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_siren.py \
    --video-path "$SIREN_VIDEO_PATH" \
    --output-dir "$SIREN_OUTPUT_DIR" \
    --epochs "$SIREN_EPOCHS" \
    --batch-size "$SIREN_BATCH_SIZE" \
    --upstream-dir "$SIREN_UPSTREAM_DIR" \
    --device "$SIREN_DEVICE" \
    --dispatch-contract "$SIREN_DISPATCH_CONTRACT" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record.
AUTH_EVAL_JSON="$OUTPUT_DIR/contest_auth_eval_cuda.json"
ARCHIVE_PATH="$OUTPUT_DIR/archive.zip"
PAYLOAD_PATH="$OUTPUT_DIR/0.bin"
if [ -f "$AUTH_EVAL_JSON" ]; then
    log "auth_eval_artifact_present path=$AUTH_EVAL_JSON"
    AUTH_EVAL_TAG_FILE="$LOG_DIR/auth_eval_tag.txt"
    if "$PYBIN" - "$AUTH_EVAL_JSON" > "$AUTH_EVAL_TAG_FILE" <<'PY'
import json, sys
try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    print("score_axis=unknown lane_tag=unknown score_claim_valid=false")
else:
    print(
        "score_axis={axis} lane_tag={tag} score_claim_valid={valid}".format(
            axis=payload.get("score_axis", "unknown"),
            tag=payload.get("lane_tag", "unknown"),
            valid=str(payload.get("score_claim_valid") is True).lower(),
        )
    )
PY
    then
        AUTH_EVAL_TAG="$(cat "$AUTH_EVAL_TAG_FILE")"
    else
        AUTH_EVAL_TAG="score_axis=unknown lane_tag=unknown score_claim_valid=false"
    fi
    log "LANE_SIREN_DONE $AUTH_EVAL_TAG auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH payload=$PAYLOAD_PATH dispatch_contract=$SIREN_DISPATCH_CONTRACT rc=$TRAIN_RC"
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON (trainer may have failed before stage 12)"
fi

exit "$TRAIN_RC"
